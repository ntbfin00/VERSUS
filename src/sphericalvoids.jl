"""
Adapted from https://github.com/franciscovillaescusa/Pylians3
"""
module SphericalVoids  # create module to be loaded in main file

include("utils.jl")
include("meshbuilder.jl")

using .Utils
using .MeshBuilder
using FFTW
using SortingLab

"""
Ouput structure to hold void data.
"""
struct VoidData
    type::String
    positions::Array{Float64,2}
    radii::Array{Float64,1}
    vsf::Array{Float64,2}
end

"""
Return input parameters specific to spherical voids.
"""
function get_params(par::Main.VoidParameters.SphericalVoidParams)
    println("\n ==== Spherical void input parameters ==== ")

    for field in propertynames(par)
        println(field, " = ", getfield(par, field))
    end

end

"""
Top-hat smoothing of density field.
"""
function smoothing(delta::Array{<:AbstractFloat,3},dims::Array{Int64,1},R::Float64,res::AbstractFloat,fft_plan)

    @info "Smoothing density field with top-hat filter of radius=$R"
    
    # compute FFT of the field
    if fft_plan == nothing
        @debug "Creating FTT plan"
        fft_plan = plan_rfft(delta)
    end

    @debug "Computing forward FT"
    delta_k = fft_plan * delta

    # loop over independent modes
    @debug "Computing frequencies"
    kkx = rfftfreq(dims[1], 1/res).^2
    kky = fftfreq(dims[2], 1/res).^2
    kkz = fftfreq(dims[3], 1/res).^2
    prefact = 2.0*pi*R
    @debug "Looping over Fourier modes"
    @inbounds @fastmath for (k,kz2) in enumerate(kkz)
        for (j,ky2) in enumerate(kky)
            for (i,kx2) in enumerate(kkx)

                # skip when kx, ky and kz equal zero
                if i==1 && j==1 && k==1
                    continue 
                end

                kR = prefact * sqrt(kx2 + ky2 + kz2)
                if abs(kR)>1e-5
                    delta_k[i,j,k] *= 3.0*(sin(kR) - cos(kR)*kR)/(kR*kR*kR)
                end

            end
        end
    end

    @debug "Computing backward FT" 
    return fft_plan, real.(fft_plan \ delta_k)

end

"""
Parallelised top-hat smoothing of density field.
"""
function smoothing(delta::Array{<:AbstractFloat,3},dims::Array{Int64,1},R::Float64,res::AbstractFloat,fft_plan,threading::Bool)

    @info "Smoothing density field with top-hat filter of radius=$R"

    # check parallelisation is to be used
    if !threading
        throw(ErrorException("Threading is set to False, cannot use this function!"))
    end 

    # compute FFT of the field
    if fft_plan == nothing
        @debug "Creating FTT plan"
        fft_plan = plan_rfft(delta)
    end

    @debug "Computing forward FT"
    delta_k = fft_plan * delta

    # loop over independent modes
    @debug "Computing frequencies"
    kkx = rfftfreq(dims[1], 1/res).^2
    kky = fftfreq(dims[2], 1/res).^2
    kkz = fftfreq(dims[3], 1/res).^2
    prefact = 2.0*pi*R
    @debug "Looping over Fourier modes"
    @inbounds @fastmath Threads.@threads for (k,kz2) in collect(enumerate(kkz))
        for (j,ky2) in enumerate(kky)
            for (i,kx2) in enumerate(kkx)

                # skip when kx, ky and kz equal zero
                if i==1 && j==1 && k==1
                    continue 
                end

                kR = prefact * sqrt(kx2 + ky2 + kz2)
                if abs(kR)>1e-5
                    delta_k[i,j,k] *= 3.0*(sin(kR) - cos(kR)*kR)/(kR*kR*kR)
                end

            end
        end
    end

    @debug "Computing backward FT"
    return fft_plan, real.(fft_plan \ delta_k)

end

"""
Identify cells below a given density threshold and sort them from lowest density.
"""
function underdense_cells(delta::Array{<:AbstractFloat,3},threshold::Float64,dims::Array{Int64,1},dims23::Int64,in_void::Array{Int8,3},delta_v::Array{Float64,1},IDs::Array{Int64,1})
    @info "Identifying underdense cells"

    # find underdense cells
    local_voids = 0
    @inbounds for k = 1:dims[3]
        for j = 1:dims[2]
            for i = 1:dims[1]

                if delta[i,j,k]<threshold && in_void[i,j,k]==0
                    local_voids += 1
                    IDs[local_voids] = dims23*(i-1) + dims[3]*(j-1) + (k-1)
                    delta_v[local_voids] = delta[i,j,k]
                end

            end
        end
    end
    @info "Found $local_voids cells with delta < $threshold"

    # sort cells underdensities
    cell_indx = fsortperm(delta_v[1:local_voids])

    IDs[cell_indx]

end

"""
Compute the volume fraction of the candidate void that belongs to previously detected voids.
"""
function void_overlap_frac(R_cand::Float64,R_void::Float64,dist2::Int64)

    @debug "Determine the void overlap fraction"

    dist = sqrt(dist2)

    a = (R_void + R_cand - dist)^2 
    b = (dist2 - 3*R_cand*R_cand - 3*R_void*R_void + 2*dist*R_cand + 2*dist*R_void + 6*R_cand*R_void)
    c = 1/(16*dist*R_cand^3)
    
    a*b*c

end
    
function periodic_bounds(dd,mid,dim)

    if abs(dd) > mid
        return abs(dd) - dim
    else
        return dd
    end

end

"""
Identify if void candidate is a new void or belongs to a previously detected void by computing distances to detected voids.

Returns 1 if another void has been detected nearby.
"""
function nearby_voids1(voids_total::Int32,dims::Array{Int64,1},middle::Array{Int64,1},i::Int64,j::Int64,k::Int64,void_radius::Array{Float64,1},void_pos::Array{Int64,2},R_grid::Float64,max_overlap_frac::Float64,bound_conds::Bool)

    overlap_frac::Float64 = 0

    # loop over all previously detected voids
    @inbounds for l = 1:voids_total

        dx = i - void_pos[l,1]
        if bound_conds
            dx = periodic_bounds(dx, middle[1], dims[1])
        end

        dy = j - void_pos[l,2]
        if bound_conds
            dy = periodic_bounds(dy, middle[2], dims[2])
        end

        dz = k - void_pos[l,3]
        if bound_conds
            dz = periodic_bounds(dz, middle[3], dims[3])
        end

        # determine distance of void from candidate
        dist2 = dx*dx + dy*dy + dz*dz
        
        # determine if voids overlap
        if dist2<((void_radius[l]+R_grid)*(void_radius[l]+R_grid))

            if max_overlap_frac == 0
                return 1
            else
                overlap_frac += void_overlap_frac(R_grid,void_radius[l],dist2)

                if overlap_frac > max_overlap_frac
                    return 1
                end

            end
        end
    end

    return 0

end


"""
Identify if void candidate is a new void or belongs to a previously detected void by determining if cells around the candidate belong to other voids.

Returns 0 if new void, else another void has been detected nearby.
"""
function nearby_voids2(Ncells::Int64,dims::Array{Int64,1},i::Int64,j::Int64,k::Int64,R_grid::Float64,R_grid2::Float64,in_void::Array{Int8,3},max_overlap_frac::Float64,bound_conds::Bool)

    overlap::Int64 = 0
    inv_void_cells = 3/(4*pi*R_grid^3)

    for n = -Ncells:Ncells        
        k1 = k+n
        if bound_conds
            k1 = mod1(k1, dims[3])
        end

        for m = -Ncells:Ncells  
            j1 = j+m
            if bound_conds
                j1 = mod1(j1, dims[2])
            end

            for l = -Ncells:Ncells
                i1 = i+l
                if bound_conds
                    i1 = mod1(i1, dims[1])
                end

                # skip if cell does not belong to another void
                if in_void[i1,j1,k1] == 0
                    continue
                # determine if void cell is within radius of candidate 
                else
                    dist2 = l*l + m*m + n*n
                    if dist2<R_grid2

                        if max_overlap_frac == 0
                            return 1
                        else
                            overlap += 1
                            if overlap*inv_void_cells > max_overlap_frac
                                return 1
                            end
                        end

                    end
                end

            end
        end
    end

    return 0

end

function nearby_voids2(Ncells::Int64,dims::Array{Int64,1},i::Int64,j::Int64,k::Int64,R_grid::Float64,R_grid2::Float64,in_void::Array{Int8,3},max_overlap_frac::Float64,bound_conds::Bool,threading::Bool)

    # check parallelisation is to be used
    if !threading
        throw(ErrorException("Threading is set to False, cannot use this function!"))
    end

    nearby_voids = Atomic(0)  # initialise atomic counter at zero
    overlap = 0
    inv_void_cells = 3/(4*pi*R_grid^3)

    # loop over all cells in cubic box around void
    Threads.@threads for n = -Ncells:Ncells

        # skip thread once a nearby void has been detected
        if (@atomic nearby_voids.counter) == 1
            continue
        end

        k1 = k+n
        if bound_conds
            k1 = mod1(k1, dims[3])
        end

        for m = -Ncells:Ncells  
            j1 = j+m
            if bound_conds
                j1 = mod1(j1, dims[2])
            end

            for l = -Ncells:Ncells
                i1 = i+l
                if bound_conds
                    i1 = mod1(i1, dims[1])
                end

                # skip if cell does not belong to another void
                if in_void[i1,j1,k1] == 0
                    continue
                # determine if void cell is within radius of candidate 
                else
                    dist2 = l*l + m*m + n*n
                    if dist2<R_grid2

                        if max_overlap_frac == 0
                            @atomicswap nearby_voids.counter = 1
                        else
                            overlap += 1
                            if overlap*inv_void_cells > max_overlap_frac
                                @atomicswap nearby_voids.counter = 1
                            end
                        end
                    end
                end


            end
        end
    end

    @atomic nearby_voids.counter

end

"""
Mark cells in radius R around void center as belonging to void.
"""
function mark_void_region!(Ncells::Int64,dims::Array{Int64,1},i::Int64,j::Int64,k::Int64,R_grid2::Float64,in_void::Array{Int8,3}, bound_conds::Bool)

    # loop over all cells in cubic box around void
    for n = -Ncells:Ncells        
        k1 = k+n
        if bound_conds
            k1 = mod1(k1, dims[3])
        end

        for m = -Ncells:Ncells  
            j1 = j+m
            if bound_conds
                j1 = mod1(j1, dims[2])
            end

            for l = -Ncells:Ncells
                i1 = i+l
                if bound_conds
                    i1 = mod1(i1, dims[1])
                end

                dist2 = l*l + m*m + n*n
                if dist2<R_grid2

                    in_void[i1,j1,k1] = Int8(1)

                end
            end
        end
    end
end

function mark_void_region!(Ncells::Int64,dims::Array{Int64,1},i::Int64,j::Int64,k::Int64,R_grid2::Float64,in_void::Array{Int8,3},bound_conds::Bool,threading::Bool)

    # check parallelisation is to be used
    if !threading
        throw(ErrorException("Threading is set to False, cannot use this function!"))
    end

    # loop over all cells in cubic box around void
    Threads.@threads for n = -Ncells:Ncells        
        k1 = k+n
        if bound_conds
            k1 = mod1(k1, dims[3])
        end

        for m = -Ncells:Ncells  
            j1 = j+m
            if bound_conds
                j1 = mod1(j1, dims[2])
            end

            for l = -Ncells:Ncells
                i1 = i+l
                if bound_conds
                    i1 = mod1(i1, dims[1])
                end

                dist2 = l*l + m*m + n*n
                if dist2<R_grid2

                    in_void[i1,j1,k1] = Int8(1)

                end
            end
        end
    end

end


"""
Determine the number of voids in given overdensity mesh with radii specified at input.

    1) Smooth density field with spherical top-hat filter of radius R (starting with largest R)
    2) Find cells with density lower than threshold (starting with lowest density)
    3) Cell determined to be centre of void with radius R if does not overlap with previously determined void, otherwise discarded
    4) This is performed for all cells lower than threshold 
    5) Then repeated with next largest smoothing radius

"""
function voidfinder(delta::Array{<:AbstractFloat,3}, box_length::Array{<:AbstractFloat,1}, box_centre::Array{<:AbstractFloat,1}, par::Main.VoidParameters.SphericalVoidParams; fft_plan = nothing, bound_conds = true, volume = prod(box_length), mask = nothing)

    @info "Void-finding on density field with spherical-based method" 
    @debug "Check voidfinding parameters" radii=par.radii dims=size(delta) box_length box_centre volume bound_conds

    if Threads.nthreads() == 1
        threading = false
    else
        threading = true 
        @info "Multithreading active: n=$(Threads.nthreads()) threads in use."
    end

    # dimesions of delta
    dims = collect(size(delta))
    middle = dims .÷ 2
    cells_total = prod(dims)
    dims23 = dims[2]*dims[3]
    box_vol = prod(box_length)

    if par.radii == [0]
        throw(ErrorException("No void radii provided. Must be provided as a 1D array."))
    end
    r_bins = size(par.radii,1)

    # check grid resolution is the same along the axes 
    res = box_length./dims
    if isapprox(res[1], res[2]; atol=0.1) && isapprox(res[2], res[3]; atol=0.1)
        res = sum(res)/3
    else
        throw(ErrorException("Mesh resolutions along each axis are not equal."))
    end

    # set minimum overdensity threshold
    threshold = par.min_dens_cut - 1

    # sort radii from largest to smallest
    Radii = sort(par.radii, rev=true)

    # check Rmin is larger than grid resolution
    Rmin = Radii[end]
    if Rmin<res
        throw(ErrorException("Rmin=$Rmin Mpc/h below grid resolution=$res Mpc/h"))
    end

    # determine the maximum possible number of voids
    vol_eff = (1.0-par.max_overlap_frac)*(4*pi*Rmin^3)/3  # minimum effective void volume
    voids_max = floor(Int64, box_vol/vol_eff)
    @debug "Maximum number of voids = $voids_max"

    # initialize arrays
    void_pos = Array{Int64}(undef,voids_max,3)  # void positions
    void_radius = Array{Float64}(undef,voids_max)  # void radii

    in_void = zeros(Int8,dims...)  # void membership cell flag
    delta_v = Array{Float64}(undef,cells_total)  # underdense cells density contrasts 
    IDs = Array{Int64}(undef,cells_total)  # underdense cell IDs

    Nvoids = zeros(Int64,r_bins)
    vsf = Array{Float64}(undef,r_bins,5)

    voids_total::Int32 = 0  # total number of voids found
    expected_filling_frac::Float64 = 0.0

    FFTW.set_num_threads(Threads.nthreads())
    
    # find voids at each input radius R
    @inbounds for (q,R) in enumerate(Radii)  

        if !threading
            fft_plan, delta_sm = smoothing(delta, dims, R, res, fft_plan)
        else
            fft_plan, delta_sm = smoothing(delta, dims, R, res, fft_plan, threading)
        end
        @debug "Smoothing complete"

        # reset survey mask
        if mask != nothing
            @debug "Resetting survey mask" length(mask)
            delta_sm[mask] .= 0.
        end

        # check if cells are below threshold
        if minimum(delta_sm)>threshold
            @info "No cells with delta < $threshold"
            continue
        end  
        
        # identify cells below density threshold
        cell_ID = underdense_cells(delta_sm, threshold, dims, dims23, in_void, delta_v, IDs)
        @debug "Underdense cells sorted" 

        R_grid = R/res
        R_grid2 = R_grid*R_grid
        Ncells = floor(Int64, R_grid + 1)

        if voids_total<(2*Ncells+1)^3
            mode = 0
        else
            mode = 1
        end

        # identify if underdense cells belong to previously detected voids
        @info "Identifying nearby voids using mode $mode"
        voids_with_R = 0  # voids found with radius R
        for ID in cell_ID
            i = ID÷dims23 + 1
            j = (ID%dims23)÷dims[3] + 1
            k = (ID%dims23)%dims[3] + 1

            # if cell belong to a void, continue
            if in_void[i,j,k] == 1
                continue
            end

            if mode==0 
                nearby_voids = nearby_voids1(voids_total, dims, middle, i, j, k, void_radius, void_pos, R_grid, par.max_overlap_frac, bound_conds)
            else
                nearby_voids = nearby_voids2(Ncells, dims, i, j, k, R_grid, R_grid2, in_void, par.max_overlap_frac, bound_conds)
                # if !threading
                    # nearby_voids = nearby_voids2(Ncells, dims, i, j, k, R_grid, R_grid2, in_void, par.max_overlap_frac, bound_conds)
                # else
                    # nearby_voids = nearby_voids2(Ncells, dims, i, j, k, R_grid, R_grid2, in_void, par.max_overlap_frac, bound_conds, threading)
                # end
            end

            # if new void found
            if nearby_voids == 0

                voids_with_R += 1
                voids_total += 1

                void_pos[voids_total,1] = i
                void_pos[voids_total,2] = j
                void_pos[voids_total,3] = k
                void_radius[voids_total] = R_grid

                # flag cells belonging to new void
                in_void[i,j,k] = 1
                mark_void_region!(Ncells, dims, i, j, k, R_grid2, in_void, bound_conds)
                # if !threading
                    # mark_void_region!(Ncells, dims, i, j, k, R_grid2, in_void, bound_conds)
                # else
                    # mark_void_region!(Ncells, dims, i, j, k, R_grid2, in_void, bound_conds, threading)
                # end

            end

        end

        Nvoids[q] = voids_with_R

        @info "Found $voids_with_R voids with radius R = $R Mpc/h"
        expected_filling_frac += voids_with_R*4*pi*R^3/(3*box_vol)
        @debug "Volume filling fraction" vff_true=sum(in_void)/cells_total vff_exp=expected_filling_frac
            
    end

    @info "Total of $voids_total voids found"
    @info "Total volume filling fraction = $(sum(in_void)/cells_total)"

    # compute the void size function (# of voids/Volume/dR)
    @info "Computing the void size function"
    @inbounds for i = 1:r_bins
        r = vcat(Radii, 0)
        Nvoids = vcat(Nvoids, 0)
        vsf[i,1] = r[i]
        vsf[i,2] = Nvoids[i]
        vsf[i,3] = Nvoids[i]/volume
        vsf[i,4] = 0.5 * (r[i] + r[i+1])  # mean(R)
        vsf[i,5] = (Nvoids[i] - Nvoids[i+1])/(volume * (r[i] - r[i+1]))  # dn/dlnR
    end

    box_shift = box_centre .- box_length/2
    void_centres = (void_pos[1:voids_total,:] .- 0.5)*res .+ box_shift'
    # output data 
    VoidData("Spherical",
             void_centres,
             void_radius[1:voids_total]*res,
             vsf)

end


"""
Void finding with galaxy positions.
"""
function voidfinder(cat::Main.VoidParameters.GalaxyCatalogue, mesh::Main.VoidParameters.MeshParams, par::Main.VoidParameters.SphericalVoidParams; fft_plan = nothing)

    box_like = (size(cat.rand_pos,1)==0)

    if box_like
        @info "VERSUS running on simulation box data"
    else
        @info "VERSUS running on survey-like data"
    end

    nbins, r_sep, vol, threshold = gal_dens_bin(cat, mesh)
    @info "Catalogue details:" r_sep vol

    # set default void radii to 3-10x mean galaxy separation
    if par.radii == [0]
        par.radii = [10:-1:2;] * r_sep
        @info "Default radii set" par.radii
    end
    # set default number of bins based on galaxy density
    if mesh.nbins_vf == [0]
        mesh.nbins_vf = nbins
        @info "Default voidfinding bins set" mesh.nbins_vf
    end

    delta, mask, box_length, box_centre = create_mesh(cat, mesh; threshold = threshold)

    voidfinder(delta, box_length, box_centre, par; fft_plan = fft_plan, bound_conds = box_like, volume = vol, mask = mask)

end


end
