"""
Adapted from https://github.com/franciscovillaescusa/Pylians3
"""
module SphericalVoids  # create module to be loaded in main file

include("utils.jl")
include("meshbuilder.jl")

using .Utils
using .MeshBuilder
using Printf
using FFTW

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
function smoothing(delta::Array{<:AbstractFloat,3},dims::Int64,middle::Int64,R::Float64,box_length::AbstractFloat)

    @info "Smoothing density field with top-hat filter of radius=$R"

    # compute FFT of the field
    @debug "Computing forward FT"
    delta_k = rfft(delta, [3,1,2])

    # loop over independent modes
    @debug "Looping over Fourier modes"
    prefact = 2.0*pi*R/box_length
    kk = fftfreq(dims, dims)
    for (i,kx) in enumerate(kk)
        kx2 = kx*kx
        for (j,ky) in enumerate(kk)
            ky2 = ky*ky
            for (k,kz) in enumerate(kk[1:middle+1])
                kz2 = kz*kz

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

    @debug "Computing backward FT" dims=size(real.(irfft(delta_k, dims, [3,1,2])))
    real.(irfft(delta_k, dims, [3,1,2]))

end

"""
Parallelised top-hat smoothing of density field.
"""
function smoothing(delta::Array{<:AbstractFloat,3},dims::Int64,middle::Int64,R::Float64,box_length::AbstractFloat,fft_plan,threading::Bool)

    @info "Smoothing density field with top-hat filter of radius=$R"

    # check parallelisation is to be used
    if !threading
        throw(ErrorException("Threading is set to False, cannot use this function!"))
    end

    # compute FFT of the field
    if fft_plan == nothing
        @debug "Creating FTT plan"
        # println("Creating fft plan...")
        fft_plan = plan_rfft(delta, [3,1,2]; num_threads=Threads.nthreads())
    end
    @debug "Computing forward FT"
    delta_k = fft_plan * delta

    # loop over independent modes
    @debug "Looping over Fourier modes"
    prefact = 2.0*pi*R/box_length
    kk = fftfreq(dims, dims)
    Threads.@threads for (i,kx) in collect(enumerate(kk))
        kx2 = kx*kx
        for (j,ky) in enumerate(kk)
            ky2 = ky*ky
            for (k,kz) in enumerate(kk[1:middle+1])
                kz2 = kz*kz

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

    @debug "Computing backward FT" dims=size(real.(fft_plan \ delta_k))
    return fft_plan, real.(fft_plan \ delta_k)

end

"""
Identify cells below a given density threshold and sort them from lowest density.
"""
function underdense_cells(delta::Array{<:AbstractFloat,3},threshold::Float64,dims::Int64,dims2::Int64,in_void::Array{Int8,3},delta_v::Array{Float64,1},IDs::Array{Int64,1})
    @info "Identifying underdense cells"
    @debug "Check mesh details" dims cells=dims^3


    # find underdense cells
    local_voids = 0
    for i = 1:dims
        for j = 1:dims
            for k = 1:dims

                if delta[i,j,k]<threshold && in_void[i,j,k]==0
                    local_voids += 1
                    IDs[local_voids] = dims2*(i-1) + dims*(j-1) + (k-1)
                    delta_v[local_voids] = delta[i,j,k]
                end

            end
        end
    end
    @info "Found $local_voids cells with delta < $threshold"
    # if par.verbose
        # @printf("Found %d cells with delta < %.2f\n", local_voids, threshold)
    # end

    # sort cells underdensities
    indx = sortperm(delta_v[1:local_voids])
    # delta_v[1:local_voids] = delta_v[indx]
    # IDs[1:local_voids] = IDs[indx]

    @debug "Underdense cells sorted"
    IDs[indx]

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
    
"""
Identify if void candidate is a new void or belongs to a previously detected void by computing distances to detected voids.

Returns 1 if another void has been detected nearby.
"""
function nearby_voids1(voids_total::Int32,dims::Int64,middle::Int64,i::Int64,j::Int64,k::Int64,void_radius::Array{Float64,1},void_pos::Array{Int64,2},R_grid::Float64,max_overlap_frac::Float64)

    nearby_voids::Int32 = 0
    overlap_frac::Float64 = 0

    # loop over all previously detected voids
    for l = 1:voids_total

        dx = i - void_pos[l,1]
        if abs(dx) > middle
            dx = abs(dx) - dims
        end

        dy = j - void_pos[l,2]
        if abs(dy) > middle
            dy = abs(dy) - dims
        end

        dz = k - void_pos[l,3]
        if abs(dz) > middle
            dz = abs(dz) - dims 
        end

        # determine distance of void from candidate
        dist2 = dx*dx + dy*dy + dz*dz
        
        # determine if voids overlap
        if dist2<((void_radius[l]+R_grid)*(void_radius[l]+R_grid))

            if max_overlap_frac == 0
                nearby_voids = 1
                break
            else
                overlap_frac += void_overlap_frac(R_grid,void_radius[l],dist2)

                if overlap_frac > max_overlap_frac
                    nearby_voids = 1
                    break
                end

            end
        end
    end

    nearby_voids

end

"""
Identify if void candidate is a new void or belongs to a previously detected void by determining if cells around the candidate belong to other voids.

Returns 0 if new void, else another void has been detected nearby.
"""
function nearby_voids2(Ncells::Int64,dims::Int64,i::Int64,j::Int64,k::Int64,R_grid::Float64,R_grid2::Float64,in_void::Array{Int8,3},max_overlap_frac::Float64)

    nearby_voids::Int32 = 0
    overlap::Int64 = 0
    inv_void_cells = 3/(4*pi*R_grid^3)

    # loop over all cells in cubic box around void
    for l = -Ncells:Ncells
        i1 = mod1(i+l, dims)

        for m = -Ncells:Ncells  
            j1 = mod1(j+m, dims)

            for n = -Ncells:Ncells        
                k1 = mod1(k+n, dims)

                # skip if cell does not belong to another void
                if in_void[i1,j1,k1] == 0
                    continue
                # determine if void cell is within radius of candidate 
                else
                    dist2 = l*l + m*m + n*n
                    if dist2<R_grid2

                        if max_overlap_frac == 0
                            nearby_voids = 1
                            break
                        else
                            overlap += 1
                            if overlap*inv_void_cells > max_overlap_frac
                                nearby_voids = 1
                                break
                            end
                        end

                    end
                end

            end
        end
    end

    nearby_voids

end

function nearby_voids2(Ncells::Int64,dims::Int64,i::Int64,j::Int64,k::Int64,R_grid::Float64,R_grid2::Float64,in_void::Array{Int8,3},max_overlap_frac::Float64,threading::Bool)

    # check parallelisation is to be used
    if !threading
        throw(ErrorException("Threading is set to False, cannot use this function!"))
    end

    nearby_voids = Threads.Atomic{Int32}(0)
    overlap::Int64 = 0
    inv_void_cells = 3/(4*pi*R_grid^3)

    # loop over all cells in cubic box around void
    Threads.@threads for l = -Ncells:Ncells

        # skip thread once a nearby void has been detected
        if nearby_voids[]>0 
            continue
        end

        i1 = mod1(i+l, dims)

        for m = -Ncells:Ncells  
            j1 = mod1(j+m, dims)

            for n = -Ncells:Ncells        
                k1 = mod1(k+n, dims)

                # skip if cell does not belong to another void
                if in_void[i1,j1,k1] == 0
                    continue
                # determine if void cell is within radius of candidate 
                else
                    dist2 = l*l + m*m + n*n
                    if dist2<R_grid2

                        if max_overlap_frac == 0
                            Threads.atomic_add!(nearby_voids,Int32(1))
                        else
                            overlap += 1
                            if overlap*inv_void_cells > max_overlap_frac
                                Threads.atomic_add!(nearby_voids,Int32(1))
                            end
                        end
                    end
                end


            end
        end
    end

    nearby_voids[]

end

"""
Mark cells in radius R around void center as belonging to void.
"""
function mark_void_region!(Ncells::Int64,dims::Int64,i::Int64,j::Int64,k::Int64,R_grid2::Float64,in_void::Array{Int8,3})

    # loop over all cells in cubic box around void
    for l = -Ncells:Ncells  
        i1 = mod1(i+l, dims)

        for m = -Ncells:Ncells  
            j1 = mod1(j+m, dims)

            for n = -Ncells:Ncells        
                k1 = mod1(k+n, dims)

                dist2 = l*l + m*m + n*n
                if dist2<R_grid2

                    in_void[i1,j1,k1] = Int8(1)

                end
            end
        end
    end
end

function mark_void_region!(Ncells::Int64,dims::Int64,i::Int64,j::Int64,k::Int64,R_grid2::Float64,in_void::Array{Int8,3},threading::Bool)

    # check parallelisation is to be used
    if !threading
        throw(ErrorException("Threading is set to False, cannot use this function!"))
    end

    # loop over all cells in cubic box around void
    Threads.@threads for l = -Ncells:Ncells  
        i1 = mod1(i+l, dims)

        for m = -Ncells:Ncells  
            j1 = mod1(j+m, dims)

            for n = -Ncells:Ncells        
                k1 = mod1(k+n, dims)

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
function voidfinder(delta::Array{<:AbstractFloat,3}, box_length::AbstractFloat, box_centre::Array{<:AbstractFloat,1}, par::Main.VoidParameters.SphericalVoidParams; fft_plan = nothing)

    @info "Void-finding on density field with spherical-based method" 
    @debug "Check voidfinding parameters" radii=par.radii dims=size(delta,1) box_length box_centre 
    # println("\n ==== Void-finding on field with spherical-based method ==== ")

    if Threads.nthreads() == 1
        threading = false
    else
        threading = true 
        @info "Multithreading active: n=$(Threads.nthreads()) threads in use."
        # println("\nMultithreading active: n=",Threads.nthreads()," threads in use.")
    end

    dims = size(delta,1)
    dims2 = dims^2
    dims3 = dims^3
    middle = dims÷2
    r_bins = size(par.radii,1)
    
    if par.radii == [0]
        throw(ErrorException("No void radii provided. Must be provided as a 1D array."))
    end

    res::Float64 = box_length/dims

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
    voids_max = floor(Int64, box_length^3/vol_eff)
    if par.verbose && par.max_overlap_frac == 0
        @info "Maximum number of voids = $voids_max"
        # @printf("\nMaximum number of voids = %d\n", voids_max)
    end

    # initialize arrays
    void_pos = Array{Int64}(undef,voids_max,3)  # void positions
    void_radius = Array{Float64}(undef,voids_max)  # void radii

    in_void = zeros(Int8,dims,dims,dims)  # void membership cell flag
    delta_v = Array{Float64}(undef,dims3)  # underdense cells density contrasts 
    IDs = Array{Int64}(undef,dims3)  # underdense cell IDs

    Nvoids = Array{Int64}(undef,r_bins)
    vsf = Array{Float64}(undef,r_bins,2)

    # find voids at each input radius R
    voids_total::Int32 = 0  # total number of voids found
    expected_filling_frac::Float64 = 0.0
    for (q,R) in enumerate(Radii)  
        # if par.verbose
            # @printf("\nSmoothing galaxy density field with top-hat filter of radius %.2f...\n", R)
        # end

        if !threading
            delta_sm = smoothing(delta, dims, middle, R, box_length)
        else
            fft_plan, delta_sm = smoothing(delta, dims, middle, R, box_length, fft_plan, threading)
        end

        # check if cells are below threshold
        if minimum(delta_sm)>threshold
            @info "No cells with delta < $threshold"
            # @printf("No cells with delta < %.2f\n", threshold)
            continue
        end  
        
        # identify cells below density threshold
        cell_ID = underdense_cells(delta_sm, threshold, dims, dims2, in_void, delta_v, IDs)

        R_grid = R/res
        R_grid2 = R_grid*R_grid
        Ncells = floor(Int64, R_grid + 1)

        if voids_total<(2*Ncells+1)^3
            mode = 0
        else
            mode = 1
        end

        # if par.verbose
            # @printf("Identifying nearby voids using mode %d...\n", mode)
        # end

        # identify if underdense cells belong to previously detected voids
        @info "Identifying nearby voids using mode $mode"
        voids_with_R = 0  # voids found with radius R
        for ID in cell_ID
            i,j,k = (ID÷dims2, (ID%dims2)÷dims, (ID%dims2)%dims) .+ 1

            # if cell belong to a void, continue
            if in_void[i,j,k] == 1
                continue
            end

            if mode==0 
                nearby_voids = nearby_voids1(voids_total, dims, middle, i, j, k, void_radius, void_pos, R_grid, par.max_overlap_frac)
            else
                if !threading
                    nearby_voids = nearby_voids2(Ncells, dims, i, j, k, R_grid, R_grid2, in_void, par.max_overlap_frac)
                else
                    nearby_voids = nearby_voids2(Ncells, dims, i, j, k, R_grid, R_grid2, in_void, par.max_overlap_frac, threading)
                end
            end

            # if new void found
            if nearby_voids == 0

                voids_with_R += 1
                voids_total += 1

                void_pos[voids_total,:] .= i,j,k
                void_radius[voids_total] = R_grid

                # flag cells belonging to new void
                in_void[i,j,k] = 1
                if !threading
                    mark_void_region!(Ncells, dims, i, j, k, R_grid2, in_void)
                else
                    mark_void_region!(Ncells, dims, i, j, k, R_grid2, in_void, threading)
                end

            end

        end

        Nvoids[q] = voids_with_R

        @info "Found $voids_with_R voids with radius R = $R Mpc/h"
        expected_filling_frac += voids_with_R*4*pi*R^3/(3*box_length^3)
        @debug "Volume filling fraction" vff_true=sum(in_void)/dims3 vff_exp=expected_filling_frac
        # @printf("Found %d voids with radius R = %.3f Mpc/h\n",voids_with_R,R)
        # if par.verbose 
            # @printf("Found %d voids with radius R >= %.3f Mpc/h\n",voids_total,R)
            # @printf("Void volume filling fraction = %.3f\n",sum(in_void)/dims3)
            # if par.max_overlap_frac == 0
                # expected_filling_frac += voids_with_R*4*pi*R^3/(3*box_length^3)
                # @printf("Expected filling fraction = %.3f\n",expected_filling_frac)
            # end
        # end
            
    end

    @info "Total of $voids_total voids found"
    @info "Total volume filling fraction = $(sum(in_void)/dims3)"
    # @printf("\nFound a total of %d voids\n",voids_total)
    # @printf("Void volume filling fraction = %.3f\n",sum(in_void)/dims3)

    # compute the void size function (# of voids/Volume/dR)
    @info "Computing the void size function"
    for i = 1:r_bins
        r = vcat(Radii, 0)
        vsf[i,1] = 0.5*(r[i]+r[i+1])
        vsf[i,2] = Nvoids[i]/(box_length^3*(r[i]-r[i+1]))
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
function voidfinder(cat::Main.MeshBuilder.GalaxyCatalogue, mesh::Main.VoidParameters.MeshParams, par::Main.VoidParameters.SphericalVoidParams; fft_plan = nothing)

    # set default void radii to 2-10x mean galaxy separation
    if par.radii == [0]
        par.radii = [2:10;] * mean_gal_sep(cat, mesh)
    end

    mesh_obj = create_mesh(cat, mesh)

    voidfinder(mesh_obj..., par; fft_plan = fft_plan)

end


end
