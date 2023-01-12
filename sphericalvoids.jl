module SphericalVoids  # create module to be loaded in main file

# package import
using Printf
using ImageFiltering
using FFTW  # don't need if using ImageFiltering

"""
Return input parameters specific to spherical voids.
"""
function get_params(par::Main.VoidParameters.VoidParams)
    println("\n ==== Spherical void input parameters ==== ")

    spherical_params = ["is_box",
                       "box_length",
                       "min_dens_cut",
                       "max_overlap_frac"]
                      
    for field in propertynames(par)
        if string(field) in spherical_params
            println(field, " = ", getfield(par, field))
        end
    end
end

# ==== may want in separate module (also used by voxel) ===== #

struct VoidData
    type::String
    positions::Array{Float64,2}
    radii::Array{Float64,1}
    vsf::Array{Float64,1}
    rbins::Array{Float64,1}
end

# function density_field(cat::Main.Tracers.TracerCat)
    # # bin galaxies to determine density field delta
    # # copy from fastmodules.allocate_gal_cic
    
    # @printf("%i tracers found", cat.size)
# end

function smoothing(delta::Array{Float64,3},r_scale::Float64,res::Float64,filter::String)
    # mesh delta must be created first by binning galaxies
    # "reflect" uses convolution instead of correlation

    # r_scale is the smoothing scale in Mpc/h
    # res is the voxel size in Mpc/h
    
    r_grid = r_scale/res
    if filter == "gaussian"
        kern = Kernel.gaussian((r_grid,r_grid,r_grid))
    # for spherical top hat, better in separate function??
    else
        d = 2*round(Int,r_grid) + 1  # Does Pylians need integer values for radii?
        # create spherical top hat kernel
        inv_vol = 3/(4*pi*r_grid^3)
        kern = ones(Float64,(d,d,d))*inv_vol
        k11 = [1,1,:]; k12 = [1,2,:]; k1dm1 = [1,d-1,:]; k1d = [1,d,:]
        k21 = [2,1,:]; k2d = [2,d,:]; 
        kdm11 = [d-1,1,:]; kdm1d = [d-1,d,:]; 
        kd1 = [d,1,:]; kd2 = [d,2,:]; kddm1 = [d,d-1,:]; kdd = [d,d,:]
        # make cubic kernel spherical
        for perm = 1:3
            for knn in (k11,k12,k1dm1,k1d,k21,k2d,kdm11,kdm1d,kd1,kd2,kddm1,kdd)
                kern[knn...] .= 0
                permute!(knn,[3,1,2])
            end
        end
        kern = centered(kern)
    end

    imfilter(delta,
             kern,
             "reflect",
             Algorithm.FFT())

end

# function top_hat_smooth(delta::Array{Float64,3},BoxSize::Float64,R::Float64,threads::Int)

# end

# =========================================================== #

"""
Compute the volume fraction of the candidate void that belongs to previously detected voids.
"""
function void_overlap_frac(R_cand::Float64,R_void::Float64,dist2::Int64)

    dist = sqrt(dist2)

    a = (R_void + R_cand - dist)^2 
    b = (dist2 - 3*R_cand*R_cand - 3*R_void*R_void + 2*dist*R_cand + 2*dist*R_void + 6*R_cand*R_void)
    c = 1/(16*dist*R_cand^3)
    
    a*b*c

end
    

"""
Identify if void candidate is a new void or belongs to a previously detected void by computing distances to detected voids.

Returns 0 if new void, else another void has been detected nearby.
"""
function nearby_voids1(voids_total::Int8,dims::Int64,middle::Int64,i::Int64,j::Int64,k::Int64,void_radius::Array{Float64,1},void_pos::Array{Int64,2},R_grid::Float64,max_overlap_frac::Float64)

    nearby_voids = Threads.Atomic{Int8}(0)
    overlap_frac::Float64 = 0

    # loop over all previously detected voids
    Threads.@threads for l = 1:voids_total
        if nearby_voids[]>0 
            continue
        end

        dx = i - void_pos[l,1]
        if dx>middle
            dx -= dims
        end
        if dx<-middle
            dx += dims
        end

        dy = j - void_pos[l,2]
        if dy>middle
            dy -= dims
        end
        if dy<-middle
            dy += dims
        end

        dz = k - void_pos[l,3]
        if dz>middle
            dz -= dims
        end
        if dz<-middle
            dz += dims
        end

        # determine distance of void from candidate
        dist2 = dx*dx + dy*dy + dz*dz
        
        # determine if voids overlap
        if dist2<((void_radius[l]+R_grid)*(void_radius[l]+R_grid))

            if max_overlap_frac == 0
                Threads.atomic_add!(nearby_voids,Int8(1))
            else
                overlap_frac += void_overlap_frac(R_grid,void_radius[l],dist2)

                if overlap_frac > max_overlap_frac
                    Threads.atomic_add!(nearby_voids,Int8(1))
                end
            end
        end
    end

    nearby_voids[]

end


"""
Identify if void candidate is a new void or belongs to a previously detected void by determining if cells around the candidate belong to other voids.

Returns 0 if new void, else another void has been detected nearby.
"""
function nearby_voids2(Ncells::Int64,dims::Int64,i::Int64,j::Int64,k::Int64,R_grid::Float64,R_grid2::Float64,in_void::Array{Int8,3},max_overlap_frac::Float64)

    nearby_voids = Threads.Atomic{Int8}(0)
    overlap::Int64 = 0
    inv_void_cells = 3/(4*pi*R_grid^3)

    # loop over all cells in cubic box around void
    Threads.@threads for l = -Ncells:Ncells  # test performance against not using threads and using threads for each nested loop

        # skip thread once a nearby void has been detected
        if nearby_voids[]>0 
            continue
        end

        i1 = i+l
        if i1>dims
            i1 -= dims
        end
        if i1<1
            i1 += dims
        end

        Threads.@threads for m = -Ncells:Ncells  

            j1 = j+m
            if j1>dims
                j1 -= dims
            end
            if j1<1
                j1 += dims
            end

            Threads.@threads for n = -Ncells:Ncells        

                k1 = k+n
                if k1>dims
                    k1 -= dims
                end
                if k1<1
                    k1 += dims
                end

                # skip if cell does not belong to another void
                if in_void[i1,j1,k1] == 0
                    continue
                # determine if void cell is within radius of candidate 
                else
                    dist2 = l*l + m*m + n*n
                    if dist2<R_grid2

                        if max_overlap_frac == 0
                            Threads.atomic_add!(nearby_voids,Int8(1))
                        else
                            overlap += 1
                            if overlap*inv_void_cells > max_overlap_frac
                                Threads.atomic_add!(nearby_voids,Int8(1))
                            end
                        end
                    end
                end


            end
        end
    end

    nearby_voids[]

end


function mark_void_region!(Ncells::Int64,dims::Int64,i::Int64,j::Int64,k::Int64,R_grid2::Float64,in_void::Array{Int8,3})

    # loop over all cells in cubic box around void
    # Threads.@threads 
    for l = -Ncells:Ncells  

        i1 = i+l
        if i1>dims
            i1 -= dims
        end
        if i1<1
            i1 += dims
        end

        # Threads.@threads 
        for m = -Ncells:Ncells  

            j1 = j+m
            if j1>dims
                j1 -= dims
            end
            if j1<1
                j1 += dims
            end

            # Threads.@threads 
            for n = -Ncells:Ncells        

                k1 = k+n
                if k1>dims
                    k1 -= dims
                end
                if k1<1
                    k1 += dims
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
function run_voidfinder(delta::Array{Float64,3},Radii::Array{Float64,1},par::Main.VoidParameters.VoidParams)
    println("\n ==== Starting the void-finding with spherical-based method ==== ")

    dims = size(delta,1)
    dims2 = dims^2
    dims3 = dims^3
    middle = dims÷2
    r_bins = size(Radii,1)

    res::Float64 = par.box_length/dims

    # set minimum overdensity threshold
    # IS THIS CORRECT??
    threshold = par.min_dens_cut - 1

    # sort radii from largest to smallest
    Radii = sort(Radii, rev=true)
    # check Rmin is larger than grid resolution
    Rmin = Radii[end]

    if Rmin<res
        throw(ErrorException("Rmin=$Rmin Mpc/h below grid resolution=$res Mpc/h"))
    end
    # determine the maximum possible number of voids
    vol_eff = (1.0-par.max_overlap_frac)*(4*pi*Rmin^3)/3  # minimum effective void volume
    voids_max = floor(Int64, par.box_length/vol_eff)
    @printf("\nMaximum number of voids = %d\n", voids_max)

    # initialize arrays
    void_pos = zeros(Int64,voids_max,3)  # void positions
    void_radius = zeros(Float64, voids_max)  # void radii

    in_void = zeros(Int8,dims,dims,dims)  # void membership cell flag
    delta_v = Array{Float64}(undef,dims3)  # underdense cells density contrasts 
    IDs = Array{Int64}(undef,dims3)  # underdense cell IDs

    Nvoids = zeros(Int64,r_bins)
    vsf = zeros(Float64,r_bins-1)
    r_bin_centres = zeros(Float64,r_bins-1)


    # create output folder
    if !isdir(par.output_folder)
        mkdir(par.output_folder)
    end


    # find voids at each input radius R
    voids_total::Int8 = 0  # total number of voids found
    expected_filling_frac::Float64 = 0.0
    for (q,R) in enumerate(Radii)  
        if par.verbose
            @printf("\nSmoothing galaxy density field with top-hat filter of radius %.2f...\n", R)
        end
        delta_sm = smoothing(delta, R, res,"top-hat")
        ## IS SMOOTHING SIGMA CORRECT??
        if minimum(delta_sm)>threshold
            @printf("No cells with delta < %.2f\n", threshold)
            continue
        end  

        # find underdense cells
        local_voids = 0
        for i = 1:dims
            for j = 1:dims
                for k = 1:dims

                    if delta_sm[i,j,k]<threshold && in_void[i,j,k]==0
                        local_voids += 1
                        IDs[local_voids] = dims2*(i-1) + dims*(j-1) + (k-1)
                        delta_v[local_voids] = delta_sm[i,j,k]
                    end

                end
            end
        end
        @printf("Found %d cells with delta < %.2f\n", local_voids, threshold)

        # sort cells underdensities
        indx = sortperm(delta_v[1:local_voids])
        delta_v[1:local_voids] = delta_v[indx]
        IDs[1:local_voids] = IDs[indx]

        R_grid = R/res
        R_grid2 = R_grid*R_grid
        Ncells = floor(Int64,R_grid + 1)

        if voids_total<(2*Ncells+1)^3
            mode = 0
        else
            mode = 1
        end

        @printf("Identifying nearby voids using mode %d...\n", mode)
        # identify if underdense cells belong to previously detected voids
        voids_with_R = 0  # voids found with radius R
        for ID in IDs[1:local_voids]
            i,j,k = (ID÷dims2, (ID%dims2)÷dims, (ID%dims2)%dims) .+ 1

            # if cell belong to a void, continue
            if in_void[i,j,k] == 1
                continue
            end

            if mode==0
                nearby_voids = nearby_voids1(voids_total, dims, middle, i, j, k, void_radius, void_pos, R_grid, par.max_overlap_frac)
            else
                nearby_voids = nearby_voids2(Ncells, dims, i, j, k, R_grid, R_grid2, in_void, par.max_overlap_frac)
            end


            # if new void found
            if nearby_voids == 0
                voids_with_R += 1
                voids_total += 1

                void_pos[voids_total,:] .= i,j,k
                void_radius[voids_total] = R_grid

                # flag cells belonging to new void
                in_void[i,j,k] = 1
                mark_void_region!(Ncells, dims, i, j, k, R_grid2, in_void)

            end

        end

        Nvoids[q] = voids_with_R

        @printf("Found %d voids with radius R = %.3f Mpc/h\n",voids_with_R,R)
        if par.verbose 
            @printf("Found %d voids with radius R >= %.3f Mpc/h\n",voids_total,R)
            @printf("Void volume filling fraction = %.3f\n",sum(in_void)/dims3)
            if par.max_overlap_frac == 0
                expected_filling_frac += voids_with_R*4*pi*R^3/(3*par.box_length^3)
                @printf("Expected filling fraction = %.3f\n",expected_filling_frac)
            end
        end
            
    end

    @printf("\nFound a total of %d voids\n",voids_total)
    @printf("Void volume filling fraction = %.3f\n",sum(in_void)/dims3)

    # compute the void size function (# of voids/Volume/dR)
    for i = 1:r_bins-1
        vsf[i] = Nvoids[i]/(par.box_length^3*(Radii[i]-Radii[i+1]))
        r_bin_centres[i] = 0.5*(Radii[i]+Radii[i+1])
    end

    # ouput data as a struct instance
    VoidData("Spherical",
             void_pos[1:voids_total,:]*res,
             void_radius[1:voids_total]*res,
             vsf,
             r_bin_centres)

end


end
