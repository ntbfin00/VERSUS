module SphericalVoids  # create module to be loaded in main file

# package import
using Printf
using ImageFiltering
using FFTW  # don't need if using ImageFiltering

function get_params(par::Main.VoidParameters.VoidParams)
    for field in propertynames(par)
        println(field, " = ", getfield(par, field))
    end
end

# ==== may want in separate module (also used by voxel) ===== #

# function density_field(cat::Main.Tracers.TracerCat)
    # # bin galaxies to determine density field delta
    # # copy from fastmodules.allocate_gal_cic
    
    # @printf("%i tracers found", cat.size)
# end

function smoothing(delta::Array{Float64,3},r_scale::Float64,binsize::Float64,filter::String)
    # mesh delta must be created first by binning galaxies
    # "reflect" uses convolution instead of correlation

    # r_scale is the smoothing scale in Mpc/h
    # binsize is the voxel size in Mpc/h
    
    r_grid = r_scale/binsize
    if filter == "gaussian"
        kern = Kernel.gaussian((r_grid,r_grid,r_grid))
    # for spherical top hat, better in separate function??
    else
        d = 2*trunc(Int,r_grid) + 1  # Does Pylians need integer values for radii?
        kern = ones(Float64,(d,d,d))/d^3
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

function run_voidfinder(delta::Array{Float64,3},radii::Array{Float64,1},par::Main.VoidParameters.VoidParams)
    println(" ==== Starting the void-finding with spherical-based method ==== ")

    # 1) Smooth density field with spherical top-hat filter of radius R (starting with largest R)
    # 2) Find cells with density lower than threshold (starting with lowest density)
    # 3) Cell determined to be centre of void with radius R if does not overlap with previously determined void, otherwise discarded
    # 4) This is performed for all cells lower than threshold 
    # 5) Then repeated with next largest smoothing radius

    dims = size(delta,1)
    dims2 = dims^2
    dims3 = dims^3

    # set minimum overdensity threshold
    # IS THIS CORRECT??
    threshold = par.min_dens_cut - 1

    # initialize arrays
    in_void = zeros(Int8,dims,dims,dims)
    delta_v = Array{Float32}(undef,dims3)
    IDs = Array{Int64}(undef,dims3)

    # sort radii from largest to smallest
    Radii = sort(radii, rev=true)

    # create output folder
    if !isdir(par.output_folder)
        mkdir(par.output_folder)
    end


    if par.is_box
        # measure the galaxy density field
        if par.verbose
            println("Allocating galaxies in cells...")
        end
        ## sys.stout.flush()
        # delta = density_field(cat)
        for R in Radii  # input void radii
            if par.verbose
                @printf("\nSmoothing galaxy density field with top-hat filter of radius %.2f...\n", R)
            end
            ## sys.stout.flush()
            delta_sm = smoothing(delta, R, par.box_length/dims,"top-hat")
            ## IS SMOOTHING SIGMA CORRECT??
            if minimum(delta_sm)>threshold
                @printf("No cells with delta < %.2f\n", threshold)
                continue
            end  

            # find underdense cells
            local_voids = 0
            for i in 1:dims
                for j in 1:dims
                    for k in 1:dims

                        if delta_sm[i,j,k]<threshold && in_void[i,j,k]==0
                            local_voids += 1
                            IDs[local_voids] = dims2*i + dims*j + k
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

            R_grid = R*dims/par.box_length

            # ln198 

            for ID in IDs[1:local_voids]
                i,j,k = ID÷dims2, (ID%dims2)÷dims, (ID%dims2)%dims

                # if cell belong to a void, continue
                if in_void[i,j,k] == 1
                    continue
                end



            end

         end
                
     end

    rhog = [0.]
    mask_cut = []


end


end
