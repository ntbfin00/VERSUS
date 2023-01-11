include("voidparameters.jl")
include("sphericalvoids.jl")  # load file into global scope

using .VoidParameters
using .SphericalVoids  # load module to use struct

#=============== Input parameters (not default) ===============#

new_params = (is_box=true,
              output_folder = "revolver_test/",
              verbose = true,
              box_length = 1.0,
              tracer_file = "rand_dist.npy",
              tracer_file_type = 2,
              do_recon = false,
              nthreads = 1)
              # max_overlap_frac=0.5)

# =========================================== #

# initialise parameter struct 
par = VoidParams(;new_params...)

# get overdensity field
using NPZ
delta = npzread("data/delta.npy")

# list input parameters
SphericalVoids.get_params(par)

# run voidfinder 
R = [0.1,0.2,0.3]  # input radii could be moved to parameters??
spherical_voids = SphericalVoids.run_voidfinder(delta, R, par)

# voidfinder output
println("\n ==== Output ==== ")
println("void type = ", spherical_voids.type)
println("void positions = ", spherical_voids.positions)
println("void radii = ", spherical_voids.radii)
println("void size function = ", spherical_voids.vsf," at R = ", spherical_voids.rbins)


# ==================================== #
# ============ TESTING =============== #

# R = 0.2
# delta_sm = SphericalVoids.smoothing(delta,R,0.05,"top-hat")
# npzwrite("data/delta_sm.npy",delta_sm)

# a = SphericalVoids.void_overlap_frac(5.0,5.0,110.0)
# b = SphericalVoids.void_overlap_frac(5.0,5.0,100.0)
# c = SphericalVoids.void_overlap_frac(2.0,2.0,sqrt(0.69)*2)
# println("expect 0 -> ",a,"expect 0 -> ",b,"expect 0.5 -> ",c)

