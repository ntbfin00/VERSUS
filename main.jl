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

# =========================================== #

# initialise parameter struct 
par = VoidParams(;params...)

# testing
SphericalVoids.get_params(par)

# find voids
# SphericalVoids.run_voidfinder(par)
