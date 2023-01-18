include("voidparameters.jl")
include("meshbuilder.jl")
include("sphericalvoids.jl")  # load file into global scope

using .VoidParameters
using .MeshBuilder
using .SphericalVoids  # load module to use struct

using NPZ

#=============== Input parameters (not default) ===============#

new_params = (is_box=true,
              output_folder = "revolver_test/",
              verbose = true,
              box_length = 1.0,
              nbins = 100,
              tracer_file = "rand_dist.npy",
              tracer_file_type = 2,
              do_recon = false,
              use_parallel=true)
              # max_overlap_frac=0.3)

# =========================================== #

# initialise parameter struct 
par = VoidParams(;new_params...)

# create density mesh
pos = npzread("data/rand_dist.npy")
weights = ones(size(pos))
cat = GalaxyCatalogue(pos, weights)
delta = create_mesh(par, cat, 0.015)  # is nbins in params correct?

# delta = npzread("data/delta.npy")

# list input parameters
SphericalVoids.get_params(par)

# run voidfinder 
Radii = [0.1,0.2,0.3]  # input radii could be moved to parameters??
spherical_voids = SphericalVoids.run_voidfinder(delta, Radii, par)

# voidfinder output
# println("\n ==== Output ==== ")
# println("void type = ", spherical_voids.type)
# println("void positions = ", spherical_voids.positions)
# println("void radii = ", spherical_voids.radii)
# println("void size function = ", spherical_voids.vsf," at R = ", spherical_voids.rbins)


# ==================================== #
# ============ TESTING =============== #
using BenchmarkTools

dims = size(delta,1)
middle = dims÷2
res = par.box_length/dims
R = Radii[3]
# R_grid = R/res
# R_grid2 = R_grid*R_grid
# Ncells = floor(Int64,R_grid + 1)
# in_void = npzread("data/in_void.npy")
# void_pos = round.(Int64,spherical_voids.positions/res)
# void_radius = spherical_voids.radii
# voids_total::Int32 = size(void_radius,1)
# i,j,k = (23,31,96)
# println("in_void=",in_void[i,j,k])

# @btime SphericalVoids.nearby_voids1(voids_total, dims, middle, i, j, k, void_radius, void_pos, R_grid, par.max_overlap_frac)
# @btime SphericalVoids.nearby_voids2(Ncells, dims, i, j, k, R_grid, R_grid2, in_void, par.max_overlap_frac)
# @btime SphericalVoids.nearby_voids2(Ncells, dims, i, j, k, R_grid, R_grid2, in_void, par.max_overlap_frac,par.use_parallel)
# @btime SphericalVoids.mark_void_region!(Ncells, dims, i, j, k, R_grid2, in_void)
# @btime SphericalVoids.mark_void_region!(Ncells, dims, i, j, k, R_grid2, in_void,par.use_parallel)

# @btime SphericalVoids.smoothing(delta, dims, middle, R, par.box_length)

# R = 0.2
# delta_sm = SphericalVoids.smoothing(delta,R,0.05,"top-hat")
# npzwrite("data/delta_sm.npy",delta_sm)

# a = SphericalVoids.void_overlap_frac(5.0,5.0,110.0)
# b = SphericalVoids.void_overlap_frac(5.0,5.0,100.0)
# c = SphericalVoids.void_overlap_frac(2.0,2.0,sqrt(0.69)*2)
# println("expect 0 -> ",a,"expect 0 -> ",b,"expect 0.5 -> ",c)


