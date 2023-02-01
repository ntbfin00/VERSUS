include("voidparameters.jl")
include("meshbuilder.jl")
include("sphericalvoids.jl")  # load file into global scope

using .VoidParameters
using .MeshBuilder
using .SphericalVoids  # load module to use struct

using ArgParse
using FITSIO
using HDF5
using YAML
using NPZ  #REMOVE

"""
Read input data and randoms files in FITS or hdf5 format.
"""
function read_input(fn)
    hdf5_ext = [".hdf", ".h4", ".hdf4", ".he2", ".h5", ".hdf5", ".he5", ".h5py"]                                                          
    if endswith(fn, ".fits")
        f = FITS(fn)
        pos = read(f,["x","y","z"])
        wts = read(f,"weights")
    elseif any(endswith.(fn, hdf5_ext))
        f = h5open(fn, "r")
        pos = read(f)
        close(f)
    else
        throw(ErrorException("Input file format not recognised. Allowed formats are .fits, $hdf5_ext"))
    end

    return pos, wts
end

s = ArgParseSettings()
@add_arg_table! s begin
    "--data"
        help = "galaxy positions and weights"
        required = true
        arg_type = String
    "--randoms"
        help = "random positions and weights"
        arg_type = String
    "--config"
        help = "optional configuration YAML file"
        arg_type = String
end

args = parse_args(ARGS, s)

# read input data
# gal_pos, gal_wts = read_input(args["data"])
# if args["randoms"] == nothing
    # cat = GalaxyCatalogue(gal_pos, gal_wts)
# else
    # rand_pos, rand_wts = read_input(args["randoms"])
    # cat = GalaxyCatalogue(gal_pos, gal_wts, rand_pos, rand_wts)
# end

# initialise parameter struct 
if args["config"] == nothing
    par = VoidParams()
elseif endswith(args["config"], ".yaml")
    config = YAML.load_file(args["config"])
    new_params = Dict()
    [new_params[Symbol(key)] = value for (key,value) in config]
    par = VoidParams(;new_params...)
else
    throw(ErrorException("Config file format not recognised. Allowed format is .yaml"))
end

# create density mesh
# delta = create_mesh(par, cat, 0.0)  #is nbins in params correct?

# npzwrite("data/delta_pyrecon_"*string(par.nbins)*".npy",delta)
delta = npzread("data/delta_pyrecon_500.npy")
delta = convert(Array{Float32,3},delta) 

# list input parameters
SphericalVoids.get_params(par)

# run voidfinder 
Radii = [0.1]#,0.2,0.3]  # input radii could be moved to parameters??
using BenchmarkTools
@btime spherical_voids = SphericalVoids.run_voidfinder(delta, Radii, par)

# voidfinder output
# println("\n ==== Output ==== ")
# println("void type = ", spherical_voids.type)
# println("void positions = ", spherical_voids.positions)
# println("void radii = ", spherical_voids.radii)
# println("void size function = ", spherical_voids.vsf," at R = ", spherical_voids.rbins)


