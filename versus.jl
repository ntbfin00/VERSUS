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
function read_input(fn::String)
    hdf5_ext = [".hdf", ".h4", ".hdf4", ".he2", ".h5", ".hdf5", ".he5", ".h5py"]                                                          
    if endswith(fn, ".fits")
        f = FITS(fn)
        # pos = read(f[2],"pos")  ## FIX
        # wts = read(f[2],"weights")  ## FIX
    elseif any(endswith.(fn, hdf5_ext))
        f = h5open(fn, "r")
        # pos = read(f)  ## FIX
        close(f)
    else
        throw(ErrorException("Input file format not recognised. Allowed formats are " * append!([".fits"],hdf5_ext)))
    end

    return pos, wts
end

"""
Save void catalogue to file in FITS or txt format.
"""
function save_void_cat(output::VoidParameters.OutputParams, fn::String, void_cat::Main.SphericalVoids.VoidData)
    # create output folder
    if !isdir(output.output_folder)
        mkdir(output.output_folder)
    end 

    out_file = output.output_folder * fn
    println("Writing void catalogue to file...")
    if output_settings.output_type == "fits"
        f = FITS(out_file*".fits", "w")
        data = Dict(String.(fieldnames(Main.SphericalVoids.VoidData)) .=> getfield.(Ref(void_cat), fieldnames(Main.SphericalVoids.VoidData)))
        write(f, data)
        close(f)
    elseif output_settings.output_type == "txt"
        writedlm(out_file*"_positions.txt",void_cat.positions)
        writedlm(out_file*"_radii.txt",void_cat.radii)
        writedlm(out_file*"_vsf.txt",void_cat.vsf)
    else
        throw(ErrorException("Output file format not recognised. Allowed formats are .fits and .txt"))
    end
    println("Void catalogue written to " * out_file)
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
        help = "configuration YAML file"
        required = true
        arg_type = String
end

args = parse_args(ARGS, s)

# load settings
if endswith(args["config"], ".yaml")
    config = YAML.load_file(args["config"])
    if any(keys(config).=="input")
        input = Dict()
        [input[Symbol(key)] = value for (key,value) in config["input"] if value != nothing]
        input_settings = InputParams(;input...)
    else
        input_settings = InputParams()
    end
    if any(keys(config).=="output")
        output = Dict()
        [output[Symbol(key)] = value for (key,value) in config["output"] if value != nothing]
        output_settings = OutputParams(;output...)
    else
        output_settings = OutputParams()
    end
    if any(keys(config).=="mesh")
        mesh = Dict()
        [mesh[Symbol(key)] = value for (key,value) in config["mesh"] if value != nothing]
        mesh_settings = MeshParams(;mesh...)
    else
        mesh_settings = MeshParams()
    end
    if input_settings.run_spherical_vf
        sph_voids = Dict()
        [sph_voids[Symbol(key)] = value for (key,value) in config["spherical_voids"] if value != nothing]
        par_sph = SphericalVoidParams(;sph_voids...)
    end
else
    throw(ErrorException("Config file format not recognised. Allowed format is .yaml"))
end

# read input data
if input_settings.build_mesh
    println("Galaxy positions taken as input.")
    gal_pos, gal_wts = read_input(args["data"])
    if args["randoms"] == nothing
        cat = GalaxyCatalogue(gal_pos, gal_wts)
    else
        rand_pos, rand_wts = read_input(args["randoms"])
        cat = GalaxyCatalogue(gal_pos, gal_wts, rand_pos, rand_wts)
    end
    delta = create_mesh(cat, mesh_settings, input_settings)
else
    println("Density mesh taken as input.")
    # delta = read_input(args["data"])   ## FIX
end

# delta = npzread("data/delta_pyrecon_500.npy")
# delta = convert(Array{Float32,3},delta) 

if input_settings.run_spherical_vf
    # list input parameters
    SphericalVoids.get_params(par_sph)
    # run voidfinder 
    spherical_voids = SphericalVoids.run_voidfinder(delta, input_settings, par_sph)
    save_void_cat(output_settings, par_sph.output_fn, spherical_voids)
else
    throw(ErrorException("No void finder selected. Cannot proceed."))
end

