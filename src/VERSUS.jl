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
using DelimitedFiles

"""
Read input data and randoms files in FITS or hdf5 format. Sky positions will are converted to cartesian positions.
"""
function read_input(input::VoidParameters.InputParams, fn::String)
    hdf5_ext = [".hdf", ".h4", ".hdf4", ".he2", ".h5", ".hdf5", ".he5", ".h5py"]                                                          
    wts_supplied = 0

    # if galaxies/randoms are supplied as input
    if input_settings.build_mesh
        if endswith(fn, ".fits")
            f = FITS(fn, "r")
            N = read_header(f[2])["NAXIS2"]
            pos = Array{AbstractFloat}(undef,N,3)
            for (i,cols) in enumerate(input.data_cols)
                if i<4
                    pos[:,i] = read(f[2], cols)
                else
                    wts_supplied = 1
                    wts = read(f[2], cols)
                end
            end
            close(f)
        elseif any(endswith.(fn, hdf5_ext))
            f = h5open(fn, "r")
            # pos = read(f)  ## FIX
            close(f)
        else
            throw(ErrorException("Input file format not recognised. Allowed formats are " * append!([".fits"],hdf5_ext)))
        end

        pos = cartesian(pos, input_settings.data_format)

        if wts_supplied == 0 
            return pos, Array{AbstractFloat}(undef,0)
        else
            return pos, wts
        end

    # if density mesh is supplied as input
    else
        if endswith(fn, ".fits")
            f = FITS(fn, "r")
            delta = read(f[1])
            close(f)
        elseif any(endswith.(fn, hdf5_ext))
            f = h5open(fn, "r")
            # pos = read(f)  ## FIX
            close(f)
        else
            throw(ErrorException("Input file format not recognised. Allowed formats are " * append!([".fits"],hdf5_ext)))
        end

        return delta 
    end

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
    println("\nWriting void catalogue to file...")
    if output_settings.output_type == "fits"
        data = Dict("positions" => void_cat.positions, "radii" => void_cat.radii)
        f = FITS(out_file * ".fits", "w")
        write(f, data)
        close(f)
        vsf = Dict("r_bins" => void_cat.vsf[:,1], "vsf" => void_cat.vsf[:,2])
        g = FITS(out_file * "_vsf.fits", "w")
        write(g, vsf)
        close(g)
    elseif output_settings.output_type == "txt"
        writedlm(out_file * "_positions.txt",void_cat.positions)
        writedlm(out_file * "_radii.txt",void_cat.radii)
        writedlm(out_file * "_vsf.txt",void_cat.vsf)
    else
        throw(ErrorException("Output file format not recognised. Allowed formats are .fits and .txt"))
    end
    println("Void catalogue written to " * output.output_folder)
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
    println("\nReading galaxy position data...")
    gal_data = read_input(input_settings, args["data"])
    if args["randoms"] == nothing
        cat = GalaxyCatalogue(gal_data...)
    else
        println("\nReading randoms position data...")
        rand_data = read_input(input_settings, args["randoms"])
        cat = GalaxyCatalogue(gal_data..., rand_data...)
    end
    delta = create_mesh(cat, mesh_settings, input_settings)
else
    println("Reading density mesh...")
    delta = read_input(input_settings, args["data"])
end

if input_settings.run_spherical_vf
    # list input parameters
    SphericalVoids.get_params(par_sph)
    # run voidfinder 
    spherical_voids = SphericalVoids.run_voidfinder(delta, input_settings, par_sph)
    save_void_cat(output_settings, par_sph.output_fn, spherical_voids)
else
    throw(ErrorException("No void finder selected. Cannot proceed."))
end

