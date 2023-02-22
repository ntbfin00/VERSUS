include("voidparameters.jl")
include("meshbuilder.jl")
include("sphericalvoids.jl")  # load file into global scope

using .VoidParameters
using .MeshBuilder
using .SphericalVoids  # load module to use struct
using ArgParse
using FITSIO
# using HDF5
using YAML
using DelimitedFiles

"""
Read input data and randoms files in FITS. Sky positions are converted to cartesian positions.
"""
function read_input(build_mesh::Bool, data_format::String, data_cols::Array{String,1}, fn::String)
    # hdf5_ext = [".hdf", ".h4", ".hdf4", ".he2", ".h5", ".hdf5", ".he5", ".h5py"]                                                          
    wts_supplied = 0

    # if galaxies/randoms are supplied as input
    if build_mesh
        if endswith(fn, ".fits")
            f = FITS(fn, "r")
            N = read_header(f[2])["NAXIS2"]
            pos = Array{AbstractFloat}(undef,N,3)
            for (i,cols) in enumerate(data_cols)
                if i<4
                    pos[:,i] = read(f[2], cols)
                else
                    wts_supplied = 1
                    wts = read(f[2], cols)
                end
            end
            close(f)
        # elseif any(endswith.(fn, hdf5_ext))
        else
            throw(ErrorException("Input file format not recognised. Allowed format is .fits"))
        end

        pos = cartesian(pos, data_format)

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
        # elseif any(endswith.(fn, hdf5_ext))
        else
            throw(ErrorException("Input file format not recognised. Allowed formats is .fits"))
        end

        return delta 
    end

end

"""
Save void catalogue to file in FITS or txt format.
"""
function save_void_cat(output_folder::String, output_type::String, fn::String, void_cat::Main.SphericalVoids.VoidData)
    # create output folder
    if !isdir(output_folder)
        mkdir(output_folder)
    end 

    out_file = output_folder * fn
    println("\nWriting void catalogue to file...")
    if output_type == "fits"
        data = Dict("positions" => void_cat.positions, "radii" => void_cat.radii)
        f = FITS(out_file * ".fits", "w")
        write(f, data)
        close(f)
        vsf = Dict("r_bins" => void_cat.vsf[:,1], "vsf" => void_cat.vsf[:,2])
        g = FITS(out_file * "_vsf.fits", "w")
        write(g, vsf)
        close(g)
    elseif output_type == "txt"
        writedlm(out_file * "_positions.txt",void_cat.positions)
        writedlm(out_file * "_radii.txt",void_cat.radii)
        writedlm(out_file * "_vsf.txt",void_cat.vsf)
    else
        throw(ErrorException("Output file format not recognised. Allowed formats are .fits and .txt"))
    end
    println("Void catalogue written to " * output_folder)
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

    # remove nothing values
    filter!(i -> !isnothing(i.second), config["input"])
    filter!(i -> !isnothing(i.second), config["output"])
    filter!(i -> !isnothing(i.second), config["mesh"])
    filter!(i -> !isnothing(i.second), config["spherical_voids"])
    
    data_format = get(config["input"], "data_format", "xyz")
    data_cols = get(config["input"], "data_cols", ["x","y","z"])
    build_mesh = get(config["input"], "build_mesh", true)
    do_recon = get(config["input"], "do_recon", false)
    run_spherical_vf = get(config["input"], "run_spherical_vf", true)
    output_type = get(config["output"], "output_type", "fits")
    output_folder = get(config["output"], "output_folder", "output/")

    if any(keys(config) .== "mesh")
        mesh = Dict()
        [mesh[Symbol(key)] = value for (key,value) in config["mesh"]]
        mesh_settings = MeshParams(;mesh...)
    else
        mesh_settings = MeshParams()
    end
    if run_spherical_vf
        sph_voids = Dict()
        [sph_voids[Symbol(key)] = value for (key,value) in config["spherical_voids"]]
        par_sph = SphericalVoidParams(;sph_voids...)
    end
else
    throw(ErrorException("Config file format not recognised. Allowed format is .yaml"))
end

# read input data
if build_mesh
    println("\nReading galaxy position data...")
    gal_data = read_input(build_mesh, data_format, data_cols, args["data"])
    if args["randoms"] == nothing
        cat = GalaxyCatalogue(gal_data...)
    else
        println("\nReading randoms position data...")
        rand_data = read_input(build_mesh, data_format, data_cols, args["randoms"])
        cat = GalaxyCatalogue(gal_data..., rand_data...)
    end
else
    println("Reading density mesh...")
    mesh = read_input(build_mesh, data_format, data_cols, args["data"])
end


# run optional reconstruction
if build_mesh && do_recon
    cat = reconstruction(cat, mesh_settings)
end

if run_spherical_vf
    # list input parameters
    SphericalVoids.get_params(par_sph)
    # run voidfinder 
    if build_mesh
        spherical_voids = SphericalVoids.voidfinder(cat, mesh_settings, par_sph)
    else
        spherical_voids = SphericalVoids.voidfinder(mesh, mesh_settings.box_length, mesh_settings.box_centre, par_sph)
    end
    save_void_cat(output_folder, output_type, par_sph.output_fn, spherical_voids)
else
    throw(ErrorException("No void finder selected. Cannot proceed."))
end

