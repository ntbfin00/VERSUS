include("src/voidparameters.jl")
include("src/meshbuilder.jl")
include("src/sphericalvoids.jl")  
include("src/utils.jl")

using .Utils
using .VoidParameters
using .MeshBuilder
using .SphericalVoids  # load module to use struct
using ArgParse
using FITSIO
# using HDF5
using YAML
using DelimitedFiles

"""
Save void catalogue to file in FITS or txt format.
"""
function save_void_cat(fn::String, output_type::String, void_cat::Main.SphericalVoids.VoidData)
    # create output folder
    if !isdir("output/")
        mkdir("output/")
    end 

    out_file = "output/" * fn
    @info "Writing void catalogue to file"
    if output_type == "fits"
        data = [void_cat.positions[:,1],void_cat.positions[:,2],void_cat.positions[:,3],void_cat.radii]
        colnames = ["x", "y", "z", "R"]
        f = FITS(out_file * ".fits", "w")
        write(f, colnames, data)
        close(f)
	vsf = [void_cat.vsf[:,1], void_cat.vsf[:,2], void_cat.vsf[:,3]]
        colnames = ["rmean", "dn/dlnR", "err"]
        g = FITS(out_file * "_vsf.fits", "w")
        write(g, colnames, vsf)
        close(g)
    elseif output_type == "txt"
        writedlm(out_file * "_positions.txt",void_cat.positions)
        writedlm(out_file * "_radii.txt",void_cat.radii)
        writedlm(out_file * "_vsf.txt",void_cat.vsf)
    else
        throw(ErrorException("Output file format not recognised. Allowed formats are .fits and .txt"))
    end
    @info "Void catalogue written to $out_file"
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
        default = nothing
    "--config"
        help = "configuration YAML file"
        required = true
        arg_type = String
end

args = parse_args(ARGS, s)

# setup logging
logger = setup_logging()

# load settings
if endswith(args["config"], ".yaml")
    config = YAML.load_file(args["config"])

    if any(keys(config) .== "input")
        filter!(i -> !isnothing(i.second), config["input"])
        data_format = get(config["input"], "data_format", "xyz")
        data_cols = get(config["input"], "data_cols", ["x","y","z"])
        do_recon = get(config["input"], "do_recon", false)
        run_spherical_vf = get(config["input"], "run_spherical_vf", true)
        build_mesh = get(config["input"], "build_mesh", true)
    else
        data_format = "xyz"
        data_cols = ["x","y","z"]
        do_recon = false
        run_spherical_vf = true
        build_mesh = true
    end

    if any(keys(config) .== "output")
        filter!(i -> !isnothing(i.second), config["output"])
        output_fn = get(config["output"], "output_fn", "void_cat")
        output_type = get(config["output"], "output_type", "fits")
    else
        output_fn = "void_cat"
        output_type = "fits"
    end

    if any(keys(config) .== "cosmo")
        filter!(i -> !isnothing(i.second), config["cosmo"])
        cosmology = Dict()
        [cosmology[Symbol(key)] = value for (key,value) in config["cosmo"]]
        cosmo = Cosmology(;cosmology...)
    else
        cosmo = Cosmology()
    end
    if any(keys(config) .== "mesh")
        filter!(i -> !isnothing(i.second), config["mesh"])
        mesh = Dict()
        [mesh[Symbol(key)] = value for (key,value) in config["mesh"]]
        mesh_settings = MeshParams(;mesh...)
    else
        mesh_settings = MeshParams()
    end
    if run_spherical_vf
        filter!(i -> !isnothing(i.second), config["spherical_voids"])
        sph_voids = Dict()
        [sph_voids[Symbol(key)] = value for (key,value) in config["spherical_voids"]]
        par_sph = SphericalVoidParams(;sph_voids...)
    end
else
    throw(ErrorException("Config file format not recognised. Allowed format is .yaml"))
end

# read input data
if build_mesh
    @info "Reading galaxy position data"
    gal_data = read_input(args["data"], build_mesh, data_format, data_cols, cosmo)
    cat = GalaxyCatalogue(gal_pos = gal_data[1], gal_wts = gal_data[2])
    gal_data = nothing
    GC.gc()

    if args["randoms"] != nothing
        @info "Reading randoms position data"
        rand_data = read_input(args["randoms"], build_mesh, data_format, data_cols, cosmo)
        cat.rand_pos = rand_data[1]
        cat.rand_wts = rand_data[2]
        rand_data = nothing
        GC.gc()
    end
else
    @info "Reading density mesh"
    mesh, box_length, box_centre = read_input(args["data"], build_mesh, data_format, data_cols, cosmo)
end


# run optional reconstruction
if build_mesh && do_recon
    cat = reconstruction(cosmo, cat, mesh_settings)
end

if run_spherical_vf
    # list input parameters
    # SphericalVoids.get_params(par_sph)
    # run voidfinder 
    if build_mesh
        spherical_voids = SphericalVoids.voidfinder(cat, mesh_settings, par_sph)
    else
        spherical_voids = SphericalVoids.voidfinder(mesh, box_length, box_centre, par_sph)
    end
    save_void_cat(output_fn * "_spherical", output_type, spherical_voids)
else
    throw(ErrorException("No void finder selected. Cannot proceed."))
end

