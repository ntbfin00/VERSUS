module Utils
export setup_logging, read_input

include("voidparameters.jl")

using .VoidParameters
using LoggingExtras
using PyCall
using FITSIO

"""
Setup the logging with timer. "level" can be used to change the depth of logging.
"""
function setup_logging(level::String = "info")
    if level == "debug"
        level = Logging.Debug
    elseif level == "warn"
        level = Logging.Warn
    else
        level = Logging.Info
    end

    t0 = time()

    timestamp_logger(logger) = TransformerLogger(logger) do log
        t = round(time() - t0, digits=3)
        merge(log, (; message = "$(log._module) | ln$(log.line) | $(t)s | $(log.message)"))
    end

    ConsoleLogger(stdout, level) |> timestamp_logger |> global_logger

    return ConsoleLogger()
end

"""
Convert sky positions and redshift to cartesian coordinates.
"""
function to_cartesian(cosmo::Main.VoidParameters.Cosmology, pos::Array{<:AbstractFloat,2}; angle::String="degrees")
    @info "Converting sky positions to cartesian"

    if angle == "degrees"
        @debug "Angle in degrees"
        degree = true
    elseif angle == "radians"
        @debug "Angle in radians"
        degree = false
    else
        throw(ErrorException("Angle type must be either 'degrees' or 'radians'."))
    end

    np = pyimport("numpy")
    cosmology = pyimport("astropy.cosmology")
    utils = pyimport("pyrecon.utils")

    # compute distances from redshifts
    @debug "Setting cosmology"
    c = cosmology.LambdaCDM(H0=cosmo.h*100, Om0=cosmo.omega_m, Ode0=cosmo.omega_l)

    ra = np.array(pos[:,1])
    dec = np.array(pos[:,2]) 
    z = np.array(pos[:,3])

    @debug "Calculating comoving distances"
    dist = c.comoving_distance(z)

    @debug "Converting to cartesian"
    utils.sky_to_cartesian(dist,ra,dec, degree=degree)
end

"""
Read input data and randoms FITS files. Sky positions are converted to cartesian positions.
"""
function read_input(fn::String, build_mesh::Bool, data_format::String, data_cols::Array{String,1}, cosmo::Main.VoidParameters.Cosmology)
    # hdf5_ext = [".hdf", ".h4", ".hdf4", ".he2", ".h5", ".hdf5", ".he5", ".h5py"]                                                          
    wts_supplied = 0

    # if galaxies/randoms are supplied as input
    if build_mesh
        if endswith(fn, ".fits")
            f = FITS(fn, "r")
            N = read_header(f[2])["NAXIS2"]
            pos = Array{AbstractFloat}(undef,N,3)
            wts = Array{AbstractFloat}(undef,N)
            for (i,col) in enumerate(data_cols)
                if i<4
                    pos[:,i] = read(f[2], col)
                else
                    wts_supplied = 1
                    wts = read(f[2], col)
                end
            end
            close(f)
        # elseif any(endswith.(fn, hdf5_ext))
        else
            throw(ErrorException("Input file format not recognised. Allowed format is .fits"))
        end

        if data_format == "xyz"
            @info "Input format: Cartesian"
            # println("Input format: Cartesian.")
        elseif data_format == "rdz"
            @info "Input format: Sky"
            # println("Input format: Sky, converting to cartesian...")
            pos = to_cartesian(cosmo, pos)
        else
            throw(ErrorException("Position data format not recognised. Only formats 'xyz' (cartesian) or 'rdz' (sky) allowed."))
        end

        if wts_supplied == 0 
            return pos, ones(size(pos,1))
        else
            @info "Weights supplied"
            # println("Weights supplied.")
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

end
