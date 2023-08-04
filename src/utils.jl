module Utils
export setup_logging, read_input, to_cartesian, Atomic

include("voidparameters.jl")

using .VoidParameters
using LoggingExtras
using PyCall
using CFITSIO
using Interpolations

cosmology = pyimport("astropy.cosmology")
utils = pyimport("pyrecon.utils")

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
Convert redshift to comoving radial distance.
"""
function z_to_dist(cosmo::Main.VoidParameters.Cosmology, z::Array{<:AbstractFloat,1})
    @debug "Setting cosmology"
    c = cosmology.LambdaCDM(H0=cosmo.h*100, Om0=cosmo.omega_m, Ode0=cosmo.omega_l)

    @debug "Converting redshifts to distances"
    z_arr = collect(minimum(z):0.01:maximum(z)+1)
    dist = c.comoving_distance(z_arr)
    fit = linear_interpolation(z_arr, dist)

    return fit(z)
end

"""
Convert sky positions and redshift to cartesian coordinates.
"""
function to_cartesian(cosmo::Main.VoidParameters.Cosmology, pos::Array{<:AbstractFloat,2}; angle::String="degrees")
    @info "Converting sky positions to cartesian"

    if angle == "degrees"
        @debug "Angle in degrees"
        degree = true
        conversion = pi/180.
    elseif angle == "radians"
        @debug "Angle in radians"
        degree = false
        conversion = 1.
    else
        throw(ErrorException("Angle type must be either 'degrees' or 'radians'."))
    end

    dist = z_to_dist(cosmo, pos[:,3])

    @debug "Converting to cartesian"
    cos_dec = cos.(pos[:,2] * conversion)
    pos[:,3] = sin.(pos[:,2] .* conversion)
    pos[:,2] = cos_dec .* sin.(pos[:,1] * conversion)
    pos[:,1] = cos_dec .* cos.(pos[:,1] * conversion)

    dist .* pos
    
end

function mem()
    println(Sys.free_memory()/Sys.total_memory())
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
            mem()
            f = fits_open_data(fn)
            mem()
            N = parse(Int,fits_read_keyword(f,"NAXIS2")[1])
            pos = Array{Float32}(undef,N,3)
            wts = Array{Float32}(undef,N)
            mem()
            for (i,col) in enumerate(data_cols)
                if i<4
                    fits_read_col(f,i,1,1,wts)
                    pos[:,i] = wts
                else
                    wts_supplied = 1
                    fits_read_col(f,i,1,1,wts)
                end
            end
            fits_close_file(f)
        # elseif any(endswith.(fn, hdf5_ext))
        else
            throw(ErrorException("Input file format not recognised. Allowed format is .fits"))
        end

        if data_format == "xyz"
            @info "Input format: Cartesian"
        elseif data_format == "rdz"
            @info "Input format: Sky"
            pos = to_cartesian(cosmo, pos)
        else
            throw(ErrorException("Position data format not recognised. Only formats 'xyz' (cartesian) or 'rdz' (sky) allowed."))
        end

        if wts_supplied == 0 
            return pos, ones(size(pos,1))
        else
            @info "Weights supplied"
            return pos, wts
        end

    # if density mesh is supplied as input
    else
        if endswith(fn, ".fits")
            f = FITS(fn, "r")
            delta = read(f[1])
            box_length = read(f[2], "box_length")
            box_centre = read(f[2], "box_centre")
            close(f)
        # elseif any(endswith.(fn, hdf5_ext))
        else
            throw(ErrorException("Input file format not recognised. Allowed formats is .fits"))
        end

        return delta, box_length, box_centre
    end

end

# atomic counter for threading
mutable struct Atomic
    @atomic counter::Int64
end

end
