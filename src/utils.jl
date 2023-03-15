module Utils
export setup_logging

# using LoggingExtras, Dates

# const date_format = "yyyy-mm-dd HH:MM:SS"

# timestamp_logger(logger) = TransformerLogger(logger) do log
    # merge(log, (; message = "$(Dates.format(now(), date_format)) $(log._module) $(log.message)"))
# end

# ConsoleLogger(stdout, Logging.Debug) |> timestamp_logger |> global_logger

using LoggingExtras

"""
Setup the logging with timer. "level" can be used to change the depth of logging.
"""
function setup_logging(; level="info")
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

# """
# Return input parameters specific to spherical voids.
# """
# function get_params(cat::Main.MeshBuilder.GalaxyCatalogue, mesh::Main.VoidParameters.MeshParams, par::Main.VoidParameters.SphericalVoidParams)
    # println("\n ==== Spherical void input parameters (galaxy field) ==== ")

    # params = Dict(key => getfield(par, key) for key ∈ fieldnames(Main.VoidParameters.SphericalVoidParams))

    # # determine default voidfinding bins
    # if mesh.nbins_vf == 0
        # r_sep = mean_gal_sep(cat, mesh)
        # nbins = rec.boxsize[1]/(0.5 * r_sep)
        # params[:nbins_vf] = optimal_binning(nbins, "above")
    # else
        # params[:nbins_vf] = mesh.nbins_vf
    # end

    # # determine default radii
    # if par.radii = [0]
        # params[:radii] = [2:10;] * mean_gal_sep(cat, mesh)
    # end

    # for (key, value) in pairs(params)
        # println(key, " = ", value)
    # end

    # # for field in propertynames(params)
            # # println(field, " = ", getfield(params, field))
    # # end 

# end

# function get_params(delta::Array{<:AbstractFloat,3}, par::Main.VoidParameters.SphericalVoidParams)
    # println("\n ==== Spherical void input parameters (density field) ==== ")

    # params = Dict(key => getfield(par, key) for key ∈ fieldnames(Main.VoidParameters.SphericalVoidParams))

    # params[:nbins_vf] = size(delta,1)
    
    # for (key, value) in pairs(params)
        # println(key, " = ", value)
    # end

# end

end
