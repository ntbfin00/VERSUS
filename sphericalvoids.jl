module SphericalVoids  # create module to be loaded in main file

# package import
using Printf

function get_params(par::Main.VoidParameters.VoidParams)
    for field in propertynames(par)
        println(field, " = ", getfield(par, field))
    end
end

function run_voidfinder(par::Main.VoidParameters.VoidParams)
    println(" ==== Starting the void-finding with spherical-based method ==== ")

    print(par.output_folder)
    if !isdir(par.output_folder)
        mkdir(par.output_folder)
    end

    rhog = [0.]
    mask_cut = []

    # @printf("%i tracers found", cat["size"])

end


end
