module MeshBuilder

export GalaxyCatalogue, create_mesh

using PyCall

np = pyimport("numpy")
recon = pyimport("pyrecon.recon")

"""
GalaxyCatalogue.positions - galaxy positions (x,y,z)
GalaxyCatalogue.weights - galaxy weights
GalaxyCatalogue.randoms - random positions (x,y,z)

Randoms set to size 0 matrix when not specified.
"""
struct GalaxyCatalogue
    positions::Array{AbstractFloat,2}
    weights::Array{AbstractFloat,2}
    randoms::Array{AbstractFloat,2}
    function GalaxyCatalogue(positions,weights,randoms=Array{AbstractFloat}(undef,0,0))
        new(positions,weights,randoms)
    end
end

function create_mesh(par::Main.VoidParameters.VoidParams,cat::GalaxyCatalogue,r_sm::Float64; boxcenter=par.box_length/2)
    println("\n ==== Creating density mesh ==== ")

    if !par.is_box && size(cat.randoms,1) != 0
        throw(ErrorException("is_box is set to false but no randoms have been supplied."))
    end

    positions = np.array(cat.positions)
    weights = np.array(cat.weights)

    # check if randoms have same weight as data
    rec = recon.BaseReconstruction(nmesh=par.nbins, boxsize=par.box_length, boxcenter=boxcenter,nthreads=Threads.nthreads()) # box center alright??

    println("Assigning galaxies to grid...")
    rec.assign_data(positions, weights)

    if par.is_box
        rec.set_density_contrast(smoothing_radius=r_sm)

        if par.do_recon
            println("Running reconstruction...")
            rec.run()
            rec_positions = rec.read_shifted_positions(positions)
            rec.assign_data(rec_positions, weights)
            rec.set_density_contrast(smoothing_radius=r_sm)
        end

    else
        randoms = np.array(cat.randoms)
        println("Assigning randoms to grid...")
        rec.assign_randoms(randoms, weights)
        rec.set_density_contrast(smoothing_radius=r_sm)

        if par.do_recon
            println("Running reconstruction...")
            rec.run()
            rec_positions = rec.read_shifted_positions(positions)
            rec.assign_data(rec_positions, weights)
            rec_randoms = rec.read_shifted_positions(randoms) # do i need this step?
            rec.assign_randoms(rec_randoms, cat.weights)
            rec.set_density_contrast(smoothing_radius=r_sm)
        end
    end

    println("Density mesh set.")
    rec.mesh_delta.value

end


end
