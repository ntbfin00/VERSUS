module MeshBuilder

export GalaxyCatalogue, create_mesh

using PyCall

np = pyimport("numpy")
recon = pyimport("pyrecon.recon")

"""
GalaxyCatalogue.gal_pos - galaxy positions (x,y,z)
GalaxyCatalogue.gal_wts - galaxy weights
GalaxyCatalogue.rand_pos - random positions (x,y,z)
GalaxyCatalogue.rand_wts - random weights

Randoms set to size 0 matrix when not specified.
"""
struct GalaxyCatalogue
    gal_pos::Array{AbstractFloat,2}
    gal_wts::Array{AbstractFloat,2}
    rand_pos::Array{AbstractFloat,2}
    rand_wts::Array{AbstractFloat,2}
    function GalaxyCatalogue(gal_pos,gal_wts,rand_pos=Array{AbstractFloat}(undef,0,0),rand_wts=Array{AbstractFloat}(undef,0,0))
        new(gal_pos,gal_wts,rand_pos,rand_wts)
    end
end

function create_mesh(cat::GalaxyCatalogue,mesh::Main.VoidParameters.MeshParams,input::Main.VoidParameters.InputParams; boxcenter=par.box_length/2)
    println("\n ==== Creating density mesh ==== ")

    if !mesh.is_box && size(cat.rand_pos,1) != 0
        throw(ErrorException("is_box is set to false but no randoms have been supplied."))
    end

    gal_pos = np.array(cat.gal_pos)
    gal_wts = np.array(cat.gal_wts)

    # check if randoms have same weight as data
    rec = recon.BaseReconstruction(nmesh=mesh.nbins, boxsize=input.box_length, boxcenter=boxcenter,dtype=mesh.dtype, nthreads=Threads.nthreads()) # box center alright??

    println("Assigning galaxies to grid...")
    rec.assign_data(gal_pos, gal_wts)

    if mesh.is_box
        rec.set_density_contrast(smoothing_radius=mesh.r_smooth)

        if mesh.do_recon
            println("Running reconstruction...")
            rec.run()
            rec_gal_pos = rec.read_shifted_positions(gal_pos)
            rec.assign_data(rec_gal_pos, gal_wts)
            rec.set_density_contrast(smoothing_radius=mesh.r_smooth)
        end

    else
        rand_pos = np.array(cat.rand_pos)
        rand_wts = np.array(cat.rand_wts)
        println("Assigning randoms to grid...")
        rec.assign_randoms(rand_pos, rand_wts)
        rec.set_density_contrast(smoothing_radius=mesh.r_smooth)

        if mesh.do_recon
            println("Running reconstruction...")
            rec.run()
            rec_gal_pos = rec.read_shifted_positions(gal_pos)
            rec.assign_data(rec_gal_pos, gal_wts)
            rec_rand_pos = rec.read_shifted_positions(rand_pos) # do i need this step?
            rec.assign_randoms(rec_rand_pos, rand_wts)
            rec.set_density_contrast(smoothing_radius=mesh.r_smooth)
        end
    end

    delta = rec.mesh_delta.value
    println(string(typeof(delta))[7:13]," density mesh set.")
    delta

end


end
