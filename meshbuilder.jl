module MeshBuilder

export GalaxyCatalogue, create_mesh

using PyCall
using FITSIO

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
    gal_wts::Array{AbstractFloat,1}
    rand_pos::Array{AbstractFloat,2}
    rand_wts::Array{AbstractFloat,1}
    function GalaxyCatalogue(gal_pos,gal_wts=Array{AbstractFloat}(undef,0),rand_pos=Array{AbstractFloat}(undef,0,0),rand_wts=Array{AbstractFloat}(undef,0))
        new(gal_pos,gal_wts,rand_pos,rand_wts)
    end
end

function create_mesh(cat::GalaxyCatalogue,mesh::Main.VoidParameters.MeshParams,input::Main.VoidParameters.InputParams)
    println("\n ==== Creating density mesh ==== ")

    if !mesh.is_box && size(cat.rand_pos,1) == 0
        throw(ErrorException("is_box is set to false but no randoms have been supplied."))
    end

    # check if randoms have same weight as data
    rec = recon.BaseReconstruction(nmesh=mesh.nbins, boxsize=[input.box_length,input.box_length,input.box_length], boxcenter=input.box_centre, boxpad=mesh.padding, dtype=mesh.dtype, nthreads=Threads.nthreads())

    println("Assigning galaxies to grid...")
    if size(cat.gal_wts,1) == 0
        rec.assign_data(cat.gal_pos)
    else
        rec.assign_data(cat.gal_pos, cat.gal_wts)
    end

    if mesh.is_box
        rec.set_density_contrast(smoothing_radius=mesh.r_smooth)

        if mesh.do_recon
            # println("Running reconstruction...")
            # rec.run()
            # rec_gal_pos = rec.read_shifted_positions(cat.gal_pos)
            # rec.assign_data(rec_gal_pos, cat.gal_wts)
            # rec.set_density_contrast(smoothing_radius=mesh.r_smooth)
        end

    else
        println("Assigning randoms to grid...")
        if size(cat.rand_wts) == 0
            rec.assign_randoms(cat.rand_pos)
        else
            rec.assign_randoms(cat.rand_pos, cat.rand_wts)
        end
        rec.set_density_contrast(smoothing_radius=mesh.r_smooth)

        if mesh.do_recon
            println("Running reconstruction...")
            # rec.run()
            # rec_gal_pos = rec.read_shifted_positions(cat.gal_pos)
            # rec.assign_data(rec_gal_pos, cat.gal_wts)
            # rec_rand_pos = rec.read_shifted_positions(cat.rand_pos) # do i need this step?
            # rec.assign_randoms(rec_rand_pos, cat.rand_wts)
            # rec.set_density_contrast(smoothing_radius=mesh.r_smooth)
        end
    end

    delta = rec.mesh_delta.value
    println(string(typeof(delta))[7:13]," density mesh set.")
    if mesh.save_mesh
        if !isdir("mesh/")
            mkdir("mesh/")
        end 
        fn = "mesh/mesh_" * string(mesh.nbins) * "_" * mesh.dtype * ".fits"
        f = FITS(fn, "w")
        write(f, delta)
        close(f)
        println(fn * " saved to file.")
    end
        
    delta

end


end
