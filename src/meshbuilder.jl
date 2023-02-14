module MeshBuilder

export GalaxyCatalogue, cartesian, reconstruction, create_mesh

using PyCall
using FITSIO

np = pyimport("numpy")
recon = pyimport("pyrecon.iterative_fft_particle")
utils = pyimport("pyrecon.utils")


"""
GalaxyCatalogue.gal_pos - galaxy positions (x,y,z)
GalaxyCatalogue.gal_wts - galaxy weights
GalaxyCatalogue.rand_pos - random positions (x,y,z)
GalaxyCatalogue.rand_wts - random weights

Weights and randoms set to size 0 matrix when not specified.
"""
mutable struct GalaxyCatalogue
    gal_pos::Array{AbstractFloat,2}
    gal_wts::Array{AbstractFloat,1}
    rand_pos::Array{AbstractFloat,2}
    rand_wts::Array{AbstractFloat,1}
    function GalaxyCatalogue(gal_pos,gal_wts=Array{AbstractFloat}(undef,0),rand_pos=Array{AbstractFloat}(undef,0,0),rand_wts=Array{AbstractFloat}(undef,0))
        new(gal_pos,gal_wts,rand_pos,rand_wts)
    end
end

function cartesian(positions::Array{AbstractFloat,2}, format::String)
    if format == "xyz"
        println("Cartesian positions provided.")
        return positions
    elseif format == "rdz"
        println("Converting sky positions to cartesian...")
        return utils.sky_to_cartesian(positions[:,3],positions[:,1],positions[:,2])
    else
        throw(ErrorException("Position data format not recognised. Only formats 'xyz' (cartesian) or 'rdz' (sky) allowed."))
    end

end


"""
Set algorithm for reconstruction.
"""
function set_recon_engine(mesh::Main.VoidParameters.MeshParams,input::Main.VoidParameters.InputParams)

    if mesh.recon_alg == "IFFTparticle"
        rec = recon.IterativeFFTParticleReconstruction(f=mesh.f,bias=mesh.bias,nmesh=mesh.nbins, boxsize=[input.box_length,input.box_length,input.box_length], boxcenter=input.box_centre, boxpad=mesh.padding, dtype=mesh.dtype, nthreads=Threads.nthreads())
        data = ["data", "data"]
    elseif mesh.recon_alg == "IFFT"
        rec = recon.IterativeFFTReconstruction(f=mesh.f,bias=mesh.bias,nmesh=mesh.nbins, boxsize=[input.box_length,input.box_length,input.box_length], boxcenter=input.box_centre, boxpad=mesh.padding, dtype=mesh.dtype, nthreads=Threads.nthreads())
        data = [cat.gal_pos, cat.gal_rand]
    elseif mesh.recon_alg == "MultiGrid"
        rec = recon.MultiGridReconstruction(nmesh=mesh.nbins, boxsize=[input.box_length,input.box_length,input.box_length], boxcenter=input.box_centre, boxpad=mesh.padding, dtype=mesh.dtype, nthreads=Threads.nthreads())
        set_cosmo(f=mesh.f,bias=mesh.bias)
        data = [cat.gal_pos, cat.gal_rand]
    else
        throw(ErrorException("Reconstruction algorithm not recognised. Allowed algorithms are IFFTparticle, IFFT and MultiGrid."))
    end

    return rec, data

end


"""
Run reconstruction on galaxy and random positions. Returns a GalaxyCatalogue object.
"""
function reconstruction(cat::Main.MeshBuilder.GalaxyCatalogue,mesh::Main.VoidParameters.MeshParams,input::Main.VoidParameters.InputParams)

    println("\n ==== Density field reconstruction ==== ")

    if !mesh.is_box && size(cat.rand_pos,1) == 0
        throw(ErrorException("is_box is set to false but no randoms have been supplied."))
    end

    rec, data = set_recon_engine(mesh, input)
    gal_pos = np.array(cat.gal_pos)

    println("Assigning galaxies to grid...")
    if size(cat.gal_wts,1) == 0
        gal_pos = np.array(cat.gal_pos)
        rec.assign_data(gal_pos)
    else
        gal_wts = np.array(cat.gal_wts)
        rec.assign_data(gal_pos, gal_wts)
    end

    if mesh.is_box
        rec.set_density_contrast(smoothing_radius=mesh.r_smooth)

        println("Running reconstruction...")
        rec.run()
        rec_gal_pos = rec.read_shifted_positions(data[1], field="rsd")
        println("Galaxy positions reconstructed.")
        return GalaxyCatalogue(rec_gal_pos, cat.gal_wts)

    else
        println("Assigning randoms to grid...")
        if size(cat.rand_wts) == 0
            rand_pos = np.array(cat.rand_pos)
            rec.assign_randoms(rand_pos)
        else
            rand_wts = np.array(cat.rand_wts)
            rec.assign_randoms(rand_pos, rand_wts)
        end
        rec.set_density_contrast(smoothing_radius=mesh.r_smooth)

        println("Running reconstruction...")
        rec.run()
        rec_gal_pos = rec.read_shifted_positions(data[1], field="rsd")
        rec_rand_pos = rec.read_shifted_positions(data[2], field="rsd") 
        println("Galaxy and random positions reconstructed.")
        return GalaxyCatalogue(rec_gal_pos, cat.gal_wts, rec_rand_pos, cat.rand_wts)
    end

end


"""
Create density mesh from galaxy and random positions.
"""
function create_mesh(cat::Main.MeshBuilder.GalaxyCatalogue,mesh::Main.VoidParameters.MeshParams,input::Main.VoidParameters.InputParams)
    println("\n ==== Creating density mesh ==== ")

    if !mesh.is_box && size(cat.rand_pos,1) == 0
        throw(ErrorException("is_box is set to false but no randoms have been supplied."))
    end

    rec = set_recon_engine(mesh, input)[1]

    println("Assigning galaxies to grid...")
    if size(cat.gal_wts,1) == 0
        rec.assign_data(cat.gal_pos)
    else
        rec.assign_data(cat.gal_pos, cat.gal_wts)
    end

    if mesh.is_box
        rec.set_density_contrast(smoothing_radius=mesh.r_smooth)

        delta = rec.mesh_delta.value
        println(string(typeof(delta))[7:13]," density mesh set.")

    else
        println("Assigning randoms to grid...")
        if size(cat.rand_wts) == 0
            rec.assign_randoms(cat.rand_pos)
        else
            rec.assign_randoms(cat.rand_pos, cat.rand_wts)
        end
        rec.set_density_contrast(smoothing_radius=mesh.r_smooth)

        delta = rec.mesh_delta.value
        println(string(typeof(delta))[7:13]," density mesh set.")
    end

    # save mesh output
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
