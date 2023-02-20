module MeshBuilder

export GalaxyCatalogue, cartesian, reconstruction, create_mesh

using PyCall
using FITSIO

np = pyimport("numpy")
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
function set_recon_engine(cat::Main.MeshBuilder.GalaxyCatalogue,mesh::Main.VoidParameters.MeshParams)

    # determine box size from input
    if mesh.is_box
        los = mesh.los
        boxsize = [mesh.box_length, mesh.box_length, mesh.box_length]
        boxcenter = mesh.box_centre
        boxpad = 1.
        pos = nothing
    # determine box size from random positions and padding
    else 
        los = nothing
        boxsize = nothing
        boxcenter = nothing
        boxpad = mesh.padding
        pos = np.array(cat.rand_pos)
    end

    if mesh.recon_alg == "IFFTparticle"
        recon = pyimport("pyrecon.iterative_fft_particle")
        rec = recon.IterativeFFTParticleReconstruction(f=mesh.f, bias=mesh.bias, los=los, nmesh=mesh.nbins, boxsize=boxsize, boxcenter=boxcenter, boxpad=boxpad, positions=pos, wrap=true, dtype=mesh.dtype, nthreads=Threads.nthreads())
        data = "data"
    elseif mesh.recon_alg == "IFFT"
        recon = pyimport("pyrecon.iterative_fft")
        rec = recon.IterativeFFTReconstruction(f=mesh.f, bias=mesh.bias, los=los, nmesh=mesh.nbins, boxsize=boxsize, boxcenter=boxcenter, boxpad=boxpad, positions=pos, wrap=true, dtype=mesh.dtype, nthreads=Threads.nthreads())
        data = np.array(cat.gal_pos)
    elseif mesh.recon_alg == "MultiGrid"
        recon = pyimport("pyrecon.multigrid")
        rec = recon.MultiGridReconstruction(f=mesh.f, bias=mesh.bias, los=los, nmesh=mesh.nbins, boxsize=boxsize, boxcenter=boxcenter, boxpad=boxpad, positions=pos, wrap=true, dtype=mesh.dtype, nthreads=Threads.nthreads())
        data = np.array(cat.gal_pos)
    else
        throw(ErrorException("Reconstruction algorithm not recognised. Allowed algorithms are IFFTparticle, IFFT and MultiGrid."))
    end

    return rec, data

end


"""
Run reconstruction on galaxy positions. Returns a GalaxyCatalogue object.
"""
function reconstruction(cat::Main.MeshBuilder.GalaxyCatalogue, mesh::Main.VoidParameters.MeshParams)
    println("\n ==== Density field reconstruction ==== ")

    if !mesh.is_box && size(cat.rand_pos,1) == 0
        throw(ErrorException("is_box is set to false but no randoms have been supplied."))
    end

    rec, data = set_recon_engine(cat, mesh)
    gal_pos = np.array(cat.gal_pos)

    println("Assigning galaxies to grid...")
    if size(cat.gal_wts,1) == 0
        gal_pos = np.array(cat.gal_pos)
        rec.assign_data(gal_pos)
    else
        gal_wts = np.array(cat.gal_wts)
        rec.assign_data(gal_pos, gal_wts)
    end

    rec.set_density_contrast(smoothing_radius=mesh.r_smooth)

    println("Running reconstruction with ", mesh.recon_alg, "...")
    rec.run()
    rec_gal_pos = rec.read_shifted_positions(data, field="rsd")
    println("Galaxy positions reconstructed.")
    
    GalaxyCatalogue(rec_gal_pos, cat.gal_wts, cat.rand_pos, cat.rand_wts)

end


"""
Create density mesh from galaxy and random positions. Returns 3D density mesh, box length and box centre.
"""
function create_mesh(cat::Main.MeshBuilder.GalaxyCatalogue, mesh::Main.VoidParameters.MeshParams)
    println("\n ==== Creating density mesh ==== ")

    if !mesh.is_box && size(cat.rand_pos,1) == 0
        throw(ErrorException("is_box is set to false but no randoms have been supplied."))
    end

    rec = set_recon_engine(cat, mesh)[1]

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
        
    return delta, rec.boxsize[1], rec.boxcenter

end


end
