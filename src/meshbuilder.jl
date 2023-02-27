module MeshBuilder

export GalaxyCatalogue, to_cartesian, reconstruction, create_mesh

using PyCall
using FITSIO

np = pyimport("numpy")
utils = pyimport("pyrecon.utils")
cosmology = pyimport("astropy.cosmology")


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


"""
Convert sky positions and redshift to cartesian coordinates. 
"""
function to_cartesian(cosmo::Main.VoidParameters.Cosmology, pos::Array{<:AbstractFloat,2}; angle::String="degrees")
    if angle == "degrees"
        degree = true
    elseif angle == "radians"
        degree = false
    else
        throw(ErrorException("Angle type must be either 'degrees' or 'radians'."))
    end

    # compute distances from redshifts
    c = cosmology.LambdaCDM(H0=cosmo.h*100, Om0=cosmo.omega_m, Ode0=cosmo.omega_l)
    pos = np.array(pos)
    dist = c.comoving_distance(pos[:,3])
    
    utils.sky_to_cartesian(dist,pos[:,1],pos[:,2], degree=degree)
end


"""
Set algorithm for reconstruction.
"""
function set_recon_engine(cosmo::Main.VoidParameters.Cosmology, cat::Main.MeshBuilder.GalaxyCatalogue, mesh::Main.VoidParameters.MeshParams, nbins::Int)

    # determine box size from input
    if mesh.is_box
        los = mesh.los
        boxsize = mesh.box_length
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

    # remove padding when creating voidfinding mesh
    if nbins == mesh.nbins_vf
        boxpad = 1.
    end

    if mesh.recon_alg == "IFFTparticle"
        recon = pyimport("pyrecon.iterative_fft_particle")
        rec = recon.IterativeFFTParticleReconstruction(f=cosmo.f, bias=cosmo.bias, los=los, nmesh=nbins, boxsize=boxsize, boxcenter=boxcenter, boxpad=boxpad, positions=pos, wrap=true, dtype=mesh.dtype, nthreads=Threads.nthreads())
        data = "data"
    elseif mesh.recon_alg == "IFFT"
        recon = pyimport("pyrecon.iterative_fft")
        rec = recon.IterativeFFTReconstruction(f=cosmo.f, bias=cosmo.bias, los=los, nmesh=nbins, boxsize=boxsize, boxcenter=boxcenter, boxpad=boxpad, positions=pos, wrap=true, dtype=mesh.dtype, nthreads=Threads.nthreads())
        data = np.array(cat.gal_pos)
    elseif mesh.recon_alg == "MultiGrid"
        recon = pyimport("pyrecon.multigrid")
        rec = recon.MultiGridReconstruction(f=cosmo.f, bias=cosmo.bias, los=los, nmesh=nbins, boxsize=boxsize, boxcenter=boxcenter, boxpad=boxpad, positions=pos, wrap=true, dtype=mesh.dtype, nthreads=Threads.nthreads())
        data = np.array(cat.gal_pos)
    else
        throw(ErrorException("Reconstruction algorithm not recognised. Allowed algorithms are IFFTparticle, IFFT and MultiGrid."))
    end

    return rec, data

end


"""
Run reconstruction on galaxy positions. Returns a GalaxyCatalogue object.
"""
function reconstruction(cosmo::Main.VoidParameters.Cosmology, cat::Main.MeshBuilder.GalaxyCatalogue, mesh::Main.VoidParameters.MeshParams)
    println("\n ==== Density field reconstruction ==== ")

    if !mesh.is_box && size(cat.rand_pos,1) == 0
        throw(ErrorException("is_box is set to false but no randoms have been supplied."))
    end

    if !mesh.is_box && mesh.padding <= 1.0
        @warn "Reconstructed positions may be wrapped back into survey region as is_box=false and padding<=1."
    end

    rec, data = set_recon_engine(cosmo, cat, mesh, mesh.nbins_recon)

    if rec.boxsize[1]/mesh.nbins_recon > mesh.r_smooth
        @warn "Smoothing scale is less than cellsize."
    end


    println("Assigning galaxies to grid...")
    gal_pos = np.array(cat.gal_pos)
    if size(cat.gal_wts,1) == 0
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

    cosmo_vf = Main.VoidParameters.Cosmology(; bias=1.)
    rec = set_recon_engine(cosmo_vf, cat, mesh, mesh.nbins_vf)[1]

    println("Assigning galaxies to grid...")
    gal_pos = np.array(cat.gal_pos)
    if size(cat.gal_wts,1) == 0
        rec.assign_data(gal_pos)
    else
        gal_wts = np.array(cat.gal_wts)
        rec.assign_data(gal_pos, gal_wts)
    end

    if mesh.is_box
        rec.set_density_contrast(smoothing_radius=0.0)

        delta = rec.mesh_delta.value
        println(string(typeof(delta))[7:13]," density mesh set.")

    else
        println("Assigning randoms to grid...")
        rand_pos = np.array(cat.rand_pos)
        if size(cat.rand_wts) == 0
            rec.assign_randoms(rand_pos)
        else
            rand_wts = np.array(cat.rand_wts)
            rec.assign_randoms(rand_pos, rand_wts)
        end
        rec.set_density_contrast(smoothing_radius=0.0)

        delta = rec.mesh_delta.value
        println(string(typeof(delta))[7:13]," density mesh set.")
    end

    # save mesh output
    if mesh.save_mesh
        if !isdir("mesh/")
            mkdir("mesh/")
        end 
        fn = "mesh/mesh_" * string(mesh.nbins_vf) * "_" * mesh.dtype * ".fits"
        f = FITS(fn, "w")
        write(f, delta)
        close(f)
        println(fn * " saved to file.")
    end
        
    return delta, rec.boxsize[1], rec.boxcenter

end


end
