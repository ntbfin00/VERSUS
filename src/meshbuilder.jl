module MeshBuilder

export GalaxyCatalogue, mean_gal_sep, to_cartesian, reconstruction, create_mesh

include("utils.jl")

using .Utils
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

function mean_gal_sep(cat::Main.MeshBuilder.GalaxyCatalogue, mesh::Main.VoidParameters.MeshParams)
        # estimate volume of survey
        if mesh.is_box
            vol = mesh.box_length^3 
        else
            l = maximum(cat.rand_pos,dims=1) - minimum(cat.rand_pos,dims=1)
            vol = prod(l)
        end

        # estimate mean galaxy density (underestimate in case of survey)
        mean_dens = size(cat.gal_pos,1)/vol

        (4 * pi * mean_dens / 3)^(-1/3)
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

    # compute distances from redshifts
    @debug "Setting cosmology"
    c = cosmology.LambdaCDM(H0=cosmo.h*100, Om0=cosmo.omega_m, Ode0=cosmo.omega_l)
    pos = np.array(pos)
    @debug "Calculating comoving distances"
    dist = c.comoving_distance(pos[:,3])
    
    @debug "Converting to cartesian"
    utils.sky_to_cartesian(dist,pos[:,1],pos[:,2], degree=degree)
end

"""
Returns the optimal number of bins for FFTs closest to the input nbins.

Round = "above" finds optimal nbins > input nbins.
Round = "below" finds optimal nbins < input nbins.
"""
function optimal_binning(nbins, round::String)
    @debug "Determining the optimal number of bins"

    p = floor(log(2,nbins))
    q = floor(log(3,(nbins - 2^p)))
    if round == "above"
        r = ceil(log(5,(nbins - 2^p - 3^q)))
        return Int(2^p + 3^q + 5^r)
    else
        return Int(2^p + 3^q)
    end

end


"""
Set algorithm for reconstruction.
"""
function set_recon_engine(cosmo::Main.VoidParameters.Cosmology, cat::Main.MeshBuilder.GalaxyCatalogue, mesh::Main.VoidParameters.MeshParams, nbins::Int)
    @debug "Initialising mesh"

    if !mesh.is_box && size(cat.rand_pos,1) == 0
        throw(ErrorException("is_box is set to false but no randoms have been supplied."))
    end

    # determine box size from input
    if mesh.is_box
        @debug "Setting mesh parameters to box inputs"
        los = mesh.los
        boxsize = mesh.box_length
        boxcenter = mesh.box_centre
        pos = nothing
    # determine box size from random positions and padding
    else 
        @debug "Setting mesh parameters to be calculated from positions"
        los = nothing
        boxsize = nothing
        boxcenter = nothing
        pos = np.array(cat.rand_pos)
    end

    # set the mesh
    if mesh.recon_alg == "IFFTparticle"
        @debug "Setting IFFTparticle mesh"
        recon = pyimport("pyrecon.iterative_fft_particle")
        rec = recon.IterativeFFTParticleReconstruction(f=cosmo.f, bias=cosmo.bias, los=los, nmesh=nbins, boxsize=boxsize, boxcenter=boxcenter, boxpad=mesh.padding, positions=pos, wrap=true, dtype=mesh.dtype, nthreads=Threads.nthreads())
        data = "data"
    elseif mesh.recon_alg == "IFFT"
        @debug "Setting IFFT mesh"
        recon = pyimport("pyrecon.iterative_fft")
        rec = recon.IterativeFFTReconstruction(f=cosmo.f, bias=cosmo.bias, los=los, nmesh=nbins, boxsize=boxsize, boxcenter=boxcenter, boxpad=mesh.padding, positions=pos, wrap=true, dtype=mesh.dtype, nthreads=Threads.nthreads())
        data = np.array(cat.gal_pos)
    elseif mesh.recon_alg == "MultiGrid"
        @debug "Setting MultiGrid mesh"
        recon = pyimport("pyrecon.multigrid")
        rec = recon.MultiGridReconstruction(f=cosmo.f, bias=cosmo.bias, los=los, nmesh=nbins, boxsize=boxsize, boxcenter=boxcenter, boxpad=mesh.padding, positions=pos, wrap=true, dtype=mesh.dtype, nthreads=Threads.nthreads())
        data = np.array(cat.gal_pos)
    else
        throw(ErrorException("Reconstruction algorithm not recognised. Allowed algorithms are IFFTparticle, IFFT and MultiGrid."))
    end


    # display only when nbins is set
    if nbins != 0
        @info "Mesh settings: box_length=$(rec.boxsize[1]), nbins=$(rec.nmesh[1])^3"
        # println("Mesh settings: box_length=", rec.boxsize[1], ", nbins=", rec.nmesh[1], "^3")
    end

    return rec, data

end


"""
Run reconstruction on galaxy positions. Returns a GalaxyCatalogue object.
"""
function reconstruction(cosmo::Main.VoidParameters.Cosmology, cat::Main.MeshBuilder.GalaxyCatalogue, mesh::Main.VoidParameters.MeshParams)
    
    @info "Performing density field reconstruction"
    # println("\n ==== Density field reconstruction ==== ")

    if !mesh.is_box && mesh.padding <= 1.0
        @warn "Reconstructed positions may be wrapped back into survey region as is_box=false and padding<=1."
    end

    # create the grid
    rec, data = set_recon_engine(cosmo, cat, mesh, mesh.nbins_recon)

    # calculate default number of bins based on box_length and smoothing radius
    if mesh.nbins_recon == 0
        @info "Calculating default bins based on smoothing radius"
        # Determine the optimum number of bins for FFTs below the smoothing radius
        nsmooth = rec.boxsize[1]/mesh.r_smooth
        nbins = optimal_binning(nsmooth, "below")
        # recalculate the mesh with new nbins
        rec, data = set_recon_engine(cosmo, cat, mesh, nbins)
    end

    if rec.boxsize[1]/rec.nmesh[1] > mesh.r_smooth
        @warn "Smoothing scale is less than cellsize."
    end

    # assign data and compute the density field
    @info "Assigning galaxies to grid"
    # println("Assigning galaxies to grid...")
    gal_pos = np.array(cat.gal_pos)
    if size(cat.gal_wts,1) == 0
        rec.assign_data(gal_pos)
    else
        gal_wts = np.array(cat.gal_wts)
        rec.assign_data(gal_pos, gal_wts)
    end

    @info "Computing density field"
    rec.set_density_contrast(smoothing_radius=mesh.r_smooth)

    # run reconstruction
    @info "Running reconstruction with $mesh.recon_alg"
    # println("Running reconstruction with ", mesh.recon_alg, "...")
    rec.run()
    rec_gal_pos = rec.read_shifted_positions(data, field="rsd")
    @info "Galaxy positions reconstructed"
    # println("Galaxy positions reconstructed.")
    
    # output reconstructed catalogue
    GalaxyCatalogue(rec_gal_pos, cat.gal_wts, cat.rand_pos, cat.rand_wts)

end

"""
Create density mesh from galaxy and random positions. Returns 3D density mesh, box length and box centre.
"""
function create_mesh(cat::Main.MeshBuilder.GalaxyCatalogue, mesh::Main.VoidParameters.MeshParams)

    @info "Creating density mesh"
    # println("\n ==== Creating density mesh ==== ")

    # set bias=1 so voids are found in the galaxy field
    # other cosmological parameters are have no effect on mesh construction 
    cosmo_vf = Main.VoidParameters.Cosmology(; bias=1.)

    # create the grid
    rec = set_recon_engine(cosmo_vf, cat, mesh, mesh.nbins_vf)[1]

    # calculate default number of voidfinding bins based on galaxy density
    if mesh.nbins_vf == 0
        @info "Calculating default bins based on galaxy density"
        r_sep = mean_gal_sep(cat, mesh)
        nbins = rec.boxsize[1]/(0.5 * r_sep)
        nbins = optimal_binning(nbins, "above")
        # recalculate the mesh with new nbins
        rec = set_recon_engine(cosmo_vf, cat, mesh, nbins)[1]
    end

    # assign data and compute the density field
    @info "Assigning galaxies to grid"
    # println("Assigning galaxies to grid...")
    gal_pos = np.array(cat.gal_pos)
    if size(cat.gal_wts,1) == 0
        rec.assign_data(gal_pos)
    else
        gal_wts = np.array(cat.gal_wts)
        rec.assign_data(gal_pos, gal_wts)
    end

    if mesh.is_box
        @info "Computing density field"
        rec.set_density_contrast(smoothing_radius=0.0)

        delta = rec.mesh_delta.value
        @info "$(string(typeof(delta))[7:13]) density mesh set"
        # println(string(typeof(delta))[7:13]," density mesh set.")

    else
        @info "Assigning randoms to grid"
        # println("Assigning randoms to grid...")
        rand_pos = np.array(cat.rand_pos)
        if size(cat.rand_wts) == 0
            rec.assign_randoms(rand_pos)
        else
            rand_wts = np.array(cat.rand_wts)
            rec.assign_randoms(rand_pos, rand_wts)
        end
        @info "Computing density field"
        rec.set_density_contrast(smoothing_radius=0.0)

        delta = rec.mesh_delta.value
        @info "$(string(typeof(delta))[7:13]) density mesh set"
        # println(string(typeof(delta))[7:13]," density mesh set.")
    end

    # save mesh output
    if mesh.save_mesh
        if !isdir("mesh/")
            mkdir("mesh/")
        end 
        fn = "mesh/" * mesh.mesh_fn * ".fits"
        f = FITS(fn, "w")
        write(f, delta)
        close(f)
        @info "$fn saved to file"
        # println(fn * " saved to file.")
    end
        
    return delta, rec.boxsize[1], rec.boxcenter

end


end
