module MeshBuilder

export gal_dens_bin, reconstruction, create_mesh

include("voidparameters.jl")
include("utils.jl")

using .VoidParameters
using .Utils
using PyCall
using FITSIO

"""
Returns the optimal number of bins for FFTWs above the input value.
"""
function optimal_binning(nbins::Array{Int,1})
    @debug "Determining the optimal number of bins"

    n_opt = zeros(3)
    for (i,n) in enumerate(nbins)
        # largest p where 2^p <= n
        max_p = ndigits(n, base=2)
        best_n = 2^max_p
        best_diff = best_n - n
        n_try = [2^max_p, 2^(max_p - 1), 3*2^(max_p - 2), 5*(max_p - 3), 7*(max_p - 3)] 
        diff = n_try .- n
        # find n which gives minimum positive difference
        n_opt[i] = minimum(diff[diff .>=0]) + n
    end

    @debug nbins_opt = Int.(n_opt)
    Int.(n_opt)

end

"""
Return both the optimal number of bins based on the galaxy density and the galaxy separation.
"""
function gal_dens_bin(cat::Main.VoidParameters.GalaxyCatalogue, mesh::Main.VoidParameters.MeshParams)
        @info "Estimating galaxy density"

        # number of bins per average galaxy separation
        f = 2

        cosmo_vf = Main.VoidParameters.Cosmology(; bias=1.)
        rec = set_recon_engine(cosmo_vf, cat, mesh, [0], 1.)[1]
        vol = prod(rec.boxsize)
        mean_dens = size(cat.gal_pos,1)/vol
        r_sep = (4 * pi * mean_dens / 3)^(-1/3)

        if !mesh.is_box
            n_itr = 4
            rand_pos = PyObject(cat.rand_pos)
            rand_wts = PyObject(cat.rand_wts)
            vol_est = 0
            # iterate for more accurate estimation
            @debug "Beginning iteration"
            for i = 1:n_itr
                # estimate nbins (underestimate)
                nbins = ceil.(Int, rec.boxsize/r_sep)
                if i == n_itr
                    @debug "Iteration complete" i, vol_est, nbins
                    # compute the final estimated survey volume
                    vol = vol_est
                    break
                end

                nbins_tot = prod(nbins)
                cell_vol = vol/nbins_tot

                @debug "Assigning randoms to estimate volume"
                # place randoms on mesh
                rec = set_recon_engine(cosmo_vf, cat, mesh, nbins, 1.)[1]
                rec.assign_randoms(rand_pos, rand_wts)

                @debug "Counting filled cells"
                # find where randoms are greater than 0.01 * average randoms per cell
                threshold = 0.01 * sum(rec.mesh_randoms.value)/nbins_tot
                filled_cells = count(q->(q > threshold), rec.mesh_randoms.value)
                vol_est = filled_cells * cell_vol

                @debug "Estimating more accurate mean density" 
                # estimate mean galaxy density
                mean_dens = size(cat.gal_pos,1)/(filled_cells*cell_vol)
                # estimate galaxy separation (overestimate)
                r_sep = (4 * pi * mean_dens / 3)^(-1/3)

                @debug "Iterating" i, vol_est, nbins

            end
        end

        nbins = ceil.(Int, f * mesh.padding*rec.boxsize/r_sep)

        return nbins, r_sep, vol

end

"""
Initialise mesh and set algorithm for reconstruction.
"""
function set_recon_engine(cosmo::Main.VoidParameters.Cosmology, cat::Main.VoidParameters.GalaxyCatalogue, mesh::Main.VoidParameters.MeshParams, nbins::Array{Int,1}, pad::Float64)
    @debug "Initialising mesh"

    if !mesh.is_box && size(cat.rand_pos,1) == 0
        throw(ErrorException("is_box is set to false but no randoms have been supplied."))
    end

    @debug "Setting mesh based on positions" nbins pad
    if mesh.is_box
        los = mesh.los
        pos = PyObject(cat.gal_pos)
    else 
        los = nothing
        pos = PyObject(cat.rand_pos)
    end

    # set the mesh
    if mesh.recon_alg == "IFFTparticle"
        @debug "Setting IFFTparticle mesh"
        recon = pyimport("pyrecon.iterative_fft_particle")
        rec = recon.IterativeFFTParticleReconstruction(f=cosmo.f, bias=cosmo.bias, los=los, nmesh=nbins, positions=pos, boxpad=pad, wrap=true, dtype=mesh.dtype, nthreads=Threads.nthreads())
        data = "data"
    elseif mesh.recon_alg == "IFFT"
        @debug "Setting IFFT mesh"
        recon = pyimport("pyrecon.iterative_fft")
        rec = recon.IterativeFFTReconstruction(f=cosmo.f, bias=cosmo.bias, los=los, nmesh=nbins, positions=pos, boxpad=pad, wrap=true, dtype=mesh.dtype, nthreads=Threads.nthreads())
        data = PyObject(cat.gal_pos)
    elseif mesh.recon_alg == "MultiGrid"
        @debug "Setting MultiGrid mesh"
        recon = pyimport("pyrecon.multigrid")
        rec = recon.MultiGridReconstruction(f=cosmo.f, bias=cosmo.bias, los=los, nmesh=nbins, positions=pos, boxpad=pad, wrap=true, dtype=mesh.dtype, nthreads=Threads.nthreads())
        data = PyObject(cat.gal_pos)
    else
        throw(ErrorException("Reconstruction algorithm not recognised. Allowed algorithms are IFFTparticle, IFFT and MultiGrid."))
    end


    return rec, data

end


function compute_density_field(cat::Main.VoidParameters.GalaxyCatalogue, mesh::Main.VoidParameters.MeshParams, rec, r_smooth::Float64)

    @info "Assigning galaxies to grid"
    gal_pos = PyObject(cat.gal_pos)
    gal_wts = PyObject(cat.gal_wts)
    rec.assign_data(gal_pos, gal_wts)

    if !mesh.is_box
        @info "Assigning randoms to grid"
        rand_pos = PyObject(cat.rand_pos)
        rand_wts = PyObject(cat.rand_wts)
        rec.assign_randoms(rand_pos, rand_wts)
    end

    @info "Computing density field"
    rec.set_density_contrast(smoothing_radius=r_smooth)

end

"""
Run reconstruction on galaxy positions. Returns a GalaxyCatalogue object.
"""
function reconstruction(cosmo::Main.VoidParameters.Cosmology, cat::Main.VoidParameters.GalaxyCatalogue, mesh::Main.VoidParameters.MeshParams)
    
    @info "Performing density field reconstruction"

    if !mesh.is_box && mesh.padding <= 1.0
        @warn "Reconstructed positions may be wrapped back into survey region as is_box=false and padding<=1."
    end

    # calculate default number of bins based on box_length and smoothing radius
    if mesh.nbins_recon == [0]
        @info "Calculating default bins based on smoothing radius"
        rec = set_recon_engine(cosmo, cat, mesh, mesh.nbins_recon, mesh.padding)[1]
        # Determine the optimum number of bins for FFTs below the smoothing radius
        nsmooth = ceil.(Int, rec.boxsize/mesh.r_smooth)
        mesh.nbins_recon = optimal_binning(nsmooth)
    end

    # create the grid
    rec, data = set_recon_engine(cosmo, cat, mesh, mesh.nbins_recon, mesh.padding)

    if maximum(rec.boxsize./rec.nmesh) > mesh.r_smooth
        @warn "Smoothing scale is less than cellsize."
    end

    compute_density_field(cat, mesh, rec, mesh.r_smooth)

    # run reconstruction
    @info "Running reconstruction with $(mesh.recon_alg)"
    rec.run()
    rec_gal_pos = rec.read_shifted_positions(data, field="rsd")
    @info "Galaxy positions reconstructed"
    
    # output reconstructed catalogue
    Main.VoidParameters.GalaxyCatalogue(rec_gal_pos, cat.gal_wts, cat.rand_pos, cat.rand_wts)

end

"""
Create density mesh from galaxy and random positions. Returns 3D density mesh, box length and box centre.
"""
function create_mesh(cat::Main.VoidParameters.GalaxyCatalogue, mesh::Main.VoidParameters.MeshParams)

    @info "Creating density mesh"

    # set bias=1 so voids are found in the galaxy field (not matter field)
    # other cosmological parameters are have no effect on mesh construction 
    cosmo_vf = Main.VoidParameters.Cosmology(; bias=1.)

    # no padding for voidfinding if box
    if mesh.is_box
        rec = set_recon_engine(cosmo_vf, cat, mesh, mesh.nbins_vf, 1.)[1]
    else
        rec = set_recon_engine(cosmo_vf, cat, mesh, mesh.nbins_vf, mesh.padding)[1]
    end

    # determine density on mesh
    compute_density_field(cat, mesh, rec, 0.)

    delta = rec.mesh_delta.value

    # compute survey mask for empty cells
    if !mesh.is_box
        threshold = 0.01 * sum(rec.mesh_randoms.value)/prod(mesh.nbins_vf)
        mask = findall(q->(q <= threshold), rec.mesh_randoms.value)
    else
        mask = nothing
    end

    @debug "Check for NaN values in mesh " any_NaN = any(isnan, delta)

    @info "$(string(typeof(delta))[7:13]) density mesh set"

    # save mesh output
    if mesh.save_mesh
        if !isdir("mesh/")
            mkdir("mesh/")
        end 
        if mesh.mesh_fn == ""
            bins = string.(mesh.nbins_vf, "_")
            fn = "mesh/mesh_" * string(bins...) * mesh.dtype * ".fits"
        else
            fn = "mesh/" * mesh.mesh_fn * ".fits"
        end
        f = FITS(fn, "w")
        write(f, delta)
        close(f)
        @info "$fn saved to file"
    end
        
    return delta, mask, rec.boxsize, rec.boxcenter

end


end
