module VoidParameters
export Cosmology, GalaxyCatalogue, MeshParams, SphericalVoidParams

using Parameters

@with_kw mutable struct Cosmology
    # for redshift to distance calculation
    h::Float32 = 0.676 # dimensionless Hubble parameter
    omega_m::Float32 = 0.31 # omega matter
    omega_l::Float32 = 0.69 # omega lambda
    # for reconstruction
    f::Float32 = 0.8  # growth rate 
    bias::Float32 = 2.  # galaxy bias 
end

@with_kw mutable struct GalaxyCatalogue
    gal_pos::Array{Float32,2}
    gal_wts::Array{Float32,1} = ones(size(gal_pos,1))
    rand_pos::Array{Float32,2} = Array{Float32}(undef,0,0)
    rand_wts::Array{Float32,1} = ones(size(rand_pos,1))
end

@with_kw mutable struct MeshParams
    dtype::String = "f4"
    nbins_vf::Array{Int,1} = [0]  # number of bins for voidfinding (calculated using galaxy density if 0) 
                                  # [MUST GIVE THE SAME CELL RESOLUTION (LENGTH/NBINS) FOR EACH DIMENSION]
    padding::Float32 = 1.1  # survey (and reconstruction) padding factor
    save_mesh::Bool = true
    mesh_fn::String = ""  # defaults to mesh_<nbins_vf>_<dtype>.fits

    # reconstruction parameters
    recon_alg::String = "IFFTparticle"
    r_smooth::Float32 = 10. # smoothing scale [Mpc/h]
    nbins_recon::Array{Int,1} = [0]  # number of bins for reconstruction (calculated using r_smooth if 0)
    los::String = "z"  # line-of-sight axis of box (disregarded if is_box=false)
end

@with_kw mutable struct SphericalVoidParams
    radii::Array{Float32,1} = [0]  # void radii [Mpc/h] (calculated using galaxy density if 0)
    min_dens_cut::Float32 = 0.2
    max_overlap_frac::Float32 = 0.
end


end
