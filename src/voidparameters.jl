module VoidParameters
export Cosmology, GalaxyCatalogue, MeshParams, SphericalVoidParams

using Parameters

@with_kw mutable struct Cosmology
    # for redshift to distance calculation
    h::Float64 = 0.676 # dimensionless Hubble parameter
    omega_m::Float64 = 0.31 # omega matter
    omega_l::Float64 = 0.69 # omega lambda
    # for reconstruction
    f::Float64 = 0.8  # growth rate 
    bias::Float64 = 2.  # galaxy bias 
end

@with_kw mutable struct GalaxyCatalogue
    gal_pos::Array{AbstractFloat,2}
    gal_wts::Array{AbstractFloat,1} = ones(size(gal_pos,1))
    rand_pos::Array{AbstractFloat,2} = Array{AbstractFloat}(undef,0,0)
    rand_wts::Array{AbstractFloat,1} = ones(size(rand_pos,1))
end

@with_kw mutable struct MeshParams
    dtype::String = "f4"
    nbins_vf::Array{Int,1} = [0]  # number of bins for voidfinding (calculated using galaxy density if 0)
    is_box::Bool = true  # set to false for survey-like data
    box_length::Array{Float64,1} = [0]  # box length [Mpc/h] (disregarded if is_box=false)
    box_centre::Array{Float64,1} = box_length./2 # box centre (disregarded if is_box=false)
    padding::Float64 = 1.2  # box padding factor (disregarded if is_box=true)
    save_mesh::Bool = true
    mesh_fn::String = ""  # defaults to mesh_<nbins_vf>_<dtype>.fits

    # reconstruction parameters
    recon_alg::String = "IFFTparticle"
    r_smooth::Float64 = 10. # smoothing scale [Mpc/h]
    nbins_recon::Array{Int,1} = [0]  # number of bins for reconstruction (calculated using r_smooth if 0)
    los::String = "z"  # line-of-sight axis of box (disregarded if is_box=false)
end

@with_kw mutable struct SphericalVoidParams
    radii::Array{Float64,1} = [0]  # void radii [Mpc/h] (calculated using galaxy density if 0)
    min_dens_cut::Float64 = 1.0
    max_overlap_frac::Float64 = 0.
end


end
