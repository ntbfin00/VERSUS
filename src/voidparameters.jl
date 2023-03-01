module VoidParameters
export Cosmology, MeshParams, SphericalVoidParams

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

@with_kw mutable struct MeshParams
    dtype::String = "f4"
    nbins_vf::Int = 500  # number of bins for voidfinding
    is_box::Bool = true  # set to false for survey-like data
    box_length::Float64 = 1500.  # box length [Mpc/h] (disregarded if is_box=false)
    box_centre::Array{Float64,1} = fill(box_length/2, 3) # box centre (disregarded if is_box=false)
    padding::Float64 = 1.2  # box padding factor (disregarded if is_box=true)
    save_mesh::Bool = true
    mesh_fn::String = "mesh_" * string(nbins_vf) * "_" * dtype

    # reconstruction parameters
    recon_alg::String = "IFFTparticle"
    r_smooth::Float64 = 12. # smoothing scale [Mpc/h]
    nbins_recon::Int = 0  # number of bins for reconstruction (calculated using mesh parameters if set to 0)
    los::String = "z"  # line-of-sight axis of box (disregarded if is_box=false)
end

@with_kw mutable struct SphericalVoidParams
    verbose::Bool = false
    radii::Array{Float64,1}  # void radii [Mpc/h]
    min_dens_cut::Float64 = 1.0
    max_overlap_frac::Float64 = 0.
end


end
