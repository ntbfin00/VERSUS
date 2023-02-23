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
    save_mesh::Bool = true
    dtype::String = "f4"
    nbins::Int = 512
    is_box::Bool = true  # set to false for survey-like data
    box_length::Float64 = 1.  # box length [Mpc/h] (disregarded if is_box=false)
    box_centre::Array{Float64,1} = fill(box_length/2, 3) # box centre (disregarded if is_box=false)
    padding::Float64 = 1.2  # box padding (disregarded if is_box=true)

    # reconstruction parameters
    recon_alg::String = "IFFTparticle"
    los::String = "z"  # line-of-sight axis of box (disregarded if is_box=false)
    r_smooth::Float64 = box_length/nbins
end

@with_kw mutable struct SphericalVoidParams
    verbose::Bool = false
    output_fn::String = "void_cat_spherical"
    radii::Array{Float64,1}  # void radii [Mpc/h]
    min_dens_cut::Float64 = 1.0
    max_overlap_frac::Float64 = 0.
end


end
