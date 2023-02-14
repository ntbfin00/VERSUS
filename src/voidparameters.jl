module VoidParameters
export InputParams, OutputParams, MeshParams, SphericalVoidParams

using Parameters

@with_kw mutable struct InputParams
    threading::Bool = false
    data_format::String = "xyz"  # xyz or rdz (if rdz, must be provided in Degrees)
    data_cols::Array{String,1} = ["x","y","z"]  # weights can be supplied as a 4th column
    box_length::Float64 = 1.
    box_centre::Array{Float64,1} = [box_length/2, box_length/2, box_length/2]
    verbose::Bool = false
    build_mesh::Bool = true
    run_spherical_vf::Bool = true
end

@with_kw mutable struct OutputParams
    output_type::String = "fits"
    output_folder::String = "output/"
end

@with_kw mutable struct MeshParams
    is_box::Bool = true
    dtype::String = "f8"
    nbins::Int = 512
    do_recon::Bool = false
    recon_alg::String = "IFFTparticle"
    padding::Float64 = 1.5
    r_smooth::Float64 = 0.
    omega_m::Float64 = 0.31
    f::Float64 = 0.8
    bias::Float64 = 2.
    save_mesh::Bool = true
end

@with_kw mutable struct SphericalVoidParams
    output_fn::String = "void_cat_spherical"
    radii::Array{Float64,1}
    min_dens_cut::Float64 = 1.0
    max_overlap_frac::Float64 = 0.
end

end
