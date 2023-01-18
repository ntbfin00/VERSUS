module VoidParameters
export VoidParams

using Parameters

@with_kw struct VoidParams
    verbose::Bool = false
    debug::Bool = false
    nthreads::Int = 4
    handle::String = "default"
    output_folder::String = ""
    omega_m::Float64 = 0.31
    do_recon::Bool = true
    nbins::Int = 512
    padding::Float64 = 200.
    smooth::Float64 = 10.
    bias::Float64 = 2.
    f::Float64 = 0.78
    niter::Int = 3
    tracer_file::String = ""
    tracer_file_type::Int = 1
    tracer_posn_cols::Array{Int,1} = [0,1,2]
    is_box::Bool = false
    box_length::Float64 = 1500.
    z_low_cut::Float64 = 0.4
    z_high_cut::Float64 = 0.73
    weights_model::Int = 1
    fkp::Bool = false
    cp::Bool = false
    noz::Bool = false
    systot::Bool = false
    veto::Bool = false
    comp::Bool = false
    random_file::String = ""
    random_file_type::Int = 1
    random_posn_cols::Array{Int,1} = [0,1,2]
    run_voxelvoids::Bool = true
    run_zobovvoids::Bool = true
    z_min::Float64 = 0.43
    z_max::Float64 = 0.70
    void_prefix::String = "Voids"
    min_dens_cut::Float64 = 1.0
    use_barycentres::Bool = true
    find_clusters::Bool = false
    cluster_prefix::String = "Clusters"
    max_dens_cut::Float64 = 1.0
    do_tesselation::Bool = true
    guard_nums::Int = 30
    use_mpi::Bool = false
    zobov_box_div::Int = 2
    zobov_buffer::Float64 = 0.08
    mask_file::String = ""
    use_z_wts::Bool = true
    use_sys_wts::Bool = true
    use_completeness_wts::Bool = true
    mock_file::String = ""
    mock_dens_ratio::Float64 = 10.
    void_min_num::Int = 5
    cluster_min_num::Int = 5
    max_overlap_frac::Float64 = 0.
    use_parallel::Bool = false

end

end
