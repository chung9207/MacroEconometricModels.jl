"""
Core Local Projection estimation and shared utility functions.

This module provides:
- Shared utility functions for all LP variants
- Core LP estimation (Jordà 2005)
- IRF extraction and cumulative IRF

Note: HAC covariance estimators are defined in covariance_estimators.jl

References:
- Jordà, Ò. (2005). "Estimation and Inference of Impulse Responses by Local Projections."
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Shared Utility Functions
# =============================================================================

"""
    create_cov_estimator(cov_type::Symbol, ::Type{T}; bandwidth::Int=0) where T

Create covariance estimator from symbol specification.
Eliminates repeated if/else patterns across LP variants.
"""
function create_cov_estimator(cov_type::Symbol, ::Type{T}; bandwidth::Int=0) where {T<:AbstractFloat}
    if cov_type == :newey_west
        NeweyWestEstimator{T}(bandwidth, :bartlett, false)
    elseif cov_type == :white
        WhiteEstimator()
    elseif cov_type == :driscoll_kraay
        DriscollKraayEstimator{T}(bandwidth, :bartlett)
    else
        throw(ArgumentError("cov_type must be :newey_west, :white, or :driscoll_kraay"))
    end
end

"""
    compute_horizon_bounds(T_obs::Int, h::Int, lags::Int) -> (t_start, t_end)

Compute valid observation bounds for horizon h.
"""
function compute_horizon_bounds(T_obs::Int, h::Int, lags::Int)
    t_start = lags + 1
    t_end = T_obs - h
    if t_end < t_start
        throw(ArgumentError("Not enough observations for horizon $h with $lags lags"))
    end
    (t_start, t_end)
end

"""
    build_response_matrix(Y::AbstractMatrix{T}, h::Int, t_start::Int, t_end::Int,
                          response_vars::Vector{Int}) where T

Build response matrix Y_h at horizon h.
"""
function build_response_matrix(Y::AbstractMatrix{T}, h::Int, t_start::Int, t_end::Int,
                                response_vars::Vector{Int}) where {T<:AbstractFloat}
    T_eff = t_end - t_start + 1
    n_response = length(response_vars)
    Y_h = Matrix{T}(undef, T_eff, n_response)
    @inbounds for (j, var) in enumerate(response_vars)
        for (i, t) in enumerate(t_start:t_end)
            Y_h[i, j] = Y[t + h, var]
        end
    end
    Y_h
end

"""
    build_control_columns!(X_h::AbstractMatrix{T}, Y::AbstractMatrix{T},
                           t_start::Int, t_end::Int, lags::Int, start_col::Int) where T

Fill control (lagged Y) columns into regressor matrix X_h.
Returns the next available column index.
"""
function build_control_columns!(X_h::AbstractMatrix{T}, Y::AbstractMatrix{T},
                                 t_start::Int, t_end::Int, lags::Int, start_col::Int) where {T<:AbstractFloat}
    n = size(Y, 2)
    col = start_col
    @inbounds for (i, t) in enumerate(t_start:t_end)
        col_local = start_col
        for lag in 1:lags
            for var in 1:n
                X_h[i, col_local] = Y[t - lag, var]
                col_local += 1
            end
        end
    end
    start_col + n * lags
end

"""
    compute_block_robust_vcov(X::AbstractMatrix{T}, U::AbstractMatrix{T},
                              cov_estimator::AbstractCovarianceEstimator) where T

Compute block-diagonal robust covariance for multi-equation system.
"""
function compute_block_robust_vcov(X::AbstractMatrix{T}, U::AbstractMatrix{T},
                                    cov_estimator::AbstractCovarianceEstimator) where {T<:AbstractFloat}
    n_eq = size(U, 2)
    k = size(X, 2)
    V = zeros(T, k * n_eq, k * n_eq)
    @inbounds for eq in 1:n_eq
        V_eq = robust_vcov(X, @view(U[:, eq]), cov_estimator)
        idx = ((eq-1)*k + 1):(eq*k)
        V[idx, idx] .= V_eq
    end
    V
end

"""
    extract_shock_irf(B::Vector{Matrix{T}}, vcov::Vector{Matrix{T}},
                      response_vars::Vector{Int}, shock_coef_idx::Int;
                      conf_level::Real=0.95) where T

Generic IRF extraction from coefficient and covariance vectors.
Works for LPModel, LPIVModel, PropensityLPModel.
"""
function extract_shock_irf(B::Vector{Matrix{T}}, vcov::Vector{Matrix{T}},
                           response_vars::Vector{Int}, shock_coef_idx::Int;
                           conf_level::Real=0.95) where {T<:AbstractFloat}
    H = length(B) - 1
    n_response = length(response_vars)
    k = size(B[1], 1)

    values = Matrix{T}(undef, H + 1, n_response)
    se = Matrix{T}(undef, H + 1, n_response)

    @inbounds for h in 0:H
        B_h = B[h + 1]
        V_h = vcov[h + 1]
        for (j, _) in enumerate(response_vars)
            values[h + 1, j] = B_h[shock_coef_idx, j]
            var_idx = (j - 1) * k + shock_coef_idx
            se[h + 1, j] = sqrt(V_h[var_idx, var_idx])
        end
    end

    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = values .- z .* se
    ci_upper = values .+ z .* se

    (values=values, se=se, ci_lower=ci_lower, ci_upper=ci_upper)
end

# Note: Kernel functions, bandwidth selection, Newey-West, White, Driscoll-Kraay,
# long_run_variance, and long_run_covariance are now in covariance_estimators.jl

# =============================================================================
# LP Matrix Construction
# =============================================================================

"""
    construct_lp_matrices(Y::AbstractMatrix{T}, shock_var::Int, h::Int, lags::Int;
                          response_vars::Vector{Int}=collect(1:size(Y,2))) where T

Construct regressor and response matrices for LP regression at horizon h.

Returns: (Y_h, X_h, valid_idx)
"""
function construct_lp_matrices(Y::AbstractMatrix{T}, shock_var::Int, h::Int, lags::Int;
                                response_vars::Vector{Int}=collect(1:size(Y, 2))) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    t_start, t_end = compute_horizon_bounds(T_obs, h, lags)
    T_eff = t_end - t_start + 1

    # Response matrix
    Y_h = build_response_matrix(Y, h, t_start, t_end, response_vars)

    # Regressor matrix: [1, shock_t, y_{t-1}, ..., y_{t-lags}]
    k = 2 + n * lags
    X_h = Matrix{T}(undef, T_eff, k)

    @inbounds for (i, t) in enumerate(t_start:t_end)
        X_h[i, 1] = one(T)
        X_h[i, 2] = Y[t, shock_var]
    end
    build_control_columns!(X_h, Y, t_start, t_end, lags, 3)

    valid_idx = collect(t_start:t_end)
    (Y_h, X_h, valid_idx)
end

# =============================================================================
# Core LP Estimation
# =============================================================================

"""
    estimate_lp(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y,2)),
                cov_type::Symbol=:newey_west, bandwidth::Int=0,
                conf_level::Real=0.95) -> LPModel{T}

Estimate Local Projection impulse response functions (Jordà 2005).

The LP regression for horizon h:
    y_{t+h} = α_h + β_h * shock_t + Γ_h * controls_t + ε_{t+h}
"""
function estimate_lp(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                     lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y, 2)),
                     cov_type::Symbol=:newey_west, bandwidth::Int=0,
                     conf_level::Real=0.95) where {T<:AbstractFloat}
    T_obs, n = size(Y)

    validate_positive(horizon, "horizon")
    @assert 1 <= shock_var <= n "shock_var must be in 1:$n"
    @assert all(1 .<= response_vars .<= n) "response_vars must be in 1:$n"
    @assert lags >= 0 "lags must be non-negative"
    @assert T_obs > lags + horizon + 1 "Not enough observations"

    cov_estimator = create_cov_estimator(cov_type, T; bandwidth=bandwidth)

    B = Vector{Matrix{T}}(undef, horizon + 1)
    residuals = Vector{Matrix{T}}(undef, horizon + 1)
    vcov = Vector{Matrix{T}}(undef, horizon + 1)
    T_eff = Vector{Int}(undef, horizon + 1)

    for h in 0:horizon
        Y_h, X_h, valid_idx = construct_lp_matrices(Y, shock_var, h, lags;
                                                     response_vars=response_vars)
        T_eff[h + 1] = length(valid_idx)

        # OLS: B_h = (X'X)^{-1} X'Y
        XtX_inv = robust_inv(X_h' * X_h)
        B_h = XtX_inv * (X_h' * Y_h)
        U_h = Y_h - X_h * B_h

        B[h + 1] = B_h
        residuals[h + 1] = U_h
        vcov[h + 1] = compute_block_robust_vcov(X_h, U_h, cov_estimator)
    end

    LPModel(Matrix{T}(Y), shock_var, response_vars, horizon, lags,
            B, residuals, vcov, T_eff, cov_estimator)
end

# Float fallback
estimate_lp(Y::AbstractMatrix, shock_var::Int, horizon::Int; kwargs...) =
    estimate_lp(Float64.(Y), shock_var, horizon; kwargs...)

# =============================================================================
# Multiple Shocks
# =============================================================================

"""
    estimate_lp_multi(Y::AbstractMatrix{T}, shock_vars::Vector{Int}, horizon::Int;
                      kwargs...) -> Vector{LPModel{T}}

Estimate LP for multiple shock variables.
"""
function estimate_lp_multi(Y::AbstractMatrix{T}, shock_vars::Vector{Int}, horizon::Int;
                           kwargs...) where {T<:AbstractFloat}
    [estimate_lp(Y, shock, horizon; kwargs...) for shock in shock_vars]
end

# =============================================================================
# LP with Orthogonalized Shocks
# =============================================================================

"""
    estimate_lp_cholesky(Y::AbstractMatrix{T}, horizon::Int;
                         lags::Int=4, cov_type::Symbol=:newey_west, kwargs...) -> Vector{LPModel{T}}

Estimate LP with Cholesky-orthogonalized shocks.
"""
function estimate_lp_cholesky(Y::AbstractMatrix{T}, horizon::Int;
                              lags::Int=4, cov_type::Symbol=:newey_west,
                              kwargs...) where {T<:AbstractFloat}
    T_obs, n = size(Y)

    var_model = estimate_var(Y, lags)
    U = var_model.U
    L = identify_cholesky(var_model)
    eps = (inv(L) * U')'

    Y_eff = Y[(lags+1):end, :]
    @assert size(eps, 1) == size(Y_eff, 1) "Dimension mismatch"

    models = Vector{LPModel{T}}(undef, n)
    for shock in 1:n
        Y_aug = hcat(eps[:, shock], Y_eff)
        models[shock] = estimate_lp(Y_aug, 1, horizon; lags=lags,
                                     response_vars=collect(2:(n+1)),
                                     cov_type=cov_type, kwargs...)
    end
    models
end

# =============================================================================
# Model Comparison
# =============================================================================

"""
    compare_var_lp(Y::AbstractMatrix{T}, horizon::Int; lags::Int=4) where T

Compare VAR-based and LP-based impulse responses.
"""
function compare_var_lp(Y::AbstractMatrix{T}, horizon::Int; lags::Int=4) where {T<:AbstractFloat}
    n = size(Y, 2)

    var_model = estimate_var(Y, lags)
    var_result = irf(var_model, horizon; method=:cholesky)

    lp_models = estimate_lp_cholesky(Y, horizon; lags=lags)
    lp_results = [lp_irf(m) for m in lp_models]

    var_values = var_result.values
    lp_values = zeros(T, horizon, n, n)

    for shock in 1:n
        for (h_idx, h) in enumerate(1:horizon)
            for resp in 1:n
                lp_values[h_idx, resp, shock] = lp_results[shock].values[h + 1, resp]
            end
        end
    end

    (var_irf=var_values, lp_irf=lp_values, difference=var_values - lp_values)
end
