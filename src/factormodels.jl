"""
Factor Models: Static (PCA), Dynamic (state-space), and Generalized Dynamic (spectral).

References:
- Bai & Ng (2002): Static factor model, information criteria
- Stock & Watson (2002): Dynamic factor models
- Forni, Hallin, Lippi & Reichlin (2000, 2005): GDFM
"""

using LinearAlgebra, Statistics, FFTW
using Distributions: Normal, quantile

# =============================================================================
# Shared Utilities
# =============================================================================

"""Standardize matrix: subtract mean, divide by std."""
function _standardize(X::AbstractMatrix{T}) where {T}
    μ, σ = mean(X, dims=1), max.(std(X, dims=1), T(1e-10))
    (X .- μ) ./ σ
end

# #############################################################################
# PART 1: STATIC FACTOR MODEL (PCA)
# #############################################################################

"""
    estimate_factors(X, r; standardize=true) -> FactorModel

Estimate static factor model Xₜ = Λ Fₜ + eₜ via PCA.
"""
function estimate_factors(X::AbstractMatrix{T}, r::Int; standardize::Bool=true) where {T<:AbstractFloat}
    T_obs, N = size(X)
    validate_factor_inputs(T_obs, N, r)

    X_orig = copy(X)
    X_proc = standardize ? _standardize(X) : X

    Σ = (X_proc'X_proc) / T_obs
    eig = eigen(Symmetric(Σ))
    idx = sortperm(eig.values, rev=true)
    λ, V = eig.values[idx], eig.vectors[:, idx]

    loadings = V[:, 1:r] * Diagonal(sqrt.(λ[1:r]))
    factors = X_proc * V[:, 1:r]

    total = sum(λ)
    expl = λ / total
    cumul = cumsum(expl)

    FactorModel{T}(X_orig, factors, loadings, λ, expl, cumul, r, standardize)
end

@float_fallback estimate_factors X

# --- StatsAPI for FactorModel ---

StatsAPI.predict(m::FactorModel) = m.factors * m.loadings'

function StatsAPI.residuals(m::FactorModel{T}) where {T}
    fitted = predict(m)
    m.standardized ? _standardize(m.X) - fitted : m.X - fitted
end

function StatsAPI.r2(m::FactorModel{T}) where {T}
    resid = residuals(m)
    X_ref = m.standardized ? _standardize(m.X) : m.X
    [max(zero(T), 1 - var(@view(resid[:, i])) / max(var(@view(X_ref[:, i])), T(1e-10)))
     for i in 1:size(m.X, 2)]
end

StatsAPI.nobs(m::FactorModel) = size(m.X, 1)
StatsAPI.dof(m::FactorModel) = size(m.X, 2) * m.r + size(m.X, 1) * m.r - m.r^2

# --- Bai-Ng Information Criteria ---

"""
    ic_criteria(X, max_factors; standardize=true)

Compute Bai-Ng (2002) IC1, IC2, IC3 and optimal factor counts.
"""
function ic_criteria(X::AbstractMatrix{T}, max_factors::Int; standardize::Bool=true) where {T<:AbstractFloat}
    T_obs, N = size(X)
    1 <= max_factors <= min(T_obs, N) || throw(ArgumentError("max_factors must be in [1, min(T,N)]"))

    IC1, IC2, IC3 = Vector{T}(undef, max_factors), Vector{T}(undef, max_factors), Vector{T}(undef, max_factors)
    NT, minNT = N * T_obs, min(N, T_obs)

    for r in 1:max_factors
        resid = residuals(estimate_factors(X, r; standardize))
        V_r = sum(resid .^ 2) / NT
        logV = log(V_r)
        pen_base = r * (N + T_obs) / NT

        IC1[r] = logV + pen_base * log(NT / (N + T_obs))
        IC2[r] = logV + pen_base * log(minNT)
        IC3[r] = logV + r * log(minNT) / minNT
    end

    (IC1=IC1, IC2=IC2, IC3=IC3, r_IC1=argmin(IC1), r_IC2=argmin(IC2), r_IC3=argmin(IC3))
end

"""Data for scree plot: (factors, explained_variance, cumulative_variance)."""
scree_plot_data(m::FactorModel) = (factors=1:length(m.eigenvalues), explained_variance=m.explained_variance,
                                    cumulative_variance=m.cumulative_variance)

# #############################################################################
# PART 2: DYNAMIC FACTOR MODEL
# #############################################################################

"""
    estimate_dynamic_factors(X, r, p; method=:twostep, standardize=true, max_iter=100, tol=1e-6)

Estimate dynamic factor model. Methods: :twostep or :em.
"""
function estimate_dynamic_factors(X::AbstractMatrix{T}, r::Int, p::Int;
    method::Symbol=:twostep, standardize::Bool=true, max_iter::Int=100,
    tol::Float64=1e-6, diagonal_idio::Bool=true
) where {T<:AbstractFloat}

    T_obs, N = size(X)
    validate_dynamic_factor_inputs(T_obs, N, r, p)
    validate_option(method, "method", (:twostep, :em))

    method == :twostep ?
        _estimate_dfm_twostep(X, r, p; standardize, diagonal_idio) :
        _estimate_dfm_em(X, r, p; standardize, max_iter, tol, diagonal_idio)
end

@float_fallback estimate_dynamic_factors X

# --- Two-Step Estimation ---

function _estimate_dfm_twostep(X::AbstractMatrix{T}, r::Int, p::Int;
    standardize::Bool=true, diagonal_idio::Bool=true
) where {T<:AbstractFloat}

    T_obs, N = size(X)
    static = estimate_factors(X, r; standardize)
    F, Λ = static.factors, static.loadings

    var_model = estimate_var(F, p)
    A = [var_model.B[(2+(lag-1)*r):(1+lag*r), :]' for lag in 1:p]
    factor_residuals, Sigma_eta = var_model.U, var_model.Sigma

    X_proc = standardize ? _standardize(X) : X
    e = X_proc - F * Λ'
    Sigma_e = diagonal_idio ? diagm(vec(var(e, dims=1))) : (e'e) / T_obs

    loglik = _compute_dfm_loglikelihood(X, F, Λ, Sigma_e, standardize)

    DynamicFactorModel{T}(copy(X), F, Λ, A, factor_residuals, Sigma_eta, Sigma_e,
        static.eigenvalues, static.explained_variance, static.cumulative_variance,
        r, p, :twostep, standardize, true, 1, loglik)
end

# --- EM Algorithm ---

function _estimate_dfm_em(X::AbstractMatrix{T}, r::Int, p::Int;
    standardize::Bool=true, max_iter::Int=100, tol::Float64=1e-6, diagonal_idio::Bool=true
) where {T<:AbstractFloat}

    T_obs, N = size(X)
    Y = standardize ? _standardize(X) : copy(X)

    init = _estimate_dfm_twostep(X, r, p; standardize, diagonal_idio)
    Λ, A, Sigma_eta, Sigma_e = copy(init.loadings), deepcopy(init.A), copy(init.Sigma_eta), copy(init.Sigma_e)

    prev_loglik, converged, iter = -Inf, false, 0

    for iteration in 1:max_iter
        iter = iteration
        F_smooth, P_smooth, Pt_smooth, loglik = _kalman_smoother_dfm(Y, Λ, A, Sigma_eta, Sigma_e, r, p)

        abs(loglik - prev_loglik) < tol * abs(prev_loglik) && iteration > 1 && (converged = true; break)
        prev_loglik = loglik

        Λ, A, Sigma_eta, Sigma_e = _em_mstep(Y, F_smooth, P_smooth, Pt_smooth, r, p, diagonal_idio)
    end

    F_smooth, P_smooth, _, loglik = _kalman_smoother_dfm(Y, Λ, A, Sigma_eta, Sigma_e, r, p)
    F = F_smooth[:, 1:r]

    T_eff = T_obs - p
    factor_residuals = zeros(T, T_eff, r)
    for t in (p+1):T_obs
        pred = sum(A[lag] * F[t-lag, :] for lag in 1:p)
        factor_residuals[t-p, :] = F[t, :] - pred
    end

    λ = sort(eigvals(Symmetric(Λ'Λ)), rev=true)
    full_λ = zeros(T, N); full_λ[1:r] = λ
    expl = λ / sum(λ)
    full_expl = zeros(T, N); full_expl[1:r] = expl
    cumul = cumsum(expl)
    full_cumul = zeros(T, N); full_cumul[1:r] = cumul; full_cumul[(r+1):end] .= 1.0

    DynamicFactorModel{T}(copy(X), F, Λ, A, factor_residuals, Sigma_eta, Sigma_e,
        full_λ, full_expl, full_cumul, r, p, :em, standardize, converged, iter, loglik)
end

# --- Kalman Filter/Smoother ---

function _kalman_smoother_dfm(Y::AbstractMatrix{T}, Λ::AbstractMatrix{T}, A::Vector{Matrix{T}},
    Sigma_eta::AbstractMatrix{T}, Sigma_e::AbstractMatrix{T}, r::Int, p::Int
) where {T<:AbstractFloat}

    T_obs, N = size(Y)
    state_dim = r * p

    Z = zeros(T, N, state_dim); Z[:, 1:r] = Λ
    T_mat = zeros(T, state_dim, state_dim)
    for lag in 1:p
        T_mat[1:r, ((lag-1)*r+1):(lag*r)] = A[lag]
    end
    p > 1 && (T_mat[(r+1):state_dim, 1:(state_dim-r)] = I(state_dim - r))

    Q = zeros(T, state_dim, state_dim); Q[1:r, 1:r] = Sigma_eta
    H = Sigma_e

    a0, P0 = zeros(T, state_dim), _compute_unconditional_covariance(T_mat, Q, state_dim)

    a_filt = zeros(T, T_obs, state_dim)
    P_filt = zeros(T, T_obs, state_dim, state_dim)
    a_pred = zeros(T, T_obs, state_dim)
    P_pred = zeros(T, T_obs, state_dim, state_dim)
    loglik, a_t, P_t = zero(T), a0, P0

    for t in 1:T_obs
        a_pred[t, :] = T_mat * a_t
        P_pred[t, :, :] = T_mat * P_t * T_mat' + Q

        v_t = Y[t, :] - Z * a_pred[t, :]
        F_t = Symmetric(Z * P_pred[t, :, :] * Z' + H)
        F_inv = try inv(F_t) catch; pinv(F_t) end
        K_t = P_pred[t, :, :] * Z' * F_inv

        a_filt[t, :] = a_pred[t, :] + K_t * v_t
        P_filt[t, :, :] = (I(state_dim) - K_t * Z) * P_pred[t, :, :]

        det_F = det(F_t)
        det_F > 0 && (loglik -= 0.5 * (N * log(2π) + log(det_F) + v_t' * F_inv * v_t))
        a_t, P_t = a_filt[t, :], P_filt[t, :, :]
    end

    a_smooth = zeros(T, T_obs, state_dim)
    P_smooth = zeros(T, T_obs, state_dim, state_dim)
    Pt_smooth = zeros(T, T_obs-1, state_dim, state_dim)

    a_smooth[T_obs, :], P_smooth[T_obs, :, :] = a_filt[T_obs, :], P_filt[T_obs, :, :]

    for t in (T_obs-1):-1:1
        P_pred_inv = try inv(Symmetric(P_pred[t+1, :, :])) catch; pinv(Symmetric(P_pred[t+1, :, :])) end
        J_t = P_filt[t, :, :] * T_mat' * P_pred_inv
        a_smooth[t, :] = a_filt[t, :] + J_t * (a_smooth[t+1, :] - a_pred[t+1, :])
        P_smooth[t, :, :] = P_filt[t, :, :] + J_t * (P_smooth[t+1, :, :] - P_pred[t+1, :, :]) * J_t'
        t < T_obs && (Pt_smooth[t, :, :] = J_t * P_smooth[t+1, :, :])
    end

    a_smooth, P_smooth, Pt_smooth, loglik
end

function _compute_unconditional_covariance(T_mat::AbstractMatrix{T}, Q::AbstractMatrix{T},
    state_dim::Int; max_iter::Int=1000, tol::Float64=1e-10
) where {T<:AbstractFloat}
    maximum(abs.(eigvals(T_mat))) >= 1.0 && return Matrix{T}(10.0 * I(state_dim))

    P = Matrix{T}(I(state_dim))
    for _ in 1:max_iter
        P_new = T_mat * P * T_mat' + Q
        norm(P_new - P) < tol * norm(P) && return Symmetric(P_new)
        P = P_new
    end
    Symmetric(P)
end

# --- EM M-Step ---

function _em_mstep(Y::AbstractMatrix{T}, F_smooth::AbstractMatrix{T}, P_smooth::AbstractArray{T,3},
    Pt_smooth::AbstractArray{T,3}, r::Int, p::Int, diagonal_idio::Bool
) where {T<:AbstractFloat}

    T_obs, N = size(Y)
    state_dim = r * p

    sum_yF = sum(Y[t, :] * F_smooth[t, 1:r]' for t in 1:T_obs)
    sum_FF = sum(F_smooth[t, 1:r] * F_smooth[t, 1:r]' + P_smooth[t, 1:r, 1:r] for t in 1:T_obs)
    Λ = sum_yF * robust_inv(sum_FF)

    sum_F_Fminus = zeros(T, r, state_dim)
    sum_Fminus_Fminus = zeros(T, state_dim, state_dim)
    for t in (p+1):T_obs
        sum_F_Fminus .+= F_smooth[t, 1:r] * F_smooth[t-1, :]'
        t > p + 1 && (sum_F_Fminus[1:r, 1:state_dim] .+= Pt_smooth[t-1, 1:r, :])
        sum_Fminus_Fminus .+= F_smooth[t-1, :] * F_smooth[t-1, :]' + P_smooth[t-1, :, :]
    end
    A_stacked = sum_F_Fminus * robust_inv(sum_Fminus_Fminus)
    A = [A_stacked[:, ((lag-1)*r+1):(lag*r)] for lag in 1:p]

    T_eff = T_obs - p
    sum_eta = zeros(T, r, r)
    for t in (p+1):T_obs
        eta = F_smooth[t, 1:r] - sum(A[lag] * F_smooth[t-lag, 1:r] for lag in 1:p)
        sum_eta .+= eta * eta' + P_smooth[t, 1:r, 1:r]
        for lag in 1:p
            cross = A[lag] * Pt_smooth[min(t-1, T_obs-1), 1:r, 1:r]'
            sum_eta .-= cross .+ cross'
        end
    end
    Sigma_eta = (sum_eta + sum_eta') / (2 * T_eff)
    min_eig = minimum(eigvals(Symmetric(Sigma_eta)))
    min_eig < 1e-8 && (Sigma_eta += (1e-8 - min_eig) * I(r))

    sum_ee = zeros(T, N, N)
    for t in 1:T_obs
        e = Y[t, :] - Λ * F_smooth[t, 1:r]
        sum_ee .+= e * e' + Λ * P_smooth[t, 1:r, 1:r] * Λ'
    end
    Sigma_e = diagonal_idio ? diagm(diag(sum_ee / T_obs)) : (sum_ee + sum_ee') / (2 * T_obs)
    min_eig = minimum(eigvals(Symmetric(Sigma_e)))
    min_eig < 1e-8 && (Sigma_e += (1e-8 - min_eig) * I(N))

    Λ, A, Sigma_eta, Sigma_e
end

# --- Log-Likelihood ---

function _compute_dfm_loglikelihood(X::AbstractMatrix{T}, F::AbstractMatrix{T},
    Λ::AbstractMatrix{T}, Sigma_e::AbstractMatrix{T}, standardize::Bool
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    Y = standardize ? _standardize(X) : X
    e = Y - F * Λ'

    Sigma_sym = Symmetric(Sigma_e)
    Sigma_inv = try inv(Sigma_sym) catch; pinv(Sigma_sym) end

    ll = -0.5 * T_obs * N * log(2π) - 0.5 * T_obs * logdet(Sigma_sym)
    ll - 0.5 * sum(e[t, :]' * Sigma_inv * e[t, :] for t in 1:T_obs)
end

# --- StatsAPI for DynamicFactorModel ---

StatsAPI.predict(m::DynamicFactorModel) = m.factors * m.loadings'

function StatsAPI.residuals(m::DynamicFactorModel{T}) where {T}
    fitted = predict(m)
    m.standardized ? _standardize(m.X) - fitted : m.X - fitted
end

function StatsAPI.r2(m::DynamicFactorModel{T}) where {T}
    resid = residuals(m)
    X_ref = m.standardized ? _standardize(m.X) : m.X
    [max(zero(T), 1 - var(@view(resid[:, i])) / max(var(@view(X_ref[:, i])), T(1e-10)))
     for i in 1:size(m.X, 2)]
end

StatsAPI.nobs(m::DynamicFactorModel) = size(m.X, 1)
StatsAPI.loglikelihood(m::DynamicFactorModel) = m.loglik

function StatsAPI.dof(m::DynamicFactorModel)
    _, N = size(m.X)
    N * m.r + m.r^2 * m.p + div(m.r * (m.r + 1), 2) + N
end

StatsAPI.aic(m::DynamicFactorModel) = -2m.loglik + 2dof(m)
StatsAPI.bic(m::DynamicFactorModel) = -2m.loglik + dof(m) * log(nobs(m))

# --- Forecasting for DynamicFactorModel ---

"""
    forecast(model::DynamicFactorModel, h; ci=false, ci_level=0.95)

Forecast factors and observables h steps ahead. Returns (factors, observables) or with CI bounds.
"""
function forecast(m::DynamicFactorModel{T}, h::Int; ci::Bool=false, ci_level::Real=0.95) where {T}
    h < 1 && throw(ArgumentError("h must be ≥ 1"))

    r, p, T_obs, N = m.r, m.p, size(m.X, 1), size(m.X, 2)
    F_last = [m.factors[T_obs-lag+1, :] for lag in 1:p]

    F_fc, X_fc = zeros(T, h, r), zeros(T, h, N)
    for horizon in 1:h
        F_h = sum(m.A[lag] * (horizon-lag >= 1 ? F_fc[horizon-lag, :] : F_last[lag-horizon+1]) for lag in 1:p)
        F_fc[horizon, :] = F_h
        X_fc[horizon, :] = m.loadings * F_h
    end

    if m.standardized
        μ, σ = vec(mean(m.X, dims=1)), max.(vec(std(m.X, dims=1)), T(1e-10))
        X_fc = X_fc .* σ' .+ μ'
    end

    !ci && return (factors=F_fc, observables=X_fc)

    n_sim = 1000
    L_eta = safe_cholesky(m.Sigma_eta)
    L_e = safe_cholesky(m.Sigma_e)

    F_sims, X_sims = zeros(T, n_sim, h, r), zeros(T, n_sim, h, N)
    for sim in 1:n_sim
        F_last_sim = copy(F_last)
        for horizon in 1:h
            F_h = sum(m.A[lag] * (horizon-lag >= 1 ? F_sims[sim, horizon-lag, :] : F_last_sim[lag-horizon+1]) for lag in 1:p)
            F_sims[sim, horizon, :] = F_h + L_eta * randn(T, r)
            X_sims[sim, horizon, :] = m.loadings * F_sims[sim, horizon, :] + L_e * randn(T, N)
        end
    end

    if m.standardized
        for sim in 1:n_sim
            X_sims[sim, :, :] = X_sims[sim, :, :] .* σ' .+ μ'
        end
    end

    α_lo, α_hi = (1 - ci_level) / 2, 1 - (1 - ci_level) / 2
    F_lo = [quantile(F_sims[:, h, j], α_lo) for h in 1:h, j in 1:r]
    F_hi = [quantile(F_sims[:, h, j], α_hi) for h in 1:h, j in 1:r]
    X_lo = [quantile(X_sims[:, h, j], α_lo) for h in 1:h, j in 1:N]
    X_hi = [quantile(X_sims[:, h, j], α_hi) for h in 1:h, j in 1:N]

    (factors=F_fc, observables=X_fc, factors_lower=F_lo, factors_upper=F_hi,
     observables_lower=X_lo, observables_upper=X_hi)
end

# --- Model Selection for DFM ---

"""Select (r, p) via AIC/BIC grid search."""
function ic_criteria_dynamic(X::AbstractMatrix{T}, max_r::Int, max_p::Int;
    standardize::Bool=true, method::Symbol=:twostep
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    1 <= max_r <= min(T_obs, N) || throw(ArgumentError("Invalid max_r"))
    1 <= max_p < T_obs - max_r || throw(ArgumentError("Invalid max_p"))

    AIC_mat, BIC_mat = fill(T(Inf), max_r, max_p), fill(T(Inf), max_r, max_p)
    for r in 1:max_r, p in 1:max_p
        p >= T_obs - r - 10 && continue
        try
            m = estimate_dynamic_factors(X, r, p; method, standardize)
            AIC_mat[r, p], BIC_mat[r, p] = aic(m), bic(m)
        catch; continue; end
    end

    aic_idx, bic_idx = argmin(AIC_mat), argmin(BIC_mat)
    (AIC=AIC_mat, BIC=BIC_mat, r_AIC=aic_idx[1], p_AIC=aic_idx[2], r_BIC=bic_idx[1], p_BIC=bic_idx[2])
end

# --- Companion Matrix & Stationarity ---

"""Companion matrix for factor VAR dynamics."""
function companion_matrix_factors(m::DynamicFactorModel{T}) where {T}
    r, p = m.r, m.p
    C = zeros(T, r * p, r * p)
    for lag in 1:p
        C[1:r, ((lag-1)*r+1):(lag*r)] = m.A[lag]
    end
    p > 1 && (C[(r+1):end, 1:(r*(p-1))] = I(r * (p - 1)))
    C
end

"""Check if factor dynamics are stationary (max|eigenvalue| < 1)."""
is_stationary(m::DynamicFactorModel) = maximum(abs.(eigvals(companion_matrix_factors(m)))) < 1.0

# #############################################################################
# PART 3: GENERALIZED DYNAMIC FACTOR MODEL (GDFM)
# #############################################################################

"""
    estimate_gdfm(X, q; standardize=true, bandwidth=0, kernel=:bartlett, r=0) -> GeneralizedDynamicFactorModel

Estimate GDFM using spectral methods. `q` = number of dynamic factors.
"""
function estimate_gdfm(X::AbstractMatrix{T}, q::Int;
    standardize::Bool=true, bandwidth::Int=0, kernel::Symbol=:bartlett, r::Int=0
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    validate_factor_inputs(T_obs, N, q; context="dynamic factors")
    validate_option(kernel, "kernel", (:bartlett, :parzen, :tukey))

    r_static = r == 0 ? q : r
    r_static < q && throw(ArgumentError("r must be >= q"))
    bandwidth = bandwidth <= 0 ? _select_bandwidth(T_obs) : bandwidth

    X_original = copy(X)
    X_proc = standardize ? _standardize(X) : X

    frequencies, spectral_X = _estimate_spectral_density(X_proc, bandwidth, kernel)
    eigenvalues, eigenvectors = _spectral_eigendecomposition(spectral_X)
    loadings = eigenvectors[:, 1:q, :]
    spectral_chi = _compute_common_spectral_density(loadings, eigenvalues[1:q, :])
    common = _reconstruct_time_domain(spectral_chi, X_proc)
    factors = _extract_time_domain_factors(X_proc, loadings, frequencies)
    var_explained = _compute_variance_explained(eigenvalues, q)

    if standardize
        μ, σ = mean(X_original, dims=1), max.(std(X_original, dims=1), T(1e-10))
        common = common .* σ .+ μ
    end
    idiosyncratic = X_original - common

    GeneralizedDynamicFactorModel{T}(X_original, factors, common, idiosyncratic, loadings,
        spectral_X, spectral_chi, eigenvalues, frequencies, q, r_static, bandwidth,
        kernel, standardize, var_explained)
end

@float_fallback estimate_gdfm X

# --- Spectral Density Estimation ---

"""Automatic bandwidth: T^(1/3)."""
_select_bandwidth(T_obs::Int) = max(3, round(Int, T_obs^(1/3)))

"""Estimate spectral density with kernel smoothing."""
function _estimate_spectral_density(X::AbstractMatrix{T}, bandwidth::Int, kernel::Symbol) where {T<:AbstractFloat}
    T_obs, N = size(X)
    n_freq = div(T_obs, 2) + 1
    frequencies = [T(2π * (j-1) / T_obs) for j in 1:n_freq]

    X_fft = fft(X, 1)
    periodogram = [X_fft[j, :] * X_fft[j, :]' / T_obs for j in 1:n_freq]
    weights = _compute_kernel_weights(bandwidth, kernel)

    spectral = Array{Complex{T},3}(undef, N, N, n_freq)
    @inbounds for j in 1:n_freq
        S = zeros(Complex{T}, N, N)
        for k in -bandwidth:bandwidth
            idx = clamp(j + k < 1 ? 2 - (j + k) : (j + k > n_freq ? 2*n_freq - (j + k) : j + k), 1, n_freq)
            S .+= weights[abs(k) + 1] * periodogram[idx]
        end
        spectral[:, :, j] = (S + S') / 2
    end
    frequencies, spectral
end

"""Compute kernel weights (bartlett, parzen, tukey)."""
function _compute_kernel_weights(bandwidth::Int, kernel::Symbol)
    weights = zeros(bandwidth + 1)
    for k in 0:bandwidth
        u = k / (bandwidth + 1)
        weights[k + 1] = kernel == :bartlett ? 1 - u :
                         kernel == :parzen ? (u <= 0.5 ? 1 - 6u^2 + 6u^3 : 2(1-u)^3) :
                         0.5 * (1 + cos(π * u))
    end
    total = weights[1] + 2sum(weights[2:end])
    weights ./ total
end

# --- Spectral Eigendecomposition & Reconstruction ---

"""Eigendecomposition at each frequency."""
function _spectral_eigendecomposition(spectral::Array{Complex{T},3}) where {T<:AbstractFloat}
    N, _, n_freq = size(spectral)
    eigenvalues, eigenvectors = Matrix{T}(undef, N, n_freq), Array{Complex{T},3}(undef, N, N, n_freq)

    @inbounds for j in 1:n_freq
        E = eigen(Hermitian(spectral[:, :, j]))
        idx = sortperm(real.(E.values), rev=true)
        eigenvalues[:, j] = real.(E.values[idx])
        eigenvectors[:, :, j] = E.vectors[:, idx]
    end
    eigenvalues, eigenvectors
end

"""Compute spectral density of common component."""
function _compute_common_spectral_density(loadings::Array{Complex{T},3}, eigenvalues::AbstractMatrix) where {T}
    N, q, n_freq = size(loadings)
    spectral_chi = Array{Complex{T},3}(undef, N, N, n_freq)
    @inbounds for j in 1:n_freq
        L = loadings[:, :, j]
        spectral_chi[:, :, j] = L * Diagonal(eigenvalues[:, j]) * L'
    end
    spectral_chi
end

"""Reconstruct common component via inverse FFT."""
function _reconstruct_time_domain(spectral_chi::Array{Complex{T},3}, X::AbstractMatrix{T}) where {T}
    T_obs, N = size(X)
    n_freq = size(spectral_chi, 3)
    X_fft = fft(X, 1)
    chi_fft = zeros(Complex{T}, T_obs, N)

    @inbounds for j in 1:n_freq
        S_chi, S_X = spectral_chi[:, :, j], X_fft[j, :] * X_fft[j, :]' / T_obs
        P = S_chi * inv(Hermitian(S_X + T(1e-10) * I))
        chi_fft[j, :] = P * X_fft[j, :]
        j > 1 && j < n_freq && (chi_fft[T_obs - j + 2, :] = conj(chi_fft[j, :]))
    end
    real(ifft(chi_fft, 1))
end

"""Extract time-domain factors via frequency-domain projection."""
function _extract_time_domain_factors(X::AbstractMatrix{T}, loadings::Array{Complex{T},3}, frequencies::Vector{T}) where {T}
    T_obs, N = size(X)
    _, q, n_freq = size(loadings)
    X_fft, F_fft = fft(X, 1), zeros(Complex{T}, T_obs, q)

    @inbounds for j in 1:n_freq
        L = loadings[:, :, j]
        F_fft[j, :] = (L' * L + T(1e-10) * I) \ (L' * X_fft[j, :])
        j > 1 && j < n_freq && (F_fft[T_obs - j + 2, :] = conj(F_fft[j, :]))
    end

    factors = real(ifft(F_fft, 1))
    for i in 1:q
        σ = std(factors[:, i])
        σ > T(1e-10) && (factors[:, i] ./= σ)
    end
    factors
end

"""Variance explained by first q factors."""
function _compute_variance_explained(eigenvalues::Matrix{T}, q::Int) where {T}
    total = mean(sum(eigenvalues, dims=1))
    [mean(eigenvalues[i, :]) / total for i in 1:q]
end

# --- StatsAPI for GDFM ---

StatsAPI.predict(m::GeneralizedDynamicFactorModel) = m.common_component
StatsAPI.residuals(m::GeneralizedDynamicFactorModel) = m.idiosyncratic
StatsAPI.nobs(m::GeneralizedDynamicFactorModel) = size(m.X, 1)
StatsAPI.dof(m::GeneralizedDynamicFactorModel) = m.q * size(m.X, 2) * length(m.frequencies) + size(m.X, 1) * m.q

function StatsAPI.r2(m::GeneralizedDynamicFactorModel{T}) where {T}
    N = size(m.X, 2)
    [one(T) - var(m.idiosyncratic[:, i]) / max(var(m.X[:, i]), T(1e-10)) for i in 1:N]
end

# --- Information Criteria for GDFM ---

"""
    ic_criteria_gdfm(X, max_q; standardize=true, bandwidth=0, kernel=:bartlett)

Information criteria for selecting number of dynamic factors.
"""
function ic_criteria_gdfm(X::AbstractMatrix{T}, max_q::Int;
    standardize::Bool=true, bandwidth::Int=0, kernel::Symbol=:bartlett
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    (max_q < 1 || max_q > N) && throw(ArgumentError("max_q must be in [1, $N]"))
    bandwidth = bandwidth <= 0 ? _select_bandwidth(T_obs) : bandwidth

    X_proc = standardize ? _standardize(X) : X
    _, spectral = _estimate_spectral_density(X_proc, bandwidth, kernel)
    eigenvalues, _ = _spectral_eigendecomposition(spectral)
    avg_eig = vec(mean(eigenvalues, dims=2))

    ratios = [avg_eig[i] / avg_eig[i+1] for i in 1:min(max_q, N-1)]
    cum_var = cumsum(avg_eig[1:max_q]) / sum(avg_eig)
    q_ratio = argmax(ratios[1:min(max_q, length(ratios))])
    q_variance = something(findfirst(>=(T(0.9)), cum_var), max_q)

    (eigenvalue_ratios=ratios, cumulative_variance=cum_var, avg_eigenvalues=avg_eig[1:max_q],
     q_ratio=q_ratio, q_variance=q_variance)
end

# --- Forecasting for GDFM ---

"""
    forecast(model::GeneralizedDynamicFactorModel, h; method=:ar) -> (common, factors)

Forecast h steps ahead using AR extrapolation of factors.
"""
function forecast(model::GeneralizedDynamicFactorModel{T}, h::Int; method::Symbol=:ar) where {T}
    h < 1 && throw(ArgumentError("h must be positive"))
    method ∉ (:ar, :spectral) && throw(ArgumentError("method must be :ar or :spectral"))

    factors_fc = _forecast_factors_ar(model.factors, h)
    L_avg = real.(model.loadings_spectral[:, :, 1])
    common_fc = factors_fc * L_avg'

    (common=common_fc, factors=factors_fc)
end

"""AR(1) forecast for factors."""
function _forecast_factors_ar(factors::Matrix{T}, h::Int) where {T<:AbstractFloat}
    T_obs, q = size(factors)
    fc = Matrix{T}(undef, h, q)

    for i in 1:q
        F = factors[:, i]
        phi = dot(F[1:end-1], F[2:end]) / dot(F[1:end-1], F[1:end-1])
        f = F[end]
        for t in 1:h
            f = phi * f
            fc[t, i] = f
        end
    end
    fc
end

# --- GDFM Utilities ---

"""Variance share explained by common component for each variable."""
function common_variance_share(m::GeneralizedDynamicFactorModel{T}) where {T}
    N = size(m.X, 2)
    [var(m.common_component[:, i]) / max(var(m.X[:, i]), T(1e-10)) for i in 1:N]
end

"""Data for plotting eigenvalues across frequencies."""
spectral_eigenvalue_plot_data(m::GeneralizedDynamicFactorModel) =
    (frequencies=m.frequencies, eigenvalues=m.eigenvalues_spectral)
