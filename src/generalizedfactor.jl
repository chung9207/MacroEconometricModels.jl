"""
Generalized Dynamic Factor Model (GDFM) via Spectral Methods.

Implements Forni, Hallin, Lippi & Reichlin (2000, 2005) GDFM:
X_t = χ_t + ξ_t (common + idiosyncratic components)

The common component has a factor structure with frequency-dependent loadings,
estimated via spectral density analysis.

References:
- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2000). The generalized dynamic-factor
  model: Identification and estimation. Review of Economics and Statistics.
- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2005). The generalized dynamic factor
  model: One-sided estimation and forecasting. Journal of the American Statistical Association.
"""

using LinearAlgebra, Statistics, FFTW, StatsAPI

# =============================================================================
# GDFM Estimation
# =============================================================================

"""
    estimate_gdfm(X, q; standardize=true, bandwidth=0, kernel=:bartlett, r=0) -> GeneralizedDynamicFactorModel

Estimate Generalized Dynamic Factor Model using spectral methods.

# Arguments
- `X`: Data matrix (T × N)
- `q`: Number of dynamic factors

# Keyword Arguments
- `standardize::Bool=true`: Standardize data
- `bandwidth::Int=0`: Kernel bandwidth (0 = automatic selection)
- `kernel::Symbol=:bartlett`: Kernel for spectral smoothing (:bartlett, :parzen, :tukey)
- `r::Int=0`: Number of static factors (0 = same as q)

# Returns
`GeneralizedDynamicFactorModel` with common/idiosyncratic components and spectral loadings.

# Example
```julia
gdfm = estimate_gdfm(X, 3)
common_variance_share(gdfm)  # Fraction of variance explained by common component
```
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

    # Spectral analysis
    frequencies, spectral_X = _estimate_spectral_density(X_proc, bandwidth, kernel)
    eigenvalues, eigenvectors = _spectral_eigendecomposition(spectral_X)
    loadings = eigenvectors[:, 1:q, :]
    spectral_chi = _compute_common_spectral_density(loadings, eigenvalues[1:q, :])
    common = _reconstruct_time_domain(spectral_chi, X_proc)
    factors = _extract_time_domain_factors(X_proc, loadings, frequencies)
    var_explained = _compute_variance_explained(eigenvalues, q)

    # Unstandardize common component if needed
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

# =============================================================================
# Bandwidth Selection
# =============================================================================

"""Automatic bandwidth selection: T^(1/3)."""
_select_bandwidth(T_obs::Int) = max(3, round(Int, T_obs^(1/3)))

# =============================================================================
# Spectral Density Estimation
# =============================================================================

"""Estimate spectral density matrix with kernel smoothing."""
function _estimate_spectral_density(X::AbstractMatrix{T}, bandwidth::Int, kernel::Symbol) where {T<:AbstractFloat}
    T_obs, N = size(X)
    n_freq = div(T_obs, 2) + 1
    frequencies = [T(2π * (j-1) / T_obs) for j in 1:n_freq]

    # Periodogram
    X_fft = fft(X, 1)
    periodogram = [X_fft[j, :] * X_fft[j, :]' / T_obs for j in 1:n_freq]

    # Kernel smoothing
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

"""Compute kernel weights for spectral smoothing."""
function _compute_kernel_weights(bandwidth::Int, kernel::Symbol)
    weights = zeros(bandwidth + 1)
    for k in 0:bandwidth
        u = k / (bandwidth + 1)
        weights[k + 1] = kernel == :bartlett ? 1 - u :
                         kernel == :parzen ? (u <= 0.5 ? 1 - 6u^2 + 6u^3 : 2(1-u)^3) :
                         0.5 * (1 + cos(π * u))  # tukey
    end
    total = weights[1] + 2sum(weights[2:end])
    weights ./ total
end

# =============================================================================
# Spectral Eigendecomposition
# =============================================================================

"""Eigendecomposition of spectral density at each frequency."""
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

# =============================================================================
# Common Component Reconstruction
# =============================================================================

"""Compute spectral density of common component from loadings and eigenvalues."""
function _compute_common_spectral_density(loadings::Array{Complex{T},3}, eigenvalues::AbstractMatrix) where {T}
    N, q, n_freq = size(loadings)
    spectral_chi = Array{Complex{T},3}(undef, N, N, n_freq)
    @inbounds for j in 1:n_freq
        L = loadings[:, :, j]
        spectral_chi[:, :, j] = L * Diagonal(eigenvalues[:, j]) * L'
    end
    spectral_chi
end

"""Reconstruct common component in time domain via inverse FFT."""
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
    # Normalize factors to unit variance
    for i in 1:q
        σ = std(factors[:, i])
        σ > T(1e-10) && (factors[:, i] ./= σ)
    end
    factors
end

"""Compute variance explained by first q factors (averaged across frequencies)."""
function _compute_variance_explained(eigenvalues::Matrix{T}, q::Int) where {T}
    total = mean(sum(eigenvalues, dims=1))
    [mean(eigenvalues[i, :]) / total for i in 1:q]
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

"""Predicted values (common component)."""
StatsAPI.predict(m::GeneralizedDynamicFactorModel) = m.common_component

"""Residuals (idiosyncratic component)."""
StatsAPI.residuals(m::GeneralizedDynamicFactorModel) = m.idiosyncratic

"""Number of observations."""
StatsAPI.nobs(m::GeneralizedDynamicFactorModel) = size(m.X, 1)

"""Degrees of freedom."""
StatsAPI.dof(m::GeneralizedDynamicFactorModel) = m.q * size(m.X, 2) * length(m.frequencies) + size(m.X, 1) * m.q

"""R² for each variable."""
function StatsAPI.r2(m::GeneralizedDynamicFactorModel{T}) where {T}
    N = size(m.X, 2)
    [one(T) - var(m.idiosyncratic[:, i]) / max(var(m.X[:, i]), T(1e-10)) for i in 1:N]
end

# =============================================================================
# Information Criteria for GDFM
# =============================================================================

"""
    ic_criteria_gdfm(X, max_q; standardize=true, bandwidth=0, kernel=:bartlett)

Information criteria for selecting number of dynamic factors.

Uses eigenvalue ratio test and cumulative variance threshold.

# Returns
Named tuple with:
- `eigenvalue_ratios`: Ratios of consecutive eigenvalues
- `cumulative_variance`: Cumulative variance explained
- `q_ratio`: Optimal q from eigenvalue ratio
- `q_variance`: Optimal q from 90% variance threshold
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

    # Eigenvalue ratio criterion
    ratios = [avg_eig[i] / avg_eig[i+1] for i in 1:min(max_q, N-1)]
    cum_var = cumsum(avg_eig[1:max_q]) / sum(avg_eig)
    q_ratio = argmax(ratios[1:min(max_q, length(ratios))])
    q_variance = something(findfirst(>=(T(0.9)), cum_var), max_q)

    (eigenvalue_ratios=ratios, cumulative_variance=cum_var, avg_eigenvalues=avg_eig[1:max_q],
     q_ratio=q_ratio, q_variance=q_variance)
end

# =============================================================================
# Forecasting
# =============================================================================

"""
    forecast(model::GeneralizedDynamicFactorModel, h; method=:ar) -> (common, factors)

Forecast h steps ahead using AR extrapolation of factors.

# Arguments
- `model`: Estimated GDFM
- `h`: Forecast horizon

# Keyword Arguments
- `method::Symbol=:ar`: Forecasting method (currently only :ar supported)

# Returns
Named tuple with forecasted common component and factors.
"""
function forecast(model::GeneralizedDynamicFactorModel{T}, h::Int; method::Symbol=:ar) where {T}
    h < 1 && throw(ArgumentError("h must be positive"))
    method ∉ (:ar, :spectral) && throw(ArgumentError("method must be :ar or :spectral"))

    factors_fc = _forecast_factors_ar(model.factors, h)
    L_avg = real.(model.loadings_spectral[:, :, 1])
    common_fc = factors_fc * L_avg'

    (common=common_fc, factors=factors_fc)
end

"""AR(1) forecast for each factor series."""
function _forecast_factors_ar(factors::Matrix{T}, h::Int) where {T<:AbstractFloat}
    T_obs, q = size(factors)
    fc = Matrix{T}(undef, h, q)

    for i in 1:q
        F = factors[:, i]
        # Estimate AR(1) coefficient
        phi = dot(F[1:end-1], F[2:end]) / dot(F[1:end-1], F[1:end-1])
        f = F[end]
        for t in 1:h
            f = phi * f
            fc[t, i] = f
        end
    end
    fc
end

# =============================================================================
# GDFM Utilities
# =============================================================================

"""
    common_variance_share(model::GeneralizedDynamicFactorModel) -> Vector

Fraction of each variable's variance explained by the common component.
"""
function common_variance_share(m::GeneralizedDynamicFactorModel{T}) where {T}
    N = size(m.X, 2)
    [var(m.common_component[:, i]) / max(var(m.X[:, i]), T(1e-10)) for i in 1:N]
end

"""
    spectral_eigenvalue_plot_data(model::GeneralizedDynamicFactorModel)

Return data for plotting eigenvalues across frequencies.
"""
spectral_eigenvalue_plot_data(m::GeneralizedDynamicFactorModel) =
    (frequencies=m.frequencies, eigenvalues=m.eigenvalues_spectral)
