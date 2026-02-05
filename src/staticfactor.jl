"""
Static Factor Model via Principal Component Analysis.

Implements Bai & Ng (2002) static factor model: X_t = Λ F_t + e_t

References:
- Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor models.
  Econometrica, 70(1), 191-221.
"""

using LinearAlgebra, Statistics, StatsAPI

# =============================================================================
# Static Factor Model Estimation
# =============================================================================

"""
    estimate_factors(X, r; standardize=true) -> FactorModel

Estimate static factor model X_t = Λ F_t + e_t via Principal Component Analysis.

# Arguments
- `X`: Data matrix (T × N), observations × variables
- `r`: Number of factors to extract

# Keyword Arguments
- `standardize::Bool=true`: Standardize data before estimation

# Returns
`FactorModel` containing factors, loadings, eigenvalues, and explained variance.

# Example
```julia
X = randn(200, 50)  # 200 observations, 50 variables
fm = estimate_factors(X, 3)  # Extract 3 factors
r2(fm)  # R² for each variable
```
"""
function estimate_factors(X::AbstractMatrix{T}, r::Int; standardize::Bool=true) where {T<:AbstractFloat}
    T_obs, N = size(X)
    validate_factor_inputs(T_obs, N, r)

    X_orig = copy(X)
    X_proc = standardize ? _standardize(X) : X

    # Eigendecomposition of sample covariance
    Σ = (X_proc'X_proc) / T_obs
    eig = eigen(Symmetric(Σ))
    idx = sortperm(eig.values, rev=true)
    λ, V = eig.values[idx], eig.vectors[:, idx]

    # Extract loadings and factors
    loadings = V[:, 1:r] * Diagonal(sqrt.(λ[1:r]))
    factors = X_proc * V[:, 1:r]

    # Variance explained
    total = sum(λ)
    expl = λ / total
    cumul = cumsum(expl)

    FactorModel{T}(X_orig, factors, loadings, λ, expl, cumul, r, standardize)
end

@float_fallback estimate_factors X

# =============================================================================
# StatsAPI Interface
# =============================================================================

"""Predicted values: F * Λ'."""
StatsAPI.predict(m::FactorModel) = m.factors * m.loadings'

"""Residuals: X - predicted."""
function StatsAPI.residuals(m::FactorModel{T}) where {T}
    fitted = predict(m)
    m.standardized ? _standardize(m.X) - fitted : m.X - fitted
end

"""R² for each variable."""
function StatsAPI.r2(m::FactorModel{T}) where {T}
    resid = residuals(m)
    X_ref = m.standardized ? _standardize(m.X) : m.X
    [max(zero(T), 1 - var(@view(resid[:, i])) / max(var(@view(X_ref[:, i])), T(1e-10)))
     for i in 1:size(m.X, 2)]
end

"""Number of observations."""
StatsAPI.nobs(m::FactorModel) = size(m.X, 1)

"""Degrees of freedom."""
StatsAPI.dof(m::FactorModel) = size(m.X, 2) * m.r + size(m.X, 1) * m.r - m.r^2

# =============================================================================
# Information Criteria (Bai & Ng 2002)
# =============================================================================

"""
    ic_criteria(X, max_factors; standardize=true)

Compute Bai-Ng (2002) information criteria IC1, IC2, IC3 for selecting the number of factors.

# Arguments
- `X`: Data matrix (T × N)
- `max_factors`: Maximum number of factors to consider

# Returns
Named tuple with IC values and optimal factor counts:
- `IC1`, `IC2`, `IC3`: Information criteria vectors
- `r_IC1`, `r_IC2`, `r_IC3`: Optimal factor counts

# Example
```julia
result = ic_criteria(X, 10)
println("Optimal factors: IC1=", result.r_IC1, ", IC2=", result.r_IC2, ", IC3=", result.r_IC3)
```
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

# =============================================================================
# Visualization Helpers
# =============================================================================

"""
    scree_plot_data(m::FactorModel)

Return data for scree plot: factor indices, explained variance, cumulative variance.

# Example
```julia
data = scree_plot_data(fm)
# Plot: data.factors vs data.explained_variance
```
"""
scree_plot_data(m::FactorModel) = (factors=1:length(m.eigenvalues), explained_variance=m.explained_variance,
                                    cumulative_variance=m.cumulative_variance)
