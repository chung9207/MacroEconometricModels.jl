"""
Type definitions for Bayesian VAR posterior results.
"""

# =============================================================================
# Bayesian VAR Posterior
# =============================================================================

"""
    BVARPosterior{T} <: Any

Posterior draws from Bayesian VAR estimation.

Replaces MCMCChains.Chains — stores i.i.d. or Gibbs draws from the
Normal-Inverse-Wishart posterior directly.

# Fields
- `B_draws::Array{T,3}`: Coefficient draws (n_draws × k × n)
- `Sigma_draws::Array{T,3}`: Covariance draws (n_draws × n × n)
- `n_draws::Int`: Number of posterior draws
- `p::Int`: Number of VAR lags
- `n::Int`: Number of variables
- `data::Matrix{T}`: Original Y matrix (for residual computation downstream)
- `prior::Symbol`: Prior used (:normal or :minnesota)
- `sampler::Symbol`: Sampler used (:direct or :gibbs)
"""
struct BVARPosterior{T<:AbstractFloat}
    B_draws::Array{T,3}       # n_draws × k × n
    Sigma_draws::Array{T,3}   # n_draws × n × n
    n_draws::Int
    p::Int
    n::Int
    data::Matrix{T}
    prior::Symbol
    sampler::Symbol
end

Base.size(post::BVARPosterior, dim::Int) = dim == 1 ? post.n_draws : error("BVARPosterior has 1 dimension (n_draws)")
Base.length(post::BVARPosterior) = post.n_draws
