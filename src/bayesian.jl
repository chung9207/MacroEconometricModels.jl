"""
Bayesian VAR estimation via Turing.jl with MCMC sampling.
"""

using Turing, MCMCChains, LinearAlgebra

# =============================================================================
# Turing Models
# =============================================================================

"""Vectorized BVAR model for gradient-based samplers (NUTS, HMC, HMCDA)."""
@model function var_bayes_vectorized(Y_eff::Matrix{T}, X::Matrix{T}, p::Int, n::Int) where {T<:AbstractFloat}
    Sigma ~ InverseWishart(n + 2, Matrix{T}(I, n, n))
    k = 1 + n * p
    b_vec ~ MvNormal(zeros(T, k * n), T(10) * I)
    B = reshape(b_vec, k, n)

    mu = X * B
    residuals = permutedims(Y_eff) .- permutedims(mu)
    Turing.@addlogprob! sum(logpdf(MvNormal(zeros(T, n), Sigma), residuals))
end

"""Sequential BVAR model for particle-based samplers (SMC, PG)."""
@model function var_bayes_sequential(Y_eff::Matrix{T}, X::Matrix{T}, p::Int, n::Int) where {T<:AbstractFloat}
    Sigma ~ InverseWishart(n + 2, Matrix{T}(I, n, n))
    k = 1 + n * p
    b_vec ~ MvNormal(zeros(T, k * n), T(10) * I)
    B = reshape(b_vec, k, n)
    mu = X * B

    for t in 1:size(Y_eff, 1)
        Y_eff[t, :] ~ MvNormal(mu[t, :], Sigma)
    end
end

# =============================================================================
# Sampler Configuration
# =============================================================================

@enum SamplerType SAMPLER_NUTS SAMPLER_HMC SAMPLER_HMCDA SAMPLER_IS SAMPLER_SMC SAMPLER_PG

"""Create Turing sampler from symbol. Supports: :nuts, :hmc, :hmcda, :is, :smc, :pg."""
function get_sampler(sampler_type::Symbol, n_adapts::Int, args::NamedTuple)
    sampler_type == :nuts && return NUTS(n_adapts, 0.65)
    sampler_type == :hmc && return HMC(get(args, :epsilon, 0.1), get(args, :n_leapfrog, 10))
    sampler_type == :hmcda && return HMCDA(n_adapts, get(args, :delta, 0.65), get(args, :lambda, 0.3))
    sampler_type == :is && return IS()
    sampler_type == :smc && return SMC(get(args, :n_particles, 100))
    sampler_type == :pg && return PG(get(args, :n_particles, 20))
    throw(ArgumentError("Unknown sampler: $sampler_type. Use :nuts, :hmc, :hmcda, :is, :smc, :pg"))
end

requires_sequential_model(s::Symbol) = s in (:smc, :pg)

# =============================================================================
# Main Estimation
# =============================================================================

"""
    estimate_bvar(Y, p; n_samples=1000, n_adapts=500, prior=:normal, hyper=nothing,
                  sampler=:nuts, sampler_args=(;)) -> Chains

Estimate Bayesian VAR via Turing.jl MCMC.

Samplers: :nuts (default), :hmc, :hmcda, :is, :smc, :pg.
Prior: :normal (default) or :minnesota with optional `hyper::MinnesotaHyperparameters`.
"""
function estimate_bvar(Y::AbstractMatrix{T}, p::Int;
    n_samples::Int=1000, n_adapts::Int=500, prior::Symbol=:normal,
    hyper::Union{Nothing,MinnesotaHyperparameters}=nothing,
    sampler::Symbol=:nuts, sampler_args::NamedTuple=(;)
) where {T<:AbstractFloat}

    T_obs, n = size(Y)
    validate_var_inputs(T_obs, n, p)

    Y_eff, X = construct_var_matrices(Y, p)

    # Apply Minnesota prior augmentation if requested
    Y_data, X_data = if prior == :minnesota
        h = isnothing(hyper) ? MinnesotaHyperparameters() : hyper
        Y_d, X_d = gen_dummy_obs(Y, p, h)
        (vcat(Y_eff, Y_d), vcat(X, X_d))
    else
        (Y_eff, X)
    end

    model = requires_sequential_model(sampler) ?
        var_bayes_sequential(Y_data, X_data, p, n) :
        var_bayes_vectorized(Y_data, X_data, p, n)

    sample(model, get_sampler(sampler, n_adapts, sampler_args), n_samples; progress=true)
end

@float_fallback estimate_bvar Y

# =============================================================================
# Chain Parameter Extraction
# =============================================================================

"""Extract (b_vecs, sigmas_mat) from MCMC chain."""
extract_chain_parameters(chain::Chains) = (Array(group(chain, :b_vec)), Array(group(chain, :Sigma)))

"""Convert chain parameters to VARModel. Provide `data` for residual computation."""
function parameters_to_model(b_vec::AbstractVector{T}, sigma_vec::AbstractVector{T},
                             p::Int, n::Int, data::AbstractMatrix{T}=Matrix{T}(undef, 0, 0)) where {T<:AbstractFloat}
    k = 1 + n * p
    B, Sigma = reshape(b_vec, k, n), reshape(sigma_vec, n, n)

    U = if !isempty(data) && size(data, 1) > p
        Y_eff, X = construct_var_matrices(data, p)
        Y_eff - X * B
    else
        Matrix{T}(undef, 0, n)
    end

    VARModel(isempty(data) ? zeros(T, 0, n) : data, p, B, U, Sigma, zero(T), zero(T), zero(T))
end

function parameters_to_model(b_vec, sigma_vec, p::Int, n::Int, data::AbstractMatrix=Matrix{Float64}(undef, 0, 0))
    T = promote_type(eltype(b_vec), eltype(sigma_vec))
    parameters_to_model(Vector{T}(b_vec), Vector{T}(sigma_vec), p, n, Matrix{T}(data))
end

# =============================================================================
# Posterior Summary
# =============================================================================

"""VARModel with posterior mean parameters."""
function posterior_mean_model(chain::Chains, p::Int, n::Int; data::AbstractMatrix=Matrix{Float64}(undef, 0, 0))
    b, s = extract_chain_parameters(chain)
    parameters_to_model(vec(mean(b, dims=1)), vec(mean(s, dims=1)), p, n, data)
end

"""VARModel with posterior median parameters."""
function posterior_median_model(chain::Chains, p::Int, n::Int; data::AbstractMatrix=Matrix{Float64}(undef, 0, 0))
    b, s = extract_chain_parameters(chain)
    parameters_to_model(vec(median(b, dims=1)), vec(median(s, dims=1)), p, n, data)
end
