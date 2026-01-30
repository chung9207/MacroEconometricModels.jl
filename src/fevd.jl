"""
Forecast Error Variance Decomposition for frequentist and Bayesian VAR models.
"""

using LinearAlgebra, Statistics, MCMCChains

# =============================================================================
# Frequentist FEVD
# =============================================================================

"""
    fevd(model, horizon; method=:cholesky, ...) -> FEVD

Compute FEVD showing proportion of h-step forecast error variance attributable to each shock.
"""
function fevd(model::VARModel{T}, horizon::Int;
    method::Symbol=:cholesky, check_func=nothing, narrative_check=nothing
) where {T<:AbstractFloat}
    irf_result = irf(model, horizon; method, check_func, narrative_check)
    decomp, props = _compute_fevd(irf_result.values, nvars(model), horizon)
    FEVD{T}(decomp, props)
end

"""Compute FEVD from IRF array: decomposition[i,j,h] = cumulative MSE contribution."""
function _compute_fevd(irfs::Array{T,3}, n::Int, horizon::Int) where {T<:AbstractFloat}
    decomp, props = zeros(T, n, n, horizon), zeros(T, n, n, horizon)
    mse = zeros(T, n, horizon)

    @inbounds for h in 1:horizon
        for i in 1:n
            total = zero(T)
            for j in 1:n
                prev = h == 1 ? zero(T) : decomp[i, j, h-1]
                decomp[i, j, h] = prev + irfs[h, i, j]^2
                total += decomp[i, j, h]
            end
            mse[i, h] = total
            total > 0 && (props[i, :, h] = decomp[i, :, h] ./ total)
        end
    end
    decomp, props
end

# =============================================================================
# Bayesian FEVD
# =============================================================================

"""
    fevd(chain, p, n, horizon; quantiles=[0.16, 0.5, 0.84], ...) -> BayesianFEVD

Compute Bayesian FEVD from MCMC chain with posterior quantiles.
"""
function fevd(chain::Chains, p::Int, n::Int, horizon::Int;
    method::Symbol=:cholesky, data::AbstractMatrix=Matrix{Float64}(undef, 0, 0),
    check_func=nothing, narrative_check=nothing, quantiles::Vector{<:Real}=[0.16, 0.5, 0.84]
)
    method == :narrative && isempty(data) && throw(ArgumentError("Narrative needs data"))

    samples = size(chain, 1)
    ET = isempty(data) ? Float64 : eltype(data)
    all_fevds = zeros(ET, samples, n, n, horizon)

    b_vecs, sigmas = extract_chain_parameters(chain)
    for s in 1:samples
        m = parameters_to_model(b_vecs[s, :], sigmas[s, :], p, n, data)
        Q = compute_Q(m, method, horizon, check_func, narrative_check; max_draws=100)
        irf_vals = compute_irf(m, Q, horizon)
        _, props = _compute_fevd(irf_vals, n, horizon)
        all_fevds[s, :, :, :] = props
    end

    q_vec = ET.(quantiles)
    fevd_q = zeros(ET, horizon, n, n, length(quantiles))
    fevd_m = zeros(ET, horizon, n, n)

    @inbounds for h in 1:horizon, v in 1:n, sh in 1:n
        d = @view all_fevds[:, v, sh, h]
        fevd_q[h, v, sh, :] = quantile(d, q_vec)
        fevd_m[h, v, sh] = mean(d)
    end

    BayesianFEVD{ET}(fevd_q, fevd_m, horizon, default_var_names(n), default_shock_names(n), q_vec)
end
