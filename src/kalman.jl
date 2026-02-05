"""
Kalman filter/smoother utilities for dynamic factor models.

These utilities are used by the dynamic factor model estimation.
"""

using LinearAlgebra

# =============================================================================
# Shared Utilities
# =============================================================================

"""Standardize matrix: subtract mean, divide by std."""
function _standardize(X::AbstractMatrix{T}) where {T}
    μ, σ = mean(X, dims=1), max.(std(X, dims=1), T(1e-10))
    (X .- μ) ./ σ
end

# =============================================================================
# Kalman Filter/Smoother for Dynamic Factor Model
# =============================================================================

"""
    _kalman_smoother_dfm(Y, Λ, A, Sigma_eta, Sigma_e, r, p) -> (a_smooth, P_smooth, Pt_smooth, loglik)

Kalman filter and smoother for state-space form of dynamic factor model.

State-space representation:
- Observation: Y_t = Z * α_t + ε_t, ε_t ~ N(0, H)
- State: α_t = T * α_{t-1} + η_t, η_t ~ N(0, Q)

Where:
- α_t = [F_t', F_{t-1}', ..., F_{t-p+1}']' (stacked factors)
- Z = [Λ, 0, ..., 0] (observation matrix)
- T = companion matrix for factor VAR
- Q = [Sigma_eta, 0; 0, 0] (state innovation covariance)
- H = Sigma_e (observation noise covariance)

Returns smoothed state estimates, covariances, and log-likelihood.
"""
function _kalman_smoother_dfm(Y::AbstractMatrix{T}, Λ::AbstractMatrix{T}, A::Vector{Matrix{T}},
    Sigma_eta::AbstractMatrix{T}, Sigma_e::AbstractMatrix{T}, r::Int, p::Int
) where {T<:AbstractFloat}

    T_obs, N = size(Y)
    state_dim = r * p

    # Build state-space matrices
    Z = zeros(T, N, state_dim); Z[:, 1:r] = Λ
    T_mat = zeros(T, state_dim, state_dim)
    for lag in 1:p
        T_mat[1:r, ((lag-1)*r+1):(lag*r)] = A[lag]
    end
    p > 1 && (T_mat[(r+1):state_dim, 1:(state_dim-r)] = I(state_dim - r))

    Q = zeros(T, state_dim, state_dim); Q[1:r, 1:r] = Sigma_eta
    H = Sigma_e

    # Initialize from unconditional distribution
    a0, P0 = zeros(T, state_dim), _compute_unconditional_covariance(T_mat, Q, state_dim)

    # Forward pass: Kalman filter
    a_filt = zeros(T, T_obs, state_dim)
    P_filt = zeros(T, T_obs, state_dim, state_dim)
    a_pred = zeros(T, T_obs, state_dim)
    P_pred = zeros(T, T_obs, state_dim, state_dim)
    loglik, a_t, P_t = zero(T), a0, P0

    for t in 1:T_obs
        # Prediction step
        a_pred[t, :] = T_mat * a_t
        P_pred[t, :, :] = T_mat * P_t * T_mat' + Q

        # Update step
        v_t = Y[t, :] - Z * a_pred[t, :]
        F_t = Symmetric(Z * P_pred[t, :, :] * Z' + H)
        F_inv = try inv(F_t) catch; pinv(F_t) end
        K_t = P_pred[t, :, :] * Z' * F_inv

        a_filt[t, :] = a_pred[t, :] + K_t * v_t
        P_filt[t, :, :] = (I(state_dim) - K_t * Z) * P_pred[t, :, :]

        # Log-likelihood contribution
        det_F = det(F_t)
        det_F > 0 && (loglik -= 0.5 * (N * log(2π) + log(det_F) + v_t' * F_inv * v_t))
        a_t, P_t = a_filt[t, :], P_filt[t, :, :]
    end

    # Backward pass: Kalman smoother
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

# =============================================================================
# Unconditional Covariance Computation
# =============================================================================

"""
    _compute_unconditional_covariance(T_mat, Q, state_dim; max_iter=1000, tol=1e-10)

Compute unconditional covariance of state vector by solving the discrete Lyapunov equation:
P = T * P * T' + Q

For stationary systems, iterates until convergence. For non-stationary systems,
returns a large diagonal matrix as fallback.
"""
function _compute_unconditional_covariance(T_mat::AbstractMatrix{T}, Q::AbstractMatrix{T},
    state_dim::Int; max_iter::Int=1000, tol::Float64=1e-10
) where {T<:AbstractFloat}
    # Check stationarity
    maximum(abs.(eigvals(T_mat))) >= 1.0 && return Matrix{T}(10.0 * I(state_dim))

    # Iterate Lyapunov equation
    P = Matrix{T}(I(state_dim))
    for _ in 1:max_iter
        P_new = T_mat * P * T_mat' + Q
        norm(P_new - P) < tol * norm(P) && return Symmetric(P_new)
        P = P_new
    end
    Symmetric(P)
end
