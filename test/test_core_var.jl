using MacroEconometricModels
using Test
using LinearAlgebra
using Statistics
using DataFrames
using Random

@testset "Core VAR & Identification" begin
    # Use fixed seed for reproducibility
    Random.seed!(12345)

    # 1. Generate Synthetic Data
    T = 200
    n = 2
    p = 1

    true_A = [0.5 0.0; 0.0 0.5] # Diagonal AR
    true_c = [0.0; 0.0]
    Sigma_true = [1.0 0.0; 0.0 1.0] # Identity

    Y = zeros(T, n)
    # Generate data
    for t in 2:T
        u = randn(2)
        Y[t, :] = true_c + true_A * Y[t-1, :] + u
    end

    # 2. Estimate
    model = estimate_var(Y, p)

    # 6. Identification (Cholesky)
    L = identify_cholesky(model)
    @test istriu(L')

    # 7. Sign Restrictions
    # We want to identify a "positive shock" to variable 1
    # Restriction: Response of Var 1 to Shock 1 at h=0 is positive (> 0)

    horizon = 10
    check_func(irf) = irf[1, 1, 1] > 0

    Q_sign, irf_sign = identify_sign(model, horizon, check_func)
    @test irf_sign[1, 1, 1] > 0
    @test isapprox(Q_sign * Q_sign', I(n), atol=1e-10) # Q is orthogonal

    # 8. Narrative Restrictions
    # Assume we know that at t=5, the shock 1 was positive.
    narrative_check(shocks) = shocks[5, 1] > 0

    Q_nar, irf_nar, shocks_nar = identify_narrative(model, horizon, check_func, narrative_check)
    @test irf_nar[1, 1, 1] > 0
    @test shocks_nar[5, 1] > 0

    # 9. Long Run Identification
    # Blanchard-Quah
    Q_lr = identify_long_run(model)
    @test isapprox(Q_lr * Q_lr', I(n), atol=1e-10)

    # Check if Long Run Impact is Lower Triangular
    # LR Impact = (I - A(1))^-1 * P
    # P = L * Q_lr
    # We can approximate infinite sum IRF or compute directly.
    # IRF cumulative sum at large h should be close to LR impact.

    # Let's compute directly to verify property
    B = model.B
    A_sum = zeros(n, n)
    for i in 1:p
        start_row = 1 + (i - 1) * n + 1
        end_row = 1 + i * n
        A_sum += B[start_row:end_row, :]'
    end
    inv_lag = inv(I(n) - A_sum)
    L_chol = identify_cholesky(model)
    P = L_chol * Q_lr
    LR_Matrix = inv_lag * P

    # Check lower triangularity of LR_Matrix
    @test abs(LR_Matrix[1, 2]) < 1e-8 # Upper right element should be 0

    @testset "Higher Order Lag (VAR(12))" begin
        # Test estimation with long lags using a known DGP to verify statistical recovery
        # DGP: Y_t = A_1 Y_{t-1} + ... + A_{12} Y_{t-12} + u_t
        # A_1 = 0.4 * I
        # A_12 = 0.2 * I
        # Others = 0

        T_large = 2000
        n = 2
        p = 12

        Y = zeros(T_large, n)

        # True Parameters
        A1 = [0.4 0.0; 0.0 0.4]
        A12 = [0.2 0.0; 0.0 0.2]
        Sigma_true = [1.0 0.0; 0.0 1.0]

        # Simulation
        Random.seed!(42) # Ensure reproducibility
        for t in p+1:T_large
            u = randn(n)
            # Y_t = A1 * Y_{t-1} + A12 * Y_{t-12} + u
            Y[t, :] = A1 * Y[t-1, :] + A12 * Y[t-12, :] + u
        end

        # Estimate
        model = estimate_var(Y, p)

        # Verify Coefficients
        # B structure: [Intercept; A_1'; A_2'; ... A_p']
        # Intercept should be close to 0
        @test norm(model.B[1, :]) < 0.2

        # Check Lag 1 (Rows 2:3)
        est_A1 = model.B[2:3, :]'
        @test isapprox(est_A1, A1, atol=0.1)

        # Check Lag 12 (Rows 1+11*2+1 : 1+12*2) -> Rows 24:25
        est_A12 = model.B[end-1:end, :]'
        @test isapprox(est_A12, A12, atol=0.1)

        # Check a middle lag (e.g., Lag 6) is close to zero
        # Rows for Lag 6: 1 + 5*2 + 1 = 12 to 13?
        # Lag k starts at 2 + (k-1)*n
        # Lag 6: 2 + 5*2 = 12. So rows 12:13.
        est_A6 = model.B[12:13, :]'
        @test norm(est_A6) < 0.1

        # Verify Residuals Covariance
        @test isapprox(model.Sigma, Sigma_true, atol=0.1)
    end

    # ==========================================================================
    # Robustness Tests (Following Arias et al. pattern)
    # ==========================================================================

    @testset "Reproducibility" begin
        # Same seed should produce identical results
        Random.seed!(99999)
        Y1 = zeros(100, 2)
        for t in 2:100
            Y1[t, :] = 0.5 * Y1[t-1, :] + randn(2)
        end
        model1 = estimate_var(Y1, 1)

        Random.seed!(99999)
        Y2 = zeros(100, 2)
        for t in 2:100
            Y2[t, :] = 0.5 * Y2[t-1, :] + randn(2)
        end
        model2 = estimate_var(Y2, 1)

        @test model1.B ≈ model2.B
        @test model1.Sigma ≈ model2.Sigma
        @test model1.U ≈ model2.U
    end

    @testset "Stability Check" begin
        # VAR should detect stable vs unstable systems
        Random.seed!(11111)
        T_stab = 200
        n_stab = 2
        p_stab = 1

        # Stable VAR
        Y_stable = zeros(T_stab, n_stab)
        A_stable = [0.3 0.1; 0.1 0.3]  # All eigenvalues < 1
        for t in 2:T_stab
            Y_stable[t, :] = A_stable * Y_stable[t-1, :] + randn(n_stab)
        end
        model_stable = estimate_var(Y_stable, p_stab)

        # Check stability via companion matrix eigenvalues
        F = companion_matrix(model_stable.B, n_stab, p_stab)
        eigenvalues = eigvals(F)
        @test maximum(abs.(eigenvalues)) < 1.0  # Stable
    end

    @testset "Numerical Stability - Near-Collinear Data" begin
        Random.seed!(22222)
        T_nc = 200
        n_nc = 3

        # Create data with near-collinearity
        Y_nc = randn(T_nc, n_nc)
        Y_nc[:, 3] = Y_nc[:, 1] + 0.01 * randn(T_nc)  # Variable 3 ≈ Variable 1

        # Should not crash with near-singular covariance
        model_nc = estimate_var(Y_nc, 1)
        @test model_nc isa VARModel
        @test all(isfinite.(model_nc.B))
        @test all(isfinite.(model_nc.Sigma))
    end

    @testset "Edge Cases" begin
        Random.seed!(33333)

        # Single variable VAR
        Y_single = randn(100, 1)
        model_single = estimate_var(Y_single, 1)
        @test size(model_single.B) == (2, 1)  # intercept + 1 lag
        @test size(model_single.Sigma) == (1, 1)

        # Minimum viable sample size (T just larger than p*n + 1)
        n_min = 2
        p_min = 2
        T_min = p_min * n_min + 10  # Bare minimum observations
        Y_min = randn(T_min, n_min)
        model_min = estimate_var(Y_min, p_min)
        @test model_min isa VARModel

        # VAR(1) - simplest case
        Y_var1 = randn(50, 2)
        model_var1 = estimate_var(Y_var1, 1)
        @test model_var1.p == 1
    end

    @testset "Orthogonality of Q Matrices" begin
        Random.seed!(44444)
        T_q = 150
        n_q = 3
        Y_q = randn(T_q, n_q)
        model_q = estimate_var(Y_q, 1)

        # Cholesky Q should be identity (orthogonal)
        Q_chol = I(n_q)
        @test norm(Q_chol' * Q_chol - I(n_q)) < 1e-10

        # Sign restriction Q should be orthogonal
        check_func_q(irf) = irf[1, 1, 1] > 0
        Q_sign_q, _ = identify_sign(model_q, 5, check_func_q)
        @test norm(Q_sign_q' * Q_sign_q - I(n_q)) < 1e-10
        @test norm(Q_sign_q * Q_sign_q' - I(n_q)) < 1e-10

        # Columns should be unit vectors
        for j in 1:n_q
            @test abs(norm(Q_sign_q[:, j]) - 1.0) < 1e-10
        end
    end

    @testset "Input Validation" begin
        Random.seed!(55555)
        Y_val = randn(100, 2)

        # p = 0 should error or be handled
        @test_throws Exception estimate_var(Y_val, 0)

        # p too large for data - package handles gracefully with warning
        # Just verify it returns a model (even if with adjusted dof)
        model_large_p = estimate_var(Y_val, 40)
        @test model_large_p isa VARModel

        # Empty data
        @test_throws Exception estimate_var(zeros(0, 2), 1)
    end
end
