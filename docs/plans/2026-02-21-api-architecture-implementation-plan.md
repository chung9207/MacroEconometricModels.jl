# API Consistency & Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Standardize the MacroEconometricModels.jl API (v0.2.5) and restructure architecture (v0.3.0).

**Architecture:** Two-phase layered approach. Phase A makes 5 non-breaking API consistency fixes. Phase B makes 4 architectural improvements (covariance registry, Kalman consolidation, forecast accessors, package extensions). Each task is TDD with commit at the end.

**Tech Stack:** Julia 1.10+, StatsAPI, PrettyTables, DataFrames, FFTW, Optim

---

## Phase A: API Consistency (v0.2.5)

### Task 1: Standardize `conf_level` parameter type

**Files:**
- Modify: `src/arima/forecast.jl:161,171,181,193`
- Modify: `src/arch/forecast.jl:37`
- Modify: `src/garch/forecast.jl:35,82,128`
- Modify: `src/sv/forecast.jl:36`
- Test: `test/arima/test_arima.jl`, `test/volatility/test_volatility.jl`

**Step 1: Write a test verifying conf_level accepts Real (not just T)**

In `test/arima/test_arima.jl`, find the ARIMA forecast test section and add:

```julia
@testset "conf_level accepts Real" begin
    Random.seed!(42)
    y = cumsum(randn(100))
    m = estimate_ar(y, 2)
    # Should accept Int, Float32, Float64 without error
    fc1 = forecast(m, 5; conf_level=0.9)   # Float64
    @test fc1.conf_level ≈ 0.9
    fc2 = forecast(m, 5; conf_level=Float32(0.9))   # Float32
    @test fc2.conf_level ≈ Float32(0.9) atol=1e-6
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -A 5 "conf_level accepts Real"`

Expected: FAIL — `MethodError: no method matching forecast(::ARModel{Float64}, ::Int64; conf_level::Float32)` because `conf_level::T` requires exact type match.

**Step 3: Fix all 9 functions**

In `src/arima/forecast.jl`, change all 4 signatures:

Line 161: `conf_level::T=T(0.95)` → `conf_level::Real=0.95`
Line 171: `conf_level::T=T(0.95)` → `conf_level::Real=0.95`
Line 181: `conf_level::T=T(0.95)` → `conf_level::Real=0.95`
Line 193: `conf_level::T=T(0.95)` → `conf_level::Real=0.95`

In each function body, where `conf_level` is used in computation, cast: `T(conf_level)`.

In `src/arch/forecast.jl`:
Line 37: `conf_level::T=T(0.95)` → `conf_level::Real=0.95`

In `src/garch/forecast.jl`:
Line 35: `conf_level::T=T(0.95)` → `conf_level::Real=0.95`
Line 82: `conf_level::T=T(0.95)` → `conf_level::Real=0.95`
Line 128: `conf_level::T=T(0.95)` → `conf_level::Real=0.95`

In `src/sv/forecast.jl`:
Line 36: `conf_level::T=T(0.95)` → `conf_level::Real=0.95`

In each function body, add `conf_level = T(conf_level)` as the first line (to maintain type stability for downstream computation).

**Step 4: Run tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: All tests pass (existing tests still work because Float64 <: Real).

**Step 5: Commit**

```bash
git add src/arima/forecast.jl src/arch/forecast.jl src/garch/forecast.jl src/sv/forecast.jl test/arima/test_arima.jl
git commit -m "Standardize conf_level::Real across all forecast functions"
```

---

### Task 2: Rename `LPForecast.forecasts` to `.forecast`

**Files:**
- Modify: `src/lp/types.jl:210-256` (struct definition + inner constructor)
- Modify: `src/lp/forecast.jl` (constructor call sites)
- Modify: `src/summary_display.jl:426,435` (field access)
- Modify: `src/plotting/forecast.jl:334,342` (field access)
- Test: `test/lp/test_lp_forecast.jl:46,56,67-68,80-81,83,111,123,134-135,140,213,232`

**Step 1: Update test file first**

In `test/lp/test_lp_forecast.jl`, replace all `.forecasts` with `.forecast`:

Lines to change: 46, 56, 67, 68, 80, 83, 111 (2 occurrences), 123 (2 occurrences), 134, 135, 140, 213 (2 occurrences), 232.

Use search-and-replace: `fc.forecasts` → `fc.forecast`, `fc_zero.forecasts` → `fc_zero.forecast`, `fc_nonzero.forecasts` → `fc_nonzero.forecast`, `fc1.forecasts` → `fc1.forecast`, `fc2.forecasts` → `fc2.forecast`, `fc_90.forecasts` → `fc_90.forecast`, `fc_99.forecasts` → `fc_99.forecast`, `fc_j.forecasts` → `fc_j.forecast`.

**Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'include("test/lp/test_lp_forecast.jl")'`

Expected: FAIL — `type LPForecast has no field forecast`

**Step 3: Rename field in struct and all source references**

In `src/lp/types.jl`, line ~211: rename field `forecasts::Matrix{T}` → `forecast::Matrix{T}`.
Also update inner constructor parameter name and all references within the struct.

In `src/lp/forecast.jl`: find all `LPForecast(` constructor calls and ensure positional argument order still matches (field rename only, order unchanged — `forecast` is still the first field).

In `src/summary_display.jl`:
- Line 426: `fc.forecasts[h, j]` → `fc.forecast[h, j]`
- Line 435: `fc.forecasts[h, j]` → `fc.forecast[h, j]`

In `src/plotting/forecast.jl`:
- Line 334: `size(fc.forecasts)` → `size(fc.forecast)`
- Line 342: `fc.forecasts[:, vi]` → `fc.forecast[:, vi]`

**Step 4: Run full test suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: All tests pass.

**Step 5: Commit**

```bash
git add src/lp/types.jl src/lp/forecast.jl src/summary_display.jl src/plotting/forecast.jl test/lp/test_lp_forecast.jl
git commit -m "Rename LPForecast.forecasts to .forecast for consistency"
```

---

### Task 3: Add `conf_level` field to `VECMForecast`

**Files:**
- Modify: `src/vecm/types.jl:135-143` (struct definition)
- Modify: `src/vecm/forecast.jl:80` (constructor call)
- Test: `test/vecm/test_vecm.jl`

**Step 1: Write a test checking VECMForecast has conf_level**

In the VECM forecast test section of `test/vecm/test_vecm.jl`, add:

```julia
@testset "VECMForecast has conf_level" begin
    fc = forecast(vecm, 10; conf_level=0.90)
    @test hasproperty(fc, :conf_level)
    @test fc.conf_level ≈ 0.90
end
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `type VECMForecast has no field conf_level`

**Step 3: Add field to struct and update constructor**

In `src/vecm/types.jl`, add `conf_level::T` field to `VECMForecast`:

```julia
struct VECMForecast{T<:AbstractFloat} <: AbstractForecastResult{T}
    levels::Matrix{T}
    differences::Matrix{T}
    ci_lower::Matrix{T}
    ci_upper::Matrix{T}
    horizon::Int
    ci_method::Symbol
    conf_level::T        # NEW FIELD
    varnames::Vector{String}
end
```

In `src/vecm/forecast.jl`, update the constructor call (line ~80) to pass `conf_level`:

```julia
VECMForecast{T}(levels, differences, ci_lower, ci_upper, h, ci_method, T(conf_level), vecm.varnames)
```

Also ensure the `forecast()` function signature accepts `conf_level` kwarg (it already does at line 46: `conf_level::Real=0.95`).

**Step 4: Run tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

**Step 5: Commit**

```bash
git add src/vecm/types.jl src/vecm/forecast.jl test/vecm/test_vecm.jl
git commit -m "Add conf_level field to VECMForecast for consistency"
```

---

### Task 4: Extensible covariance estimator registry

**Files:**
- Modify: `src/core/covariance.jl` (add registry at end of file)
- Modify: `src/lp/core.jl:40-55` (replace if-chain with registry lookup)
- Modify: `src/MacroEconometricModels.jl` (export `register_cov_estimator!`)
- Test: `test/core/test_covariance.jl` or new section in `test/lp/test_lp.jl`

**Step 1: Write a test for the registry**

Add to `test/lp/test_lp.jl`:

```julia
@testset "Covariance estimator registry" begin
    # Built-in estimators work
    est = MacroEconometricModels.create_cov_estimator(:newey_west, Float64)
    @test est isa NeweyWestEstimator
    est = MacroEconometricModels.create_cov_estimator(:white, Float64)
    @test est isa WhiteEstimator
    est = MacroEconometricModels.create_cov_estimator(:driscoll_kraay, Float64)
    @test est isa DriscollKraayEstimator

    # Unknown estimator throws informative error
    @test_throws ArgumentError MacroEconometricModels.create_cov_estimator(:unknown, Float64)

    # Custom estimator can be registered
    struct TestCovEstimator <: AbstractCovarianceEstimator end
    register_cov_estimator!(:test_cov, TestCovEstimator)
    @test haskey(MacroEconometricModels._COV_REGISTRY, :test_cov)

    # Clean up
    delete!(MacroEconometricModels._COV_REGISTRY, :test_cov)
end
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `_COV_REGISTRY` not defined.

**Step 3: Implement the registry**

At the end of `src/core/covariance.jl`, add:

```julia
# =============================================================================
# Covariance Estimator Registry
# =============================================================================

"""
    _COV_REGISTRY

Registry mapping Symbol names to covariance estimator types.
Use `register_cov_estimator!` to add custom estimators.
"""
const _COV_REGISTRY = Dict{Symbol, Type}(
    :newey_west => NeweyWestEstimator,
    :white => WhiteEstimator,
    :driscoll_kraay => DriscollKraayEstimator,
)

"""
    register_cov_estimator!(name::Symbol, T::Type{<:AbstractCovarianceEstimator})

Register a custom covariance estimator type for use in LP and other estimators.

# Example
```julia
struct MyCovEstimator <: AbstractCovarianceEstimator end
register_cov_estimator!(:my_cov, MyCovEstimator)
# Now usable: estimate_lp(Y, 1, 20; cov_type=:my_cov)
```
"""
function register_cov_estimator!(name::Symbol, ::Type{T}) where {T<:AbstractCovarianceEstimator}
    _COV_REGISTRY[name] = T
end
```

Replace the if-chain in `src/lp/core.jl:40-55` with:

```julia
"""
    create_cov_estimator(cov_type::Symbol, ::Type{T}; bandwidth::Int=0) where T

Create covariance estimator from symbol specification using the global registry.
"""
function create_cov_estimator(cov_type::Symbol, ::Type{T}; bandwidth::Int=0) where {T<:AbstractFloat}
    EstType = get(_COV_REGISTRY, cov_type, nothing)
    if EstType === nothing
        valid = join(sort(collect(keys(_COV_REGISTRY))), ", ")
        throw(ArgumentError("Unknown cov_type :$cov_type. Available: $valid"))
    end
    if EstType <: NeweyWestEstimator
        NeweyWestEstimator{T}(bandwidth, :bartlett, false)
    elseif EstType <: DriscollKraayEstimator
        DriscollKraayEstimator{T}(bandwidth, :bartlett)
    else
        EstType()
    end
end
```

Add to exports in `src/MacroEconometricModels.jl`:

```julia
export register_cov_estimator!
```

**Step 4: Run full test suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: All tests pass (existing LP tests use `:newey_west`/`:white` which still work).

**Step 5: Commit**

```bash
git add src/core/covariance.jl src/lp/core.jl src/MacroEconometricModels.jl test/lp/test_lp.jl
git commit -m "Add extensible covariance estimator registry"
```

---

### Task 5: Add forecast accessor functions

**Files:**
- Modify: `src/core/types.jl:97-104` (add accessor functions after AbstractForecastResult)
- Modify: `src/MacroEconometricModels.jl` (export accessors)
- Test: new section in `test/core/test_coverage_gaps.jl` or `test/core/test_utils.jl`

**Step 1: Write tests for forecast accessors**

Add to `test/core/test_utils.jl`:

```julia
@testset "Forecast accessor functions" begin
    Random.seed!(42)
    Y = randn(100, 3)

    # VARForecast
    m = estimate_var(Y, 2)
    fc = forecast(m, 5)
    @test point_forecast(fc) === fc.forecast
    @test lower_bound(fc) === fc.ci_lower
    @test upper_bound(fc) === fc.ci_upper
    @test forecast_horizon(fc) == fc.horizon

    # ARIMAForecast
    y = cumsum(randn(100))
    am = estimate_ar(y, 2)
    afc = forecast(am, 5)
    @test point_forecast(afc) === afc.forecast
    @test lower_bound(afc) === afc.ci_lower
    @test upper_bound(afc) === afc.ci_upper
    @test forecast_horizon(afc) == afc.horizon
end
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `point_forecast` not defined.

**Step 3: Implement accessors**

In `src/core/types.jl`, after the `AbstractForecastResult` definition (line ~104), add:

```julia
# =============================================================================
# Forecast Accessor Functions
# =============================================================================

"""
    point_forecast(f::AbstractForecastResult)

Return the point forecast values (Vector or Matrix).
"""
point_forecast(f::AbstractForecastResult) = f.forecast

"""
    lower_bound(f::AbstractForecastResult)

Return the lower confidence interval bound.
"""
lower_bound(f::AbstractForecastResult) = f.ci_lower

"""
    upper_bound(f::AbstractForecastResult)

Return the upper confidence interval bound.
"""
upper_bound(f::AbstractForecastResult) = f.ci_upper

"""
    forecast_horizon(f::AbstractForecastResult)

Return the forecast horizon (number of steps ahead).
"""
forecast_horizon(f::AbstractForecastResult) = f.horizon
```

Note: `FactorForecast` has different field names (`.observables`, `.observables_lower`, `.observables_upper`). Add overrides AFTER `FactorForecast` is defined. In `src/factor/kalman.jl`, after the `FactorForecast` struct definition (~line 53), add:

```julia
point_forecast(f::FactorForecast) = f.observables
lower_bound(f::FactorForecast) = f.observables_lower
upper_bound(f::FactorForecast) = f.observables_upper
```

For `VECMForecast` which uses `.levels` instead of `.forecast`, add override in `src/vecm/types.jl` after the struct:

```julia
point_forecast(f::VECMForecast) = f.levels
```

Export in `src/MacroEconometricModels.jl`:

```julia
export point_forecast, lower_bound, upper_bound, forecast_horizon
```

**Step 4: Run tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

**Step 5: Commit**

```bash
git add src/core/types.jl src/factor/kalman.jl src/vecm/types.jl src/MacroEconometricModels.jl test/core/test_utils.jl
git commit -m "Add forecast accessor functions for uniform interface"
```

---

### Task 6: Standardize `report()` vs `show()` convention

**Files:**
- Audit: `src/summary.jl`, `src/summary_display.jl`
- Modify: Files where convention is violated
- Test: Existing tests should still pass

**Step 1: Audit current behavior**

Check each model type's `report()` and `show()`:
- `report()` should be the FULL output (coefficient tables, diagnostics)
- `show()` should be COMPACT (type, dimensions, key stats)

Read `src/summary.jl` to see all `report()` definitions.
Read `src/summary_display.jl` to see all `show()` definitions.

Identify violations: models where `report()` simply calls `show()` (should be the opposite — `show()` is compact, `report()` is full).

**Step 2: Fix violations**

For each violation, ensure:
- `show(io, model)`: print type name, dimensions, 2-3 key statistics (AIC/BIC, nobs)
- `report(model)`: print full publication output (delegates to `show(stdout, model)` is OK if `show` is already full, but then needs a separate compact method for REPL display)

The convention should be:
```julia
# REPL display (compact)
Base.show(io::IO, ::MIME"text/plain", m::ModelType) = _show_compact(io, m)
# Full report
report(m::ModelType) = _show_full(stdout, m)
```

**Step 3: Run tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

**Step 4: Commit**

```bash
git add src/summary.jl src/summary_display.jl
git commit -m "Standardize report() vs show() convention across model types"
```

---

### Task 7: Version bump to v0.2.5 and full test run

**Files:**
- Modify: `Project.toml` (version field)

**Step 1: Bump version**

In `Project.toml`, change `version = "0.2.4"` to `version = "0.2.5"`.

**Step 2: Run full test suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: All 7200+ tests pass, 2 broken (pre-existing).

**Step 3: Commit**

```bash
git add Project.toml
git commit -m "Bump version to v0.2.5"
```

---

## Phase B: Architecture (v0.3.0)

### Task 8: Consolidate Kalman filter core operations

**Files:**
- Create: `src/core/kalman.jl`
- Modify: `src/factor/kalman.jl` (use shared core)
- Modify: `src/arima/kalman.jl` (use shared core)
- Modify: `src/nowcast/kalman_missing.jl` (use shared core)
- Modify: `src/MacroEconometricModels.jl` (include new file)
- Test: `test/factor/test_factormodel.jl`, `test/arima/test_arima.jl`, `test/nowcast/test_nowcast.jl`

**Step 1: Write equivalence tests**

Create `test/core/test_kalman.jl`:

```julia
@testset "Core Kalman operations" begin
    Random.seed!(42)
    n, m = 4, 2

    # Setup state-space matrices
    F = 0.9 * I(n) |> Matrix{Float64}
    H = randn(m, n)
    Q = 0.1 * I(n) |> Matrix{Float64}
    R = 0.05 * I(m) |> Matrix{Float64}
    x0 = zeros(n)
    P0 = I(n) |> Matrix{Float64}

    # Generate data
    T_obs = 50
    states = zeros(T_obs, n)
    obs = zeros(T_obs, m)
    x = copy(x0)
    for t in 1:T_obs
        x = F * x + cholesky(Q).L * randn(n)
        states[t, :] = x
        obs[t, :] = H * x + cholesky(R).L * randn(m)
    end

    @testset "predict step" begin
        x_pred, P_pred = MacroEconometricModels._kalman_predict(x0, P0, F, Q)
        @test size(x_pred) == (n,)
        @test size(P_pred) == (n, n)
        @test x_pred ≈ F * x0
        @test P_pred ≈ F * P0 * F' + Q
        @test issymmetric(P_pred) || P_pred ≈ P_pred'
    end

    @testset "update step" begin
        x_pred, P_pred = MacroEconometricModels._kalman_predict(x0, P0, F, Q)
        y = obs[1, :]
        x_upd, P_upd, v, S, K = MacroEconometricModels._kalman_update(x_pred, P_pred, y, H, R)
        @test size(x_upd) == (n,)
        @test size(P_upd) == (n, n)
        @test size(v) == (m,)
        @test size(S) == (m, m)
        @test size(K) == (n, m)
        @test v ≈ y - H * x_pred
    end

    @testset "full filter-smoother cycle" begin
        # Run filter
        x_filt, P_filt, x_pred_all, P_pred_all, ll = MacroEconometricModels._kalman_filter(
            obs, F, H, Q, R, x0, P0)
        @test size(x_filt) == (T_obs, n)
        @test isfinite(ll)

        # Run smoother
        x_smooth, P_smooth = MacroEconometricModels._kalman_smoother(
            x_filt, P_filt, x_pred_all, P_pred_all, F)
        @test size(x_smooth) == (T_obs, n)

        # Smoother should reduce variance
        for t in 1:T_obs
            @test tr(P_smooth[t]) <= tr(P_filt[t]) + 1e-10
        end
    end
end
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `_kalman_predict`, `_kalman_filter`, `_kalman_smoother` not defined in core.

**Step 3: Create `src/core/kalman.jl`**

Extract shared predict/update/smoother operations. This file should contain:
- `_kalman_predict(x, P, F, Q)` — state prediction
- `_kalman_update(x_pred, P_pred, y, H, R)` — measurement update
- `_kalman_filter(Y, F, H, Q, R, x0, P0)` — full forward pass
- `_kalman_smoother(x_filt, P_filt, x_pred, P_pred, F)` — RTS smoother

Then update `factor/kalman.jl`, `arima/kalman.jl`, `nowcast/kalman_missing.jl` to call these shared operations instead of reimplementing them.

Add to `src/MacroEconometricModels.jl` include order — after `core/covariance.jl`, before `factor/`:

```julia
include("core/kalman.jl")
```

**Step 4: Verify numerical equivalence**

Run the existing test suites that exercise each Kalman implementation:

```bash
julia --project=. -e 'include("test/factor/test_factormodel.jl")'
julia --project=. -e 'include("test/arima/test_arima.jl")'
julia --project=. -e 'include("test/nowcast/test_nowcast.jl")'
```

All must pass with identical numerical results.

**Step 5: Commit**

```bash
git add src/core/kalman.jl src/factor/kalman.jl src/arima/kalman.jl src/nowcast/kalman_missing.jl src/MacroEconometricModels.jl test/core/test_kalman.jl
git commit -m "Consolidate Kalman filter core operations into src/core/kalman.jl"
```

---

### Task 9: Display code consolidation

**Files:**
- Modify: `src/summary_display.jl` (consolidate 40 methods)
- Modify: `src/summary_tables.jl` (if shared helpers move here)
- Test: `test/core/test_display_backends.jl`, `test/core/test_summary.jl`

**Step 1: Identify the common pattern**

All `print_table` methods follow this pattern:
1. Call `table(obj, args...)` to get a DataFrame/Matrix
2. Define column labels
3. Call `_pretty_table(io, data; title=..., column_labels=..., alignment=...)`

All `Base.show` methods follow:
1. Build spec data (key-value pairs)
2. Call `_pretty_table` for spec table
3. Build main data matrix
4. Call `_pretty_table` for each sub-table

**Step 2: Extract shared helpers**

Add to `src/summary_display.jl` near the top:

```julia
"""
    _show_spec_table(io, title, pairs)

Display a specification table (key-value pairs) used in most show() methods.
"""
function _show_spec_table(io::IO, title::String, pairs::Vector{<:Pair})
    data = [first(p) last(p) for p in pairs]
    _pretty_table(io, data; title=title, column_labels=["", ""], alignment=[:l, :r])
end
```

**Step 3: Refactor each `Base.show` method**

Replace repeated spec-table construction (e.g., `spec_data = ["Variables" n_vars; ...]`) with `_show_spec_table(io, title, pairs)` calls.

Example — `Base.show(io::IO, irf::ImpulseResponse{T})` becomes:

```julia
function Base.show(io::IO, irf::ImpulseResponse{T}) where {T}
    ci_str = irf.ci_type == :none ? "None" : string(irf.ci_type)
    _show_spec_table(io, "Impulse Response Functions", [
        "Variables" => length(irf.variables),
        "Shocks" => length(irf.shocks),
        "Horizon" => irf.horizon,
        "CI" => ci_str,
    ])
    # ... rest of display logic (per-shock tables stay as-is)
end
```

**Step 4: Run tests to verify display output unchanged**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

Focus on display/summary tests:
```bash
julia --project=. -e 'include("test/core/test_display_backends.jl")'
julia --project=. -e 'include("test/core/test_summary.jl")'
```

**Step 5: Commit**

```bash
git add src/summary_display.jl
git commit -m "Consolidate display helpers to reduce summary_display.jl duplication"
```

---

### Task 10: Package extensions for optional dependencies

**Files:**
- Modify: `Project.toml` (add weakdeps, extensions sections)
- Create: `ext/MacroEconometricModelsFFTWExt.jl`
- Create: `ext/MacroEconometricModelsDataFramesExt.jl`
- Create: `ext/MacroEconometricModelsPrettyTablesExt.jl`
- Modify: `src/MacroEconometricModels.jl` (conditional includes)
- Modify: `src/factor/generalized.jl` (FFTW gated)
- Modify: `src/data/examples.jl` (DataFrames gated)
- Modify: `src/summary_display.jl` (PrettyTables gated)
- Test: Full test suite (with and without optional deps loaded)

**CAUTION**: This is the highest-risk task. The approach is:
1. Move FFTW usage to an extension (smallest scope — only `estimate_gdfm`)
2. If successful, move DataFrames
3. If successful, move PrettyTables (widest scope — affects all display)

**Step 1: Start with FFTW extension only**

In `Project.toml`, add:

```toml
[weakdeps]
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"

[extensions]
MacroEconometricModelsFFTWExt = "FFTW"
```

Move FFTW from `[deps]` to `[weakdeps]`.

Create `ext/MacroEconometricModelsFFTWExt.jl`:

```julia
module MacroEconometricModelsFFTWExt

using MacroEconometricModels
using FFTW

# Move GDFM spectral computation here
# Functions that use fft/ifft from FFTW

end
```

In `src/factor/generalized.jl`, gate FFTW calls behind `@static if isdefined(Main, :FFTW)` or use the extension mechanism.

**Step 2: Test with FFTW loaded**

```bash
julia --project=. -e 'using FFTW; using MacroEconometricModels; m = estimate_gdfm(randn(100,5), 2); println("OK")'
```

**Step 3: Test without FFTW loaded**

```bash
julia --project=. -e 'using MacroEconometricModels; println(typeof(estimate_var(randn(100,3), 2)))'
```

Should work without FFTW. `estimate_gdfm` should throw an informative error when FFTW is not loaded.

**Step 4: Run full test suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

Note: Tests that use GDFM will need `using FFTW` in the test file.

**Step 5: Commit FFTW extension**

```bash
git add Project.toml ext/MacroEconometricModelsFFTWExt.jl src/factor/generalized.jl
git commit -m "Move FFTW to package extension (optional dependency)"
```

**Step 6: Repeat for DataFrames and PrettyTables**

Follow the same pattern. DataFrames gates: `load_example()`, `table()`, `to_dataframe()`. PrettyTables gates: `_pretty_table()`, `print_table()`.

These are more complex and may require stub functions that throw helpful errors when the extension isn't loaded.

**Step 7: Commit each extension separately**

---

### Task 11: Version bump to v0.3.0 and final test run

**Files:**
- Modify: `Project.toml`

**Step 1: Bump version**

`version = "0.2.5"` → `version = "0.3.0"`

**Step 2: Run full test suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: All tests pass.

**Step 3: Commit**

```bash
git add Project.toml
git commit -m "Bump version to v0.3.0"
```

---

## Verification Checklist

After all tasks:

- [ ] `julia --project=. -e 'using Pkg; Pkg.test()'` — all tests pass
- [ ] `conf_level::Real` in all forecast functions (grep: `grep -rn "conf_level::T" src/` returns 0 results)
- [ ] `LPForecast.forecast` (not `.forecasts`) — grep: `grep -rn "\.forecasts" src/` returns 0 results
- [ ] `VECMForecast` has `conf_level` field
- [ ] `register_cov_estimator!` exported and working
- [ ] `point_forecast()`, `lower_bound()`, `upper_bound()`, `forecast_horizon()` exported
- [ ] `_kalman_predict`, `_kalman_update` in `src/core/kalman.jl`
- [ ] `summary_display.jl` line count < 500
- [ ] FFTW is a `[weakdeps]` not `[deps]`
- [ ] `julia --project=. -e 'using MacroEconometricModels; estimate_var(randn(100,3), 2)'` works without FFTW
