# API Consistency & Architecture Improvement Design

**Date**: 2026-02-21
**Version target**: Phase A → v0.2.5 (non-breaking), Phase B → v0.3.0 (breaking)
**Scope**: Code quality, API standardization, architectural improvements

---

## Motivation

Audit of the MacroEconometricModels.jl codebase (117 files, ~38k LOC) identified 22+ API inconsistencies, ~800 LOC of display duplication, unnecessary data storage in model types, non-extensible dispatch patterns, and heavy optional dependencies loaded for all users. This design addresses the highest-impact issues in two phases.

---

## Phase A: API Consistency (v0.2.5, non-breaking)

### A1. Standardize `conf_level` parameter type

**Problem**: ARIMA/volatility use `conf_level::T=T(0.95)`, LP/Factor/VECM/BVAR use `conf_level::Real=0.95`.

**Solution**: Standardize to `conf_level::Real=0.95` everywhere.

**Files to change** (9 functions):
- `src/arima/forecast.jl` — 4 functions (ARModel, MAModel, ARMAModel, ARIMAModel)
- `src/arch/forecast.jl` — 1 function (ARCHModel)
- `src/garch/forecast.jl` — 3 functions (GARCHModel, EGARCHModel, GJRGARCHModel)
- `src/sv/forecast.jl` — 1 function (SVModel)

**Change pattern**: `conf_level::T=T(0.95)` → `conf_level::Real=0.95`, then `T(conf_level)` at point of use.

### A2. Fix forecast result type inconsistencies

**Problem**: Field naming varies across forecast types.

**Changes**:
1. `LPForecast.forecasts` → `.forecast` (rename field)
2. Add `conf_level::T` field to `VECMForecast`
3. Update all constructors and call sites

**Files to change**:
- `src/lp/types.jl` — rename field
- `src/lp/forecast.jl` — update constructor calls
- `src/vecm/types.jl` — add field
- `src/vecm/forecast.jl` — pass conf_level to constructor
- `src/summary_display.jl` — update field references
- `src/plotting/forecast.jl` — update field references
- Tests: `test/lp/test_lp_forecast.jl`, `test/vecm/test_vecm.jl`

### A3. Consolidate display code

**Problem**: 40 methods in `summary_display.jl` (977 lines) with repeated formatting logic.

**Solution**: Extract shared helper:
```julia
function _render_table_display(io::IO, title::String, tbl::DataFrame;
                               footnotes::Vector{String}=String[])
    println(io, "\n", title)
    println(io, "─"^min(78, length(title)+4))
    print_table(io, tbl)
    for note in footnotes
        println(io, note)
    end
end
```

Each type's `show()` becomes 3-5 lines calling this helper.

**Target**: Reduce `summary_display.jl` from ~977 lines to ~400 lines.

### A4. Add missing docstrings

**Functions needing docstrings**:
- `estimate_gdfm` — generalized dynamic factor model
- `propensity_irf`, `propensity_diagnostics` — LP propensity score functions
- `pvar_hansen_j`, `pvar_mmsc`, `pvar_lag_selection` — PVAR specification tests
- `identify_fastica`, `identify_jade`, `identify_sobi`, `identify_dcov`, `identify_hsic` — ICA identification
- `identify_student_t`, `identify_mixture_normal`, `identify_pml`, `identify_skew_normal` — ML identification

**Format**: Standard `# Arguments` / `# Returns` / `# Example` / `# References` sections.

### A5. Standardize `report()` vs `show()` convention

**Convention**:
- `Base.show(io, model)` = compact summary (type, dimensions, key stats). 5-10 lines max.
- `report(model)` = full publication-quality output (coefficient tables, diagnostics, residual stats). Unbounded length.

**Audit and fix** models where this convention is violated (some have `report()` calling `show()`, which inverts the relationship).

---

## Phase B: Architecture (v0.3.0, breaking)

### B1. Package extensions for optional dependencies

**Problem**: FFTW (~8MB), DataFrames (~5MB), PrettyTables (~5MB) loaded for all users. Load time ~30-60s.

**Solution**: Julia 1.9+ package extensions.

**Project.toml changes**:
```toml
[weakdeps]
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"

[extensions]
MacroEconometricModelsFFTWExt = "FFTW"
MacroEconometricModelsDataFramesExt = "DataFrames"
MacroEconometricModelsPrettyTablesExt = "PrettyTables"
```

**Extension modules** (new `ext/` directory):
- `ext/MacroEconometricModelsFFTWExt.jl` — gates `estimate_gdfm()` spectral computation
- `ext/MacroEconometricModelsDataFramesExt.jl` — gates `load_example()`, `table()`, DataFrame conversion
- `ext/MacroEconometricModelsPrettyTablesExt.jl` — gates `print_table()` formatted output

**Fallback**: When PrettyTables is absent, `print_table()` uses `println`-based plain text. When DataFrames is absent, `table()` throws `ExtensionError("DataFrames required")`.

**Expected improvement**: Load time from ~60s → ~15s for VAR-only users.

### B2. Eliminate unnecessary data storage

**Problem**: VARModel, LPModel, VECMModel store both raw data (`Y`) and residuals (`U`). BVARPosterior stores `data` (never used downstream).

**Solution**:

**VARModel**: Remove `Y` field. Keep `U` (needed for `report()` residual diagnostics and Granger causality). Add `_T_eff::Int` field for effective sample size. Users needing raw data should keep their own reference.

**VECMModel**: Same as VARModel — remove `Y`, keep `U`.

**BVARPosterior**: Remove `data` field. Add `_T_eff::Int` and `_n::Int` for dimensions.

**LPModel**: Keep both `Y` (needed for re-estimation at different horizons) and `residuals` (per-horizon, needed for inference). No change.

**Migration**:
- Deprecate `model.Y` access with `@deprecate` warning in v0.2.5
- Remove field in v0.3.0
- Update all internal code that accesses `model.Y` to use stored dimensions or `residuals()`

### B3. Consolidate Kalman filters

**Problem**: Three independent implementations: `src/factor/kalman.jl`, `src/arima/kalman.jl`, `src/nowcast/kalman_missing.jl`.

**Solution**: Create `src/core/kalman.jl` with shared core:

```julia
# Core Kalman operations (shared)
function _kalman_predict(x, P, F, Q)
    x_pred = F * x
    P_pred = F * P * F' + Q
    return x_pred, P_pred
end

function _kalman_update(x_pred, P_pred, y, H, R)
    v = y - H * x_pred
    S = H * P_pred * H' + R
    K = P_pred * H' * robust_inv(S)
    x_upd = x_pred + K * v
    P_upd = (I - K * H) * P_pred
    return x_upd, P_upd, v, S, K
end

function _kalman_smooth(x_filt, P_filt, x_pred, P_pred, F)
    J = P_filt * F' * robust_inv(P_pred)
    x_smooth = x_filt + J * (x_smooth_next - x_pred_next)
    P_smooth = P_filt + J * (P_smooth_next - P_pred) * J'
    return x_smooth, P_smooth, J
end
```

Module-specific wrappers handle state-space formulation:
- Factor: `_factor_kalman_filter(F, Lambda, R, ...)` sets up H, F, Q from factor model
- ARIMA: `_arima_state_space(phi, theta, ...)` constructs companion form
- Nowcast: `_kalman_missing_data(y, W, ...)` handles NaN observations

**Include order**: `src/core/kalman.jl` included after `core/utils.jl`, before `factor/`.

### B4. Extensible covariance estimator registry

**Problem**: `create_cov_estimator()` uses if-chain, blocking user extensions.

**Solution**:
```julia
# src/core/covariance.jl (add at top)
const _COV_REGISTRY = Dict{Symbol, Type}()

function register_cov_estimator!(name::Symbol, T::Type{<:AbstractCovarianceEstimator})
    _COV_REGISTRY[name] = T
end

# Register built-in estimators
register_cov_estimator!(:newey_west, NeweyWestEstimator)
register_cov_estimator!(:white, WhiteEstimator)
register_cov_estimator!(:driscoll_kraay, DriscollKraayEstimator)

# src/lp/core.jl (replace if-chain)
function create_cov_estimator(cov_type::Symbol, ::Type{T}; bandwidth::Int=0) where {T}
    EstType = get(_COV_REGISTRY, cov_type, nothing)
    if EstType === nothing
        valid = join(keys(_COV_REGISTRY), ", ")
        throw(ArgumentError("Unknown cov_type=$cov_type. Available: $valid"))
    end
    # Construct with appropriate kwargs
    return _construct_cov_estimator(EstType, T; bandwidth=bandwidth)
end
```

Export `register_cov_estimator!` so users/extensions can add custom estimators.

### B5. Abstract forecast accessors

**Problem**: 6 forecast types share fields but no common interface.

**Solution**: Define accessor functions on `AbstractForecastResult`:
```julia
# src/core/types.jl
point_forecast(f::AbstractForecastResult) = f.forecast
lower_bound(f::AbstractForecastResult) = f.ci_lower
upper_bound(f::AbstractForecastResult) = f.ci_upper
forecast_horizon(f::AbstractForecastResult) = f.horizon

# Override for special cases
point_forecast(f::FactorForecast) = f.observables
lower_bound(f::FactorForecast) = f.observables_lower
upper_bound(f::FactorForecast) = f.observables_upper
```

Single generic `show()`:
```julia
function Base.show(io::IO, f::AbstractForecastResult{T}) where T
    h = forecast_horizon(f)
    fc = point_forecast(f)
    println(io, "$(typeof(f).name) — $(h)-step forecast")
    # ... common display logic using accessors
end
```

### B6. Display helper consolidation

**Problem**: 40 methods × ~20 lines each = 800 lines of near-identical code.

**Solution**: `_render_table_display()` helper (see A3). In Phase B, also unify `Base.show()` methods for all `AbstractAnalysisResult` subtypes using accessor-based dispatch.

**Target**: `summary_display.jl` from 977 → ~350 lines.

---

## Implementation Order

### Phase A (v0.2.5)
1. A1: conf_level standardization (~1 hour)
2. A2: Forecast field fixes (~3 hours)
3. A4: Missing docstrings (~1 day)
4. A5: report/show convention (~1 day)
5. A3: Display consolidation (~2 days)
6. Full test suite run + version bump

### Phase B (v0.3.0)
1. B4: Covariance registry (~1 day) — lowest risk
2. B3: Kalman consolidation (~2-3 days) — internal only
3. B5: Forecast accessors (~1-2 days)
4. B6: Display helper (~2 days)
5. B1: Package extensions (~3-4 days) — highest risk
6. B2: Remove model.Y (~3-4 days) — most breaking
7. Full test suite run + version bump

---

## Risk Assessment

| Item | Risk | Mitigation |
|------|------|------------|
| A2 (LPForecast rename) | Low | grep + replace, test coverage good |
| B1 (Package extensions) | Medium | Requires Julia 1.9+; test with/without deps |
| B2 (Remove model.Y) | High | Deprecation warning in v0.2.5 first |
| B3 (Kalman consolidation) | Medium | Numerical equivalence tests for all 3 callers |
| B4 (Cov registry) | Low | Drop-in replacement, same behavior |

---

## Success Criteria

- All 7200+ existing tests pass
- No new test failures
- `conf_level` parameter type is `::Real` in all 23 functions
- Forecast types have consistent field access via accessors
- `summary_display.jl` reduced by >50% in LOC
- Package load time <20s without optional deps (Phase B)
- All exported functions have docstrings
