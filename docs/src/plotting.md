# Visualization

MacroEconometricModels.jl includes a zero-dependency visualization system that renders interactive HTML/SVG charts using inline [D3.js](https://d3js.org/) v7. No additional packages are required — D3.js is loaded from CDN at runtime.

The unified `plot_result()` function dispatches on 31 result types, producing self-contained HTML documents with interactive tooltips that work in browsers and Jupyter notebooks.

## Quick Start

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Estimate a VAR and plot IRFs
Y = randn(200, 3)
m = estimate_var(Y, 2)
r = irf(m, 20; ci_type=:bootstrap, reps=500)
p = plot_result(r)

# Save to file
save_plot(p, "irf_plot.html")

# Open in browser
display_plot(p)
```

```@raw html
<iframe src="../assets/plots/quickstart_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Output Type

All `plot_result()` methods return a [`PlotOutput`](@ref) struct:

```julia
struct PlotOutput
    html::String  # Complete self-contained HTML document
end
```

### Displaying Results

| Context | How it works |
|---|---|
| **Jupyter/IJulia** | Automatic inline display via `MIME"text/html"` |
| **REPL** | `display_plot(p)` opens in default browser |
| **File** | `save_plot(p, "path.html")` writes HTML to disk |
| **Programmatic** | Access `p.html` directly |

```julia
# Save to file
save_plot(p, "my_plot.html")

# Open in default browser
display_plot(p)

# Access raw HTML
println(length(p.html), " bytes")
```

---

## Impulse Response Functions

### Frequentist IRF

```julia
Y = randn(200, 3)
m = estimate_var(Y, 2)
r = irf(m, 20; ci_type=:bootstrap, reps=500)

# Full n_vars x n_shocks grid
p = plot_result(r)

# Single response variable and shock
p = plot_result(r; var=1, shock=1)

# String-based selection (uses variable names)
p = plot_result(r; var="Var 1", shock="Shock 2")

# Custom title
p = plot_result(r; title="Monetary Policy Shock")
```

```@raw html
<iframe src="../assets/plots/irf_freq.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**kwargs**: `var` (Int/String), `shock` (Int/String), `ncols` (Int), `title` (String), `save_path` (String)

### Bayesian IRF

```julia
post = estimate_bvar(Y, 2; n_draws=1000)
r = irf(post, 20)
p = plot_result(r)                    # Full grid
p = plot_result(r; var=1, shock=1)    # Single panel
```

```@raw html
<iframe src="../assets/plots/irf_bayesian.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Displays posterior median with credible band from the widest quantile interval.

### Local Projection IRF

```julia
lp = estimate_lp(Y, 1, 20; lags=2)
r = lp_irf(lp)
p = plot_result(r)          # All response variables
p = plot_result(r; var=1)   # Single variable
```

```@raw html
<iframe src="../assets/plots/irf_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Structural LP

```julia
slp = structural_lp(Y, 20; method=:cholesky, lags=2)
p = plot_result(slp)
```

```@raw html
<iframe src="../assets/plots/irf_structural_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Forecast Error Variance Decomposition

### Frequentist FEVD

```julia
m = estimate_var(Y, 2)
f = fevd(m, 20)
p = plot_result(f)          # All variables (stacked area)
p = plot_result(f; var=1)   # Single variable
```

```@raw html
<iframe src="../assets/plots/fevd_freq.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Rendered as stacked area charts with proportions summing to 1.0.

### Bayesian FEVD

```julia
post = estimate_bvar(Y, 2; n_draws=1000)
f = fevd(post, 20)
p = plot_result(f)
p = plot_result(f; stat=:mean)  # Default: posterior mean
```

```@raw html
<iframe src="../assets/plots/fevd_bayesian.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### LP-FEVD

```julia
slp = structural_lp(Y, 20; method=:cholesky, lags=2)
f = lp_fevd(slp, 20)
p = plot_result(f)
p = plot_result(f; bias_corrected=true)  # Default
```

```@raw html
<iframe src="../assets/plots/fevd_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Historical Decomposition

### Frequentist HD

```julia
m = estimate_var(Y, 2)
T_eff = size(m.Y, 1) - m.p
hd = historical_decomposition(m, T_eff)
p = plot_result(hd)          # All variables
p = plot_result(hd; var=1)   # Single variable
```

```@raw html
<iframe src="../assets/plots/hd_freq.html" width="100%" height="600" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Each variable produces two panels: stacked bar chart of shock contributions and a line chart comparing actual values with the sum of contributions.

### Bayesian HD

```julia
post = estimate_bvar(Y, 2; n_draws=1000)
T_eff = size(Y, 1) - 2
hd = historical_decomposition(post, T_eff)
p = plot_result(hd)
```

```@raw html
<iframe src="../assets/plots/hd_bayesian.html" width="100%" height="600" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Time Series Filters

All five filter types produce a two-panel figure: trend vs. original series and the extracted cycle component.

### Hodrick-Prescott

```julia
y = cumsum(randn(200))
p = plot_result(hp_filter(y))
```

```@raw html
<iframe src="../assets/plots/filter_hp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Hamilton (2018)

```julia
p = plot_result(hamilton_filter(y); original=y)
```

```@raw html
<iframe src="../assets/plots/filter_hamilton.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Beveridge-Nelson

```julia
p = plot_result(beveridge_nelson(y))
```

```@raw html
<iframe src="../assets/plots/filter_bn.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Baxter-King

```julia
p = plot_result(baxter_king(y); original=y)
```

```@raw html
<iframe src="../assets/plots/filter_bk.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Boosted HP (Phillips & Shi 2021)

```julia
p = plot_result(boosted_hp(y))
```

```@raw html
<iframe src="../assets/plots/filter_boosted_hp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

!!! note "Original series"
    Hamilton and Baxter-King filters produce shorter output than the input. Pass `original=y` to overlay the full original series in the plot.

---

## Forecasts

### ARIMA Forecast

```julia
y = randn(200)
ar = estimate_ar(y, 2)
fc = forecast(ar, 20)

# Forecast only
p = plot_result(fc)

# With recent history
p = plot_result(fc; history=y, n_history=30)
```

```@raw html
<iframe src="../assets/plots/forecast_arima.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Volatility Forecast

```julia
gm = estimate_garch(y, 1, 1)
fc = forecast(gm, 10)
p = plot_result(fc)
p = plot_result(fc; history=gm.conditional_variance)
```

```@raw html
<iframe src="../assets/plots/forecast_volatility.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### VECM Forecast

```julia
Y = cumsum(randn(150, 3), dims=1)
vecm = estimate_vecm(Y, 2; rank=1)
fc = forecast(vecm, 10)
p = plot_result(fc)          # All variables
p = plot_result(fc; var=1)   # Single variable
```

```@raw html
<iframe src="../assets/plots/forecast_vecm.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Factor Forecast

```julia
X = randn(200, 20)
fm = estimate_dynamic_factors(X, 2, 1)
fc = forecast(fm, 10)

p = plot_result(fc)                          # Factor forecasts
p = plot_result(fc; type=:observable, var=1) # Observable forecast
```

```@raw html
<iframe src="../assets/plots/forecast_factor.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### LP Forecast

```julia
Y = randn(100, 3)
lp = estimate_lp(Y, 1, 10; lags=2)
shock_path = zeros(10); shock_path[1] = 1.0
fc = forecast(lp, shock_path)
p = plot_result(fc)
```

```@raw html
<iframe src="../assets/plots/forecast_lp.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Volatility Models

ARCH, GARCH, EGARCH, and GJR-GARCH models produce a three-panel diagnostic figure: return series, conditional volatility, and standardized residuals with +/-2 standard deviation bounds.

```julia
y = randn(500)

p = plot_result(estimate_arch(y, 2))
p = plot_result(estimate_garch(y, 1, 1))
p = plot_result(estimate_egarch(y, 1, 1))
p = plot_result(estimate_gjr_garch(y, 1, 1))
```

```@raw html
<iframe src="../assets/plots/model_garch.html" width="100%" height="700" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Stochastic Volatility

The SV model shows posterior volatility with quantile credible bands:

```julia
m = estimate_sv(y; n_samples=2000, burnin=1000)
p = plot_result(m)
```

```@raw html
<iframe src="../assets/plots/model_sv.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Factor Models

Factor models display a scree plot of eigenvalues and the extracted factor time series:

```julia
X = randn(200, 20)

# Static factors
p = plot_result(estimate_factors(X, 3))

# Dynamic factors
p = plot_result(estimate_dynamic_factors(X, 2, 1))
```

```@raw html
<iframe src="../assets/plots/model_factor_static.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Data Containers

### TimeSeriesData

```julia
d = TimeSeriesData(randn(100, 3); varnames=["GDP", "CPI", "RATE"])
p = plot_result(d)                      # All variables
p = plot_result(d; vars=["GDP", "CPI"]) # Subset
```

```@raw html
<iframe src="../assets/plots/data_timeseries.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### PanelData

```julia
using DataFrames
df = DataFrame(group=repeat(1:3, inner=20), time=repeat(1:20, 3),
    x=randn(60), y=randn(60))
pd = xtset(df, :group, :time)
p = plot_result(pd)
```

```@raw html
<iframe src="../assets/plots/data_panel.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Panel data plots show each variable in a separate panel with one line per group.

---

## Nowcasting

### Nowcast Result

```julia
Y = randn(100, 5)
Y[end, end] = NaN  # Missing quarterly observation
dfm = nowcast_dfm(Y, 4, 1; r=2, p=1)
nr = nowcast(dfm)
p = plot_result(nr)
```

```@raw html
<iframe src="../assets/plots/nowcast_result.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Displays smoothed target variable with nowcast and forecast values annotated in the title.

### Nowcast News

```julia
X_old = randn(100, 5); X_old[end, end] = NaN
X_new = copy(X_old); X_new[end, end] = 0.5
dfm = nowcast_dfm(X_old, 4, 1; r=2, p=1)
nn = nowcast_news(X_new, X_old, dfm, 5)
p = plot_result(nn)
```

```@raw html
<iframe src="../assets/plots/nowcast_news.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Bar chart showing per-release impact on the nowcast revision.

---

## Common Options

All `plot_result()` methods accept these keyword arguments:

| Kwarg | Type | Default | Description |
|---|---|---|---|
| `title` | `String` | `""` | Override auto-generated title |
| `save_path` | `String` or `nothing` | `nothing` | Auto-save HTML to path |
| `ncols` | `Int` | `0` (auto) | Number of columns in multi-panel grid |

Type-specific kwargs (e.g., `var`, `shock`, `history`, `stat`, `bias_corrected`) are documented for each method above.

---

## Chart Types

Three D3.js chart types cover all use cases:

| Chart | Used for | Features |
|---|---|---|
| **Line** | IRF, forecasts, filters, volatility, data series | CI bands, dashed lines, zero reference line, tooltips |
| **Stacked area** | FEVD | Proportions summing to 1.0, per-shock coloring |
| **Bar** | Historical decomposition, nowcast news | Stacked or grouped, diverging stack for negative values |

All charts include interactive tooltips and a consistent color palette.

## References

- Bostock, M., Ogievetsky, V., & Heer, J. (2011). D3: Data-Driven Documents. *IEEE Transactions on Visualization and Computer Graphics*, 17(12), 2301–2309. DOI: 10.1109/TVCG.2011.185
