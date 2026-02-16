# Generate HTML plot files for documentation embedding
# Run with: julia --project=docs docs/generate_plots.jl

using MacroEconometricModels
using DataFrames
using Random

Random.seed!(42)

# Output directory
const PLOT_DIR = joinpath(@__DIR__, "src", "assets", "plots")
mkpath(PLOT_DIR)

function save(name::String, p::PlotOutput)
    path = joinpath(PLOT_DIR, name)
    save_plot(p, path)
    println("  ✓ $name")
end

println("Generating documentation plots...")

# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------
Y3   = randn(200, 3)          # 3-variable series (VAR/LP/FEVD/HD)
y1   = randn(200)             # univariate (ARIMA, volatility)
y_rw = cumsum(randn(200))     # random walk (filters)
X20  = randn(200, 20)         # wide panel (factor models)
Y_ci = cumsum(randn(150, 3), dims=1)  # cointegrated-like (VECM)

# ---------------------------------------------------------------------------
# 1. Quick Start IRF
# ---------------------------------------------------------------------------
m_var = estimate_var(Y3, 2)
r_qs  = irf(m_var, 20; ci_type=:bootstrap, reps=500)
save("quickstart_irf.html", plot_result(r_qs))

# ---------------------------------------------------------------------------
# 2. Frequentist IRF (full grid)
# ---------------------------------------------------------------------------
save("irf_freq.html", plot_result(r_qs))

# ---------------------------------------------------------------------------
# 3. Bayesian IRF
# ---------------------------------------------------------------------------
post = estimate_bvar(Y3, 2; n_draws=1000)
r_birf = irf(post, 20)
save("irf_bayesian.html", plot_result(r_birf))

# ---------------------------------------------------------------------------
# 4. LP IRF
# ---------------------------------------------------------------------------
lp_m = estimate_lp(Y3, 1, 20; lags=2)
r_lp = lp_irf(lp_m)
save("irf_lp.html", plot_result(r_lp))

# ---------------------------------------------------------------------------
# 5. Structural LP IRF
# ---------------------------------------------------------------------------
slp = structural_lp(Y3, 20; method=:cholesky, lags=2)
save("irf_structural_lp.html", plot_result(slp))

# ---------------------------------------------------------------------------
# 6. Frequentist FEVD
# ---------------------------------------------------------------------------
f_freq = fevd(m_var, 20)
save("fevd_freq.html", plot_result(f_freq))

# ---------------------------------------------------------------------------
# 7. Bayesian FEVD
# ---------------------------------------------------------------------------
f_bay = fevd(post, 20)
save("fevd_bayesian.html", plot_result(f_bay))

# ---------------------------------------------------------------------------
# 8. LP-FEVD
# ---------------------------------------------------------------------------
f_lp = lp_fevd(slp, 20)
save("fevd_lp.html", plot_result(f_lp))

# ---------------------------------------------------------------------------
# 9. Frequentist HD
# ---------------------------------------------------------------------------
T_eff = size(m_var.Y, 1) - m_var.p
hd_freq = historical_decomposition(m_var, T_eff)
save("hd_freq.html", plot_result(hd_freq))

# ---------------------------------------------------------------------------
# 10. Bayesian HD
# ---------------------------------------------------------------------------
T_eff_b = size(Y3, 1) - 2
hd_bay = historical_decomposition(post, T_eff_b)
save("hd_bayesian.html", plot_result(hd_bay))

# ---------------------------------------------------------------------------
# 11-15. Time Series Filters
# ---------------------------------------------------------------------------
save("filter_hp.html",         plot_result(hp_filter(y_rw)))
save("filter_hamilton.html",   plot_result(hamilton_filter(y_rw); original=y_rw))
save("filter_bn.html",         plot_result(beveridge_nelson(y_rw)))
save("filter_bk.html",         plot_result(baxter_king(y_rw); original=y_rw))
save("filter_boosted_hp.html", plot_result(boosted_hp(y_rw)))

# ---------------------------------------------------------------------------
# 16. ARIMA Forecast
# ---------------------------------------------------------------------------
ar = estimate_ar(y1, 2)
fc_ar = forecast(ar, 20)
save("forecast_arima.html", plot_result(fc_ar; history=y1, n_history=30))

# ---------------------------------------------------------------------------
# 17. Volatility Forecast
# ---------------------------------------------------------------------------
gm = estimate_garch(y1, 1, 1)
fc_vol = forecast(gm, 10)
save("forecast_volatility.html", plot_result(fc_vol; history=gm.conditional_variance))

# ---------------------------------------------------------------------------
# 18. VECM Forecast
# ---------------------------------------------------------------------------
vecm_m = estimate_vecm(Y_ci, 2; rank=1)
fc_vecm = forecast(vecm_m, 10)
save("forecast_vecm.html", plot_result(fc_vecm))

# ---------------------------------------------------------------------------
# 19. Factor Forecast
# ---------------------------------------------------------------------------
fm = estimate_dynamic_factors(X20, 2, 1)
fc_fm = forecast(fm, 10)
save("forecast_factor.html", plot_result(fc_fm))

# ---------------------------------------------------------------------------
# 20. LP Forecast
# ---------------------------------------------------------------------------
Y_small = randn(100, 3)
lp_fc_m = estimate_lp(Y_small, 1, 10; lags=2)
shock_path = zeros(10); shock_path[1] = 1.0
fc_lp = forecast(lp_fc_m, shock_path)
save("forecast_lp.html", plot_result(fc_lp))

# ---------------------------------------------------------------------------
# 21. GARCH diagnostic
# ---------------------------------------------------------------------------
save("model_garch.html", plot_result(gm))

# ---------------------------------------------------------------------------
# 22. SV posterior volatility
# ---------------------------------------------------------------------------
sv_m = estimate_sv(y1; n_samples=2000, burnin=1000)
save("model_sv.html", plot_result(sv_m))

# ---------------------------------------------------------------------------
# 23. Static factor model
# ---------------------------------------------------------------------------
fm_static = estimate_factors(X20, 3)
save("model_factor_static.html", plot_result(fm_static))

# ---------------------------------------------------------------------------
# 24. TimeSeriesData
# ---------------------------------------------------------------------------
d_ts = TimeSeriesData(randn(100, 3); varnames=["GDP", "CPI", "RATE"])
save("data_timeseries.html", plot_result(d_ts))

# ---------------------------------------------------------------------------
# 25. PanelData
# ---------------------------------------------------------------------------
df = DataFrame(group=repeat(1:3, inner=20), time=repeat(1:20, 3),
    x=randn(60), y=randn(60))
pd = xtset(df, :group, :time)
save("data_panel.html", plot_result(pd))

# ---------------------------------------------------------------------------
# 26. Nowcast result
# ---------------------------------------------------------------------------
Y_nc = randn(100, 5)
Y_nc[end, end] = NaN
dfm_nc = nowcast_dfm(Y_nc, 4, 1; r=2, p=1)
nr = nowcast(dfm_nc)
save("nowcast_result.html", plot_result(nr))

# ---------------------------------------------------------------------------
# 27. Nowcast news
# ---------------------------------------------------------------------------
X_old = randn(100, 5); X_old[end, end] = NaN
X_new = copy(X_old); X_new[end, end] = 0.5
dfm_news = nowcast_dfm(X_old, 4, 1; r=2, p=1)
nn = nowcast_news(X_new, X_old, dfm_news, 5)
save("nowcast_news.html", plot_result(nn))

println("\nDone! Generated $(length(readdir(PLOT_DIR))) HTML files in $PLOT_DIR")
