# Factor Models

This chapter covers static factor models for dimensionality reduction in large macroeconomic panels, including estimation via principal components and information criteria for selecting the number of factors.

## Introduction

Factor models are fundamental tools in macroeconometrics for extracting common sources of variation from large panels of economic indicators. They enable:

1. **Dimensionality Reduction**: Summarize ``N`` variables with ``r \ll N`` factors
2. **Forecasting**: Use factors as predictors in regressions (diffusion indices)
3. **Structural Analysis**: Identify common shocks driving multiple series
4. **FAVAR**: Combine factors with VARs for high-dimensional structural analysis

**Reference**: Stock & Watson (2002a, 2002b), Bai & Ng (2002)

---

## The Static Factor Model

### Model Specification

The static factor model decomposes an ``N``-dimensional vector of observables ``x_t`` into common and idiosyncratic components:

```math
x_{it} = \lambda_i' F_t + e_{it}, \quad i = 1, \ldots, N, \quad t = 1, \ldots, T
```

In matrix form:

```math
X = F \Lambda' + E
```

where:
- ``X`` is the ``T \times N`` data matrix
- ``F`` is the ``T \times r`` matrix of latent factors
- ``\Lambda`` is the ``N \times r`` matrix of factor loadings
- ``E`` is the ``T \times N`` matrix of idiosyncratic errors
- ``r`` is the number of factors (with ``r \ll \min(T, N)``)

### Assumptions

**Factors and Loadings**:
- ``E[F_t] = 0``, ``\text{Var}(F_t) = I_r`` (normalization)
- ``\frac{1}{T} \sum_t F_t F_t' \xrightarrow{p} \Sigma_F`` positive definite
- ``\frac{1}{N} \Lambda' \Lambda \xrightarrow{p} \Sigma_\Lambda`` positive definite

**Idiosyncratic Errors**:
- ``E[e_{it}] = 0``
- Weak cross-sectional and temporal dependence allowed
- Weak correlation with factors: ``\frac{1}{NT} \sum_{i,t} |E[F_t e_{it}]| \to 0``

**Reference**: Bai & Ng (2002), Bai (2003)

---

## Estimation via Principal Components

### Principal Components Analysis (PCA)

The factors and loadings are estimated by minimizing the sum of squared idiosyncratic errors:

```math
\min_{F, \Lambda} \sum_{i=1}^N \sum_{t=1}^T (x_{it} - \lambda_i' F_t)^2
```

subject to the normalization ``F'F/T = I_r``.

### Solution

The solution involves the eigenvalue decomposition of ``X'X`` (or ``XX'``):

**Case 1**: ``T < N`` (short panel)
- Compute ``XX'`` (``T \times T`` matrix)
- ``\hat{F} = \sqrt{T} \times`` (first ``r`` eigenvectors of ``XX'``)
- ``\hat{\Lambda} = X' \hat{F} / T``

**Case 2**: ``N \leq T`` (tall panel)
- Compute ``X'X`` (``N \times N`` matrix)
- ``\hat{\Lambda} = \sqrt{N} \times`` (first ``r`` eigenvectors of ``X'X``)
- ``\hat{F} = X \hat{\Lambda} / N``

### Data Preprocessing

Before estimation, data is typically:
1. **Demeaned**: Center each series to have zero mean
2. **Standardized**: Scale each series to have unit variance

This prevents high-variance series from dominating the factor extraction.

### Identification

The factors and loadings are identified only up to an ``r \times r`` invertible rotation. If ``(F, \Lambda)`` is a solution, so is ``(FH, \Lambda H^{-1})`` for any invertible ``H``.

The normalization ``F'F/T = I_r`` and ``\Lambda'\Lambda`` diagonal pins down rotation up to sign.

**Reference**: Stock & Watson (2002a), Bai & Ng (2002)

### Julia Implementation

```julia
using Macroeconometrics

# X is T×N data matrix
# Estimate r-factor model

model = estimate_factors(X, r;
    standardize = true,    # Standardize data
    method = :pca          # Principal components
)

# Access results
F = model.factors          # T×r estimated factors
Λ = model.loadings         # N×r estimated loadings
```

---

## Determining the Number of Factors

### The Selection Problem

Choosing ``r`` is crucial:
- Too few factors: Omitted common variation, biased estimates
- Too many factors: Overfitting, including noise as signal

### Bai & Ng (2002) Information Criteria

Bai & Ng propose three information criteria:

**IC1**:
```math
IC_1(r) = \log \hat{\sigma}^2(r) + r \cdot \frac{N + T}{NT} \log\left( \frac{NT}{N+T} \right)
```

**IC2**:
```math
IC_2(r) = \log \hat{\sigma}^2(r) + r \cdot \frac{N + T}{NT} \log(C_{NT}^2)
```

**IC3**:
```math
IC_3(r) = \log \hat{\sigma}^2(r) + r \cdot \frac{\log(C_{NT}^2)}{C_{NT}^2}
```

where:
- ``\hat{\sigma}^2(r) = \frac{1}{NT} \sum_{i,t} \hat{e}_{it}^2`` is the average squared residual
- ``C_{NT}^2 = \min(N, T)``

**Selection Rule**: Choose ``\hat{r}`` that minimizes ``IC_k(r)`` over ``r \in \{1, \ldots, r_{max}\}``.

**Properties**:
- IC2 and IC3 perform best in simulations
- All three are consistent: ``\hat{r} \xrightarrow{p} r_0`` as ``N, T \to \infty``

**Reference**: Bai & Ng (2002)

### Julia Implementation

```julia
using Macroeconometrics

# Compute IC for r = 1, ..., r_max
r_max = 10
ic = ic_criteria(X, r_max)

# Optimal number by each criterion
println("IC1 selects: ", ic.r_IC1, " factors")
println("IC2 selects: ", ic.r_IC2, " factors")
println("IC3 selects: ", ic.r_IC3, " factors")

# IC values for all r
for r in 1:r_max
    println("r=$r: IC1=$(ic.IC1[r]), IC2=$(ic.IC2[r]), IC3=$(ic.IC3[r])")
end
```

---

## Scree Plot Analysis

### Visual Factor Selection

The scree plot displays eigenvalues (or variance explained) against factor number. The "elbow" in the plot suggests the number of significant factors.

### Variance Explained

For each factor ``j``:

**Individual Variance**:
```math
\text{VarExp}_j = \frac{\mu_j}{\sum_{k=1}^N \mu_k}
```

**Cumulative Variance**:
```math
\text{CumVarExp}_r = \sum_{j=1}^r \text{VarExp}_j
```

where ``\mu_j`` is the ``j``-th largest eigenvalue of ``X'X/T`` (or ``XX'/N``).

### Julia Implementation

```julia
using Macroeconometrics

model = estimate_factors(X, r)

# Get scree plot data
scree = scree_plot_data(model)

# Variance explained
for j in 1:min(10, length(scree.factors))
    println("Factor $j: $(round(scree.explained_variance[j]*100, digits=2))% ",
            "(cumulative: $(round(scree.cumulative_variance[j]*100, digits=2))%)")
end
```

---

## Model Diagnostics

### R-squared for Each Variable

The ``R^2`` measures how much of variable ``i``'s variation is explained by the common factors:

```math
R^2_i = 1 - \frac{\sum_t \hat{e}_{it}^2}{\sum_t (x_{it} - \bar{x}_i)^2}
```

Variables with low ``R^2`` are mainly driven by idiosyncratic shocks.

### Julia Implementation

```julia
using Macroeconometrics

model = estimate_factors(X, r)

# R² for each variable
r2_values = r2(model)

# Summary statistics
println("Mean R²: ", round(mean(r2_values), digits=3))
println("Median R²: ", round(median(r2_values), digits=3))
println("Min R²: ", round(minimum(r2_values), digits=3))
println("Max R²: ", round(maximum(r2_values), digits=3))

# Variables well-explained by factors
well_explained = findall(r2_values .> 0.7)
```

### Fitted Values and Residuals

```julia
# Fitted values: X̂ = FΛ'
X_fitted = predict(model)

# Residuals: E = X - X̂
resid = residuals(model)

# Model statistics
println("Number of observations: ", nobs(model))
println("Degrees of freedom: ", dof(model))
```

---

## Applications

### Diffusion Index Forecasting

Use factors as predictors for forecasting a target variable ``y_{t+h}``:

```math
y_{t+h} = \alpha + \beta' \hat{F}_t + \gamma' y_{t:t-p} + \varepsilon_{t+h}
```

Factors summarize information from a large panel, improving forecast accuracy.

**Reference**: Stock & Watson (2002b)

### Factor-Augmented VAR (FAVAR)

Combine factors with key observable variables in a VAR:

```math
\begin{bmatrix} y_t \\ F_t \end{bmatrix} = A_1 \begin{bmatrix} y_{t-1} \\ F_{t-1} \end{bmatrix} + \cdots + A_p \begin{bmatrix} y_{t-p} \\ F_{t-p} \end{bmatrix} + u_t
```

This allows structural analysis with high-dimensional information sets.

**Reference**: Bernanke, Boivin & Eliasz (2005)

### Example: FAVAR Setup

```julia
using Macroeconometrics

# Estimate factors from large panel X
fm = estimate_factors(X, r)
F = fm.factors

# Combine with key observables (e.g., FFR, GDP, inflation)
Y_key = data[:, [:FFR, :GDP, :CPI]]
Y_favar = hcat(Y_key, F)

# Estimate FAVAR
favar_model = fit(VARModel, Y_favar, p)

# Structural analysis
irf_favar = irf(favar_model, H; method=:cholesky)
```

---

## Asymptotic Theory

### Consistency of Factor Estimates

Under the assumptions of Bai & Ng (2002), as ``T, N \to \infty``:

```math
\frac{1}{T} \sum_{t=1}^T \|\hat{F}_t - H F_t\|^2 = O_p\left( \frac{1}{\min(N, T)} \right)
```

where ``H`` is an ``r \times r`` rotation matrix.

The factors are consistently estimated up to rotation at rate ``\min(\sqrt{N}, \sqrt{T})``.

### Distribution Theory

For large ``N, T``, the factor estimates are asymptotically normal:

```math
\sqrt{T} (\hat{F}_t - H F_t) \xrightarrow{d} N(0, V)
```

where ``V`` depends on the cross-sectional and temporal dependence structure.

**Reference**: Bai (2003), Bai & Ng (2006)

---

## Comparison with Other Methods

### Static vs. Dynamic Factor Models

| Aspect | Static FM | Dynamic FM |
|--------|-----------|------------|
| **Model** | ``X_t = \Lambda F_t + e_t`` | ``X_t = \Lambda(L) f_t + e_t`` |
| **Factors** | Contemporaneous | May include lags |
| **Estimation** | PCA | Spectral methods, Kalman filter |
| **Use case** | Large N, moderate T | Time series dynamics important |

**Reference**: Forni, Hallin, Lippi & Reichlin (2000)

### Maximum Likelihood Estimation

ML estimation assumes Gaussian factors and errors:

```math
F_t \sim N(0, I_r), \quad e_t \sim N(0, \Psi)
```

Estimated via EM algorithm. More efficient than PCA if model is correctly specified, but computationally intensive.

---

## References

### Core Theory

- Bai, J. (2003). "Inferential Theory for Factor Models of Large Dimensions." *Econometrica*, 71(1), 135-171.
- Bai, J., & Ng, S. (2002). "Determining the Number of Factors in Approximate Factor Models." *Econometrica*, 70(1), 191-221.
- Bai, J., & Ng, S. (2006). "Confidence Intervals for Diffusion Index Forecasts and Inference for Factor-Augmented Regressions." *Econometrica*, 74(4), 1133-1150.
- Stock, J. H., & Watson, M. W. (2002a). "Forecasting Using Principal Components from a Large Number of Predictors." *Journal of the American Statistical Association*, 97(460), 1167-1179.
- Stock, J. H., & Watson, M. W. (2002b). "Macroeconomic Forecasting Using Diffusion Indexes." *Journal of Business & Economic Statistics*, 20(2), 147-162.

### Applications

- Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). "Measuring the Effects of Monetary Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach." *Quarterly Journal of Economics*, 120(1), 387-422.
- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2000). "The Generalized Dynamic-Factor Model: Identification and Estimation." *Review of Economics and Statistics*, 82(4), 540-554.
- McCracken, M. W., & Ng, S. (2016). "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business & Economic Statistics*, 34(4), 574-589.
