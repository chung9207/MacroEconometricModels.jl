# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
Critical value tables for unit root tests.
"""

# =============================================================================
# Critical Value Tables
# =============================================================================

# MacKinnon (2010) response surface coefficients for ADF/PP p-values
# Format: (β∞, β₁, β₂) for τ = β∞ + β₁/T + β₂/T²
const MACKINNON_ADF_COEFS = Dict(
    # No constant, no trend (nc)
    :none => Dict(
        1  => (-2.5658, -1.960, -10.04),   # 1%
        5  => (-1.9393, -0.398,  -0.0),    # 5%
        10 => (-1.6156, -0.181,  -0.0)     # 10%
    ),
    # Constant only (c)
    :constant => Dict(
        1  => (-3.4336, -5.999, -29.25),
        5  => (-2.8621, -2.738,  -8.36),
        10 => (-2.5671, -1.438,  -4.48)
    ),
    # Constant and trend (ct)
    :trend => Dict(
        1  => (-3.9638, -8.353, -47.44),
        5  => (-3.4126, -4.039, -17.83),
        10 => (-3.1279, -2.418,  -7.58)
    )
)

# KPSS critical values (Kwiatkowski et al. 1992, Table 1)
const KPSS_CRITICAL_VALUES = Dict(
    :constant => Dict(1 => 0.739, 5 => 0.463, 10 => 0.347),
    :trend    => Dict(1 => 0.216, 5 => 0.146, 10 => 0.119)
)

# Zivot-Andrews critical values (Zivot & Andrews 1992, Table 4)
const ZA_CRITICAL_VALUES = Dict(
    :constant => Dict(1 => -5.34, 5 => -4.80, 10 => -4.58),
    :trend    => Dict(1 => -4.80, 5 => -4.42, 10 => -4.11),
    :both     => Dict(1 => -5.57, 5 => -5.08, 10 => -4.82)
)

# Ng-Perron critical values (Ng & Perron 2001, Table 1)
const NGPERRON_CRITICAL_VALUES = Dict(
    :constant => Dict(
        :MZa => Dict(1 => -13.8, 5 => -8.1, 10 => -5.7),
        :MZt => Dict(1 => -2.58, 5 => -1.98, 10 => -1.62),
        :MSB => Dict(1 => 0.174, 5 => 0.233, 10 => 0.275),
        :MPT => Dict(1 => 1.78, 5 => 3.17, 10 => 4.45)
    ),
    :trend => Dict(
        :MZa => Dict(1 => -23.8, 5 => -17.3, 10 => -14.2),
        :MZt => Dict(1 => -3.42, 5 => -2.91, 10 => -2.62),
        :MSB => Dict(1 => 0.143, 5 => 0.168, 10 => 0.185),
        :MPT => Dict(1 => 4.03, 5 => 5.48, 10 => 6.67)
    )
)

# Johansen critical values (Osterwald-Lenum 1992)
# Format: [10%, 5%, 1%] for each n-r (number of common trends)

# Case 1 (:none) — No deterministic terms
# Trace test critical values
const JOHANSEN_TRACE_CV_NONE = Dict(
    1 => [2.69, 3.84, 6.63],
    2 => [13.33, 15.41, 20.04],
    3 => [26.79, 29.68, 35.65],
    4 => [43.95, 47.21, 54.46],
    5 => [64.84, 68.52, 76.07],
    6 => [85.18, 90.39, 104.20],
    7 => [118.99, 124.25, 136.06],
    8 => [151.38, 157.11, 168.92],
    9 => [186.54, 192.89, 206.95],
    10 => [224.63, 231.26, 247.18]
)

# Case 1 — Max eigenvalue test critical values
const JOHANSEN_MAX_CV_NONE = Dict(
    1 => [2.69, 3.84, 6.63],
    2 => [12.07, 14.07, 18.63],
    3 => [18.60, 20.97, 25.52],
    4 => [24.73, 27.07, 32.24],
    5 => [30.67, 33.46, 38.77],
    6 => [36.25, 39.43, 44.59],
    7 => [42.06, 45.28, 51.30],
    8 => [48.43, 51.42, 57.07],
    9 => [54.01, 57.12, 62.80],
    10 => [59.00, 62.81, 68.83]
)

# Case 2 (:constant) — Restricted constant in cointegrating relation
# Trace test critical values (Osterwald-Lenum 1992, Table 1*)
const JOHANSEN_TRACE_CV_CONSTANT = Dict(
    1 => [7.52, 9.24, 12.97],
    2 => [17.85, 19.96, 24.60],
    3 => [32.00, 34.91, 41.07],
    4 => [49.65, 53.12, 60.16],
    5 => [71.86, 76.07, 84.45],
    6 => [97.18, 102.14, 111.01],
    7 => [126.58, 131.70, 143.09],
    8 => [159.48, 165.58, 177.20],
    9 => [196.37, 202.92, 215.74],
    10 => [236.54, 244.15, 257.68]
)

# Case 2 — Max eigenvalue test critical values
const JOHANSEN_MAX_CV_CONSTANT = Dict(
    1 => [7.52, 9.24, 12.97],
    2 => [13.75, 15.67, 20.20],
    3 => [19.77, 22.00, 26.81],
    4 => [25.56, 28.14, 33.24],
    5 => [31.66, 34.40, 39.79],
    6 => [37.45, 40.30, 46.82],
    7 => [43.25, 46.45, 51.91],
    8 => [49.18, 52.00, 57.95],
    9 => [54.71, 57.42, 63.71],
    10 => [60.29, 63.57, 69.94]
)

# Case 4 (:trend) — Restricted trend in cointegrating relation, unrestricted constant
# Trace test critical values (Osterwald-Lenum 1992, Table 2*)
const JOHANSEN_TRACE_CV_TREND = Dict(
    1 => [10.49, 12.53, 16.31],
    2 => [22.76, 25.32, 30.45],
    3 => [39.06, 42.44, 48.45],
    4 => [59.14, 62.99, 70.05],
    5 => [83.20, 87.31, 96.58],
    6 => [110.42, 114.90, 124.75],
    7 => [141.01, 146.76, 155.36],
    8 => [175.16, 182.82, 192.89],
    9 => [212.66, 222.21, 233.13],
    10 => [254.46, 264.22, 277.71]
)

# Case 4 — Max eigenvalue test critical values
const JOHANSEN_MAX_CV_TREND = Dict(
    1 => [10.49, 12.53, 16.31],
    2 => [16.85, 18.96, 23.65],
    3 => [23.11, 25.54, 30.34],
    4 => [29.12, 31.46, 36.65],
    5 => [34.75, 37.52, 42.36],
    6 => [40.91, 43.97, 48.94],
    7 => [46.32, 49.51, 54.71],
    8 => [52.16, 55.24, 61.57],
    9 => [57.87, 60.81, 66.76],
    10 => [63.37, 66.91, 72.64]
)
