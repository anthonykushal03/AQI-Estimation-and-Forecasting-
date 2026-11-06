"""Project-wide configuration: paths, periods, column names, model settings and random seeds."""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_TASK1_DIR = FIG_DIR / "task1_estimation"
FIG_TASK2_DIR = FIG_DIR / "task2_forecasting"
FIG_DIAG_DIR = FIG_DIR / "diagnostics"
METRIC_DIR = RESULTS_DIR / "metrics"               # headline results of both tasks
EXPLAIN_DIR = RESULTS_DIR / "explainability"       # SHAP importance tables
DIAG_DIR = RESULTS_DIR / "diagnostics"             # data-quality, leakage and feature-selection checks
PRED_DIR = RESULTS_DIR / "predictions"

# ---------------------------------------------------------------- periods
YEARS = range(2020, 2025)
TRAIN_END_YEAR = 2023          # 2020-2023 -> training
TEST_YEAR = 2024               # 2024      -> testing
VALIDATION_YEAR = 2023         # used only for feature decisions inside the training years

# ---------------------------------------------------------------- columns
PM25, PM10 = "PM2.5 (µg/m³)", "PM10 (µg/m³)"
NO, NO2, NOX = "NO (µg/m³)", "NO2 (µg/m³)", "NOx (ppb)"
CO, TOLUENE, BENZENE = "CO (mg/m³)", "Toluene (µg/m³)", "Benzene (µg/m³)"
AT, RH, WS, WD = "AT (°C)", "RH (%)", "WS (m/s)", "WD (deg)"

POLLUTANTS = [PM25, PM10, NO, NO2, NOX, "NH3 (µg/m³)", "SO2 (µg/m³)", CO, "Ozone (µg/m³)", BENZENE, TOLUENE]
WEATHER = [AT, RH, WS, WD, "SR (W/mt2)", "BP (mmHg)"]
ALL_FEATURES = POLLUTANTS + WEATHER

# ---------------------------------------------------------------- Task 1
CORR_THRESHOLD = 0.3
# Pollutants with r > 0.3 on measured training days, minus Toluene (174-day gap, see task1 features)
TASK1_FEATURES = [PM25, PM10, NO, NO2, NOX, CO]
N_PCA_COMPONENTS = 4

# ---------------------------------------------------------------- Task 2
FC_INPUTS = [PM25, PM10, CO, NO2, AT, RH, WS]
SETUPS = {"Strict": 1, "Same-day": 0}      # lag (days) applied to pollutant and weather inputs
LEAKAGE_TEST_DAYS = 25
ARIMA_ORDER_GRID = dict(p=range(4), d=range(2), q=range(3))
N_BOOTSTRAP = 2000

# 30-day ARIMA experiment
ARIMA_30DAY_ORDER = (30, 2, 0)
HOLDOUT_DAYS = 30

# CPCB breakpoints (concentration low, high, index low, high)
PM10_BREAKPOINTS = [(0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200),
                    (250, 350, 200, 300), (350, 430, 300, 400), (430, 1000, 400, 500)]
PM25_BREAKPOINTS = [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200),
                    (90, 120, 200, 300), (120, 250, 300, 400), (250, 1000, 400, 500)]

# ---------------------------------------------------------------- models (fixed in advance, not tuned on 2024)
SEED = 42
RF_PARAMS = dict(n_estimators=500, random_state=SEED)
XGB_PARAMS = dict(n_estimators=100, learning_rate=0.1, max_depth=3)
