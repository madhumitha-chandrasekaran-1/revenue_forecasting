"""Shared configuration for every stage of the pipeline."""
import os
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLD_EXPORT_DIR = REPO_ROOT / "data" / "gold"
IMAGES_DIR = REPO_ROOT / "docs" / "images"
LOCAL_RUNTIME_DIR = REPO_ROOT / ".local_runtime"   # Spark warehouse + MLflow store when not on Databricks

SCHEMA = os.environ.get("RF_SCHEMA", "revenue_forecasting")
SEED = 42

# --- Simulation -------------------------------------------------------------
N_CUSTOMERS = 10_000
START_DATE = date(2022, 1, 1)
N_MONTHS = 36

SEGMENTS = ["Enterprise", "Mid-Market", "SMB"]
SEG_DIST = [0.10, 0.25, 0.65]
PLANS = ["Basic", "Pro", "Enterprise"]
PLAN_MRR = {"Basic": 99, "Pro": 299, "Enterprise": 999}
PLAN_MIX = {
    "Enterprise": [0.05, 0.25, 0.70],
    "Mid-Market": [0.20, 0.55, 0.25],
    "SMB":        [0.60, 0.35, 0.05],
}
CHANNELS = ["Organic", "Paid Search", "Partner", "Outbound Sales"]
CHANNEL_MIX = {
    "Enterprise": [0.15, 0.05, 0.30, 0.50],
    "Mid-Market": [0.30, 0.20, 0.25, 0.25],
    "SMB":        [0.45, 0.40, 0.10, 0.05],
}
CHANNEL_CHURN_MULT = {"Organic": 0.9, "Paid Search": 1.3, "Partner": 0.8, "Outbound Sales": 1.0}
ANNUAL_CONTRACT_SHARE = {"Enterprise": 0.80, "Mid-Market": 0.50, "SMB": 0.15}

# Target annual logo churn; the generator's base hazards are tuned to land near these.
TARGET_ANNUAL_CHURN = {"Enterprise": 0.05, "Mid-Market": 0.12, "SMB": 0.22}

# --- Modeling ---------------------------------------------------------------
TEST_MONTHS = 3          # last N labelled months held out for evaluation
CALIBRATION_MONTHS = 3   # N labelled months before the test window, used for probability calibration

FEATURES = [
    "mrr", "tenure_months", "months_to_renewal", "contract_annual",
    "segment_enc", "plan_enc", "channel_paid", "channel_partner", "channel_outbound",
    "logins", "rolling_3m_logins", "rolling_6m_logins", "login_trend", "logins_vs_baseline",
    "support_tickets", "rolling_3m_tickets",
    "mrr_change_1m", "mrr_change_3m", "downgraded_6m",
]

# Chosen on the calibration window (never the test window): shallow trees generalised best,
# deeper / longer models over-fit the ~1.5% churn class (validation PR-AUC 0.13 vs 0.11).
XGB_PARAMS = dict(
    n_estimators=200, max_depth=2, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=50, reg_lambda=5, eval_metric="aucpr",
    random_state=SEED, verbosity=0,
)

# --- Forecasting ------------------------------------------------------------
FORECAST_HORIZON = 12
BACKTEST_HORIZON = 6
N_SIMULATIONS = 1_000
SIGNAL_DECAY = 0.75  # monthly weight kept on today's risk signal before reverting to the base rate

SCENARIOS = {
    #        churn hazard multiplier, new-bookings multiplier
    "Bear": (1.20, 0.90),
    "Base": (1.00, 1.00),
    "Bull": (0.80, 1.10),
}

# --- Retention playbook -----------------------------------------------------
OUTREACH_COST = 400.0   # fully loaded CSM cost per targeted account
SAVE_RATE = 0.25        # share of expected churn prevented by outreach

# --- Chart palette (validated light-mode categorical slots) ------------------
COLORS = {
    "series_1": "#2a78d6", "series_2": "#eb6834", "series_3": "#1baf7a",
    "text": "#0b0b0b", "text_muted": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb",
}
