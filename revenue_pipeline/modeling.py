"""Churn models, evaluation, explanations and forward hazard projection."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

RATIO_FEATURES = ["login_trend", "logins_vs_baseline"]

FEATURE_LABELS = {
    "mrr": "MRR", "tenure_months": "Tenure (months)", "months_to_renewal": "Months to renewal",
    "contract_annual": "Annual contract", "segment_enc": "Segment", "plan_enc": "Plan tier",
    "channel_paid": "Paid-search acquisition", "channel_partner": "Partner acquisition",
    "channel_outbound": "Outbound acquisition", "logins": "Logins this month",
    "rolling_3m_logins": "Logins, 3-mo avg", "rolling_6m_logins": "Logins, 6-mo avg",
    "login_trend": "Login trend vs prior quarter", "logins_vs_baseline": "Logins vs own baseline",
    "support_tickets": "Support tickets", "rolling_3m_tickets": "Tickets, 3-mo avg",
    "mrr_change_1m": "MRR change, 1 mo", "mrr_change_3m": "MRR change, 3 mo", "downgraded_6m": "Recent downgrade",
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[config.FEATURES].astype(float)
    X[RATIO_FEATURES] = X[RATIO_FEATURES].clip(0, 3)
    X[["mrr_change_1m", "mrr_change_3m"]] = X[["mrr_change_1m", "mrr_change_3m"]].clip(-0.5, 0.5)
    return X


def labelled(feat: pd.DataFrame) -> pd.DataFrame:
    return feat[feat["churn_next_month"].notna()]


def month_offset(month, n) -> pd.Timestamp:
    return pd.Timestamp(month) + pd.DateOffset(months=n)


class PlattCalibrator:
    """Logistic regression on the raw score's log-odds.

    Class weighting multiplies the odds of churn by roughly `scale_pos_weight`, i.e. shifts log-odds by a
    near-constant amount, which is exactly what a one-feature logistic fit undoes. Being strictly monotone
    it also leaves the ranking untouched (isotonic regression, tried first, created ties and hit 1.0).
    """

    def fit(self, raw, y):
        self.model = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(raw), np.asarray(y))
        return self

    def predict(self, raw):
        return self.model.predict_proba(self._logit(raw))[:, 1]

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)


class ChurnModel:
    """Gradient-boosted classifier + calibration map from raw scores to probabilities."""

    def __init__(self, booster, calibrator):
        self.booster = booster
        self.calibrator = calibrator

    def predict_raw(self, df):
        return self.booster.predict_proba(prepare_features(df))[:, 1]

    def predict_proba(self, df):
        return self.calibrator.predict(self.predict_raw(df))


def fit_churn_model(feat: pd.DataFrame, last_train_month) -> ChurnModel:
    """Train on labelled rows up to `last_train_month`; the final CALIBRATION_MONTHS of those fit the calibrator.

    scale_pos_weight up-weights the rare churners so the trees learn their patterns; that inflates raw
    scores, which is why they are recalibrated before being used as probabilities.
    """
    data = labelled(feat)
    data = data[data["month"] <= pd.Timestamp(last_train_month)]
    calib_start = month_offset(data["month"].max(), -(config.CALIBRATION_MONTHS - 1))
    train, calib = data[data["month"] < calib_start], data[data["month"] >= calib_start]

    y = train["churn_next_month"].astype(int)
    booster = xgb.XGBClassifier(scale_pos_weight=(y == 0).sum() / (y == 1).sum(), **config.XGB_PARAMS)
    booster.fit(prepare_features(train), y)

    calibrator = PlattCalibrator().fit(booster.predict_proba(prepare_features(calib))[:, 1],
                                       calib["churn_next_month"].astype(int))
    return ChurnModel(booster, calibrator)


def fit_logistic(train: pd.DataFrame):
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(prepare_features(train), train["churn_next_month"].astype(int))
    return model


def login_drop_score(df: pd.DataFrame) -> np.ndarray:
    """Rule-of-thumb baseline: the further logins fall below the customer's own average, the riskier."""
    return 1 - prepare_features(df)["logins_vs_baseline"].fillna(1).to_numpy()


def login_drop_flag(df: pd.DataFrame, threshold=0.7) -> np.ndarray:
    return (prepare_features(df)["logins_vs_baseline"].fillna(1) < threshold).to_numpy()


def evaluate(y, p, arr, top_frac=0.10) -> dict:
    y, p, arr = np.asarray(y), np.asarray(p), np.asarray(arr)
    top = np.argsort(-p)[: int(len(p) * top_frac)]
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, np.clip(p, 0, 1)),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "precision_top10": y[top].mean(),
        "recall_top10": y[top].sum() / y.sum(),
        "churned_arr_captured_top10": (y[top] * arr[top]).sum() / (y * arr).sum(),
        "predicted_churners": p.sum(),
        "actual_churners": y.sum(),
    }


def flag_metrics(y, flag) -> dict:
    y, flag = np.asarray(y), np.asarray(flag)
    return {"flagged_share": flag.mean(), "precision": y[flag].mean(), "recall": y[flag].sum() / y.sum()}


def capture_curve(y, p, arr, steps=50) -> pd.DataFrame:
    """Share of churners and churned ARR caught when contacting the top-k% riskiest accounts."""
    order = np.argsort(-np.asarray(p))
    y, arr = np.asarray(y)[order], np.asarray(arr)[order]
    cut = np.linspace(0, len(y), steps + 1).astype(int)[1:]
    return pd.DataFrame({
        "pct_targeted": cut / len(y),
        "churners_captured": np.cumsum(y)[cut - 1] / y.sum(),
        "churned_arr_captured": np.cumsum(y * arr)[cut - 1] / (y * arr).sum(),
    })


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    return df.groupby("bin").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"))


# --- Explanations -------------------------------------------------------------

REASON_TEMPLATES = {
    "logins_vs_baseline":  lambda r: f"Logins at {r.logins_vs_baseline:.0%} of usual",
    "login_trend":         lambda r: f"Logins {r.login_trend - 1:+.0%} vs prior quarter",
    "logins":              lambda r: f"Only {r.logins:.0f} logins this month",
    "rolling_3m_logins":   lambda r: f"Low 3-month usage ({r.rolling_3m_logins:.0f} logins/mo)",
    "rolling_6m_logins":   lambda r: f"Low 6-month usage ({r.rolling_6m_logins:.0f} logins/mo)",
    "support_tickets":     lambda r: f"{r.support_tickets:.0f} support tickets this month",
    "rolling_3m_tickets":  lambda r: f"Elevated tickets ({r.rolling_3m_tickets:.1f}/mo)",
    "months_to_renewal":   lambda r: "Renewal due next month" if r.months_to_renewal == 1 and r.contract_annual
                                     else "Month-to-month contract" if not r.contract_annual
                                     else f"Annual renewal in {r.months_to_renewal:.0f} months",
    "contract_annual":     lambda r: "Month-to-month contract" if not r.contract_annual
                                     else "Renewal due next month" if r.months_to_renewal == 1
                                     else f"Annual renewal in {r.months_to_renewal:.0f} months",
    "mrr_change_1m":       lambda r: f"MRR {r.mrr_change_1m:+.0%} last month",
    "mrr_change_3m":       lambda r: f"MRR {r.mrr_change_3m:+.0%} over 3 months",
    "downgraded_6m":       lambda r: "Downgraded plan in last 6 months",
    "tenure_months":       lambda r: f"New customer ({r.tenure_months:.0f} months)",
    "segment_enc":         lambda r: f"{r.segment} segment",
    "plan_enc":            lambda r: f"{r.plan_type} plan",
    "channel_paid":        lambda r: f"Acquired via {r.acquisition_channel}",
    "channel_partner":     lambda r: f"Acquired via {r.acquisition_channel}",
    "channel_outbound":    lambda r: f"Acquired via {r.acquisition_channel}",
    "mrr":                 lambda r: f"Small account (${r.mrr:,.0f} MRR)",
}


def shap_values(model: ChurnModel, df: pd.DataFrame) -> pd.DataFrame:
    """Per-row SHAP contributions (log-odds) from the boosted trees."""
    import shap

    X = prepare_features(df)
    values = shap.TreeExplainer(model.booster).shap_values(X)
    return pd.DataFrame(values, columns=config.FEATURES, index=df.index)


def top_reasons(df: pd.DataFrame, contributions: pd.DataFrame, n=3) -> pd.DataFrame:
    """Human-readable top-n risk drivers per row (features pushing churn risk up the most)."""
    context = prepare_features(df).join(df[["segment", "plan_type", "acquisition_channel"]])
    out = []
    for idx, row in contributions.iterrows():
        drivers = row[row > 0].sort_values(ascending=False).index
        reasons = list(dict.fromkeys(REASON_TEMPLATES[f](context.loc[idx]) for f in drivers))[:n]
        out.append(reasons + [None] * (n - len(reasons)))
    return pd.DataFrame(out, columns=[f"reason_{i + 1}" for i in range(n)], index=df.index)


# --- Forward projection ---------------------------------------------------------

def base_rates(feat: pd.DataFrame) -> pd.DataFrame:
    """Historical next-month churn rate by segment x contract x whether a renewal is due."""
    data = labelled(feat).assign(renewal_next=lambda d: (d["months_to_renewal"] == 1).astype(int))
    return (data.groupby(["segment_enc", "contract_annual", "renewal_next"])["churn_next_month"]
                .mean().rename("base_rate").reset_index())


def project_hazards(model: ChurnModel, snapshot: pd.DataFrame, rates: pd.DataFrame,
                    horizon=config.FORECAST_HORIZON, decay=config.SIGNAL_DECAY) -> np.ndarray:
    """Monthly churn hazard for each snapshot customer over the next `horizon` months -> (n_customers, horizon).

    Month 1 is the model's calibrated score. Later months roll tenure and renewal timing forward and blend
    the model score toward the historical base rate, because today's behavioural signal fades (at-risk
    accounts often recover) - holding it constant for a year would overstate churn.
    """
    hazards = np.zeros((len(snapshot), horizon))
    for h in range(horizon):
        future = snapshot.copy()
        future["tenure_months"] = snapshot["tenure_months"] + h
        future["months_to_renewal"] = np.where(
            future["contract_annual"] == 1, 12 - future["tenure_months"] % 12, 1)
        future["renewal_next"] = (future["months_to_renewal"] == 1).astype(int)
        p_base = future.merge(rates, on=["segment_enc", "contract_annual", "renewal_next"], how="left")["base_rate"]
        weight = decay ** h
        hazards[:, h] = weight * model.predict_proba(future) + (1 - weight) * p_base.fillna(0).to_numpy()
    return hazards
