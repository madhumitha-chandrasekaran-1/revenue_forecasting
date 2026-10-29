"""ARR forecasting: top-down Prophet, naive baselines, and a bottom-up Monte Carlo built on the churn model."""
import logging

import numpy as np
import pandas as pd

from . import config, modeling

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

MACRO_SHOCK_SD = 0.15   # churn shock shared by every customer within a simulated year (economy, pricing change, ...)


def monthly_arr(feat: pd.DataFrame) -> pd.DataFrame:
    return feat.groupby("month", as_index=False)["arr"].sum().rename(columns={"month": "ds", "arr": "y"})


def new_bookings(feat: pd.DataFrame) -> pd.DataFrame:
    """ARR added by customers in their first month."""
    first = feat[feat["tenure_months"] == 0]
    return first.groupby("month", as_index=False)["arr"].sum().rename(columns={"month": "ds", "arr": "y"})


def expansion_rates(feat: pd.DataFrame) -> dict:
    """Mean month-over-month ARR multiplier for retained customers, per segment (plus 'all' for new cohorts)."""
    df = feat.sort_values(["customer_id", "month"])
    prev = df.groupby("customer_id")["arr"].shift()
    ratio = (df["arr"] / prev).where(df["tenure_months"] > 0).clip(0.5, 2.0)
    rates = ratio.groupby(df["segment"]).mean().to_dict()
    rates["all"] = ratio.mean()
    return rates


def early_life_hazard(feat: pd.DataFrame) -> float:
    data = modeling.labelled(feat)
    return data.loc[data["tenure_months"] < 12, "churn_next_month"].mean()


def _prophet(series: pd.DataFrame, interval_width=0.8, uncertainty_samples=1000):
    from prophet import Prophet

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                    changepoint_prior_scale=0.05, interval_width=interval_width,
                    uncertainty_samples=uncertainty_samples)
    model.fit(series)
    return model


def prophet_forecast(series: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Top-down forecast of a monthly series; returns month, yhat and an 80% interval."""
    model = _prophet(series)
    future = model.make_future_dataframe(periods=horizon, freq="MS").tail(horizon)
    fc = model.predict(future)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "month"})


def new_bookings_paths(series: pd.DataFrame, horizon: int, n_sims: int, seed=config.SEED) -> np.ndarray:
    """Sampled future paths of monthly new-bookings ARR from Prophet -> (n_sims, horizon)."""
    np.random.seed(seed)  # Prophet samples with the global numpy RNG
    model = _prophet(series, uncertainty_samples=n_sims)
    future = model.make_future_dataframe(periods=horizon, freq="MS").tail(horizon)
    return np.clip(model.predictive_samples(future)["yhat"].T, 0, None)


def simulate_arr(snapshot: pd.DataFrame, hazards: np.ndarray, expansion: dict, new_paths: np.ndarray,
                 new_cohort_hazard: float, churn_mult=1.0, bookings_mult=1.0,
                 n_sims=config.N_SIMULATIONS, seed=config.SEED) -> dict:
    """Monte Carlo of month-end ARR for the existing book plus new cohorts.

    Each simulation draws one shared churn shock (so customers churn together in a bad year, which is what
    makes the interval honest) and then an independent churn month per customer. With survival S(h) and
    shock m, a customer is alive at h when U < S(h)^m, i.e. log(U)/m < log S(h) - evaluated without
    materialising an (n_sims x customers x months) array.
    """
    rng = np.random.default_rng(seed)
    horizon = hazards.shape[1]
    shock = np.exp(rng.normal(0, MACRO_SHOCK_SD, n_sims)) * churn_mult

    log_survival = np.cumsum(np.log1p(-np.clip(hazards, 0, 0.999)), axis=1)               # (n, H)
    growth = snapshot["segment"].map(expansion).to_numpy()[:, None] ** np.arange(1, horizon + 1)
    arr_path = snapshot["arr"].to_numpy()[:, None] * growth                                # (n, H)
    draw = np.log(rng.random((n_sims, len(snapshot)))) / shock[:, None]                    # (S, n)

    existing = np.empty((n_sims, horizon))
    for h in range(horizon):
        existing[:, h] = (draw < log_survival[:, h]).astype(np.float32) @ arr_path[:, h]

    new = np.zeros((n_sims, horizon))
    bookings = new_paths[:n_sims] * bookings_mult
    retain = (1 - np.clip(new_cohort_hazard * shock, 0, 1)) * expansion["all"]           # (S,)
    for k in range(horizon):
        for h in range(k, horizon):
            new[:, h] += bookings[:, k] * retain ** (h - k)

    no_churn = arr_path.sum(axis=0)
    return {"total": existing + new, "existing": existing, "new": new,
            "churned": no_churn[None, :] - existing}


def summarise(sim: dict, months) -> pd.DataFrame:
    total = sim["total"]
    return pd.DataFrame({
        "month": months,
        "arr_forecast": np.median(total, axis=0),
        "arr_lower": np.percentile(total, 10, axis=0),
        "arr_upper": np.percentile(total, 90, axis=0),
        "arr_mean": total.mean(axis=0),
        "arr_existing": sim["existing"].mean(axis=0),
        "arr_new": sim["new"].mean(axis=0),
        "arr_churned": sim["churned"].mean(axis=0),
    })


def bottom_up_forecast(feat: pd.DataFrame, as_of, horizon: int, model=None, scenarios=None,
                       n_sims=config.N_SIMULATIONS) -> pd.DataFrame:
    """Full bottom-up forecast as of a month, using only data observable at that month.

    If no model is supplied, one is trained on labels resolved by `as_of` (rows up to as_of - 1 month).
    """
    as_of = pd.Timestamp(as_of)
    history = feat[feat["month"] <= as_of]
    known = history[history["month"] < as_of]          # labels for these rows resolved by as_of
    model = model or modeling.fit_churn_model(known, modeling.month_offset(as_of, -1))

    snapshot = history[history["month"] == as_of].reset_index(drop=True)
    hazards = modeling.project_hazards(model, snapshot, modeling.base_rates(known), horizon)
    expansion = expansion_rates(history)
    paths = new_bookings_paths(new_bookings(history), horizon, n_sims)
    cohort_hazard = early_life_hazard(known)
    months = pd.date_range(modeling.month_offset(as_of, 1), periods=horizon, freq="MS")

    out = []
    for name, (churn_mult, bookings_mult) in (scenarios or {"Base": (1.0, 1.0)}).items():
        sim = simulate_arr(snapshot, hazards, expansion, paths, cohort_hazard,
                           churn_mult, bookings_mult, n_sims)
        out.append(summarise(sim, months).assign(scenario=name))
    return pd.concat(out, ignore_index=True)


def backtest(feat: pd.DataFrame, as_of, horizon=config.BACKTEST_HORIZON) -> pd.DataFrame:
    """Forecast `horizon` months from `as_of` with each method and line them up against actual ARR."""
    as_of = pd.Timestamp(as_of)
    hist = monthly_arr(feat)
    train = hist[hist["ds"] <= as_of]
    actual = hist[(hist["ds"] > as_of)].head(horizon).reset_index(drop=True)
    steps = np.arange(1, horizon + 1)

    last = train["y"].iloc[-1]
    drift = (last - train["y"].iloc[-13]) / 12
    prophet = prophet_forecast(train, horizon)
    bottom_up = bottom_up_forecast(feat, as_of, horizon)

    return pd.DataFrame({
        "month": actual["ds"],
        "actual": actual["y"],
        "naive_last_value": last,
        "linear_drift": last + drift * steps,
        "prophet_top_down": prophet["yhat"].to_numpy(),
        "bottom_up_p50": bottom_up["arr_forecast"].to_numpy(),
        "bottom_up_p10": bottom_up["arr_lower"].to_numpy(),
        "bottom_up_p90": bottom_up["arr_upper"].to_numpy(),
        "prophet_lower": prophet["yhat_lower"].to_numpy(),
        "prophet_upper": prophet["yhat_upper"].to_numpy(),
    })


def backtest_scores(bt: pd.DataFrame) -> pd.DataFrame:
    methods = ["naive_last_value", "linear_drift", "prophet_top_down", "bottom_up_p50"]
    rows = []
    for m in methods:
        err = (bt[m] - bt["actual"]) / bt["actual"]
        rows.append({"method": m, "mape": err.abs().mean(), "bias": err.mean(),
                     "final_month_error": err.iloc[-1]})
    scores = pd.DataFrame(rows)
    scores["interval_coverage"] = np.nan
    scores.loc[scores["method"] == "bottom_up_p50", "interval_coverage"] = (
        bt["actual"].between(bt["bottom_up_p10"], bt["bottom_up_p90"]).mean())
    scores.loc[scores["method"] == "prophet_top_down", "interval_coverage"] = (
        bt["actual"].between(bt["prophet_lower"], bt["prophet_upper"]).mean())
    return scores
