"""Turn churn scores into a retention plan: who to contact, and whether it pays for itself."""
import numpy as np
import pandas as pd

from . import config

STRATEGIES = {
    "ARR at risk (prob x ARR)": "revenue_risk",
    "Churn probability":        "churn_probability",
    "Largest ARR":              "arr",
}


def targeting_curve(risk: pd.DataFrame, cost=config.OUTREACH_COST, save_rate=config.SAVE_RATE,
                    seed=config.SEED) -> pd.DataFrame:
    """Expected first-year ARR saved vs outreach cost when contacting the top-N accounts under each ranking.

    `churn_probability` is the 12-month probability, so expected loss = P(churn within a year) x ARR.
    """
    expected_loss = (risk["churn_probability"] * risk["arr"]).to_numpy()
    orders = {name: np.argsort(-risk[col].to_numpy()) for name, col in STRATEGIES.items()}
    orders["Random"] = np.random.default_rng(seed).permutation(len(risk))

    n = np.arange(1, len(risk) + 1)
    frames = []
    for name, order in orders.items():
        saved = save_rate * np.cumsum(expected_loss[order])
        frames.append(pd.DataFrame({
            "strategy": name, "accounts_contacted": n,
            "expected_arr_saved": saved, "outreach_cost": n * cost,
            "net_value": saved - n * cost,
        }))
    return pd.concat(frames, ignore_index=True)


def best_plan(curve: pd.DataFrame) -> pd.DataFrame:
    best = curve.loc[curve.groupby("strategy")["net_value"].idxmax()].copy()
    best["roi"] = best["net_value"] / best["outreach_cost"]
    return best.sort_values("net_value", ascending=False).reset_index(drop=True)


def sensitivity(risk: pd.DataFrame, save_rates=(0.10, 0.15, 0.25, 0.35), costs=(200, 400, 800)) -> pd.DataFrame:
    """Optimal plan under the 'ARR at risk' ranking for a grid of save-rate and cost assumptions."""
    rows = []
    for s in save_rates:
        for c in costs:
            curve = targeting_curve(risk, cost=c, save_rate=s)
            top = best_plan(curve[curve["strategy"] == "ARR at risk (prob x ARR)"]).iloc[0]
            rows.append({"save_rate": s, "cost_per_account": c,
                         "accounts_to_contact": int(top["accounts_contacted"]),
                         "expected_arr_saved": top["expected_arr_saved"], "net_value": top["net_value"]})
    return pd.DataFrame(rows)
