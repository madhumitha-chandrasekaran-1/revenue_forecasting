import numpy as np
import pandas as pd

from revenue_pipeline import forecasting, modeling, retention


def test_platt_calibration_preserves_ranking_and_fixes_scale():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.001, 0.05, 20_000)
    y = rng.random(20_000) < true_p
    inflated = 1 / (1 + (1 - true_p) / (true_p * 50))      # odds x50, as class weighting does
    calibrated = modeling.PlattCalibrator().fit(inflated, y).predict(inflated)

    assert np.array_equal(np.argsort(inflated), np.argsort(calibrated))
    assert abs(calibrated.sum() - y.sum()) / y.sum() < 0.1
    assert inflated.sum() > 5 * y.sum()


def _snapshot(n=50):
    return pd.DataFrame({"segment": ["SMB"] * n, "arr": np.full(n, 1_000.0)})


def test_simulation_without_churn_or_growth_keeps_arr_flat():
    snap = _snapshot()
    sim = forecasting.simulate_arr(snap, np.zeros((len(snap), 6)), {"SMB": 1.0, "all": 1.0},
                                   np.zeros((100, 6)), new_cohort_hazard=0.0, n_sims=100)
    assert np.allclose(sim["total"], snap["arr"].sum())


def test_certain_churn_empties_the_existing_book():
    snap = _snapshot()
    sim = forecasting.simulate_arr(snap, np.ones((len(snap), 6)), {"SMB": 1.0, "all": 1.0},
                                   np.zeros((100, 6)), new_cohort_hazard=0.0, n_sims=100)
    # Hazards are capped at 0.999, so a stray survivor in month 1 is possible; none by month 6.
    assert sim["existing"][:, 0].mean() < 0.01 * snap["arr"].sum()
    assert np.allclose(sim["existing"][:, -1], 0)


def test_scenario_multipliers_order_outcomes():
    snap = _snapshot(500)
    hazards = np.full((len(snap), 12), 0.02)
    paths = np.full((300, 12), 5_000.0)
    p50 = {}
    for name, (churn_mult, bookings_mult) in {"Bear": (1.2, 0.9), "Base": (1, 1), "Bull": (0.8, 1.1)}.items():
        sim = forecasting.simulate_arr(snap, hazards, {"SMB": 1.0, "all": 1.0}, paths, 0.02,
                                       churn_mult, bookings_mult, n_sims=300)
        p50[name] = np.median(sim["total"][:, -1])
    assert p50["Bear"] < p50["Base"] < p50["Bull"]


def test_targeting_by_expected_loss_beats_random():
    rng = np.random.default_rng(1)
    risk = pd.DataFrame({"churn_probability": rng.beta(1, 40, 2_000), "arr": rng.lognormal(8, 1, 2_000)})
    risk["revenue_risk"] = risk["churn_probability"] * risk["arr"]
    plan = retention.best_plan(retention.targeting_curve(risk, cost=50, save_rate=0.25)).set_index("strategy")
    assert plan.loc["ARR at risk (prob x ARR)", "net_value"] >= plan["net_value"].max() - 1e-6
    assert plan.loc["ARR at risk (prob x ARR)", "net_value"] > plan.loc["Random", "net_value"]
