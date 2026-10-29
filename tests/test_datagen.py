import pandas as pd

from revenue_pipeline import datagen


def test_simulation_is_deterministic(simulated):
    again = datagen.simulate(n_customers=len(simulated[0]), seed=7)
    for first, second in zip(simulated, again):
        pd.testing.assert_frame_equal(first, second)


def test_revenue_only_between_signup_and_churn(simulated):
    customers, mrr, _ = simulated
    rows = mrr.merge(customers, on="customer_id")
    assert (rows["month"] >= rows["signup_date"]).all()
    churned = rows["churn_date"].notna()
    assert (rows.loc[churned, "month"] < rows.loc[churned, "churn_date"]).all()


def test_churners_pay_through_the_month_before_churn(simulated):
    customers, mrr, _ = simulated
    last_paid = mrr.groupby("customer_id")["month"].max()
    churners = customers.dropna(subset=["churn_date"]).set_index("customer_id")
    expected = (pd.to_datetime(churners["churn_date"]) - pd.DateOffset(months=1)).dt.strftime("%Y-%m-%d")
    assert (last_paid.loc[churners.index] == expected).all()


def test_annual_contracts_only_churn_at_renewal(simulated):
    customers, _, _ = simulated
    annual = customers[(customers["contract_type"] == "Annual") & customers["churn_date"].notna()]
    tenure = ((pd.to_datetime(annual["churn_date"]).dt.to_period("M")
               - pd.to_datetime(annual["signup_date"]).dt.to_period("M")).apply(lambda d: d.n))
    assert len(annual) > 0 and (tenure % 12 == 0).all()


def test_raw_defects_are_injected(simulated):
    _, mrr, activity = simulated
    mrr_raw, activity_raw = datagen.add_raw_defects(mrr, activity)
    assert len(mrr_raw) > len(mrr) and mrr_raw["mrr"].isna().any()
    assert (activity_raw["logins"] < 0).any()
