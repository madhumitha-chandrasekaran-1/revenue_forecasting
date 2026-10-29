import numpy as np
import pandas as pd

from conftest import register_silver, truncate
from revenue_pipeline import config, features

FEATURE_COLUMNS = [c for c in config.FEATURES] + ["arr", "plan_type"]


def test_label_fires_only_on_last_paying_month(spark, simulated, to_pandas):
    customers, mrr, activity = simulated
    register_silver(spark, customers, mrr, activity)
    gold = to_pandas(features.build_gold_features(spark))

    last_month = gold["month"].max()
    assert gold.loc[gold["month"] == last_month, "churn_next_month"].isna().all()

    positives = gold[gold["churn_next_month"] == 1]
    churn = pd.to_datetime(customers.set_index("customer_id")["churn_date"])
    assert (positives["month"] + pd.DateOffset(months=1)
            == churn.loc[positives["customer_id"]].to_numpy()).all()
    assert positives["customer_id"].is_unique
    assert len(positives) == churn.notna().sum()


def test_features_are_point_in_time(spark, simulated, to_pandas):
    """Features for month t must not change when everything after t is deleted from the source data."""
    customers, mrr, activity = simulated
    register_silver(spark, customers, mrr, activity)
    full = to_pandas(features.build_gold_features(spark))

    as_of = "2023-06-01"
    register_silver(spark, *truncate(customers, mrr, activity, as_of))
    past = to_pandas(features.build_gold_features(spark))

    full_past = full[full["month"] <= as_of].reset_index(drop=True)
    assert len(past) == len(full_past)
    pd.testing.assert_frame_equal(past[["customer_id", "month"] + FEATURE_COLUMNS],
                                  full_past[["customer_id", "month"] + FEATURE_COLUMNS], check_dtype=False)


def test_renewal_countdown(spark, simulated, to_pandas):
    customers, mrr, activity = simulated
    register_silver(spark, customers, mrr, activity)
    gold = to_pandas(features.build_gold_features(spark))

    annual = gold[gold["contract_annual"] == 1]
    assert annual["months_to_renewal"].between(1, 12).all()
    assert np.array_equal(annual["months_to_renewal"], 12 - annual["tenure_months"] % 12)
    assert (gold.loc[gold["contract_annual"] == 0, "months_to_renewal"] == 1).all()
