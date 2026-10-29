import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from revenue_pipeline import datagen  # noqa: E402

N_TEST_CUSTOMERS = 400


@pytest.fixture(scope="session")
def spark():
    """Plain local Spark (no Delta, no metastore) - the tests only need the SQL engine."""
    from pyspark.sql import SparkSession

    session = (SparkSession.builder.master("local[2]").appName("revenue_pipeline_tests")
               .config("spark.sql.shuffle.partitions", "4")
               .config("spark.sql.session.timeZone", "UTC")
               .config("spark.sql.execution.arrow.pyspark.enabled", "true")
               .config("spark.ui.enabled", "false")
               .getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def simulated():
    return datagen.simulate(n_customers=N_TEST_CUSTOMERS, seed=7)


def register_silver(spark, customers, mrr, activity):
    """Expose pandas frames as the silver_* views the feature SQL reads, with silver's column types."""
    spark.createDataFrame(customers).selectExpr(
        "CAST(customer_id AS INT) AS customer_id", "CAST(signup_date AS DATE) AS signup_date",
        "CAST(churn_date AS DATE) AS churn_date", "segment", "initial_plan", "acquisition_channel", "contract_type",
    ).createOrReplaceTempView("silver_customers")
    spark.createDataFrame(mrr).selectExpr(
        "CAST(customer_id AS INT) AS customer_id", "CAST(month AS DATE) AS month", "plan_type", "CAST(mrr AS DOUBLE) AS mrr",
    ).createOrReplaceTempView("silver_mrr")
    spark.createDataFrame(activity).selectExpr(
        "CAST(customer_id AS INT) AS customer_id", "CAST(month AS DATE) AS month",
        "CAST(logins AS INT) AS logins", "CAST(support_tickets AS INT) AS support_tickets",
    ).createOrReplaceTempView("silver_activity")


def truncate(customers, mrr, activity, as_of: str):
    """The world as it looked at `as_of`: later rows don't exist yet and later churn isn't known."""
    customers = customers[customers["signup_date"] <= as_of].copy()
    customers.loc[customers["churn_date"] > as_of, "churn_date"] = None
    return customers, mrr[mrr["month"] <= as_of], activity[activity["month"] <= as_of]


@pytest.fixture
def to_pandas():
    def convert(df):
        out = df.toPandas()
        out["month"] = pd.to_datetime(out["month"])
        return out.sort_values(["customer_id", "month"]).reset_index(drop=True)
    return convert
