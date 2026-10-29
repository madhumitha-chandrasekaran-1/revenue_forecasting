"""Data-quality checks for the silver layer. Each check is a SQL count of violating rows."""
import pandas as pd

SILVER_CHECKS = [
    ("customers: unique customer_id",
     "SELECT COUNT(*) - COUNT(DISTINCT customer_id) FROM silver_customers"),
    ("customers: churn after signup",
     "SELECT COUNT(*) FROM silver_customers WHERE churn_date IS NOT NULL AND churn_date <= signup_date"),
    ("mrr: unique (customer_id, month)",
     "SELECT COUNT(*) FROM (SELECT customer_id, month FROM silver_mrr GROUP BY 1, 2 HAVING COUNT(*) > 1)"),
    ("mrr: positive amounts",
     "SELECT COUNT(*) FROM silver_mrr WHERE mrr IS NULL OR mrr <= 0"),
    ("mrr: known customer",
     "SELECT COUNT(*) FROM silver_mrr m LEFT ANTI JOIN silver_customers c USING (customer_id)"),
    ("mrr: no revenue before signup or after churn",
     """SELECT COUNT(*) FROM silver_mrr m JOIN silver_customers c USING (customer_id)
        WHERE m.month < c.signup_date OR (c.churn_date IS NOT NULL AND m.month >= c.churn_date)"""),
    ("activity: unique (customer_id, month)",
     "SELECT COUNT(*) FROM (SELECT customer_id, month FROM silver_activity GROUP BY 1, 2 HAVING COUNT(*) > 1)"),
    ("activity: non-negative counts",
     "SELECT COUNT(*) FROM silver_activity WHERE logins < 0 OR support_tickets < 0"),
]


def run_checks(spark, checks=SILVER_CHECKS, raise_on_failure=True) -> pd.DataFrame:
    results = pd.DataFrame(
        [(name, int(spark.sql(sql).first()[0])) for name, sql in checks],
        columns=["check", "violations"],
    )
    results["passed"] = results["violations"] == 0
    if raise_on_failure and not results["passed"].all():
        failed = results.loc[~results["passed"], "check"].tolist()
        raise AssertionError(f"Data-quality checks failed: {failed}")
    return results
