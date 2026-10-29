"""Gold feature table: one row per customer-month, every feature computed from data at or before that month.

Label: churn_next_month = 1 when the customer has no revenue in the following month. Rows in the
final observed month have a NULL label (the outcome is not known yet) and are the rows we score.
"""

GOLD_FEATURES_SQL = """
WITH base AS (
    SELECT m.customer_id, m.month, m.plan_type, m.mrr,
           a.logins, a.support_tickets,
           c.segment, c.acquisition_channel, c.contract_type, c.signup_date, c.churn_date,
           CASE m.plan_type WHEN 'Basic' THEN 0 WHEN 'Pro' THEN 1 WHEN 'Enterprise' THEN 2 END AS plan_enc
    FROM silver_mrr m
    JOIN silver_activity a USING (customer_id, month)
    JOIN silver_customers c USING (customer_id)
),
windowed AS (
    SELECT *,
        CAST(months_between(month, signup_date) AS INT)                       AS tenure_months,
        AVG(logins)          OVER (PARTITION BY customer_id ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS rolling_3m_logins,
        AVG(logins)          OVER (PARTITION BY customer_id ORDER BY month ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS rolling_6m_logins,
        AVG(logins)          OVER (PARTITION BY customer_id ORDER BY month ROWS BETWEEN 5 PRECEDING AND 3 PRECEDING) AS prior_3m_logins,
        AVG(logins)          OVER (PARTITION BY customer_id ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS baseline_logins,
        AVG(support_tickets) OVER (PARTITION BY customer_id ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS rolling_3m_tickets,
        LAG(mrr, 1)          OVER (PARTITION BY customer_id ORDER BY month)    AS mrr_lag_1m,
        LAG(mrr, 3)          OVER (PARTITION BY customer_id ORDER BY month)    AS mrr_lag_3m,
        plan_enc - LAG(plan_enc) OVER (PARTITION BY customer_id ORDER BY month) AS plan_change
    FROM base
)
SELECT
    customer_id, month, segment, plan_type, acquisition_channel, contract_type,
    mrr, mrr * 12                                                            AS arr,
    tenure_months,
    CASE WHEN contract_type = 'Annual' THEN 12 - tenure_months % 12 ELSE 1 END AS months_to_renewal,
    CASE WHEN contract_type = 'Annual' THEN 1 ELSE 0 END                     AS contract_annual,
    CASE segment WHEN 'SMB' THEN 0 WHEN 'Mid-Market' THEN 1 WHEN 'Enterprise' THEN 2 END AS segment_enc,
    plan_enc,
    CASE WHEN acquisition_channel = 'Paid Search'    THEN 1 ELSE 0 END       AS channel_paid,
    CASE WHEN acquisition_channel = 'Partner'        THEN 1 ELSE 0 END       AS channel_partner,
    CASE WHEN acquisition_channel = 'Outbound Sales' THEN 1 ELSE 0 END       AS channel_outbound,
    logins, rolling_3m_logins, rolling_6m_logins,
    rolling_3m_logins / NULLIF(prior_3m_logins, 0)                           AS login_trend,
    logins / NULLIF(baseline_logins, 0)                                      AS logins_vs_baseline,
    support_tickets, rolling_3m_tickets,
    COALESCE(mrr / NULLIF(mrr_lag_1m, 0) - 1, 0)                             AS mrr_change_1m,
    COALESCE(mrr / NULLIF(mrr_lag_3m, 0) - 1, 0)                             AS mrr_change_3m,
    MAX(CASE WHEN plan_change < 0 THEN 1 ELSE 0 END)
        OVER (PARTITION BY customer_id ORDER BY month ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS downgraded_6m,
    CASE
        WHEN month >= (SELECT MAX(month) FROM silver_mrr) THEN NULL
        WHEN churn_date = ADD_MONTHS(month, 1)            THEN 1
        ELSE 0
    END                                                                      AS churn_next_month
FROM windowed
"""


def build_gold_features(spark):
    return spark.sql(GOLD_FEATURES_SQL)
