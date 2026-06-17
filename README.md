# Revenue Forecasting & Churn Risk Pipeline

End-to-end SaaS revenue analytics pipeline built on **Databricks** (PySpark + Delta Lake) with a **Tableau** dashboard. Simulates 10,000 customers over 36 months, predicts churn risk using XGBoost, and projects ARR for the next 12 months using Prophet.

---

## Project Structure

```
revenue_forecasting/
├── RevenueForecastingNB.ipynb   # Main Databricks notebook
├── Dashboard.twb                # Tableau workbook
├── gold_arr_forecast.csv        # Prophet ARR forecast (historical + 12-month)
├── gold_monthly_arr.csv         # Monthly ARR history
├── gold_revenue_risk.csv        # Per-customer churn probability & ARR at risk
├── gold_risk_segment.csv        # Risk aggregated by segment
├── gold_risk_plan.csv           # Risk aggregated by plan type
├── gold_scenario_forecast.csv   # Bear/Base/Bull scenario forecasts
└── gold_scenario_risk.csv       # Scenario risk per customer
```

---

## Pipeline Overview

### Data Generation
Synthetic customer data is generated with realistic SaaS churn dynamics:
- **10,000 customers** across Enterprise (10%), Mid-Market (25%), and SMB (65%) segments
- Three plan tiers: Basic ($99/mo), Pro ($299/mo), Enterprise ($999/mo)
- Annual churn rates: Enterprise 5%, Mid-Market 12%, SMB 22%
- MRR drift (+0.2%/month with noise) and behavioral signals (logins, support tickets) that degrade pre-churn

### Medallion Architecture (Delta Lake)

| Layer | Tables | Description |
|-------|--------|-------------|
| Bronze | `bronze_customers`, `bronze_mrr`, `bronze_activity` | Raw ingested data |
| Silver | `silver_customers`, `silver_mrr`, `silver_activity` | Cleaned, typed |
| Gold | `gold_customer_features`, `gold_revenue_risk`, `gold_arr_forecast`, `gold_scenario_*`, `gold_risk_by_*` | Analytics-ready |

### Churn Prediction (XGBoost)
Features engineered per customer-month:
- MRR, ARR, MRR growth rate
- Tenure months
- Logins (raw + 3-month and 6-month rolling averages)
- Support tickets (raw + 3-month rolling average)
- Segment and plan encodings

Trained with temporal split (last 3 months held out).

**Test AUC: 0.9855**

Top features by importance: `logins`, `rolling_ticket_3m`, `segment_enc`

### ARR Forecasting (Prophet)
- Fits on 36 months of historical monthly ARR
- Projects 12 months forward with yearly seasonality
- Forecast range: ~$34.7M (Jan 2025) to ~$43.2M (Dec 2025)

### Scenario Analysis
Three scenarios applied to both risk and forecast:

| Scenario | Churn Adjustment | Dec 2025 ARR Forecast |
|----------|-----------------|----------------------|
| Bear | +10% churn | ~$41.0M |
| Base | No change | ~$43.2M |
| Bull | -10% churn | ~$45.3M |

### Current Risk Snapshot (Dec 2024)
- Total ARR: **$33.2M**
- ARR at Risk: **$2.0M (6.1%)**
- SMB drives 81% of at-risk revenue despite lower ARR per customer

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Databricks / PySpark | Distributed data processing |
| Delta Lake | Lakehouse storage with ACID transactions |
| XGBoost | Churn probability scoring |
| Prophet | Time-series ARR forecasting |
| pandas / numpy | Local data manipulation |
| Tableau | Dashboard & visualization |

---

## Running the Notebook

1. Upload `RevenueForecastingNB.ipynb` to a Databricks workspace
2. Attach to a cluster with Unity Catalog or a schema with write access
3. Run all cells top to bottom — each cell prints confirmation output
4. Export the gold CSV files and connect `Dashboard.twb` in Tableau Desktop

Dependencies installed inline:
```
%pip install xgboost prophet python-dateutil
```
