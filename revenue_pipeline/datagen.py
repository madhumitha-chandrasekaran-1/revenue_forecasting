"""Synthetic SaaS book of business.

Churn is driven by a hidden health state rather than a hard-coded countdown, so behavioural
signals are informative but imperfect:
  * customers drift into an "at-risk" state (logins fall, tickets rise, contractions) and
    often recover without churning -> false positives for any early-warning model;
  * healthy customers have random usage dips (holidays, re-orgs) -> more false positives;
  * healthy customers can still churn with no warning (budget cuts, acquisitions) -> silent churn;
  * annual contracts can only churn at renewal.

Conventions: `month` is the first day of a month. `churn_date` is the first month with no
revenue, so a customer's last row is the month before `churn_date`.
"""
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from . import config

BASE_HAZARD = {"Enterprise": 0.0036, "Mid-Market": 0.0076, "SMB": 0.0125}  # healthy, monthly
P_DECLINE = {"Enterprise": 0.015, "Mid-Market": 0.025, "SMB": 0.035}       # healthy -> at-risk
P_RECOVER = 0.25                                                           # at-risk -> healthy
AT_RISK_HAZARD_MULT = 6.0
EARLY_LIFE_HAZARD_MULT = 1.3   # first 6 months

BASE_LOGINS = {"Enterprise": 45, "Mid-Market": 25, "SMB": 12}
BASE_TICKETS = {"Enterprise": 1.5, "Mid-Market": 1.0, "SMB": 0.5}
PLAN_USAGE = {"Basic": 0.8, "Pro": 1.0, "Enterprise": 1.2}
LOGIN_SEASONALITY = [1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.92, 0.9, 1.0, 1.0, 0.97, 0.85]
SIGNUP_SEASONALITY = [0.85, 0.9, 1.0, 1.0, 1.0, 0.95, 0.9, 0.85, 1.05, 1.15, 1.15, 1.1]
P_EXPAND = {"Enterprise": 0.04, "Mid-Market": 0.03, "SMB": 0.015}


def _signup_months(rng, n_customers, n_months):
    growth = 1 + 0.04 * np.arange(n_months)
    season = np.array([SIGNUP_SEASONALITY[(config.START_DATE.month - 1 + m) % 12] for m in range(n_months)])
    weights = growth * season
    return np.sort(rng.choice(n_months, size=n_customers, p=weights / weights.sum()))


def _login_multiplier(at_risk, months_at_risk, rng):
    if at_risk:
        return max(0.35, 0.85 - 0.15 * months_at_risk) * rng.uniform(0.8, 1.2)
    if rng.random() < 0.06:                     # temporary dip unrelated to churn
        return rng.uniform(0.4, 0.75)
    return 1.0


def simulate(n_customers=config.N_CUSTOMERS, n_months=config.N_MONTHS, seed=config.SEED):
    """Simulate clean customers / MRR / activity tables as pandas DataFrames."""
    rng = np.random.default_rng(seed)
    months = [config.START_DATE + relativedelta(months=m) for m in range(n_months)]

    segments = rng.choice(config.SEGMENTS, size=n_customers, p=config.SEG_DIST)
    signup_idx = _signup_months(rng, n_customers, n_months)

    customers, mrr_rows, activity_rows = [], [], []
    for cid in range(1, n_customers + 1):
        seg = str(segments[cid - 1])
        start = int(signup_idx[cid - 1])
        plan_idx = int(rng.choice(3, p=config.PLAN_MIX[seg]))
        channel = str(rng.choice(config.CHANNELS, p=config.CHANNEL_MIX[seg]))
        annual = rng.random() < config.ANNUAL_CONTRACT_SHARE[seg]
        propensity = rng.lognormal(0, 0.35)
        mrr = float(config.PLAN_MRR[config.PLANS[plan_idx]])
        initial_plan = config.PLANS[plan_idx]

        at_risk, months_at_risk, churn_idx = False, 0, None
        for m in range(start, n_months):
            tenure = m - start
            if m > start:
                # Churn decision for month m uses the state observed in month m-1.
                if not annual or tenure % 12 == 0:
                    h = BASE_HAZARD[seg] * config.CHANNEL_CHURN_MULT[channel]
                    h *= AT_RISK_HAZARD_MULT if at_risk else 1.0
                    h *= EARLY_LIFE_HAZARD_MULT if tenure <= 6 else 1.0
                    p_churn = 1 - (1 - min(h, 0.95)) ** 12 if annual else h
                    if rng.random() < p_churn:
                        churn_idx = m
                        break

                if at_risk:
                    if rng.random() < P_RECOVER:
                        at_risk, months_at_risk = False, 0
                    else:
                        months_at_risk += 1
                elif rng.random() < P_DECLINE[seg]:
                    at_risk, months_at_risk = True, 1

                if not at_risk and rng.random() < P_EXPAND[seg]:
                    mrr *= rng.uniform(1.05, 1.20)
                if rng.random() < (0.05 if at_risk else 0.005):
                    mrr *= rng.uniform(0.75, 0.95)
                if not at_risk and plan_idx < 2 and rng.random() < 0.004:
                    plan_idx += 1
                    mrr *= config.PLAN_MRR[config.PLANS[plan_idx]] / config.PLAN_MRR[config.PLANS[plan_idx - 1]]
                elif at_risk and plan_idx > 0 and rng.random() < 0.03:
                    plan_idx -= 1
                    mrr *= config.PLAN_MRR[config.PLANS[plan_idx]] / config.PLAN_MRR[config.PLANS[plan_idx + 1]]

            month = months[m]
            login_rate = (BASE_LOGINS[seg] * PLAN_USAGE[config.PLANS[plan_idx]] * propensity
                          * LOGIN_SEASONALITY[month.month - 1] * _login_multiplier(at_risk, months_at_risk, rng))
            ticket_rate = BASE_TICKETS[seg] * (1.8 if at_risk else 1.0) + (3.0 if rng.random() < 0.04 else 0.0)

            mrr_rows.append((cid, str(month), config.PLANS[plan_idx], round(mrr, 2)))
            activity_rows.append((cid, str(month), int(rng.poisson(login_rate)), int(rng.poisson(ticket_rate))))

        customers.append((
            cid, str(months[start]), seg, initial_plan, channel,
            "Annual" if annual else "Monthly",
            str(months[churn_idx]) if churn_idx is not None else None,
        ))

    customers = pd.DataFrame(customers, columns=[
        "customer_id", "signup_date", "segment", "initial_plan", "acquisition_channel",
        "contract_type", "churn_date"])
    mrr = pd.DataFrame(mrr_rows, columns=["customer_id", "month", "plan_type", "mrr"])
    activity = pd.DataFrame(activity_rows, columns=["customer_id", "month", "logins", "support_tickets"])
    return customers, mrr, activity


def add_raw_defects(mrr, activity, seed=config.SEED, dup_rate=0.003, null_rate=0.001):
    """Make the 'raw' feeds look like real ingests: duplicate rows, null amounts, sentinel values."""
    rng = np.random.default_rng(seed + 1)
    mrr, activity = mrr.copy(), activity.copy()
    mrr["mrr"] = mrr["mrr"].astype("float64")
    mrr.loc[rng.random(len(mrr)) < null_rate, "mrr"] = np.nan
    activity.loc[rng.random(len(activity)) < null_rate, "logins"] = -1
    mrr = pd.concat([mrr, mrr.sample(frac=dup_rate, random_state=seed)], ignore_index=True)
    activity = pd.concat([activity, activity.sample(frac=dup_rate, random_state=seed)], ignore_index=True)
    return mrr, activity


def realised_annual_churn(customers, months=12):
    """Share of customers active at each month start who churned within the next 12 months, by segment."""
    c = customers.assign(
        signup=pd.to_datetime(customers["signup_date"]),
        churn=pd.to_datetime(customers["churn_date"]),
    )
    end = pd.Timestamp(config.START_DATE) + pd.DateOffset(months=config.N_MONTHS)
    rows = []
    for start in pd.date_range(config.START_DATE, end - pd.DateOffset(months=months), freq="MS"):
        window_end = start + pd.DateOffset(months=months)
        active = c[(c["signup"] <= start) & (c["churn"].isna() | (c["churn"] > start))]
        churned = active["churn"].notna() & (active["churn"] <= window_end)
        rows.append(active.assign(churned=churned)[["segment", "churned"]])
    return pd.concat(rows).groupby("segment")["churned"].mean()
