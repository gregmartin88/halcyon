"""Check that ZIP tiers 4-5 are over-routed to Queue B after controlling for
claim amount and vehicle age.

Runs (1) raw Queue B rates by tier, (2) a stratified comparison within
claim-amount x vehicle-age cells, and (3) a logistic regression (numpy IRLS).

Usage:
    python verify_bias.py [data/claims.csv]
"""

import sys

import numpy as np
import pandas as pd


def logit_fit(X, y, iters=50):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return beta, se


def main(path):
    df = pd.read_csv(path, parse_dates=["claim_date"])
    df["queue_b"] = (df["adjuster_queue"] == "B").astype(int)
    df["high_tier"] = df["zip_tier"] >= 4
    print(f"{len(df):,} rows, {df.claim_date.min().date()} to {df.claim_date.max().date()}\n")

    print("1) Raw Queue B rate by ZIP tier")
    print(df.groupby("zip_tier")["queue_b"].agg(rate="mean", n="size").round(3).to_string(), "\n")

    print("2) Stratified: Queue B rate within amount-quintile x vehicle-age-band cells")
    df["amt_q"] = pd.qcut(df["claim_amount"], 5, labels=False)
    df["veh_band"] = pd.cut(df["vehicle_age"], [-1, 3, 7, 12, 99], labels=False)
    cell = df.groupby(["amt_q", "veh_band", "high_tier"])["queue_b"].agg(["mean", "size"]).unstack("high_tier")
    cell = cell.dropna()
    w = cell[("size", True)] + cell[("size", False)]
    diff = cell[("mean", True)] - cell[("mean", False)]
    print(f"   cells: {len(cell)}, tiers 4-5 higher in {(diff > 0).sum()} of them")
    print(f"   size-weighted gap (tiers 4-5 minus 1-3): {np.average(diff, weights=w):+.3f}\n")

    print("3) Logistic regression: queue_b ~ log(amount) + vehicle_age + tier dummies (ref = tier 1)")
    X = pd.DataFrame({
        "intercept": 1.0,
        "log_amount": np.log(df["claim_amount"]),
        "vehicle_age": df["vehicle_age"],
    })
    for t in (2, 3, 4, 5):
        X[f"tier_{t}"] = (df["zip_tier"] == t).astype(float)
    beta, se = logit_fit(X.to_numpy(float), df["queue_b"].to_numpy(float))
    res = pd.DataFrame({"coef": beta, "se": se, "z": beta / se, "odds_ratio": np.exp(beta)}, index=X.columns)
    print(res.round(3).to_string(), "\n")

    print("   + policy tenure, prior claims, claim type (fuller controls)")
    X["tenure_lt_12m"] = (df["policy_tenure_months"] < 12).astype(float)
    X["prior_claims"] = df["prior_claims_3yr"]
    X["liability"] = (df["claim_type"] == "liability").astype(float)
    beta, se = logit_fit(X.to_numpy(float), df["queue_b"].to_numpy(float))
    res = pd.DataFrame({"coef": beta, "se": se, "z": beta / se, "odds_ratio": np.exp(beta)}, index=X.columns)
    print(res.loc[[f"tier_{t}" for t in (2, 3, 4, 5)]].round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/claims.csv")
