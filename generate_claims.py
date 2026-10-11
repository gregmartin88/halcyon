"""Generate a synthetic auto-insurance claims dataset with an injected proxy bias.

The dataset is intended as a test fixture for fairness / disparate-impact
auditing. Routing to Queue B (Special Investigation / High-Scrutiny) is driven
by legitimate risk factors, plus a deliberately injected penalty for claims
from ZIP-code tiers 4 and 5 that is NOT explained by claim amount, vehicle age,
or any other column. The ground-truth bias parameters are recorded in
`BIAS_LOGIT` below and in `data/claims_metadata.json`.

Usage:
    python generate_claims.py [--rows 6000] [--seed 42] [--out data/claims.csv]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-06-30")

# Share of claims by ZIP-code tier (1 = lowest-risk proxy, 5 = highest).
ZIP_TIER_WEIGHTS = {1: 0.20, 2: 0.24, 3: 0.24, 4: 0.18, 5: 0.14}

# Synthetic 3-digit ZIP prefixes per tier (fictional; not real geography).
ZIP3_BY_TIER = {
    1: ["101", "104", "117", "122", "135"],
    2: ["203", "211", "226", "238", "249"],
    3: ["305", "318", "327", "336", "342"],
    4: ["407", "414", "429", "433", "448"],
    5: ["502", "516", "521", "537", "544"],
}

CLAIM_TYPES = {"collision": 0.46, "comprehensive": 0.22, "liability": 0.20, "glass": 0.12}
CLAIM_TYPE_BASE_AMOUNT = {"collision": 8.6, "comprehensive": 8.2, "liability": 9.0, "glass": 6.4}

# --- Routing model (log-odds of Queue B) -----------------------------------
ROUTING_INTERCEPT = -2.35
ROUTING_COEF = {
    "log_amount_z": 0.85,      # per SD of log(claim amount)
    "vehicle_age": 0.045,      # per year
    "tenure_lt_12m": 0.55,     # new policy
    "prior_claims": 0.40,      # per prior claim in 3 yrs
    "liability": 0.30,         # liability claims get more scrutiny
    "latent_fraud": 1.60,      # true (unobserved) fraud signal
}
# Injected proxy bias: extra log-odds of Queue B by ZIP tier, applied
# independently of every legitimate factor above.
BIAS_LOGIT = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.85, 5: 1.25}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_rows: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # Dates: mild weekday effect and a winter bump in Jan-Feb.
    days = pd.date_range(START_DATE, END_DATE, freq="D")
    day_w = np.where(days.dayofweek < 5, 1.0, 0.65) * np.where(days.month <= 2, 1.15, 1.0)
    claim_date = rng.choice(days, size=n_rows, p=day_w / day_w.sum())

    zip_tier = rng.choice(list(ZIP_TIER_WEIGHTS), size=n_rows, p=list(ZIP_TIER_WEIGHTS.values()))
    zip3 = np.array([rng.choice(ZIP3_BY_TIER[t]) for t in zip_tier])

    # Mild, realistic confounding: higher tiers skew to older vehicles and
    # shorter tenure, so a naive comparison overstates the bias and a
    # controlled one is required to isolate it.
    vehicle_age = np.clip(rng.gamma(2.2, 2.6, n_rows) + 0.6 * (zip_tier - 3), 0, 25).round().astype(int)
    tenure_mean = 46 - 3.5 * (zip_tier - 3)
    policy_tenure_months = np.clip(rng.gamma(1.6, tenure_mean / 1.6), 1, 240).round().astype(int)
    prior_claims = rng.poisson(0.35 + 0.04 * (zip_tier - 3).clip(0), n_rows)

    claim_type = rng.choice(list(CLAIM_TYPES), size=n_rows, p=list(CLAIM_TYPES.values()))
    base = np.array([CLAIM_TYPE_BASE_AMOUNT[c] for c in claim_type])
    # Older vehicles -> lower repair value; noise is lognormal.
    log_amount = base - 0.025 * vehicle_age + rng.normal(0, 0.75, n_rows)
    claim_amount = np.clip(np.exp(log_amount), 150, 150_000).round(2)

    # Latent fraud is independent of ZIP tier by construction.
    latent_fraud = rng.random(n_rows) < 0.06

    log_amt = np.log(claim_amount)
    log_amount_z = (log_amt - log_amt.mean()) / log_amt.std()
    logit = (
        ROUTING_INTERCEPT
        + ROUTING_COEF["log_amount_z"] * log_amount_z
        + ROUTING_COEF["vehicle_age"] * vehicle_age
        + ROUTING_COEF["tenure_lt_12m"] * (policy_tenure_months < 12)
        + ROUTING_COEF["prior_claims"] * prior_claims
        + ROUTING_COEF["liability"] * (claim_type == "liability")
        + ROUTING_COEF["latent_fraud"] * latent_fraud
        + np.vectorize(BIAS_LOGIT.get)(zip_tier)
    )
    p_queue_b = sigmoid(logit)
    queue_b = rng.random(n_rows) < p_queue_b
    adjuster_queue = np.where(queue_b, "B", "A")

    # Outcomes. Queue B investigates, so it catches more fraud but also denies
    # or trims more legitimate claims and takes longer.
    outcome = np.empty(n_rows, dtype=object)
    labels = ["approved_full", "approved_partial", "denied", "withdrawn"]
    probs = {
        ("A", False): [0.86, 0.09, 0.03, 0.02],
        ("A", True): [0.55, 0.15, 0.22, 0.08],
        ("B", False): [0.62, 0.22, 0.11, 0.05],
        ("B", True): [0.08, 0.12, 0.62, 0.18],
    }
    for (q, f), p in probs.items():
        mask = (adjuster_queue == q) & (latent_fraud == f)
        outcome[mask] = rng.choice(labels, size=mask.sum(), p=p)

    paid_ratio = np.select(
        [outcome == "approved_full", outcome == "approved_partial"],
        [rng.uniform(0.95, 1.0, n_rows), rng.uniform(0.35, 0.85, n_rows)],
        default=0.0,
    )
    amount_paid = (claim_amount * paid_ratio).round(2)

    days_to_close = np.where(
        queue_b, rng.gamma(4.0, 9.0, n_rows), rng.gamma(2.0, 4.0, n_rows)
    ).round().astype(int) + 1

    df = pd.DataFrame({
        "claim_id": [f"CLM-{i:06d}" for i in range(1, n_rows + 1)],
        "claim_date": pd.to_datetime(claim_date).date,
        "zip3": zip3,
        "zip_tier": zip_tier,
        "claim_type": claim_type,
        "claim_amount": claim_amount,
        "vehicle_age": vehicle_age,
        "policy_tenure_months": policy_tenure_months,
        "prior_claims_3yr": prior_claims,
        "adjuster_queue": adjuster_queue,
        "final_outcome": outcome,
        "amount_paid": amount_paid,
        "days_to_close": days_to_close,
        "_latent_fraud": latent_fraud,
        "_p_queue_b": p_queue_b.round(4),
    })
    df = df.sort_values(["claim_date", "claim_id"]).reset_index(drop=True)
    df["claim_id"] = [f"CLM-{i:06d}" for i in range(1, n_rows + 1)]

    # Ground truth is split out so the main file carries no hint of the bias.
    truth = pd.DataFrame({
        "claim_id": df["claim_id"],
        "latent_fraud": df.pop("_latent_fraud").astype(int),
        "p_queue_b": df.pop("_p_queue_b"),
    })
    return df, truth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/claims.csv"))
    args = ap.parse_args()

    df, truth = generate(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    truth_path = args.out.with_name(args.out.stem + "_ground_truth.csv")
    truth.to_csv(truth_path, index=False)

    meta = {
        "rows": len(df),
        "seed": args.seed,
        "date_range": [str(START_DATE.date()), str(END_DATE.date())],
        "routing_intercept": ROUTING_INTERCEPT,
        "routing_coefficients": ROUTING_COEF,
        "injected_bias_logit_by_zip_tier": BIAS_LOGIT,
        "note": "latent_fraud is independent of zip_tier; any tier effect on routing "
                "beyond the legitimate factors is the injected bias.",
    }
    meta_path = args.out.with_name(args.out.stem + "_metadata.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {len(df):,} rows to {args.out} (+ {truth_path.name}, {meta_path.name})")


if __name__ == "__main__":
    main()
