# Halcyon: synthetic claims dataset with an injected proxy bias

A synthetic auto-insurance claims dataset (6,000 rows, 2026-01-01 to 2026-06-30) built as a
**test fixture for fairness and disparate-impact auditing**. A known bias is planted in adjuster
routing: claims from ZIP-code tiers 4 and 5 go to Queue B (Special Investigation / High-Scrutiny)
more often than the legitimate risk factors can explain. Because the size of the bias is known,
you can check whether an audit method finds it.

All data is fictional. The ZIP prefixes are made up and don't match real geography.

## Files

| File | Contents |
|---|---|
| `data/claims.csv` | The dataset you audit. Nothing in it marks the bias. |
| `data/claims_ground_truth.csv` | Per-claim `latent_fraud` flag and the true `p_queue_b` probability. Keep this away from the method under test. |
| `data/claims_metadata.json` | Seed, routing coefficients and the injected bias values. |
| `generate_claims.py` | Generator (`--rows`, `--seed`, `--out`). |
| `verify_bias.py` | Checks that the bias holds up under controls. |

## Columns (`claims.csv`)

`claim_id`, `claim_date`, `zip3` (fictional 3-digit ZIP prefix), `zip_tier` (1–5, the ZIP-code proxy),
`claim_type`, `claim_amount`, `vehicle_age` (years), `policy_tenure_months`, `prior_claims_3yr`,
`adjuster_queue` (`A` = Fast-Track, `B` = Special Investigation / High-Scrutiny),
`final_outcome` (`approved_full`, `approved_partial`, `denied`, `withdrawn`), `amount_paid`, `days_to_close`.

## How routing works

The chance of going to Queue B is a logistic model. Legitimate inputs raise it: larger claims,
older vehicles, policies under 12 months old, prior claims, liability claims, and an unobserved
fraud signal that has no connection to ZIP tier. On top of that, an **injected penalty** adds
**+0.85** log-odds (odds ×2.3) for tier 4 and **+1.25** (odds ×3.5) for tier 5.

Higher tiers also skew a little toward older vehicles and shorter tenure. So the raw gap
overstates the bias, and you need controls to measure it correctly.

Queue B outcomes are worse even for honest claims: more partial payments and denials, and claims
take longer to close. That means the routing bias also shows up as a gap in outcomes.

## Verification (seed 42)

```
python generate_claims.py && python verify_bias.py
```

- Raw Queue B rate: tiers 1–3 ≈ 18–20%, tier 4 ≈ 33%, tier 5 ≈ 43%.
- Within amount-quintile × vehicle-age cells, tiers 4–5 have the higher Queue B rate in 20 of 20
  cells. The weighted gap is +18 points.
- Logistic regression controlling for log(amount) and vehicle age: tier 4 coef 0.72 (OR 2.0),
  tier 5 coef 1.18 (OR 3.2), both z > 6. Tiers 2–3 show no significant effect.

## Routing fairness monitor

```
pip install -r requirements.txt
python monitor_routing.py data/claims.csv --out reports [--group zip_tier]
```

It writes the following to `reports/`:

| Output | What it shows |
|---|---|
| `FLAG_SUMMARY.md` | A one-page summary in plain language: overall status, a flag table, findings and next steps. |
| `selection_rates.png` | The Queue B rate for each group with a 95% interval. Next to it is the rate the group would get if it were routed like the best-treated group, given its own claim amounts and vehicle ages. |
| `disparity_ratio.png` | The four-fifths impact ratio for each group (Fast-Track rate ÷ the best group's rate), both raw and adjusted for claim amount and vehicle age. |
| `false_investigation.png` | False investigations, meaning claims sent to Queue B and then paid in full. Shown two ways: as a share of each group's investigations, and per 100 claimants in the group. |
| `drift.png` | Each month's Queue B rate and impact ratio, compared with a trailing three-month baseline. |
| `group_metrics.csv`, `monthly_drift.csv` | The numbers behind the charts. |

Flags and their default thresholds (set at the top of the script):

- **4/5ths:** impact ratio below 0.80.
- **Over-routed:** observed ÷ expected Queue B rate above 1.25.
- **Burden:** at least 5 more false investigations per 100 claimants than the best group.
- **Drift:** a month moves at least 3 points from its baseline, with p < 0.05.

The over-routed flag is there because the four-fifths test is lenient when the favourable rate is high for every
group. Tier 4 shows this: at 0.81 it passes the four-fifths test, yet it is investigated 1.8× as often as tier 2.

On a copy of the data generated with the injected bias set to zero, the monitor raises no flags.
