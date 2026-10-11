# Claim routing fairness monitor — flag summary

**Data:** `data/claims.csv` · 6,000 claims · Jan 01 2026 – Jun 30 2026 · grouped by `zip_tier` · run Oct 11 2026  
**Overall status: RED — action needed**

## Bottom line

Claims from **Tier 4, Tier 5** are sent to Special Investigation (Queue B) much more often than claims from other areas. Tier 5 claims go to Queue B **42%** of the time, against **18%** for Tier 2. Most of that gap remains after accounting for claim amount and vehicle age, so it cannot be explained by those groups simply filing bigger claims or driving older cars. The extra investigations mostly land on honest customers: their claims are paid in full afterwards, but only after a slower, more intrusive process. The pattern is steady from month to month, which points to how routing works rather than a recent change.

## Flags

| Group | Sent to Queue B | Expected* | Impact ratio (≥0.80 ok) | Found nothing | Honest claimants investigated / 100 | Flags |
|---|---|---|---|---|---|---|
| Tier 1 | 20% | 18% | 0.98 | 55% | 10.9 | — |
| Tier 2 | 18% | 19% | 1.00 | 54% | 9.9 | — |
| Tier 3 | 19% | 19% | 0.99 | 46% | 8.6 | — |
| Tier 4 | 33% | 19% | 0.81 | 59% | 19.8 | ⚠ over-routed, burden |
| Tier 5 | 42% | 20% | **0.70** | 60% | 25.6 | ⚠ 4/5ths, over-routed, burden |

*Expected = the Queue B rate the group would get if routed like Tier 2 (the best-treated group), given its own claim amounts and vehicle ages. Book-wide Queue B rate: 25%.

## What each check found

- **Tier 4** — routed to Queue B 1.7× as often as its claims would predict when routed like Tier 2; Fast-Track impact ratio 0.81 (adjusted 0.83); 19.8 of every 100 claimants were investigated and then paid in full, vs 9.9 for Tier 2.
- **Tier 5** — routed to Queue B 2.1× as often as its claims would predict when routed like Tier 2; Fast-Track impact ratio 0.70 (adjusted 0.72); 25.6 of every 100 claimants were investigated and then paid in full, vs 9.9 for Tier 2.
- **Why Tier 4 passes the four-fifths test but is still flagged:** the test compares Fast-Track rates, which are high for everyone (67% vs 82%), so a large difference in investigations (33% vs 18%) barely moves the ratio. Measured on investigations instead, Tier 4 claimants are 1.8× as likely to be investigated as the least-investigated group.
- **Drift:** 1 of 15 group-months moved significantly against their trailing three-month baseline (checked 2026-04, 2026-05, 2026-06). Moved: Tier 4 2026-04 (+10.6 pts). With 15 checks at p < 0.05, about 0.8 would flag by chance alone, so treat an isolated hit as a watch item and confirm next month. Either way, the disparity itself was present from the first month (see drift.png).
- **Monthly four-fifths check:** below 0.80 in Tier 4 3/6 months, Tier 5 6/6 months.

## Recommended next steps

1. Review the routing rules and model features for anything tied to location (ZIP code, territory, or a stand-in for it such as a territory risk score).
2. Re-test routing with location features removed or neutralised, and confirm that fraud catch rate holds up.
3. Until fixed, consider second-level review before a flagged-group claim enters Queue B.
4. Keep this monitor running monthly; escalate if any impact ratio falls further or a drift flag appears.

<sub>Method: impact ratio = group Fast-Track rate ÷ highest group's Fast-Track rate (flag < 0.8). Over-routed = observed ÷ expected Queue B rate > 1.25, with expected rates from the reference group's claim-amount quintile × vehicle-age band cells. Burden flag = ≥ 5 more honest-claimant investigations per 100 than the best group. Drift = month vs pooled prior 3 months, flagged at ≥ 3 pts and p < 0.05 (two-proportion z-test). The four-fifths rule is a screening heuristic, not a legal finding. Charts: selection_rates.png, disparity_ratio.png, false_investigation.png, drift.png.</sub>
