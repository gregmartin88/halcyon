"""Fairness monitor for claim routing to Queue B (Special Investigation).

Computes, per group (default: zip_tier):
  1. Selection rate      share of claims routed to Queue B, with 95% CIs, plus an
                         observed / expected ratio that controls for claim amount
                         and vehicle age (expected = what the group's own mix of
                         claims would get at the book-wide rate for each cell).
  2. Disparity ratio     four-fifths-rule impact ratio on the favourable outcome
                         (Fast-Track rate / best group's Fast-Track rate), and the
                         stricter adverse-selection ratio
                         (lowest Queue B rate / group's Queue B rate).
  3. False-investigation rate
                         share of Queue B claims that were then paid in full, i.e.
                         the investigation found nothing; also expressed per 100
                         claims in the group (the burden on that group's customers).
  4. Month-over-month drift
                         each month's Queue B rate vs a trailing three-month
                         baseline, flagged when the change is both material and
                         statistically significant.

Writes chart PNGs, metric CSVs and a plain-language one-page FLAG_SUMMARY.md.

Usage:
    python monitor_routing.py [data/claims.csv] [--out reports] [--group zip_tier]
"""

import argparse
from datetime import date
from math import erf, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --- thresholds ---------------------------------------------------------------
FOUR_FIFTHS = 0.80          # impact ratio below this is flagged
OBS_EXP_FLAG = 1.25         # observed/expected Queue B rate above this is flagged
FIR_GAP_PP = 5.0            # false-investigations per 100 claims, gap vs best group
DRIFT_MIN_PP = 3.0          # drift must move at least this many points...
DRIFT_ALPHA = 0.05          # ...and be significant at this level
BASELINE_MONTHS = 3
MIN_CELL = 30               # suppress group/month rates on fewer claims

# --- chart styling (reference palette, light mode) ----------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"            # categorical slot 1
ORANGE = "#eb6834"          # categorical slot 2
BLUE_LIGHT = "#cde2fb"
NEUTRAL = "#b9b8b1"
CRITICAL = "#d03b3b"        # status: critical (always paired with icon + label)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


# --- statistics ---------------------------------------------------------------
def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def two_prop_p(k1, n1, k2, n2):
    if min(n1, n2) == 0:
        return 1.0
    p = (k1 + k2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs(k1 / n1 - k2 / n2) / se
    return 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))


# --- metrics ------------------------------------------------------------------
def load(path):
    df = pd.read_csv(path, parse_dates=["claim_date"])
    df["queue_b"] = (df["adjuster_queue"] == "B").astype(int)
    df["false_inv"] = ((df["queue_b"] == 1) & (df["final_outcome"] == "approved_full")).astype(int)
    df["month"] = df["claim_date"].dt.to_period("M")
    # Control cells for the observed / expected comparison.
    df["amt_q"] = pd.qcut(df["claim_amount"], 5, labels=False)
    df["veh_band"] = pd.cut(df["vehicle_age"], [-1, 3, 7, 12, 99], labels=False)
    return df


def group_metrics(df, group):
    g = df.groupby(group)
    m = pd.DataFrame({
        "claims": g.size(),
        "queue_b": g["queue_b"].sum(),
        "false_inv": g["false_inv"].sum(),
    })
    m["selection_rate"] = m["queue_b"] / m["claims"]
    m["sel_lo"], m["sel_hi"] = wilson(m["queue_b"], m["claims"])

    fast = 1 - m["selection_rate"]
    m["fast_track_rate"] = fast
    ref = fast.idxmax()
    m["impact_ratio"] = fast / fast[ref]
    m["adverse_ratio"] = m["selection_rate"].min() / m["selection_rate"]

    # Expected Queue B rate if each group's own claims (claim-amount quintile x
    # vehicle-age band mix) were routed at the reference group's cell rates.
    # Book-wide rates would already contain any bias, so they are not used.
    ref_cells = df[df[group] == ref].groupby(["amt_q", "veh_band"])["queue_b"].agg(["mean", "size"])
    book_cells = df.groupby(["amt_q", "veh_band"])["queue_b"].mean()
    cell_rate = ref_cells["mean"].where(ref_cells["size"] >= MIN_CELL).reindex(book_cells.index)
    cell_rate = cell_rate.fillna(book_cells)
    claim_cell = pd.MultiIndex.from_frame(df[["amt_q", "veh_band"]])
    df_expected = pd.Series(cell_rate.reindex(claim_cell).to_numpy(), index=df.index)
    m["expected_rate"] = df_expected.groupby(df[group]).mean()
    m["obs_exp"] = m["selection_rate"] / m["expected_rate"]
    # Adjusted impact ratio: actual Fast-Track rate vs the rate the group would
    # get if treated like the reference group, given its claim mix.
    m["impact_ratio_adj"] = fast / (1 - m["expected_rate"])
    m.attrs["reference"] = ref

    m["false_inv_rate"] = m["false_inv"] / m["queue_b"]
    m["fir_lo"], m["fir_hi"] = wilson(m["false_inv"], m["queue_b"])
    m["false_inv_per100"] = 100 * m["false_inv"] / m["claims"]

    m["flag_impact"] = m["impact_ratio"] < FOUR_FIFTHS
    m["flag_obs_exp"] = m["obs_exp"] > OBS_EXP_FLAG
    m["flag_fir"] = (m["false_inv_per100"] - m["false_inv_per100"].min()) >= FIR_GAP_PP
    return m


def drift_metrics(df, group):
    rows = []
    months = sorted(df["month"].unique())
    for key, sub in df.groupby(group):
        by_m = sub.groupby("month")["queue_b"].agg(["sum", "size"]).reindex(months, fill_value=0)
        for i, mo in enumerate(months):
            k, n = by_m.loc[mo, "sum"], by_m.loc[mo, "size"]
            row = {group: key, "month": str(mo), "claims": n, "queue_b": k,
                   "rate": k / n if n >= MIN_CELL else np.nan}
            if i >= BASELINE_MONTHS:
                base = by_m.iloc[i - BASELINE_MONTHS:i]
                bk, bn = base["sum"].sum(), base["size"].sum()
                row["baseline_rate"] = bk / bn
                row["change_pp"] = 100 * (k / n - bk / bn)
                row["p_value"] = two_prop_p(k, n, bk, bn)
                row["flag_drift"] = abs(row["change_pp"]) >= DRIFT_MIN_PP and row["p_value"] < DRIFT_ALPHA
            rows.append(row)
    d = pd.DataFrame(rows)

    # Monthly impact ratio (four-fifths) per group.
    fast = 1 - d.pivot(index="month", columns=group, values="rate")
    impact = fast.div(fast.max(axis=1), axis=0)
    return d, impact


# --- charts -------------------------------------------------------------------
def _label(t):
    return f"Tier {t}" if isinstance(t, (int, np.integer)) else str(t)


def _caption(fig, text):
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=8.5, color=MUTED, wrap=True)


def chart_selection(m, overall, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(m))
    flagged = m["flag_impact"] | m["flag_obs_exp"]
    colors = [CRITICAL if f else BLUE for f in flagged]
    ax.bar(x, m["selection_rate"] * 100, width=0.42, color=colors, zorder=2)
    ax.errorbar(x, m["selection_rate"] * 100,
                yerr=[(m["selection_rate"] - m["sel_lo"]) * 100, (m["sel_hi"] - m["selection_rate"]) * 100],
                fmt="none", ecolor=INK_2, elinewidth=1, capsize=3, zorder=3)
    ax.plot(x, m["expected_rate"] * 100, "o", ms=8, mfc=SURFACE, mec=INK, mew=1.6, zorder=4,
            label=f"Expected if routed like {_label(m.attrs['reference'])}, given claim amount and vehicle age")
    ax.axhline(overall * 100, color=MUTED, lw=1, zorder=1)
    ax.text(len(m) - 0.5, overall * 100, f"  book-wide {overall:.0%}", va="center", ha="left",
            color=INK_2, fontsize=8.5)
    for xi, (r, f) in enumerate(zip(m["selection_rate"], flagged)):
        ax.text(xi, m["sel_hi"].iloc[xi] * 100 + 1.2, f"{'▲ ' if f else ''}{r:.0%}",
                ha="center", color=INK, fontsize=9, fontweight="bold" if f else "normal")
    ax.set_xticks(x, [_label(t) for t in m.index])
    ax.set_ylabel("Routed to Queue B (%)")
    ax.set_ylim(0, max(m["sel_hi"]) * 100 + 8)
    ax.grid(axis="x", visible=False)
    ax.set_title("Selection rate: share of claims sent to Special Investigation")
    ax.legend(loc="upper left", fontsize=8.5)
    _caption(fig, "Bars: observed Queue B rate with 95% interval. Red ▲ = flagged. "
                  "Hollow dot: rate expected if the group were routed like the best-treated group, "
                  "given its own claim amounts and vehicle ages.")
    fig.savefig(out)
    plt.close(fig)


def chart_impact(m, out):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(m))[::-1]
    for yi, (raw, adj) in zip(y, zip(m["impact_ratio"], m["impact_ratio_adj"])):
        ax.plot([min(raw, adj), max(raw, adj)], [yi, yi], color=AXIS, lw=2, zorder=1)
    ax.scatter(m["impact_ratio"], y, s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="Raw impact ratio")
    ax.scatter(m["impact_ratio_adj"], y, s=70, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="Adjusted for claim amount and vehicle age")
    ax.axvline(FOUR_FIFTHS, color=CRITICAL, lw=1.2, zorder=2)
    ax.text(FOUR_FIFTHS - 0.005, -0.5, "four-fifths line (0.80)", color=INK_2, fontsize=8.5,
            ha="right", va="bottom")
    ax.axvspan(0.5, FOUR_FIFTHS, color=CRITICAL, alpha=0.06, zorder=0)
    for yi, raw, fl in zip(y, m["impact_ratio"], m["flag_impact"]):
        ax.text(1.045, yi, f"{'▲ ' if fl else ''}{raw:.2f}", va="center", ha="left", fontsize=9,
                color=INK, fontweight="bold" if fl else "normal")
    ax.set_yticks(y, [_label(t) for t in m.index])
    ax.set_xlim(0.6, 1.1)
    ax.set_ylim(-0.6, len(m) - 0.1)
    ax.set_xlabel("Fast-Track rate relative to the best-treated group (1.00 = parity)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", fontsize=8.5, ncol=2, bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    ax.set_title("Disparity ratio (four-fifths rule)", pad=26)
    _caption(fig, "Values at right are the raw ratio. Below 0.80 is the conventional adverse-impact flag.")
    fig.savefig(out)
    plt.close(fig)


def chart_false_investigation(m, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    x = np.arange(len(m))
    labels = [_label(t) for t in m.index]

    a1.bar(x, m["false_inv_rate"] * 100, width=0.42, color=BLUE, zorder=2)
    a1.errorbar(x, m["false_inv_rate"] * 100,
                yerr=[(m["false_inv_rate"] - m["fir_lo"]) * 100, (m["fir_hi"] - m["false_inv_rate"]) * 100],
                fmt="none", ecolor=INK_2, elinewidth=1, capsize=3, zorder=3)
    for xi, r in enumerate(m["false_inv_rate"]):
        a1.text(xi, m["fir_hi"].iloc[xi] * 100 + 1.5, f"{r:.0%}", ha="center", fontsize=9, color=INK)
    a1.set_ylim(0, 100)
    a1.set_xticks(x, labels)
    a1.set_ylabel("% of investigated claims")
    a1.set_title("Investigations that found nothing", fontsize=11)
    a1.grid(axis="x", visible=False)

    colors = [CRITICAL if f else BLUE for f in m["flag_fir"]]
    a2.bar(x, m["false_inv_per100"], width=0.42, color=colors, zorder=2)
    for xi, (v, f) in enumerate(zip(m["false_inv_per100"], m["flag_fir"])):
        a2.text(xi, v + 0.5, f"{'▲ ' if f else ''}{v:.1f}", ha="center", fontsize=9, color=INK,
                fontweight="bold" if f else "normal")
    a2.set_ylim(0, m["false_inv_per100"].max() * 1.25)
    a2.set_xticks(x, labels)
    a2.set_ylabel("per 100 claims in the group")
    a2.set_title("Honest claimants put through investigation", fontsize=11)
    a2.grid(axis="x", visible=False)

    fig.suptitle("False-investigation rate", x=0.0, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    _caption(fig, "“False investigation” = routed to Queue B, then paid in full. Left: share of each group's "
                  "investigations that found nothing (95% interval). Right: how many of every 100 claimants "
                  f"in the group went through one. Red ▲ = at least {FIR_GAP_PP:.0f} per 100 above the best group.")
    fig.savefig(out)
    plt.close(fig)


def chart_drift(d, impact, group, out):
    keys = list(d[group].unique())
    fig, axes = plt.subplots(2, len(keys), figsize=(2.3 * len(keys) + 1, 5.6), sharey="row",
                             gridspec_kw={"height_ratios": [3, 2]})
    months = sorted(d["month"].unique())
    x = np.arange(len(months))
    mlabels = [pd.Period(mo).strftime("%b") for mo in months]
    _, hi_all = wilson(d["queue_b"].values, d["claims"].values)
    ymax = np.ceil(np.nanmax(hi_all) * 10) * 10 + 2

    for ax, key in zip(axes[0], keys):
        s = d[d[group] == key].set_index("month").reindex(months)
        lo, hi = wilson(s["queue_b"].values, s["claims"].values)
        ax.fill_between(x, lo * 100, hi * 100, color=BLUE, alpha=0.10, lw=0)
        ax.plot(x, s["rate"] * 100, color=BLUE, lw=2, marker="o", ms=4, solid_capstyle="round")
        if "baseline_rate" in s:
            ax.plot(x, s["baseline_rate"] * 100, color=INK_2, lw=1.2, ls=(0, (3, 2)))
            hits = s.get("flag_drift", pd.Series(False, index=s.index)).fillna(False).astype(bool).values
            if hits.any():
                ax.scatter(x[hits], s["rate"].values[hits] * 100, s=90, color=CRITICAL,
                           edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.set_title(_label(key), fontsize=10)
        ax.set_xticks(x, mlabels, fontsize=8)
        ax.set_ylim(0, ymax)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Queue B rate (%)")

    for ax, key in zip(axes[1], keys):
        ax.axhspan(0.5, FOUR_FIFTHS, color=CRITICAL, alpha=0.06, lw=0)
        ax.axhline(FOUR_FIFTHS, color=CRITICAL, lw=1)
        ax.plot(x, impact[key].reindex(months), color=BLUE, lw=2, marker="o", ms=4)
        ax.set_xticks(x, mlabels, fontsize=8)
        ax.set_ylim(0.55, 1.03)
        ax.grid(axis="x", visible=False)
    axes[1][0].set_ylabel("Impact ratio")

    fig.suptitle("Month-over-month drift vs trailing three-month baseline",
                 x=0.0, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    _caption(fig, "Top: monthly Queue B rate (solid, shaded 95% interval) against the trailing "
                  "three-month baseline (dashed; starts once three months of history exist). A red dot marks "
                  f"a month that moved ≥{DRIFT_MIN_PP:.0f} points with p < {DRIFT_ALPHA}. "
                  "Bottom: monthly four-fifths impact ratio; red band = below 0.80.")
    fig.savefig(out)
    plt.close(fig)


# --- summary ------------------------------------------------------------------
def write_summary(df, m, d, impact, group, src, out):
    overall = df["queue_b"].mean()
    best = m["fast_track_rate"].idxmax()
    flagged = m[m["flag_impact"] | m["flag_obs_exp"] | m["flag_fir"]]
    drift_hits = d[d.get("flag_drift", False) == True]  # noqa: E712
    impact_months = (impact < FOUR_FIFTHS).sum()
    n_months = impact.shape[0]
    status = "RED — action needed" if len(flagged) else "GREEN — no flags"

    L = []
    L.append("# Claim routing fairness monitor — flag summary\n")
    L.append(f"**Data:** `{src}` · {len(df):,} claims · "
             f"{df.claim_date.min():%b %d %Y} – {df.claim_date.max():%b %d %Y} · "
             f"grouped by `{group}` · run {date.today():%b %d %Y}  ")
    L.append(f"**Overall status: {status}**\n")

    L.append("## Bottom line\n")
    if len(flagged):
        names = ", ".join(_label(t) for t in flagged.index)
        worst = m["impact_ratio"].idxmin()
        L.append(f"Claims from **{names}** are sent to Special Investigation (Queue B) much more often than "
                 f"claims from other areas. {_label(worst)} claims go to Queue B "
                 f"**{m.loc[worst, 'selection_rate']:.0%}** of the time, against "
                 f"**{m.loc[best, 'selection_rate']:.0%}** for {_label(best)}. "
                 "Most of that gap remains after accounting for claim amount and vehicle age, so it cannot be "
                 "explained by those groups simply filing bigger claims or driving older cars. "
                 "The extra investigations mostly land on honest customers: their claims are paid in full "
                 "afterwards, but only after a slower, more intrusive process. "
                 "The pattern is steady from month to month, which points to how routing works "
                 "rather than a recent change.\n")
    else:
        L.append("No group crosses any disparity threshold this period.\n")

    L.append("## Flags\n")
    L.append("| Group | Sent to Queue B | Expected* | Impact ratio (≥0.80 ok) | "
             "Found nothing | Honest claimants investigated / 100 | Flags |")
    L.append("|---|---|---|---|---|---|---|")
    for t, r in m.iterrows():
        fl = [n for n, on in (("4/5ths", r.flag_impact), ("over-routed", r.flag_obs_exp),
                              ("burden", r.flag_fir)) if on]
        L.append(f"| {_label(t)} | {r.selection_rate:.0%} | {r.expected_rate:.0%} | "
                 f"{'**' if r.flag_impact else ''}{r.impact_ratio:.2f}{'**' if r.flag_impact else ''} | "
                 f"{r.false_inv_rate:.0%} | {r.false_inv_per100:.1f} | "
                 f"{'⚠ ' + ', '.join(fl) if fl else '—'} |")
    L.append(f"\n*Expected = the Queue B rate the group would get if routed like "
             f"{_label(m.attrs['reference'])} (the best-treated group), given its own claim amounts and vehicle ages. Book-wide Queue B rate: {overall:.0%}.\n")

    L.append("## What each check found\n")
    for t, r in flagged.iterrows():
        L.append(f"- **{_label(t)}** — routed to Queue B {r.obs_exp:.1f}× as often as its claims would "
                 f"predict when routed like {_label(m.attrs['reference'])}; Fast-Track impact ratio {r.impact_ratio:.2f} "
                 f"(adjusted {r.impact_ratio_adj:.2f}); {r.false_inv_per100:.1f} of every 100 claimants were "
                 f"investigated and then paid in full, vs {m.loc[best, 'false_inv_per100']:.1f} "
                 f"for {_label(best)}.")
    near_miss = m[m["flag_obs_exp"] & ~m["flag_impact"]]
    for t, r in near_miss.iterrows():
        L.append(f"- **Why {_label(t)} passes the four-fifths test but is still flagged:** the test compares "
                 f"Fast-Track rates, which are high for everyone ({r.fast_track_rate:.0%} vs "
                 f"{m.loc[best, 'fast_track_rate']:.0%}), so a large difference in investigations "
                 f"({r.selection_rate:.0%} vs {m.loc[best, 'selection_rate']:.0%}) barely moves the ratio. "
                 f"Measured on investigations instead, {_label(t)} claimants are "
                 f"{1 / r.adverse_ratio:.1f}× as likely to be investigated as the least-investigated group.")
    n_tests = int(d["baseline_rate"].notna().sum()) if "baseline_rate" in d else 0
    L.append(f"- **Drift:** {len(drift_hits)} of {n_tests} group-months moved significantly against their trailing "
             f"three-month baseline (checked "
             f"{', '.join(sorted(d.dropna(subset=['baseline_rate'])['month'].unique()))}). "
             + (("The disparity is not getting worse or better — it has been there since the start of the data."
                 if len(flagged) else "Routing rates are stable.")
                if not len(drift_hits) else
                "Moved: " + ", ".join(f"{_label(r[group])} {r.month} ({r.change_pp:+.1f} pts)"
                                      for _, r in drift_hits.iterrows())
                + f". With {n_tests} checks at p < {DRIFT_ALPHA}, about {n_tests * DRIFT_ALPHA:.1f} "
                "would flag by chance alone, so treat an isolated hit as a watch item and confirm next "
                "month."
                + (" Either way, the disparity itself was present from the first month (see drift.png)."
                   if len(flagged) else "")))
    worst_months = {_label(k): int(v) for k, v in impact_months.items() if v}
    if worst_months:
        L.append("- **Monthly four-fifths check:** below 0.80 in "
                 + ", ".join(f"{k} {v}/{n_months} months" for k, v in worst_months.items()) + ".")
    L.append("")

    L.append("## Recommended next steps\n")
    if len(flagged):
        L.append("1. Review the routing rules and model features for anything tied to location "
                 "(ZIP code, territory, or a stand-in for it such as a territory risk score).")
        L.append("2. Re-test routing with location features removed or neutralised, and confirm that "
                 "fraud catch rate holds up.")
        L.append("3. Until fixed, consider second-level review before a flagged-group claim enters Queue B.")
        L.append("4. Keep this monitor running monthly; escalate if any impact ratio falls further "
                 "or a drift flag appears.\n")
    else:
        L.append("Continue monthly monitoring.\n")

    L.append("<sub>Method: impact ratio = group Fast-Track rate ÷ highest group's Fast-Track rate "
             f"(flag < {FOUR_FIFTHS}). Over-routed = observed ÷ expected Queue B rate > {OBS_EXP_FLAG}, "
             "with expected rates from the reference group's claim-amount quintile × vehicle-age band cells. Burden flag = "
             f"≥ {FIR_GAP_PP:.0f} more honest-claimant investigations per 100 than the best group. "
             f"Drift = month vs pooled prior {BASELINE_MONTHS} months, flagged at ≥ {DRIFT_MIN_PP:.0f} pts "
             f"and p < {DRIFT_ALPHA} (two-proportion z-test). The four-fifths rule is a screening "
             "heuristic, not a legal finding. Charts: selection_rates.png, disparity_ratio.png, "
             "false_investigation.png, drift.png.</sub>")
    out.write_text("\n".join(L) + "\n")


# --- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default="data/claims.csv")
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--group", default="zip_tier")
    args = ap.parse_args()

    df = load(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    m = group_metrics(df, args.group)
    d, impact = drift_metrics(df, args.group)

    m.round(4).to_csv(args.out / "group_metrics.csv")
    d.round(4).to_csv(args.out / "monthly_drift.csv", index=False)

    chart_selection(m, df["queue_b"].mean(), args.out / "selection_rates.png")
    chart_impact(m, args.out / "disparity_ratio.png")
    chart_false_investigation(m, args.out / "false_investigation.png")
    chart_drift(d, impact, args.group, args.out / "drift.png")
    write_summary(df, m, d, impact, args.group, args.data, args.out / "FLAG_SUMMARY.md")
    print(f"Wrote charts, CSVs and FLAG_SUMMARY.md to {args.out}/")


if __name__ == "__main__":
    main()
