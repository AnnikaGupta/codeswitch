"""Computes per-(axis, condition, model, metric) effect sizes vs. that model's
baseline, with a Bonferroni-corrected significance flag, and writes them into
stats_summary. Deterministic, not LLM-driven -- the whole point is that the
agent shouldn't be the one deciding what counts as a "real" finding.

Usage: uv run scripts/compute_stats.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from db_schema import connect, DB_PATH_DEFAULT  # noqa: E402

METRICS = [
    "refusal_flag",
    "informality_score",
    "explicit_gender_flag",
    "implicit_gender_flag",
    "narrowness_score",
]

N_COMPARISONS = 12  # non-baseline conditions; Bonferroni correction divides alpha by this


def main(db_path: str = DB_PATH_DEFAULT):
    con = connect(db_path)
    df = con.execute("""
        SELECT r.model, c.axis, c.condition_id, s.*
        FROM scores s
        JOIN responses r USING (response_id)
        JOIN conditions c USING (condition_id)
    """).fetchdf()

    if df.empty:
        print("No scored responses yet -- run generate_benchmark.py and score_benchmark.py first.")
        return

    z_crit = norm.ppf(1 - 0.05 / (2 * N_COMPARISONS))
    rows = []

    for model, model_df in df.groupby("model"):
        baseline_df = model_df[model_df["condition_id"] == "baseline"]
        for metric in METRICS:
            vals = pd.to_numeric(baseline_df[metric], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            baseline_mean = vals.mean()
            baseline_se = vals.std(ddof=1) / max(len(vals) ** 0.5, 1) if len(vals) > 1 else 0.0

            for condition_id, cond_df in model_df.groupby("condition_id"):
                if condition_id == "baseline":
                    continue
                axis = cond_df["axis"].iloc[0]
                vals_c = pd.to_numeric(cond_df[metric], errors="coerce").dropna()
                n = len(vals_c)
                if n == 0:
                    continue
                mean_c = vals_c.mean()
                se_c = vals_c.std(ddof=1) / max(n ** 0.5, 1) if n > 1 else 0.0

                diff = mean_c - baseline_mean
                se_diff = (se_c ** 2 + baseline_se ** 2) ** 0.5
                ci_low = diff - z_crit * se_diff
                ci_high = diff + z_crit * se_diff
                sig = (ci_low > 0) or (ci_high < 0)

                rows.append((axis, condition_id, model, metric, n, mean_c, baseline_mean,
                             diff, ci_low, ci_high, bool(sig)))

    con.execute("DELETE FROM stats_summary")
    con.executemany(
        "INSERT INTO stats_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.close()
    sig_count = sum(1 for r in rows if r[-1])
    print(f"Wrote {len(rows)} stats_summary rows ({sig_count} Bonferroni-significant) to {db_path}")


if __name__ == "__main__":
    main()
