"""Computes per-(axis, condition, model, metric) effect sizes vs. that model's
baseline, with a Bonferroni-corrected significance flag, and writes them into
stats_summary. Deterministic, not LLM-driven -- the whole point is that the
agent shouldn't be the one deciding what counts as a "real" finding.

The same 8 task templates repeat across every condition, so responses aren't
independent samples: some templates just run narrower/more formal than others
regardless of condition (baseline narrowness alone ranges ~1.0-3.0 across
templates). Treating all 16 responses per cell as iid understates uncertainty.
Instead this pairs by template: for each template, take (condition mean -
that template's own baseline mean), then test whether the mean of those
per-template paired differences is nonzero, with a t-distribution on
n_templates - 1 degrees of freedom (n_templates <= 8, not 16).

Usage: uv run scripts/compute_stats.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import t as t_dist

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
        SELECT r.model, r.template_id, c.axis, c.condition_id, s.*
        FROM scores s
        JOIN responses r USING (response_id)
        JOIN conditions c USING (condition_id)
    """).fetchdf()

    if df.empty:
        print("No scored responses yet -- run generate_benchmark.py and score_benchmark.py first.")
        return

    alpha = 0.05 / N_COMPARISONS
    rows = []

    for model, model_df in df.groupby("model"):
        baseline_df = model_df[model_df["condition_id"] == "baseline"]
        for metric in METRICS:
            baseline_vals = pd.to_numeric(baseline_df[metric], errors="coerce")
            baseline_by_template = baseline_vals.groupby(baseline_df["template_id"]).mean()
            overall_baseline_mean = baseline_vals.dropna().mean()
            if pd.isna(overall_baseline_mean):
                continue

            for condition_id, cond_df in model_df.groupby("condition_id"):
                if condition_id == "baseline":
                    continue
                axis = cond_df["axis"].iloc[0]
                cond_vals = pd.to_numeric(cond_df[metric], errors="coerce")
                n_responses = cond_vals.notna().sum()
                if n_responses == 0:
                    continue
                mean_c = cond_vals.mean()

                cond_by_template = cond_vals.groupby(cond_df["template_id"]).mean()
                common = cond_by_template.index.intersection(baseline_by_template.index)
                diffs = (cond_by_template[common] - baseline_by_template[common]).dropna()
                n_templates = len(diffs)

                if n_templates < 2:
                    # Can't estimate across-template variance; report descriptively, never "significant".
                    rows.append((axis, condition_id, model, metric, int(n_responses), mean_c,
                                 overall_baseline_mean, mean_c - overall_baseline_mean, None, None, False))
                    continue

                diff_mean = diffs.mean()
                diff_se = diffs.std(ddof=1) / (n_templates ** 0.5)
                if diff_se == 0:
                    ci_low = ci_high = diff_mean
                    sig = diff_mean != 0
                else:
                    t_crit = t_dist.ppf(1 - alpha / 2, df=n_templates - 1)
                    ci_low = diff_mean - t_crit * diff_se
                    ci_high = diff_mean + t_crit * diff_se
                    sig = (ci_low > 0) or (ci_high < 0)

                rows.append((axis, condition_id, model, metric, int(n_responses), mean_c,
                             overall_baseline_mean, diff_mean, ci_low, ci_high, bool(sig)))

    con.execute("DELETE FROM stats_summary")
    con.executemany(
        "INSERT INTO stats_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.close()
    sig_count = sum(1 for r in rows if r[-1])
    print(f"Wrote {len(rows)} stats_summary rows ({sig_count} Bonferroni-significant, "
          f"template-paired t-test, df<=7) to {db_path}")


if __name__ == "__main__":
    main()
