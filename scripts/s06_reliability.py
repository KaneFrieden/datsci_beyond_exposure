#!/usr/bin/env python3
"""
Our s06 script seeks to examine the reliability of the word-level estimates obtained in s05.
In s05, we fit a generalized linear model (GLM) for each lemma to estimate its retention level
and decay slope. However, these estimates are subject to uncertainty due to limited data and
variability in learner responses. Therefore s06 now asks: how much of the observed variation
might reflect differences between lemmas, rather than uncertainty in the estimates?
This script reads the item outcomes from s05 and summarizes how much the fitted parameters
vary across lemmas. It compares this variation with the standard errors and saves the results
in a JSON report for further analysis.
Use --limit to inspect a small alphabetical subset without writing a file; these values are
not study results.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

# we compare the variation in the lemma estimates with their standard errors. This helps us
# assess how reliable the decay slope and retention level estimates are across lemmas.
def summarize(estimates: pd.Series, errors: pd.Series) -> dict:
    values = estimates.to_numpy(dtype=float)
    se = errors.to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all(): # we need at least two finite estimates to compute the variation and reliability metrics
        raise ValueError("At least two finite estimates are needed.")
    if not np.isfinite(se).all() or (se <= 0).any(): # we also need all standard errors to be finite and positive to compute the inverse-variance weights
        raise ValueError("Standard errors must be finite and positive.")

    # a smaller SE pinpoints a more precise estimate, so it receives more weight.
    weights = 1.0 / se**2
    weighted_mean = float(np.average(values, weights=weights))
    # the Q statistic measures variation around the weighted mean. Tau-squared estimates by contrast how much of that
    # variation remains between lemmas after accounting for their estimation uncertainty.
    q = float(np.sum(weights * (values - weighted_mean)**2))
    degrees_of_freedom = len(values) - 1
    c = float(weights.sum() - np.sum(weights**2) / weights.sum())
    tau2 = max(0.0, (q - degrees_of_freedom) / c) if c > 0 else 0.0
    observed_variance = float(np.var(values, ddof=1))
    mean_error_variance = float(np.mean(se**2))

    return {
        "lemmas": len(values),
        "weighted_mean": weighted_mean,
        "observed_variance": observed_variance,
        "mean_squared_standard_error": mean_error_variance,
        "variance_ratio": observed_variance / mean_error_variance,
        "median_standard_error": float(np.median(se)),
        "q": q,
        "q_degrees_of_freedom": degrees_of_freedom,
        "q_p_value_approximate": float(chi2.sf(q, degrees_of_freedom)),
        "tau_squared": tau2,
        "tau": float(np.sqrt(tau2)),
        "i_squared": max(0.0, (q - degrees_of_freedom) / q) if q > 0 else 0.0,
        "typical_estimate_reliability_approximate": (
            tau2 / (tau2 + mean_error_variance) if tau2 + mean_error_variance > 0 else 0.0
        ),
        "effective_n_inverse_variance_weights": float(weights.sum()**2 / np.sum(weights**2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="inspect the first N lemmas without writing an output")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must not be negative")

    config.require_data_root()
    folder = config.DATA_INTERMEDIATE
    input_path = folder / "item_outcomes.parquet"
    source_report = folder / "outcomes_report.json"
    output_path = folder / "reliability_report.json"
    if not input_path.is_file() or not source_report.is_file():
        raise RuntimeError("s05 outcome table and report must exist before s06.")
    if not args.limit and output_path.exists():
        raise RuntimeError(f"s06 output already exists: {output_path}")

    outcomes = pd.read_parquet(input_path)
    previous_report = json.loads(source_report.read_text(encoding="utf-8"))
    if outcomes.empty or outcomes["lemma"].duplicated().any():
        raise ValueError("s05 must contain one non-empty row per lemma.")
    if len(outcomes) != previous_report["eligible_lemmas"]:
        raise ValueError("The s05 table and report disagree on the lemma count.")
    if args.limit:
        # a small alphabetical subset can be useful for inspection, but it is not a study result and should not be written to the output file
        outcomes = outcomes.sort_values("lemma").head(args.limit).copy()

    # we only include lemmas for which the GLM has converged. unstable retention levels are
    # excluded from the level summary, but the same lemmas can still enter the slope summary.
    converged = outcomes.loc[outcomes["glm_converged"]]
    slope = converged.dropna(subset=["decay_slope", "decay_slope_se"])
    level = converged.loc[~converged["level_unstable"]].dropna(
        subset=["retention_level", "retention_level_se"]
    )
    report = {
        "source_eligible_lemmas": previous_report["eligible_lemmas"],
        "lemmas_inspected": len(outcomes),
        "converged": len(converged),
        "level_warning_excluded_from_level_summary": int(converged["level_unstable"].sum()),
        "decay_slope": summarize(slope["decay_slope"], slope["decay_slope_se"]),
        "retention_level": summarize(level["retention_level"], level["retention_level_se"]),
        "method": "inverse-variance Q; DerSimonian-Laird tau-squared; descriptive noise comparison",
        "notes": [
            "The level warning does not exclude the same lemma from the slope summary.",
            "Tau-squared is clipped at zero and depends on the model and standard errors.",
            "Typical-estimate reliability is an approximation, not an individual lemma score.",
            "Q's p-value assumes independent, correctly estimated errors; learners can occur under several lemmas.",
            "This report does not test whether lexical properties explain the differences.",
        ],
    }
    if args.limit:
        print(json.dumps(report, indent=2))
        print("Inspection mode: no output file written.")
        return
    with tempfile.TemporaryDirectory(prefix="s06_", dir=folder) as temp:
        staged = Path(temp) / output_path.name
        staged.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged.replace(output_path)
    print(f"Saved reliability summary for {len(slope)} slopes and {len(level)} levels.")
    print("Approximate typical-estimate reliability: "
          f"slope {report['decay_slope']['typical_estimate_reliability_approximate']:.1%}; "
          f"level {report['retention_level']['typical_estimate_reliability_approximate']:.1%}.")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"Reliability check failed: {exc}") from None
