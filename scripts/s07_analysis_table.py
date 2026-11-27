#!/usr/bin/env python3
"""With s07, we bring together the lemma inventory, lexical features and
estimated outcomes. We check that the three sources match and keep one row
per eligible lemma. We also mark which lemmas have the data needed for later
models and report how many are missing. The choice of controls and the
standardisation of predictors happen later.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

# the input table and report are produced by s03, s04 and s05.
# The output table and report are then used in s08 and s09.
def input_table(name: str) -> pd.DataFrame:
    path = config.DATA_INTERMEDIATE / name
    if not path.is_file():
        raise RuntimeError(f"Missing input table: {path}")
    table = pd.read_parquet(path)
    if table.empty or "lemma" not in table or table["lemma"].isna().any():
        raise ValueError(f"Input table needs non-empty lemma values: {path}")
    if table["lemma"].duplicated().any():
        raise ValueError(f"Input table contains duplicate lemmas: {path}")
    return table

# s03, s04 and s05 each write a report. We read these reports to check
# that their counts agree with the tables before joining them.
def input_report(name: str) -> dict:
    path = config.DATA_INTERMEDIATE / name
    if not path.is_file():
        raise RuntimeError(f"Missing input report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

# we only use an estimate if it is finite, has a finite positive standard error, and the GLM converged.
def usable_estimate(table: pd.DataFrame, value: str, error: str) -> pd.Series:
    values = pd.to_numeric(table[value], errors="coerce").to_numpy(dtype=float)
    errors = pd.to_numeric(table[error], errors="coerce").to_numpy(dtype=float)
    return pd.Series(
        table["glm_converged"].to_numpy(dtype=bool)
        & np.isfinite(values) & np.isfinite(errors) & (errors > 0),
        index=table.index,
    )

# here we build the analysis table and report, checking that the three sources match and
# marking which lemmas have usable estimates.
def build_table() -> tuple[pd.DataFrame, dict]:
    inventory = input_table("item_inventory.parquet")
    features = input_table("feature_table.parquet")
    outcomes = input_table("item_outcomes.parquet")
    inventory_report = input_report("inventory_report.json")
    feature_report = input_report("feature_report.json")
    outcomes_report = input_report("outcomes_report.json")

    if len(inventory) != inventory_report["lemmas_after_lag_filter"]:
        raise ValueError("The s03 inventory and report disagree.")
    if len(features) != feature_report["inventory_lemmas"] or set(features["lemma"]) != set(inventory["lemma"]):
        raise ValueError("The s04 features do not cover exactly the s03 inventory.")
    if len(outcomes) != outcomes_report["eligible_lemmas"]:
        raise ValueError("The s05 outcomes and report disagree.")
    eligible = inventory.loc[inventory["meets_min_events"]].copy()
    if len(eligible) != inventory_report["lemmas_qualified"] or set(eligible["lemma"]) != set(outcomes["lemma"]):
        raise ValueError("The s05 outcome lemmas differ from the s03 eligible lemmas.")

    # two earlier stages had the same values. we compare them before keeping one copy.
    tags = inventory[["lemma", "frm_mean_grammatical_tags"]].merge(
        features[["lemma", "frm_mean_grammatical_tags"]], on="lemma", validate="one_to_one",
        suffixes=("_s03", "_s04"),
    )
    if not np.allclose(tags["frm_mean_grammatical_tags_s03"],
                       tags["frm_mean_grammatical_tags_s04"], equal_nan=True):
        raise ValueError("The s03 and s04 grammatical-tag values differ.")
    event_counts = eligible[["lemma", "n_events", "n_distinct_lags"]].merge(
        outcomes[["lemma", "n_events", "n_distinct_lags"]], on="lemma", validate="one_to_one",
        suffixes=("_s03", "_s05"),
    )
    if not event_counts["n_events_s03"].eq(event_counts["n_events_s05"]).all():
        raise ValueError("The s03 and s05 event counts differ.")
    if not event_counts["n_distinct_lags_s03"].eq(event_counts["n_distinct_lags_s05"]).all():
        raise ValueError("The s03 and s05 distinct-lag counts differ.")

    feature_columns = [name for name in features.columns if name != "frm_mean_grammatical_tags"]
    # we start with all eligible lemmas, then merge in the lexical features and outcomes. we mark which lemmas have usable estimates.
    outcome_columns = [name for name in outcomes.columns if name not in {"n_events", "n_distinct_lags"}]
    table = eligible.merge(features[feature_columns], on="lemma", how="left", validate="one_to_one")
    table = table.merge(outcomes[outcome_columns], on="lemma", how="left", validate="one_to_one")
    if len(table) != len(eligible) or table["lemma"].duplicated().any():
        raise ValueError("The merge lost or duplicated eligible lemmas.")

    predictors = list(feature_report["features"])
    if len(predictors) != 12 or any(name not in table for name in predictors):
        raise ValueError("The s04 report must define the twelve planned lexical predictors.")
    # we keep incomplete rows visible, but mark the lemmas usable for both outcome models.
    lexical_values = table[predictors].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    table["has_all_lexical_features"] = np.isfinite(lexical_values).all(axis=1)
    # we exclude uncertain levels, but keep their slopes when the slope estimate is usable.
    table["level_usable"] = usable_estimate(table, "retention_level", "retention_level_se") & ~table["level_unstable"]
    table["slope_usable"] = usable_estimate(table, "decay_slope", "decay_slope_se")
    table["main_sample"] = table["has_all_lexical_features"] & table["level_usable"] & table["slope_usable"]
    table = table.sort_values("lemma").reset_index(drop=True)

    possible_controls = [
        "exp_mean_log1p_history_seen", "exp_mean_log1p_history_correct",
        "exp_mean_log1p_history_wrong", "n_learners", "exp_mean_session_seen",
        "pos_dominant",
    ]
    if any(name not in table for name in possible_controls):
        raise ValueError("The s03 inventory is missing a planned control column.")
    report = {
        "inventory_lemmas": len(inventory),
        "eligible_lemmas": len(table),
        "has_all_lexical_features": int(table["has_all_lexical_features"].sum()),
        "level_usable": int(table["level_usable"].sum()),
        "slope_usable": int(table["slope_usable"].sum()),
        "main_sample_both_outcomes": int(table["main_sample"].sum()),
        "missing_by_lexical_predictor": {name: int(table[name].isna().sum()) for name in predictors},
        "missing_by_possible_control": {name: int(table[name].isna().sum()) for name in possible_controls},
        "lemmas_with_mixed_pos_below_80_percent": int(table["pos_dominant_share"].lt(0.8).sum()),
        "predictors": predictors,
        "notes": [
            "All eligible lemmas remain in the table, including incomplete rows.",
            "main_sample is a common candidate sample for comparing level and slope models.",
            "The models must standardise predictors on their chosen analysis sample, not here.",
            "Earlier practice of this word is not a measure of curriculum position.",
            "The choice of primary control model and weighting is still pending for s08 and s09.",
        ],
    }
    if report["has_all_lexical_features"] != feature_report["complete_cases_qualified"]:
        raise ValueError("s07 complete-case count differs from the s04 coverage report.")
    return table, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", type=int, default=0,
                        help="print N selected rows and coverage without writing output files")
    args = parser.parse_args()
    if args.inspect < 0:
        parser.error("--inspect must not be negative")
    config.require_data_root()
    output_dir = config.DATA_PROCESSED
    table_path = output_dir / "analysis_table.parquet"
    report_path = output_dir / "analysis_table_report.json"
    if not args.inspect and (table_path.exists() or report_path.exists()):
        raise RuntimeError("An s07 output already exists; use a fresh data root.")
    table, report = build_table()
    if args.inspect:
        print(json.dumps(report, indent=2))
        print(table[["lemma", "n_events", "has_all_lexical_features", "level_usable",
                     "slope_usable", "main_sample"]].head(args.inspect).to_string(index=False))
        print("Inspection mode: no output files written.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="s07_", dir=output_dir) as temp:
        staging = Path(temp)
        table.to_parquet(staging / table_path.name, index=False)
        (staging / report_path.name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (staging / table_path.name).replace(table_path)
        (staging / report_path.name).replace(report_path)
    print(f"Saved {len(table)} eligible lemmas; {report['main_sample_both_outcomes']} "
          "have all lexical features and both usable outcomes.")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"Analysis table failed: {exc}") from None
