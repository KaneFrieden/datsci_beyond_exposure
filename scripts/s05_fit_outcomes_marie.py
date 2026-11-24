#!/usr/bin/env python3
"""With this file, we fit a retention level and a time coefficient for each eligible lemma.

Input: s03 events_analysis.parquet, item_inventory.parquet and inventory_report.json.
Output: item_outcomes.parquet and outcomes_report.json in the intermediate folder.
Purpose: The fitted retention level is the log odds of a correct response at a common reference point.
You may use --limit for an inspection run that does not write output files.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402


# here we just quickly define some constants and the columns we expect in the input and output dataframes.
# The reference point is six previous encounters and a lag of seven days, which we use to compute
# the retention level. MIN_REFERENCE_EVENTS checks that enough events have exactly six previous encounters.
# The minimum number of events per lemma comes from s03. LEVEL_SE_WARNING flags uncertain level estimates.
# The model columns are those that are used in the GLM fitting, while the outcome columns are those that will be saved in the output dataframe.
# Each event gives a number of correct answers out of one or more attempts. The binomial model uses both counts,
# so the same proportion from one attempt does not carry as much information as one from many attempts.
REFERENCE_HISTORY_SEEN = 6
REFERENCE_LAG_DAYS = 7.0
MIN_REFERENCE_EVENTS = 200
LEVEL_SE_WARNING = 5.0
MODEL_COLUMNS = (
    "lemma", "delta_days", "history_seen", "history_correct", "history_wrong",
    "session_seen", "session_correct",
)
OUTCOME_COLUMNS = (
    "lemma", "n_events", "retention_level", "retention_level_se", "decay_slope",
    "decay_slope_se", "cov_level_slope", "half_life_days", "half_life_extrapolated",
    "slope_nonnegative", "glm_converged", "level_unstable", "design_condition_number",
    "n_distinct_lags", "n_wrong_sessions", "fit_error",
)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]: # read the input files from the intermediate folder, check that they exist and are valid, and return them as a tuple of dataframes and a dictionary.
    config.require_data_root()
    folder = config.DATA_INTERMEDIATE
    paths = [folder / "events_analysis.parquet", folder / "item_inventory.parquet",
             folder / "inventory_report.json"]
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Missing s03 input: {path}")
    events = pd.read_parquet(paths[0], columns=list(MODEL_COLUMNS))
    inventory = pd.read_parquet(paths[1], columns=["lemma", "n_events", "meets_min_events"])
    report = json.loads(paths[2].read_text(encoding="utf-8"))
    if events.empty or inventory.empty: # just catch a bunch of potential issues with the input data, like missing files, empty dataframes, or inconsistent counts.
        raise ValueError("s03 inputs must not be empty.")
    if events[list(MODEL_COLUMNS)].isna().any().any():
        raise ValueError("The s03 event table has missing model values.")
    if inventory["lemma"].isna().any() or inventory["lemma"].duplicated().any():
        raise ValueError("The s03 inventory needs one row per lemma.")
    if (events["delta_days"] <= 0).any() or (events["session_seen"] <= 0).any():
        raise ValueError("Lag and number of session trials must be positive.")
    if ((events["session_correct"] < 0) |
        (events["session_correct"] > events["session_seen"])).any():
        raise ValueError("Session successes must be between zero and the trial count.")
    if ((events["history_wrong"] < 0) |
        (events["history_seen"] != events["history_correct"] + events["history_wrong"])).any():
        raise ValueError("The history counts are inconsistent.")
    counts = events.groupby("lemma", observed=True).size()
    actual = inventory["lemma"].map(counts)
    if actual.isna().any() or not actual.eq(inventory["n_events"]).all():
        raise ValueError("The inventory event counts do not match the event table.")
    min_events = int(report["min_events_per_lemma"])
    if not inventory["meets_min_events"].eq(inventory["n_events"] >= min_events).all():
        raise ValueError("The inventory eligibility flags do not match its recorded threshold.")
    if len(events) != report["events_after_lag_filter"] or len(inventory) != report["lemmas_after_lag_filter"]:
        raise ValueError("The s03 report does not match its data files.")
    return events, inventory, report


def reference_vector(events: pd.DataFrame) -> tuple[np.ndarray, dict]: # just computing the reference vector for the retention level at the common reference point
    at_reference = events.loc[events["history_seen"] == REFERENCE_HISTORY_SEEN]
    if len(at_reference) < MIN_REFERENCE_EVENTS:
        raise ValueError("Too few events with six previous encounters for the reference point.")
    correct = float(at_reference["history_correct"].mean())
    wrong = float(at_reference["history_wrong"].mean())
    session_seen = float(at_reference["session_seen"].mean())
    vector = np.array([
        1.0, np.log2(REFERENCE_LAG_DAYS), np.log1p(REFERENCE_HISTORY_SEEN),
        np.log1p(correct), np.log1p(wrong), session_seen,
    ])
    details = { # some details that are useful for reporting and debugging
        "history_seen": REFERENCE_HISTORY_SEEN, "lag_days": REFERENCE_LAG_DAYS,
        "history_correct": correct, "history_wrong": wrong,
        "session_seen": session_seen, "n_at_reference": len(at_reference),
        "history_seen_median": float(events["history_seen"].median()),
    }
    return vector, details


def design_matrix(events: pd.DataFrame) -> np.ndarray: # this is the heart of the GLM fitting, where we construct the design matrix for the model.
    # The design matrix has one row per event and one column per model term. log2 makes the time coefficient
    # describe a change per doubling of the lag. log1p also works for zero previous encounters and reduces
    # the scale of large history counts.
    return np.column_stack([
        np.ones(len(events)), np.log2(events["delta_days"].to_numpy(float)),
        np.log1p(events["history_seen"].to_numpy(float)),
        np.log1p(events["history_correct"].to_numpy(float)),
        np.log1p(events["history_wrong"].to_numpy(float)),
        events["session_seen"].to_numpy(float),
    ])


def fit_one_lemma(events: pd.DataFrame, reference: np.ndarray) -> dict: # this function fits a GLM for a single lemma, using the design matrix and the reference vector.
    # It returns a dictionary with the fitted parameters and various diagnostics, including whether the model converged,
    # whether the slope is non-negative, and whether the retention level estimate is unstable.
    # If there are any issues during fitting (e.g., perfect separation, non-finite coefficients), it saves the error message in the output dictionary.
    x = design_matrix(events)
    successes = events["session_correct"].to_numpy(float)
    trials = events["session_seen"].to_numpy(float)
    result = {
        "n_events": len(events), "n_distinct_lags": int(events["delta_days"].nunique()),
        "n_wrong_sessions": int((successes < trials).sum()), "fit_error": None,
        "retention_level": np.nan, "retention_level_se": np.nan,
        "decay_slope": np.nan, "decay_slope_se": np.nan,
        "cov_level_slope": np.nan, "half_life_days": np.nan,
        "half_life_extrapolated": False, "slope_nonnegative": False,
        "glm_converged": False, "level_unstable": False,
        "design_condition_number": float(np.linalg.cond(x)),
    }
    if np.linalg.matrix_rank(x) < x.shape[1]:
        result["fit_error"] = "design matrix has insufficient rank"
        return result
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = sm.GLM(
                np.column_stack([successes, trials - successes]), x,
                family=sm.families.Binomial(),
            ).fit(maxiter=100)
        if not fitted.converged:
            result["fit_error"] = "GLM did not converge"
            return result
        if any("PerfectSeparation" in warning.category.__name__ for warning in caught):
            result["fit_error"] = "perfect separation warning"
            return result
        beta = np.asarray(fitted.params, dtype=float)
        covariance = np.asarray(fitted.cov_params(), dtype=float)
        if not np.isfinite(beta).all() or not np.isfinite(covariance).all():
            result["fit_error"] = "non-finite coefficients or covariance"
            return result
        # we do this because the level combines several coefficients, so its uncertainty must also include how their estimates vary together.
        level_variance = float(reference @ covariance @ reference)
        slope_variance = float(covariance[1, 1])
        if level_variance < 0 or slope_variance < 0:
            result["fit_error"] = "negative estimated variance"
            return result
        level = float(reference @ beta)
        slope = float(beta[1])
        level_se = float(np.sqrt(level_variance))
        slope_se = float(np.sqrt(slope_variance))
        half_life = np.nan
        if slope < 0:
            log2_half_life = np.log2(REFERENCE_LAG_DAYS) - level / slope
            half_life = float(np.exp2(log2_half_life)) if log2_half_life < 1024 else np.inf
        result.update({
            "retention_level": level, "retention_level_se": level_se,
            "decay_slope": slope, "decay_slope_se": slope_se,
            "cov_level_slope": float(reference @ covariance[:, 1]),
            "half_life_days": half_life,
            "half_life_extrapolated": bool(
                slope < 0 and (
                    not np.isfinite(half_life) or
                    half_life < events["delta_days"].min() or
                    half_life > events["delta_days"].max()
                )
            ),
            "slope_nonnegative": bool(slope >= 0),
            "glm_converged": True, "level_unstable": bool(level_se > LEVEL_SE_WARNING),
        })
    except (FloatingPointError, np.linalg.LinAlgError, PerfectSeparationError, ValueError) as exc:
        result["fit_error"] = f"{type(exc).__name__}: {exc}"
    return result


def fit_all(events: pd.DataFrame, inventory: pd.DataFrame, reference: np.ndarray) -> pd.DataFrame: # repeatedly fit the GLM for all eligible lemmas, and return a dataframe with the results. We also print progress every 100 lemmas to keep track of the fitting process.
    eligible = inventory.loc[inventory["meets_min_events"], "lemma"].sort_values().tolist()
    groups = dict(tuple(events.loc[events["lemma"].isin(eligible)].groupby("lemma", observed=True)))
    rows = []
    for number, lemma in enumerate(eligible, start=1):
        rows.append({"lemma": lemma, **fit_one_lemma(groups[lemma], reference)})
        if number % 100 == 0:
            print(f"Fitted {number:,}/{len(eligible):,} lemmas", flush=True)
    return pd.DataFrame(rows, columns=OUTCOME_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="fit only the first N eligible lemmas without writing files")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must not be negative")
    outputs = [config.DATA_INTERMEDIATE / "item_outcomes.parquet",
               config.DATA_INTERMEDIATE / "outcomes_report.json"]
    if not args.limit and any(path.exists() for path in outputs):
        raise RuntimeError("An s05 output already exists; use a fresh data root.")
    events, inventory, inventory_report = read_inputs()
    reference, reference_details = reference_vector(events)
    if args.limit:
        selected = inventory.loc[inventory["meets_min_events"], "lemma"].sort_values().head(args.limit)
        inventory = inventory.loc[inventory["lemma"].isin(selected)]
    outcomes = fit_all(events, inventory, reference)
    fitted = outcomes["glm_converged"]
    report = {
        "s03_min_lag_days": inventory_report["min_lag_days"],
        "s03_min_events_per_lemma": inventory_report["min_events_per_lemma"],
        "reference": reference_details,
        "model": "binomial GLM with logit link; one fit per lemma",
        "predictors": ["log2(delta_days)", "log1p(history_seen)",
                       "log1p(history_correct)", "log1p(history_wrong)", "session_seen"],
        "retention_level_scale": "log odds at the common reference point",
        "decay_slope_scale": "change in log odds per doubling of lag",
        "level_se_warning_threshold": LEVEL_SE_WARNING,
        "eligible_lemmas": len(outcomes), "converged": int(fitted.sum()),
        "failed": int((~fitted).sum()),
        "failure_reasons": outcomes.loc[~fitted, "fit_error"].value_counts().to_dict(),
        "level_unstable": int(outcomes["level_unstable"].sum()),
        "slope_nonnegative": int(outcomes["slope_nonnegative"].sum()),
        "half_life_extrapolated": int(outcomes["half_life_extrapolated"].sum()),
        "median_level_se": float(outcomes.loc[fitted, "retention_level_se"].median()),
        "median_slope_se": float(outcomes.loc[fitted, "decay_slope_se"].median()),
        "notes": [
            "A level-SE warning does not automatically exclude an item or its slope.",
            "A non-negative slope has no finite half-life in this model.",
            "Repeated observations per learner and scheduling are not modelled explicitly.",
        ],
    }
    if args.limit:
        print(json.dumps(report, indent=2))
        print("Inspection mode: no output files written.")
        return
    if len(outcomes) != inventory_report["lemmas_qualified"]:
        raise ValueError("The number of fitted lemmas differs from the s03 report.")
    with tempfile.TemporaryDirectory(prefix="s05_", dir=config.DATA_INTERMEDIATE) as temp:
        stage = Path(temp)
        outcomes.to_parquet(stage / outputs[0].name, index=False)
        (stage / outputs[1].name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for output in outputs:
            (stage / output.name).replace(output)
    print(f"Saved {len(outcomes):,} lemma fits; {report['converged']:,} converged; "
          f"{report['level_unstable']:,} have a level-SE warning.")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"Outcome fitting failed: {exc}") from None
