#!/usr/bin/env python3

"""This file looks at the learning events and lexeme descriptions, inspects possible exclusions, 
and builds an inventory of lemmas with explicit selection limits. You can run it without arguments to 
inspect the data and see how many events and lemmas would remain after various exclusions.
Or you can run it with --min-lag-days and --min-events to build an inventory candidate, 
which will be saved in Parquet format along with a JSON report. The script checks for missing lexeme descriptions, 
invalid counts, and other potential issues before proceeding with the analysis."""


from __future__ import annotations

import json
import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

# The script defines two main functions: inspect_selection and build_inventory.
# inspect_selection takes the joined events and lexemes, applies a series of exclusions based on flags
# (like wildcards, prpers, non-word lemmas, etc.), and returns the remaining events along with a report summarizing 
# the exclusions and candidate lag and count thresholds.
# build_inventory takes the filtered events, applies the minimum lag and event thresholds, aggregates the data by lemma, 
# and returns the analysis, inventory, and a final report with statistics about the qualified lemmas and events.

def inspect_selection(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]: 
    flags = ("is_wildcard", "is_prpers", "is_nonword_lemma", # these are the flags we will use to exclude events and lexemes from the inventory
             "has_no_regular_pos", "is_unparsed")
    individual = { # here we count how many events and distinct lexemes are flagged for each exclusion criterion
        name: {"events": int(events[name].sum()),
               "lexemes": int(events.loc[events[name], "lexeme_id"].nunique())}
        for name in flags
    }
    kept = events
    chain = [{"step": "selected language pair", "events": len(kept)}] # we will keep track of how many events remain after each exclusion step
    for name in flags:
        removed = int(kept[name].sum()) 
        kept = kept.loc[~kept[name]]
        chain.append({"step": f"after excluding {name}",
                      "removed_at_this_step": removed, "events": len(kept)})
    if not (kept["delta_days"] >= 0).all(): 
        raise ValueError("Some time intervals are missing or negative.")

    lag_options = {} # we will compute how many events and lemmas remain after applying different minimum lag thresholds (0, 0.25, 0.5, 1, 2 days)
    for days in (0, 0.25, 0.5, 1, 2):
        subset = kept.loc[kept["delta_days"] >= days]
        counts = subset.groupby("lemma").size()
        lag_options[str(days)] = {
            "events": len(subset), "lemmas": len(counts),
            "lemmas_at_100_events": int((counts >= 100).sum()),
            "lemmas_at_200_events": int((counts >= 200).sum()),
            "lemmas_at_400_events": int((counts >= 400).sum()),
        }

    pos_counts = kept.groupby(["lemma", "pos"]).size().rename("events").reset_index() 
    pos_totals = pos_counts.groupby("lemma")["events"].transform("sum")
    pos_counts["share"] = pos_counts["events"] / pos_totals
    dominant = pos_counts.sort_values(["lemma", "events", "pos"],
                                      ascending=[True, False, True]).drop_duplicates("lemma")
    mixed = pos_counts.groupby("lemma")["pos"].nunique()
    report = {
        "individual_flags_before_exclusion": individual,
        "filter_chain": chain,
        "lag_days_after_structural_exclusions": {
            "min": float(kept["delta_days"].min()),
            "median": float(kept["delta_days"].median()),
            "max": float(kept["delta_days"].max()),
        },
        "candidate_lag_and_count_thresholds": lag_options,
        "word_classes": {
            "lemmas_with_multiple_tags": int((mixed > 1).sum()),
            "lemmas_with_dominant_tag_below_80_percent": int((dominant["share"] < 0.8).sum()),
            "examples_low_dominance": dominant.loc[dominant["share"] < 0.8]
                .sort_values("share").head(15)[["lemma", "pos", "share"]].to_dict("records"),
        },
    }
    return kept, report

# Here we build an inventory of lemmas with explicit selection limits. 
# We filter the events based on the minimum lag and event thresholds,
# aggregate the data by lemma, and compute various statistics for the final report.
def build_inventory(events: pd.DataFrame, min_lag_days: float,
                    min_events: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    analysis = events.loc[events["delta_days"] >= min_lag_days].copy()
    analysis["history_wrong"] = analysis["history_seen"] - analysis["history_correct"]
    if (analysis["history_wrong"] < 0).any():
        raise ValueError("Some events have more previous correct answers than previous attempts.")
    for source in ("history_seen", "history_correct", "history_wrong"):
        analysis[f"log1p_{source}"] = np.log1p(analysis[source])
    groups = analysis.groupby("lemma")
    inventory = groups.agg(  # 
        n_events=("lexeme_id", "size"),
        n_lexemes=("lexeme_id", "nunique"),
        n_learners=("user_id", "nunique"),
        session_seen_total=("session_seen", "sum"),
        session_correct_total=("session_correct", "sum"),
        median_lag_days=("delta_days", "median"),
        min_lag_days=("delta_days", "min"),
        max_lag_days=("delta_days", "max"),
        n_distinct_lags=("delta_days", "nunique"),
        mean_history_seen=("history_seen", "mean"),
        exp_mean_log1p_history_seen=("log1p_history_seen", "mean"),
        exp_mean_log1p_history_correct=("log1p_history_correct", "mean"),
        exp_mean_log1p_history_wrong=("log1p_history_wrong", "mean"),
        exp_mean_session_seen=("session_seen", "mean"),
        frm_mean_grammatical_tags=("n_grammatical_tags", "mean"),
    ).reset_index()
    pos = analysis.groupby(["lemma", "pos"]).size().rename("n").reset_index()
    pos = pos.sort_values(["lemma", "n", "pos"], ascending=[True, False, True])
    dominant = pos.drop_duplicates("lemma").copy()
    pos_total = pos.groupby("lemma")["n"].sum()
    dominant["pos_dominant_share"] = dominant["n"] / dominant["lemma"].map(pos_total)
    inventory = inventory.merge(
        dominant[["lemma", "pos", "pos_dominant_share"]].rename(columns={"pos": "pos_dominant"}),
        on="lemma", validate="one_to_one",
    )
    inventory["recall_rate"] = inventory["session_correct_total"] / inventory["session_seen_total"]
    if inventory["recall_rate"].isna().any() or (inventory["recall_rate"] > 1).any():
        raise ValueError("Some lemmas have invalid answer counts.")
    inventory["meets_min_events"] = inventory["n_events"] >= min_events
    qualified = inventory.loc[inventory["meets_min_events"]]
    report = {"min_lag_days": min_lag_days, "min_events_per_lemma": min_events,
              "events_after_lag_filter": len(analysis),
              "lemmas_after_lag_filter": len(inventory),
              "lemmas_qualified": len(qualified),
              "events_in_qualified_lemmas": int(qualified["n_events"].sum())}
    return analysis, inventory, report

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-lag-days", type=float)
    parser.add_argument("--min-events", type=int)
    args = parser.parse_args()
    if (args.min_lag_days is None) != (args.min_events is None):
        parser.error("Give both thresholds for an inventory, or neither for inspection only.")
    if args.min_lag_days is not None and (args.min_lag_days <= 0 or args.min_events < 1):
        parser.error("An inventory needs a positive minimum lag and at least one event.")
    config.require_data_root()
    language_pair = f"{config.LEARNING_LANGUAGE}_{config.UI_LANGUAGE}"
    events_path = config.DATA_INTERMEDIATE / f"traces_{language_pair}.parquet"
    lexemes_path = config.DATA_INTERMEDIATE / "lexemes.parquet"
    report_path = config.DATA_INTERMEDIATE / "inventory_join_report.json"
    inspection_path = config.DATA_INTERMEDIATE / "inventory_inspection.json"
    analysis_path = config.DATA_INTERMEDIATE / "events_analysis.parquet"
    inventory_path = config.DATA_INTERMEDIATE / "item_inventory.parquet"
    final_report_path = config.DATA_INTERMEDIATE / "inventory_report.json"

    for source in (events_path, lexemes_path):
        if not source.is_file():
            raise RuntimeError(f"Hey! The input is missing: {source}")
    outputs = [report_path, inspection_path]
    if args.min_lag_days is not None:
        outputs += [analysis_path, inventory_path, final_report_path]
    if any(path.exists() for path in outputs):
        raise RuntimeError("Oops! An s03 output already exists; choose a fresh data root.")

    # for now we only need the lexeme IDs from the events and
    # the IDs and descriptions from s02, but we will use other columns later
    events = pd.read_parquet(events_path, columns=["lexeme_id"])
    lexemes = pd.read_parquet(lexemes_path, columns=["lexeme_id", "lexeme_string"])
    if events.empty or lexemes.empty:
        raise ValueError("One of the input tables is empty.")
    if events["lexeme_id"].isna().any() or lexemes.isna().any().any():
        raise ValueError("A lexeme ID or description is missing.")
    # every lexeme ID should have only one description, otherwise one event
    # could appear several times after we connect the tables.
    if lexemes["lexeme_id"].duplicated().any():
        raise ValueError("The s02 table has more than one row for a lexeme ID.")

    joined = events.merge(lexemes, on="lexeme_id", how="left", validate="many_to_one", indicator=True)
    missing = joined["_merge"].eq("left_only")
    if len(joined) != len(events):
        raise ValueError("The join changed the number of learning events.")
    if missing.any():
        examples = events.loc[missing, "lexeme_id"].drop_duplicates().head(5).tolist()
        raise ValueError(f"{int(missing.sum())} events have no lexeme description. Example IDs: {examples}")

    report = {
        "input_events": len(events),
        "input_lexemes": len(lexemes),
        "joined_events": len(joined),
        "events_without_lexeme": int(missing.sum()),
        "distinct_event_lexeme_ids": int(events["lexeme_id"].nunique()),
    }
    full_events = pd.read_parquet(events_path)
    detail = pd.read_parquet(lexemes_path, columns=[
        "lexeme_id", "lemma", "pos", "is_wildcard", "is_prpers",
        "is_nonword_lemma", "has_no_regular_pos", "is_unparsed",
        "n_grammatical_tags",
    ])
    full_events = full_events.merge(detail, on="lexeme_id", how="left", validate="many_to_one")
    if len(full_events) != len(events):
        raise ValueError("The full join changed the number of learning events.")
    kept, inspection = inspect_selection(full_events)
    analysis = inventory = final_report = None
    if args.min_lag_days is not None:
        analysis, inventory, final_report = build_inventory(
            kept, args.min_lag_days, args.min_events
        )

    with tempfile.TemporaryDirectory(prefix="s03_", dir=config.DATA_INTERMEDIATE) as temp:
        staging = Path(temp)
        for path, content in ((report_path, report), (inspection_path, inspection)):
            (staging / path.name).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
        if analysis is not None:
            analysis.to_parquet(staging / analysis_path.name, index=False)
            inventory.to_parquet(staging / inventory_path.name, index=False)
            (staging / final_report_path.name).write_text(
                json.dumps(final_report, indent=2) + "\n", encoding="utf-8"
            )
        for target in outputs:
            (staging / target.name).replace(target)

    print(f"Joined {len(joined):,} events to {len(lexemes):,} lexemes; no missing IDs.")
    print(f"Saved: {report_path}")
    print(f"Saved: {inspection_path}")
    if analysis is not None:
        print(f"Inventory: {len(inventory):,} lemmas, {len(analysis):,} events; "
              f"{int(inventory['meets_min_events'].sum()):,} lemmas reach the event threshold.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Inventory join failed: {exc}") from None
