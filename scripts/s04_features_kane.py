#!/usr/bin/env python3
"""
With this script, we calculate lexical features for every lemma in the s03 inventory.

Inputs are the s03 inventory, five lexical norm files, English fastText vectors,
and a local English-to-Spanish translation model. The script writes one feature
table, a translation audit table, and a coverage report. You can use the --sample flag for a
small run that does not write output files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402


SOURCES = { # we use a dict of dicts to store the source information, including the filename and pinned SHA-256 value for each source
    "norms": {
        "aoa": (
            "AoA_51715_words.zip",
            "ce1c17a65d3b0b1130b6bcfda47a416b4f6625d187d1fe7d83e618453c3d4f89",
        ),
        "concreteness": (
            "Concreteness_ratings_Brysbaert_et_al_BRM.txt",
            "0b4082dbd38585b0ee1fd258145b7a50592f8d0d98e5fc6b6844ceef3cd8ecc8",
        ),
        "prevalence": (
            "WordprevalencesSupplementaryfilefirstsubmission.xlsx",
            "3b2cbee33f7a4eab50e71a7a09461fe28c32bb16c53435327b94cf7b951b76af",
        ),
        "subtlex_us": (
            "SUBTLEX-US_frequency_list_with_PoS_information_final_text_version.zip",
            "f9d2e082439cd7fd060768a7f1e717f00ea0a2d2268b8340f39dae3c66bf6c82",
        ),
        "warriner": (
            "Ratings_Warriner_et_al.csv",
            "85f6d7e35069b0eff3d9d88c4dbe08e6ae7dbdb81eac64be3f89ed0f5079d8e3",
        ),
    },
    "vectors": {
        "fasttext_en_vec": (
            "crawl-300d-2M.vec.zip",
            "5bfffffbabdab299d4c9165c47275e8f982807a6eaca37ee1f71d3a79ddb544d",
        ),
    },
}

# here we define some constants that will be used throughout the script, including the model name and revision for the translation model,
# a set of Spanish determiners, the number of neighbours to consider for semantic distance calculations,
# and a default vocabulary limit for loading fastText vectors
MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"
MODEL_REVISION = "5bc4493d463cf000c1f0b50f8d56886a392ed4ab"
SPANISH_DETERMINERS = {"el", "la", "los", "las", "un", "una", "unos", "unas", "lo"}
N_NEIGHBOURS = 40
DEFAULT_VOCAB_LIMIT = 200_000

PREDICTORS = (
    "fam_log_frequency", "fam_age_of_acquisition", "fam_prevalence",
    "mea_concreteness", "mea_semantic_distance", "mea_valence", "mea_arousal",
    "frm_length_chars", "frm_length_syllables", "frm_letters_per_phoneme",
    "frm_mean_grammatical_tags", "trn_norm_levenshtein_es",
)


def sha256_of(path: Path) -> str: # we define a function to compute the SHA-256 hash of a file, which is used to verify the integrity of source files
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources() -> None: # we define a function to verify the integrity of the source files by checking their existence, size, and SHA-256 hash against the pinned values in the SOURCES dictionary
    config.require_data_root()
    for folder, entries in SOURCES.items():
        source_dir = config.DATA_RAW / folder
        manifest_path = source_dir / f"{folder}_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Missing source manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        for key, (filename, pinned_sha256) in entries.items():
            record = manifest[key]
            if record["filename"] != filename or record["sha256"].lower() != pinned_sha256:
                raise RuntimeError(f"Unexpected source details in {manifest_path}: {key}")
            source_path = source_dir / filename
            if not source_path.is_file():
                raise RuntimeError(f"Missing source file: {source_path}")
            if source_path.stat().st_size != int(record["bytes"]):
                raise RuntimeError(f"File size differs from the manifest: {source_path}")
            if sha256_of(source_path) != pinned_sha256:
                raise RuntimeError(f"SHA-256 differs from the pinned value: {source_path}")
            print(f"Verified {folder}/{filename}")

    print("All six lexical source files match their pinned SHA-256 values.")

# a helper function to read a tab-separated or Excel file from a zip archive and return a pandas DataFrame.
def read_zipped(path: Path, inner_name: str, *, excel: bool = False) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        if inner_name not in archive.namelist():
            raise ValueError(f"Expected {inner_name} inside {path}")
        with archive.open(inner_name) as source:
            if excel:
                return pd.read_excel(io.BytesIO(source.read()), engine="openpyxl")
            return pd.read_csv(source, sep="\t")

# this function is used to clean and tidy a DataFrame containing lexical norms, including stripping whitespace, case folding,
# converting numeric columns, and collapsing duplicate lemmas by averaging their values
def tidy_norm(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    frame = frame.copy()
    frame["lemma"] = frame["lemma"].astype("string").str.strip().str.casefold()
    frame = frame.loc[frame["lemma"].notna() & frame["lemma"].ne("")]
    for column in frame.columns.drop("lemma"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    duplicates = len(frame) - frame["lemma"].nunique()
    return frame.groupby("lemma", as_index=False).mean(numeric_only=True), duplicates

# we define a function to load and process the lexical norms from the source files, including AoA, SUBTLEX-US,
# concreteness, Warriner, and prevalence norms. The function merges these norms with the inventory lemmas and
# returns a DataFrame of features along with a report of the number of matched lemmas and duplicates for each norm source
def load_norms(lemmas: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw = config.DATA_RAW / "norms"
    aoa = read_zipped(raw / SOURCES["norms"]["aoa"][0], "AoA_51715_words.xlsx", excel=True)
    aoa = pd.DataFrame({
        "lemma": aoa["Word"],
        "fam_age_of_acquisition": aoa["AoA_Kup_lem"],
        "fam_prevalence": aoa["Perc_known_lem"],
        "frm_length_syllables": aoa["Nsyll"],
        "aoa_nletters": aoa["Nletters"],
        "aoa_nphon": aoa["Nphon"],
    })
    aoa, aoa_duplicates = tidy_norm(aoa) # we tidy the AoA DataFrame and count the number of duplicate lemmas
    aoa["frm_length_syllables"] = aoa["frm_length_syllables"].where(
        aoa["frm_length_syllables"] > 0
    )
    aoa["frm_letters_per_phoneme"] = (
        aoa["aoa_nletters"] / aoa["aoa_nphon"].where(aoa["aoa_nphon"] > 0)
    )
    aoa = aoa.drop(columns=["aoa_nletters", "aoa_nphon"])

    with zipfile.ZipFile(raw / SOURCES["norms"]["subtlex_us"][0]) as archive: # now we read the SUBTLEX-US frequency list from the zip archive and tidy it
        inner = "SUBTLEX-US frequency list with PoS information text version.txt"
        with archive.open(inner) as source:
            subtlex = pd.read_csv(source, sep="\t", usecols=["Word", "Lg10WF"])
    subtlex, subtlex_duplicates = tidy_norm(subtlex.rename(columns={
        "Word": "lemma", "Lg10WF": "fam_log_frequency",
    }))

    concrete = pd.read_csv( # here we read the concreteness ratings from a text file and tidy it as well
        raw / SOURCES["norms"]["concreteness"][0], sep="\t",
        usecols=["Word", "Conc.M"],
    ).rename(columns={"Word": "lemma", "Conc.M": "mea_concreteness"})
    concrete, concrete_duplicates = tidy_norm(concrete)

    warriner = pd.read_csv( # same for the Warriner et al. ratings, which include valence and arousal measures
        raw / SOURCES["norms"]["warriner"][0],
        usecols=["Word", "V.Mean.Sum", "A.Mean.Sum"],
    ).rename(columns={
        "Word": "lemma", "V.Mean.Sum": "mea_valence", "A.Mean.Sum": "mea_arousal",
    })
    warriner, warriner_duplicates = tidy_norm(warriner)

    prevalence = pd.read_excel( # finally, we read the prevalence ratings from an Excel file and tidy it
        raw / SOURCES["norms"]["prevalence"][0],
        sheet_name="Prevalence", usecols=["Word", "Prevalence"], engine="openpyxl",
    ).rename(columns={"Word": "lemma", "Prevalence": "fam_prevalence_dedicated"})
    prevalence, prevalence_duplicates = tidy_norm(prevalence)

    output = lemmas[["lemma", "frm_mean_grammatical_tags"]].copy()
    output["frm_length_chars"] = output["lemma"].str.len().astype("float64")
    frames = {
        "aoa": (aoa, aoa_duplicates),
        "subtlex_us": (subtlex, subtlex_duplicates),
        "concreteness": (concrete, concrete_duplicates),
        "warriner": (warriner, warriner_duplicates),
        "prevalence": (prevalence, prevalence_duplicates),
    }
    report = {}
    for name, (frame, duplicates) in frames.items():
        # we keep all words, even those with missing ratings, so we can see which source
        # reduces the sample. Dropping them now would also prevent models without that rating.
        output = output.merge(frame, on="lemma", how="left", validate="one_to_one")
        columns = frame.columns.drop("lemma")
        matched = int(output[list(columns)].notna().any(axis=1).sum())
        if matched == 0:
            raise ValueError(f"No inventory lemmas matched the {name} source.")
        report[name] = {
            "source_words": len(frame), "duplicate_rows_collapsed": int(duplicates),
            "lemmas_with_a_value": matched,
        }
    if output["fam_prevalence"].dropna().between(0, 1).eq(False).any():
        raise ValueError("AoA percent-known values are outside 0 to 1.")
    return output, report

# we hereby define a function to load the fastText vectors from the zip archive,
# normalize them, and return a matrix of vectors, a mapping of words to their indices in the matrix,
# a mapping of target words to their vectors, and a count of invalid vector rows.
# this is meant to be used for calculating semantic features for the inventory lemmas by computing the mean semantic distance
# to their nearest neighbours in the fastText vector space.
def load_vectors(targets: set[str], vocab_limit: int) -> tuple[np.ndarray, dict, dict, int]:
    archive_path = config.DATA_RAW / "vectors" / SOURCES["vectors"]["fasttext_en_vec"][0]
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("crawl-300d-2M.vec") as source:
            header = source.readline().decode("utf-8").split()
            if len(header) != 2:
                raise ValueError("Unexpected fastText header.")
            total, dimension = map(int, header)
            matrix = np.empty((min(vocab_limit, total), dimension), dtype=np.float32)
            neighbor_index: dict[str, int] = {}
            target_vectors: dict[str, np.ndarray] = {}
            invalid = 0
            for raw_line in source:
                word_bytes, separator, values = raw_line.partition(b" ")
                if not separator:
                    invalid += 1
                    continue
                word = word_bytes.decode("utf-8").casefold()
                if len(neighbor_index) >= vocab_limit and word not in targets:
                    continue
                vector = np.fromstring(values.decode("utf-8"), sep=" ", dtype=np.float32)
                if len(vector) != dimension or not np.isfinite(vector).all():
                    invalid += 1
                    continue
                if len(neighbor_index) < vocab_limit and word not in neighbor_index:
                    neighbor_index[word] = len(neighbor_index)
                    matrix[len(neighbor_index) - 1] = vector
                if word in targets and word not in target_vectors:
                    target_vectors[word] = vector
    matrix = matrix[:len(neighbor_index)]
    if len(matrix) <= N_NEIGHBOURS:
        raise ValueError("The fastText neighbour vocabulary is too small.")
    lengths = np.linalg.norm(matrix, axis=1)
    matrix /= np.where(lengths > 0, lengths, 1)[:, None]
    return matrix, neighbor_index, target_vectors, invalid

# with the following function, we calculate semantic features for the inventory lemmas by computing the mean semantic distance to their nearest neighbours
# in the fastText vector space.
def semantic_features(lemmas: pd.Series, vocab_limit: int, batch_size: int) -> tuple[pd.DataFrame, dict]:
    target_set = set(lemmas)
    matrix, neighbor_index, vectors, invalid = load_vectors(target_set, vocab_limit)
    present = [word for word in lemmas if word in vectors]
    distances = np.full(len(lemmas), np.nan, dtype=np.float64)
    lookup = {word: row for row, word in enumerate(lemmas)}
    for start in range(0, len(present), batch_size):
        batch = present[start:start + batch_size]
        target_matrix = np.stack([vectors[word] for word in batch])
        lengths = np.linalg.norm(target_matrix, axis=1)
        target_matrix /= np.where(lengths > 0, lengths, 1)[:, None]
        similarities = target_matrix @ matrix.T
        for row, word in enumerate(batch):
            if word in neighbor_index: # counting a word's perfect similarity with itself would make its average distance artificially small
                similarities[row, neighbor_index[word]] = -np.inf
        nearest = np.argpartition(-similarities, kth=N_NEIGHBOURS - 1, axis=1)[:, :N_NEIGHBOURS]
        scores = np.take_along_axis(similarities, nearest, axis=1)
        for row, word in enumerate(batch):
            distances[lookup[word]] = float((1 - scores[row]).mean())
    result = pd.DataFrame({
        "lemma": lemmas.to_numpy(), "mea_semantic_distance": distances,
        "in_fasttext_vocab": np.isfinite(distances),
    })
    report = {
        "neighbours": N_NEIGHBOURS, "neighbour_vocab_limit": vocab_limit,
        "neighbour_selection": "first N distinct case-folded words in fastText frequency order",
        "neighbour_vocab_loaded": len(matrix), "targets_in_full_vector_file": len(present),
        "targets_missing": len(lemmas) - len(present), "invalid_vector_rows": invalid,
    }
    return result, report


def levenshtein(left: str, right: str) -> int: # just a simple implementation of the Levenshtein distance algorithm to compute the
    # edit distance between two strings, which is used to calculate the normalized Levenshtein distance between English lemmas and their Spanish translations.
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1, previous[column] + 1,
                previous[column - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def clean_translation(value: str) -> tuple[str, bool]: # basically a function to clean the Spanish translations by stripping leading
    # determiners and returning the cleaned translation along with a boolean indicating whether a determiner was stripped
    words = value.casefold().strip().split()
    stripped = len(words) > 1 and words[0] in SPANISH_DETERMINERS
    if stripped:
        words = words[1:]
    return " ".join(words), stripped

# main function to generate Spanish translations for the inventory lemmas using a MarianMT translation model,
# clean the translations, and compute various features such as normalized Levenshtein distance,
# whether the translation is identical to the original lemma, and whether it is a multiword expression
# The MarianMT model is a pre-trained neural machine translation model from the Hugging Face Transformers library, and we use it to translate English lemmas to Spanish in batches for efficiency.
def translation_features(
    lemmas: pd.Series, model_snapshot: Path, batch_size: int,
) -> tuple[pd.DataFrame, dict]:
    if model_snapshot.name != MODEL_REVISION:
        raise ValueError(f"Translation model must use revision {MODEL_REVISION}.")
    for filename in ("config.json", "pytorch_model.bin", "source.spm", "target.spm"):
        if not (model_snapshot / filename).is_file():
            raise RuntimeError(f"Translation model is incomplete: {model_snapshot / filename}")
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(model_snapshot, local_files_only=True)
    model = MarianMTModel.from_pretrained(model_snapshot, local_files_only=True)
    model.eval()
    translations = []
    with torch.inference_mode():
        for start in range(0, len(lemmas), batch_size):
            batch = lemmas.iloc[start:start + batch_size].tolist()
            encoded = tokenizer(batch, return_tensors="pt", padding=True)
            generated = model.generate(
                **encoded, num_beams=1, do_sample=False, max_new_tokens=24,
            )
            translations.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    del model
    audit = pd.DataFrame({"lemma": lemmas.to_numpy(), "es_translation": translations})
    cleaned = audit["es_translation"].map(clean_translation)
    audit["es_cleaned"] = cleaned.map(lambda pair: pair[0])
    audit["determiner_stripped"] = cleaned.map(lambda pair: pair[1])
    audit["is_echo"] = audit["es_cleaned"].eq(audit["lemma"])
    audit["is_multiword"] = audit["es_cleaned"].str.contains(" ", regex=False)
    audit["trn_norm_levenshtein_es"] = [
        levenshtein(english, spanish) / max(len(english), len(spanish)) if spanish else np.nan
        for english, spanish in zip(audit["lemma"], audit["es_cleaned"])
    ]
    report = {
        "model": MODEL_NAME, "revision": MODEL_REVISION,
        "decoding": "greedy, no sampling", "translated_lemmas": len(audit),
        "determiners_stripped": int(audit["determiner_stripped"].sum()),
        "identical_or_echo": int(audit["is_echo"].sum()),
        "multiword_after_stripping": int(audit["is_multiword"].sum()),
        "empty_translation": int(audit["es_cleaned"].eq("").sum()),
        "echo_examples": audit.loc[audit["is_echo"], "lemma"].head(20).tolist(),
    }
    return audit, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0, help="inspect N lemmas without writing outputs")
    parser.add_argument("--vocab-limit", type=int, default=DEFAULT_VOCAB_LIMIT)
    parser.add_argument("--semantic-batch", type=int, default=32)
    parser.add_argument("--translation-batch", type=int, default=32)
    parser.add_argument("--model-snapshot", type=Path, default=(
        config.DATA_RAW.parent / "models" / "huggingface" / "hub"
        / "models--Helsinki-NLP--opus-mt-en-es" / "snapshots" / MODEL_REVISION
    ))
    args = parser.parse_args()
    if args.sample < 0 or args.vocab_limit <= N_NEIGHBOURS:
        parser.error("The sample must be non-negative and the vocabulary must exceed 40 words.")
    if args.semantic_batch < 1 or args.translation_batch < 1:
        parser.error("Both batch sizes must be positive.")

    config.require_data_root()
    inventory_path = config.DATA_INTERMEDIATE / "item_inventory.parquet"
    if not inventory_path.is_file():
        raise RuntimeError(f"Missing s03 inventory: {inventory_path}")
    outputs = [
        config.DATA_INTERMEDIATE / "feature_table.parquet",
        config.DATA_INTERMEDIATE / "translation_audit.parquet",
        config.DATA_INTERMEDIATE / "feature_report.json",
    ]
    if not args.sample and any(path.exists() for path in outputs):
        raise RuntimeError("An s04 output already exists; use a fresh data root.")
    inventory = pd.read_parquet(inventory_path, columns=[
        "lemma", "n_events", "meets_min_events", "frm_mean_grammatical_tags",
    ])
    if inventory.empty or inventory["lemma"].isna().any() or inventory["lemma"].duplicated().any():
        raise ValueError("The s03 inventory needs one non-empty row per lemma.")
    inventory["lemma"] = inventory["lemma"].astype("string").str.casefold()
    if inventory["lemma"].duplicated().any():
        raise ValueError("Two inventory lemmas become identical after case folding.")
    inventory = inventory.sort_values("lemma").reset_index(drop=True)
    if args.sample:
        inventory = inventory.head(args.sample).copy()
    verify_sources()
    norms, norm_report = load_norms(inventory)
    audit, translation_report = translation_features(
        inventory["lemma"], args.model_snapshot, args.translation_batch,
    )
    semantic, semantic_report = semantic_features(
        inventory["lemma"], args.vocab_limit, args.semantic_batch,
    )
    features = norms.merge(
        semantic, on="lemma", how="left", validate="one_to_one",
    ).merge(
        audit[["lemma", "trn_norm_levenshtein_es"]],
        on="lemma", how="left", validate="one_to_one",
    )
    if len(features) != len(inventory) or set(features["lemma"]) != set(inventory["lemma"]):
        raise ValueError("The feature merge lost or duplicated inventory lemmas.")
    qualified = inventory["meets_min_events"].to_numpy(dtype=bool)
    report = {
        "inventory_lemmas": len(inventory),
        "qualified_lemmas": int(qualified.sum()),
        "features": {
            name: {
                "available_all": int(features[name].notna().sum()),
                "available_qualified": int(features.loc[qualified, name].notna().sum()),
            }
            for name in PREDICTORS
        },
        "complete_cases_all": int(features[list(PREDICTORS)].notna().all(axis=1).sum()),
        "complete_cases_qualified": int(features.loc[qualified, list(PREDICTORS)].notna().all(axis=1).sum()),
        "norm_sources": norm_report, "semantic": semantic_report,
        "translation": translation_report,
        "notes": [
            "AoA percent-known is the primary prevalence value; dedicated prevalence is a comparison column.",
            "Mean grammatical-tag count is an event-weighted proxy, not a literal inflection count.",
            "Larger semantic distance means a sparser neighbourhood.",
            "An identical translation can be a real cognate or a model echo.",
        ],
    }
    if args.sample:
        print(json.dumps(report, indent=2))
        print("Sample mode: no output files written.")
        return
    config.DATA_INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="s04_", dir=config.DATA_INTERMEDIATE) as temp:
        staging = Path(temp)
        features.to_parquet(staging / outputs[0].name, index=False)
        audit.to_parquet(staging / outputs[1].name, index=False)
        (staging / outputs[2].name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for output in outputs:
            (staging / output.name).replace(output)
    print(f"Saved features for {len(features):,} lemmas; "
          f"{report['complete_cases_qualified']:,} qualified lemmas have all 12 predictors.")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"Feature calculation failed: {exc}") from None
