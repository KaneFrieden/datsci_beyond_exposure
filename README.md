# Beyond Exposure: What Makes Words Stick?

This repository contains our analysis of Duolingo learning traces. We examine
whether properties of English words relate to recall and to how recall changes
with time since practice. We build the analysis in small, reviewable stages and
record our checks and intermediate results along the way.

## Current stage

`scripts/s00_verify_inputs_marie.py` checks the raw dataset and the output
locations. `scripts/s01_subset_kane.py` selects the language group, and
`scripts/s02_lexemes_marie.py` examines the word descriptions. The shared
`scripts/s03_inventory_zusammen.py` builds a lemma inventory and reports how
selection rules affect the data. We are now working on word features in s04.

## Data locations

Use a separate Python 3.12 environment for this repository. The local `.venv`
is ignored by Git and can be recreated without changing the project files.

Set `DATSCI_DATA_ROOT` to a new output directory containing a
`.datsci-data-root` marker. Set `DATSCI_RAW_ROOT` when reusing an existing
raw-data directory. Choose a separate output directory for this analysis. When
`DATSCI_RAW_ROOT` is set, s00 checks that the two locations do not overlap.
See `env.example.sh` for the
variables. A machine-specific `env.sh` is ignored by Git.

Run `.venv/bin/python scripts/s00_verify_inputs_marie.py` after setting the variables.
The raw directory must contain the dataset and its `dataset_manifest.json`.
The script verifies the raw file's size and MD5 against that local manifest
without writing to the raw-data directory.
Downloading the dataset is now handled by `scripts/download_dataset.py`, which also creates a local manifest.
