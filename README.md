# CAPSTONE-CLIP

Canonical research workflows and the latest completed reproducibility artifacts for the CAPSTONE-CLIP paper. This release replaces the previous collection of notebook versions with descriptive Python entry points and explicit artifact folders.

## Commands

Use Python 3.12+ and a suitable CUDA/Colab environment. Install dependencies with `python -m pip install -r requirements.txt`. The training workflows retain the original Colab setup, model-download and Drive-mount behavior. They need the benchmark datasets and feature caches from the original experiment environment. `--root` changes the original Drive project path; it does not download datasets.

```bash
python code/train_benchmark.py --root /content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing
python code/train_protocol_baselines.py --root /content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing
python code/run_robustness_checks.py --root /content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing
python code/run_overlap_decontamination.py --root /content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing
python code/postprocess_results.py --root /content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing
python code/analyze_statistics.py
python code/validate_release.py
```

These are independent workflows, not mandatory sequential steps. `train_benchmark.py` is the last archived full benchmark source associated with the primary run tables; `train_protocol_baselines.py` is the completed protocol-matched comparator source. `run_robustness_checks.py` is the final clean revision suite containing the five-width sweep, all-six controls, second-caption-corpus check and RSICD–AID sensitivity. `run_overlap_decontamination.py` is the latest completed 29 October paired RSITMD–UCMerced rerun. `postprocess_results.py` produces historical descriptive tables and figures. The canonical final inferential analysis is `analyze_statistics.py`; it reads the archived inputs and writes regenerated outputs to `generated/statistical_analysis`.

All Python code has zero comment tokens and no docstrings. File names and command-line help provide the entry-point descriptions. Training was not repeated when preparing this release.

## Paper artifact map

| Material | Folder |
|---|---|
| Locked fold and support manifests | [manifests](manifests) |
| Fixed configurations | [configs](configs) |
| Canonical training and postprocessing code | [code](code) |
| Archived run tables | [results/archived_runs](results/archived_runs) |
| Run-level architecture ablations | [results/architecture_ablations](results/architecture_ablations) |
| All-six exposure extension | [results/all_six_exposure](results/all_six_exposure) |
| Five-width sweep, r = 32, 64, 128, 256, 512 | [results/five_width_sweep](results/five_width_sweep) |
| Second-caption-corpus robustness | [results/second_caption_corpus](results/second_caption_corpus) |
| Cross-corpus overlap audit | [results/cross_corpus_overlap](results/cross_corpus_overlap) |
| RSICD–AID decontamination sensitivity | [results/aid_decontamination](results/aid_decontamination) |

Additional folders contain the latest RSITMD–UCMerced rerun, protocol-matched baseline tables, descriptive postprocessing and the final dataset-level statistical revision.

## Manifests and evidence

Each of the six dataset folders contains `image_index.csv`, `fold_assignments.csv` and `support_assignments.csv`. Image paths are relative to the dataset root. `test_fold` assigns each record to its one outer test fold; the training pool for a fold is its complement. Support membership is explicit for every fold, shot count and support seed, in the original sampling order. The ordered path digests, fold hashes and support hashes were verified against the archived locked manifests.

The original benchmark configuration is an archived JSON. Other configuration files are extracted source defaults and explicit revision schedules, labelled as such. File provenance and hashes are recorded under `provenance`.

**Read [data integrity notes](docs/data_integrity.md) before interpreting the AID results.** The archived AID index includes a verified duplicate-content pair in different folds. Original results and assignments are preserved and the limitation is disclosed. Human review of the latest overlap candidates is incomplete. This package supplies reproducibility evidence; it does not certify that historical datasets were free of leakage.

The repository contains metadata, source code and run tables, not dataset imagery or multi-gigabyte feature tensors. The existing GPL-3.0 license is retained.
