# Data integrity and scope

The released index lists reconstruct the archived protocol; they do not represent new experiment runs. All 60 train/test-index hashes, 750 support-index hashes and six ordered path digests match the original archives.

AID contains two byte-identical source images with the same Drive title `port_200.jpg`. The archived ordered path digest confirms their mounted names were `AID/Port/port_200 (1).jpg` and `AID/Port/port_200.jpg`. They occur at indices 6772 and 6773, in test folds 3 and 0. Both original source files have SHA-256 `46eeaabfaf66e91f3337a99b6a5b7a6bcb126b590587b2d9fe338cc32984e450` (see validation JSON for authoritative hash). The historical split is disjoint by record index, but these two records are not distinct image content. This is a known duplicate-content leakage risk; this release does not silently modify historical assignments or scores. Correcting it requires a separately versioned split and new runs.

The source AID class counts are Industrial=389 and Port=381. The release preserves these recorded counts rather than substituting nominal dataset counts.

The RSICD–AID decontamination sensitivity removes the two cross-corpus candidates recorded in the archived run table. This is different from the within-AID duplicate above and from the latest RSITMD–UCMerced conservative exclusion rerun.

The latest UCMerced source status explicitly records human_review_completed=false. Candidate similarities are screening evidence, not human-confirmed duplicates.

All archived metric tables are original source bytes. Accuracy, Macro-F1 and ECE in raw tables use fractions; NLL is unscaled. Summary columns ending `_pp` use percentage points. Existing tables contain heterogeneous coverage and reused primary runs; the release validation reports their observed grids. The architecture ablation table includes Accuracy, MacroF1, ECE and NLL at run level.

Historical paper postprocessing code is retained for descriptive tables and figures. Its older inferential outputs are not canonical. Use `analyze_statistics.py` and `results/statistical_analysis` for the final 29 October dataset-level statistical revision.
