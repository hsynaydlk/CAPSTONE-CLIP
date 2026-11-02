# CAPSTONE reviewer-revision experiment report

- Wall-clock used: 3.84 h
- Hard budget: 4.90 h
- Backbone: openclip_b32
- Target validation: not used

## R1-#4 — Cross-corpus overlap audit
- Completed caption→target combinations: 12 / 12
- Maximum observed frozen-feature cosine similarity: 0.995741

## R1-#7 — Adapter-width sweep
- Widths represented: [32, 64, 128, 256, 512]
- Best macro accuracy in completed sweep: width=512, 87.957%
- Interpret only over the tested width interval; do not claim capacity is universally irrelevant.

## R1-#5/#6 — Exposure controls
- Datasets represented: 6 / 6
- Controls represented: ['image_only', 'joint_full', 'no_caption', 'shuffled_caption']
- The shuffled-caption control is objective-matched and isolates correspondence more cleanly than the image-only control.

## R1-#9 — Empirical variability scale
- Median support-seed SD: 0.842 pp
- 95th percentile support-seed SD: 3.696 pp
- Use this only as descriptive empirical context; it is not a formally calibrated equivalence margin.

## R1-#12 — Caption-corpus robustness
- Datasets represented: 6
- Corpora represented: ['RSICD', 'RSITMD']

## Files
- Tables: `/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing/reviewer_revision_final_clean_20261006/tables`
- Figures: `/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing/reviewer_revision_final_clean_20261006/figures`

The response letter should quote the actual completed rows and should not claim completion for stages stopped by the wall-clock guard.