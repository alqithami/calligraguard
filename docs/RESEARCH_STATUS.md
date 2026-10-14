# Research assessment and next decisions

Assessment date: 14 October 2026. Basis: the recovered legacy audit, completed v2b/repair studies, and the frozen source and operational status of the ongoing v3 experiment. This document does not claim new held-out performance results or provide a completed manuscript revision.

## Executive assessment

The project is substantially stronger as a benchmark and experimental program than the original manuscript, but methodological novelty and production validity are still unresolved. V3 addresses the weak-comparison objection through larger pretrained multi-task baselines, pinned author anomaly-detection implementations, stronger classical comparisons, and controlled source/rendering/script evaluations. More computation does not by itself establish a new method.

The current defensible research question is: **How reliably can glyph-level outline defects be detected, localized, and typed when harmless rendering differences, observation limits, and source-family shift are explicitly represented?** This is more specific than the original claim of a generally raster-vector-grounded multimodal inspector.

At the dated [snapshot](../experiments/v3/provenance/RUN_SNAPSHOT_20261014.json), 10 of 22 configured fits had completed. The third SegFormer fit was active; no v3 held-out evaluation was complete. The snapshot is historical, not a live status endpoint. Training checkpoints are not final results.

## What has improved

**Task validity.** The corrected generator makes finite-area protrusions and localized notches, deforms outlines locally, and removes/displaces identified combining marks without intentionally deleting the base glyph. It retains attempted/rejected interventions and separates intervention labels from observation-level visibility. In v3, 39 generated observations are outside the primary visible-defect population rather than silently relabeled as clean.

**Nontrivial normal controls.** Benign subpixel shift, supersampling, gamma changes, and combinations prevent every legitimate query from being bit-identical to its reference. The legacy exact-reference case was too favorable to unfiltered differencing to support the claimed learning advantage. The new controls are procedural rendering changes, not expert approval of arbitrary design revisions.

**Experimental design.** V3 uses 34 pinned sources, explicit derivative grouping, 82,551 training records from 1,536 base glyphs, separate development and normal-calibration populations, and held-out source/rendering/resolution/script conditions. The 135,844 primary records are not 135,844 independent glyphs or source families. Three Arabic held-out sources remain a limited family population.

**Prediction tasks.** The supervised models now train and report an image score, a dense mask, and a five-type classifier. The geometry ablation has a tested gradient path into its added input channels. Training uses separately seeded heads/minibatch order; shared pretrained encoder weights are intentional.

**Evidence discipline.** Sources and data are frozen before fitting; the coordinator waits for all configured fits before freezing a checkpoint registry and scoring held-out data. Calibration uses only normal examples. The code reports both image and parent-any-view false-positive rates, and no test-based score inversion is performed.

## What remains weak or unsupported

### Methodological novelty

U-Net, DeepLabV3+, SegFormer, multi-task heads, and added outline-distance/tangent fields are conventional components. The geometry variant adds structured reference features to the input; it is not an explicit candidate/reference contour matcher, learned vector-token cross-attention system, active renderer, or validated editable-segment attribution method.

The earlier learned crop selector did not beat the full-resolution raster baseline. The subsequent repair ablations improved some crop scores but did not reverse that conclusion. These findings are preserved in [DEVELOPMENT_RESULTS.md](DEVELOPMENT_RESULTS.md), including the initial seed-handling caveat. They must not be described as positive evidence for an acquisition advantage.

Even a successful v3 geometry ablation would first establish a useful inductive bias. It would not automatically establish a major architecture contribution. A stronger method claim requires a distinct capability with matched-information and matched-cost controls and an explanation of when/why that capability matters.

### Realism and external validity

V3 remains synthetic, with the same intervention mechanisms across training and the current held-out domains. Latin transfer is not evidence on real font-production failures. The study has no unseen operator-mechanism test, independently adjudicated production-defect collection, foundry workflow evaluation, or repair-time study. It does not cover full text shaping, ligatures in context, general GPOS/GSUB correctness, or hinting in production engines.

Known source lineages are grouped conservatively, but complete genealogical independence has not been independently adjudicated. Renderer/resolution domains share held-out parent glyphs, so their observations are correlated and should be analyzed as paired domain shifts, not independent replications.

### Statistical and baseline scope

The calibration population has 284 distinct base-glyph groups. The nominal 1% operating point therefore depends on very few upper-tail group scores. The current within-family group bootstrap conditions on the threshold and observed source families; it does not capture calibration-threshold uncertainty or generalization to an unseen family population. Realized parent FPR is essential and cannot be replaced by a smaller image FPR.

The supervised checkpoint criterion combines AUROC, Dice and type macro-F1, not the final low-FPR endpoint alone. This choice is frozen and must be disclosed. Supervised and normal-only methods also differ in information, supervision, pretraining, resolution, and segmentation-threshold calibration. PatchCore's 1,024-normal-image/1%-coreset budget is explicit, not evidence that full-budget PatchCore has been exhausted. EfficientAD and several other relevant controls are not included in this campaign.

### Reproducibility and deployment

Publication of source/configuration/provenance is not release of complete results. V3 prediction dumps, masks, final checkpoints, plots and the paper-to-artifact registry remain pending. Resume and checkpoint logic exist, but arbitrary crash-point and cross-hardware replay have not been exhaustively tested. The current timing records include particular loading/preprocessing paths and do not establish an end-to-end active-rendering efficiency advantage.

## What to expect next

1. Finish the fixed training queue without changing locked sources or selecting models from test results. Publish no winning v3 method until all relevant runs and evaluations are verified.
2. Freeze the actual saved checkpoint registry, then run the five held-out conditions and the 12 classical configurations. Separate referred/supervised and normal-only comparisons, and retain all seeds and failures.
3. Independently validate IDs, masks, checkpoint hashes, recorded optimization work, thresholds, actual FPR, and metric recomputation. Report paired, per-family and per-defect differences, with the right uncertainty and observation denominators. Regenerate every manuscript table and figure from an explicit run/population mapping.
4. Make the contribution decision from the evidence. Strong raster models plus a negligible geometry benefit favor a benchmark/inspection paper. A consistent geometry benefit supports a narrower representation result, not an automatic breakthrough. Severe renderer/family failures should be reported as the problem exposed by the benchmark, not concealed with post-test retuning.
5. Add real/toolchain cases and qualified independent assessment before claiming production relevance. A new methodological or task revision after seeing these tests requires a new declared evaluation plan rather than treating already inspected results as untouched confirmation.

## Publication recommendation

Do not resubmit the existing manuscript unchanged. Its title, architecture description, numerical narrative and old claims need revision against the completed audited evidence. Retain the glyph-level problem and its three tasks; remove unsupported dense-SVG, broad universal-inspection, or superiority language. A benchmark-first paper is the most defensible current direction. A top-tier method-paper claim is not yet established, and acceptance cannot be inferred from run duration, model size, or the number of comparisons.

The ongoing experiment should finish before another small architecture search is initiated. The next major deliverable is a verified, interpretable results matrix and a manuscript whose claims are traceable to it—not another untested title or a premature final draft.
