# CalligraGuard / CFDefect

Research code for glyph-level defect inspection: controlled outline interventions, paired raster evidence, defect localization, type labeling, and independently calibrated false-alarm evaluation.

## Current research version

The expanded **CFDefect-v3 computational benchmark is in progress**. Its frozen scientific source is now under [`experiments/v3/`](experiments/v3/README.md). This is a research snapshot, not a completed leaderboard or submission-ready release.

- [V3 instructions and scope](experiments/v3/README.md)
- [Dated run snapshot](experiments/v3/provenance/RUN_SNAPSHOT_20261014.json)
- [Current assessment and next decisions](docs/RESEARCH_STATUS.md)
- [Legacy audit and claim corrections](docs/LEGACY_AUDIT.md)
- [Completed development studies, kept separate from v3](docs/DEVELOPMENT_RESULTS.md)
- [Source-font provenance](docs/DATA_PROVENANCE.md)
- [Remaining release requirements](docs/UPLOAD_CHECKLIST.md)

**Historical warning:** `cfdefect/`, `DATASET_SPLITS/`, `runs/`, `runs_split/`, `runs_2m/`, and the original `tables/` belong to the legacy experiment. An audit found invalid or invisible generated positives, destructive small-component filtering, weak family separation, and manuscript/code discrepancies. Those files are preserved, not silently overwritten. Their presence is not validation of the original paper's superiority claims. Read the audit before reusing them.

## What v3 adds

The verified build contains **135,883 generated records**, of which **135,844** are in the primary observation-level population and **39** are separately retained as observation-only interventions. These are rendered records, not independent font families or independent physical defects.

| Partition | Primary records |
|---|---:|
| Training | 82,551 |
| Development | 10,543 |
| Normal calibration | 4,260 |
| New Arabic source-family evaluation | 11,930 |
| Renderer-shift evaluation | 11,912 |
| 128-pixel native rendering | 3,979 |
| 512-pixel native rendering | 3,985 |
| Latin transfer | 6,684 |

There are 34 pinned font sources: 19 training, 3 development, 4 calibration, 3 new Arabic evaluation, and 5 Latin evaluation sources. Known derivatives are grouped; complete genealogical independence is not asserted. Training records derive from 1,536 distinct base glyphs; calibration uses 284 base-glyph groups.

The five defined interventions are `spur`, `notch`, `local_deformation`, `missing_mark`, and `misplaced_mark`. A notch is not claimed to sever a whole stroke. Arabic cases use isolated base-plus-combining-mark shaping, not full text-line layout validation.

The configured comparisons are 16 pretrained supervised fits, 6 normal-only fits, and 12 classical configurations. Supervised models predict a detection score, a pixel mask, and five defect types. The normal-only methods use pinned author PatchCore and Dinomaly implementations with documented adaptations; they do not receive a query-specific clean reference at inference. Geometry conditioning is an ablation, not established methodological novelty.

## Reproduction

Use the Python 3.11/CUDA instructions in [`experiments/v3/README.md`](experiments/v3/README.md), not the legacy root requirements. All 14 frozen scientific/configuration files were checked against the active study's source hashes before publication. The new public setup helper is separate from those scientific files and does not alter the running study.

Do not copy provenance snapshots into live output locations as completion markers. `STUDY_COMPLETE.json`, produced by an actual run after fitting and held-out evaluation, is the computational completion indicator. It is not a guarantee of publication readiness.

## Evidence and limitations

V3 held-out results are not yet released. Completed v2b and repair studies are exploratory, use different populations, and must not be pooled with v3. No real-defect study, expert adjudication, full shaping validation, or active-rendering efficiency advantage is established. Calibration reports realized image- and parent-level false positives; it does not guarantee a target FPR under distribution shift.

Font binaries, server addresses, SSH details, private credentials, and unfinished checkpoints are not included in this update. No final arXiv identifier or publication claim is supplied. The root license applies to this repository's code; third-party implementations and font inputs retain their own licenses.

## Citation

Use [`CITATION.cff`](CITATION.cff) for the software and identify the exact repository commit and experiment version used. Do not cite historical manuscript numbers as verified v3 results.
