# CalligraGuard

**Computational closeout — 24 October 2026.** The expanded v3 benchmark is complete: **22 fits, 22 five-condition evaluations, and 12 classical configurations**. The source-available historical inspection studies and the final official **Diffenator3 1.1.4** comparison have also completed. This repository no longer describes v3 as a running study.

**These results do not establish a superior new inspection architecture, a uniquely correct inverse source attribution method, or production false-positive guarantees.** Computational completion is not submission readiness. No independently adjudicated 20-event revision benchmark or timed designer study has been performed.

## Start here

| Resource | Contents |
|---|---|
| [Current research status](docs/RESEARCH_STATUS.md) | Completed studies, contribution boundaries and unperformed validation |
| [Computational closeout](docs/COMPUTATIONAL_CLOSEOUT_20261024.md) | Execution, preservation and infrastructure status |
| [Official-tool comparison](docs/DIFFENATOR3_COMPARISON_20261024.md) | All 16 configured comparisons, native report counts and limitations |
| [Frozen v3 source](experiments/v3/) | Original scientific code and source/data locks; no retraining during closeout |
| [Verified results and source bundles](reproducibility/20261024/) | Completed numerical summaries, checksums, historical QA code and official-tool runner |
| [Legacy audit](docs/LEGACY_AUDIT.md) | Why original manuscript-era artifacts are not validated current results |

## Two separate information contracts

**Source-available font-build QA:** both source/binary revisions, acquisition settings and rendered images are available. Direct structural, visual and dependency-aware comparison are mandatory baselines. Detecting a difference does not establish that a change is unwanted.

**Raster-only source explanation:** a trusted reference source and candidate images are available, but the original candidate source and historical patch are evaluator-only. An approximate rendering explanation is not necessarily the true editing history. Failure to find a counterexample establishes neither necessity nor infeasibility.

These tracks must not be merged into an information-unmatched leaderboard.

## Completed v3 benchmark

The campaign finished **17 October 2026 at 18:25:42 UTC**. Its recorded audit verified 22 fits, 110 learned model-condition evaluations, 60 classical configuration-condition evaluations, 940,500 learned per-run prediction records, 14 frozen scientific source hashes and 25 model-artifact hashes. It structurally checked 846,780 packed masks. Audit limitations are retained: it did not repeat all model inference or independently recompute every mask/ground-truth intersection.

The corpus contains **135,844 primary records** plus **39 separately retained observation-only interventions**. Training has 82,551 records from 1,536 base glyphs; normal calibration has 4,260 views from 284 base-glyph groups. Renderer and resolution conditions reuse held-out parents; Latin transfer uses the same synthetic geometric mechanisms. Counts of renderings are not counts of independent real defects.

The geometry variant is a controlled reference-field ablation, not an established new architecture. Results and actual false-positive rates under shift must be reported rather than selecting a favorable in-domain AUROC. Normal-only PatchCore/Dinomaly methods receive different information and supervision from referenced supervised models.

## Historical and production-tool findings

Later studies examined authentic source histories, original Urdu binaries, reconstructed Malayalam masters, numerical source fitting and explicit ambiguity. Ordinary alignment, sparse fitting and source-dependency tracing remained strong controls. Four changed Urdu glyph records affect 34 rendered glyphs; dependency tracing agrees with all 27 supported source-swap text configurations. The source-available comparison is not a learned defect classifier or a designer-time study.

The final official Diffenator3 comparison completed **16/16 invocations**. Its native output reports the Urdu historical repair through 1,032 word-image records, and all six Malayalam reconstructions through structural differences. The tool's sampling/renderer and native thresholds differ from the earlier exhaustive FreeType comparison; these counts are not defect counts or directly matched recall. Full details and unfavorable control outcomes are retained in the comparison note.

## Reproducibility and preservation

The original v3 scientific files remain byte-for-byte frozen. Public bundles add results and executed source without fonts, credentials or cloud host details. Some historical scripts retain their original workspace layout; they are provenance snapshots, not all portable entry points. Reacquire licensed inputs according to the pinned source manifests. The large model/data/prediction archives and complete native reports are privately preserved rather than committed as Git objects.

Read [the bundle manifest](reproducibility/20261024/MANIFEST.json) and verify SHA-256 before extracting into a new directory. The 14 October README is preserved [here](docs/README_20261014.md); its running-status statements are historical.

## Original manuscript-era material

`cfdefect/`, older `runs/`, tables and the original dataset/split artifacts remain for traceability. They are **historical and subject to the legacy audit**, not a replacement for the corrected v3 evidence. Do not reuse old superiority claims or treat a file's presence as proof of a validated result.

## Publication state

The computational phase is closed for manuscript consolidation. Expert intent labels, a blinded independent revision-event evaluation, and reviewer-time benefits are **not established**. No manuscript was rewritten as part of this closeout. The repository documents both positive and negative results without claiming that further GPU training is required to finish the completed campaigns.
