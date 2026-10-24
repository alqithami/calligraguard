# CFDefect-v3: completed computational benchmark

**Completed 17 October 2026 at 18:25:42 UTC; results published in the 24 October closeout.** This directory retains the original 14 frozen scientific/configuration files. No scientific source was changed to publish results or shut down compute.

The exact earlier README, with detailed construction, training, evaluation and setup instructions, is preserved at [provenance/README_20261014.md](provenance/README_20261014.md). References there to an ongoing run describe the historical 14 October snapshot, not current status.

## Results

The completed audit records **22/22 fits**, **22/22 model evaluations across five conditions**, and **12/12 classical configurations across five conditions**. The result bundle contains `RESULTS_SUMMARY.json`, `COMPLETION_VERIFIED.json`, `VERIFIED_FITS.json` and `PREDICTION_INVENTORY.json`:

- [Completed results bundle](../../reproducibility/20261024/v3_completed_results.tar.xz)
- [Bundle checksums and scope](../../reproducibility/20261024/MANIFEST.json)
- [Current research status](../../docs/RESEARCH_STATUS.md)

The observed campaign's dataset metadata SHA-256 is `e9054589f930f566270cf69daa205e1740ffa81190ef25d42d1f5756f1669790`.

The aggregate learned-results SHA-256 is `df3a9605ff10dffc4090e09e223861965662eb6a5bec690ee678e6380a5ce9a1`.

The classical-results SHA-256 is `ecbfb4e2fc53ff5a8a48a6c732d3a03bb8f3c01a4c3c9e3c7c58773a4537999c`.

## Scope

Referenced U-Net/ResNet-34 (5 seeds), DeepLabV3+/ResNet-50 (3), SegFormer-B2 (3), reference-geometry U-Net (5), author-implementation PatchCore (3) and Dinomaly (3) constitute the 22 fits. The normal-only and referenced supervised tracks have different information/supervision. The geometry branch adds reference distance/tangent/boundary fields; it is not validated as a new source-attribution architecture.

There are 135,844 primary examples and 39 retained observation-only interventions. Renderer/resolution tests share held-out parent glyphs, and Latin examples use the same geometric intervention machinery. Actual image FPR and parent-any-view FPR must accompany nominal calibration targets. Calibration and within-family bootstrap assumptions do not imply a production or arbitrary-shift guarantee.

## Reproduction

Use a new Linux/CUDA workspace and the instructions in the preserved README. Begin with `sha256sum -c SOURCE_SHA256SUMS`. Do not invoke the coordinator over an archived completed directory merely to recreate a status marker. Existing trained models and predictions are preserved in private archival backups; they are not bundled in Git.

`setup_public.sh` is a separately packaged helper, not one of the frozen scientific files; its prior syntax checks are not a new clean-machine reproduction claim. `STUDY_COMPLETE.json` indicates configured computational completion, not final-paper readiness.

The audit reads compressed predictions, recomputes scalar detection/threshold metrics and checks mask structure. It does not independently re-render every mask intersection or retrain models. Retain these limitations with any numerical table.
