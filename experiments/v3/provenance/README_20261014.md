# CFDefect-v3: expanded computational benchmark

**Ongoing study, not final results.** This directory publishes the 14 scientific/configuration files frozen for the active benchmark. Their bytes match the original study's source hashes. The immutable source/data lock is in `provenance/EXPERIMENT_LOCK.json`; the dated status snapshot is not a live monitor.

## Scope and tracks

| Track | Methods | Information and supervision |
|---|---|---|
| Supervised referenced | U-Net/ResNet-34 (5 seeds), DeepLabV3+/ResNet-50 (3), SegFormer-B2 (3) | Candidate and trusted reference rasters, explicit differences; image/mask/type training labels |
| Reference geometry ablation | U-Net/ResNet-34 plus reference outline fields (5 seeds) | Same raster evidence plus distance/tangent/boundary fields from the trusted outline |
| Normal-only | Author PatchCore (3), author Dinomaly (3) | Normal training images; candidate grayscale expanded to RGB at inference, no paired clean reference |
| Classical referenced | 12 score configurations | Candidate/reference images, estimated alignment/gamma and structural/boundary residuals |

The normal-only tracks are not equal-information comparisons with referenced supervised models. PatchCore uses a recorded source-stratified budget of at most 1,024 normal images and a 1% feature coreset; Dinomaly uses the normal training pool and 10,000 iterations. Full-frame resize, score aggregation and segmentation-threshold adaptations are recorded in `normal_ad.py`. Do not describe these as an untouched reproduction of every original author's experimental setting.

The geometry branch is an inductive-bias ablation, not a novel contour-attention, active acquisition, or editable-segment-attribution method. It adds four dense trusted-reference fields to the U-Net input, with added first-layer weights initially zero. It does not receive injected defect type, severity, modified-point indices, or the candidate vector source as inference features.

## Verified build

135,883 generated records; 135,844 primary records; 39 retained observation-only interventions. Training: 82,551 records from 1,536 base glyphs. Normal calibration: 4,260 views from 284 base-glyph groups. Full counts and source partitioning are in the root README and provenance validation report.

Arabic examples are isolated base-plus-combining-mark shapes. The three marks are U+064E, U+064F and U+0650. The defined operators are `spur`, `notch`, `local_deformation`, `missing_mark`, and `misplaced_mark`; the last two are not used for Latin. Normal controls include identity, subpixel translation, supersampling, gamma and combined changes.

The renderer/resolution tests share held-out parents with `test_family`. They are controlled shifts, not additional independent families. Latin uses the same geometric generator and is synthetic transfer, not real-defect validation. Primary metrics are conditional on the declared observation being visible; retain `all_records.jsonl`, `observability_only.jsonl` and rejected-attempt accounting for broader analysis.

## Reproduce in a new Linux/CUDA workspace

Do not run these commands over an active experiment directory. An existing maintainer-hosted run does not need to pull this repository update.

From the repository root:

```bash
cd experiments/v3
sha256sum -c SOURCE_SHA256SUMS
```

Use Python 3.11 and a CUDA-compatible NVIDIA driver. On an Ubuntu/Debian host, install the runtime libraries before setup:

```bash
sudo apt-get install -y --no-install-recommends libcairo2 libegl1 libgl1 git
PYTHON_BIN=python3.11 bash setup_public.sh
.venv/bin/python -m unittest -v validate_and_test.EvaluatorTests
```

`setup_public.sh` is a new portable packaging helper, not part of the 14-file scientific source lock. It creates an isolated environment and checks imports; it was syntax-checked for publication but not used to rebuild the already-running environment. The frozen `requirements.txt` preserves an obsolete comment about dependency inheritance; the public helper does not inherit another research environment. It installs CUDA PyTorch explicitly and includes the repaired `tabulate`/`colorama` requirements.

A complete run can be started in a detached session:

```bash
tmux new-session -d -s cfdefect-v3 'exec .venv/bin/python -u run_benchmark.py > RUN.log 2>&1'
cat STATUS.json
tail -n 30 -F RUN.log
```

The coordinator acquires pinned sources, builds and validates data, tests all model APIs, freezes the study, fits all 22 configurations, freezes their checkpoint registry, evaluates the held-out partitions, and runs the classical comparisons. Downloads include pretrained weights and source fonts in local acquisition caches; they are not bundled with the repository.

To resume an interrupted run, first confirm that no coordinator or child training process is active; retain the unchanged sources, data, environment and saved checkpoints, then invoke the same coordinator. Do not edit source files after `EXPERIMENT_LOCK.json` is created. Resume is implemented, but arbitrary crash-point recovery has not been exhaustively fault-injection tested.

`geometry_helpers.py` retains its original v2b header and standalone entry point so its hash remains identical; v3 imports its geometry primitives and runs `build_benchmark.py`, not that older standalone build.

## Evaluation rules

Supervised fits use 20-40 epochs with a fixed patience rule, BF16, 16-example microbatches and two-step accumulation. The final incomplete accumulation group is normalized separately. A shuffled incomplete final training microbatch is omitted to avoid singleton BatchNorm batches; all development/evaluation records remain.

Checkpoint selection is fixed at 0.5 development AUROC + 0.3 positive Dice + 0.2 type macro-F1. This is not low-FPR checkpoint optimization. Normal-only methods use their recorded fixed fit budgets without defective labels.

Detection thresholds use strictly greater-than comparisons against order statistics of maximum normal score per base glyph. Report both image FPR and parent-any-view FPR. Group bootstrap intervals resample base glyphs within the observed source families and condition on the chosen threshold; they do not include calibration-threshold uncertainty or uncertainty about entirely new families.

Supervised mask threshold is 0.5. Normal-only and classical mask thresholds come from normal calibration map pixels, with their definitions recorded in the outputs. These are different localization protocols and must be stated in tables. Pixel AUROC from supervised outputs is a 4,096-bin histogram approximation. No real-defect realism, formal shift guarantee, runtime advantage, or method superiority is implied by passing the software tests.

## Completion and release

Each `runs/<method>/seed_<seed>/COMPLETE.json` certifies only that fit. `MODEL_REGISTRY_LOCK.json` precedes held-out scoring. `EVALUATION_COMPLETE.json` records per-run evaluation. `STUDY_COMPLETE.json` is written only after the configured fits, evaluations and classical results finish. It explicitly does not certify final-paper readiness.

Provenance files here are historical read-only snapshots, not substitute success markers. The final predictions, masks, checkpoints, curves and dataset manifests are not published in this ongoing snapshot. See `../../docs/UPLOAD_CHECKLIST.md` for the post-run release audit.
