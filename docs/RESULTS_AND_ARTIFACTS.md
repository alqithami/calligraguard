# Results and artifact status

This file supersedes the older assumption that a public metrics JSON or generated table is automatically the scientific source of truth. Measurements must be connected to the exact data population, inference inputs, score orientation, checkpoint and validity audit.

| Stage | Code/data status | Result status | Permitted interpretation |
|---|---|---|---|
| Legacy CFDefect | Historical `cfdefect/`, original split/run/table directories preserved | Existing numbers subject to [legacy audit](LEGACY_AUDIT.md) | Historical measurements, not a validated original superiority claim |
| V2b feasibility | Completed on a new nuisance-varied development corpus | Summary in [DEVELOPMENT_RESULTS.md](DEVELOPMENT_RESULTS.md) | Exploratory screen; shared initial weights across named seeds acknowledged |
| Contour diagnosis and repair | Completed diagnostic/ablation work, original experiments unchanged | Separate development results with independent initialization seeds | Diagnosis, not novel-method confirmation or independent test evidence |
| Expanded v3 | Frozen source under `experiments/v3/`; verified build and running fits | Held-out result release pending | Benchmark implementation and dated progress, not a finished leaderboard |

V3 publication contains exact source/configuration hashes, a real 34-source acquisition manifest, a dataset validation summary, and a dated operational snapshot. It does not include incomplete model predictions or mark partial runs complete. The v3 final checkpoint/prediction/data/figure archive is not yet released.

Completion has three different meanings: `COMPLETE.json` for an individual fit; `EVALUATION_COMPLETE.json` for that fit's held-out scoring; `STUDY_COMPLETE.json` for the configured computational study. None alone establishes real-world validity or methodological novelty. Provenance copies under `experiments/v3/provenance/` are not executable-run success markers.

A release-grade row needs the dataset manifest hash, source lock, checkpoint or memory-bank digest, exact input/supervision budget, complete per-example predictions, threshold/calibration population, and a recomputation script. A manuscript figure also needs its population and source-data mapping. These will be assessed after the active run; no figure or manuscript has been falsely marked synchronized by this update.

The old `docs/RUNS_MANIFEST.csv` and `docs/FONTS_MANIFEST.csv`, where present, are legacy inventory documents, not the current v3 source manifest. Use [DATA_PROVENANCE.md](DATA_PROVENANCE.md) and the new versioned provenance files. Outstanding work is consolidated in [UPLOAD_CHECKLIST.md](UPLOAD_CHECKLIST.md).
