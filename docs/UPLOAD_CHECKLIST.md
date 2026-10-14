# Remaining public-release requirements

Status: v3 source and provenance snapshot published while the experiment is running. The legacy tree is preserved. Do not upload unfinished results as a final release or replace old paper numbers by guesses.

## After computational completion

- Verify all 22 configured fits, their saved checkpoints/memory banks, optimizer histories where applicable, independent seeds, and every per-run evaluation marker. Check the 12 classical configurations and all five held-out conditions before accepting `STUDY_COMPLETE.json`.
- Recompute metrics from complete, unique, finite per-example predictions. Verify raw score orientation, calibration/test group separation, actual parent and image FPR, mask conventions, per-type outcomes, and the observation-only denominator.
- Release final prediction/mask files, exact checkpoint or archive links, full generated-data metadata and rejected-attempt/observation manifests, environment lock and source commit. Include checksums and a durable artifact inventory. Font binaries remain excluded.
- Add figure-generation scripts plus source data and a one-to-one manuscript-table/figure mapping. Preserve development-vs-test distinctions and label differing input/supervision budgets.
- Independently validate checkpoint replay and representative full evaluation, including the effect of batching/numerical settings. Resume support is not proof that every possible interruption is recovered identically.

## Before calling the paper submission-ready

- Complete a manuscript revision grounded in verified results. Reconcile or remove the old architecture description, SVG-V claims, score-orientation inconsistencies and unresolved figure populations. Do not advertise a final manuscript until it exists and has been checked.
- Adjudicate source ancestry and realistic defect/benign-change labels, and collect independent real/toolchain cases. Synthetic Latin/renderer transfer is not a substitute for external production evidence.
- State the methodological contribution that actually survives the comparisons. A conventional geometry-channel ablation or many successful fits is not, by itself, a novel architecture.
- Review statistical uncertainty, calibration group count, per-family robustness, threshold-selection differences and remaining baseline limitations. Any post-test development needs a declared new evaluation plan.
- Create a release only after the included artifacts exist and match the documentation. Cite the exact software commit; do not invent an arXiv identifier or DOI.

## Legacy artifacts

Any original run still cited needs its own complete provenance and recovered files. Some legacy run directories remain partial; the new v3 source release does not fill those historical gaps. Legacy proprietary font inputs are not assumed redistributable. Old `paper/` assets are not a v3 final draft.
