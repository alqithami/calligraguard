# Public-release check and remaining data-access boundary

## Completed on 25 October 2026

`verify_public_release.py` downloads the five closeout bundles and fourteen frozen v3 scientific files directly from the pinned public revision `c2aa1af2b7f816b710d96d84f3637755006d8da2`. It checks SHA-256, Git blob identities, archive members, Python syntax, the full 30-row learned and 60-row classical result grids, 340 seed-mean/standard-deviation calculations, and the 16-row official-tool summary.

The executed fresh-public-download check passed at 12:13:24 UTC. Its receipt is `PUBLIC_RELEASE_VERIFIED.json`.

```bash
python3 reproducibility/20261025/verify_public_release.py --output public_release_check.json
```

The optional output must be new. The script uses only the Python standard library, keeps public downloads in memory, never extracts archive paths, and requires no GPU or retired-server access. `--checkout PATH` reads an existing checkout instead of downloading pinned public bytes.

## What this check does not establish

It is a release-integrity and reporting-arithmetic check, **not** raw-score AUROC/FPR recalculation, pixel-mask intersection validation, a new inference/training run, or complete access to every experimental artifact. The official-tool check reads the released count summary, not the full native reports.

| Supporting material | Current public state | Required for stronger reproduction claim |
|---|---|---|
| Frozen v3 scientific code and study plans | Published and hash-checked | Pinned environment and input acquisition must remain explicit |
| Completed v3 seed-level metric summaries | Published and arithmetic-checked | Score/label/group records plus evaluator needed to recount raw-score metrics |
| Checkpoints, image corpus and packed mask streams | Preserved privately, not included in these bundles | Accessible licensed inputs/outputs or an explicit complete reproduction route |
| Official-tool runner and 16-row summary | Published | Full native report records and recount command remain to be released |
| Historical QA and constrained-sparse source | Published selected source snapshots | Remaining case traces and portability/recount checks must be linked per table |
| Expert intent labels and reviewer-time data | Not collected | Not claimed as evidence |

The underlying research is preserved in verified private archives. Public packaging of additional records is incomplete. These gaps are not repaired by declaring the computational campaign complete, and this check must not be cited as full experimental replication.

No scientific model, result, threshold or dataset was changed. No new experiment or server was started. A raw-support export was stopped by a local free-space preflight before archive extraction or data writes. Existing backups were left unchanged. New storage/workspace is a packaging requirement, not a reason to repeat the completed GPU campaign.
