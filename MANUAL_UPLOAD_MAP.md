# Manual upload map

Upload these files to the repository paths below.

| Bundle file | Upload path in repo | Action |
|---|---|---|
| `README.md` | `/README.md` | replace |
| `CITATION.cff` | `/CITATION.cff` | new |
| `.gitignore` | `/.gitignore` | new |
| `requirements.txt` | `/requirements.txt` | replace |
| `environment.yml` | `/environment.yml` | new |
| `RELEASE_CHECKLIST.md` | `/RELEASE_CHECKLIST.md` | new |
| `LICENSE-MIT.txt` | `/LICENSE` | rename **only if** you want MIT |
| `LICENSE-NOASSERTION.md` | `/LICENSE-NOASSERTION.md` | optional reference |
| `DATASET_SPLITS/README.md` | `/DATASET_SPLITS/README.md` | new |
| `docs/REPRODUCIBILITY.md` | `/docs/REPRODUCIBILITY.md` | new |
| `docs/RESULTS_AND_ARTIFACTS.md` | `/docs/RESULTS_AND_ARTIFACTS.md` | new |
| `docs/RUNS_MANIFEST.csv` | `/docs/RUNS_MANIFEST.csv` | new |
| `docs/FONTS_MANIFEST_TEMPLATE.csv` | `/docs/FONTS_MANIFEST_TEMPLATE.csv` | new |
| `docs/PAPER_RELEASE_NOTES.md` | `/docs/PAPER_RELEASE_NOTES.md` | new |
| `paper/README.md` | `/paper/README.md` | new |

## Recommended order

1. Replace the root `README.md`.
2. Add `CITATION.cff`, `.gitignore`, `environment.yml`, and the new docs files.
3. Rename `LICENSE-MIT.txt` to `LICENSE` **only if** you want MIT for the code.
4. Fill `docs/FONTS_MANIFEST_TEMPLATE.csv` and update `docs/RUNS_MANIFEST.csv` with any missing artifacts.
5. Add the actual manuscript files under `paper/`.
