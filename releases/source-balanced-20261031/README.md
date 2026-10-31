# Source-balanced manuscript: versioned release preparation

**Scientific version: 31 October 2026. Preparation: 1 November 2026.**

The approved package and PDFs have passed local integrity checks. **Binary upload is pending; no GitHub Release or journal submission is represented as completed.** The connected repository-writing tool can commit these records, but has no release-asset upload action. The checked local GitHub CLI was not authenticated. Do not request or paste a token into chat; sign in directly with GitHub CLI.

## Preparation files

- `ASSET_MANIFEST.json`: exact expected file names, sizes, SHA-256 identities and PDF page counts.
- `LOCAL_PREFLIGHT.json`: actual local archive/PDF check, with limits.
- `CalligraGuard_Completed_20261031_Fresh_Verified.json`: unchanged prior scientific assembly/build/replay receipt.
- `publish_release.py`: local check by default; explicit `--publish` is required for external publication.
- `test_release.py` and `PUBLISHER_TESTED.json`: eight local safety/lifecycle tests. Remote upload is mocked in those tests.
- `RELEASE_NOTES.md`: pre-submission research-release notes, not a claim of acceptance.

## Use

Download the approved `CalligraGuard_Complete_Package_20261031.zip` to Downloads. Use a checkout of this repository and Python 3.9+:

```bash
python3 releases/source-balanced-20261031/publish_release.py
```

The default performs only local checks. It verifies every manifest-listed member and extracts only the three approved PDFs. It does not modify the original ZIP, run scientific code, or upload anything.

To publish after direct GitHub sign-in:

```bash
gh auth login --hostname github.com
python3 releases/source-balanced-20261031/publish_release.py \
  --archive "$HOME/Downloads/CalligraGuard_Complete_Package_20261031.zip" \
  --target-commit "$(git rev-parse HEAD)" --publish
```

The helper creates or resumes the exact-tag draft, refuses conflicting targets/assets, uploads without clobbering, downloads and hashes every asset, then publishes a **pre-release**. It additionally checks unauthenticated public downloads and writes a local `PUBLICATION_VERIFIED.json` only after success. It never deletes an asset or changes the scientific package. A failed upload remains a draft unless the failure occurs during post-publication access verification; inspect the actual release state rather than assuming success.

## Scope not to overstate

The 31 MB ZIP includes the complete manuscript, panel-qualification records and final historical records. It includes only compact transfer records. The separate 542 MB raw transfer archive and large original v3 assets are not uploaded by this helper. Publishing this release does not complete every data-access requirement or authorize a journal submission.

The current approved PDF still needs final affiliation/declarations and a factual updated release link before journal upload. Keep the original scientific version immutable; make an explicitly versioned administrative revision after author declarations and public access are confirmed.
