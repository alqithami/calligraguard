# Non-file actions still required

These steps cannot be completed just by uploading files.

1. In GitHub repository settings, change the **About** description so it matches the current README.
   Recommended text:
   `Code and released artifacts for Arabic glyph defect inspection and the CFDefect benchmark.`

2. Create a GitHub **Release** and attach any large checkpoints / prediction dumps that you do not want to keep in git.

3. Replace the placeholder `docs/FONTS_MANIFEST.csv` with the real font manifest before calling the repository complete.

4. If you want the repository itself to store the manuscript snapshot, upload `paper/manuscript.pdf` and `paper/source.zip`, or attach them to a Release and update `paper/README.md` accordingly.
