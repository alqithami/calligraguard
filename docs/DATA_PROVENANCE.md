# Source-font and data provenance

## V3 input manifest

[`experiments/v3/provenance/FONTS.csv`](../experiments/v3/provenance/FONTS.csv) contains 34 actually acquired source records, not placeholder rows. Each row records the source directory, embedded family/version metadata, grouping, split, relative filename, font-file SHA-256 and license-notice SHA-256. Font files themselves are not included.

All sources were acquired from the pinned Google Fonts tree:

`66928797f271fed5ee428544e7229685e2d90127`

The pinned source root is:

`https://raw.githubusercontent.com/google/fonts/66928797f271fed5ee428544e7229685e2d90127/ofl/`

For each CSV row, append the URL-encoded `filename` to this root for the font source. Append `<source_directory>/OFL.txt` for its license notice and `<source_directory>/METADATA.pb` for source metadata. Square brackets, commas and other filename characters must be URL-encoded. `acquire_sources.py` implements the acquisition and verifies the cache against the saved manifest. Metadata and license notices are fetched into the local acquisition cache.

| Source split | Acquired sources |
|---|---:|
| Training | 19 |
| Development | 3 |
| Normal calibration | 4 |
| Held-out Arabic | 3 |
| Held-out Latin | 5 |

Amiri and Ruwudu share the declared `amiri_lineage` group. Noto Arabic source variants share a conservative `noto_arabic` group. Known grouping does not establish fully adjudicated genealogical independence. Dataset glyph IDs, marks, render variants and interventions stay within the assigned base/source group; paired renderer/resolution conditions deliberately share held-out parents.

The immutable dataset manifest hash is `e9054589f930f566270cf69daa205e1740ffa81190ef25d42d1f5756f1669790`. The build contains 135,883 generated observations, 135,844 primary visible-defect/normal observations, and 39 separately recorded observation-only interventions. Final full metadata and artifact archives remain pending; the publication includes source and validation provenance, not the whole image corpus.

## License and distribution scope

The acquisition gate found the expected SIL Open Font License notice for each source and records its hash. This is provenance, not a claim that the root MIT license covers fonts or third-party models. Font binaries are excluded from this repository update and generated downloadable research packages. Third-party implementations are fetched at their recorded commits and retain their own licenses. Any later derived-data release needs its notices and distribution review.

## Historical sources

The original CFDefect corpus used system-font sources including proprietary fonts and fallback glyphs. The v3 CSV must not be used to retroactively claim provenance or distribution rights for that legacy corpus. The old `docs/FONTS_MANIFEST.csv`, where present, belongs to the legacy inventory and is not the new acquisition manifest.
