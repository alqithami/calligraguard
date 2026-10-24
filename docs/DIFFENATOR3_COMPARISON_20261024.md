# Official Diffenator3 comparison — 24 October 2026

## Executed comparison

The unmodified PyPI package **diffenator3==1.1.4** ran in an isolated Python 3.11.17 environment on the existing research server. All **16/16 invocations exited successfully and produced parseable native JSON**, with no timeouts. Execution finished at **03:37:15 UTC on 24 October 2026**. Each native output and stderr file was checked against its per-case execution hash receipt; report counts were independently recounted from the JSON.

Command for each pair:

```bash
diffenator3 --json --pretty --max-changes 100000 --jobs 2 BEFORE_FONT AFTER_FONT
```

Two pair workers were used; each invocation had a 240-second limit. The comparison used the tool's default table/glyph/word/language checks, built-in word inventories and native reporting settings. The installed CLI exposed no common ppem/threshold override for matching the prior exhaustive FreeType study. Do not interpret this as an equal-renderer, equal-text, equal-threshold recall experiment.

## Input population

Seven historical-pair settings come from **two already-inspected repair events**: original complete Noto Nastaliq Urdu binaries and six reconstructed Noto Sans Malayalam masters. The Malayalam CFF reconstructions map the observed glyph to ASCII A; they are not complete released fonts, production Malayalam shaping, interpolation or hinting builds.

Two pairs are exact replay controls. Five are controlled representation/metadata modifications. Two are deliberately translated synthetic controls; intent is known by construction, not independently adjudicated natural typography.

## Native results

| Pair | Native changed table keys | Glyph-image records | Word-image records |
|---|---|---:|---:|
| Urdu historical repair | head | 0 | 1032 |
| Urdu exact replay | none | 0 | 0 |
| Malayalam master 1 | CFF, head | 0 | 0 |
| Malayalam master 2 | CFF, head | 0 | 0 |
| Malayalam master 3 | CFF, head | 0 | 0 |
| Malayalam master 4 | CFF, head | 0 | 0 |
| Malayalam master 5 | CFF, head | 0 | 0 |
| Malayalam master 6 | CFF, head | 0 | 0 |
| Malayalam exact replay | none | 0 | 0 |
| Malayalam smooth metadata | none | 0 | 0 |
| Malayalam cyclic start | none | 0 | 0 |
| Malayalam declared translation | CFF, head | 1 | 0 |
| Urdu metadata only | head, name | 0 | 0 |
| Urdu cyclic contour start | head | 0 | 0 |
| Urdu all contours reversed | head | 0 | 44 |
| Urdu declared translation | head | 1 | 0 |

`CFF ` is the actual four-character table tag; trailing space is omitted in the table for readability. Counts refer to native report records, not unique faults, independent examples, or expert-confirmed defects.

## Interpretation

The official production tool already exposes the Urdu repair through word-rendering differences. CalligraGuard cannot claim unique detection of that event. The six Malayalam pairs have structural reports despite no default glyph-image entry; absence of that entry does not mean their pixels are identical at every size.

The whole-outline reversal control produces 44 native word records. Earlier CalligraGuard controls inspected a particular native glyph under a different renderer/context/tolerance. This discrepancy is retained; it is not counted as an official-tool false positive or a CalligraGuard superiority result without matched rendering and expert adjudication.

Neither tool infers whether a visible revision is wanted from difference magnitude alone. No independently labeled defect/legitimate-revision classification accuracy is established by these two events.

## Reproduction and records

Executed preparation/runner source and hash-verified native count records are in `reproducibility/20261024/official_comparison_source.tar.xz`. That preparation uses the unchanged `common.py` and `run_study.py` from the source-QA bundle. Original private font files are not redistributed. The pinned source hashes are retained in the acquisition records and source-QA code. The full native JSON reports, stderr, exact protocol, input-pair hashes and environment freeze are preserved in the closeout backup.

The environment used attrs 26.1.0, diffenator3 1.1.4, fonttools 4.67.0, freetype-py 2.5.1, glyphsLib 6.16.2, numpy 2.4.6, openstep_plist 0.5.2, pillow 13.0.0, ufoLib2 0.19.1, uharfbuzz 0.57.0 and unicodedata2 18.0.0. These describe the executed environment, not a claim about current latest versions.

## Primary tool references

- Official source and CLI semantics: https://github.com/googlefonts/diffenator3
- Executed package version: https://pypi.org/project/diffenator3/1.1.4/

This is the final bounded external computational comparison before manuscript consolidation. It does not replace the unperformed independent expert revision-review study.
