# Legacy audit and claim corrections

Audit scope: all 9,860 pilot and all 247,996 large-build raster pairs/masks in the recovered original experiment; no image-read errors. Saved predictions from 16 run directories were also checked. This is a summary of the completed audit, not a claim that the old data are scientifically valid. The original files and the old [quickstart](LEGACY_QUICKSTART.md) remain available for historical inspection. The quickstart is not the current setup guide and contains superseded badges/instructions.

## Baseline comparison changes the conclusion

| Original population | Filtered TemplateDiff AUROC | Unfiltered maximum/mean difference AUROC | Unfiltered nonzero-difference recall | Observed clean FPR |
|---|---:|---:|---:|---:|
| Pilot, 9,860 | 0.7067697769 | 0.8274594320 | 0.6549188641 | 0 |
| Large, 247,996 | 0.7149994355 | 0.8583135750 | 0.7166271499 | 0 |

The original TemplateDiff score thresholds and morphologically filters the difference mask, removing many small changes before pooling. Removing that filtering reverses the broad claimed advantage over direct differencing: the recovered fine-tuned model's 0.8030558789 AUROC is below the large-build raw-difference AUROC of 0.8583135750.

These audit operating points are descriptive on the existing population; they are not independently calibrated deployment evaluations. Every normal query equals its own reference, so nonzero raw differences separate raster-visible positives from normal pairs. Excluding invisible positives alone would make that exact-reference task trivial, not a stronger learned-detection benchmark.

## Generated labels and visible outcomes

| Large-build label | Audited outcome |
|---|---|
| `spur`, 42,548 records | 38,195 query/reference pairs are pixel-identical; all 42,548 stored masks are empty |
| `missing_diacritic`, 42,520 records | All 42,520 query images are blank; 3,408 references were already blank |
| `jitter`, 42,872 records | 42,693 stored masks are empty |

The spur operator departs from and retraces the same segment, making a zero-area excursion rather than a controlled finite-width protrusion. The missing-mark fallback can remove a whole path that contains the entire glyph. The old small-component filtering can erase micro-defect supervision. Consequently, the old per-type findings cannot simply be interpreted as successful semantic mark inspection and failure on real tiny spurs.

## Implementation and reporting inconsistencies

The recovered legacy model is a custom U-Net, not the EfficientNet-B0-backed network drawn in the manuscript. SVG-V conditions features after the mask is computed, so changing SVG-V does not directly change that fixed model's mask-derived detection score. Shared training gradients from the auxiliary heads are a separate effect and must not be described as direct mask conditioning.

The conflicting U-Net AUROCs near 0.435 and 0.565 correspond to raw versus reversed score orientation. They must not be silently reconciled by choosing the larger number. The approximately 0.795 and 0.803 large-model values are from different unfine-tuned/fine-tuned runs, not automatically an arithmetic error. The figure's 225,060 population count is not yet reconciled with the 247,996-record evaluation; no verified extra partition should be invented.

The old split has `Arial` in training and `Arial Bold` in test, and `Courier New Bold` in training versus `Courier New` in validation. Sample-ID disjointness does not establish the claimed font-family disjointness. The v3 source plan replaces this with explicit source/known-lineage grouping.

## Consequence for reuse

Historical tables are records of an older pipeline, not validated evidence of universal superiority, production FPR control, semantic vector grounding, or real-defect performance. They are not v3 results. Manuscript revisions must distinguish verified measurements, invalid task constructions, unresolved reporting discrepancies, and later independent experiments.

V2b introduced new finite-area/mark-specific operators and explicit observation partitions. V3 adds nuisance-varied normal controls, stronger baselines and source-defined domain shifts. Those corrections do not retroactively validate old numbers or themselves prove novel methodology.
