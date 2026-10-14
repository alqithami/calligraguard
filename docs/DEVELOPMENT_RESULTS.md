# Completed development studies (not v3 held-out results)

These summaries were read from verified saved predictions and workload records. All numbers are retained with their original population and evaluation status. They are not a single leaderboard and must not be pooled across stages.

## V2b feasibility study

18 fits, six configurations and three named seeds, completed 13 October 2026. Exploratory screen: 2,002 records (720 normal, 1,282 defective). Training: 5,347 records from 384 parent glyphs. Calibration: 144 normal parent groups. One of the 10,694 generated records was separately retained as unobservable at the declared high-resolution mask criterion; 10,693 were primary records.

**Seed qualification:** a software test reset the PyTorch seed before model construction. The three runs used different minibatch order but shared initial weights. They are real distinct fits, not three independent initializations. This was discovered later and corrected in the repair study and v3. Shared pretrained weights in v3 are intentional; its trainable heads and minibatch randomness are separately seeded.

Means across the three v2b fits:

| Configuration | AUROC | Recall at nominal 1% calibration target | Actual image FPR | Actual parent-any-view FPR |
|---|---:|---:|---:|---:|
| Raster 64 | 0.868607 | 0.611544 | 0.004630 | 0.018519 |
| Raster 256 | 0.982919 | 0.885335 | 0.003241 | 0.016204 |
| Raw-residual-selected contour crops | 0.930289 | 0.660426 | 0.002778 | 0.011574 |
| Random contour crops | 0.648848 | 0.230109 | 0.001389 | 0.006944 |
| All contour crops | 0.858144 | 0.637546 | 0.009722 | 0.048611 |
| Learned contour selection | 0.856069 | 0.606084 | 0.000000 | 0.000000 |

The untrained raw-maximum 256-pixel comparator had AUROC 0.970668, recall 0.742590 and image FPR 0.001389 on this screen. Different realized FPRs mean the recall differences are not exactly matched-FPR comparisons. In particular, a smaller image FPR does not imply the parent-level calibration target is achieved under source shift.

The full raster model was strongest; the learned selector did not establish a performance or rendering-efficiency advantage. The verified full review archive has SHA-256 `66b1b08358f1c67ae80e64c8d956b0cd7c52ac2d7d38da2d3f052a67a897dcf0`. Its full binary contents are not included in this source-only update.

## Read-only contour diagnosis

Nine saved checkpoints were diagnosed using development examples and normal calibration, without reopening the exploratory screen. Across 858 positive development examples, all 32 candidate windows reached at least some annotated defect evidence in 100% of cases and covered 99.7895% of mask pixels on average. Learned four-window selection reached evidence in 83.3333% of cases; raw-residual selection reached 85.5478%.

The discrete top-k path did not carry final image-loss gradients into selector parameters, although auxiliary mask-support supervision did train the selector. Some whole-image positive labels were assigned to selected patches without annotated defect evidence. Candidate coverage, crop sampling and loss consistency therefore needed separate controls before claiming a learned acquisition benefit.

## Controlled repair study

12 fits, four configurations, three genuinely distinct initialization/shuffle seeds, 12 epochs each. 24,192 total optimizer updates. Training population unchanged: 5,347 examples from 384 parents in eight source families. The reported population is **1,338 development examples from two families**, also used for best-epoch selection. These scores are not an independent final test and are not directly comparable to the v2b screen scores above.

| Configuration | Mean development AUROC | Mean recall at nominal 1% calibration target | Mean actual image FPR |
|---|---:|---:|---:|
| Raster 256 | 0.969797 | 0.868687 | 0.002083 |
| Four raw-selected crops, 24x24 sampling | 0.910369 | 0.653846 | 0.003472 |
| Four raw-selected crops, 40x40 sampling | 0.924730 | 0.729604 | 0.000694 |
| 40x40 crops, selected-evidence training targets | 0.920644 | 0.709402 | 0.000000 |

Sampling crops at higher resolution helped this control, but changing the image target to selected-evidence support did not further improve the aggregate results. The full-resolution raster baseline remained stronger. Dataset intervention labels were not altered by the selected-evidence training objective, and missed defects remained positive during evaluation.

The verified repair archive has SHA-256 `73bae732c0f190009a3fe0b7b6bd7e619e560b94adfce25409bc2d862200376f`. These completed development studies motivated the larger multi-task v3 benchmark; they do not establish a new state-of-the-art method, exact low-FPR control, real-defect validity, or a publishable active-rendering result.
