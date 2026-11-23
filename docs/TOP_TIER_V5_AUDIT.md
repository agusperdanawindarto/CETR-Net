# CETR-Net IEEE Access — Penakluk Top Tier V5, 23 Nov 2026

## What was audited
The latest V5 is a line-by-line consistency revision of the source-preserving V4 Word manuscript, not a new experimental run. It retains the official FIVES V2 200-record/197-visual-group locked test, three independent backbones (45 epochs each), and seven head variants (eight epochs per seed; 21 runs).

## Primary evidence (no positive superiority claim)
Mean Dice: frozen backbone 0.851540 vs. predeclared dual_full_asymmetric CETR-Net 0.849389, paired difference -0.002152.
Mean IoU: 0.749034 vs. 0.745639.
Mean clDice: 0.871657 vs. 0.870490.
All seed differences negative; no refinement variant exceeds the independent backbone in mean official-test Dice.

## RQ/contribution/comparator alignment
C1/RQ1: paired missing- and false-link supervision tested against natural, break-only, false-only, symmetric dual-error controls and no-refiner backbone. The hypothesis is NOT supported in held-out overlap/topology metrics.
C2/RQ2: image-conditioned routing tested against geometry-only, confidence-only and symmetric-weight controls using the same frozen per-seed backbone and head family. It improves some weaker refinement variants but does NOT exceed the backbone; this is NOT a MaskVSC matched comparison.
C3/RQ3: three training seeds, one-time official FIVES V2 test, 197 screened visual groups, paired group bootstrap and fixed case-ID genuine prediction panels. Mean primary effect negative. Transfer, efficiency and unmatched SOTA do not belong to an answered RQ.

## Literature hierarchy
Main method predecessor: Zhou et al. 2025, MaskVSC, IEEE TMI, DOI 10.1109/TMI.2025.3538336 (NOT independently trained in this work).
Main data source: Jin et al. 2022, FIVES, Scientific Data, DOI 10.1038/s41597-022-01564-3.
Metric reference: Shit et al. 2021, clDice (measured).
Supporting literature: U-Net, Attention U-Net, GKE-TUNet, STR-Net and related topology work (NOT trained head-to-head here).

## V5 repairs relative to V4
1. Related Work paragraph still incorrectly required external transfer and efficiency for completed RQ3; rewritten to restrict evidence to FIVES V2.
2. Methods paragraph inaccurately called executed CETR-Net learning components 'unexecuted'; rewritten to reflect frozen-backbone refiner fitting.
3. Framework-to-RQ paragraph still claimed cross-dataset generalization and compute cost for RQ3; corrected.
4. Equation (11) and its explanatory paragraph had counted unique byte hashes (199), contradicting the 197 visual-group estimator used in results; the native OMML equation now uses G and R_g for the 197 groups/annotation records.
5. Qualitative subsection corrected to 'Figures 8-10'; case IDs 001/080/160 are authentic archived seed-42 outputs, shown as original RGB, annotation, unchanged backbone, proposed prediction.
6. Unchanged quantitative results and IEEE Access original two-column layout/author bios/OMML equations. Exported with Microsoft Word: 14 PDF pages; no empty page or beyond-page text detected.

## Formatting
Uses the retained IEEE Access template, body 10-point Times New Roman, captions/references 8-point, template heading/title fonts, double column, 20 inline graphics, and native editable equation objects.

## Mendeley: not linked yet
The DOCX retains STATIC [n] citations and references, NOT Mendeley Cite fields. The included BibTeX is import-ready; activating live Mendeley Cite requires Word and the author's signed-in Mendeley account. Do not describe this as already connected.

## Submission blockers
Actual capacity-matched independent MaskVSC/Attention U-Net/U-Net comparisons, candidate-level action-error verification, external-validation and cost profiling have not been executed. An IEEE Access superiority claim would not be justified. FIVES has already been exposed at official test; avoid post-test parameter selection. More experiments need a fresh held-out design. The manuscript is evidence-aligned but NOT unconditionally ready for acceptance/submission, particularly without live Mendeley and bibliography audit.

[executed on device: DESKTOP-FVE5NQR (981dd20a-9648-473e-8e4d-2ac3d3a4fb5a)]
