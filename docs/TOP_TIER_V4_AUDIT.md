# Penakluk Top Tier v4: CETR-Net IEEE Access audit

## Observed outcome
The locked FIVES V2 official held-out test contains 200 annotated records and 197 visually screened image groups. Three independent 45-epoch backbones (42/123/2025) and 21 eight-epoch refiner runs were evaluated. The predeclared dual_full_asymmetric proposed model gives mean Dice 0.849389 versus its same-seed frozen backbone 0.851540 (difference -0.002152). Mean IoU: 0.745639 vs 0.749034. Mean clDice: 0.870490 vs 0.871657. All seed-level primary differences were negative and no refiner exceeded the baseline mean Dice.

## Literature roles
MAIN METHOD PREDECESSOR: MaskVSC by Zhou et al. (2025), IEEE Transactions on Medical Imaging, DOI 10.1109/TMI.2025.3538336. Not trained as a comparator.
MAIN DATA REFERENCE: FIVES, Jin et al. (2022), Scientific Data, DOI 10.1038/s41597-022-01564-3.
SUPPORTING references: U-Net (Ronneberger 2015), Attention U-Net (Schlemper 2019), GKE-TUNet (Qiu 2024), STR-Net (Wang 2026). These separate methods were not executed in our official V2 study.
TOPOLOGY METRIC: clDice (Shit et al. 2021), actually calculated on official test.

## Main contribution -> research question -> actual comparator -> evidence
C1/RQ1: Paired CDES restore/reject supervision versus unperturbed natural, break-only, false-only and dual symmetric controls, plus frozen same-seed backbone. None beat baseline; dual does not dominate single-error on Dice/IoU/clDice.
C2/RQ2: Candidate image-evidence routing versus geometry-only and confidence-only heads and symmetric-weight full heads, using frozen seed-matched backbones. Proposed mean Dice 0.849389 > geometry-only 0.846104 and confidence-only 0.848910, but < backbone 0.851540; symmetric dual 0.849384 nearly identical.
C3/RQ3: Three seeds, one-time 200-record test, 197 macro-groups, validation-only checkpoint and threshold selection, paired cluster bootstrap, source hashes, fixed-ID actual qualitative images 001/080/160. Mean proposed delta -0.002152. Limited to within-FIVES reproducibility, NOT external-domain claims.

## Executed research framework
Each independently trained backbone is FROZEN for refiner fitting. Branch A: backbone P0, threshold 0.5, no correction. Branch B: prediction-only P0 candidate graph, RGB/probability/geometry descriptors, two action heads and spatial correction, trained using training-only synthetic CDES masks. Seven refiner variants on 3 seeds. Figure 2 revised accordingly. No joint end-to-end backbone training.

## Qualitative evidence
Three fixed-ID original Kaggle official held-out samples, seed 42, are inserted as Figures 8-10: RGB fundus / reference vessel mask / unchanged frozen backbone / preregistered primary CETR-Net. No synthetic redraw or cherry-picking based on sample scores. Source Jin et al. FIVES (2022), CC BY 4.0. Numeric summary covers all 197 groups; qualitative figures are illustrative only.

## Template audit
Original IEEE Access double-column Word document retained, including equations, title, authors, header, and author biographies. Body Paper Paragraph: Times New Roman 10 pt; Caption: 8 pt; Reference paragraphs: 8 pt; Title Helvetica 22 pt. Microsoft Word PDF export gives 14 pages, contact-sheet visual review completed. Equations remain embedded OMML; 20 inline graphics.

## Mendeley integration status: NOT LIVE
The DOCX contains ZERO active ADDIN CSL_CITATION / Mendeley fields. Its [n] reference markers and References section are STATIC text. The 14-entry BibTeX is supplied for Mendeley Reference Manager import. The corresponding author must sign in to Mendeley Cite within Microsoft Word, replace each static marker using Insert Citation, and regenerate References via Mendeley. Never claim that providing BibTeX equals live Word citation linkage.

## Submission blockers
- No independent capacity-matched MaskVSC, U-Net or Attention U-Net head-to-head runs on this same protocol; no SOTA claim.
- No complete candidate-level missed-vessel/false-link error benchmark or FLOPs/latency comparisons.
- No independent DRIVE cross-dataset transfer or patient-disjoint clinical cohort.
- Live Mendeley Word fields and editorial DOI/metadata review pending.
- Do not tune any future variant against already observed official FIVES V2 results.

Decision: Evidence-backed revision delivered, but NOT ready for an IEEE Access superiority claim or unconditional submission.
