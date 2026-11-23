# IV. EXPERIMENTAL RESULTS AND DISCUSSION (IEEE Access section draft)

## A. Experimental design and integrity

The executed FIVES V2 evaluation uses 508 training, 90 validation and 200 official held-out annotation records, grouped into 197 visually screened image-content groups for the macro-score. Independent backbone instances were fitted for seeds 42, 123 and 2025, with 45 backbone epochs per seed; 7 topology variants were then fitted against each frozen seed-matched backbone for 8 epochs, producing 21 refiner runs. Candidate proposals at inference depend solely on predictions, not held-out annotations. Model selection and the threshold (0.5) were frozen using validation before the held-out experiment. The predeclared primary comparison is dual_full_asymmetric versus the frozen backbone, with 2,000 paired group-bootstrap resamples for per-seed Dice uncertainty. These are controlled within-FIVES observations rather than independent clinical cases or external-domain evidence.

## B. Primary held-out performance

Across three seeds, the independent backbone achieved mean grouped Dice 0.851540, IoU 0.749034, and clDice 0.871657. Proposed dual_full_asymmetric achieved Dice 0.849389, IoU 0.745639, and clDice 0.870490. Thus, the preregistered primary outcome was worse by 0.002152 Dice points (approximately −0.253% relative), with decreases on every seed: −0.002205, −0.002199, and −0.002052 for seeds 42, 123, and 2025, respectively. Per-seed paired 95% cluster-bootstrap intervals exclude zero on the negative side, but no population-wide or three-seed significance claim follows from this observation alone. The proposal cannot be described as providing accuracy, overlap, or clDice improvements.

## C. Ablation analyses and research questions

Across the seven refinement variants, the best *mean* held-out Dice was obtained by false_only (0.850137), still lower than the independent backbone (0.851540). The unperturbed/natural control gave 0.849690, break_only 0.849537, dual_full_symmetric 0.849384, dual_geometry 0.846104, dual_confidence 0.848910, and dual_full_asymmetric 0.849389. This pattern indicates that the tested dual-error setup did not jointly surpass its constituent single-action controls and fails to establish a positive answer to RQ1. A distinct benefit from image-evidence-conditioned routing over simpler alternatives (RQ2) is also unsupported under these tested choices: the geometry-only condition is notably poorer, but adding more features is not sufficient to overcome the backbone benchmark.

For RQ3, the consistent direction of the three-seed primary difference and corresponding declines in IoU/clDice strengthen the conclusion that the implemented refinement did not improve within-FIVES overlap/topology metrics. Nevertheless, external DRIVE transfer, complete matched-capacity SOTA comparators, per-link missed/spurious correction scores, and formal runtime/FLOPs measurements were not completed. Those unmeasured objectives must not be treated as demonstrated, nor should this result be generalized beyond the 2-D FIVES masks.

## D. Interpretation, limitations and next experiments

Unlike region-level gradient optimization, discrete candidate graph construction and action-conditioned spatial modifications may introduce systematic false edits near thin or ambiguous vessels. The observed uniform Dice decrement is consistent with, but does not prove, such an error mechanism. Isolating causal factors requires saved candidate-level edit traces, centerline restoration/break metrics, controlled synthetic error challenges, and matched-parameter controls. The frozen backbone is stronger than all tested refinement variants, so further method development requires a *new*, clearly separated study; the official test should not be used iteratively to retune the current reported experiment. The legitimate contribution of the present work is the reproducible, negative evaluation of an explicit two-action connectivity-correction hypothesis.

## E. Reproducibility and research integrity

Experiments were executed on Kaggle using two authenticated accounts; source code, frozen checkpoint hashes, evaluation protocol, per-record and groupwise metrics, and aggregated tables are archived. Original fundus images and ground-truth masks are not redistributed. The final manuscript should be updated with a verified IEEE-numbered reference list, independent baseline training, and appropriately licensed qualitative figure panels if available.
