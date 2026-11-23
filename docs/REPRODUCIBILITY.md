# Reproducing CETR-Net FIVES V2 Evidence

This is an as-executed research archive, **not a single-click cloud installation**. Several scripts retain machine-specific paths and Kaggle input mounts; adapt local paths to your authorized installation without modifying frozen benchmark logic.

1. Obtain the licensed FIVES release from its official data distribution. Do not copy images or masks from private Kaggle inputs into this public repository.
2. Rebuild the content-screened V2 split in \`protocol/v2_preprocessing.json\`; verify 508 training and 90 validation records per seed and save the SHA-256 audit.
3. Independently train backbone seeds 42, 123, 2025 (45 epochs in executed runs). Match the original checkpoint manifest and processing functions.
4. With each frozen seed-matched backbone, train 4 CDES plus 3 routing ablations for 8 epochs each. Select the best checkpoint by **validation only**.
5. Freeze the model/checkpoint hashes and threshold (0.5) in \`protocol/frozen_fives_v2_test_plan.json\` **before** official test access.
6. Run held-out evaluation once on the complete 200 annotation records, with grouping over 197 visually screened groups and compare to the frozen independent baseline.
7. To audit rather than rerun, load \`results/official_fives_v2_final_report.json\` and \`results/official_fives_v2_cross_seed_table.csv\`; verify per-seed SHA-256 manifests.
8. Do **not** choose a different proposed variant or recalibrate thresholds because its official-test performance appears more favorable.

### Readout and audit scope

- Test Dice is a grouped macro-score, not an annotation-level micro-average or clinical patient-level statistic.
- The paired cluster-bootstrap intervals are *within-seed, within-test-set* uncertainty, not proof of across-population significance.
- Original pilot leakage was identified and disqualified; only inference that does **not** use ground truth to form test/validation proposals is acceptable.
- Some source training scripts refer to laptop paths, but executed official test was computed on Kaggle GPU. **Do not execute GPU training on the local laptop** under the original operational restriction.
- Test input images, masks, patient metadata and weights are not provided; full reproduction requires their legal acquisition and corresponding archived checksum.

Remaining experiments for a stronger journal claim: independent capacity-matched baselines, geometry-and-clDice-specific controlled tests, paired topology false-link/recovery metrics, cross-dataset transfer and computational cost.
