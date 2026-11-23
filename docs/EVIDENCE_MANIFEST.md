# Evidence and integrity index

- Aggregated primary JSON: \`results/official_fives_v2_final_report.json\`
- Frozen plan: \`protocol/frozen_fives_v2_test_plan.json\`
- Cross-seed metrics: \`results/official_fives_v2_cross_seed_table.csv\`
- Per-seed held-out checksums: \`results/per_seed/seed_{42,123,2025}/VERIFIED_OFFICIAL_FIVES_TEST.json\`
- Per-group metrics: \`results/per_seed/seed_*/groupwise_197.json\`
- Per-record metrics: \`results/per_seed/seed_*/per_record_200.json\`
- Six frozen refiner group manifests: \`results/refinement/seed_*_{cdes,routing}.json\`
- Source backbone integrity: \`results/backbones/\`

Only hashes, code, and numeric outcomes are published. A checksum merely establishes consistency with the source archival state; it does not prove absence of train/test conceptual leakage by itself. Frozen protocol and test-time source inspection are part of the methodological evidence.

The manuscript draft previously described a historical 510/90 development partition; the executed audit after filtering used 508/90. Historical 199 unique test byte hashes and 197 visually screened groups are different counting definitions and must not be equated.
