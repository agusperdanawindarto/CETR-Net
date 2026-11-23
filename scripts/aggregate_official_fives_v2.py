"""Aggregation of 3 completed, frozen official FIVES V2 test seeds.
Run ONLY after all 3 Kaggle seed outputs have passed manifest and checksum checks.
CPU-only; never perform training, threshold fitting, or choose winner based on test.
"""
import json,hashlib,csv,datetime,statistics
from pathlib import Path
import numpy as np
B=Path(r"D:\RISETKOLABORASI\CETR-Net\EXPERIMEN")
O=B/"RESULTS_KAGGLE"/"OFFICIAL_TEST_V2"
plan=json.loads((O/"FROZEN_FIVES_V2_TEST_PLAN.json").read_text(encoding="utf8"))
SEEDS=(42,123,2025)
names=plan["secondary_variants"]
assert len(names)==7
data={}
for s in SEEDS:
    folder=O/f"seed_{s}"/"CETR_OFFICIAL_TEST_V2"
    f=folder/"OFFICIAL_FIVES_V2_SEED_SUMMARY.json"
    assert f.is_file(),("Unfinished official FIVES evaluation",s,f)
    j=json.loads(f.read_text(encoding="utf8"))
    assert j["status"]=="COMPLETE" and j["seed"]==s
    assert j["n_records"]==200 and j["n_unique_visual_groups"]==197
    assert j["test_archive_sha256"]==plan["test_archive_sha256"]
    assert j["backbone_sha256"]==plan["backbones_sha256"][str(s)]
    assert j["validation_only_model_selection"] and not j["test_time_ground_truth_proposals"]
    assert j["threshold"]==.5 and j["primary_variant"]==plan["primary_predeclared"]
    assert set(j["refiners"])==set(names)
    group=json.loads((folder/"groupwise_197.json").read_text(encoding="utf8"))
    record=json.loads((folder/"per_record_200.json").read_text(encoding="utf8"))
    assert len(group)==197 and len(record)==200
    assert len({x["record_key"] for x in record})==200
    assert len({x["near_group"] for x in record})==197
    assert all(len(x["variants"])==7 for x in record)
    data[s]=j
summary={
 "status":"ALL_THREE_OFFICIAL_FIVES_V2_SEEDS_VERIFIED",
 "test_type":"one-time locked FIVES V2 official held-out test",
 "completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "seeds":[42,123,2025],"test_records":200,"near_duplicate_visual_groups":197,
 "groupwise_metric":"Dice, IoU and clDice across 197 screened visual groups",
 "frozen_validation_selection":True,
 "predeclared_primary":"dual_full_asymmetric vs independent_backbone",
 "per_seed":{str(s):{"backbone_dice":data[s]["backbone"]["dice"],
   "primary_dice":data[s]["refiners"]["dual_full_asymmetric"]["dice"],
   "primary_delta_dice":data[s]["refiners"]["dual_full_asymmetric"]["vs_backbone"]["paired_delta_mean"],
   "primary_delta_ci_95":data[s]["refiners"]["dual_full_asymmetric"]["vs_backbone"]["paired_delta_95pct_cluster_bootstrap_ci"],
   "validation_selected_variant":data[s]["prior_validation_best_variant"]}
   for s in SEEDS},
 "cross_seed":{},
 "interpretation_rules":[
  "No test-based variant/threshold selection: all seven variants were frozen before test access",
  "Official FIVES V2 test alone does not establish cross-dataset generalization",
  "No SOTA superiority claim unless independent baselines evaluated identically",
  "N=3 seeds; descriptive means and SD only, avoid unsupported significance claims"]
}
rows=[]
methods=["independent_backbone",*names]
for method in methods:
    metrics={}
    for metric in ("dice","iou","cldice"):
        vals=[]
        for s in SEEDS:
            z=data[s]["backbone"] if method=="independent_backbone" else data[s]["refiners"][method]
            vals.append(z[metric])
        metrics[metric]={"by_seed":{str(s):vals[i] for i,s in enumerate(SEEDS)},
                          "mean":float(np.mean(vals)),"sample_sd":float(np.std(vals,ddof=1)),
                          "n_seeds":3}
    summary["cross_seed"][method]=metrics
    rows.append({"method":method,"seed42_dice":metrics["dice"]["by_seed"]["42"],
      "seed123_dice":metrics["dice"]["by_seed"]["123"],
      "seed2025_dice":metrics["dice"]["by_seed"]["2025"],
      "mean_dice":metrics["dice"]["mean"],"sd_dice":metrics["dice"]["sample_sd"],
      "mean_iou":metrics["iou"]["mean"],"mean_cldice":metrics["cldice"]["mean"]})
with (O/"OFFICIAL_FIVES_V2_CROSS_SEED_TABLE.csv").open("w",newline="",encoding="utf8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
tmp=O/"OFFICIAL_FIVES_V2_FINAL_REPORT.json.tmp"
tmp.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
tmp.replace(O/"OFFICIAL_FIVES_V2_FINAL_REPORT.json")
print("ALL_OFFICIAL_FIVES_V2_EVALUATIONS_COMPLETE_AND_VERIFIED",json.dumps(summary["per_seed"]),flush=True)
print("TABLE",O/"OFFICIAL_FIVES_V2_CROSS_SEED_TABLE.csv",flush=True)
