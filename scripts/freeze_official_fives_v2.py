"""One-time preregistered official FIVES V2 evaluation plan, BEFORE running labels.
Stages 21 previously trained heads and immutable source hashes as a PRIVATE Kaggle dataset.
No test-set inference, metric computation, threshold fitting, or GPU use.
"""
from pathlib import Path
import json,os,shutil,hashlib,datetime
B=Path(r"D:\RISETKOLABORASI\CETR-Net\EXPERIMEN")
assert (B/"LAPTOP_GPU_DISABLED.flag").is_file()
OUT=B/"RESULTS_KAGGLE"/"OFFICIAL_TEST_V2"
OUT.mkdir(parents=True,exist_ok=True)
DST=B/"kaggle_transfer"/"MIGRATE_AGUSPERDANA"/"fives_official_heads"
DST.mkdir(parents=True,exist_ok=True)
HEAD_DST=DST/"heads"
HEAD_DST.mkdir(exist_ok=True)
VARIANTS=("natural_full_symmetric","break_only","false_only","dual_full_symmetric","dual_geometry","dual_confidence","dual_full_asymmetric")
GROUP={"natural_full_symmetric":"cdes","break_only":"cdes","false_only":"cdes",
       "dual_full_symmetric":"cdes","dual_geometry":"routing","dual_confidence":"routing","dual_full_asymmetric":"routing"}
SEEDS=(42,123,2025)
def sha(path):
    with path.open("rb") as fp:return hashlib.file_digest(fp,"sha256").hexdigest()
def save(path,data):
    content=json.dumps(data,indent=2,ensure_ascii=False)+"\n"
    if path.is_file():
        if path.read_text(encoding="utf8")!=content:raise RuntimeError("Frozen preregistration already exists with different contents: "+str(path))
    else:path.write_text(content,encoding="utf8")
base_manifest=json.loads((B/"kaggle_transfer"/"MIGRATE_AGUSPERDANA"/"verified_backbones"/"SOURCE_MANIFEST.json").read_text(encoding="utf8"))
heldout=json.loads((B/"kaggle_transfer"/"heldout_test_dataset"/"HELDOUT_AUDIT.json").read_text(encoding="utf8"))
assert heldout["records"]==200 and heldout["visual_groups"]==197
test_zip=B/"kaggle_transfer"/"MIGRATE_AGUSPERDANA"/"fives_official_test"/"fives_v2_official_heldout_512.zip"
assert sha(test_zip)==heldout["archive_sha256"]
variants_data={}
best={}
for seed in SEEDS:
    per_seed={}
    for variant in VARIANTS:
        group=GROUP[variant]
        name=f"cetr-v2-full-refiner-seed{seed}-{group}"
        if (seed,group) in ((123,"cdes"),(123,"routing"),(2025,"cdes")):
            root=B/"RESULTS_KAGGLE"/"FULL_STAGE2"/name
            report=json.loads((root/"VERIFIED_FULL_STAGE2_MANIFEST.json").read_text(encoding="utf8"))
            report_item=report["variants"][variant]
            expected_hash=report_item["best_head_sha256"]
        else:
            root=B/"RESULTS_KAGGLE"/"MIGRATION_AGUSPERDANA"/name
            report=json.loads((root/"VERIFIED_MIGRATED_STAGE2.json").read_text(encoding="utf8"))
            report_item=report["variants"][variant]
            expected_hash=report_item["head_sha256"]
        assert report["seed"]==seed and report["group"]==group
        assert report["official_test_scored"] is False
        assert report_item["epochs"]==8
        infile=root/"cetr_v2"/"CETR_V2_STAGE2_EXPERIMENTS"/f"seed_{seed}"/variant/"best_head_weights.npz"
        assert infile.is_file(),infile
        current_hash=sha(infile)
        assert expected_hash==current_hash, ("Corrupted refiner best checkpoint",seed,variant)
        dest=HEAD_DST/f"seed{seed}_{variant}.npz"
        if dest.exists():assert sha(dest)==current_hash
        else:shutil.copy2(infile,dest)
        per_seed[variant]={
          "checkpoint_file":f"heads/{dest.name}",
          "checkpoint_sha256":current_hash,
          "best_validation_dice":report_item["best_validation_dice"],
          "best_validation_epoch":report_item["best_epoch"],
          "checkpoint_source":report["source_kernel"],
          "epochs_completed":8}
    best_variant=max(VARIANTS,key=lambda v:per_seed[v]["best_validation_dice"])
    variants_data[str(seed)]=per_seed
    best[str(seed)]={"variant":best_variant,"validation_dice":per_seed[best_variant]["best_validation_dice"]}
for seed in SEEDS:
    src=B/"kaggle_transfer"/"MIGRATE_AGUSPERDANA"/"verified_backbones"/f"backbone_seed{seed}.weights.h5"
    assert sha(src)==base_manifest["seeds"][str(seed)]["sha256"],("Backbone mismatch",seed)
source_code={}
for name in ("cetr_core_ablation.py","cetr_v2_suite.py"):
    p=B/name
    source_code[name]=sha(p)
import numpy as np
for filename in HEAD_DST.glob("*.npz"):
    with np.load(filename) as a:
        for family in ("shared","restore","reject"):
            assert family+"_0" in a.files,(filename,family)
        assert all(np.all(np.isfinite(a[k])) for k in a.files),filename
plan={
 "title":"CETR-Net FIVES V2 one-time held-out evaluation, pre-registered from VALIDATION ONLY",
 "frozen_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "dataset_protocol":"FIVES_V2_NEARDUP_SCREENED",
 "test_archive_sha256":heldout["archive_sha256"],
 "test_records":200,"unique_visual_groups":197,
 "training_records_per_seed":508,"validation_records_per_seed":90,
 "seeds":list(SEEDS),"threshold":0.5,
 "metric_definitions":{"dice":"(2*intersection+1)/(pred_area+gt_area+1)",
                       "iou":"(intersection+1)/(union+1)",
                       "cldice":"centerline Dice using binary skeleton",
                       "aggregate":"unweighted mean of 197 visually grouped case means",
                       "ci":"2000 paired cluster bootstrap resamples at 197 visual group level"},
 "primary_predeclared":"dual_full_asymmetric",
 "primary_comparator":"independent_backbone",
 "secondary_variants":list(VARIANTS),
 "ablation_reviewer_check_passed":True,
 "model_selection_source":"validation only, best epoch for each of 21 configurations",
 "validation_best_variants_descriptive_only":best,
 "code_sha256":source_code,
 "backbones_sha256":{s:base_manifest["seeds"][s]["sha256"] for s in map(str,SEEDS)},
 "variants":variants_data,
 "inference_protocol":"prediction_only_before_any_ground_truth_access; no test-time GT-conditioned proposals; no TTA and no post-test threshold changes",
 "rules":["One fixed test execution per seed over all 7 predefined ablations; no test-based model selection",
          "Do NOT change threshold, weights, variants, model choice, or interpretation based on official-test labels",
          "Never use official test to train or validate",
          "Do not imply significance without CI or corrected hypotheses",
          "Do not claim superiority over other SOTA without protocol-matched independent baseline results"]
}
# Freeze source plan ONCE; identical copied test plan in Kaggle dataset.
master=OUT/"FROZEN_FIVES_V2_TEST_PLAN.json"
if master.is_file():
    previous=json.loads(master.read_text(encoding="utf8"))
    plan["frozen_utc"]=previous["frozen_utc"]
save(master,plan)
save(DST/"FROZEN_FIVES_V2_TEST_PLAN.json",plan)
for name in ("cetr_core_ablation.py","cetr_v2_suite.py"):
    source=B/name
    dest=DST/name
    if dest.exists():assert sha(dest)==sha(source)
    else:shutil.copy2(source,dest)
metadata={
 "id":"agusperdana/cetr-v2-frozen-refiner-heads-private",
 "title":"CETR V2 Frozen Refiner Heads for Official Test",
 "subtitle":"Twenty-one validated refiner checkpoints and frozen protocol",
 "licenses":[{"name":"unknown"}],
 "description":"Private 21 fully-trained CETR-Net FIVES V2 refiner heads, independent three-seed checkpoints, and pre-registered official held-out test evaluation. No test labels in this dataset."
}
save(DST/"dataset-metadata.json",metadata)
print("FIVES_OFFICIAL_PLAN_FROZEN",master,flush=True)
print("OFFICIAL_TEST_HEADS_VALIDATED",len(list(HEAD_DST.glob("*.npz"))),"FOR_3_SEEDS_7_VARIANTS",flush=True)
print("PRIMARY_HYPOTHESIS",plan["primary_predeclared"],"vs",plan["primary_comparator"],flush=True)
print("VALIDATION_BEST_DESCRIPTIVE",json.dumps(best),flush=True)
print("TEST_ARCHIVE_SHA256",heldout["archive_sha256"],flush=True)
