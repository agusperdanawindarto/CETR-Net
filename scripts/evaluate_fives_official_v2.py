"""LOCKED ONLY: official FIVES V2 test evaluation with frozen models and prediction-only proposals.
Refuse execution unless all 3 model+refiner seeds and frozen pre-registered plan exist.
This script is deliberately NOT run until training/ablation and model selection are complete.
"""
import os,json,hashlib,csv,datetime
from pathlib import Path
from collections import defaultdict
import numpy as np,tensorflow as tf
from cetr_core_ablation import CETRNet,candidate
from cetr_v2_suite import get_rows,load,scores,candidates_for_inference,atomic
WORK=Path("/mnt/d/RISETKOLABORASI/CETR-Net/EXPERIMEN")
TESTPLAN=WORK/"FROZEN_FIVES_V2_TEST_PLAN.json"
if not TESTPLAN.exists():raise RuntimeError("Official test locked: frozen test plan missing")
plan=json.loads(TESTPLAN.read_text(encoding="utf8"))
assert plan["dataset_protocol"]=="FIVES_V2_NEARDUP_SCREENED"
assert plan["threshold"]==0.5 and plan["seeds"]==[42,123,2025]
assert plan["ablation_reviewer_check_passed"] is True
W={
    42:WORK/"RESULTS_LOCAL"/"stage2_V2_seed42"/"seed42_backbone_frozen.weights.h5",
    123:WORK/"RESULTS_KAGGLE"/"independent_backbones_v1"/"cetr_v2"/"runs"/"seed_123"/"warmup_V2_FULL"/"best_backbone.weights.h5",
    2025:WORK/"RESULTS_KAGGLE"/"independent_backbones_v1"/"cetr_v2"/"runs"/"seed_2025"/"warmup_V2_FULL"/"best_backbone.weights.h5"
}
H={
    42:WORK/"RESULTS_LOCAL"/"stage2_V2_seed42"/"full"/"seed_42"/"dual_full_asymmetric"/"best_head_weights.npz",
    123:WORK/"RESULTS_LOCAL"/"stage2_V2_seed123"/"full"/"seed_123"/"dual_full_asymmetric"/"best_head_weights.npz",
    2025:WORK/"RESULTS_LOCAL"/"stage2_V2_seed2025"/"full"/"seed_2025"/"dual_full_asymmetric"/"best_head_weights.npz"
}
for seed in plan["seeds"]:
    for path in (W[seed],H[seed]):
        if not path.is_file():raise FileNotFoundError("Official-test lock: missing "+str(path))
    with W[seed].open("rb") as f:checksum=hashlib.file_digest(f,"sha256").hexdigest()
    assert plan["checkpoint_sha256"][str(seed)]==checksum,"Backbone changed after preregistration"
test=get_rows(WORK,42,"test")
assert len(test)==200,"Official FIVES test record count mismatch"
groups={}
groups_path=WORK/"SPLITS_NEARDUP_SAFE_V2"/"test_group_weights.csv"
with groups_path.open(newline="",encoding="utf8") as f:
    for r in csv.DictReader(f):
        groups[r["record_key"]]=r
assert len(groups)==200
OUT=WORK/"RESULTS_OFFICIAL_TEST_V2"
if OUT.exists():raise FileExistsError("Official test output already exists; refuse overwrite")
OUT.mkdir(parents=True)
for g in tf.config.list_physical_devices("GPU"):tf.config.experimental.set_memory_growth(g,True)
allruns=[]
for seed in plan["seeds"]:
    tf.keras.backend.clear_session()
    net=CETRNet(shape=(512,512,3))
    net.backbone.load_weights(str(W[seed]))
    dummy=tf.zeros((1,512,512,3),tf.float32)
    e=candidate((15,15),(28,28),"restore",0,(512,512))
    net.forward(dummy,[e],np.zeros((512,512),"float32"),training=False,evidence_mode="full")
    with np.load(H[seed]) as weight:
        for prefix,layer in (("shared",net.shared),("restore",net.restore),("reject",net.reject)):
            for i,v in enumerate(layer.weights):v.assign(weight[prefix+"_"+str(i)])
    recs=[]
    for row in test:
        image,mask=load(row,512)
        p0,_=net.backbone(image,training=False)
        mask0=p0.numpy()[0,:,:,0]
        binary,cands=candidates_for_inference(mask0)
        pred,_,_,_,_=net.forward(image,cands,binary,training=False,evidence_mode="full")
        ground=mask.numpy()[0,:,:,0]
        m_ref=scores(ground,pred.numpy()[0,:,:,0])
        m_base=scores(ground,mask0)
        key=row["record_key"]
        group=groups[key]["near_group"]
        recs.append({"seed":seed,"record_key":key,"group":group,
             **{"refiner_"+k:v for k,v in m_ref.items()},
             **{"backbone_"+k:v for k,v in m_base.items()},
             "candidate_count":len(cands)})
    groupscores=defaultdict(list)
    for row in recs:groupscores[row["group"]].append(row)
    grouped={key:{met:float(np.mean([r[met] for r in rs])) for met in
             ("refiner_dice","refiner_iou","refiner_cldice","backbone_dice","backbone_iou","backbone_cldice")}
             for key,rs in groupscores.items()}
    assert len(grouped)==197,"Near-duplicate group count unexpected"
    rng=np.random.default_rng(seed+300)
    vals=np.array([item["refiner_dice"] for item in grouped.values()])
    bootstrap=[float(np.mean(rng.choice(vals,size=len(vals),replace=True))) for _ in range(2000)]
    summary={"seed":seed,"n_record":len(recs),"n_unique_visual_groups":len(grouped),
        "fives_v2_test_groupwise_mean_dice":float(np.mean(vals)),
        "fives_v2_test_groupwise_dice_ci_95":list(np.percentile(bootstrap,[2.5,97.5])),
        "backbone_mean_dice":float(np.mean([v["backbone_dice"] for v in grouped.values()])),
        "refiner_minus_backbone_dice":float(np.mean([v["refiner_dice"]-v["backbone_dice"] for v in grouped.values()])),
        "limitations":["2D annotation only","Visual grouping not patient identity","No statistical comparison to MaskVSC until same V2 protocol"]}
    out=OUT/("seed_"+str(seed))
    out.mkdir(parents=True)
    atomic(out/"per_record.json",recs)
    atomic(out/"groupwise_197.json",grouped)
    atomic(out/"summary.json",summary)
    allruns.append(summary)
    print("OFFICIAL_TEST_SEED_COMPLETE",seed,summary,flush=True)
atomic(OUT/"FINAL_TEST_SUMMARY.json",{"completed_at":datetime.datetime.now().astimezone().isoformat(),
    "protocol":"FIVES_V2","test_groups":197,"seeds":allruns,
    "evidence":"prediction-derived candidate proposals; no test-time GT filtering"})
print("OFFICIAL_FIVES_V2_TEST_COMPLETE",len(allruns),flush=True)
