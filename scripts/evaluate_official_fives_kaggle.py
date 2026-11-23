"""LOCKED FIVES V2 OFFICIAL held-out evaluation — Kaggle GPU, single seed per kernel.
Compares frozen independent backbone vs seven predeclared refinement variants, with 
ground truth read only after all prediction maps have been generated for each record.
NO training, no validation selection, no test-time threshold fitting or GT-conditioned proposals.
"""
import os,sys,datetime,hashlib,json,zipfile,time,platform,random
from pathlib import Path
from collections import defaultdict
import numpy as np,cv2,tensorflow as tf
import skimage
from cetr_core_ablation import CETRNet,candidate
from cetr_v2_suite import candidates_for_inference,scores,VARIANTS

ROOT=Path("/kaggle/working")
OUT=ROOT/"CETR_OFFICIAL_TEST_V2"
OUT.mkdir(parents=True,exist_ok=True)
SEED=int(os.environ.get("CETR_OFFICIAL_SEED","42"))
assert SEED in (42,123,2025)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
gpus=tf.config.list_physical_devices("GPU")
assert gpus,"Kaggle GPU required for requested official evaluation; refusing CPU fallback"
for g in gpus:
    try:tf.config.experimental.set_memory_growth(g,True)
    except RuntimeError:pass
def current():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def store(path,obj):
    path=Path(path)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    tmp.replace(path)
def lookup(name):
    hits=list(Path("/kaggle/input").rglob(name))
    assert len(hits)==1,("Expected one exact input file",name,[str(p) for p in hits])
    return hits[0]
test_manifest=lookup("heldout_manifest.json")
test_base=test_manifest.parent
model_manifest=lookup("SOURCE_MANIFEST.json")
plan_path=lookup("FROZEN_FIVES_V2_TEST_PLAN.json")
plan=json.loads(plan_path.read_text(encoding="utf8"))
source=json.loads(model_manifest.read_text(encoding="utf8"))
assert plan["dataset_protocol"]=="FIVES_V2_NEARDUP_SCREENED"
assert plan["threshold"]==.5 and plan["seeds"]==[42,123,2025]
assert plan["test_records"]==200 and plan["unique_visual_groups"]==197
assert plan["model_selection_source"].startswith("validation only")
assert plan["ablation_reviewer_check_passed"] is True
assert plan["primary_predeclared"]=="dual_full_asymmetric"
assert plan["primary_comparator"]=="independent_backbone"
assert len(plan["secondary_variants"])==7
assert plan["inference_protocol"].startswith("prediction_only")
audit=json.loads(lookup("HELDOUT_AUDIT.json").read_text(encoding="utf8"))
assert audit["archive_sha256"]==plan["test_archive_sha256"],"Extracted held-out archive provenance mismatch"
assert audit["records"]==plan["test_records"]==200 and audit["visual_groups"]==197
head_cache=lookup("seed42_natural_full_symmetric.npz").parent.parent
assert head_cache.name=="heads_archive","Unexpected head checkpoint archive mount"
assert len(list(head_cache.rglob("*.npz")))==21
print("FROZEN_HEADS_EXTRACTED_MOUNT_VERIFIED",len(list(head_cache.rglob("*.npz"))),flush=True)
assert source["official_test_scored"] is False
weights=model_manifest.parent/f"backbone_seed{SEED}.weights.h5"
assert weights.exists() and sha(weights)==source["seeds"][str(SEED)]["sha256"]
assert sha(weights)==plan["backbones_sha256"][str(SEED)]
assert source["seeds"][str(SEED)]["epochs_logged"]==45
for source_name in ("cetr_core_ablation.py","cetr_v2_suite.py"):
    path=Path(__file__).parent/source_name
    assert sha(path)==plan["code_sha256"][source_name],("Inference source mismatch",source_name)
VAR_ORDER=plan["secondary_variants"]
assert VAR_ORDER==["natural_full_symmetric","break_only","false_only","dual_full_symmetric",
                   "dual_geometry","dual_confidence","dual_full_asymmetric"]
heads={}
for variant in VAR_ORDER:
    ref=plan["variants"][str(SEED)][variant]
    path=head_cache/ref["checkpoint_file"]
    assert path.is_file()
    assert sha(path)==ref["checkpoint_sha256"],("Frozen head mismatch",SEED,variant)
    assert ref["epochs_completed"]==8
    with np.load(path) as npz:
        names=tuple(npz.files)
        weights_by_kind={}
        for family in ("shared","restore","reject"):
            indexed=[]
            for key in names:
                if key.startswith(family+"_"):
                    ix=int(key.split("_")[-1])
                    indexed.append((ix,np.asarray(npz[key]).copy()))
            indexed.sort()
            assert indexed and [i for i,_ in indexed]==list(range(len(indexed)))
            assert all(np.all(np.isfinite(v)) for _,v in indexed)
            weights_by_kind[family]=[v for _,v in indexed]
    heads[variant]=weights_by_kind
print("FIVES_TEST_MODEL_PROVENANCE_VALID",{"seed":SEED,"backbone_sha256":sha(weights),
      "variants":VAR_ORDER,"test_sha256":audit["archive_sha256"]},flush=True)
model=CETRNet(shape=(512,512,3))
model.backbone.load_weights(str(weights))
model.backbone.trainable=False
dummy=tf.zeros((1,512,512,3),tf.float32)
probe=candidate((20,20),(32,32),"restore",0,(512,512))
model.forward(dummy,[probe],np.zeros((512,512),np.float32),training=False,evidence_mode="full")
HEAD_LAYERS={"shared":model.shared,"restore":model.restore,"reject":model.reject}
for variant in VAR_ORDER:
    for family,layer in HEAD_LAYERS.items():
        learned=heads[variant][family]
        assert len(learned)==len(layer.get_weights()),(variant,family)
        for old,new in zip(learned,layer.get_weights()):
            assert old.shape==new.shape,(variant,family,old.shape,new.shape)
del dummy
img_dir=OUT/"qualitative_fixed_ids"
img_dir.mkdir(exist_ok=True)
PER_RECORD=OUT/"per_record_progress.jsonl"
if PER_RECORD.exists():raise RuntimeError("Official seed evaluation already started: refusing overwrite")
start=time.monotonic()
metric_keys=("dice","iou","cldice")
if True:
    records=json.loads(test_manifest.read_text(encoding="utf8"))
    assert len(records)==200
    assert len({r["record_key"] for r in records})==200
    assert len({r["file_id"] for r in records})==200
    assert len({r["near_group"] for r in records})==197
    assert all(r["group_size"]>=1 for r in records)
    fixed_ids={records[j]["record_key"] for j in (0,39,79,119,159,199)}
    all_rows=[]
    with PER_RECORD.open("a",encoding="utf8") as f:
        for i,rec in enumerate(records,1):
            fid=rec["file_id"]
            img=cv2.imread(str(test_base/"images"/(fid+".png")),cv2.IMREAD_COLOR)
            assert img is not None and img.shape[:2]==(512,512)
            im_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            x=tf.convert_to_tensor(im_rgb.astype("float32")[None]/255.)
            p0,_=model.backbone(x,training=False)
            pp=np.asarray(p0.numpy()[0,:,:,0],dtype=np.float32)
            binary,cands=candidates_for_inference(pp)
            assert len(cands)<=12
            predictions={}
            for variant in VAR_ORDER:
                for family,layer in HEAD_LAYERS.items():
                    layer.set_weights(heads[variant][family])
                mode=VARIANTS[variant]["evidence"]
                p,_,_,_,_=model.forward(x,cands,binary,training=False,evidence_mode=mode)
                predictions[variant]=np.asarray(p.numpy()[0,:,:,0],dtype=np.float32)
                assert predictions[variant].shape==(512,512)
                assert np.all(np.isfinite(predictions[variant]))
            # Crucial: open test annotations ONLY AFTER all model predictions exist.
            gt=cv2.imread(str(test_base/"masks"/(fid+".png")),cv2.IMREAD_GRAYSCALE)
            assert gt is not None and gt.shape==(512,512)
            truth=(gt>127).astype(np.float32)
            assert set(np.unique(truth)).issubset({0.,1.})
            bas=scores(truth,pp)
            variant_scores={v:scores(truth,predictions[v]) for v in VAR_ORDER}
            row={"record_key":rec["record_key"],"near_group":rec["near_group"],
                 "file_id":fid,"seed":SEED,"n_candidates":len(cands),
                 "backbone":bas,"variants":variant_scores}
            all_rows.append(row)
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
            f.flush()
            if rec["record_key"] in fixed_ids:
                target=predictions["dual_full_asymmetric"]
                panel=np.concatenate([
                    im_rgb,
                    cv2.cvtColor((truth*255).astype(np.uint8),cv2.COLOR_GRAY2RGB),
                    cv2.cvtColor(((pp>.5)*255).astype(np.uint8),cv2.COLOR_GRAY2RGB),
                    cv2.cvtColor(((target>.5)*255).astype(np.uint8),cv2.COLOR_GRAY2RGB)],axis=1)
                out=img_dir/(f"{i:03d}_input_GT_backbone_proposed.png")
                assert cv2.imwrite(str(out),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
            if i==1 or i%25==0 or i==200:
                state={"status":"RUNNING","seed":SEED,"n_processed":i,"n_total":200,
                       "elapsed_sec":round(time.monotonic()-start,2),"time_utc":current()}
                store(OUT/"PROGRESS.json",state)
                print("FIVES_OFFICIAL_TEST_PROGRESS",json.dumps(state),flush=True)
                del state
assert len(all_rows)==200
visual_groups=sorted({r["near_group"] for r in all_rows})
assert len(visual_groups)==197
groupwise={}
for grp in visual_groups:
    subgroup=[x for x in all_rows if x["near_group"]==grp]
    groupwise[grp]={"n_records":len(subgroup),
      "backbone":{k:float(np.mean([r["backbone"][k] for r in subgroup])) for k in metric_keys},
      "variants":{v:{k:float(np.mean([r["variants"][v][k] for r in subgroup])) for k in metric_keys} for v in VAR_ORDER}}
store(OUT/"groupwise_197.json",groupwise)
store(OUT/"per_record_200.json",all_rows)
def ci_and_delta(a,b,seedcode):
    a=np.asarray(a,np.float64)
    b=np.asarray(b,np.float64)
    assert len(a)==len(b)==197
    rng=np.random.default_rng(seedcode)
    draw=rng.integers(0,197,size=(2000,197))
    differences=a-b
    bs=differences[draw].mean(axis=1)
    return {"paired_delta_mean":float(np.mean(differences)),
            "paired_delta_95pct_cluster_bootstrap_ci":[float(z) for z in np.quantile(bs,[.025,.975])],
            "n_visual_groups":197}
by_model={"independent_backbone":{
    k:float(np.mean([groupwise[g]["backbone"][k] for g in visual_groups])) for k in metric_keys}}
baseline_dice=[groupwise[g]["backbone"]["dice"] for g in visual_groups]
variant_results={}
for idx,v in enumerate(VAR_ORDER):
    values=[groupwise[g]["variants"][v]["dice"] for g in visual_groups]
    m={k:float(np.mean([groupwise[g]["variants"][v][k] for g in visual_groups])) for k in metric_keys}
    m["vs_backbone"]=ci_and_delta(values,baseline_dice,SEED*100+idx)
    m["vs_natural"]=ci_and_delta(values,[groupwise[g]["variants"]["natural_full_symmetric"]["dice"] for g in visual_groups],SEED*200+idx)
    variant_results[v]=m
summary={
    "status":"COMPLETE","seed":SEED,"source":"Kaggle GPU private official FIVES V2 held-out",
    "completed_utc":current(),"n_records":200,"n_unique_visual_groups":197,
    "n_variants":7,"threshold":plan["threshold"],"primary_variant":"dual_full_asymmetric",
    "primary_comparator":"independent_backbone",
    "prior_validation_best_variant":plan["validation_best_variants_descriptive_only"][str(SEED)]["variant"],
    "backbone":by_model["independent_backbone"],"refiners":variant_results,
    "test_archive_sha256":plan["test_archive_sha256"],
    "backbone_sha256":plan["backbones_sha256"][str(SEED)],
    "frozen_plan_sha256":sha(plan_path),"validation_only_model_selection":True,
    "test_time_ground_truth_proposals":False,
    "groupwise_statistics":"unweighted mean across 197 screened visual groups, paired 2000 bootstrap CI",
    "limitations":["Near-duplicate visual grouping does not demonstrate patient-disjoint test",
                   "2D FIVES masks only; no 3D anatomical connectivity validation",
                   "No newly trained protocol-matched MaskVSC or SOTA baselines in this test",
                   "Secondary 7-variant comparisons are descriptive; no test-time model selection"],
    "runtime":{"tf":tf.__version__,"numpy":np.__version__,"skimage":skimage.__version__,
               "kaggle_gpu_count":len(gpus),"elapsed_seconds":round(time.monotonic()-start,2)}}
store(OUT/"OFFICIAL_FIVES_V2_SEED_SUMMARY.json",summary)
store(OUT/"PROGRESS.json",{"status":"COMPLETE","seed":SEED,"n_processed":200,"n_total":200,"time_utc":current()})
print("FIVES_V2_OFFICIAL_TEST_SEED_COMPLETE",SEED,
      "BASELINE_DICE",round(summary["backbone"]["dice"],6),
      "PRIMARY_DICE",round(summary["refiners"]["dual_full_asymmetric"]["dice"],6),
      "PRIMARY_DELTA",round(summary["refiners"]["dual_full_asymmetric"]["vs_backbone"]["paired_delta_mean"],6),
      flush=True)
