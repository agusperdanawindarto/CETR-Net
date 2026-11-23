"""CPU-only Kaggle preflight validates private dataset mounts and frozen hashes.
Does not train, use test ground-truth masks, or score official FIVES test.
"""
import json,ast
from pathlib import Path
B=Path(r"D:\RISETKOLABORASI\CETR-Net\EXPERIMEN")
out=B/"kaggle_transfer"/"OFFICIAL_FIVES_V2_KERNELS"/"cetr-fives-v2-official-preflight-cpu"
out.mkdir(parents=True,exist_ok=True)
src=r'''
import hashlib,json,zipfile
from pathlib import Path
P=Path("/kaggle/input")
def locate(name):
    hits=list(P.rglob(name))
    assert len(hits)==1,(name,[str(z) for z in hits])
    return hits[0]
plan_path=locate("FROZEN_FIVES_V2_TEST_PLAN.json")
plan=json.loads(plan_path.read_text(encoding="utf8"))
model=json.loads(locate("SOURCE_MANIFEST.json").read_text(encoding="utf8"))
test_manifest=locate("heldout_manifest.json")
head_base=locate("seed42_natural_full_symmetric.npz").parent.parent
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
audit=json.loads(locate("HELDOUT_AUDIT.json").read_text(encoding="utf8"))
assert audit["archive_sha256"]==plan["test_archive_sha256"]
assert audit["records"]==200 and audit["visual_groups"]==197
assert plan["seeds"]==[42,123,2025] and plan["threshold"]==.5
assert plan["primary_predeclared"]=="dual_full_asymmetric"
records=json.loads(test_manifest.read_text(encoding="utf8"))
assert len(records)==200 and len(set(x["near_group"] for x in records))==197
assert all((test_manifest.parent/"images"/(x["file_id"]+".png")).is_file() for x in records)
assert all((test_manifest.parent/"masks"/(x["file_id"]+".png")).is_file() for x in records)
assert len(list(head_base.rglob("*.npz")))==21
for seed in plan["seeds"]:
    for var,record in plan["variants"][str(seed)].items():
        weight=head_base/record["checkpoint_file"]
        assert weight.is_file() and sha(weight)==record["checkpoint_sha256"]
assert sum(len(x) for x in plan["variants"].values())==21
for seed in plan["seeds"]:
    w=model["seeds"][str(seed)]
    backbone=model.keys()
    candidates=list(P.rglob(w["weight_file"]))
    assert len(candidates)==1
    assert sha(candidates[0])==w["sha256"]==plan["backbones_sha256"][str(seed)]
for file in ("cetr_core_ablation.py","cetr_v2_suite.py"):
    source=locate(file)
    assert sha(source)==plan["code_sha256"][file]
o=Path("/kaggle/working")/"CETR_FIVES_V2_OFFICIAL_PREFLIGHT.json"
o.write_text(json.dumps({"status":"PASS","test_records":200,"visual_groups":197,
"head_checkpoints":21,"backbones":3,"threshold":.5,
"test_labels_scored":False,"ground_truth_masks_opened":False,
"datasets":["heldout-private","frozen-heads-private","backbones-private"],
"test_archive_sha256":plan["test_archive_sha256"]},indent=2))
print("FIVES_OFFICIAL_PREFLIGHT_PASS_200_197_21",flush=True)
'''
ast.parse(src)
nb={"cells":[{"cell_type":"markdown","metadata":{},"source":["# CETR FIVES V2 Official GPU Evaluation — CPU provenance preflight\n","NO GPU training and NO test labels scored. Verifies all frozen Kaggle datasets/checkpoints BEFORE official once-per-seed scoring.\n"]},{"cell_type":"code","metadata":{},"outputs":[],"execution_count":None,"source":src.splitlines(keepends=True)}],
"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":4}
(out/"preflight.ipynb").write_text(json.dumps(nb,indent=2),encoding="utf8")
meta={"id":"agusperdana/cetr-fives-v2-official-preflight-cpu","title":"cetr-fives-v2-official-preflight-cpu",
 "code_file":"preflight.ipynb","language":"python","kernel_type":"notebook",
 "is_private":True,"enable_gpu":False,"enable_internet":False,
 "dataset_sources":["agusperdana/cetr-fives-v2-heldout-test-private",
 "agusperdana/cetr-v2-frozen-refiner-heads-private","agusperdana/cetr-v2-backbones-private"],
 "kernel_sources":[],"competition_sources":[],"model_sources":[]}
(out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2))
print("OFFICIAL_TEST_CPU_PREFLIGHT_NOTEBOOK_READY",out)
