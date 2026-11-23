"""Build three sealed Kaggle GPU-only test notebooks, each one per seed.
Every notebook evaluates all seven preregistered variants on the entire 200-case official held-out,
without examining test labels for proposal generation or checkpoint selection.
"""
from pathlib import Path
import os,json,hashlib,ast,datetime
B=Path(r"D:\RISETKOLABORASI\CETR-Net\EXPERIMEN")
assert (B/"LAPTOP_GPU_DISABLED.flag").exists()
src=B/"evaluate_official_fives_kaggle.py"
out=B/"kaggle_transfer"/"OFFICIAL_FIVES_V2_KERNELS"
out.mkdir(parents=True,exist_ok=True)
eval_code=src.read_text(encoding="utf8")
ast.parse(eval_code)
plan=json.loads((B/"RESULTS_KAGGLE"/"OFFICIAL_TEST_V2"/"FROZEN_FIVES_V2_TEST_PLAN.json").read_text(encoding="utf8"))
assert len(plan["variants"])==3 and sum(len(d) for d in plan["variants"].values())==21
assert plan["primary_predeclared"]=="dual_full_asymmetric" and plan["threshold"]==.5
def cell(src):
    ast.parse(src)
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.splitlines(keepends=True)}
for seed in (42,123,2025):
    name=f"cetr-fives-v2-official-test-seed{seed}"
    folder=out/name
    folder.mkdir(exist_ok=True,parents=True)
    init=r'''
import os,sys,json,hashlib,runpy
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["CETR_OFFICIAL_SEED"]="SEED_PLACEHOLDER"
import tensorflow as tf
gpus=tf.config.list_physical_devices("GPU")
print("OFFICIAL_TEST_ENV",{"seed":os.environ["CETR_OFFICIAL_SEED"],
                           "tf":tf.__version__,"gpu_count":len(gpus)},flush=True)
assert gpus,"GPU Kaggle required; laptop GPU prohibited and no CPU substitution"
work=Path("/kaggle/working")
p=list(Path("/kaggle/input").rglob("FROZEN_FIVES_V2_TEST_PLAN.json"))
assert len(p)==1
plan=json.loads(p[0].read_text(encoding="utf8"))
for fname in ("cetr_core_ablation.py","cetr_v2_suite.py"):
    matched=list(Path("/kaggle/input").rglob(fname))
    assert len(matched)==1,("Module missing or nonunique",fname)
    with matched[0].open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
    assert digest==plan["code_sha256"][fname],"Frozen code checksum changed"
    (work/fname).write_bytes(matched[0].read_bytes())
sys.path.insert(0,str(work))
print("OFFICIAL_FIVES_CODE_SOURCES_VERIFIED",flush=True)
'''
    init=init.replace("SEED_PLACEHOLDER",str(seed))
    emit='''from pathlib import Path
import hashlib,runpy
work=Path("/kaggle/working")
payload=PAYLOAD_REPR
code_file=work/"evaluate_official_fives_kaggle.py"
code_file.write_text(payload,encoding="utf8")
digest=hashlib.sha256(payload.encode("utf8")).hexdigest()
print("EVALUATOR_SCRIPT_SHA256",digest,flush=True)
runpy.run_path(str(code_file),run_name="__main__")
'''.replace("PAYLOAD_REPR",repr(eval_code))
    nb={
      "cells":[{"cell_type":"markdown","metadata":{},"source":[
         f"# CETR-Net V2 — OFFICIAL FIVES 200-test evaluation, independent seed {seed}\n",
         "Locked one-time held-out test: 200 original records / 197 near-duplicate-screened groups.\n",
         "Seven frozen full-data ablation checkpoints; primary hypothesis dual_full_asymmetric vs independent backbone. Threshold 0.5. Validation-only checkpoint choice. No test-derived model selection and no GPU laptop training.\n"]},
         cell(init),cell(emit)],
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},
      "nbformat":4,"nbformat_minor":4}
    file=folder/f"official_fives_seed{seed}.ipynb"
    file.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding="utf8")
    meta={
      "id":f"agusperdana/{name}",
      "title":name,
      "code_file":file.name,
      "language":"python","kernel_type":"notebook",
      "is_private":True,"enable_gpu":True,"enable_internet":False,
      "machine_shape":"NvidiaTeslaT4",
      "dataset_sources":[
        "agusperdana/cetr-fives-v2-heldout-test-private",
        "agusperdana/cetr-v2-frozen-refiner-heads-private",
        "agusperdana/cetr-v2-backbones-private"],
      "kernel_sources":[],"competition_sources":[],"model_sources":[]}
    (folder/"kernel-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf8")
    print("KAGGLE_OFFICIAL_NOTEBOOK_READY",name,"byte_length",file.stat().st_size,
          "evaluator_sha256",hashlib.sha256(eval_code.encode("utf8")).hexdigest(),flush=True)
(B/"RESULTS_KAGGLE"/"OFFICIAL_TEST_V2"/"EVALUATOR_SHA256.txt").write_text(hashlib.sha256(eval_code.encode("utf8")).hexdigest()+"\n",encoding="utf8")
