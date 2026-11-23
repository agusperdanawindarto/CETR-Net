from pathlib import Path
import json
B=Path(r"D:\RISETKOLABORASI\CETR-Net\EXPERIMEN")
D=B/"kaggle_transfer"/"OFFICIAL_FIVES_V2_KERNELS"/"cetr-fives-v2-mount-inspector-cpu"
D.mkdir(parents=True,exist_ok=True)
src=r'''
import os,json
from pathlib import Path
b=Path("/kaggle/input")
checks=("heldout_manifest.json","fives_v2_official_heldout_512.zip",
        "HELDOUT_AUDIT.json","heads_archive.zip","seed42_natural_full_symmetric.npz",
        "FROZEN_FIVES_V2_TEST_PLAN.json","SOURCE_MANIFEST.json","backbone_seed42.weights.h5")
out={}
for name in checks:
    hits=list(b.rglob(name))
    out[name]=[str(x) for x in hits[:5]]
    print("MOUNT_MATCH",name,out[name],flush=True)
print("MOUNT_ROOT",list(str(x) for x in b.iterdir()),flush=True)
summary=Path("/kaggle/working")/"CETR_FIVES_V2_MOUNT_INSPECTOR.json"
summary.write_text(json.dumps(out,indent=2))
'''
nb={"cells":[{"cell_type":"code","metadata":{},"outputs":[],"execution_count":None,"source":src.splitlines(keepends=True)}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":4}
(D/"inspection.ipynb").write_text(json.dumps(nb,indent=2),encoding="utf8")
meta={"id":"agusperdana/cetr-fives-v2-mount-inspector-cpu",
"title":"cetr-fives-v2-mount-inspector-cpu","code_file":"inspection.ipynb",
"language":"python","kernel_type":"notebook","is_private":True,
"enable_gpu":False,"enable_internet":False,
"dataset_sources":["agusperdana/cetr-fives-v2-heldout-test-private",
"agusperdana/cetr-v2-frozen-refiner-heads-private","agusperdana/cetr-v2-backbones-private"],
"kernel_sources":[],"competition_sources":[],"model_sources":[]}
(D/"kernel-metadata.json").write_text(json.dumps(meta,indent=2))
print("MOUNT_INSPECTOR_READY")
