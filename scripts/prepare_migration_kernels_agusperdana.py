"""Migrate ONLY unfinished full refiner Kaggle notebooks to account agusperdana.
Completed 3 groups are preserved in private evidence dataset; no retraining.
All training uses Kaggle GPU; laptop GPU remains disabled.
"""
import json,ast,sys,hashlib
from pathlib import Path
B=Path(r"D:\RISETKOLABORASI\CETR-Net\EXPERIMEN")
assert (B/"LAPTOP_GPU_DISABLED.flag").exists()
SRC=B/"kaggle_transfer"/"FULL_STAGE2"
DST=B/"kaggle_transfer"/"MIGRATE_AGUSPERDANA"/"KERNELS"
MODEL=json.loads((B/"kaggle_transfer"/"MIGRATE_AGUSPERDANA"/"verified_backbones"/"SOURCE_MANIFEST.json").read_text())
TARGETS=((2025,"routing"),(42,"cdes"),(42,"routing"))
for seed,group in TARGETS:
    slug=f"cetr-v2-full-refiner-seed{seed}-{group}"
    source=SRC/slug
    nb_file=source/f"CETR_FULL_{seed}_{group}.ipynb"
    nb=json.loads(nb_file.read_text(encoding="utf8"))
    assert len(nb["cells"])==7 and nb["nbformat"]==4
    assert f"seed-{seed}" in "".join(nb["cells"][0]["source"])
    nb["cells"][0]["source"]=[
       f"# CETR-Net V2 migrated full-data refinement — {seed} / {group}\n",
       "Kaggle account agusperdana. Preserves provenance of independently trained frozen encoder on original ardhanabdillah account. FIVES V2 split 508 train/90 val. This is validation-only and uses no official test data.\n",
       f"Predeclared variant group: {group}, 8 epochs/variant. All training on Kaggle GPU, not laptop GPU.\n"
    ]
    first="".join(nb["cells"][1]["source"])
    assert 'ardhanabdillah/cetr-fives-v2-private-dev' in first
    first=first.replace('ardhanabdillah/cetr-fives-v2-private-dev','agusperdana/cetr-fives-v2-private-dev')
    assert "agusperdana/cetr-fives-v2-private-dev" in first
    nb["cells"][1]["source"]=first.splitlines(keepends=True)
    expected=MODEL["seeds"][str(seed)]["sha256"]
    validate=f'''
import os,json,hashlib
from pathlib import Path
seed={seed}
expected="{expected}"
found=[]
for here,dirs,files in os.walk("/kaggle/input",followlinks=True):
    if len(Path(here).relative_to("/kaggle/input").parts)>8:
        dirs.clear()
        continue
    if f"backbone_seed{{seed}}.weights.h5" in files:
        found.append(Path(here)/f"backbone_seed{{seed}}.weights.h5")
assert len(found)==1,"Migration checkpoint missing or ambiguous (expected unique Kaggle private model source)"
weights=found[0]
with weights.open("rb") as fp:
    got=hashlib.file_digest(fp,"sha256").hexdigest()
assert got==expected,"Migrated checkpoint differs from source SHA-256"
provenance_path=weights.parent/"SOURCE_MANIFEST.json"
assert provenance_path.is_file(),"Missing provenance manifest"
provenance=json.loads(provenance_path.read_text(encoding="utf8"))
assert provenance["seeds"][str(seed)]["sha256"]==expected
assert provenance["seeds"][str(seed)]["epochs_logged"]==45
assert provenance["official_test_scored"] is False
print("MIGRATED_BACKBONE_SHA256_OK",seed,got,flush=True)
'''
    nb["cells"][4]["source"]=validate.splitlines(keepends=True)
    assert "FULL_STAGE2_RUNNING" in "".join(nb["cells"][5]["source"])
    for c in nb["cells"]:
        if c["cell_type"]=="code":ast.parse("".join(c["source"]))
    out=DST/slug
    out.mkdir(parents=True,exist_ok=True)
    path=out/f"CETR_FULL_{seed}_{group}.ipynb"
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding="utf8")
    meta={
       "id":f"agusperdana/{slug}","title":slug,"code_file":path.name,
       "language":"python","kernel_type":"notebook","is_private":True,
       "enable_gpu":True,"enable_internet":False,"machine_shape":"NvidiaTeslaT4",
       "dataset_sources":["agusperdana/cetr-fives-v2-private-dev","agusperdana/cetr-v2-backbones-private"],
       "kernel_sources":[],"competition_sources":[],"model_sources":[]}
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf8")
    print("MIGRATED_NOTEBOOK_READY",meta["id"],"SOURCE_MODEL_SHA",expected[:16],"CODE_CELLS",sum(c["cell_type"]=="code" for c in nb["cells"]),flush=True)
# Store migration run order and source proof, without attaching FIVES official test in any training notebook.
manifest={"account_source":"ardhanabdillah","account_destination":"agusperdana",
 "training_data":"agusperdana/cetr-fives-v2-private-dev",
 "backbone_weights":"agusperdana/cetr-v2-backbones-private",
 "source_verified_groups":["123-cdes","123-routing","2025-cdes"],
 "remaining_groups":[f"{s}-{g}" for s,g in TARGETS],
 "official_test_in_training":False,
 "local_gpu_training":False}
(DST/"MIGRATION_PLAN.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
