from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,os,sys
OLD=Path(r"D:\RISETKOLABORASI\Artikel Siap Kirim\1 NEXT PUBLICATION\4\EXPERIMEN_TAMBAHAN\IEEE_ACCESS_REVISION_EXPERIMENTS")
S=OLD/"LEAKSAFE_FIVES_SPLITS_V1"
D=OLD/"DATASET"/"FIVES"
R=OLD/"LEAKSAFE_FIVES_RETRAIN_V1"/"full"
OUT=Path(__file__).resolve().parent
def win(p):
    return Path(p.replace("/mnt/d/","D:/")) if p.startswith("/mnt/d/") else Path(p)
def rows(p):
    with open(p,encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
rep={"source":str(OLD),"fives":{},"seed_splits":{},"baseline_checkpoints":{},"errors":[]}
for part in ("train","test"):
    imgs={x.stem.lower():x for x in (D/part/"Original").glob("*.png")}
    masks={x.stem.lower():x for x in (D/part/"Ground truth").glob("*.png")}
    rep["fives"][part]={"images":len(imgs),"masks":len(masks),"paired":len(imgs.keys()&masks.keys()),"missing_masks":list(imgs.keys()-masks.keys())[:10]}
allpaths={}
for seed in (42,123,2025):
    splits={}
    for part in ("train","validation","test"):
        dat=rows(S/f"seed_{seed}"/f"fives_{part}_split.csv")
        splits[part]=dat
        for row in dat:
            for k in ("image","mask"):
                fp=win(row[k]);allpaths[str(fp)]=(fp,row["image_sha256"] if k=="image" else row["mask_sha256"])
    groups={p:set(r["image_sha256"] for r in rr) for p,rr in splits.items()}
    cross={f"{a}_{b}":len(groups[a]&groups[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))}
    rep["seed_splits"][str(seed)]={"records":{p:len(v) for p,v in splits.items()},"unique_image_hashes":{p:len(groups[p]) for p in splits},"hash_overlap":cross,"non_ok":{p:sum(r.get("status")!="OK" for r in v) for p,v in splits.items()}}
    if any(cross.values()):rep["errors"].append("hash overlap: "+str(seed))
    if [len(splits[p]) for p in ("train","validation","test")]!=[510,90,200]:rep["errors"].append("count discrepancy: "+str(seed))
for model in ("U_Net","ResUNet","Attention_UNet","EAG_UNet","MS_SE_Attention_UNet"):
    rep["baseline_checkpoints"][model]={str(seed):{"checkpoint":(R/model/f"seed_{seed}"/"best.weights.h5").exists(),"evaluation":(R/model/f"seed_{seed}"/"evaluation_summary.json").exists()} for seed in (42,123,2025)}
missing=[];bad_hash=[];verified=0
for s,(p,digest) in allpaths.items():
    if not p.is_file(): missing.append(s);continue
    if digest:
        h=hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda:f.read(2*1024*1024),b""):h.update(chunk)
        verified+=1
        if h.hexdigest().lower()!=digest.lower():bad_hash.append(s)
rep["files"]={"listed":len(allpaths),"sha256_verified":verified,"missing":missing[:20],"missing_count":len(missing),"bad_sha256":bad_hash[:20],"bad_sha256_count":len(bad_hash)}
if missing or bad_hash:rep["errors"].append("missing or corrupt source files")
if any(not v["checkpoint"] or not v["evaluation"] for vv in rep["baseline_checkpoints"].values() for v in vv.values()):rep["errors"].append("baseline artifact missing")
rep["status"]="PASS" if not rep["errors"] else "FAIL"
(OUT/"data_integrity_report.json").write_text(json.dumps(rep,indent=2),encoding="utf-8")
print(json.dumps(rep,indent=2))
