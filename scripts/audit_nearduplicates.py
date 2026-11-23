"""Conservative perceptual-hash candidate audit. Similar hashes are NOT proof of leakage."""
import csv,json,cv2,math
from pathlib import Path
from itertools import combinations
OLD=Path(r"D:\RISETKOLABORASI\Artikel Siap Kirim\1 NEXT PUBLICATION\4\EXPERIMEN_TAMBAHAN\IEEE_ACCESS_REVISION_EXPERIMENTS")
def win(p):
    return Path(p.replace("/mnt/d/","D:/")) if p.startswith("/mnt/d/") else Path(p)
S=OLD/"LEAKSAFE_FIVES_SPLITS_V1"
OUT=Path(__file__).resolve().parent
def get(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def phash(path):
    im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if im is None:raise FileNotFoundError(path)
    sm=cv2.resize(im,(32,32),interpolation=cv2.INTER_AREA).astype("float32")
    dct=cv2.dct(sm)[:8,:8].reshape(-1)
    med=float(sorted(dct[1:])[31])
    return sum((1<<i) for i,v in enumerate(dct[1:]) if v>med)
seedset={}
hashes={}
for seed in (42,123,2025):
    sr={}
    for part in ("train","validation","test"):
        dat=get(S/f"seed_{seed}"/f"fives_{part}_split.csv")
        uniq={}
        for r in dat:uniq.setdefault(r["image_sha256"],r)
        sr[part]=list(uniq.values())
        for r in uniq.values():
            if r["image_sha256"] not in hashes:hashes[r["image_sha256"]]=phash(win(r["image"]))
    seedset[seed]=sr
rep={"method":"pHash 32x32 DCT, lower frequency 63-bit; candidate visual similarities only","thresholds":[4,6,8],"per_seed":{},"not_patient_level_proof":True}
for seed,sr in seedset.items():
    details={}
    for part_a,part_b in (("train","validation"),("train","test"),("validation","test")):
        candidates=[]
        for a in sr[part_a]:
            ha=hashes[a["image_sha256"]]
            for b in sr[part_b]:
                hb=hashes[b["image_sha256"]]
                hamming=(ha^hb).bit_count()
                if hamming<=8:
                    candidates.append({"a":a["id"],"b":b["id"],"hamming":hamming,"a_sha":a["image_sha256"],"b_sha":b["image_sha256"]})
        candidates.sort(key=lambda x:x["hamming"])
        details[f"{part_a}_vs_{part_b}"]={"counts":{str(t):sum(x["hamming"]<=t for x in candidates) for t in (4,6,8)},"top_candidates":candidates[:30]}
    rep["per_seed"][str(seed)]=details
(OUT/"near_duplicate_candidate_report.json").write_text(json.dumps(rep,indent=2),encoding="utf8")
print(json.dumps(rep,indent=2))
