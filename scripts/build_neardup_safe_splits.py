"""Construct a NEW conservative split; do NOT modify official FIVES files or historical manifests.
Near-duplicate groups are pHash/SSIM/ORB review candidates; no patient-level assertion.
"""
import csv,json,random
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(r"D:\RISETKOLABORASI\Artikel Siap Kirim\1 NEXT PUBLICATION\4\EXPERIMEN_TAMBAHAN\IEEE_ACCESS_REVISION_EXPERIMENTS\LEAKSAFE_FIVES_SPLITS_V1")
WORK=Path(__file__).resolve().parent
OUT=WORK/"SPLITS_NEARDUP_SAFE_V2"
OUT.mkdir(exist_ok=True)
with (ROOT/"all_800_image_sha256_manifest.csv").open(encoding="utf8",newline="") as f: rows=list(csv.DictReader(f))
evidence=json.loads((WORK/"full_near_duplicate_audit.json").read_text())
orig_header=list(rows[0])
shas=sorted({r["image_sha256"] for r in rows})
parent={h:h for h in shas}
def find(x):
    if parent[x]!=x:parent[x]=find(parent[x])
    return parent[x]
def union(a,b):
    aa,bb=find(a),find(b)
    if aa!=bb:parent[max(aa,bb)]=min(aa,bb)
for p in evidence["pairs"]:union(p["a_hash"],p["b_hash"])
grp={h:find(h) for h in shas}
train0=[r for r in rows if r["official_split"]=="official_train"]
test0=[r for r in rows if r["official_split"]=="official_test"]
test_groups={grp[r["image_sha256"]] for r in test0}
excluded=[r for r in train0 if grp[r["image_sha256"]] in test_groups]
train_ok=[r for r in train0 if grp[r["image_sha256"]] not in test_groups]
if len(excluded)<2:raise RuntimeError("Expected at least two vulnerable official-train records")
train_groups=defaultdict(list)
for r in train_ok:train_groups[grp[r["image_sha256"]]].append(r)
expected=Counter(r["label_token"] for r in train_ok)
colnames=orig_header+["record_key","near_duplicate_group","protocol"]
def write(path,data,split):
    with path.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=colnames);w.writeheader()
        for row in data:
            x={**row,"split":split,"record_key":row["official_split"]+"/"+row["id"],"near_duplicate_group":grp[row["image_sha256"]],"protocol":"near_duplicate_screened_v2"}
            w.writerow(x)
reports={}
for seed in (42,123,2025):
    best=None
    keys=list(train_groups)
    for trial in range(200):
        rng=random.Random(seed*10000+trial); cand=keys[:]; rng.shuffle(cand)
        selected=[];count=0
        for k in cand:
            size=len(train_groups[k])
            if count+size<=90:
                selected.append(k);count+=size
            if count==90:break
        vc=Counter(r["label_token"] for k in selected for r in train_groups[k])
        score=abs(count-90)*100 + sum(abs(vc[l]-90*expected[l]/len(train_ok)) for l in expected)
        if best is None or score<best[0]:best=(score,set(selected))
    valkeys=best[1]
    training=[r for r in train_ok if grp[r["image_sha256"]] not in valkeys]
    validation=[r for r in train_ok if grp[r["image_sha256"]] in valkeys]
    train_gp={grp[r["image_sha256"]] for r in training}
    val_gp={grp[r["image_sha256"]] for r in validation}
    assert not (train_gp & val_gp or train_gp & test_groups or val_gp & test_groups)
    assert len(training)+len(validation)+len(excluded)==600
    dest=OUT/f"seed_{seed}";dest.mkdir(exist_ok=True)
    for part,rs in (("train",training),("validation",validation),("test",test0)):
        write(dest/f"fives_{part}_split.csv",rs,part)
    reports[str(seed)]={"train":len(training),"validation":len(validation),"test":len(test0),"excluded":len(excluded),"train_groups":len(train_gp),"validation_groups":len(val_gp),"test_groups":len(test_groups),"train_labels":dict(Counter(r["label_token"] for r in training)),"validation_labels":dict(Counter(r["label_token"] for r in validation)),"zero_group_overlap":True}
with (OUT/"excluded_train_records.csv").open("w",encoding="utf8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=colnames);w.writeheader()
    for r in excluded:w.writerow({**r,"split":"excluded","record_key":r["official_split"]+"/"+r["id"],"near_duplicate_group":grp[r["image_sha256"]],"protocol":"near_duplicate_screened_v2"})
test_weights=Counter(grp[r["image_sha256"]] for r in test0)
with (OUT/"test_group_weights.csv").open("w",encoding="utf8",newline="") as f:
    w=csv.writer(f);w.writerow(["record_key","sha256","near_group","group_size","per_record_weight"])
    for r in test0:
        g=grp[r["image_sha256"]]
        w.writerow([r["official_split"]+"/"+r["id"],r["image_sha256"],g,test_weights[g],1/test_weights[g]])
summary={"input":"FIVES 800 records preserved, original official test unchanged","official_train_records":600,"train_excluded":len(excluded),"excluded_record_keys":[r["official_split"]+"/"+r["id"] for r in excluded],"official_test_records":len(test0),"unique_test_image_groups":len(test_weights),"near_duplicate_pairs_reviewed":len(evidence["pairs"]),"by_seed":reports,"data_policy":["All baseline/proposed methods must be retrained under V2 for direct comparisons","Official FIVES test labels never used for training, proposals, checkpoint or threshold selection","pHash/SSIM/ORB is a conservative near-duplicate screen, not patient-disjoint proof","Historical SHA-only V1 manifests remain unchanged"]}
(OUT/"v2_integrity_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
print(json.dumps(summary,indent=2))
