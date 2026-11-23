"""FIVES full source-image near-duplicate screening; no changes to source/splits.
Thresholds are predeclared screening heuristics, not patient identity proofs.
"""
import csv,json,time,cv2,numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity
S=Path(r"D:\RISETKOLABORASI\Artikel Siap Kirim\1 NEXT PUBLICATION\4\EXPERIMEN_TAMBAHAN\IEEE_ACCESS_REVISION_EXPERIMENTS\LEAKSAFE_FIVES_SPLITS_V1")
W=Path(__file__).resolve().parent
src=S/"all_800_image_sha256_manifest.csv"
with src.open(encoding="utf8",newline="") as f:entries=list(csv.DictReader(f))
unique={}
for row in entries:unique.setdefault(row["image_sha256"],row)
data=list(unique.values())
ims={};ph={};key={}
for row in data:
    h=row["image_sha256"];path=row["image"].replace("/mnt/d/","D:/")
    im=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if im is None:raise FileNotFoundError(path)
    sm=cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA)
    ims[h]=sm
    dct=cv2.dct(cv2.resize(im,(32,32)).astype("float32"))[:8,:8].reshape(-1)
    med=float(np.median(dct[1:]))
    ph[h]=sum(1<<i for i,v in enumerate(dct[1:]) if v>med)
    key[h]=row
a=np.arange(128)-63.5;yy,xx=np.meshgrid(a,a,indexing="ij")
roi=(xx*xx+yy*yy)<(0.38*128)**2
gray488={};kp={}
orb=cv2.ORB_create(nfeatures=1600)
bf=cv2.BFMatcher(cv2.NORM_HAMMING)
def orb_match(h):
    if h not in kp:
        path=key[h]["image"].replace("/mnt/d/","D:/")
        gray488[h]=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(448,448))
        kp[h]=orb.detectAndCompute(gray488[h],None)
    return kp[h]
start=time.time();stats={"phash_lte8":0,"correlation_lte8":0,"ssim_pass":0,"orb_pass":0}
suspects=[]
hashes=list(unique)
for i,ha in enumerate(hashes):
    A=ims[ha][roi].astype(float);astd=A.std()
    for hb in hashes[i+1:]:
        dist=(ph[ha]^ph[hb]).bit_count()
        if dist>8:continue
        stats["phash_lte8"]+=1
        B=ims[hb][roi].astype(float);bstd=B.std()
        corr=float(np.corrcoef(A,B)[0,1]) if astd and bstd else 0.
        if corr<.92:continue
        stats["correlation_lte8"]+=1
        ssim=float(structural_similarity(ims[ha],ims[hb],data_range=255))
        if ssim<.82:continue
        stats["ssim_pass"]+=1
        k1,d1=orb_match(ha);k2,d2=orb_match(hb)
        good=[]
        if d1 is not None and d2 is not None:
            for pair in bf.knnMatch(d1,d2,k=2):
                if len(pair)==2 and pair[0].distance<.75*pair[1].distance:good.append(pair[0])
        inl=0
        if len(good)>=8:
            p1=np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            p2=np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            _,mask=cv2.findHomography(p1,p2,cv2.RANSAC,4.)
            if mask is not None:inl=int(mask.sum())
        frac=inl/len(good) if good else 0.
        if inl>=12 and frac>=.6:
            stats["orb_pass"]+=1
            suspects.append({"a_id":key[ha]["id"],"b_id":key[hb]["id"],"a_split":key[ha]["official_split"],"b_split":key[hb]["official_split"],"a_hash":ha,"b_hash":hb,"hamming":dist,"corr":round(corr,4),"ssim":round(ssim,4),"orb_inliers":inl,"orb_total":len(good),"orb_fraction":round(frac,3)})
parent={h:h for h in hashes}
def find(h):
    if parent[h]!=h:parent[h]=find(parent[h])
    return parent[h]
def union(a,b):
    pa,pb=find(a),find(b)
    if pa!=pb:parent[pb]=pa
for r in suspects:union(r["a_hash"],r["b_hash"])
groups={}
for h in hashes:groups.setdefault(find(h),[]).append(h)
clusters=[]
for vs in groups.values():
    if len(vs)<=1:continue
    rec=[{"id":key[h]["id"],"split":key[h]["official_split"],"sha":h,"label":key[h]["label_token"]} for h in vs]
    clusters.append({"members":rec,"cross_official_train_test":len(set(r["split"] for r in rec))>1})
clusters.sort(key=lambda r:(not r["cross_official_train_test"],-len(r["members"])))
exclude=sorted(set(rec["id"] for grp in clusters if grp["cross_official_train_test"] for rec in grp["members"] if rec["split"]=="official_train"))
summary={"method":"pHash<=8, central ROI Pearson>=.92, full SSIM>=.82, ORB RANSAC>=12 inliers, inlier fraction>=.6","source_records":len(entries),"unique_sha_images":len(unique),"candidate_stages":stats,"high_confidence_pairs":len(suspects),"linked_groups":len(clusters),"cross_train_test_groups":sum(c["cross_official_train_test"] for c in clusters),"proposed_train_exclusion_IDs":exclude,"elapsed_seconds":round(time.time()-start,1),"limits":["Potentially misses near duplicates with pHash distance >8, rotations, crop transformations","Does not establish patient identity","Candidates require visual confirmation","Original official test and historical manifests are not modified"]}
out={"summary":summary,"pairs":suspects,"clusters":clusters}
(W/"full_near_duplicate_audit.json").write_text(json.dumps(out,indent=2),encoding="utf8")
print(json.dumps(summary,indent=2))
