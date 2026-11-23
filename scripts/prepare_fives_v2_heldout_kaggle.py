"""Package strictly held-out FIVES V2 official TEST for a separate PRIVATE Kaggle dataset.
Never mount it in training or ablation notebooks. No local GPU use.
"""
from pathlib import Path
import csv,json,hashlib,zipfile,cv2
ROOT=Path(r"D:\RISETKOLABORASI\CETR-Net\EXPERIMEN")
source=ROOT/"SPLITS_NEARDUP_SAFE_V2"
dest=ROOT/"kaggle_transfer"/"heldout_test_dataset"
dest.mkdir(parents=True,exist_ok=True)
with (source/"seed_42"/"fives_test_split.csv").open(encoding="utf8",newline="") as f:
    records=list(csv.DictReader(f))
assert len(records)==200
with (source/"test_group_weights.csv").open(encoding="utf8",newline="") as f:
    group_rows=list(csv.DictReader(f))
grp={r["record_key"]:r for r in group_rows}
assert len(grp)==200
assert len({r["near_group"] for r in group_rows})==197
archive=dest/"fives_v2_official_heldout_512.zip"
manifest=[]
with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as z:
    for i,row in enumerate(records,1):
        assert row["official_split"]=="official_test"
        rid=row["record_key"].replace("/","_")
        key=row["record_key"]
        assert key in grp
        image=Path(row["image"].replace("/mnt/d/","D:/"))
        mask=Path(row["mask"].replace("/mnt/d/","D:/"))
        with image.open("rb") as f:source_image_hash=hashlib.file_digest(f,"sha256").hexdigest()
        with mask.open("rb") as f:source_mask_hash=hashlib.file_digest(f,"sha256").hexdigest()
        assert source_image_hash==row["image_sha256"],("Bad original image sha",rid)
        assert source_mask_hash==row["mask_sha256"],("Bad original mask sha",rid)
        im=cv2.imread(str(image),cv2.IMREAD_COLOR)
        gt=cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE)
        assert im is not None and gt is not None
        im=cv2.resize(im,(512,512),interpolation=cv2.INTER_LINEAR)
        gt=cv2.resize(gt,(512,512),interpolation=cv2.INTER_NEAREST)
        good,data=cv2.imencode(".png",im,[cv2.IMWRITE_PNG_COMPRESSION,3])
        assert good
        z.writestr("images/"+rid+".png",data.tobytes())
        good,data=cv2.imencode(".png",gt,[cv2.IMWRITE_PNG_COMPRESSION,3])
        assert good
        z.writestr("masks/"+rid+".png",data.tobytes())
        manifest.append({"record_key":key,"file_id":rid,"image_sha256_original":source_image_hash,
          "mask_sha256_original":source_mask_hash,
          "near_group":grp[key]["near_group"],
          "group_size":int(grp[key]["group_size"]),
          "weight":float(grp[key]["per_record_weight"])})
        if i%50==0:print("TEST_PACKAGE_PROGRESS",i,"/",len(records),flush=True)
    z.writestr("heldout_manifest.json",json.dumps(manifest,indent=2))
    z.writestr("README.txt","Official FIVES V2 TEST only. 200 records, 197 visually grouped cases. NEVER mount as training or validation data. Threshold/model must be frozen BEFORE use. Images resized 512 and masks nearest. Private user research evidence.\n")
metadata={"title":"CETR FIVES V2 Official Heldout Private",
    "subtitle":"Private 200-record held-out FIVES V2 evaluation package",
    "id":"ardhanabdillah/cetr-fives-v2-heldout-test-private",
    "licenses":[{"name":"unknown"}],
    "description":"Official FIVES test only, 200 records and 197 near-duplicate groups. Private data for FINAL locked evaluation, not training, checkpoint selection or ablation."}
(dest/"dataset-metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf8")
with archive.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
info={"records":200,"visual_groups":197,"archive_sha256":sha,"size_bytes":archive.stat().st_size,
    "test_labels_heldout":True,"protocol":"V2","training_access_prohibited":True}
(dest/"HELDOUT_AUDIT.json").write_text(json.dumps(info,indent=2),encoding="utf8")
print("HELDOUT_PRIVATE_DATASET_READY",json.dumps(info),flush=True)
