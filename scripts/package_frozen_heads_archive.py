from pathlib import Path
import zipfile,json,hashlib
ROOT=Path(r"D:\RISETKOLABORASI\CETR-Net\EXPERIMEN")
DST=ROOT/"kaggle_transfer"/"MIGRATE_AGUSPERDANA"/"fives_official_heads"
PL=json.loads((DST/"FROZEN_FIVES_V2_TEST_PLAN.json").read_text(encoding="utf8"))
target=DST/"heads_archive.zip"
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert len(list((DST/"heads").glob("*.npz")))==21
with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_STORED) as z:
    for seed,objs in PL["variants"].items():
        for var,fields in objs.items():
            source=DST/fields["checkpoint_file"]
            assert sha(source)==fields["checkpoint_sha256"],source
            z.write(source,arcname=fields["checkpoint_file"].replace("\\","/"))
with zipfile.ZipFile(target) as z:
    assert len(z.namelist())==21
    for seed,objs in PL["variants"].items():
        for var,fields in objs.items():
            assert hashlib.sha256(z.read(fields["checkpoint_file"])).hexdigest()==fields["checkpoint_sha256"]
print("HEAD_ARCHIVE_21_VERIFIED",target.stat().st_size,sha(target),flush=True)
