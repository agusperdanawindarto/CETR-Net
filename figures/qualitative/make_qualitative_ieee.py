from pathlib import Path
from PIL import Image,ImageOps,ImageDraw,ImageFont
import hashlib,json
R=Path(r"D:\RISETKOLABORASI\CETR-Net\EXPERIMEN\RESULTS_KAGGLE\OFFICIAL_TEST_V2\seed_42\CETR_OFFICIAL_TEST_V2\qualitative_fixed_ids")
O=Path(__file__).resolve().parent
headers=["Fundus image","Reference mask","Frozen backbone","CETR-Net (primary)"]
font=ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf",22)
small=ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf",17)
result=[]
for case in ("001","080","160"):
    src=R/f"{case}_input_GT_backbone_proposed.png"
    assert src.is_file(),src
    im=Image.open(src).convert("RGB")
    assert im.width==im.height*4,(im.size,src)
    s=im.height
    out=Image.new("RGB",(2*s+30,2*s+130),"white")
    drawer=ImageDraw.Draw(out)
    drawer.text((13,10),f"Fixed official held-out case {case} | seed 42 | prediction-only inference",fill=(25,42,65),font=font)
    for n in range(4):
        x=12+(n%2)*(s+6)
        y=65+(n//2)*(s+15)
        crop=im.crop((n*s,0,(n+1)*s,s))
        out.paste(crop,(x,y))
        drawer.text((x+7,y+s-28),headers[n],font=small,fill="white",stroke_width=2,stroke_fill="black")
    dest=O/f"Fig8_case_{case}_official_test_FIVES.png"
    out.save(dest,dpi=(600,600),optimize=True)
    with src.open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
    result.append({"case_id":case,"source_path":str(src),"source_sha256":digest,"source_pixels":im.size,"figure_file":dest.name,"panels":headers})
(O/"QUALITATIVE_SOURCE_PROVENANCE.json").write_text(json.dumps({"dataset":"FIVES (Jin et al. 2022), license see official distribution","seed":42,"fixed_cases":result,"description":"Unmodified image slices from frozen test outputs; only transposed into labeled 2x2 panels, no mask editing, no score-based selection."},indent=2),encoding="utf8")
print("QUAL_IMAGES_CREATED", [(a["case_id"],(O/a["figure_file"]).stat().st_size) for a in result])
