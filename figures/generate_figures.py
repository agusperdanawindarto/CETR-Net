"""Generate camera-ready analytical charts from frozen FIVES V2 official results."""
from pathlib import Path
import json,csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
R=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent
a=json.loads((R/"results/official_fives_v2_final_report.json").read_text(encoding="utf8"))
t={v["method"]:v for v in csv.DictReader((R/"results/official_fives_v2_cross_seed_table.csv").open(encoding="utf8"))}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9,"xtick.labelsize":7,"ytick.labelsize":8})
def save(fig,name):
    fig.savefig(O/(name+".png"),dpi=600,bbox_inches="tight",facecolor="white")
    fig.savefig(O/(name+".svg"),bbox_inches="tight",facecolor="white")
    plt.close(fig)
# Fig 5
fig,ax=plt.subplots(figsize=(3.44,1.9))
ax.set_xlim(0,10);ax.set_ylim(0,4.4);ax.axis("off")
boxes=[(.15,2.65,"V2 development","508 train | 90 val"),
(3.65,2.65,"Backbone x3","45 epochs/seed"),
(7.15,2.65,"CDES + routing","21 x 8 epochs"),
(3.65,.50,"Freeze on val","weights; threshold 0.5"),
(7.15,.50,"Official test","200 records | 197 groups")]
for x,y,title,sub in boxes:
    ax.add_patch(FancyBboxPatch((x,y),2.7,1.05,boxstyle="round,pad=0.13",
     facecolor="#e4f1f4",edgecolor="#267286",lw=1))
    ax.text(x+1.35,y+.71,title,ha="center",va="center",fontweight="bold",fontsize=7.6)
    ax.text(x+1.35,y+.3,sub,ha="center",va="center",fontsize=6.5)
for start,end in [((2.94,3.18),(3.56,3.18)),((6.43,3.18),(7.05,3.18)),((6.43,1.0),(7.05,1.0))]:
    ax.annotate("",end,start,arrowprops=dict(arrowstyle="->",color="#475569",lw=1.1))
ax.annotate("",(5,1.65),(8.45,2.45),arrowprops=dict(arrowstyle="->",color="#475569",lw=1.0,connectionstyle="arc3,rad=.1"))
ax.text(5,4.1,"Prediction-only inference; no annotations supplied to proposals",ha="center",fontsize=7,color="#334155")
fig.tight_layout(pad=.25);save(fig,"Fig5_experimental_protocol_600dpi")
# Fig 6
fig,ax=plt.subplots(figsize=(3.44,2.05))
for i,seed in enumerate((42,123,2025)):
    y=2-i;s=a["per_seed"][str(seed)]
    lo,hi=s["primary_delta_ci_95"]
    ax.plot([lo,hi],[y,y],color="#a11f29",lw=2.2)
    ax.scatter(s["primary_delta_dice"],y,s=25,color="#7e121e",zorder=4)
ax.axvline(0,ls="--",color="#334155")
ax.set_yticks((2,1,0),("Seed 42","Seed 123","Seed 2025"))
ax.set_xlim(-.0033,.00035);ax.set_xticks((-.003,-.002,-.001,0))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f"{x:+.3f}"))
ax.set_title("Official FIVES V2 | paired primary difference")
ax.set_xlabel("Delta Dice (CETR-Net minus backbone)")
ax.grid(axis="x",alpha=.2);ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(pad=.7);save(fig,"Fig6_official_primary_delta_600dpi")
# Fig 7
pairs=[("Natural/no counterfactual","natural_full_symmetric"),
("Break-only","break_only"),("False-only","false_only"),
("Dual symmetric","dual_full_symmetric"),("Geometry-only","dual_geometry"),
("Confidence-only","dual_confidence"),("Proposed dual/asymmetric","dual_full_asymmetric")]
baseline=float(t["independent_backbone"]["mean_dice"])
diff=[float(t[k]["mean_dice"])-baseline for label,k in pairs]
fig,ax=plt.subplots(figsize=(3.44,2.8))
ax.barh(range(7),diff,color=["#ba3943" if v<-.004 else "#417c99" for v in diff],height=.64)
ax.set_yticks(range(7),[v[0] for v in pairs]);ax.invert_yaxis()
ax.axvline(0,color="#0f172a",lw=.9);ax.set_xlim(-.006,.0002)
ax.set_xlabel("Delta mean Dice versus backbone (3 seeds)")
ax.set_title("Official-test ablation contrasts")
ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f"{x:+.3f}"))
ax.tick_params(axis="y",labelsize=6.7);ax.grid(axis="x",alpha=.18)
ax.spines[["top","right"]].set_visible(False)
fig.subplots_adjust(left=.48,right=.99,top=.86,bottom=.16)
save(fig,"Fig7_ablation_dice_difference_600dpi")
print("FIGURES_OK",len(list(O.glob("*.png"))))
