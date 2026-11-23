"""CETR-Net V2 topology-stage experimental runner.
Train/validation ONLY; accepts prepared /kaggle/working/cetr_v2 or local WSL WORK.
Pilot phase is NOT a final paper result; complete multi-seed independently initialized backbones and official test are a separate locked evaluation.
"""
import os, json, csv, time, random, hashlib, argparse, traceback
from pathlib import Path
import numpy as np, cv2, tensorflow as tf
from skimage.morphology import skeletonize
from cetr_core_ablation import (
    CETRNet, supervised_candidates, inference_candidates, training_loss, line_coverage, soft_dice
)
VARIANTS={
 "natural_full_symmetric":dict(cdes="natural",evidence="full",wr=.3,wd=.3),
 "break_only":dict(cdes="break",evidence="full",wr=.3,wd=.3),
 "false_only":dict(cdes="false",evidence="full",wr=.3,wd=.3),
 "dual_geometry":dict(cdes="dual",evidence="geometry",wr=.3,wd=.3),
 "dual_confidence":dict(cdes="dual",evidence="confidence",wr=.3,wd=.3),
 "dual_full_symmetric":dict(cdes="dual",evidence="full",wr=.3,wd=.3),
 "dual_full_asymmetric":dict(cdes="dual",evidence="full",wr=.45,wd=.75),
}
def get_rows(work,seed,part):
    p=work/"SPLITS_NEARDUP_SAFE_V2"/("seed_"+str(seed))/("fives_"+part+"_split.csv")
    with p.open(encoding="utf8",newline="") as f:return list(csv.DictReader(f))
def load(row,size=512):
    img=cv2.imread(row["image"],cv2.IMREAD_COLOR)
    mask=cv2.imread(row["mask"],cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:raise FileNotFoundError((row["image"],row["mask"]))
    img=cv2.cvtColor(cv2.resize(img,(size,size)),cv2.COLOR_BGR2RGB).astype("float32")/255.
    mask=(cv2.resize(mask,(size,size),interpolation=cv2.INTER_NEAREST)>127).astype("float32")
    return tf.convert_to_tensor(img[None]),tf.convert_to_tensor(mask[None,:,:,None])
def candidates_from_model(pred,truth):
    mask=(pred>=.5).astype("float32")
    cands=inference_candidates(mask,max_candidates=12)
    scored=[]
    for c in cands:
        gt=line_coverage(truth,c["a"],c["b"])
        if .35<gt<.7:continue
        c["label"]=float(gt>=.7) if c["kind"]=="restore" else float(gt<=.35)
        scored.append(c)
    return mask,scored
def candidates_for_inference(pred):
    """Inference proposals MUST be independent from ground-truth annotations.
    Only preliminary prediction may determine retained candidate paths.
    """
    mask=(pred>=.5).astype("float32")
    cands=inference_candidates(mask,max_candidates=12)
    return mask,cands

def candidates_from_cdes(pred,truth,rng,mode):
    mask=(pred>=.5).astype("float32")
    try:
        _,original,removed,inserted=supervised_candidates(truth,rng)
    except ValueError:
        return candidates_from_model(pred,truth)
    original=[c for c in original if 
         (mode=="dual" or (mode=="break" and c["kind"]=="restore") or 
          (mode=="false" and c["kind"]=="reject"))]
    if mode in ("break","dual"):mask[removed>0]=0.
    if mode in ("false","dual"):mask[inserted>0]=1.
    return mask,original
def scores(y,p):
    pred=np.asarray(p>.5)
    truth=np.asarray(y>.5)
    inter=np.logical_and(pred,truth).sum()
    dice=float((2*inter+1)/(pred.sum()+truth.sum()+1))
    iou=float((inter+1)/(np.logical_or(pred,truth).sum()+1))
    skpred=skeletonize(pred)
    sktruth=skeletonize(truth)
    a=float((skpred&truth).sum()+1e-5)/(float(skpred.sum())+1e-5)
    b=float((sktruth&pred).sum()+1e-5)/(float(sktruth.sum())+1e-5)
    cl=float((2*a*b+1e-5)/(a+b+1e-5))
    return {"dice":dice,"iou":iou,"cldice":cl}
def atomic(path,obj):
    temp=Path(str(path)+".tmp")
    temp.write_text(json.dumps(obj,indent=2),encoding="utf8")
    temp.replace(path)
def run_one(work,weight_path,seed,variant,epochs,train_n,val_n,size,report_root):
    cfg=VARIANTS[variant]
    train=get_rows(work,seed,"train")
    val=get_rows(work,seed,"validation")
    assert len(train)==508 and len(val)==90 and size==512
    if train_n>0:train=train[:min(train_n,len(train))]
    if val_n>0:val=val[:min(val_n,len(val))]
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    random.seed(seed);rng=np.random.default_rng(seed)
    net=CETRNet((size,size,3))
    net.backbone.load_weights(str(weight_path))
    net.backbone.trainable=False
    optim=tf.keras.optimizers.Adam(3e-4)
    out=report_root/("seed_"+str(seed))/variant
    out.mkdir(parents=True,exist_ok=True)
    history=[]
    now=time.time()
    init=load(train[0],size)
    p0,_=net.backbone(init[0],training=False)
    mask,cands=candidates_from_cdes(p0.numpy()[0,:,:,0],init[1].numpy()[0,:,:,0],rng,"dual")
    net.forward(init[0],cands,mask,training=False,evidence_mode=cfg["evidence"])
    epvar=tf.Variable(0,dtype=tf.int64,trainable=False)
    ckpt=tf.train.Checkpoint(shared=net.shared,restore_head=net.restore,
                              reject_head=net.reject,optimizer=optim,epoch=epvar)
    manager=tf.train.CheckpointManager(ckpt,str(out/"resume_checkpoint"),max_to_keep=2)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print("REFINER_RESUME",seed,variant,manager.latest_checkpoint,flush=True)
    best_path=out/"best_state.json"
    best=json.loads(best_path.read_text(encoding="utf8")) if best_path.exists() else {"best_dice":-1.,"best_epoch":0}
    for epoch in range(int(epvar.numpy()),epochs):
        rng=np.random.default_rng(seed*100000+epoch)
        indices=rng.permutation(len(train))
        losses=[];action_counts={"restore":0,"reject":0}
        for k in indices:
            row=train[int(k)]
            x,y=load(row,size)
            truth=y.numpy()[0,:,:,0]
            p0,_=net.backbone(x,training=False)
            pp=p0.numpy()[0,:,:,0]
            if cfg["cdes"]=="natural":
                mask,cands=candidates_from_model(pp,truth)
            else:
                mask,cands=candidates_from_cdes(pp,truth,rng,cfg["cdes"])
            if not cands:continue
            with tf.GradientTape() as tape:
                pred,_,qr,qd,act=net.forward(x,cands,mask,training=False,evidence_mode=cfg["evidence"])
                loss,components=training_loss(y,pred,qr,qd,cands,
                    top_weight=.025,restore_weight=cfg["wr"],reject_weight=cfg["wd"])
            train_vars=net.shared.trainable_variables+net.restore.trainable_variables+net.reject.trainable_variables
            grads=tape.gradient(loss,train_vars)
            valid=[(g,v) for g,v in zip(grads,train_vars) if g is not None]
            if not valid or not all(bool(tf.reduce_all(tf.math.is_finite(g))) for g,v in valid):
                raise RuntimeError("no valid finite head gradients")
            optim.apply_gradients(valid)
            losses.append(float(loss.numpy()))
            for c in cands: action_counts[c["kind"]]+=1
        observed=[];pair_rows=[]
        for row in val:
            x,y=load(row,size)
            truth=y.numpy()[0,:,:,0]
            p0,_=net.backbone(x,training=False)
            pp=p0.numpy()[0,:,:,0]
            mask,cs=candidates_for_inference(pp)
            pred,_,qr,qd,_=net.forward(x,cs,mask,training=False,evidence_mode=cfg["evidence"])
            real=pred.numpy()[0,:,:,0]
            met=scores(truth,real)
            base=scores(truth,pp)
            met.update({"id":row["id"],"base_dice":base["dice"],"base_cldice":base["cldice"],
                        "n_candidates":len(cs)})
            observed.append(met)
            # Ground truth is read only AFTER prediction for diagnostics, never proposal filtering.
            for j,c in enumerate(cs):
                gt_coverage=line_coverage(truth,c["a"],c["b"])
                if .35<gt_coverage<.7:
                    continue  # ambiguous edge has no binary evaluation label
                truth_action=(gt_coverage>=.7) if c["kind"]=="restore" else (gt_coverage<=.35)
                q=float(qr[j].numpy()) if c["kind"]=="restore" else float(qd[j].numpy())
                pair_rows.append({"sample":row["id"],"kind":c["kind"],
                                  "ground_truth_action_label":float(truth_action),
                                  "acceptance_probability":q,"accepted":q>=.5})
        if not observed:raise RuntimeError("validation has no records")
        report={"method":"CETR-Net stage2 preliminary controlled experiment",
                "status":"full_validation_only" if len(train)==508 and len(val)==90 else "validation_pilot_NOT_official_test",
                "seed":seed,"variant":variant,"config":cfg,"epoch":epoch+1,
                "train_count":len(train),"validation_count":len(val),
                "train_loss_mean":float(np.mean(losses)) if losses else None,
                "train_samples_with_candidates":len(losses),
                "train_action_candidate_counts":action_counts,
                "val_mean_dice":float(np.mean([r["dice"] for r in observed])),
                "val_mean_cldice":float(np.mean([r["cldice"] for r in observed])),
                "val_mean_iou":float(np.mean([r["iou"] for r in observed])),
                "val_backbone_mean_dice":float(np.mean([r["base_dice"] for r in observed])),
                "val_mean_delta_dice":float(np.mean([r["dice"]-r["base_dice"] for r in observed])),
                "num_val_candidates":len(pair_rows),"validation_candidate_protocol":"prediction_only_before_metrics",
                "seconds_elapsed":round(time.time()-now,1),
                "paired_scores":observed,
                "candidate_scores":pair_rows,
                 "limitations":["Backbone is frozen during refiner stage and each seed must use its independently-trained matching backbone",
                               "FIVES validation, NOT official held-out test",
                               "Head-only optimization, not end-to-end training",
                               "No independent MaskVSC baseline run",
                               "Candidate 2D annotations are not 3D topology"]}
        history.append(report)
        atomic(out/"progress.json",report)
        with (out/"epoch_history.jsonl").open("a",encoding="utf8") as f:
            f.write(json.dumps({"epoch":epoch+1,"dice":report["val_mean_dice"],
                       "cldice":report["val_mean_cldice"],"train_loss":report["train_loss_mean"]})+"\n")
        if report["val_mean_dice"]>best["best_dice"]:
            best={"best_dice":report["val_mean_dice"],"best_epoch":epoch+1}
            atomic(best_path,best)
            np.savez_compressed(out/"best_head_weights.npz",
              **{("shared_"+str(i)):w.numpy() for i,w in enumerate(net.shared.weights)},
              **{("restore_"+str(i)):w.numpy() for i,w in enumerate(net.restore.weights)},
              **{("reject_"+str(i)):w.numpy() for i,w in enumerate(net.reject.weights)})
        epvar.assign(epoch+1)
        manager.save(checkpoint_number=epoch+1)
        print("EPOCH_STAGE2",seed,variant,epoch+1,round(report["val_mean_dice"],5),
              round(report["val_mean_delta_dice"],5),"samples",len(losses),flush=True)
    np.savez_compressed(out/"head_weights.npz",
       **{("shared_"+str(i)):w.numpy() for i,w in enumerate(net.shared.weights)},
       **{("restore_"+str(i)):w.numpy() for i,w in enumerate(net.restore.weights)},
       **{("reject_"+str(i)):w.numpy() for i,w in enumerate(net.reject.weights)})
    atomic(out/"completed.json",{"variant":variant,"seed":seed,"completed_at":time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime()),"final":history[-1]})
    return history[-1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--work",default="/kaggle/working/cetr_v2")
    p.add_argument("--weights",required=True)
    p.add_argument("--mode",choices=("preflight","pilot","fullstage"),default="pilot")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--variant-list",type=str,default="",help="comma-separated predeclared variants")
    p.add_argument("--epochs",type=int,default=8)
    a=p.parse_args()
    w=Path(a.work)
    out=w/"CETR_V2_STAGE2_EXPERIMENTS"
    out.mkdir(exist_ok=True,parents=True)
    with (out/"RUN_STATUS.json").open("w") as f:json.dump({"mode":a.mode,"started":time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime())},f)
    configs={"preflight":{"seeds":[42],"variants":["dual_full_asymmetric"],"train_n":4,"val_n":2,"epochs":1},
             "pilot":{"seeds":[42,123,2025],"variants":list(VARIANTS),"train_n":24,"val_n":10,"epochs":1},
             "fullstage":{"seeds":[42,123,2025],"variants":list(VARIANTS),"train_n":508,"val_n":90,"epochs":20}}
    c=configs[a.mode]
    if a.mode=="fullstage":
        if a.seed not in (42,123,2025):raise ValueError("Unsupported FIVES V2 seed")
        if a.epochs<1 or a.epochs>40:raise ValueError("Unsupported epoch setting")
        c["seeds"]=[a.seed]
        c["epochs"]=a.epochs
        if a.variant_list:
            variants=a.variant_list.split(",")
            if len(set(variants))!=len(variants) or any(v not in VARIANTS for v in variants):
                raise ValueError("Invalid or duplicate ablation variants")
            c["variants"]=variants
    print("FULLSTAGE_PROTOCOL",json.dumps(c),flush=True)
    completed=[]
    for seed in c["seeds"]:
        for variant in c["variants"]:
            print("START_STAGE2",seed,variant,flush=True)
            try:
                report=run_one(w,Path(a.weights),seed,variant,c["epochs"],c["train_n"],c["val_n"],512,out)
                completed.append({"seed":seed,"variant":variant,
                                  "val_mean_dice":report["val_mean_dice"],
                                  "delta_dice":report["val_mean_delta_dice"],
                                  "train_count":report["train_count"],"validation_count":report["validation_count"]})
            except Exception as err:
                error={"seed":seed,"variant":variant,"error":repr(err),"traceback":traceback.format_exc()}
                atomic(out/"LAST_ERROR.json",error)
                print("STAGE2_FAILED",json.dumps(error),flush=True)
                raise
            atomic(out/"SUMMARY.json",{"mode":a.mode,"completed":completed,
                  "full_training_results":bool(a.mode=="fullstage"),
                 "official_test_scored":False,
                 "baseline_seed":int(a.seed) if a.mode=="fullstage" else "pretrained_seed42",
                 "all_runs_passed":len(completed)==len(c["seeds"])*len(c["variants"])})
    atomic(out/"RUN_STATUS.json",{"mode":a.mode,"status":"FINISHED","num_completed":len(completed)})
    print("CETR_V2_STAGE2_COMPLETE",len(completed),flush=True)
if __name__=="__main__":
    main()
