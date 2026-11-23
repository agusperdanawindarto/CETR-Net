"""CETR-Net stage-2 topology training. Stage 1 warmup must be complete.
Uses original FIVES TRAIN labels only in training candidate supervision. Validation generates candidates from P0 only.
Never accesses official FIVES test. Checkpoints saved separately from historical baseline.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
import argparse,json,time,random,hashlib
from pathlib import Path
import numpy as np,tensorflow as tf
from cetr_core import CETRNet, supervised_candidates, inference_candidates, training_loss, soft_dice, soft_cldice, line_coverage
from train_backbone import WORK,get_rows,load,atom_json

def model_candidates(mask,y):
    cands=inference_candidates(mask,max_candidates=12)
    keep=[]
    for e in cands:
        ratio=line_coverage(y,e["a"],e["b"])
        if .3<ratio<.7:continue
        e["label"]=float(ratio>=.7) if e["kind"]=="restore" else float(ratio<=.3)
        keep.append(e)
    return keep

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--seed",type=int,choices=(42,123,2025),required=True)
    pa.add_argument("--epochs",type=int,default=120)
    pa.add_argument("--size",type=int,default=512)
    pa.add_argument("--max-images",type=int,default=0)
    pa.add_argument("--patience",type=int,default=15)
    a=pa.parse_args()
    tf.random.set_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    gpu=tf.config.list_physical_devices("GPU")
    if not gpu:raise RuntimeError("GPU required")
    for g in gpu:tf.config.experimental.set_memory_growth(g,True)
    train,htrain=get_rows(a.seed,"train");val,hval=get_rows(a.seed,"validation")
    if a.max_images:train=train[:a.max_images];val=val[:min(8,len(val))]
    warm=WORK/"runs"/f"seed_{a.seed}"/("warmup_V2_FULL" if not a.max_images else f"warmup_V2_SMOKE_{a.max_images}")
    best_warm=warm/"best_backbone.weights.h5"
    if not best_warm.exists():raise FileNotFoundError("No warmup weights: "+str(best_warm))
    out=WORK/"runs"/f"seed_{a.seed}"/("refiner_V2_FULL" if not a.max_images else f"refiner_V2_SMOKE_{a.max_images}")
    out.mkdir(parents=True,exist_ok=True)
    conf={"seed":a.seed,"size":a.size,"train_images":len(train),"validation_images":len(val),"train_split_sha256":htrain,"validation_split_sha256":hval,"loss":"BCE+Dice + 0.3 Restore BCE + 0.3 Reject BCE + 0.1 softclDice","training_proposals":"50% annotation-corrupted, 50% model-derived where eligible","val_proposals":"predictions only","official_test_access":False}
    cfg=out/"config.json"
    if cfg.exists():
        old=json.loads(cfg.read_text())
        for k in ("seed","size","train_images","validation_images","train_split_sha256","validation_split_sha256"):
            if old[k]!=conf[k]:raise RuntimeError("Refuse incompatible resume: "+k)
    else:atom_json(cfg,conf)
    net=CETRNet((a.size,a.size,3))
    # Build heads on a real TRAIN image to make checkpoint variable restoration deterministic.
    x0,y0=load(train[0],a.size)
    rng0=np.random.default_rng(a.seed)
    m0,c0,_,_=supervised_candidates(y0.numpy()[0,:,:,0],rng0)
    net.forward(x0,c0,m0,training=False)
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
    ep=tf.Variable(0,dtype=tf.int64,trainable=False)
    idx=tf.Variable(0,dtype=tf.int64,trainable=False)
    step=tf.Variable(0,dtype=tf.int64,trainable=False)
    ck=tf.train.Checkpoint(net=net,optimizer=optimizer,epoch=ep,sample=idx,step=step)
    manager=tf.train.CheckpointManager(ck,str(out/"checkpoint_resume"),max_to_keep=3)
    if manager.latest_checkpoint:
        ck.restore(manager.latest_checkpoint).expect_partial()
        print("RESUME_REFINER",manager.latest_checkpoint,"epoch",int(ep.numpy()),"index",int(idx.numpy()),flush=True)
    else:
        net.backbone.load_weights(str(best_warm))
        print("INITIALIZED_FROM_FIVES_WARMUP",best_warm,flush=True)
    statepath=out/"best_state.json"
    state=json.loads(statepath.read_text()) if statepath.exists() else {"best_dice":-1.,"best_cldice":-1.,"best_epoch":-1,"bad_epochs":0}
    start=int(ep.numpy())
    for epoch in range(start,a.epochs):
        rng=np.random.default_rng(a.seed*1000+epoch)
        order=rng.permutation(len(train))
        losses=[]
        t0=time.time()
        first=int(idx.numpy()) if epoch==start else 0
        for k in range(first,len(train)):
            row=train[order[k]]
            x,y=load(row,a.size)
            truth=y.numpy()[0,:,:,0]
            use_prediction=bool(rng.random()<.5)
            if use_prediction:
                p_pre,_=net.backbone(x,training=False)
                mask=(p_pre.numpy()[0,:,:,0]>.5).astype("float32")
                cands=model_candidates(mask,truth)
            else:
                cands=[]
            if not cands:
                try:mask,cands,_,_=supervised_candidates(truth,rng)
                except ValueError:
                    idx.assign(k+1);continue
            with tf.GradientTape() as tape:
                pred,_,qr,qd,_=net.forward(x,cands,mask,training=True)
                loss,parts=training_loss(y,pred,qr,qd,cands,top_weight=.1)
            grads=tape.gradient(loss,net.trainable_variables)
            safe=[(g,v) for g,v in zip(grads,net.trainable_variables) if g is not None]
            if not safe or any(not bool(tf.reduce_all(tf.math.is_finite(g))) for g,_ in safe):raise RuntimeError("Invalid gradient")
            optimizer.apply_gradients(safe)
            losses.append(float(loss.numpy()))
            idx.assign(k+1);step.assign_add(1)
            if (k+1)%32==0:
                manager.save(checkpoint_number=int(step.numpy()))
                print("REFINER_PROGRESS",epoch+1,k+1,"/",len(train),"loss",round(float(np.mean(losses)),5),flush=True)
        scores=[];cls=[]
        for row in val:
            x,y=load(row,a.size)
            p0,_=net.backbone(x,training=False)
            m=(p0.numpy()[0,:,:,0]>.5).astype("float32")
            cs=inference_candidates(m)
            p,_,_,_,_=net.forward(x,cs,m,training=False)
            scores.append(float(soft_dice(y,p).numpy()))
            cls.append(float(soft_cldice(y,p).numpy()))
        dice=float(np.mean(scores));cldice=float(np.mean(cls))
        better=(dice>state["best_dice"]) and (state["best_cldice"]<0 or cldice>=state["best_cldice"]-.01)
        if better:
            ck.write(str(out/"best_model_checkpoint"))
            state.update({"best_dice":dice,"best_cldice":cldice,"best_epoch":epoch+1,"bad_epochs":0})
        else:state["bad_epochs"]+=1
        atom_json(statepath,state)
        ep.assign(epoch+1);idx.assign(0)
        manager.save(checkpoint_number=int(step.numpy()))
        rec={"epoch":epoch+1,"train_loss":float(np.mean(losses)) if losses else None,"val_dice":dice,"val_cldice":cldice,"seconds":round(time.time()-t0,1),"best":bool(better),"skipped":len(train)-len(losses)}
        with (out/"epoch_history.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(rec)+"\n")
        print("REFINER_EPOCH_COMPLETED",json.dumps(rec),flush=True)
        if state["bad_epochs"]>=a.patience:
            print("EARLY_STOP",json.dumps(state),flush=True)
            break
    print("REFINER_TRAINING_EXIT",str(out),flush=True)

if __name__=="__main__":
    if (Path(__file__).resolve().parent/"LAPTOP_GPU_DISABLED.flag").exists():
        raise SystemExit("LAPTOP_GPU_DISABLED: CETR-Net training must run on Kaggle only")
    main()
