"""Checkpoint-resumable FIVES backbone training. Validation-only selection; NO official test access."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
import argparse,csv,hashlib,json,time,random
from pathlib import Path
import cv2,numpy as np,tensorflow as tf
from cetr_core import CETRNet,soft_dice

BASE=Path("/mnt/d/RISETKOLABORASI/Artikel Siap Kirim/1 NEXT PUBLICATION/4/EXPERIMEN_TAMBAHAN/IEEE_ACCESS_REVISION_EXPERIMENTS")
WORK=Path("/mnt/d/RISETKOLABORASI/CETR-Net/EXPERIMEN")
def get_rows(seed,part):
    src=WORK/"SPLITS_NEARDUP_SAFE_V2"/f"seed_{seed}"/f"fives_{part}_split.csv"
    with src.open(encoding="utf8",newline="") as f:return list(csv.DictReader(f)),hashlib.sha256(src.read_bytes()).hexdigest()
def load(row,size):
    img=cv2.imread(row["image"],cv2.IMREAD_COLOR)
    mask=cv2.imread(row["mask"],cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:raise FileNotFoundError(row["image"])
    img=cv2.cvtColor(cv2.resize(img,(size,size)),cv2.COLOR_BGR2RGB).astype("float32")/255.
    mask=(cv2.resize(mask,(size,size),interpolation=cv2.INTER_NEAREST)>127).astype("float32")
    return tf.convert_to_tensor(img[None]),tf.convert_to_tensor(mask[None,:,:,None])
def augment_train(x,y,rng):
    """Geometric flips share image/mask transforms; intensity changes image only."""
    xa=x.numpy().copy()
    ya=y.numpy().copy()
    for i in range(len(xa)):
        if rng.random()<.5:
            xa[i]=np.flip(xa[i],axis=1);ya[i]=np.flip(ya[i],axis=1)
        if rng.random()<.5:
            xa[i]=np.flip(xa[i],axis=0);ya[i]=np.flip(ya[i],axis=0)
        contrast=float(rng.uniform(.90,1.10))
        brightness=float(rng.uniform(-.05,.05))
        xa[i]=np.clip((xa[i]-.5)*contrast+.5+brightness,0.,1.)
    return tf.convert_to_tensor(xa),tf.convert_to_tensor(ya)
def atom_json(path,obj):
    tmp=Path(str(path)+".tmp")
    tmp.write_text(json.dumps(obj,indent=2),encoding="utf8")
    tmp.replace(path)
def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--seed",type=int,choices=[42,123,2025],required=True)
    pa.add_argument("--epochs",type=int,default=120)
    pa.add_argument("--size",type=int,default=512)
    pa.add_argument("--max-images",type=int,default=0,help="0 means full train/val, otherwise development smoke")
    pa.add_argument("--patience",type=int,default=15)
    a=pa.parse_args()
    tf.random.set_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    gpus=tf.config.list_physical_devices("GPU")
    if not gpus:raise RuntimeError("GPU is required for this experiment; refuse accidental CPU-only long run")
    for g in gpus:tf.config.experimental.set_memory_growth(g,True)
    train,htrain=get_rows(a.seed,"train");val,hval=get_rows(a.seed,"validation")
    if a.max_images:
        train=train[:a.max_images];val=val[:min(8,len(val))]
    out=WORK/"runs"/f"seed_{a.seed}"/("warmup_V2_FULL" if not a.max_images else f"warmup_V2_SMOKE_{a.max_images}")
    out.mkdir(parents=True,exist_ok=True)
    config={"seed":a.seed,"epochs":a.epochs,"size":a.size,"train_images":len(train),"validation_images":len(val),"train_split_sha256":htrain,"validation_split_sha256":hval,"optimizer":"Adam","learning_rate":1e-4,"batch_size":4,"augmentation":"train-only horizontal-vertical flips and RGB contrast 0.9-1.1 brightness +-0.05","loss":"BCE+softDice","selection":"validation Dice","official_test_access":False}
    config_path=out/"config.json"
    if config_path.exists():
        old=json.loads(config_path.read_text(encoding="utf8"))
        for key in ("seed","size","train_images","validation_images","train_split_sha256","validation_split_sha256","batch_size","augmentation"):
            if old[key]!=config[key]:raise RuntimeError("Incompatible resume configuration "+key)
    else:atom_json(config_path,config)
    net=CETRNet(shape=(a.size,a.size,3))
    opt=tf.keras.optimizers.Adam(learning_rate=1e-4)
    epoch_var=tf.Variable(0,dtype=tf.int64,trainable=False)
    sample_var=tf.Variable(0,dtype=tf.int64,trainable=False)
    step_var=tf.Variable(0,dtype=tf.int64,trainable=False)
    ckpt=tf.train.Checkpoint(backbone=net.backbone,optimizer=opt,epoch=epoch_var,sample=sample_var,step=step_var)
    mgr=tf.train.CheckpointManager(ckpt,str(out/"checkpoint_resume"),max_to_keep=3)
    if mgr.latest_checkpoint:
        ckpt.restore(mgr.latest_checkpoint).expect_partial()
        print("RESUME",mgr.latest_checkpoint,"epoch",int(epoch_var.numpy()),"sample",int(sample_var.numpy()),flush=True)
    else: print("NEW_RUN",str(out),flush=True)
    best_path=out/"best_state.json"
    state=json.loads(best_path.read_text(encoding="utf8")) if best_path.exists() else {"best_dice":-1.,"best_epoch":-1,"bad_epochs":0}
    start_epoch=int(epoch_var.numpy())
    for epoch in range(start_epoch,a.epochs):
        rng=np.random.default_rng(a.seed+epoch)
        order=rng.permutation(len(train))
        epoch_losses=[];t0=time.time()
        first=int(sample_var.numpy()) if epoch==start_epoch else 0
        if first%4:raise RuntimeError("Unsafe checkpoint offset: must land on batch boundary")
        for k in range(first,len(train),4):
            indices=order[k:k+4]
            xy=[load(train[i],a.size) for i in indices]
            x=tf.concat([pair[0] for pair in xy],axis=0)
            y=tf.concat([pair[1] for pair in xy],axis=0)
            # Augmentation keyed to epoch/batch; checkpoint resumes deterministically.
            aug_rng=np.random.default_rng(a.seed*1000000+epoch*10000+k)
            x,y=augment_train(x,y,aug_rng)
            with tf.GradientTape() as tape:
                p,_=net.backbone(x,training=True)
                loss=tf.reduce_mean(tf.keras.losses.binary_crossentropy(y,p))+1-soft_dice(y,p)
            grad=tape.gradient(loss,net.backbone.trainable_variables)
            if not all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in grad):
                raise RuntimeError("Invalid gradient at "+str((epoch,k)))
            opt.apply_gradients(zip(grad,net.backbone.trainable_variables))
            epoch_losses.append(float(loss.numpy()))
            completed=min(k+4,len(train))
            sample_var.assign(completed);step_var.assign_add(1)
            if completed%32==0:
                mgr.save(checkpoint_number=int(step_var.numpy()))
                print("PROGRESS epoch",epoch+1,"sample",completed,"of",len(train),"loss",round(float(np.mean(epoch_losses)),5),flush=True)
        dices=[]
        for row in val:
            x,y=load(row,a.size)
            p,_=net.backbone(x,training=False)
            dices.append(float(soft_dice(y,p).numpy()))
        val_dice=float(np.mean(dices))
        is_best=val_dice>state["best_dice"]
        if is_best:
            net.backbone.save_weights(str(out/"best_backbone.weights.h5"))
            state.update({"best_dice":val_dice,"best_epoch":epoch+1,"bad_epochs":0})
        else:state["bad_epochs"]+=1
        atom_json(best_path,state)
        epoch_var.assign(epoch+1);sample_var.assign(0)
        mgr.save(checkpoint_number=int(step_var.numpy()))
        entry={"epoch":epoch+1,"train_loss_mean":float(np.mean(epoch_losses)) if epoch_losses else None,"val_soft_dice":val_dice,"seconds":round(time.time()-t0,1),"best_so_far":bool(is_best),"samples":len(train)}
        with (out/"epoch_history.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(entry)+"\n")
        print("EPOCH_COMPLETED",json.dumps(entry),flush=True)
        if state["bad_epochs"]>=a.patience:
            print("EARLY_STOP",json.dumps(state),flush=True)
            break
    print("BACKBONE_TRAINING_EXIT",str(out),flush=True)

if __name__=="__main__":
    if (Path(__file__).resolve().parent/"LAPTOP_GPU_DISABLED.flag").exists():
        raise SystemExit("LAPTOP_GPU_DISABLED: CETR-Net training must run on Kaggle only")
    main()
