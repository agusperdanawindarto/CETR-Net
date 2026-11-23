"""CETR-Net experimental prototype, NOT a completed peer-reviewed result.
Real fundus RGB is required at inference. Annotation-derived perturbations are TRAINING-ONLY.
No operations write into historical EAG-UNet folders.
"""
import math
import cv2
import numpy as np
import tensorflow as tf
from skimage.morphology import skeletonize


def conv_block(x, ch):
    for _ in range(2):
        x = tf.keras.layers.Conv2D(ch, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
    return x


def make_backbone(shape=(256, 256, 3)):
    image = tf.keras.Input(shape=shape, name="RGB_fundus")
    x, skips = image, []
    for c in (24, 48, 96, 192):
        x = conv_block(x, c)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D()(x)
    x = conv_block(x, 384)
    for c, skip in zip((192, 96, 48, 24), reversed(skips)):
        x = tf.keras.layers.UpSampling2D(interpolation="bilinear")(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = conv_block(x, c)
    p = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="P0")(x)
    return tf.keras.Model(image, [p, x], name="cetr_initial_encoder_decoder")


def line_points(a, b, count=16):
    yy = np.rint(np.linspace(a[0], b[0], count)).astype(int)
    xx = np.rint(np.linspace(a[1], b[1], count)).astype(int)
    return np.stack([yy, xx], axis=-1)


def line_coverage(binary, a, b):
    pts = line_points(a, b, 25)
    return float(np.mean(binary[pts[:, 0], pts[:, 1]]))


def choose_pair(coords, Y, kind, rng, min_d=9, max_d=27):
    if len(coords) == 0:
        raise ValueError("No skeleton coordinates; cannot construct topology examples")
    for _ in range(1800):
        a = coords[int(rng.integers(len(coords)))]
        b = coords[int(rng.integers(len(coords)))]
        dist = float(np.linalg.norm(a-b))
        if dist < min_d or dist > max_d: continue
        ratio = line_coverage(Y, a, b)
        if kind == "true" and ratio >= .78: return tuple(a), tuple(b)
        if kind == "false" and ratio <= .30: return tuple(a), tuple(b)
    raise ValueError("No eligible "+kind+" candidate; skip rather than fabricate training labels")


def candidate(a, b, kind, label, shape):
    h, w = shape
    path = np.clip(line_points(a, b, 20), [0, 0], [h-1, w-1]).astype("int32")
    ker = np.zeros((h, w), "float32")
    cv2.line(ker, (int(a[1]), int(a[0])), (int(b[1]), int(b[0])), 1, 3, cv2.LINE_8)
    ker = cv2.GaussianBlur(ker, (5,5), .7)
    ker = np.clip(ker, 0, 1)
    return {"path":path, "kernel":ker, "kind":kind, "label":float(label), "a":a, "b":b}


def supervised_candidates(y, rng):
    """Training ONLY: construct four controls with ground-truth labels.
    Missing true arc and added background bridge are supervised distinctly.
    """
    y = np.asarray(y>0.5, dtype=np.uint8)
    h, w = y.shape
    sk = skeletonize(y>0).astype("uint8")
    coords = np.argwhere(sk>0)
    ar,br = choose_pair(coords,y,"true",rng)
    af,bf = choose_pair(coords,y,"false",rng)
    # Other two are negative action controls: nothing is corrupted along them.
    candidates = [
        candidate(ar,br,"restore",1,(h,w)),
        candidate(af,bf,"reject",1,(h,w)),
        candidate(af,bf,"restore",0,(h,w)),
        candidate(ar,br,"reject",0,(h,w)),
    ]
    removed = ((candidates[0]["kernel"]>.5)&(y==1)).astype("uint8")
    inserted = ((candidates[1]["kernel"]>.5)&(y==0)).astype("uint8")
    if (removed & (1-y)).any() or (inserted & y).any():
        raise AssertionError("counterfactual masks violate Y constraints")
    corrupted = (y * (1-removed)) | inserted
    return corrupted.astype("float32"), candidates, removed, inserted


def inference_candidates(preliminary_binary, max_candidates=12, radius=25):
    """Inference depends only on P0, no ground-truth or other labels."""
    mask = np.asarray(preliminary_binary>0.5, dtype="uint8")
    h,w = mask.shape
    sk = skeletonize(mask>0).astype("uint8")
    n = cv2.filter2D(sk,-1,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT)-sk
    endpoints = np.argwhere((sk>0)&(n==1))
    pts = np.argwhere(sk>0)
    proposals=[]; seen=set()
    for i,a in enumerate(endpoints):
        for b in endpoints[i+1:]:
            dis=np.linalg.norm(a-b)
            if 7<dis<radius:
                k=tuple(sorted((tuple(a),tuple(b))))
                if k not in seen:
                    proposals.append(candidate(tuple(a),tuple(b),"restore",0,(h,w)))
                    seen.add(k)
            if len(proposals)>=max_candidates//2:break
        if len(proposals)>=max_candidates//2:break
    # Existing centerline segments are possible false-link candidates.
    for i in np.linspace(0,max(0,len(pts)-1),min(len(pts),max_candidates*2),dtype=int):
        if len(proposals)>=max_candidates:break
        a=pts[i]
        near=pts[np.sum((pts-a)**2,axis=1)<400]
        if len(near)<2: continue
        b=near[-1]
        if 7<np.linalg.norm(a-b)<radius:
            proposals.append(candidate(tuple(a),tuple(b),"reject",0,(h,w)))
    return proposals[:max_candidates]


class CETRNet(tf.keras.Model):
    def __init__(self, shape=(256,256,3)):
        super().__init__(name="CETRNet")
        self.backbone=make_backbone(shape)
        self.shared=tf.keras.layers.Dense(64,activation="relu")
        self.restore=tf.keras.layers.Dense(1,activation="sigmoid",name="restore_head")
        self.reject=tf.keras.layers.Dense(1,activation="sigmoid",name="reject_head")

    def forward(self, image, candidate_list, proposal_mask, training=False):
        """RGB image [1,H,W,3]; proposal_mask [H,W], graph from predictions or training perturbations."""
        p0, feat=self.backbone(image,training=training)
        H,W=int(p0.shape[1]),int(p0.shape[2])
        if not candidate_list:
            return p0,p0,tf.zeros((0,),tf.float32),tf.zeros((0,),tf.float32),tf.constant(0.)
        image0=image[0]; p00=p0[0]; f0=feat[0]
        desc=[]
        for e in candidate_list:
            idx=tf.constant(e["path"],dtype=tf.int32)
            rgb=tf.reduce_mean(tf.gather_nd(image0,idx),axis=0)
            featvec=tf.reduce_mean(tf.gather_nd(f0,idx),axis=0)
            prob=tf.gather_nd(p00,idx)[:,0]
            ent=-prob*tf.math.log(prob+1e-6)-(1-prob)*tf.math.log(1-prob+1e-6)
            dy=(e["b"][0]-e["a"][0])/H
            dx=(e["b"][1]-e["a"][1])/W
            geom=tf.constant([dy,dx,math.sqrt(dy*dy+dx*dx)],tf.float32)
            desc.append(tf.concat([rgb,featvec,tf.stack([tf.reduce_mean(prob),tf.math.reduce_std(prob),tf.reduce_mean(ent)]),geom],axis=0))
        z=self.shared(tf.stack(desc))
        qr=tf.reshape(self.restore(z),(-1,))
        qd=tf.reshape(self.reject(z),(-1,))
        m=tf.constant(proposal_mask,dtype=tf.float32)
        ker=tf.constant(np.stack([e["kernel"] for e in candidate_list]),tf.float32)
        rt=tf.constant(np.array([float(e["kind"]=="restore") for e in candidate_list]),tf.float32)
        dt=1.-rt
        add=ker*(1.-m)[None,:,:]*rt[:,None,None]*qr[:,None,None]
        delete=ker*m[None,:,:]*dt[:,None,None]*qd[:,None,None]
        R=1.-tf.reduce_prod(1.-add,axis=0)
        D=1.-tf.reduce_prod(1.-delete,axis=0)
        if bool(tf.reduce_any(R*D>1e-6)):raise AssertionError("opposing pixel edits")
        P=tf.clip_by_value(p00[:,:,0]+(1.-p00[:,:,0])*R-p00[:,:,0]*D,1e-5,1-1e-5)
        return tf.expand_dims(P[None,:,:],-1),p0,qr,qd,tf.reduce_mean(R+D)


def soft_dice(y,p):
    return (2*tf.reduce_sum(y*p)+1.)/(tf.reduce_sum(y)+tf.reduce_sum(p)+1.)


def _soft_erode(x):
    return tf.minimum(-tf.nn.max_pool2d(-x,ksize=(3,1),strides=1,padding="SAME"),
                      -tf.nn.max_pool2d(-x,ksize=(1,3),strides=1,padding="SAME"))


def _soft_open(x):
    return tf.nn.max_pool2d(_soft_erode(x),ksize=3,strides=1,padding="SAME")


def _soft_skeleton(x,iters=8):
    sk=tf.nn.relu(x-_soft_open(x))
    for _ in range(iters):
        x=_soft_erode(x)
        delta=tf.nn.relu(x-_soft_open(x))
        sk=sk+tf.nn.relu(delta-sk*delta)
    return sk


def soft_cldice(y,p):
    """Established soft-clDice formulation, NOT a new CETR-Net loss."""
    sy=_soft_skeleton(y)
    sp=_soft_skeleton(p)
    prec=(tf.reduce_sum(sp*y)+1e-5)/(tf.reduce_sum(sp)+1e-5)
    sens=(tf.reduce_sum(sy*p)+1e-5)/(tf.reduce_sum(sy)+1e-5)
    return (2*prec*sens+1e-5)/(prec+sens+1e-5)


def training_loss(y,p,qr,qd,cands,top_weight=0.1):
    seg=tf.reduce_mean(tf.keras.losses.binary_crossentropy(y,p))+(1-soft_dice(y,p))
    restore_idx=[i for i,e in enumerate(cands) if e["kind"]=="restore"]
    reject_idx=[i for i,e in enumerate(cands) if e["kind"]=="reject"]
    def action_loss(scores,indices):
        if not indices:return tf.constant(0.,tf.float32)
        pred=tf.gather(scores,indices)
        truth=tf.constant([cands[i]["label"] for i in indices],tf.float32)
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(truth,pred))
    lr=action_loss(qr,restore_idx)
    ld=action_loss(qd,reject_idx)
    lt=1-soft_cldice(y,p) if top_weight>0 else tf.constant(0.,tf.float32)
    return seg+.3*lr+.3*ld+top_weight*lt,{"seg":seg,"restore":lr,"reject":ld,"topology":lt}
