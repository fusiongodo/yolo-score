import torch
import config 

# ANCHORS = c.ANCHORS
# N = c.N
img_w, img_h = 1.0, 1.0


def decode_fn(t):
    N, A = t.shape[0], t.shape[2]
    tx, ty, tw, th = t[...,0], t[...,1], t[...,2], t[...,3]

    ii = torch.arange(N, device=t.device, dtype=t.dtype).view(N,1,1)
    jj = torch.arange(N, device=t.device, dtype=t.dtype).view(1,N,1)

    cx = (ii + tx) / N
    cy = (jj + ty) / N

    anch = torch.as_tensor(config.ANCHORS[:A], device=t.device, dtype=t.dtype)  # (A,2)
    aw = anch[:,0].view(1,1,A)
    ah = anch[:,1].view(1,1,A)

    w = aw * torch.exp(tw)
    h = ah * torch.exp(th)

    return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], -1)


def pred_to_iou(pred, tgt):
    # elementwise IoU (same anchor index) – anchors are identical anyway
    bp = decode_fn(pred)                 # (N,N,A,4)
    bt = decode_fn(tgt)                  # (N,N,A,4)
    return iou_xyxy(bp, bt).to(pred.dtype)   # (N,N,A)



def iou_xyxy(a, b, eps=1e-9):
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    x1 = torch.maximum(a[...,0], b[...,0])
    y1 = torch.maximum(a[...,1], b[...,1])
    x2 = torch.minimum(a[...,2], b[...,2])
    y2 = torch.minimum(a[...,3], b[...,3])

    inter = (x2-x1).clamp(0) * (y2-y1).clamp(0)
    area_a = (a[...,2]-a[...,0]).clamp(0) * (a[...,3]-a[...,1]).clamp(0)
    area_b = (b[...,2]-b[...,0]).clamp(0) * (b[...,3]-b[...,1]).clamp(0)
    union  = area_a + area_b - inter
    return (inter + eps) / (union + eps)

def debug_pred_to_iou():
    N,A,C = config.N, config.A, config.C
    t = torch.zeros((N,N,A,5+C))
    t[...,0:2] = 0.5                      # tx,ty
    bp = decode_fn(t)
    bt = decode_fn(t)

    print("max |bp-bt|:", (bp-bt).abs().max().item())

    m   = iou_xyxy(bp.unsqueeze(-2), bt.unsqueeze(-3))   # (N,N,A,A)
    diag= torch.diagonal(m, dim1=-2, dim2=-1)

    print("IoU diag mean/min/max:",
          diag.mean().item(), diag.min().item(), diag.max().item())

    bad = (diag < 0.999999).nonzero(as_tuple=False)
    if bad.numel():
        cx,cy,a = bad[0].tolist()
        print("bad idx:", cx,cy,a)
        print("bp:", bp[cx,cy,a])
        print("bt:", bt[cx,cy,a])
        print("m:",  m[cx,cy])

def unit_precision(pred, tgt, iou, conf_thr=0.5):

    conf   = pred[...,4] >= conf_thr
    pred_c = pred[...,5:].argmax(-1)
    tgt_c  = tgt[...,5:].argmax(-1)
    gt_obj = tgt[...,4] > 0

    tp_mask = conf & gt_obj & (pred_c == tgt_c)
    fp_mask = conf & (~gt_obj | (gt_obj & (pred_c != tgt_c)))

    # only count IoU where GT exists
    tp = (iou * tp_mask).sum()

    # count FP as *counts*, not IoU (or use iou*gt_obj==0 => 0 anyway)
    fp = fp_mask.sum().float()

    denom = tp + fp
    return (tp / denom).item() if denom > 0 else 0.0


#for testing only
def logit_to_target(tensor):
    pred = tensor.clone()
    
    pred[..., [0, 1, 4]] = torch.sigmoid(pred[..., [0, 1, 4]])
    argmax = torch.argmax(pred[..., 5:], dim=-1)

    one_hot = torch.zeros_like(pred[..., 5:])
    one_hot.scatter_(-1, argmax.unsqueeze(-1), 1.0)

    pred[..., 5:] = one_hot
    return pred

#Todo: introduce randomness?
def average_precision(model, eval_dataset, device, n_samples=100):
    precisions = []
    model = model.to(device)
    model.eval()
    for s in range(n_samples):
        img, tgt = eval_dataset[s]
        img = img.unsqueeze(0).unsqueeze(0).to(device) # 1, 1, H, W
        tgt = tgt.to(device)
        pred = model.forward(img).squeeze(0)
        pred = logit_to_target(pred)
        iou = pred_to_iou(pred, tgt)
        precisions.append(unit_precision(pred, tgt, iou))
    return sum(precisions) / len(precisions)





#######################################################################
#######################################################################
#######################################################################
###################### Debug unit_precision  ##########################
#######################################################################
#######################################################################
#######################################################################

import torch, config
from math import isclose

# ---- helpers -------------------------------------------------
def assert_shape(x, shape):
    assert x.shape == shape, f"shape {x.shape} != {shape}"

def near(x, y, tol=1e-5): return isclose(float(x), float(y), rel_tol=tol, abs_tol=tol)

def dbg_masks(pred, tgt, iou, conf_thr=0.5):
    conf = pred[...,4]
    pc   = pred[...,5:].argmax(-1)
    tc   = tgt[...,5:].argmax(-1)
    tp_m = (conf >= conf_thr) & (pc == tc)
    print("conf≥thr:", conf.ge(conf_thr).sum().item(),
          "pc==tc:", (pc==tc).sum().item(),
          "tp_m:", tp_m.sum().item(),
          "iou>0:", (iou>0).sum().item())

# ---- 0. primitive boxes --------------------------------------
def _tiny_iou_test():
    a = torch.tensor([0.,0.,1.,1.])
    b = torch.tensor([0.,0.,1.,1.])
    c = torch.tensor([0.,0.,0.5,0.5])
    assert near(iou_xyxy(a,b), 1.0)
    assert near(iou_xyxy(a,c), 0.25)

# ---- 1. decode_fn --------------------------------------------
def test_decode_identity():
    N, A = config.N, config.A
    t = torch.zeros((N,N,A,5+config.C))
    # set tx=ty=0.5, tw=th=0 → width = anchor_w, height = anchor_h, centered in cell
    t[...,0:2] = 0.5
    box = decode_fn(t)
    assert_shape(box, (N,N,A,4))
    # box coords must be within [0,1]
    assert (box>=0).all() and (box<=1).all()

# ---- 2. pred_to_iou ------------------------------------------
def test_pred_to_iou_perfect():
    N,A = config.N, config.A
    t = torch.zeros((N,N,A,5+config.C))
    t[...,0:2] = 0.5
    iou = pred_to_iou(t, t)
    assert_shape(iou, (N,N,A))
    assert near(iou.mean(), 1.0)

# ---- 3. dummyTensor logic ------------------------------------
def test_dummy_patterns(d):
    one  = d.dummyTensor("one")
    half = d.dummyTensor("half")
    assert_shape(one,  (config.N,config.N,config.A,5+config.C))
    assert one[...,4].sum() == config.N*config.N*config.A
    assert half[...,4].sum() == (config.N//2)*config.N*config.A

# ---- 4. unit_precision sanity --------------------------------
def test_unit_precision_simple(d):
    one  = d.dummyTensor("one")
    half = d.dummyTensor("half")
    iou  = pred_to_iou(one, half)
    p_oh = unit_precision(one, half, iou)    # pred=one, tgt=half
    p_ho = unit_precision(half, one, iou)    # pred=half, tgt=one
    p_oo = unit_precision(one,  one, iou)

    print("oh, ho, oo =", p_oh, p_ho, p_oo)
    dbg_masks(one, half, iou)
    dbg_masks(half, one, iou)

import dataset

def runTests(gt_df):
    _tiny_iou_test()
    test_decode_identity()
    test_pred_to_iou_perfect()
    d = dataset.CroppedDummyset(gt_df)
    test_dummy_patterns(d)
    test_unit_precision_simple(d)
