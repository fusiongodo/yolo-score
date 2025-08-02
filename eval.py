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




#for testing only
def logit_to_target(tensor):
    pred = tensor.clone()
    
    pred[..., [0, 1, 4]] = torch.sigmoid(pred[..., [0, 1, 4]])
    argmax = torch.argmax(pred[..., 5:], dim=-1)

    one_hot = torch.zeros_like(pred[..., 5:])
    one_hot.scatter_(-1, argmax.unsqueeze(-1), 1.0)

    pred[..., 5:] = one_hot
    return pred

def prec_rec_cls(tp_mask, fp_mask, fn_mask):
    prec_cls, rec_cls = dict(), dict()
    for i in range(136):
        tp_mask_cls = tp_mask[..., 5] == i
        fp_mask_cls = fp_mask[..., 5] == i
        fn_mask_cls = fn_mask[..., 5] == i

        tp_cls = tp_mask_cls.sum().float().item()
        fp_cls = fp_mask_cls.sum().float().item()
        fn_cls = fn_mask_cls.sum().float().item()

        
        if tp_cls + fp_cls > 0.0:
            precision = tp_cls / (tp_cls + fp_cls)
        else:
            precision = 0.0
        if tp_cls + fn_cls > 0.0:
            recall = tp_cls / (tp_cls + fn_cls)
        else:
            recall = 0.0

        prec_cls[i] = precision
        rec_cls[i] = recall
    return prec_cls, rec_cls

def calc_masks(pred, tgt, iou, conf_thr):
        pred_conf   = pred[...,4] >= conf_thr
        tgt_conf = tgt[...,4] > 0.2 # just any value between 0 and 1 will do
        pred_cls = pred[...,5:].argmax(-1)
        tgt_cls  = tgt[...,5:].argmax(-1)

        tp_mask = pred_conf & tgt_conf & (pred_cls == tgt_cls) & (iou >= conf_thr)
        fp_mask = pred_conf & (
            (~tgt_conf) | 
            (tgt_conf & (pred_cls != tgt_cls)) | 
            (iou < conf_thr )
            ) 
        fn_mask = tgt_conf &(
            (~pred_conf) |
            (pred_cls != tgt_cls) |
            (iou < conf_thr)
        )
        return tp_mask, fp_mask, fn_mask, tgt_conf

def unit_precision_recall(tp_mask, fp_mask, fn_mask):
    tp = tp_mask.sum().float().item()
    fp = fp_mask.sum().float().item()
    fn = fn_mask.sum().float().item()
    
    if tp + fn > 0.0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
    if tp + fp > 0.0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    return precision, recall



def avg_precision_recall(model, eval_dataset, device, n_samples=100, conf_thr=0.5):
    print("avg_precision called")

    tps, fps, fns = [], [], []
    num_zero, num_one, num_two, active_anchor_one, active_anchor_two  = [], [], [], [], []
    max_two = [0, 0]
    model = model.to(device)
    model.eval()
    for s in range(n_samples):
        img, tgt = eval_dataset[(s * 20) % len(eval_dataset)]
        img = img.unsqueeze(0).unsqueeze(0).to(device) # 1, 1, H, W
        tgt = tgt.to(device)
        pred = model.forward(img).squeeze(0)
        pred = logit_to_target(pred)
        iou = pred_to_iou(pred, tgt)
        tp_mask, fp_mask, fn_mask, tgt_conf = calc_masks(pred, tgt, iou, conf_thr=conf_thr)

        tp = tp_mask.sum().float().item()
        fp = fp_mask.sum().float().item()
        fn = fn_mask.sum().float().item()

        tps.append(tp)
        fps.append(fp)
        fns.append(fn)


        #active_per_cell = tgt_conf.sum(dim=-1) # number of anchors per cell from tgt_conf
        # active_one = ((tgt_conf[..., 0] == 1) & (active_per_cell == 1)).sum().item()
        # active_two = ((tgt_conf[..., 1] == 1)  & (active_per_cell == 1)).sum().item()
        # active_anchor_one.append(active_one)
        # active_anchor_two.append(active_two)
        # zero = (active_per_cell == 0).sum().item()
        # one = (active_per_cell == 1).sum().item()
        # two = (active_per_cell == 2).sum().item()
        # if two > max_two[1]:
        #     max_two = [s, two]
        # num_zero.append(zero)
        # num_one.append(one)
        # num_two.append(two)
        
        

    # avg_zero = sum(num_zero) / len(num_zero)
    # avg_one = sum(num_one) / len(num_one)
    # #check if all ones are at anchor zero
    # avg_two = sum(num_two) / len(num_two)
    # avg_anchor_one = sum(active_anchor_one) / len(active_anchor_one)
    # avg_anchor_two = sum(active_anchor_two) / len(active_anchor_two)

    # print(f"avg_zero, avg_one, avg_two: {avg_zero, avg_one, avg_two}")
    # print(f"max_two: {max_two}")
    # print(f"avg active in anchor one vs. two: {avg_anchor_one, avg_anchor_two}")

    tp, fp, fn = sum(tps), sum(fps), sum(fns)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
        
    return recall, precision, tp, fp, fn





#######################################################################
#######################################################################
#######################################################################
###################### Debug unit_precision  ##########################
#######################################################################
#######################################################################
#######################################################################


def debug_pred_to_iou(obj_thresh = 0.5):
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
