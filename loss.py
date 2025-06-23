import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from torch.utils.data import DataLoader



def _decode_boxes(tx, ty, tw, th):
    """Convert (tx,ty,tw,th) in cell‑space to (tx,ty,w,h)."""
    anchors = torch.tensor(config.ANCHORS, device=tx.device, dtype=tx.dtype)  # (A,2)
    aw = anchors[:, 0].view(1, 1, 1, -1)
    ah = anchors[:, 1].view(1, 1, 1, -1)
    w = torch.exp(tw) * aw
    h = torch.exp(th) * ah
    return tx, ty, w, h


def _bbox_iou(box1, box2):
    """IoU of two boxes given (tx,ty,w,h) in the same cell."""
    b1_x1 = box1[..., 0] - box1[..., 2] / 2
    b1_y1 = box1[..., 1] - box1[..., 3] / 2
    b1_x2 = box1[..., 0] + box1[..., 2] / 2
    b1_y2 = box1[..., 1] + box1[..., 3] / 2
    b2_x1 = box2[..., 0] - box2[..., 2] / 2
    b2_y1 = box2[..., 1] - box2[..., 3] / 2
    b2_x2 = box2[..., 0] + box2[..., 2] / 2
    b2_y2 = box2[..., 1] + box2[..., 3] / 2
    inter_x1 = torch.maximum(b1_x1, b2_x1)
    inter_y1 = torch.maximum(b1_y1, b2_y1)
    inter_x2 = torch.minimum(b1_x2, b2_x2)
    inter_y2 = torch.minimum(b1_y2, b2_y2)
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1 + area2 - inter_area + 1e-9
    return inter_area / union

# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------

class YOLOv2Loss(nn.Module):
    """Composite YOLO‑v2 loss returning total and breakdown dict.

    Matching step is vectorised over all ground‑truth boxes to avoid the
    quadruple‑nested Python loop.
    """

    def __init__(self, l_xy=5.0, l_wh=5.0, l_obj=1.0, l_noobj=0.5, l_cls=1.0):
        super().__init__()
        self.l_xy = l_xy
        self.l_wh = l_wh
        self.l_obj = l_obj
        self.l_noobj = l_noobj
        self.l_cls = l_cls

    # ------------------------------------------------------------------
    def forward(self, pred, target):
        device = pred.device
        N, S, _, A, _ = pred.shape

        # ---------------- Split prediction & target tensors ------------
        def _split(t):
            tx, ty, tw, th, to = torch.split(t[..., :5], 1, dim=-1)
            return tx.squeeze(-1), ty.squeeze(-1), tw.squeeze(-1), th.squeeze(-1), to.squeeze(-1), t[..., 5:]

        p_tx, p_ty, p_tw, p_th, p_to, p_cls = _split(pred)
        g_tx, g_ty, g_tw, g_th, g_to, g_cls = _split(target)

        # Mask of GT objects
        obj_mask = g_to > 0.5  # (N,S,S,A)

        # ---------------- Decode to box params for IoU -----------------
        p_tx_s, p_ty_s, p_w, p_h = _decode_boxes(torch.sigmoid(p_tx), torch.sigmoid(p_ty), p_tw, p_th)
        g_tx_s, g_ty_s, g_w, g_h = _decode_boxes(g_tx, g_ty, g_tw, g_th)

        pred_boxes = torch.stack((p_tx_s, p_ty_s, p_w, p_h), dim=-1)   # (N,S,S,A,4)
        gt_boxes   = torch.stack((g_tx_s, g_ty_s, g_w, g_h), dim=-1)   # (N,S,S,A,4)

        # ---------------- Vectorised greedy matching -------------------
        # Flatten grid for easier indexing
        P = pred_boxes.view(N, -1, A, 4)           # (N, S², A, 4)
        G = gt_boxes.view(N, -1, A, 4)
        mask = obj_mask.view(N, -1, A)             # (N, S², A)

                # IoU between each GT box and the A predictions in its cell
        ious = _bbox_iou(P.unsqueeze(3), G.unsqueeze(2))  # (N,S²,A_pred,A_gt)
        # Mask out cells where there is no GT (objectness==0)
        mask_exp = mask.unsqueeze(2).expand_as(ious)      # (N,S²,A_pred,A_gt)
        ious = ious.masked_fill(~mask_exp, -1)

        # Greedy pick best pred anchor per GT box
        _, best_pred_idx = ious.max(dim=2)  # over A_pred → (N,S²,A_gt) = ious.max(dim=2)  # over A_pred → (N,S²,A_gt)

        # Build aligned_target by scattering
        aligned_target = torch.zeros_like(target.view(N, -1, A, 5 + config.C))
        anchor_taken = torch.zeros((N, S * S, A), dtype=torch.bool, device=device)

        gt_indices = mask.nonzero(as_tuple=False)  # (M,3) with dims (n, cell, a_gt)
        for n, cell, a_gt in gt_indices:
            a_pred = best_pred_idx[n, cell, a_gt].item()
            if anchor_taken[n, cell, a_pred]:
                continue  # this anchor already assigned in this cell
            anchor_taken[n, cell, a_pred] = True
            aligned_target[n, cell, a_pred] = target.view(N, -1, A, 5 + config.C)[n, cell, a_gt]

        aligned_target = aligned_target.view_as(target)
        obj_mask = aligned_target[..., 4] > 0.5
        noobj_mask = ~obj_mask

        # ------------------- Loss components ---------------------------
        loss_xy = self.l_xy * (
            F.mse_loss(torch.sigmoid(p_tx)[obj_mask], aligned_target[..., 0][obj_mask], reduction='sum') +
            F.mse_loss(torch.sigmoid(p_ty)[obj_mask], aligned_target[..., 1][obj_mask], reduction='sum'))

        loss_wh = self.l_wh * (
            F.mse_loss(p_tw[obj_mask], aligned_target[..., 2][obj_mask], reduction='sum') +
            F.mse_loss(p_th[obj_mask], aligned_target[..., 3][obj_mask], reduction='sum'))

        loss_obj = self.l_obj * F.binary_cross_entropy_with_logits(p_to[obj_mask], torch.ones_like(p_to[obj_mask]), reduction='sum')
        loss_noobj = self.l_noobj * F.binary_cross_entropy_with_logits(p_to[noobj_mask], torch.zeros_like(p_to[noobj_mask]), reduction='sum')

        if obj_mask.any():
            tgt_labels = torch.argmax(aligned_target[..., 5:][obj_mask], dim=-1)
            loss_cls = self.l_cls * F.cross_entropy(p_cls[obj_mask], tgt_labels, reduction='sum')
        else:
            loss_cls = pred.sum() * 0

        total = (loss_xy + loss_wh + loss_obj + loss_noobj + loss_cls) / N
        breakdown = {
            'total': total.detach(),
            'l_xy': loss_xy.detach() / N,
            'l_wh': loss_wh.detach() / N,
            'l_obj': loss_obj.detach() / N,
            'l_noobj': loss_noobj.detach() / N,
            'l_cls': loss_cls.detach() / N,
        }
        return total, breakdown
    

# ------------------------------------------------------------------
# Evaluation: mean Average Precision (mAP @ IoU=0.5)
# ------------------------------------------------------------------

def _decode_predictions(pred, score_thresh=0.5):
    """Decode raw model output (N,S,S,A,5+C) → list of detections per image.

    • Applies sigmoid to tx, ty, to.
    • Decodes (w,h) **before** masking so anchor broadcast matches.
    • Filters by (objectness * class‑score) > score_thresh.
    """
    N, S, _, A, _ = pred.shape

    tx = torch.sigmoid(pred[..., 0])  # (N,S,S,A)
    ty = torch.sigmoid(pred[..., 1])
    tw = pred[..., 2]
    th = pred[..., 3]
    to = torch.sigmoid(pred[..., 4])
    cls_logits = pred[..., 5:]               # (N,S,S,A,C)
    cls_scores, cls_idx = torch.max(torch.softmax(cls_logits, dim=-1), dim=-1)  # (N,S,S,A)

    # decode boxes (broadcast with anchors)
    _, _, w, h = _decode_boxes(tx, ty, tw, th)        # (N,S,S,A)

    # flatten everything to (N*S*S*A)
    boxes   = torch.stack([tx, ty, w, h], dim=-1).reshape(-1,4)
    scores  = (to * cls_scores).reshape(-1)           # combined conf
    labels  = cls_idx.reshape(-1)
    img_ids = torch.arange(N).view(N,1,1,1).expand(N,S,S,A).reshape(-1)  # which image each pred belongs to

    keep = scores > score_thresh
    boxes, scores, labels, img_ids = boxes[keep], scores[keep], labels[keep], img_ids[keep]

    # build per‑image output list
    outs = []
    for n in range(N):
        sel = img_ids == n
        outs.append({
            'boxes': boxes[sel],
            'scores': scores[sel],
            'labels': labels[sel]
        })
    return outs


def _compute_ap(rec, prec):
    """11‑point interpolation (VOC 2007)"""
    ap = 0.0
    for t in torch.linspace(0,1,11):
        if (rec>=t).any():
            ap += prec[rec>=t].max()
    return ap/11


def average_precision(model: nn.Module, dataset, device='cpu', iou_thresh=0.5, score_thresh=0.5, max_batches: int | None = 3):
    """Compute mAP@0.5 using at most *max_batches* batches from *dataset*.

    Parameters
    ----------
    model : nn.Module
    dataset : Dataset
    device : str
    iou_thresh : float
    score_thresh : float
    max_batches : int | None
        If given, stop evaluation after this many batches (quick sanity‑check).
    """
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0) 
    model.eval(); model.to(device)
    num_classes = config.C
    # per‑class lists
    TP, FP, scores, total_gt = [[] for _ in range(num_classes)], [[] for _ in range(num_classes)], [[] for _ in range(num_classes)], [0]*num_classes

    with torch.no_grad():
        for b_idx, (imgs, targets) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            imgs = imgs.to(device)
            preds = model(imgs).cpu()
            detections = _decode_predictions(preds, score_thresh)
            for det, tgt in zip(detections, targets):
                # ground truth boxes per class
                gt_boxes_per_cls = {c: [] for c in range(num_classes)}
                obj = tgt[...,4] > 0.5
                if obj.any():
                    # flatten anchor index for each GT box in this image
                    anc_idx = obj.nonzero(as_tuple=False)[:, -1]  # anchor id (last dim)  # anchor id 0..A-1
                    g_tx = tgt[..., 0][obj]
                    g_ty = tgt[..., 1][obj]
                    g_tw = tgt[..., 2][obj]
                    g_th = tgt[..., 3][obj]
                    anc = torch.tensor(config.ANCHORS, device=tgt.device, dtype=tgt.dtype)[anc_idx]
                    gw = torch.exp(g_tw) * anc[:, 0]
                    gh = torch.exp(g_th) * anc[:, 1]
                    gboxes = torch.stack([g_tx, g_ty, gw, gh], dim=-1)
                    glabels = tgt[..., 5:][obj].argmax(dim=-1)
                    for b, l in zip(gboxes, glabels):
                        gt_boxes_per_cls[int(l.item())].append({'box':b, 'matched':False})
                        total_gt[int(l)] += 1
                # predictions per class sorted by score
                for box, score, label in zip(det['boxes'], det['scores'], det['labels']):
                    c = int(label.item())
                    scores[c].append(score.item())
                    match = False
                    for gt in gt_boxes_per_cls[c]:
                        if not gt['matched'] and _bbox_iou(box, gt['box']) >= iou_thresh:
                            match = True; gt['matched'] = True; break
                    TP[c].append(1 if match else 0)
                    FP[c].append(0 if match else 1)

    APs = []
    for c in range(num_classes):
        if total_gt[c] == 0:
            continue
        # sort by score desc
        if len(scores[c])==0:
            APs.append(0.0); continue
        scores_c = torch.tensor(scores[c])
        order = torch.argsort(scores_c, descending=True)
        tp_c = torch.tensor(TP[c])[order].float()
        fp_c = torch.tensor(FP[c])[order].float()
        tp_c = torch.cumsum(tp_c, dim=0)
        fp_c = torch.cumsum(fp_c, dim=0)
        rec = tp_c / (total_gt[c]+1e-9)
        prec = tp_c / (tp_c + fp_c + 1e-9)
        APs.append(_compute_ap(rec, prec).item())
    mAP = sum(APs)/len(APs) if APs else 0.0
    print(f"mAP@{iou_thresh}: {mAP:.4f}")
    return mAP
