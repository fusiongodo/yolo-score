import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from torch.utils.data import DataLoader
from dataset import DummyDataLoader



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
    
 #------------------------------------------------------------------
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

def mean_average_precision(model: nn.Module, dataset, device='cpu', iou_thresh=0.5, score_thresh=0.5, max_batches: int | None = 3, dummy=False, setchoice = "one", modelchoice = "one"):
    """Compute mAP@IoU using at most *max_batches* batches (fast eval)."""
    if dummy:
        loader = DummyDataLoader(dataset, choice=setchoice, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    model.to(device)
    num_classes = config.C
    TP, FP, Conf = [[] for _ in range(num_classes)], [[] for _ in range(num_classes)], [[] for _ in range(num_classes)]
    total_gt = [0] * num_classes

    with torch.no_grad():
        for b_idx, (imgs, targets) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            imgs = imgs.to(device)
            if dummy:
                preds = model.dummy_forward(batch_size=2, mode=modelchoice)
            else:
                preds = model(imgs).cpu()
            for pred, tgt in zip(preds, targets):
                stats = unit_precision(pred, tgt, device='cpu', iou_thresh=iou_thresh, score_thresh=score_thresh, return_stats=True)
                for c in range(num_classes):
                    TP[c].extend(stats['TP'][c])
                    FP[c].extend(stats['FP'][c])
                    Conf[c].extend(stats['Conf'][c])
                    total_gt[c] += stats['total_gt'][c]

    APs = []
    for c in range(num_classes):
        if total_gt[c] == 0:
            continue
        if len(Conf[c]) == 0:
            APs.append(0.0)
            continue
        scores_c = torch.tensor(Conf[c])
        order = torch.argsort(scores_c, descending=True)
        tp_c = torch.tensor(TP[c])[order].float()
        fp_c = torch.tensor(FP[c])[order].float()
        tp_c = torch.cumsum(tp_c, 0)
        fp_c = torch.cumsum(fp_c, 0)
        rec = tp_c / (total_gt[c] + 1e-9)
        prec = tp_c / (tp_c + fp_c + 1e-9)
        APs.append(_compute_ap(rec, prec).item())

    mAP = sum(APs) / len(APs) if APs else 0.0
    print(f"mAP@{iou_thresh}: {mAP:.4f}  (evaluated on {min(max_batches, len(loader)) if max_batches else len(loader)} batch(es))")
    return mAP




def _compute_ap(rec, prec):
    """11‑point interpolation (VOC 2007)"""
    ap = 0.0
    for t in torch.linspace(0,1,11):
        if (rec>=t).any():
            ap += prec[rec>=t].max()
    return ap/11


def _boxes_to_corners(box):
    """(cx,cy,w,h) → (x1,y1,x2,y2) with same scale."""
    x1 = box[..., 0] - box[..., 2] / 2
    y1 = box[..., 1] - box[..., 3] / 2
    x2 = box[..., 0] + box[..., 2] / 2
    y2 = box[..., 1] + box[..., 3] / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)


def _box_iou_matrix(boxes1, boxes2):
    """Vectorised IoU for two sets of boxes in (cx,cy,w,h)."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)), device=boxes1.device)
    b1 = _boxes_to_corners(boxes1)  # (M,4)
    b2 = _boxes_to_corners(boxes2)  # (N,4)
    # broadcast corners
    tl = torch.maximum(b1[:, None, :2], b2[None, :, :2])  # top‑left
    br = torch.minimum(b1[:, None, 2:], b2[None, :, 2:])  # bottom‑right
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - inter + 1e-9
    return inter / union


def unit_precision(pred, gt, device='cpu', iou_thresh=0.5, score_thresh=0.5, return_stats=False):
    pred = pred.to(device)
    gt = gt.to(device)
    N = config.N
    A = config.A
    C = config.C
    S_over_N = config.S / config.N

    # Decode GT
    obj_gt = gt[..., 4] > score_thresh
    if obj_gt.any():
        obj_indices_gt = obj_gt.nonzero(as_tuple=False)
        i_gt = obj_indices_gt[:, 0]
        j_gt = obj_indices_gt[:, 1]
        a_gt = obj_indices_gt[:, 2]
        tx_gt = gt[i_gt, j_gt, a_gt, 0]
        ty_gt = gt[i_gt, j_gt, a_gt, 1]
        tw_gt = gt[i_gt, j_gt, a_gt, 2]
        th_gt = gt[i_gt, j_gt, a_gt, 3]
        anchors = torch.tensor(config.ANCHORS, device=device)
        anc_w_gt = anchors[a_gt, 0]
        anc_h_gt = anchors[a_gt, 1]
        cx_gt = (j_gt.float() + tx_gt) / N
        cy_gt = (i_gt.float() + ty_gt) / N
        w_gt = torch.exp(tw_gt) * anc_w_gt * S_over_N
        h_gt = torch.exp(th_gt) * anc_h_gt * S_over_N
        gboxes = torch.stack((cx_gt, cy_gt, w_gt, h_gt), dim=-1)
        glabels = gt[i_gt, j_gt, a_gt, 5:].argmax(dim=-1)
    else:
        gboxes = torch.empty((0, 4), device=device)
        glabels = torch.empty((0,), dtype=torch.long, device=device)

    # Decode predictions (treat as GT-like: no activations)
    i_grid, j_grid, a_grid = torch.meshgrid(
        torch.arange(N, device=device),
        torch.arange(N, device=device),
        torch.arange(A, device=device),
        indexing='ij'
    )
    flat_i = i_grid.reshape(-1)
    flat_j = j_grid.reshape(-1)
    flat_a = a_grid.reshape(-1)
    flat_pred = pred.reshape(-1, 5 + C)
    tx = flat_pred[:, 0]
    ty = flat_pred[:, 1]
    tw = flat_pred[:, 2]
    th = flat_pred[:, 3]
    obj = flat_pred[:, 4]
    class_logits = flat_pred[:, 5:]
    labels = class_logits.argmax(dim=-1)
    scores = obj
    mask = scores > score_thresh
    if not mask.any():
        p_boxes = torch.empty((0, 4), device=device)
        p_scores = torch.empty((0,), device=device)
        p_labels = torch.empty((0,), dtype=torch.long, device=device)
    else:
        valid_tx = tx[mask]
        valid_ty = ty[mask]
        valid_tw = tw[mask]
        valid_th = th[mask]
        valid_scores = scores[mask]
        valid_labels = labels[mask]
        valid_a = flat_a[mask]
        valid_j = flat_j[mask]
        valid_i = flat_i[mask]
        anchors = torch.tensor(config.ANCHORS, device=device)
        anc_w = anchors[valid_a, 0]
        anc_h = anchors[valid_a, 1]
        cx = (valid_j.float() + valid_tx) / N
        cy = (valid_i.float() + valid_ty) / N
        w = torch.exp(valid_tw) * anc_w * S_over_N
        h = torch.exp(valid_th) * anc_h * S_over_N
        p_boxes = torch.stack((cx, cy, w, h), dim=-1)
        p_scores = valid_scores
        p_labels = valid_labels

    # Matching to compute TP and FP
    if return_stats:
        TP_per = [[] for _ in range(C)]
        FP_per = [[] for _ in range(C)]
        Conf_per = [[] for _ in range(C)]
        total_gt_per = [0] * C
        for c in range(C):
            total_gt_per[c] = (glabels == c).sum().item()
        unique_classes = torch.unique(torch.cat((p_labels, glabels)))
        for cls in unique_classes:
            c = int(cls.item())
            mask_pred = p_labels == c
            if mask_pred.sum() == 0:
                continue
            p_boxes_c = p_boxes[mask_pred]
            p_scores_c = p_scores[mask_pred]
            order = torch.argsort(p_scores_c, descending=True)
            p_boxes_c = p_boxes_c[order]
            p_scores_c = p_scores_c[order]
            mask_gt = glabels == c
            g_boxes_c = gboxes[mask_gt]
            if g_boxes_c.size(0) == 0:
                for k in range(p_boxes_c.size(0)):
                    TP_per[c].append(0)
                    FP_per[c].append(1)
                    Conf_per[c].append(p_scores_c[k].item())
                continue
            matched_gt = torch.zeros(g_boxes_c.size(0), dtype=torch.bool, device=device)
            for k in range(p_boxes_c.size(0)):
                box = p_boxes_c[k:k+1]
                ious = _box_iou_matrix(box, g_boxes_c).squeeze(0)
                ious[matched_gt] = -1.0
                max_iou, idx = ious.max(0)
                if max_iou >= iou_thresh:
                    TP_per[c].append(1)
                    FP_per[c].append(0)
                    matched_gt[idx] = True
                else:
                    TP_per[c].append(0)
                    FP_per[c].append(1)
                Conf_per[c].append(p_scores_c[k].item())
        return {'TP': TP_per, 'FP': FP_per, 'Conf': Conf_per, 'total_gt': total_gt_per}
    else:
        TP = 0
        FP = 0
        total_gt_count = gboxes.size(0)
        if total_gt_count == 0 and p_boxes.size(0) == 0:
            return 1.0
        if p_boxes.size(0) == 0 and total_gt_count > 0:
            return 0.0
        unique_classes = torch.unique(torch.cat((p_labels, glabels)))
        for cls in unique_classes:
            c = int(cls.item())
            mask_pred = p_labels == c
            if mask_pred.sum() == 0:
                continue
            p_boxes_c = p_boxes[mask_pred]
            p_scores_c = p_scores[mask_pred]
            order = torch.argsort(p_scores_c, descending=True)
            p_boxes_c = p_boxes_c[order]
            p_scores_c = p_scores_c[order]
            mask_gt = glabels == c
            g_boxes_c = gboxes[mask_gt]
            if g_boxes_c.size(0) == 0:
                FP += p_boxes_c.size(0)
                continue
            matched_gt = torch.zeros(g_boxes_c.size(0), dtype=torch.bool, device=device)
            for k in range(p_boxes_c.size(0)):
                box = p_boxes_c[k:k+1]
                ious = _box_iou_matrix(box, g_boxes_c).squeeze(0)
                ious[matched_gt] = -1.0
                max_iou, idx = ious.max(0)
                if max_iou >= iou_thresh:
                    TP += 1
                    matched_gt[idx] = True
                else:
                    FP += 1
        if TP + FP == 0:
            return 0.0
        return TP / (TP + FP)