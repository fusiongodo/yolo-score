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

        # ---------------- Split prediction & target tensors, apply sigmoid to to ------------
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
    






####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################