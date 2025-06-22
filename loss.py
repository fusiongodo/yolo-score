import torch
import torch.nn as nn
import torch.nn.functional as F
import config




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






class YOLOv2Loss(nn.Module):
    """Composite YOLO‑v2 loss returning total and breakdown dict."""

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
        # Split prediction tensor --------------------------------------
        pred_tx, pred_ty, pred_tw, pred_th, pred_to = torch.split(pred[..., :5], 1, dim=-1)
        pred_tx, pred_ty = pred_tx.squeeze(-1), pred_ty.squeeze(-1)
        pred_tw, pred_th = pred_tw.squeeze(-1), pred_th.squeeze(-1)
        pred_to = pred_to.squeeze(-1)
        pred_cls = pred[..., 5:]
        # Split target tensor ------------------------------------------
        tgt_tx, tgt_ty, tgt_tw, tgt_th, tgt_to = torch.split(target[..., :5], 1, dim=-1)
        tgt_tx, tgt_ty = tgt_tx.squeeze(-1), tgt_ty.squeeze(-1)
        tgt_tw, tgt_th = tgt_tw.squeeze(-1), tgt_th.squeeze(-1)
        tgt_to = tgt_to.squeeze(-1)
        tgt_cls = target[..., 5:]
        # Masks (initial) ----------------------------------------------
        obj_mask = tgt_to > 0.5
        # Decode to cell‑space for IoU matching -------------------------
        p_tx_s, p_ty_s, p_w, p_h = _decode_boxes(torch.sigmoid(pred_tx), torch.sigmoid(pred_ty), pred_tw, pred_th)
        g_tx_s, g_ty_s, g_w, g_h = _decode_boxes(tgt_tx, tgt_ty, tgt_tw, tgt_th)
        # Greedy matching ----------------------------------------------
        aligned_target = torch.zeros_like(target)
        anchor_taken = torch.zeros((N, S, S, A), dtype=torch.bool, device=device)
        for n in range(N):
            for i in range(S):
                for j in range(S):
                    for a_gt in range(A):
                        if not obj_mask[n, i, j, a_gt]:
                            continue
                        gt_box = torch.stack([g_tx_s[n, i, j, a_gt], g_ty_s[n, i, j, a_gt], g_w[n, i, j, a_gt], g_h[n, i, j, a_gt]])
                        pred_boxes = torch.stack([p_tx_s[n, i, j, :], p_ty_s[n, i, j, :], p_w[n, i, j, :], p_h[n, i, j, :]], dim=-1)
                        ious = _bbox_iou(pred_boxes, gt_box.unsqueeze(0))
                        ious[anchor_taken[n, i, j]] = -1
                        best_a = torch.argmax(ious)
                        anchor_taken[n, i, j, best_a] = True
                        aligned_target[n, i, j, best_a] = target[n, i, j, a_gt]
        tgt_to_aligned = aligned_target[..., 4]
        obj_mask = tgt_to_aligned > 0.5
        noobj_mask = ~obj_mask
        # Loss components ---------------------------------------------
        loss_xy = self.l_xy * (F.mse_loss(torch.sigmoid(pred_tx)[obj_mask], aligned_target[..., 0][obj_mask], reduction='sum') +
                               F.mse_loss(torch.sigmoid(pred_ty)[obj_mask], aligned_target[..., 1][obj_mask], reduction='sum'))
        loss_wh = self.l_wh * (F.mse_loss(pred_tw[obj_mask], aligned_target[..., 2][obj_mask], reduction='sum') +
                               F.mse_loss(pred_th[obj_mask], aligned_target[..., 3][obj_mask], reduction='sum'))
        loss_obj = self.l_obj * F.binary_cross_entropy_with_logits(pred_to[obj_mask], torch.ones_like(pred_to[obj_mask]), reduction='sum')
        loss_noobj = self.l_noobj * F.binary_cross_entropy_with_logits(pred_to[noobj_mask], torch.zeros_like(pred_to[noobj_mask]), reduction='sum')
        if obj_mask.any():
            tgt_labels = torch.argmax(aligned_target[..., 5:][obj_mask], dim=-1)
            loss_cls = self.l_cls * F.cross_entropy(pred_cls[obj_mask], tgt_labels, reduction='sum')
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