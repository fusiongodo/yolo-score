import torch
import os
import json
import pandas as pd
from PIL import Image, ImageDraw
from importlib import reload
import config
import torch.nn as nn
import numpy as np
from IPython.display import display
reload(config)





class DataExtractor:
    def __init__(self, slim = False):
        self.original_data = config.filepath
        if not slim:
            self.filepath = config.filepath
            self.normalized_path = os.path.join(
                os.path.dirname(self.filepath),
                "gt_space.json"
            )
        else:
            self.filepath = config.slimpath
            self.normalized_path = os.path.join(
                os.path.dirname(self.filepath),
                "gt_space_slim.json"
            )
        # save next to deepscores_train.json:
        
        
        self.data = None
        self.anns_df = None
        print(f"Saving to: {self.normalized_path}")

    def loadData(self):
        with open(self.filepath, 'r') as f:
            return json.load(f)
        
    def loadSlimData(self):
        with open(self.original_data, 'r') as f:
            data =  json.load(f)

        # 2) collect annotation IDs actually used by the 11 images
        used_ids = {ann_id for img in data["images"] for ann_id in img["ann_ids"]}

        # 3) build a pared-down annotations dict (only a_bbox, cat_id, img_id)
        keep_keys = {"a_bbox", "cat_id", "img_id", "area"}
        data["annotations"] = {
            ann_id: {k: v for k, v in ann.items() if k in keep_keys}
            for ann_id, ann in data["annotations"].items()
            if ann_id in used_ids
        }

        print(f"kept {len(data['annotations'])} annotations; "
            f"each with keys {sorted(keep_keys)}")

        # 4) save compact JSON (no pretty-printing)
        with open(config.slimpath, "w") as f:
            json.dump(data, f, separators=(",", ":"))

        print(f"filtered file written →  {config.slimpath}")

    def createMergedAnnsDF(self):
        images = pd.DataFrame(self.data["images"]).set_index("id")
        anns   = pd.DataFrame(self.data["annotations"]).transpose()
        anns["img_id"] = anns["img_id"].astype(images.index.dtype)
        anns = (
            anns
            .merge(images[['filename','width','height']],
                   left_on='img_id', right_index=True)
            .loc[lambda df: ~df['area'].astype(float).eq(0)]
        )
        anns['cat_id'] = anns['cat_id'].apply(
            lambda x: x[0] if isinstance(x, (list,tuple)) and x else x
        )
        self.anns_df = anns
        return anns

    def normalizedData(self, grouped = False):
        if os.path.exists(self.normalized_path):
            return pd.read_json(self.normalized_path)

        if self.data is None:
            self.data = self.loadData()
        if self.anns_df is None:
            self.createMergedAnnsDF()

        xy = pd.DataFrame(self.anns_df['a_bbox'].tolist(),
                          columns=['x1','y1','x2','y2'],
                          index=self.anns_df.index)
        anns = pd.concat([self.anns_df, xy], axis=1)

        # center + size normalized
        anns['x'] = ((anns.x1 + anns.x2) * .5) / anns.width
        anns['y'] = ((anns.y1 + anns.y2) * .5) / anns.height
        anns['w'] = (anns.x2 - anns.x1) / anns.width
        anns['h'] = (anns.y2 - anns.y1) / anns.height

        # grid cell + offsets
        anns['cx'] = (anns.x * config.S).astype(int)
        anns['cy'] = (anns.y * config.S).astype(int)
        anns['tx'] = (anns.x * config.S - anns.cx).astype(np.float32)
        anns['ty'] = (anns.y * config.S - anns.cy).astype(np.float32)
        anns['tw'] = (np.log((anns.w / config.ANCHORS[0,0]).clip(lower=1e-9))).astype(np.float32)
        anns['th'] = (np.log((anns.h / config.ANCHORS[0,1]).clip(lower=1e-9))).astype(np.float32)
        anns["img_id"] = anns["img_id"].astype(int)


        df = (
            anns
            .rename(columns={'cat_id':'class_id'})
            [['img_id','filename','cx','cy','tx','ty','tw','th','class_id']]
            .reset_index(drop=True)
        )
        df['class_id'] = df.class_id.astype(int)
        df["class_id"] = df["class_id"] - 1
        df = df[df.class_id < config.C]

        if grouped:
            df = df.groupby('img_id')

        # ensure dir & save
        os.makedirs(os.path.dirname(self.normalized_path), exist_ok=True)
        df.to_json(self.normalized_path, orient='records', indent=2)
        return df
    
    def croppedData(self, grouped: bool = True):
        """        
            cx_loc / cy_loc ∈ [0, N-1]  (0-39)
        """
        df = self.normalizedData(grouped=False).copy()

        S, N = config.S, config.N             # 120, 40
        if S % N:
            raise ValueError("config.N must evenly divide config.S")

        crops_per_dim = S // N               # 3
        # ------- which crop does the box belong to? -------------------------------
        df["crop_row"] = (df["cy"] // N).astype(int)   # 0..2
        df["crop_col"] = (df["cx"] // N).astype(int)   # 0..2
        df["crop_id"]  = df["crop_row"] * crops_per_dim + df["crop_col"]

        # ------- cell indices *inside* that 40×40 crop ----------------------------
        df["cx_loc"] = (df["cx"] % N).astype(int)       # 0..39
        df["cy_loc"] = (df["cy"] % N).astype(int)       # 0..39

        col_order = [
            "img_id", "crop_id", "cx_loc", "cy_loc",
            "tx", "ty", "tw", "th",
            "class_id", "filename",
            "cx", "cy", "crop_row", "crop_col"
        ]
        df = df[col_order]

        # --- cache on disk so subsequent calls are cheap ----------------------------
        cropped_path = os.path.join(
            os.path.dirname(self.filepath),
            f"gt_space_{N}x{N}.json"
        )
        if not os.path.exists(cropped_path):
            os.makedirs(os.path.dirname(cropped_path), exist_ok=True)
            df.to_json(cropped_path, orient='records', indent=2)

        # --- optional hierarchical grouping ----------------------------------------
        if grouped:


            df["crop_uid"] = df.groupby(['img_id', 'crop_id']).ngroup()

        #remove staff line bboxes
        df = df[df.class_id != 134]
        return df


save_dir = "model_dumps"



def saveModel(filename: str, model: nn.Module, dump_dir = "model_dumps"):
    """Save model weights to ./model_dumps/<filename>.pth"""
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"{filename}.pth" if not filename.endswith('.pth') else filename)
    torch.save(model.state_dict(), path)
    if hasattr(model, 'device'):
        loc = getattr(model, 'device')
    else:
        loc = next(model.parameters()).device
    print(f"[saveModel] Saved weights to {path} (device={loc})")


def loadModel(filename: str, model: nn.Module, device: str = 'cpu', dir = 'model_dumps') -> nn.Module:
    path = os.path.join(dir, f"{filename}.pth" if not filename.endswith('.pth') else filename)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"[loadMode] Loaded weights from {path}")
    else:
        print(f"[loadMode] No checkpoint found at {path} — starting fresh")
    return model







def load_crop_image(img_path, crop_row, crop_col):
    res_factor = config.S // config.N  # e.g., 3
    effective_full_size = config.RES * res_factor

    cell_px = effective_full_size / config.S
    crop_px = config.N * cell_px
    left_px = int(round(crop_col * crop_px))
    top_px = int(round(crop_row * crop_px))
    right_px = int(round((crop_col + 1) * crop_px))
    lower_px = int(round((crop_row + 1) * crop_px))
    scale = config.RES / (right_px - left_px)

    full_img = Image.open(img_path).convert("RGB").resize((effective_full_size, effective_full_size))
    crop_img = full_img.crop((left_px, top_px, right_px, lower_px)).resize((config.RES, config.RES))

    crop_img = crop_img.convert("L") 

    return crop_img, left_px, top_px, scale, effective_full_size





# used in visualize.ipynb
def render_prediction(image, target, iou, colour=(0, 255, 0, 200), obj_thresh = 0.5,

                             out_dir="evaluation_crops_from_dataset",
                             name="crop.png"):
    if image.dim() == 2:
        # (H, W)
        h, w = image.shape
        if h != w:
            raise ValueError("Image must be square.")
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
    elif image.dim() == 3 and image.shape[0] == 1:
        # (1, H, W)
        h, w = image.shape[1:]
        if h != w:
            raise ValueError("Image must be square.")
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
    else:
        raise ValueError("Image must be (H,W) or (1,H,W) tensor for grayscale.")
    
    crop_img = Image.fromarray(img_np, mode='L').convert("RGBA") #changed for colored boxes

    # Draw setup
    draw = ImageDraw.Draw(crop_img, "RGBA")  #changed for colored boxes

    

    N, _, A, _ = target.shape
    side_size = config.RES  # Assuming this is the side size (e.g., 224)

    # Create meshgrid for i, j, a
    i_grid, j_grid, a_grid = torch.meshgrid(
        torch.arange(N, device=target.device),
        torch.arange(N, device=target.device),
        torch.arange(A, device=target.device),
        indexing='ij'
    )

    flat_iou = iou.reshape(-1)
    mask = flat_iou > obj_thresh # e.g., 0.2

    # Flatten everything
    flat_i = i_grid.reshape(-1)[mask]
    flat_j = j_grid.reshape(-1)[mask]
    flat_a = a_grid.reshape(-1)[mask]
    flat_params = target[..., :5].reshape(-1, 5)[mask]  # (N*N*A, 5), where [:,4] is obj
    

    # Mask for valid boxes
    mask = flat_params[:, 4] >= obj_thresh

    if not mask.any():
        # No boxes to draw
        pass  # Proceed to save, etc.

    # Extract valid components
    valid_i = flat_i[mask]
    valid_j = flat_j[mask]
    valid_a = flat_a[mask]
    tx = flat_params[mask, 0]
    ty = flat_params[mask, 1]
    tw = flat_params[mask, 2]
    th = flat_params[mask, 3]

    # Relative coordinates in crop (0-1)
    cx_rel = (valid_j + tx) / N
    cy_rel = (valid_i + ty) / N

    # Anchors as tensor (assuming config.ANCHORS is list of lists/tuples, shape (A, 2))
    anchors = torch.tensor(config.ANCHORS, device=target.device, dtype=torch.float32)  # (A, 2)

    # Compute w_rel and h_rel
    scale_factor = config.S / config.N  # Use float division
    w_rel = torch.exp(tw) * anchors[valid_a, 0] * scale_factor
    h_rel = torch.exp(th) * anchors[valid_a, 1] * scale_factor

    # Absolute pixel corners (vectorized _rel2abs)
    cx_abs = cx_rel * side_size
    cy_abs = cy_rel * side_size
    w_abs = w_rel * side_size
    h_abs = h_rel * side_size
    x0 = cx_abs - w_abs / 2
    y0 = cy_abs - h_abs / 2
    x1 = cx_abs + w_abs / 2
    y1 = cy_abs + h_abs / 2

# Now, draw in a loop (since PIL drawing isn't vectorized, but this is fast for typical num_boxes << N*N*A)
    for k in range(len(x0)):
        #draw.rectangle([x0[k].item(), y0[k].item(), x1[k].item(), y1[k].item()], outline=128, width=2)
        draw.rectangle(
            [x0[k].item(), y0[k].item(), x1[k].item(), y1[k].item()],
            outline=colour,  # Use provided color (e.g., green (0, 255, 0, 200) or red (255, 0, 0, 200))
            width=1
        )

    # Save and return
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, name)
    crop_img.save(out_path)
    return crop_img



# def render_demo(image, pred, obj_thresh=0.5,
#                       colour=(0, 255, 0, 200),
#                       out_dir="evaluation_crops_from_dataset",
#                       name="crop.png"):
#     # image: (H,W) or (1,H,W) tensor in [0,1]; pred: (N,N,A,5+C) on any device
#     import os, torch
#     from PIL import Image, ImageDraw
#     import numpy as np

#     # --- image to RGBA ---
#     if image.dim() == 3 and image.shape[0] == 1:
#         image = image[0]
#     if image.dim() != 2:
#         raise ValueError("image must be (H,W) or (1,H,W) grayscale")

#     H, W = image.shape
#     if H != W:
#         raise ValueError("Image must be square.")
#     side_size = H

#     img_np = (image.detach().to("cpu").numpy() * 255).astype(np.uint8)
#     crop_img = Image.fromarray(img_np, mode="L").convert("RGBA")
#     draw = ImageDraw.Draw(crop_img, "RGBA")

#     # --- boxes from pred ---
#     device = pred.device
#     N, _, A, _ = pred.shape
#     i_idx, j_idx, a_idx = torch.meshgrid(
#         torch.arange(N, device=device),
#         torch.arange(N, device=device),
#         torch.arange(A, device=device),
#         indexing="ij"
#     )

#     # objectness
#     obj = pred[..., 4]
#     keep = obj >= obj_thresh
#     if not keep.any():
#         os.makedirs(out_dir, exist_ok=True)
#         crop_img.save(os.path.join(out_dir, name))
#         return crop_img

#     tx = pred[..., 0][keep]
#     ty = pred[..., 1][keep]
#     tw = pred[..., 2][keep]
#     th = pred[..., 3][keep]
#     ii = i_idx[keep]
#     jj = j_idx[keep]
#     aa = a_idx[keep]

#     # centers relative to crop
#     cx_rel = (jj + tx) / N
#     cy_rel = (ii + ty) / N

#     anchors = torch.tensor(config.ANCHORS, dtype=torch.float32, device=device)  # (A,2)
#     scale_factor = float(config.S) / float(config.N)  # e.g. 120/40=3
#     w_rel = torch.exp(tw) * anchors[aa, 0] * scale_factor
#     h_rel = torch.exp(th) * anchors[aa, 1] * scale_factor

#     # to pixels
#     cx = cx_rel * side_size
#     cy = cy_rel * side_size
#     w  = w_rel * side_size
#     h  = h_rel * side_size

#     x0 = (cx - w/2).tolist()
#     y0 = (cy - h/2).tolist()
#     x1 = (cx + w/2).tolist()
#     y1 = (cy + h/2).tolist()

#     for a,b,c,d in zip(x0,y0,x1,y1):
#         draw.rectangle([a,b,c,d], outline=colour, width=1)

#     os.makedirs(out_dir, exist_ok=True)
#     crop_img.save(os.path.join(out_dir, name))
#     return crop_img


def render_demo(image, pred, obj_thresh=0.5,
                colour=(0, 255, 0, 200),
                out_dir="evaluation_crops_from_dataset",
                name="crop.png",
                class_names=None,          # e.g. ["notehead", "rest", ...]
                show_score=True):
    import os, torch
    from PIL import Image, ImageDraw
    import numpy as np

    # --- image to RGBA ---
    if image.dim() == 3 and image.shape[0] == 1:
        image = image[0]
    if image.dim() != 2:
        raise ValueError("image must be (H,W) or (1,H,W) grayscale")
    H, W = image.shape
    if H != W:
        raise ValueError("Image must be square.")
    side_size = H

    img_np = (image.detach().to("cpu").numpy() * 255).astype(np.uint8)
    crop_img = Image.fromarray(img_np, mode="L").convert("RGBA")
    draw = ImageDraw.Draw(crop_img, "RGBA")

    # --- indices ---
    device = pred.device
    N, _, A, K = pred.shape  # K = 5 + C
    C = K - 5
    i_idx, j_idx, a_idx = torch.meshgrid(
        torch.arange(N, device=device),
        torch.arange(N, device=device),
        torch.arange(A, device=device),
        indexing="ij"
    )

    # objectness + keep
    obj = pred[..., 4]
    keep = obj >= obj_thresh
    if not keep.any():
        os.makedirs(out_dir, exist_ok=True)
        crop_img.save(os.path.join(out_dir, name))
        return crop_img

    # box params
    tx = pred[..., 0][keep]; ty = pred[..., 1][keep]
    tw = pred[..., 2][keep]; th = pred[..., 3][keep]
    ii = i_idx[keep];        jj = j_idx[keep]; aa = a_idx[keep]
    # class ids (argmax over one-hot)
    cls_id = pred[..., 5:].argmax(dim=-1)[keep]  # (M,)
    score  = obj[keep]                            # (M,)

    # centers relative to crop
    cx_rel = (jj + tx) / N
    cy_rel = (ii + ty) / N

    anchors = torch.tensor(config.ANCHORS, dtype=torch.float32, device=device)  # (A,2)
    scale_factor = float(config.S) / float(config.N)
    w_rel = torch.exp(tw) * anchors[aa, 0] * scale_factor
    h_rel = torch.exp(th) * anchors[aa, 1] * scale_factor

    # to pixels
    cx = (cx_rel * side_size).tolist()
    cy = (cy_rel * side_size).tolist()
    w  = (w_rel  * side_size).tolist()
    h  = (h_rel  * side_size).tolist()

    for x_c, y_c, ww, hh, c_id, sc in zip(cx, cy, w, h, cls_id.tolist(), score.tolist()):
        x0 = x_c - ww/2; y0 = y_c - hh/2
        x1 = x_c + ww/2; y1 = y_c + hh/2
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=1)

        # --- tiny id label, always readable ---
        label = str(c_id)
        fs = max(6, min(9, int(0.10 * min(ww, hh))))  # keep very small
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("DejaVuSans.ttf", fs)
        except Exception:
            font = None  # fallback to default

        tx = max(0, min(side_size - 1, int(x0) + 1))
        ty = max(0, min(side_size - 1, int(y0) + 1))

        # Draw black text with a thin white outline
        draw.text(
            (tx, ty), label,
            fill=(0, 0, 0, 255),
            font=font,
            stroke_width=1,
            stroke_fill=(255, 255, 255, 255)
        )



    os.makedirs(out_dir, exist_ok=True)
    crop_img.save(os.path.join(out_dir, name))
    return crop_img