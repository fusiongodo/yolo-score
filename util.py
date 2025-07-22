import torch
import os
import json
import pandas as pd
from PIL import Image, ImageDraw
from importlib import reload
import config
import torch.nn as nn
import numpy as np

import torch
import os
import json
import pandas as pd
import config
import numpy as np


import torch
import os
import json
import pandas as pd
import config
import numpy as np

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

def loadModel(checkpoint_path,model, optimizer):
    path = os.path.join(save_dir, checkpoint_path)
    print(f"loadmodel: path: {path}")
    if os.path.isfile(path):
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['opt'])
        print(f"Model {checkpoint_path} successfully loaded")
    else:
        print(f"loadmodel: path: {path} does not exist")


def saveModel(checkpoint_path,model, optimizer):
    path = os.path.join(save_dir, checkpoint_path)
    torch.save({
        'model': model.state_dict(),
        'opt':   optimizer.state_dict()
    }, path)
    print(f"Checkpoint {checkpoint_path} saved.")



def saveModel(filename: str, model: nn.Module):
    """Save model weights to ./model_dumps/<filename>.pth"""
    dump_dir = 'model_dumps'
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"{filename}.pth" if not filename.endswith('.pth') else filename)
    torch.save(model.state_dict(), path)
    if hasattr(model, 'device'):
        loc = getattr(model, 'device')
    else:
        loc = next(model.parameters()).device
    print(f"[saveModel] Saved weights to {path} (device={loc})")


def loadModel(filename: str, model: nn.Module, device: str = 'cpu') -> nn.Module:
    """Load weights into `model` if checkpoint exists, else return unchanged model."""
    dump_dir = 'model_dumps'
    path = os.path.join(dump_dir, f"{filename}.pth" if not filename.endswith('.pth') else filename)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"[loadMode] Loaded weights from {path}")
    else:
        print(f"[loadMode] No checkpoint found at {path} — starting fresh")
    return model

def _rel2abs(cx, cy, w, h, side_size=config.RES):
    cx, cy, w, h = cx * side_size, cy * side_size, w * side_size, h * side_size
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


def _gt_box_from_row(row):
    """Decode (cx,cy,tx,ty,tw,th) row to absolute pixel rectangle."""
    cx = (row.cx + row.tx) / config.S
    cy = (row.cy + row.ty) / config.S
    w  = np.exp(row.tw) * config.ANCHORS[0][0]
    h  = np.exp(row.th) * config.ANCHORS[0][1]
    return _rel2abs(cx, cy, w, h)


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

def drawCropBoxes(crop_rows, crop_img, top_px, left_px, scale, effective_full_size):

    draw = ImageDraw.Draw(crop_img, "L")
    for _, row in crop_rows.iterrows():
        cx_full = (row.cx + row.tx) / config.S
        cy_full = (row.cy + row.ty) / config.S
        w_full = np.exp(row.tw) * config.ANCHORS[0][0]
        h_full = np.exp(row.th) * config.ANCHORS[0][1]
        x0, y0, x1, y1 = _rel2abs(cx_full, cy_full, w_full, h_full, side_size=effective_full_size)
        box = [(x0 - left_px) * scale,
            (y0 - top_px) * scale,
            (x1 - left_px) * scale,
            (y1 - top_px) * scale]
        #draw.rectangle(box, outline=(0, 255, 0, 200), width=2)
        draw.rectangle(box, outline=128, width=2)


    # draw = ImageDraw.Draw(crop_img, "RGBA")
    # for _, row in crop_rows.iterrows():
    #     cx_full = (row.cx + row.tx) / config.S
    #     cy_full = (row.cy + row.ty) / config.S
    #     w_full = np.exp(row.tw) * config.ANCHORS[0][0]
    #     h_full = np.exp(row.th) * config.ANCHORS[0][1]
    #     x0, y0, x1, y1 = _rel2abs(cx_full, cy_full, w_full, h_full)
    #     box = [(x0 - left_px) * scale,
    #         (y0 - top_px) * scale,
    #         (x1 - left_px) * scale,
    #         (y1 - top_px) * scale]
    #     draw.rectangle(box, outline=(0, 255, 0, 200), width=2)



def render_crop_from_dataset(image, target, colour=(0, 255, 0, 200),
                             out_dir="evaluation_crops_from_dataset",
                             name="crop.png"):
    """
    Renders the crop image with ground truth bounding boxes decoded from the target tensor.
    
    Args:
        image (torch.Tensor): The crop image tensor (CHW, float32 0-1).
        target (torch.Tensor): The target tensor (N, N, A, 5 + C).
        colour (tuple): The color for the bounding boxes.
        out_dir (str): Directory to save the rendered image.
        name (str): Filename for the saved image.
    
    Returns:
        PIL.Image: The rendered crop image with bounding boxes.
    """
    # Convert torch image (CHW float 0-1) to PIL (assuming image is square)
    # Convert torch image to PIL (handle grayscale cases)
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
    
    crop_img = Image.fromarray(img_np, mode='L')

    # Draw setup
    draw = ImageDraw.Draw(crop_img, "L")

# Decode and draw boxes from target
    # N, _, A, _ = target.shape
    # for i in range(N):  # cy_loc rows
    #     for j in range(N):  # cx_loc cols
    #         for a in range(A):  # anchors
    #             obj = target[i, j, a, 4].item()
    #             if obj != 1.0:
    #                 continue
    #             tx = target[i, j, a, 0].item()
    #             ty = target[i, j, a, 1].item()
    #             tw = target[i, j, a, 2].item()
    #             th = target[i, j, a, 3].item()
                
    #             # Relative coordinates in crop (0-1)
    #             cx_rel = (j + tx) / N
    #             cy_rel = (i + ty) / N
    #             w_rel = np.exp(tw) * config.ANCHORS[a][0] * (config.S // config.N)
    #             h_rel = np.exp(th) * config.ANCHORS[a][1] * (config.S // config.N)
                
    #             # Absolute pixel corners
    #             x0, y0, x1, y1 = util._rel2abs(cx_rel, cy_rel, w_rel, h_rel)
                
    #             # Draw the box
    #             draw.rectangle([x0, y0, x1, y1], outline=128, width=2)
    # Parallelized box decoding
    N, _, A, _ = target.shape
    side_size = config.RES  # Assuming this is the side size (e.g., 224)

    # Create meshgrid for i, j, a
    i_grid, j_grid, a_grid = torch.meshgrid(
        torch.arange(N, device=target.device),
        torch.arange(N, device=target.device),
        torch.arange(A, device=target.device),
        indexing='ij'
    )

    # Flatten everything
    flat_i = i_grid.reshape(-1)
    flat_j = j_grid.reshape(-1)
    flat_a = a_grid.reshape(-1)
    flat_params = target[..., :5].reshape(-1, 5)  # (N*N*A, 5), where [:,4] is obj

    # Mask for valid boxes
    mask = flat_params[:, 4] == 1.0

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
        draw.rectangle([x0[k].item(), y0[k].item(), x1[k].item(), y1[k].item()], outline=128, width=2)

    # Save and return
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, name)
    crop_img.save(out_path)
    print(f"[render_crop_from_dataset] saved → {out_path}")
    return crop_img