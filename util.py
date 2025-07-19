import torch
import os
import json
import pandas as pd
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
    
    def croppedData(self, grouped: bool = False):
        """
        Split the S×S grid into N×N crops and assign every bbox to its crop.

        Added columns
        -------------
        crop_id      : flat index 0 … N²-1
        crop_row/col : 2-D crop position inside the N×N crop grid
        cx_loc/cy_loc: cell indices inside the crop-local m×m grid,
                       where m = S // N

        Other regression targets (tx, ty, tw, th) stay unchanged.
        """
        # --- start from the global (S×S) label table ---
        df = self.normalizedData(grouped=False).copy()

        S, N = config.S, config.N
        if S % N:
            raise ValueError(f"config.N ({N}) must evenly divide config.S ({S})")
        m = S // N                                     # cells per crop edge

        # --- determine crop indices -------------------------------------------------
        df['crop_row'] = (df['cy'] // m).astype(int)   # 0 … N-1
        df['crop_col'] = (df['cx'] // m).astype(int)
        df['crop_id']  = df['crop_row'] * N + df['crop_col']

        # --- cell coordinates inside the crop grid ----------------------------------
        df['cx_loc'] = (df['cx'] % m).astype(int)
        df['cy_loc'] = (df['cy'] % m).astype(int)

        # --- tidy column order (keep globals for reference) -------------------------
        col_order = [
            'img_id', 'crop_id', 'cx_loc', 'cy_loc',    # crop-local info
            'tx', 'ty', 'tw', 'th',                    # regression targets
            'class_id', 'filename',                    # meta
            'cx', 'cy', 'crop_row', 'crop_col'         # global refs (optional)
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
            df = df.groupby(['img_id', 'crop_id'])

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

