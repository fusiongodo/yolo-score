import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from importlib import reload
import config
import util
import random

reload(util)
reload(config)





class CroppedDataset(Dataset):
    """
    gt_df is already grouped by "crop_id"
    """
    def __init__(self, gt_df,  type = "train"):

        self.gt_df = gt_df
        self.img_dir = config.img_dir

        
        
        self.crop_uids = gt_df['crop_uid'].unique()
        n = len(self.crop_uids)
        n_train, n_val, n_test = int(0.8 * n), int(0.1 * n), int(0.1 * n)
        if type == "train":
            self.crop_uids = self.crop_uids[:n_train]
        elif type == "val":
            self.crop_uids = self.crop_uids[n_train : (n_train + n_val)]
        elif type == "test":
            self.crop_uids = self.crop_uids[(n_train + n_val) : ]

    def __len__(self):
        return len(self.crop_uids)

    def __getitem__(self, idx):
        crop_uid = self.crop_uids[idx]
        ann = self.gt_df[self.gt_df.crop_uid == crop_uid].copy()
        crop_row = int(ann.crop_row.iloc[0])
        crop_col = int(ann.crop_col.iloc[0])


        img_path = os.path.join(self.img_dir, ann['filename'].iloc[0])
        try:
            crop_img, left_px, top_px, scale, effective_full_size = util.load_crop_image(img_path, crop_row, crop_col)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
        arr = np.array(crop_img, dtype=np.float32) / 255.0
        # HWC to CHW
        image = torch.from_numpy(arr).permute(0, 1)
        


        target = torch.zeros(
            (config.N, config.N, config.A, 5 + config.C),
            dtype=torch.float32
        )


        # Populate target per anchor slot
        for _, row in ann.iterrows():
            # row['cx'] is column index, row['cy'] is row index
            i = int(row['cy_loc'])
            j = int(row['cx_loc'])
            # Assign to first free anchor in this cell
            cell_objness = target[i, j, :, 4]  #torch.tensor([0., 1., 0., 1., 0.])   # shape (5,)
            free_anchors = torch.nonzero(cell_objness == 0).flatten() #at index 0,2,4
            if free_anchors.numel() == 0:
                continue
            a = int(free_anchors[0].item()) ##next/first free anchor
            # Bounding‑box offsets and scales
            target[i, j, a, 0] = row['tx']
            target[i, j, a, 1] = row['ty']
            target[i, j, a, 2] = row['tw']
            target[i, j, a, 3] = row['th']
            # Objectness
            target[i, j, a, 4] = 1.0
            # One‑hot class
            cls = int(row['class_id'])
            if cls < config.C:
                target[i, j, a, 5 + cls] = 1.0

        return image, target
    
