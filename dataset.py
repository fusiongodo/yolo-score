import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import config

class ObjectDetectionDataset(Dataset):
    """
    PyTorch Dataset for object detection tasks.

    Expects a DataFrame with columns ['img_id','filename','cx','cy','tx','ty','tw','th','class_id']
    and a directory of images.

    Returns:
      image tensor: FloatTensor of shape (3, 896, 896)
      target tensor: FloatTensor of shape (config.S, config.S, config.A, 5 + config.C)
    """
    def __init__(self, gt_df, img_dir):
        self.gt_df = gt_df
        self.img_dir = img_dir
        # Unique image IDs
        self.img_ids = gt_df['img_id'].unique()
        # Group annotations by img_id for fast lookup
        self.grouped = gt_df.groupby('img_id')

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann = self.grouped.get_group(img_id)

        # Load image
        img_path = os.path.join(self.img_dir, ann['filename'].iloc[0])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")


        image = image.resize((896, 896))
        arr = np.array(image, dtype=np.float32) / 255.0
        # HWC to CWH 
        image = torch.from_numpy(arr).permute(2, 1, 0)

 
        target = torch.zeros(
            (config.S, config.S, config.A, 5 + config.C),
            dtype=torch.float32
        )

        # img_id        int32
        # filename     object
        # cx            int32
        # cy            int32
        # tx          float32
        # ty          float32
        # tw          float32
        # th          float32
        # class_id      int32

        for _, row in ann.iterrows():
            i = int(row['cx'])
            j = int(row['cy'])
            # Assign to the first available anchor slot in this cell
            cell_objness = target[i, j, :, 4]
            # find anchors where objectness == 0
            free_anchors = torch.nonzero(cell_objness == 0).flatten()
            if free_anchors.numel() == 0:
            # No free anchor slots left for this cell, skip this box
                continue
            a = int(free_anchors[0].item())
            # Bounding‐box offsets and scales
            target[i, j, a, 0] = row['tx']
            target[i, j, a, 1] = row['ty']
            target[i, j, a, 2] = row['tw']
            target[i, j, a, 3] = row['th']
            # Objectness
            target[i, j, a, 4] = 1.0
            # One‐hot class
            cls = int(row['class_id'])
            if cls < config.C:
                target[i, j, a, 5 + cls] = 1.0

        return image, target
