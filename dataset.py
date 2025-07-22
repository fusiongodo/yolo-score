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


class MyDataset(Dataset):
    """
    PyTorch Dataset for object detection tasks.

    Expects a DataFrame with columns ['img_id','filename','cx','cy','tx','ty','tw','th','class_id']
    and a directory of images.

    Returns:
      image tensor: FloatTensor of shape (3, 896, 896)
      target tensor: FloatTensor of shape (config.S, config.S, config.A, 5 + config.C)
    """
    def __init__(self, gt_df, img_dir = config.img_dir, eval = False):
        self.gt_df = gt_df
        self.img_dir = img_dir
        # Unique image IDs
        if not eval:
            self.img_ids = gt_df['img_id'].unique()
        if eval:
            self.img_ids = gt_df['img_id'].unique()[:192]
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

        ### Resize the image and normalize
        image = image.resize((896, 896))
        arr = np.array(image, dtype=np.float32) / 255.0
        # HWC to CHW
        image = torch.from_numpy(arr).permute(2, 0, 1)

        # Create target tensor using config values directly
        target = torch.zeros(
            (config.S, config.S, config.A, 5 + config.C),
            dtype=torch.float32
        )

        # Populate target per anchor slot
        for _, row in ann.iterrows():
            # row['cx'] is column index, row['cy'] is row index
            i = int(row['cy'])
            j = int(row['cx'])
            # Assign to first free anchor in this cell
            cell_objness = target[i, j, :, 4]
            free_anchors = torch.nonzero(cell_objness == 0).flatten()
            if free_anchors.numel() == 0:
                continue
            a = int(free_anchors[0].item())
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
    




class CroppedDataset(Dataset):
    """
    gt_df is already grouped by "crop_id"
    """
    def __init__(self, gt_df, img_dir = config.img_dir,  eval = False):
        self.gt_df = gt_df
        self.img_dir = img_dir
        # Unique image IDs
        if not eval:
            self.crop_uids = gt_df['crop_uid'].unique()
        if eval:
            self.crop_uids = gt_df['crop_uid'].unique()[:192]

    def __len__(self):
        return len(self.crop_uids)

    def __getitem__(self, idx):
        crop_uid = self.crop_uids[idx]
        ann = self.gt_df[self.gt_df.crop_uid == crop_uid]
        crop_row = int(ann.crop_row.iloc[0])
        crop_col = int(ann.crop_col.iloc[0])

        # Load Crop
        img_path = os.path.join(self.img_dir, ann['filename'].iloc[0])
        try:
            crop_img, left_px, top_px, scale, effective_full_size = util.load_crop_image(img_path, crop_row, crop_col)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
        arr = np.array(crop_img, dtype=np.float32) / 255.0
        # HWC to CHW
        image = torch.from_numpy(arr).permute(0, 1)

        # Create target tensor using config values directly
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
    

    
class CroppedDummyset(Dataset):
    """
    gt_df is already grouped by "crop_id"
    """
    def __init__(self, gt_df, img_dir = config.img_dir):
        self.gt_df = gt_df
        self.img_dir = img_dir
        # Unique image IDs
        if not eval:
            self.crop_uids = gt_df['crop_uid'].unique()
        if eval:
            self.crop_uids = gt_df['crop_uid'].unique()[:192]

    def __len__(self):
        return len(self.crop_uids)
    

    def dummyTensor(self, choice):#zero, one, half
        shape = (config.N, config.N, config.A, 5 + config.C)
        target = torch.zeros(shape, dtype=torch.float32)
        if choice == 'one':
            target.fill_(1.0)
        elif choice == 'half':
            half = config.N // 2
            target[:, :half, :, :] = 1.0  # Horizontal split along symmetry plane (top half 1s, bottom 0s; adjust as needed)
        # 'zero' is already handled by initialization
        return target

    def __getitem__(self, idx):
        crop_uid = self.crop_uids[idx]
        ann = self.gt_df[self.gt_df.crop_uid == crop_uid]
        crop_row = int(ann.crop_row.iloc[0])
        crop_col = int(ann.crop_col.iloc[0])

        # Load Crop
        img_path = os.path.join(self.img_dir, ann['filename'].iloc[0])
        try:
            crop_img, left_px, top_px, scale, effective_full_size = util.load_crop_image(img_path, crop_row, crop_col)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
        arr = np.array(crop_img, dtype=np.float32) / 255.0
        # HWC to CHW
        image = torch.from_numpy(arr).permute(0, 1)

        target = self.dummyTensor(choice = "one")

        return image, target
    




from torch.utils.data import DataLoader
import torch
import config

class DummyDataLoader(DataLoader):
    def __init__(self, dummyset, choice="zero", **kwargs):
        super().__init__(dummyset, **kwargs)
        self.choice = choice
        self.dummyset = dummyset

    def __iter__(self):
        for batch in super().__iter__():
            images, _ = batch
            dummy_target = self.dummyset.dummyTensor(self.choice)
            targets = torch.stack([dummy_target.clone() for _ in range(len(images))])
            yield images, targets
