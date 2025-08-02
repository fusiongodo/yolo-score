import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from importlib import reload
import config
import util
import random
import time

reload(util)
reload(config)





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
    def __init__(self, gt_df,  mode = "train"):

        self.gt_df = gt_df
        self.img_dir = config.img_dir

        
        
        self.crop_uids = gt_df['crop_uid'].unique()
        n = len(self.crop_uids)
        n_train, n_val, n_test = int(0.8 * n), int(0.1 * n), int(0.1 * n)
        if mode == "train":
            self.crop_uids = self.crop_uids[:n_train]
        elif mode == "val":
            self.crop_uids = self.crop_uids[n_train : (n_train + n_val)]
        elif mode == "test":
            self.crop_uids = self.crop_uids[(n_train + n_val) : ]
        print(f"{mode}-dataset created containing {len(self.crop_uids)} crops")

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
    

class CroppedPreloadedDataset(Dataset):
    def __init__(self, gt_df,  mode = "train", val_gt = None, train_gt = None):

        self.gt_df = gt_df
        self.img_dir = config.img_dir
        self.mode = mode

        
        
        self.crop_uids = gt_df['crop_uid'].unique()
        n = len(self.crop_uids)
        n_train, n_val, n_test = int(0.8 * n), int(0.1 * n), int(0.1 * n)
        if mode == "train":
            self.crop_uids = self.crop_uids[:n_train]
        elif mode == "val":
            self.crop_uids = self.crop_uids[n_train : (n_train + n_val)]
        elif mode == "test":
            self.crop_uids = self.crop_uids[(n_train + n_val) : ]

        self.images = []
        self.targets = []
        if (mode == "train" and train_gt) or (mode == "val" and val_gt):
            if val_gt:
                self.images, self.targets = val_gt[0], val_gt[1]
            elif train_gt:
                self.images, self.targets = train_gt[0], train_gt[1]
        else:
            d_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preloaded_data", f"{self.mode}")
            save_path =  os.path.join(d_dir, "preloaded_dataset.pt")

            if not (os.path.exists(d_dir)):          
                print(f"number of image, target pairs: {len(self.crop_uids)}")           
                os.makedirs(os.path.join(d_dir), exist_ok=True)
                start = time.time()
                counter = 0
                for idx in range(len(self)):
                    counter += 1
                    if counter % 100 == 0:
                        print(f"{self.mode}_dataset: 100 elements loaded into RAM")
                    img, tgt = self._load_item(idx)
                    self.images.append(img)
                    self.targets.append(tgt)
                end = time.time()
                print(f"dataset.py: All images saved to disk within {end - start:.4f} seconds\n Save Images.")

                torch.save((self.images, self.targets), save_path)
                print(f"dataset.py: images and targets saved at {d_dir}")
            else:
                start = time.time()
                print(f"Load images and targets from {os.path.join(d_dir, 'preloaded_dataset.pt')}")
                self.images, self.targets = torch.load(save_path)
                end = time.time()
                print(f"dataset.py: All images loaded into RAM within {end - start:.4f} seconds\n Save Images.")

        
        
    def __len__(self):
        return len(self.crop_uids)

    def _load_item(self, idx):
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

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


    





    
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
    

    def dummyTensor(self, choice: str = "zero", *, seed: int = 0) -> torch.Tensor:
        """
        Return a (N, N, A, 5+C) tensor with hand-crafted objectness patterns.

        Choices
        -------
        zero           : all zeros  (already the default)
        one            : all ones
        half           : top half = 1
        inverse_half   : bottom half = 1     (new)
        vertical_half  : **left** half = 1   (fixed)
        quarter        : top-left quarter = 1
        checker        : checkerboard pattern (50 % coverage)
        random25       : ~25 % cells = 1, deterministic via `seed`
        """
        N, A, C = config.N, config.A, config.C
        tgt = torch.zeros((N, N, A, 5 + C), dtype=torch.float32)

        def _fill_cell(t, i, j):
            t[i, j, :, 4] = 1.0          # objectness
            t[i, j, :, 0:2] = 0.5        # centre at cell-centre
            t[i, j, :, 2:4] = -10.0      # log-w/h  →  exp(-10) ≈ 4.5e-5
            t[i, j, :, 5] = 1.0          # class-0 one-hot
        # ------------------------------------------------------------------ #

        if choice == "one":                      # every cell
            for i in range(N):
                for j in range(N):
                    _fill_cell(tgt, i, j)

        elif choice == "half":                   # top half
            for i in range(N // 2):
                for j in range(N):
                    _fill_cell(tgt, i, j)

        elif choice == "inverse_half":           # bottom half
            for i in range(N // 2, N):
                for j in range(N):
                    _fill_cell(tgt, i, j)

        elif choice == "vertical_half":          # left half
            for i in range(N):
                for j in range(N // 2):
                    _fill_cell(tgt, i, j)

        elif choice == "quarter":                # top-left quarter
            for i in range(N // 2):
                for j in range(N // 2):
                    _fill_cell(tgt, i, j)

        elif choice == "checker":                # every-other cell
            for i in range(N):
                for j in range(N):
                    if (i + j) % 2 == 0:
                        _fill_cell(tgt, i, j)

        elif choice == "random25":
            g = torch.Generator().manual_seed(seed)
            mask = torch.rand((N, N), generator=g) < 0.25
            idx = mask.nonzero(as_tuple=False)
            for i, j in idx:
                _fill_cell(tgt, int(i), int(j))

        # 'zero' already all-zeros
        return tgt


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
