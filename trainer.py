import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import time
from importlib import import_module
from torch.utils.data import DataLoader
import loss as l
import eval
import time
import util
from dataset import MyDataset, CroppedDataset


class Trainer:
    """Training loop with logging, periodic mAP evaluation and checkpoints.

    Every *save_interval* epochs the model weights are saved.  Every second
    epoch (2, 4, 6, …) an evaluation dataset (created with `eval=True`) is
    run through `average_precision`.
    """

    def __init__(self, model, loss_fn, gt_df, dataloader, epochs,
                 modelname='run', device=None, lr=1e-4, verbose=True,
                 save_interval=1, eval_batch_size=4):
        
        # --------------- core ----------------
        self.model = model.to(device or 'cpu')
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device or 'cpu'
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.modelname = modelname
        self.save_interval = max(1, int(save_interval))
        self.verbose = verbose


        self.eval_dataset = CroppedDataset(gt_df,  mode = "val")
        self.eval_loader  = DataLoader(self.eval_dataset, batch_size=eval_batch_size,
                                       shuffle=False, num_workers=2, pin_memory=True)
        # --------------- logging ----------------
        self.logdir = getattr(config, 'logdir', os.path.join(os.path.dirname(config.filepath), 'logs'))
        os.makedirs(self.logdir, exist_ok=True)
        self.logfile = os.path.join(self.logdir, f'{modelname}_log.txt')
        with open(self.logfile, 'w') as f:
            f.write('# step total l_xy l_wh l_obj l_noobj l_cls mAP\n')

    # ------------------------------------------------------------------
    def _log_epoch(self, epoch_msg: str):
            with open(self.logfile, 'a') as f:
                f.write(epoch_msg + "\n")
            if self.verbose:
                print(epoch_msg)

    # ------------------------------------------------------------------
    def run(self):
        counter = 0 
        steps_per_epoch = len(self.dataloader)
        for epoch in range(1, self.epochs + 1):
            epoch_totals = {k: 0.0 for k in ['total','l_xy','l_wh','l_obj','l_noobj','l_cls']}
            for imgs, targets in self.dataloader:
                imgs = imgs.unsqueeze(1)
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                self.opt.zero_grad()
                loss, metrics = self.loss_fn(self.model(imgs), targets)
                loss.backward(); self.opt.step()
                counter += self.dataloader.batch_size
                if counter // 2000 == 0:
                    print("2000 crops processed")
                    mAP = eval.average_precision(self.model, self.eval_dataset, device=self.device, n_samples=100)
                    print(mAP)
                    counter -= 2000
                for k in epoch_totals:
                    epoch_totals[k] += metrics[k]
            # ----- mAP on eval subset -----
            mAP = eval.average_precision(self.model, self.eval_dataset, device=self.device, n_samples=100)
            avg = {k: epoch_totals[k] / steps_per_epoch for k in epoch_totals}
            epoch_line = (f"Epoch {epoch} mAP:{mAP:e}" + ' '.join([f"{k}:{avg[k]:.4f}" for k in ['total','l_xy','l_wh','l_obj','l_noobj','l_cls']]))
            self._log_epoch(epoch_line)
            # checkpointing
            if epoch % self.save_interval == 0:
                ts = time.strftime('%d%m%H%M')
                ckpt_name = f"{self.modelname}_ep{epoch}_{ts}"
                util.saveModel(ckpt_name, self.model)
                if self.verbose:
                    print(f"[Trainer] Saved checkpoint {ckpt_name}.pth")
        


