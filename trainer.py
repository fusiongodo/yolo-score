import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import time
from torch.utils.data import DataLoader

class Trainer:
    """Training loop that logs to one file and saves checkpoints every
    *save_interval* epochs (caller still handles Ctrl-C).
    """

    def __init__(self, model, loss_fn, dataloader, epochs, modelname='run',
                 device=None, lr=1e-4, verbose=True, save_interval=1):
        self.model = model.to(device or 'cpu')
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device or 'cpu'
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.modelname = modelname
        self.save_interval = max(1, int(save_interval))
        # logging
        self.logdir = getattr(config, 'logdir', os.path.join(os.path.dirname(config.filepath), 'logs'))
        os.makedirs(self.logdir, exist_ok=True)
        self.logfile = os.path.join(self.logdir, f'{modelname}_log.txt')
        with open(self.logfile, 'w') as f:
            f.write('# step total l_xy l_wh l_obj l_noobj l_cls\n')
        self.verbose = verbose

    # ------------------------------------------------------------------
    def _log(self, step_global: int, metrics: dict):
        msg = f"{step_global:06d} " + ' '.join([f"{k}:{v:.4f}" for k, v in metrics.items()])
        with open(self.logfile, 'a') as f:
            f.write(msg + "\n")
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    def run(self):
        import time  # local import to avoid unused when not training
        steps_per_epoch = len(self.dataloader)
        log_interval = max(1, steps_per_epoch // 2)
        global_step = 0
        for epoch in range(1, self.epochs + 1):
            running = {k: 0.0 for k in ['total', 'l_xy', 'l_wh', 'l_obj', 'l_noobj', 'l_cls']}
            epoch_totals = {k: 0.0 for k in running}
            since_last = 0
            for step, (imgs, targets) in enumerate(self.dataloader, 1):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                self.opt.zero_grad()
                preds = self.model(imgs)
                loss, metrics = self.loss_fn(preds, targets)
                loss.backward()
                self.opt.step()
                for k in running:
                    running[k] += metrics[k].item()
                    epoch_totals[k] += metrics[k].item()
                since_last += 1
                global_step += 1
                if step % log_interval == 0 or step == steps_per_epoch:
                    avg = {k: running[k] / since_last for k in running}
                    self._log(global_step, avg)
                    running = {k: 0.0 for k in running}; since_last = 0
            # epoch average
            avg_epoch = {k: epoch_totals[k] / steps_per_epoch for k in epoch_totals}
            epoch_msg = f'Epoch {epoch} avg ' + ' '.join([f"{k}:{v:.4f}" for k, v in avg_epoch.items()])
            with open(self.logfile, 'a') as f:
                f.write(epoch_msg + "\n")
            if self.verbose:
                print(epoch_msg)
            # checkpoint every save_interval epochs
            if epoch % self.save_interval == 0:
                ts = time.strftime('%d%m%H%M')  # ddmmHHMM
                ckpt_name = f"{self.modelname}_ep{epoch}_{ts}"
                saveModel(ckpt_name, self.model)
                if self.verbose:
                    print(f"[Trainer] Saved checkpoint {ckpt_name}.pth")

# ------------------------------------------------------------------
# Model checkpoint helpers
# ------------------------------------------------------------------

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
        print(f"[loadModel] Loaded weights from {path}")
    else:
        print(f"[loadModel] No checkpoint found at {path} — starting fresh")
        return model
