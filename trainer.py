import os
import torch
import config
class Trainer:
    """Simple training loop with logging every quarter epoch."""
    def __init__(self, model, loss_fn, dataloader, epochs, device=None, lr=1e-4):
        self.model = model.to(device or 'cpu')
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device or 'cpu'
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        # log dir
        self.logdir = os.path.join(os.path.dirname(config.filepath), 'logs')
        os.makedirs(self.logdir, exist_ok=True)

    def _log(self, epoch, step, metrics):
        """Append metrics dict to log file."""
        fname = os.path.join(self.logdir, f'epoch_{epoch:03d}.txt')
        with open(fname, 'a') as f:
            msg = f"step {step:05d} " + ' '.join([f"{k}:{v:.4f}" for k, v in metrics.items()]) + '\n'
            f.write(msg)

    def run(self):
        steps_per_epoch = len(self.dataloader)
        log_interval = max(1, steps_per_epoch // 4)  # roughly every quarter epoch
        for epoch in range(1, self.epochs + 1):
            running = {k: 0.0 for k in ['total', 'l_xy', 'l_wh', 'l_obj', 'l_noobj', 'l_cls']}
            for step, (imgs, targets) in enumerate(self.dataloader, 1):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                self.opt.zero_grad()
                preds = self.model(imgs)
                loss, metrics = self.loss_fn(preds, targets)
                loss.backward()
                self.opt.step()
                # accumulate
                for k in running.keys():
                    running[k] += metrics[k].item()
                # periodic log
                if step % log_interval == 0 or step == steps_per_epoch:
                    avg = {k: running[k] / log_interval for k in running.keys()}
                    self._log(epoch, step + (epoch-1)*steps_per_epoch, avg)
                    running = {k: 0.0 for k in running.keys()}
            # epoch summary
            epoch_file = os.path.join(self.logdir, f'epoch_{epoch:03d}.txt')
            with open(epoch_file, 'a') as f:
                f.write('--- end of epoch ---\n')