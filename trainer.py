class Trainer:
    """Simple training loop with console + file logging every quarter‑epoch."""

    def __init__(self, model, loss_fn, dataloader, epochs, device=None, lr=1e-4, verbose=True):
        self.model = model.to(device or 'cpu')
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device or 'cpu'
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        # log dir strictly <config.logdir>
        self.logdir = getattr(config, 'logdir', os.path.join(os.path.dirname(config.filepath), 'logs'))
        os.makedirs(self.logdir, exist_ok=True)
        self.verbose = verbose

    # ------------------------------------------------------------------
    def _log(self, epoch: int, step_global: int, metrics: dict):
        """Append metrics dict to log file and optionally print to console."""
        fname = os.path.join(self.logdir, f'epoch_{epoch:03d}.txt')
        msg = f"step {step_global:05d} " + ' '.join([f"{k}:{v:.4f}" for k, v in metrics.items()])
        with open(fname, 'a') as f:
            f.write(msg + "\n")
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    def run(self):
        steps_per_epoch = len(self.dataloader)
        log_interval = max(1, steps_per_epoch // 20)  # roughly every quarter epoch
        for epoch in range(1, self.epochs + 1):
            running = {k: 0.0 for k in ['total', 'l_xy', 'l_wh', 'l_obj', 'l_noobj', 'l_cls']}
            since_last = 0  # number of steps accumulated since last log
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
                since_last += 1
                # periodic log
                if step % log_interval == 0 or step == steps_per_epoch:
                    avg = {k: running[k] / since_last for k in running.keys()}
                    global_step = (epoch - 1) * steps_per_epoch + step
                    self._log(epoch, global_step, avg)
                    running = {k: 0.0 for k in running.keys()}
                    since_last = 0
            # epoch summary marker
            with open(os.path.join(self.logdir, f'epoch_{epoch:03d}.txt'), 'a') as f:
                f.write('--- end of epoch ---\n')
            if self.verbose:
                print(f'Epoch {epoch} finished')