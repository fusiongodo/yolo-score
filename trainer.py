
class Trainer:
    """Training loop that logs to one file per run. Model saving is *not* handled
    internally; caller controls checkpoints and KeyboardInterrupt handling.
    """

    def __init__(self, model, loss_fn, dataloader, epochs, modelname='run',
                 device=None, lr=1e-4, verbose=True):
        self.model = model.to(device or 'cpu')
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device or 'cpu'
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        # logging ------------------------------------------------------
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
        steps_per_epoch = len(self.dataloader)
        log_interval = max(1, steps_per_epoch // 10)
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
            avg_epoch = {k: epoch_totals[k] / steps_per_epoch for k in epoch_totals}
            epoch_msg = f'Epoch {epoch} avg ' + ' '.join([f"{k}:{v:.4f}" for k, v in avg_epoch.items()])
            with open(self.logfile, 'a') as f:
                f.write(epoch_msg + "\n")
            if self.verbose:
                print(epoch_msg)
