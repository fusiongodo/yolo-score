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
from dataset import CroppedDataset
from ModelSeries import LossRecord, LearnConfig, Record, ModelSeries
import config as c
import util








class Trainer:

    def __init__(self, modelseries, learn_config : LearnConfig, epochs, checkpoint_rate, num_workers, batch_size ):
        
        # --------------- core ----------------
        self.model = modelseries.model.to("cuda")
        self.modelseries : ModelSeries = modelseries
        self.testConfig() #test if moderseries config aligns with config.py

        d = learn_config.toDict()
        self.loss_fn = l.YOLOv2Loss(l_xy = d["c_xy"], l_wh=d["c_wh"], l_obj=d["c_obj"], l_noobj=d["c_noobj"], l_cls=d["c_cls"], iou_obj = d["iou_obj"])

        self.learn_config = learn_config #loss paramters and lr
        self.checkpoint_rate = checkpoint_rate
        self.batch_size = batch_size

        self.device = "cuda"
        self.opt = torch.optim.Adam(self.model.parameters(), lr=learn_config.toDict()["c_lr"])


        

        de = util.DataExtractor()
        gt_df = de.croppedData()
   
        self.eval_dataset = CroppedDataset(gt_df, mode = "val")
        self.train_dataset = CroppedDataset(gt_df, mode = "train")
   
       
        self.eval_loader  = DataLoader(self.eval_dataset, batch_size=self.batch_size,
                                       shuffle=False, num_workers=num_workers, pin_memory=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, 
                                       shuffle=True,  num_workers=num_workers, )
        
        self.rec_counter = 0 
        self.index_counter = 0
        self.lossRecord = LossRecord()

        
        self.epochs = epochs
        self.n_epochs = epochs
        self.start_epoch = self.modelseries.getEpoch()
        self.current_epoch = None


    
    def testConfig(self):
        assert c.S == self.modelseries.S, "config.S unequal to ModelSeries S parameter"
        assert c.N == self.modelseries.N, "config.N unequal to ModelSeries N parameter"
        assert c.A == self.modelseries.A, "config.A unequal to ModelSeries A parameter"
        assert c.RES == self.modelseries.RES, "config.RES unequal to ModelSeries RES parameter"

    def addRecord(self, epoch):
        start = time.time()
        mREC, mAP, (tp, fp, fn), (precision_per_cls, recall_per_cls) = eval.avg_precision_recall(self.model, self.eval_dataset, device=self.device, n_samples=100)
        end = time.time()
        print(f"Elapsed: {end - start:.4f} seconds")
        
        r = Record(self.rec_counter, epoch, mAP, mREC, self.learn_config, self.lossRecord)
        self.modelseries.addRecord(r)
        self.modelseries.eval_records.addEvalRecord(precision_per_cls, recall_per_cls, self.current_epoch)
        self.lossRecord = LossRecord()
        self.rec_counter = 0

    def keyboardInterrupt(self):
        mREC, mAP, (tp, fp, fn), (precision_per_cls, recall_per_cls) = eval.avg_precision_recall(self.model, self.eval_dataset, device=self.device, n_samples=100)
        self.modelseries.eval_records.addEvalRecord(precision_per_cls, recall_per_cls, self.current_epoch)
        self.addRecord(self.current_epoch)
        self.modelseries.saveJsonData()
        self.modelseries.addCheckpoint(self.model)

    def visualize(self):
        self.model.eval()
        #(image, target, colour=(0, 255, 0, 200), obj_thres = 0.5, out_dir="evaluation_crops_from_dataset", name="crop.png"):
        for i in range(20):
            idx = i * 200
            try:
                image, target = self.eval_dataset[idx % len(self.eval_dataset)]
                image, target = image.to("cuda"), target.to("cuda")
            except Exception:
                continue
            image = image.unsqueeze(0)  # [1, H, W]

            with torch.no_grad():
                pred = self.model(image)  # [1, 1, H, W] -> model -> [1, N, N, A, 5+C]
                pred = pred.squeeze(0)  # [N, N, A, 5+C]

            pred = eval.logit_to_target(pred)

            iou = eval.pred_to_iou(pred, target)
            tp_mask, fp_mask, fn_mask, tgt_conf, pred_cls, tgt_cls = eval.calc_masks(pred, target, iou, conf_thr=0.5)
            precision, recall = eval.unit_precision_recall(tp_mask, fp_mask, fn_mask)

            iou = eval.pred_to_iou(pred, target)
            mAP = f"{precision:.4f}"
            mREC = f"{recall:.4f}"
            dir = os.path.join(self.modelseries.series_dir, "predictions", f"{self.modelseries.getEpoch()}")
            rendered = util.render_prediction(image.squeeze(0), pred, iou, out_dir = dir, name = f"crop_{i}_{mAP}_{mREC}.png", obj_thresh=0.01)
        self.model.train()

    def run(self, num_workers = 0):
        
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            self.current_epoch = epoch
            for imgs, targets in self.train_loader:
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                self.opt.zero_grad()
                loss, loss_dict = self.loss_fn(self.model(imgs), targets)
                loss.backward(); self.opt.step()

                self.rec_counter += self.train_loader.batch_size
                self.index_counter += self.train_loader.batch_size
                self.lossRecord.addLossDictionary(loss_dict)

 
            self.addRecord(epoch)
            self.visualize()
            if((epoch - self.start_epoch + 1) % self.checkpoint_rate == 0):
                print(f"epoch {epoch}/{self.n_epochs + self.start_epoch} checkpoint added at checkpoint_rate{self.checkpoint_rate}")
                self.modelseries.addCheckpoint(self.model)
            
        #keep current checkpoint, overwrite current model weights
        self.modelseries.addCheckpoint(self.model)


            
 
            
        


