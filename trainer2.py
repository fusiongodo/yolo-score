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
        self.loss_fn = l.YOLOv2Loss(l_xy = d["c_xy"], l_wh=d["c_wh"], l_obj=d["c_obj"], l_noobj=d["c_noobj"], l_cls=d["c_cls"])

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
        mAP = eval.avg_precision_recall(self.model, self.eval_dataset, device=self.device, n_samples=100)
        end = time.time()
        
        print(f"Elapsed: {end - start:.4f} seconds")
        
        r = Record(self.rec_counter, epoch, mAP, self.learn_config, self.lossRecord)
        self.modelseries.addRecord(r)
        self.lossRecord = LossRecord()
        self.rec_counter = 0

    def keyboardInterrupt(self):
        self.addRecord(self.current_epoch)
        self.modelseries.saveJsonData()
        self.modelseries.saveCheckpoint(self.model)

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
                pred = self.model(image.unsqueeze(0))  # [1, 1, H, W] -> model -> [1, N, N, A, 5+C]
                pred = pred.squeeze(0)  # [N, N, A, 5+C]

            pred = eval.logit_to_target(pred)

            iou = eval.pred_to_iou(pred, target)
            mAP = f"{eval.unit_iou_precision(pred, target, iou):.4f}"
            dir = os.path.join(self.modelseries.series_dir, "predictions", f"{self.modelseries.getEpoch()}")
            util.render_crop_from_dataset(image, pred, out_dir = dir, name = f"crop_{i}_{mAP}.png")
        self.model.train()

    def run(self, num_workers = 0):
        switch_debug = True
        
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            self.current_epoch, start = epoch, time.time() # needed for keyboardInterrupt
            print(f"trainer.run(): Epoch {self.current_epoch}")
            for imgs, targets in self.train_loader:
                imgs = imgs.unsqueeze(1) #in dataset verlegen?
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                self.opt.zero_grad()
                loss, loss_dict = self.loss_fn(self.model(imgs), targets)
                loss.backward(); self.opt.step()

                self.rec_counter += self.train_loader.batch_size
                self.index_counter += self.train_loader.batch_size
                self.lossRecord.addLossDictionary(loss_dict)
            if switch_debug:
                self.modelseries.addCheckpoint(self.model)
                print("debug: addCheckpoint()")
                switch_debug = False
                

                        
            end = time.time()
            print(f"Epoch duration without mAP and prediction rendering: {end - start:.4f} seconds")
            self.addRecord(epoch)
            self.visualize()
            if((epoch - self.start_epoch + 1) % self.checkpoint_rate == 0):
                print(f"epoch {epoch}/{self.n_epochs + self.start_epoch} checkpoint added at checkpoint_rate{self.checkpoint_rate}")
                self.modelseries.addCheckpoint(self.model)
            
        #keep current checkpoint, overwrite current model weights
        self.modelseries.addCheckpoint(self.model)


            
 
            
        


