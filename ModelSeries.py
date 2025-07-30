import pandas as pd
import config as c
import os
import model as m
import util
import json
import IPython.display as pdis
from importlib import reload
reload(util)
reload(c)

    
    
    
    
class LearnConfig:
    def __init__(self, c_xy, c_wh, c_obj, c_noobj, c_cls, c_lr):
        self.c_xy = c_xy
        self.c_wh = c_wh
        self.c_obj = c_obj
        self.c_noobj = c_noobj
        self.c_cls = c_cls
        self.c_lr = c_lr
    
    def toDict(self):
        return {
            "c_xy": self.c_xy,
            "c_wh": self.c_wh,
            "c_obj": self.c_obj,
            "c_noobj": self.c_noobj,
            "c_cls": self.c_cls,
            "c_lr" : self.c_lr
        }
    

class LossRecord:
    def __init__(self):
        self.l_total = []
        self.l_xy = []
        self.l_wh = []
        self.l_obj = []
        self.l_noobj = []
        self.l_cls = []

    def addLossDictionary(self, d):
        self.addLoss(d["total"], d["l_xy"], d["l_wh"], d["l_obj"], d["l_noobj"], d["l_cls"])

    def addLoss(self, total, xy, wh, obj, noobj, cls):
        self.l_total.append(total)
        self.l_xy.append(xy)
        self.l_wh.append(wh)
        self.l_obj.append(obj)
        self.l_noobj.append(noobj)
        self.l_cls.append(cls)


    
    def toDict(self):
        return {
            "l_total": sum(self.l_total) / len(self.l_total),
            "l_xy": sum(self.l_xy) / len(self.l_xy),
            "l_wh": sum(self.l_wh) / len(self.l_wh),
            "l_obj": sum(self.l_obj) / len(self.l_obj),
            "l_noobj": sum(self.l_noobj) / len(self.l_noobj),
            "l_cls": sum(self.l_cls) / len(self.l_cls),
        }
    
class Record:
    def __init__(self, n_crops, epoch, mAP, learn_configs : LearnConfig, losses : LossRecord): 
        
        print(f"Record.__init__: {n_crops} images processed")
        losses = losses.toDict()
        learn_configs = learn_configs.toDict()
        self.checkpoint_idx = 0
        self.n_crops = n_crops
        self.epoch = epoch
        self.mAP = mAP
        self.c_xy = learn_configs["c_xy"]
        self.c_wh = learn_configs["c_wh"]
        self.c_obj = learn_configs["c_obj"]
        self.c_noobj = learn_configs["c_noobj"]
        self.c_cls = learn_configs["c_cls"]
        self.c_lr = learn_configs["c_lr"]
        self.l_total = losses["l_total"]
        self.l_xy = losses["l_xy"]
        self.l_wh = losses["l_wh"]
        self.l_obj = losses["l_obj"]
        self.l_noobj = losses["l_noobj"]
        self.l_cls = losses["l_cls"]

    def toDict(self):
        return {
            "checkpoint_idx": 0,
            "n_crops": self.n_crops,
            "epoch" : self.epoch,
            "mAP": self.mAP,
            "c_xy": self.c_xy,
            "c_wh": self.c_wh,
            "c_obj": self.c_obj,
            "c_noobj": self.c_noobj,
            "c_cls": self.c_cls,
            "c_lr" : self.c_lr,
            "l_total": self.l_total,
            "l_xy": self.l_xy,
            "l_wh": self.l_wh,
            "l_obj": self.l_obj,
            "l_noobj": self.l_noobj,
            "l_cls": self.l_cls
        }

class ModelSeries:
    def __init__(self, name, model = m.YOLOResNet(), model_descr = "not provided", mode = ""):
        self.name = name
        self.model = model
        self.model_descr = model_descr#nur bei erstmaligem erstellen nötig
        #self.series_dir = os.path.join(c.models, self.name)
        self.S = c.S
        self.N = c.N
        self.A = c.A
        self.RES = c.RES
        self.series_dir = os.path.join(c.models, f"{self.S}@{self.N}@{self.A}@{self.RES}", self.name)
        self.checkpoint = 0
        self.mode = mode
        columns = "checkpoint_idx, n_crops, epoch, mAP, c_lr, c_xy, c_wh, c_obj, c_noobj, c_cls, l_total, l_xy, l_wh, l_obj, l_noobj, l_cls".split(", ")
        # Define data types for each column
        dtypes = {
            "checkpoint_idx": "Int64",  # Nullable integer for checkpoint index
            "n_crops": "Int64",        # Nullable integer for crop count
            "epoch": "Int64",          # Nullable integer for epoch
            "mAP": "float64",          # Float for mean average precision
            "c_lr": "float64",         # Float for learning rate
            "c_xy": "float64",         # Float for loss coefficients
            "c_wh": "float64",
            "c_obj": "float64",
            "c_noobj": "float64",
            "c_cls": "float64",
            "l_total": "float64",      # Float for loss values
            "l_xy": "float64",
            "l_wh": "float64",
            "l_obj": "float64",
            "l_noobj": "float64",
            "l_cls": "float64"
        }
        self.records = pd.DataFrame(columns = columns).astype(dtypes)
        if(os.path.exists(self.series_dir)):
            try:
                self.loadJsonData()
                print("ModelSeries: loadJsonData()")
                if mode == "profiling":
                    pdis.display(pdis.HTML(self.records.iloc[-5:].to_html()))
                    
            except Exception:
                print("ModelSeries: loadjsonData() Error")
            
            if mode == "training":
                try:
                    self.model = self.loadLatestCheckpoint(self.model)
                except Exception:
                    print("Modelseries: loadLastCheckpoint Error")



    def addRecord(self, newRecord):
        newRecord.checkpoint_idx = self.checkpoint
        self.records = pd.concat([self.records, pd.DataFrame([newRecord.toDict()])], ignore_index = True)
        self.saveJsonData()

    def getEpoch(self):
        try:
            epoch = self.records.iloc[-1]["epoch"]
            return int(epoch)
        except IndexError:
            return 0
    
    def addCheckpoint(self, model):
        try:
            epoch_idx = int(self.records.iloc[-1]["epoch"])
        except Exception:
            epoch_idx = 0
        #self.checkpoint = checkpoint_idx + 1
        self.saveCheckpoint(model, epoch_idx)

    def saveCheckpoint(self, model, epoch_idx):
        filename = f"{epoch_idx}.pth"
        dir = os.path.join(self.series_dir, "checkpoints")
        os.makedirs(dir, exist_ok=True)
        util.saveModel(filename, model, dir)


    def loadLatestCheckpoint(self, model):
        dir_path = os.path.join(self.series_dir, "checkpoints")
        checkpoint_id = int(max([int(f.split(".")[0]) for f in os.listdir(dir_path)]))
        filename =  f"{checkpoint_id}.pth"
        return util.loadModel(filename, model, dir = dir_path) 

    
    def loadJsonData(self):
        filename = f"{self.name}_.json"
        with open(os.path.join(self.series_dir, filename), 'r') as f:
            data = json.load(f)
        self.model_descr = data["model_descr"]
        self.S = data["S"]
        self.N = data["N"]
        self.A = data["A"]
        self.RES = data["RES"]
        self.records = pd.DataFrame(data["records"])
        
        checkpoint_idx = self.records.iloc[-1]["checkpoint_idx"] #assume pandas dataframe is still sorted after main index
    
    def saveJsonData(self):
        data = {
            "name" : self.name,
            "model_descr" : self.model_descr,
            "S" : self.S,
            "N" : self.N,
            "A" : self.A,
            "RES" : self.RES,
            "records" : self.records.to_dict(orient="records"), 
            "checkpoint" : self.checkpoint
        }
        filename = f"{self.name}_.json"
        filepath = os.path.join(self.series_dir, filename)
        os.makedirs(self.series_dir, exist_ok=True)
        with open(filepath, 'w') as f: #complete overwrite
            json.dump(data, f, indent = 4)
