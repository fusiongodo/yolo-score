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




class EvalRecord:
    def __init__(self, gt_df, model_name, save_dir):
        columns = "class_id, class_name, n_occurences, n_images, epoch, mAP, mREC".split(", ")
        self.df = pd.DataFrame(columns = columns)
        self.gt_df = gt_df
        self.model_name = model_name
        self.save_dir = save_dir

                # Load class names JSON into a dict
        with open(os.path.join(c.img_dir, "..", "class_names.json"), "r", encoding="utf-8") as f:
            self.class_names = json.load(f)  # keys as strings

        # Convert keys to int for direct index lookup
        self.class_names = {int(k): v for k, v in self.class_names.items()}
        self.class_stats = self.calc_class_stats()

    def calc_class_stats(self):
        # group stats
        stats = (
            self.gt_df.groupby("class_id")
                .agg(
                    n_occurences=("class_id", "size"),
                    n_images=("img_id", "nunique")
                )
                .reset_index()
        )

        # add class_name
        stats["class_name"] = stats["class_id"].map(self.class_names)

        # add empty columns to match EvalRecord schema
        stats["epoch"] = None
        stats["mAP"]   = None
        stats["mREC"]  = None

        # reorder to match self.df columns
        stats = stats[["class_id", "class_name", "n_occurences", "n_images", "epoch", "mAP", "mREC"]]

        # sort by n_occurences
        stats = stats.sort_values("n_occurences", ascending=False).reset_index(drop=True)

        return stats
    
    #prec_per_cls and rec_per_cls are torch tensors
    def addEvalRecord(self, prec_per_cls, rec_per_cls, epoch):
        ids = list(range(len(prec_per_cls)))
        occ_map = dict(zip(self.class_stats["class_id"], self.class_stats["n_occurences"]))
        img_map = dict(zip(self.class_stats["class_id"], self.class_stats["n_images"]))
        data = {
            "class_id": list(range(len(prec_per_cls))),
            "class_name": [self.class_names[i] for i in ids],
            "mAP": prec_per_cls.detach().cpu().numpy(),
            "mREC": rec_per_cls.detach().cpu().numpy(),
            "n_occurences": [occ_map.get(i, 0) for i in ids],
            "n_images": [img_map.get(i, 0) for i in ids],
            "epoch": [epoch] * len(ids) 
        }

        df_new = pd.DataFrame(data)
        self.df = pd.concat([self.df, df_new], ignore_index=True)
        self.saveJsonData()

    def saveJsonData(self):
        data = {
            "eval_records": self.df.to_dict(orient="records")
        }
        filename = f"{self.model_name}_eval.json"
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"EvalRecord: saved → {filepath}")

    def loadJsonData(self, model_name, save_dir):
        filename = f"{model_name}_eval.json"
        filepath = os.path.join(save_dir, filename)
        if not os.path.exists(filepath):
            print(f"EvalRecord: no eval JSON found at {filepath}")
            return
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.df = pd.DataFrame(data["eval_records"])
    
    
    
class LearnConfig:
    def __init__(self, c_xy, c_wh, c_obj, c_noobj, c_cls, c_lr, iou_obj = False):
        self.c_xy = c_xy
        self.c_wh = c_wh
        self.c_obj = c_obj
        self.c_noobj = c_noobj
        self.c_cls = c_cls
        self.c_lr = c_lr
        self.iou_obj = iou_obj
    
    def toDict(self):
        return {
            "c_xy": self.c_xy,
            "c_wh": self.c_wh,
            "c_obj": self.c_obj,
            "c_noobj": self.c_noobj,
            "c_cls": self.c_cls,
            "c_lr" : self.c_lr,
            "iou_obj" : self.iou_obj
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
    def __init__(self, n_crops, epoch, mAP, mREC, learn_configs : LearnConfig, losses : LossRecord): 
        
        print(f"Record.__init__: {n_crops} images processed")
        losses = losses.toDict()
        learn_configs = learn_configs.toDict()
        self.checkpoint_idx = 0
        self.n_crops = n_crops
        self.epoch = epoch
        self.mAP = mAP
        self.mREC = mREC
        self.c_xy = learn_configs["c_xy"]
        self.c_wh = learn_configs["c_wh"]
        self.c_obj = learn_configs["c_obj"]
        self.c_noobj = learn_configs["c_noobj"]
        self.c_cls = learn_configs["c_cls"]
        self.c_lr = learn_configs["c_lr"]
        self.iou_obj = learn_configs["iou_obj"] 
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
            "mREC": self.mREC,
            "c_xy": self.c_xy,
            "c_wh": self.c_wh,
            "c_obj": self.c_obj,
            "c_noobj": self.c_noobj,
            "c_cls": self.c_cls,
            "c_lr" : self.c_lr,
            "iou_obj" : self.iou_obj,
            "l_total": self.l_total,
            "l_xy": self.l_xy,
            "l_wh": self.l_wh,
            "l_obj": self.l_obj,
            "l_noobj": self.l_noobj,
            "l_cls": self.l_cls
        }

class ModelSeries:
    def __init__(self, name, gt_df, model = m.YOLOResNet(), model_descr = "not provided",  mode = "training"):
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
        columns = "checkpoint_idx, n_crops, epoch, mAP, mREC, c_lr, iou_obj, c_xy, c_wh, c_obj, c_noobj, c_cls, l_total, l_xy, l_wh, l_obj, l_noobj, l_cls".split(", ")
        # Define data types for each column
        dtypes = {
            "checkpoint_idx": "Int64",  # Nullable integer for checkpoint index
            "n_crops": "Int64",        # Nullable integer for crop count
            "epoch": "Int64",          # Nullable integer for epoch
            "mAP": "float64",          # Float for mean average precision
            "mREC": "float64",
            "c_lr": "float64", 
            "iou_obj": "bool",
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
        self.eval_records = EvalRecord(gt_df, self.name, self.series_dir)
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
                except Exception as e:
                    print(f"Modelseries init(): loadLastCheckpoint Error → {e}")
        



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
        files = [
            int(f.split(".")[0]) for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f)) and f.split(".")[0].isdigit()
        ]
        print(f"loadLatestCheckpoint available files: {files}")
        checkpoint_id = int(max(files))
        print(f"loadLatestCheckpoint picked file: {checkpoint_id}")
        filename =  f"{checkpoint_id}.pth"
        print(f"latest checkpoint: {filename}")
        return util.loadModel(filename, model, dir = dir_path) 

    
    def loadJsonData(self):
        filename = f"{self.name}_.json"
        try:
            with open(os.path.join(self.series_dir, filename), 'r') as f:
                data = json.load(f)
        except Exception:
            print(f"ModelSeries: {Exception.with_traceback}")
        self.model_descr = data["model_descr"]
        self.S = data["S"]
        self.N = data["N"]
        self.A = data["A"]
        self.RES = data["RES"]
        self.records = pd.DataFrame(data["records"])
        checkpoint_idx = self.records.iloc[-1]["checkpoint_idx"] #assume pandas dataframe is still sorted after main index

        self.eval_records.loadJsonData(self.name, self.series_dir)
    
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
        self.eval_records.saveJsonData()
