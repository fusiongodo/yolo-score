import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

img_dir  = os.path.join(BASE_DIR, "ds2_dense", "ds2_dense", "images")
filepath = os.path.join(BASE_DIR, "ds2_dense", "ds2_dense", "deepscores_train.json")
slimpath = os.path.join(BASE_DIR, "ds2_dense", "ds2_dense", "deepscores_train_slim.json")
logdir   = os.path.join(BASE_DIR, "logs")
checkpoints = os.path.join(BASE_DIR, "checkpoints")
models = os.path.join(BASE_DIR, "models")

S = 60
A = 2
C = 136
N = 60
RES = 720

ANCHORS = np.array([
    [0.02080085, 0.01517667],
    [0.02080085, 0.01517667]
])
