# config.py
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

img_dir  = os.path.join(BASE_DIR, "ds2_dense", "ds2_dense", "images")
filepath = os.path.join(BASE_DIR, "ds2_dense", "ds2_dense", "deepscores_train.json")
slimpath = os.path.join(BASE_DIR, "ds2_dense", "ds2_dense", "deepscores_train_slim.json")
logdir   = os.path.join(BASE_DIR, "logs")

S = 30
A = 2
C = 136
N = 10

RES = 480

ANCHORS = np.array([
    [0.02080085, 0.01517667],
    [0.02080085, 0.01517667]
])
