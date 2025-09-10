
## Clone Repository
Clone only the main branch (ca. 550 MB) instead of the whole repository (>2GB due to ultralytics branch).
```
git clone --single-branch --branch main --depth 1 https://gitlab.hs-flensburg.de/alha7503/obb_anns_hausarbeit.git
```
# Setup
setup.ipynb
Execute first code cell -> pip install gdown (Google Drive Python API)  
Execute second code cell -> download DeepscoresV2 Dataset  

# Dataset Statistics
dataset_statistics.ipynb   
Annoation density distribution for various detection grid resolutions, class occurence scatter plot and table.   

# Demo, Model Size, Recall and Precision, Training Losses
model_and_training_evaluation.ipynb  
Render an image of your choice: paste image into presentation/demo/images (as .png). Provide the name of the image file(e. g. image.png) as second argument to demo() function.  
Model Size  
Precision-Recall Curve  
Training Losses and mAP/REC per epoch.

# observe_training.ipynb
Track losses and model performance during training.  

# ModelSeries.py
Central class to track training progress associated with a model and its configurations. You can inspect the visualization of all gathered data in model_and_training_evaluation.ipynb. ModelSeries creates model weight checkpoints in regular intervals. LearnConfig is the datastructure used to track changes in loss parameters per epoch. EvalRecord tracks overall and per-class model performance per epoch.  

# training.ipynb
Configure and initiate training of the model. Uses Trainer class from trainer2.py  

# config.py
Adjusting eigher N, S or RES requires stride adjustments within model.py  
Parameter N: Detection head resolution of the model.  
Parameter S: Grid Resolution applied on the input image. e.g. if N=S we feed the complete image into the model, if S // N = 3 we divide the input image  into nine crops that are processed by the model individually.  
Parameter RES: Input image resolution.  
Parameter A: Number of anchors.  
Parameter ANCHORS: Anchor box dimensions.  

# loss.py
See lines 153-171 to understand inconcistency regarding mAP (evaluation) and loss function implementation.  

# util.py
class DataExtractor: Processes ds2_dense\deepscores_train.json data based on config.py settings to obtain formatted dataframe for model training and evaluation.  
Contains many helper functions used to load and save model weights, viualize the model predictions, etc.  

# eval.py
Functionalities related to model evaluation (mAP, mREC, ...)  



