# Model & Dataset Selection
MODEL_NAME = "bfor_fcos"  # "fcos" or "bfor"
DATASET = "voc"  # "voc" or "coco"

import os

# Data
DATA_PATH = (
    "/mnt/external_ssd/Datasets"
    if os.path.exists("/mnt/external_ssd/Datasets")
    else "Data"
)
TRAIN_SIZE = 0.9
BATCH_SIZE = 4
NUM_WORKERS = 4
SEED = 42

# FCOS Specific Settings
FCOS_PRETRAINED_BACKBONE = False  # ResNet-50 ImageNet-1K pretrained weights
FCOS_NUM_CLASSES = 21  # 20 classes for VOC + 1 background

# B-FOR Specific Settings
N_CHANNELS = 128
DROP_RATE = 0.2
ALPHA = 0.1
LAMBDA_CTR = 1.0
K = 9
ALL_CLASSES = True

# Training
EPOCHS = 100
OPTIMIZER = "sgd" if MODEL_NAME in ["fcos", "bfor_fcos"] else "adam"
LR = 0.001 if MODEL_NAME in ["fcos", "bfor_fcos"] else 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4 if MODEL_NAME in ["fcos", "bfor_fcos"] else 0.0
LR_PATIENCE = 5
LR_FACTOR = 0.3
MIN_LR_SCHEDULER = 1e-9
EARLY_STOPPING_PATIENCE = 10
MAX_GRAD_NORM = 10.0
DEVICE = "cuda"

if MODEL_NAME == "fcos":
    SAVE_PATH = "best_fcos_voc20.pt"
elif MODEL_NAME == "bfor_fcos":
    SAVE_PATH = "best_bfor_fcos_voc20.pt"
else:
    SAVE_PATH = "best_model_voc20.pt"
