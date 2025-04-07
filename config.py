import os.path as osp

# Dataset configuration
MIL_DATASET = "mil-pcam"
PATH_TO_ZIP = f"{MIL_DATASET}-dummy.zip"
ROOT = "/dataset"
TRAIN_DIR = osp.join(ROOT, MIL_DATASET, "train")
VALID_DIR = osp.join(ROOT, MIL_DATASET, "valid")

# Training configuration
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 20
