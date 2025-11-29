from src.test import TestConfig, test, DataConfig
from src.train import TrainConfig, train
from src.loaders import DataConfig
import os

# optimum_kfold = DataConfig(
#     dataset='optimum', 
#     cohort_selection_mode='kfold', 
#     fold=0,
# )

fold = int(os.getenv("SLURM_ARRAY_TASK_ID", "0")) 
checkpoint_path = f"/checkpoint/pwilson/17539347_{fold}/best.pth"

# test it as if we hadn't trained it
cfg = TestConfig(
    output_dir='.test', device='cpu', checkpoint=checkpoint_path,
)
test(cfg)



