import os
import sys
import argparse
import submitit

sys.path.append(os.getcwd())

from src.train import TrainConfig, train, DataConfig


cfg = TrainConfig(
    model="prostnfound_plus_final",
    device="cuda",
    cutoff_epoch=10,
    data=DataConfig(
        fold=int(os.getenv("SLURM_ARRAY_TASK_ID", "0")),
        dataset="optimum",
        image_size=512,
        mask_size=128,
        cohort_selection_mode="kfold",
        augmentations="translate",
    ),
    wandb_kw=dict(
        group="prostnfound_plus_final_optimum_finetune_v1",
    ),
    save_best_weights=True,
    tracked_metric="val/core_auc",
)

checkpoint_dir = cfg.checkpoint_dir


if __name__ == "__main__":

    dst = os.path.join(
        os.getcwd(), "logs", "pnf_optimum_finetune_v1", f"fold_{cfg.data.fold}"
    )
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    # make it easier to find
    os.symlink(checkpoint_dir, dst)

    train(cfg)
    # executor.submit(train, cfg)
