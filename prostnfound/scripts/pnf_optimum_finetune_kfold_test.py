import argparse
import json
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
from src.test import TestConfig, test, DataConfig
from src.train import TrainConfig, train
from src.loaders import DataConfig
import os
from src.train import TrainConfig, train, DataConfig
from os.path import join
from pnf_optimum_finetune import cfg


parser = argparse.ArgumentParser()
parser.add_argument("--stage")
parser.add_argument('--groupname')

args = parser.parse_args()
groupname = args.groupname


if args.stage == "train":
    cfg = TrainConfig(
        model="prostnfound_plus_final",
        cutoff_epoch=10,
        data=DataConfig(
            fold=int(os.getenv("SLURM_ARRAY_TASK_ID", "0")),
            dataset="optimum",
            image_size=512,
            mask_size=128,
            cohort_selection_mode="kfold",
            augmentations="translate",
            batch_size=8, 
            grade_group_for_positive_label=2,  # GG2+ is positive
        ),
        wandb_kw=dict(
            group=groupname,
        ),
        save_best_weights=True,
        tracked_metric="val/core_auc",
        treat_gg1_as_benign=False,
    )

    checkpoint_dir = cfg.checkpoint_dir

    if checkpoint_dir:
        dst = os.path.join(os.getcwd(), "logs", groupname, f"fold_{cfg.data.fold}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        # make it easier to find
        os.symlink(checkpoint_dir, dst)

    train(cfg)

    fold = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    checkpoint = f"logs/{groupname}/fold_{fold}/best.pth"
    dir = Path(os.path.dirname(checkpoint))

    cfg_kw = dict(postprocess=True, save_visualizations=True)

    # test it as if we hadn't trained it
    cfg = TestConfig(
        model="prostnfound_plus_final",
        data_cfg=cfg.data,
        output_dir=join(dir, "test_no_finetune"),
        **cfg_kw,
    )

    cfg = TestConfig(
        data_cfg=cfg.data_cfg,
        output_dir=join(dir, "test_finetune"),
        checkpoint=f"logs/{groupname}/fold_{fold}/best.pth",
        **cfg_kw,
    )
    test(cfg)

elif args.stage == "aggregate_metrics":
    metrics = []
    for fold in range(5):
        dir = Path(f"logs/{groupname}/fold_{fold}") / "test_finetune"
        with open(dir / "metrics.json", "r") as f:
            metrics.append(json.load(f))
            metrics[-1]["finetune"] = True
            metrics[-1]["fold"] = fold
        dir = Path(f"logs/{groupname}/fold_{fold}") / "test_no_finetune"
        with open(dir / "metrics.json", "r") as f:
            metrics.append(json.load(f))
            metrics[-1]["finetune"] = False
            metrics[-1]["fold"] = fold

    with open("pnf_optimum_finetune_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    def flatten_dict(d):
        output = {}
        for k, v in d.items():
            if isinstance(v, dict):
                for k2, v2 in flatten_dict(v).items():
                    output[f"{k}_{k2}"] = v2
            else:
                output[k] = v
        return output

    metrics = [flatten_dict(m) for m in metrics]
    df = pd.DataFrame(metrics)

    df.to_csv("pnf_optimum_finetune_metrics.csv", index=False)

    metrics = ["GG2+_hmap_auroc", "GG2+_PRI-MUS_auroc"]
    df = df[["fold", "finetune"] + metrics]
    # df = df.groupby('fold').mean().reset_index()
    df = df.melt(
        id_vars=["fold", "finetune"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )

    new_df = df.groupby(["finetune", "metric"]).mean().reset_index()
    print(new_df)

    new_df["fold"] = "mean"
    df = pd.concat([df, new_df], ignore_index=True)

    import seaborn as sns

    fig, ax = plt.subplots(ncols=2, figsize=(10, 6))
    sns.barplot(
        data=df.query("finetune == False"), x="fold", y="value", hue="metric", ax=ax[0]
    )
    ax[0].set_title("Without Fine-tuning")
    ax[0].set_ylim(0.7, 1)
    sns.barplot(
        data=df.query("finetune == True"), x="fold", y="value", hue="metric", ax=ax[1]
    )
    ax[1].set_title("With Fine-tuning")
    ax[1].set_ylim(0.7, 1)
    plt.savefig("pnf_optimum_finetune_metrics.png")

    print(df)
