"""
Test script for ProstNFound-RL models

This is adapted from test.py to support RL models with attention mechanisms.
"""

import argparse
from collections import defaultdict
import json
import os
from argparse import ArgumentParser, Namespace
import time

import PIL
import hydra
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import rich_argparse
import torch
from PIL import Image
import medAI
from medAI.layers.masked_prediction_module import get_bags_of_predictions
from medAI.modeling.prostnfound_rl import ProstNFoundRL
from medAI.utils.accumulators import DataFrameCollector
from medAI.utils.argparse import UpdateDictAction

import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm

from medAI.datasets.nct2013 import data_accessor
from medAI.modeling import list_models, create_model
from src.loss import MaskedPredictionModule
from src.loaders import get_dataloaders
from src.evaluator import show_heatmap_prediction
from train_rl import ProstNFoundMeta
from src.evaluator import CancerLogitsHeatmapsEvaluator as Evaluator


@hydra.main(config_path="cfg", config_name="test_rl", version_base="1.3")
def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    else: 
        state = None

    train_args = Namespace(**state["args"])

    # IMPORTANT: Always use the model config from the checkpoint
    # This ensures we load the correct RL model architecture
    print(f"Checkpoint model: {train_args.model}")
    print(f"Checkpoint model_kw: {train_args.model_kw}")
    
    # Override test config with checkpoint's model config
    args.model = train_args.model
    args.model_kw = train_args.model_kw

    if args.save_checkpoint:
        torch.save(state, os.path.join(args.output_dir, "checkpoint.pth"))

    # Saving test-time config is always safe
    OmegaConf.save(args, os.path.join(args.output_dir, "test_args.yaml"))

    # Older training configs saved as plain dicts/Namespaces can contain
    # objects OmegaConf cannot serialize (e.g. types, unions). Since this
    # is only for bookkeeping, fail softly if saving them doesn't work.
    try:
        OmegaConf.save(
            state["args"],
            os.path.join(args.output_dir, "train_args.yaml"),
        )
    except Exception as e:
        print(f"Warning: could not save train_args.yaml: {e}")

    # Create model and detect if it's RL
    base_model = create_model(args.model, **args.model_kw)
    is_rl_model = isinstance(base_model, ProstNFoundRL)
    print(f"Model type: {type(base_model)}")
    print(f"Is RL model: {is_rl_model}")
    
    model = ProstNFoundMeta(base_model, is_rl=is_rl_model)
    print(model.load_state_dict(state["model"], strict=False))
    model.to(args.device)
    model.eval()
    if args.torch_compile:
        model = torch.compile(model)

    if args.get("data"):
        loaders = get_dataloaders(args.data, mode="test")
    elif "data" in vars(train_args):
        loaders = get_dataloaders(train_args.data, mode="test")
    else:
        loaders = get_dataloaders(train_args, mode="test")

    # maybe calibrate the temperature and bias of the model
    if args.calibration_mode == "pixel":
        do_calibration_pixel_wise_balanced_bce(
            model, loaders, args.calibrate_bias, args.calibrate_temperature
        )
    elif args.calibration_mode == "bag":
        do_calibration_bag_wise(
            model, loaders, args.calibrate_bias, args.calibrate_temperature
        )

    evaluator = Evaluator(
        log_images=False, include_patient_metrics=args.get('include_patient_metrics', False)
    )
    accumulator = defaultdict(list)
    
    # For RL models, also accumulate attention point statistics
    if is_rl_model:
        rl_accumulator = defaultdict(list)

    loader = loaders[args.split]

    # warmup
    for _ in range(10):
        batch = next(iter(loader))
        if is_rl_model:
            model(batch, deterministic=True)
        else:
            model(batch)

    for i, data in enumerate(tqdm(loader)):

        # measure inference
        t0 = time.perf_counter()

        with torch.amp.autocast_mode.autocast(
            device_type=torch.device(args.device).type, enabled=args.use_amp
        ):
            with torch.inference_mode():
                # Use deterministic policy for RL models
                if is_rl_model:
                    data = model(data, deterministic=True)
                else:
                    data = model(data)

        if args.postprocess:
            cancer_logits = data.pop("cancer_logits")
            heatmap = cancer_logits[0, 0].sigmoid().cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            # blur and upsample
            
            import skimage

            blurred = skimage.filters.gaussian(heatmap, sigma=1.5)
            upsampled = skimage.transform.resize(blurred, (256, 256), order=1, anti_aliasing=True)
            upsampled = (upsampled * 255).astype(np.uint8)
            heatmap = upsampled
            data["cancer_probs"] = (torch.tensor(heatmap) / 255.0)[None, None, ...]
        else:
            # get raw heatmap and also save as png
            heatmap = data["cancer_logits"][0, 0].sigmoid().cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = Image.fromarray(heatmap)

        if args.device == "cuda":
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - t0
        accumulator["infer_time"].append(infer_time)
        
        # Store RL-specific information
        if is_rl_model and 'rl_attention_coords' in data:
            rl_accumulator['attention_coords'].append(data['rl_attention_coords'].cpu())
            if 'rl_attention_map' in data and data['rl_attention_map'] is not None:
                rl_accumulator['attention_maps'].append(data['rl_attention_map'].cpu())

        if args.save_raw_heatmaps:
            # get raw heatmap and also save as png
            heatmap = Image.fromarray(heatmap)
            os.makedirs(os.path.join(args.output_dir, "raw_heatmaps"), exist_ok=True)
            heatmap.save(
                os.path.join(
                    args.output_dir, "raw_heatmaps", data["core_id"][0] + ".png"
                )
            )

        if args.save_rendered_heatmaps:

            patient_id = data['patient_id'][0]
            core_id = data['core_id'][0]
        
            output_file = os.path.join(
                args.output_dir, 
                "heatmaps", 
                patient_id,
                f"{core_id}.{args.save_format}"
            )
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            show_heatmap_prediction(data)
            
            # Overlay attention points if RL model
            if is_rl_model and 'rl_attention_coords' in data:
                coords = data['rl_attention_coords'][0].cpu().numpy()
                plt.scatter(coords[:, 0], coords[:, 1], c='red', marker='x', 
                           s=200, linewidths=3, label='RL Attention')
                plt.legend()
            
            plt.savefig(
                output_file,
                format=args.save_format,
            )
            plt.close()

        evaluator(data)

    table = evaluator.accumulator.compute()
    table.to_csv(os.path.join(args.output_dir, "metrics_by_core.csv"))

    metrics = evaluator.aggregate_metrics()
    metrics["infer_time"] = np.array(accumulator["infer_time"]).mean()
    metrics = {k: float(v) for k, v in metrics.items()}
    
    # Add RL-specific metrics
    if is_rl_model and rl_accumulator:
        print("\n=== RL Attention Statistics ===")
        all_coords = torch.cat(rl_accumulator['attention_coords'], dim=0)  # (N, k, 2)
        metrics['rl_attention_mean_x'] = float(all_coords[:, :, 0].mean())
        metrics['rl_attention_mean_y'] = float(all_coords[:, :, 1].mean())
        metrics['rl_attention_std_x'] = float(all_coords[:, :, 0].std())
        metrics['rl_attention_std_y'] = float(all_coords[:, :, 1].std())
        
        print(f"Average attention X: {metrics['rl_attention_mean_x']:.2f} ± {metrics['rl_attention_std_x']:.2f}")
        print(f"Average attention Y: {metrics['rl_attention_mean_y']:.2f} ± {metrics['rl_attention_std_y']:.2f}")
        
        # Save attention coordinates for further analysis
        np.save(
            os.path.join(args.output_dir, "rl_attention_coords.npy"),
            all_coords.numpy()
        )

    print("\n=== Test Metrics ===")
    print(json.dumps(metrics, indent=4))
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


def do_calibration_pixel_wise_balanced_bce(
    model,
    loaders,
    calibrate_bias=True,
    calibrate_temperature=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # extract all pixel predictions from val loader
    pixel_preds, pixel_labels, core_ids = extract_all_pixel_predictions(
        model, loaders["val"], device
    )
    core_ids = np.array(core_ids)

    # fit temperature and bias to center and scale the predictions
    temp = nn.Parameter(torch.ones(1))
    bias = nn.Parameter(torch.zeros(1))

    from torch.optim import LBFGS

    params = []
    if calibrate_bias:
        params.append(bias)
    if calibrate_temperature:
        params.append(temp)

    optim = LBFGS(params, lr=1e-3, max_iter=100, line_search_fn="strong_wolfe")

    # weight the loss to account for class imbalance
    pos_weight = (1 - pixel_labels).sum() / pixel_labels.sum()
    # encourage sensitivity over specificity
    pos_weight *= 1.6

    def closure():
        optim.zero_grad()
        logits = pixel_preds / temp + bias
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits[:, 0], pixel_labels)
        loss.backward()
        return loss

    for i in range(10):
        print(optim.step(closure))

    model.temperature.data.copy_(temp)
    model.bias.data.copy_(bias)


def do_calibration_bag_wise(
    model,
    loaders,
    calibrate_bias=True,
    calibrate_temperature=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    bags_of_logits, involvement, label = extract_all_bag_predictions(
        model, loaders["val"], device
    )

    # fit temperature and bias to center and scale the predictions
    log_temp = nn.Parameter(torch.zeros(1, device=device))
    bias = nn.Parameter(torch.zeros(1, device=device))

    from torch.optim import LBFGS

    pos_weight = (1 - label).sum() / label.sum()

    params = []
    if calibrate_bias:
        params.append(bias)
    if calibrate_temperature:
        params.append(log_temp)

    optim = LBFGS(params, lr=1e-1, max_iter=100)

    def closure():
        optim.zero_grad()
        loss = torch.tensor(0.0, device=device)
        for bag_i, involvement_i, label_i in zip(bags_of_logits, involvement, label):
            bag_i = bag_i / log_temp.exp() + bias
            bag_i = bag_i.sigmoid()
            bag_i_mean = bag_i.mean()
            loss_i = (
                -involvement_i * bag_i_mean.log()
                - (1 - involvement_i) * (1 - bag_i_mean).log()
            )
            if label_i:
                loss_i = loss_i * pos_weight
            loss = loss + loss_i
        loss.backward()
        return loss

    for i in range(10):
        print(optim.step(closure))

    model.temperature.data.copy_(log_temp.exp())
    model.bias.data.copy_(bias)


@torch.no_grad()
def extract_all_bag_predictions(model, loader, device):

    bags_of_logits = []
    involvement = []
    label = []
    
    is_rl_model = hasattr(model, 'is_rl') and model.is_rl

    for data in tqdm(loader, f"Running model..."):
        if is_rl_model:
            data = model(data, deterministic=True)
        else:
            data = model(data)
        bags_of_logits.extend(
            get_bags_of_predictions(
                data["cancer_logits"], data["prostate_mask"], data["needle_mask"]
            )
        )
        involvement.append(data["involvement"].to(device))
        label.append(data["label"].to(device))

    involvement = torch.cat(involvement)
    label = torch.cat(label)

    return bags_of_logits, involvement, label


def extract_all_pixel_predictions(model, loader, device):
    pixel_labels = []
    pixel_preds = []
    core_ids = []

    model.eval()
    model.to(device)
    
    is_rl_model = hasattr(model, 'is_rl') and model.is_rl

    for i, data in enumerate(tqdm(loader)):
        with torch.no_grad():
            if is_rl_model:
                data = model(data, deterministic=True)
            else:
                data = model(data)

            prostate_mask = data["prostate_mask"].to(device)
            needle_mask = data["needle_mask"].to(device)
            heatmap_logits = data["cancer_logits"]
            label = data["label"]
            core_id = data["core_id"]

            # compute predictions
            masks = (prostate_mask > 0.5) & (needle_mask > 0.5)

            predictions, batch_idx = MaskedPredictionModule()(heatmap_logits, masks)

            labels = torch.zeros(len(predictions), device=predictions.device)
            for i in range(len(predictions)):
                labels[i] = label[batch_idx[i]]
            pixel_preds.append(predictions.cpu())
            pixel_labels.append(labels.cpu())

            core_ids.extend(core_id[batch_idx[i]] for i in range(len(predictions)))

    pixel_preds = torch.cat(pixel_preds)
    pixel_labels = torch.cat(pixel_labels)

    return pixel_preds, pixel_labels, core_ids


if __name__ == "__main__":
    main()

