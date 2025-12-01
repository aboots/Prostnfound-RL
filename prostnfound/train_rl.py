"""
Training script for ProstNFound-RL with GRPO

This is a modified version of train.py that supports RL training with GRPO.
"""

import argparse
from collections import defaultdict
import json
import logging
import os
from tempfile import mkdtemp
import typing as tp
from argparse import ArgumentParser, BooleanOptionalAction

import PIL
import hydra
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import wandb
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from medAI.modeling.prostnfound import ProstNFound
from medAI.modeling.prostnfound_rl import ProstNFoundRL
from medAI.modeling.grpo import GRPO, GRPOTrainer, create_grpo_optimizer
from medAI.modeling.setr import SETR
from torch.nn import functional as F
from tqdm import tqdm
from torchvision.transforms import v2 as T

from medAI.modeling.registry import create_model, list_models, register_model
from medAI.modeling import *
from medAI.utils.argparse import UpdateDictAction
from medAI.utils.reproducibility import (
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
)
from medAI.utils.accumulators import DataFrameCollector
from src.loss_new import (
    build_loss,
)
from src.rl_loss import RLRewardComputer, build_rl_reward_computer
from medAI.layers.masked_prediction_module import (
    MaskedPredictionModule,
)
from src.loaders import get_dataloaders
from src.evaluator import CancerLogitsHeatmapsEvaluator as Evaluator
from PIL import Image


OmegaConf.register_new_resolver('getenv', os.getenv)


def main(cfg):
    # setup
    logging.basicConfig(
        level=logging.INFO if not cfg.debug else logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info("Setting up RL experiment")

    if cfg.debug:
        cfg.name = "debug_rl"

    wandb.init(
        config=OmegaConf.to_object(cfg),
        **cfg.get("wandb", {}),
        project=cfg.get("project", "prostate_cancer_detection_rl"),
        name=cfg.name,
    )
    _tmpdir = mkdtemp()
    OmegaConf.save(cfg, os.path.join(_tmpdir, "train_config.yaml"), resolve=True) 
    wandb.save(os.path.join(_tmpdir, "train_config.yaml"), base_path=_tmpdir, policy="now")
    cfg.wandb_url = wandb.run.url if wandb.run else None

    if cfg.checkpoint_dir is not None:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        exp_state_path = os.path.join(cfg.checkpoint_dir, "experiment_state_rl.pth")
        if os.path.exists(exp_state_path):
            logging.info("Loading experiment state from experiment_state_rl.pth")
            state = torch.load(exp_state_path, weights_only=False)
        else:
            logging.info("No experiment state found - starting from scratch")
            state = None
    else:
        state = None

    set_global_seed(cfg.seed)

    logging.info("Setting up RL model")

    # Create model
    model = create_model(cfg.model, **cfg.model_kw)
    print(f"Model: {type(model)}")
    print(f"Model kwargs: {cfg.model_kw}")
    
    # Check if it's an RL model
    is_rl_model = isinstance(model, ProstNFoundRL)
    logging.info(f"Is RL model: {is_rl_model}")
    
    model = ProstNFoundMeta(model, **cfg.get('metamodel', {}), is_rl=is_rl_model)

    model.to(cfg.device)
    if cfg.get('torch_compile', False):
        logging.info("Compiling model with torch.compile")
        torch.compile(model)
    logging.info("Model setup complete")
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    logging.info(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    
    # Load checkpoint if provided
    if cfg.model_checkpoint:
        model_state = torch.load(cfg.model_checkpoint, map_location="cpu")
        if "model" in model_state:
            model_state = model_state["model"]
        msg = model.load_state_dict(model_state, strict=False)
        logging.info(f"Loaded model from {cfg.model_checkpoint} with message `{msg}`.")
    if state is not None:
        model.load_state_dict(state["model"])

    # setup criterion
    criterion = build_loss(cfg)

    loaders = get_dataloaders(cfg.data)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # Setup RL components if enabled
    use_rl = cfg.get('use_rl', False)
    if use_rl and is_rl_model:
        logging.info("Setting up RL training components")
        
        # Number of samples per image for within-image comparison
        num_samples_per_image = cfg.get('rl_num_samples_per_image', 4)
        logging.info(f"Using within-image comparison with {num_samples_per_image} samples per image")
        
        # Create GRPO algorithm with within-image comparison support
        grpo = GRPO(
            clip_eps=cfg.get('rl_clip_eps', 0.2),
            entropy_coef=cfg.get('rl_entropy_coef', 0.01),
            value_coef=cfg.get('rl_value_coef', 0.5),
            max_grad_norm=cfg.get('rl_max_grad_norm', 0.5),
            gamma=cfg.get('rl_gamma', 1.0),
            normalize_advantages=cfg.get('rl_normalize_advantages', True),
            num_samples_per_image=num_samples_per_image,
        )
        
        # Create reward computer
        reward_computer = build_rl_reward_computer(cfg)
        
        logging.info(f"RL mode: {cfg.get('rl_mode', 'grpo')}")
    else:
        grpo = None
        reward_computer = None

    optimizer, lr_scheduler = setup_optimizer(cfg, model, train_loader)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])

    scaler = torch.amp.GradScaler('cuda')
    if state is not None:
        scaler.load_state_dict(state["gradient_scaler"])

    epoch = 0 if state is None else state["epoch"]
    logging.info(f"Starting at epoch {epoch}")
    best_score = 0 if state is None else state["best_score"]
    logging.info(f"Best score so far: {best_score}")
    if state is not None:
        rng_state = state["rng"]
        set_all_rng_states(rng_state)

    def get_state():
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "gradient_scaler": scaler.state_dict(),
            "rng": get_all_rng_states(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": OmegaConf.to_object(cfg),
        }

    def save_checkpoint(name):
        state = get_state()
        if cfg.checkpoint_dir is not None:
            logging.info(f"Saving experiment snapshot to {cfg.checkpoint_dir}")
            torch.save(state, os.path.join(cfg.checkpoint_dir, name))
            if cfg.get('save_checkpoint_wandb', False):
                wandb.save(
                    os.path.join(cfg.checkpoint_dir, name),
                    base_path=cfg.checkpoint_dir,
                    policy="now",
                )

    for epoch in range(epoch, cfg.epochs):
        if cfg.cutoff_epoch is not None and epoch > cfg.cutoff_epoch:
            break
        logging.info(f"Epoch {epoch}")

        save_checkpoint("experiment_state_rl.pth")

        if use_rl and is_rl_model:
            run_rl_train_epoch(
                cfg,
                model,
                train_loader,
                criterion,
                optimizer,
                lr_scheduler,
                scaler,
                grpo,
                reward_computer,
                epoch,
                desc="train",
            )
        else:
            run_train_epoch(
                cfg,
                model,
                train_loader,
                criterion,
                optimizer,
                lr_scheduler,
                scaler,
                epoch,
                desc="train",
            )

        if cfg.run_val:
            val_metrics = run_eval_epoch(cfg, model, val_loader, epoch, desc="val")

            if val_metrics is not None:
                tracked_metric = val_metrics[cfg.tracked_metric]
                new_record = tracked_metric > best_score
            else:
                new_record = None

            if new_record:
                best_score = tracked_metric
                logging.info(f"New best score: {best_score}")

            if cfg.run_test and new_record or cfg.get('test_every_epoch', False):
                logging.info("Running test set")
                metrics = run_eval_epoch(cfg, model, test_loader, epoch, desc="test")
            
            if new_record and cfg.get('save_best_weights', True):
                save_checkpoint("best_rl.pth")

    logging.info("Finished RL training")


def compute_coords_inside_prostate_stats(rl_coords, prostate_mask):
    """
    Compute statistics about how many RL attention coordinates are inside vs outside prostate.
    
    Args:
        rl_coords: Attention coordinates (B, num_points, 2) in [x, y] format, in 256x256 space
        prostate_mask: Prostate segmentation mask (B, 1, H, W), may be downsampled
        
    Returns:
        dict with statistics
    """
    if rl_coords is None:
        return {}
    
    B, num_points, _ = rl_coords.shape
    _, _, H_mask, W_mask = prostate_mask.shape
    
    # CRITICAL FIX: Resize mask to match coordinate space (256x256)
    COORD_SPACE_SIZE = 256
    if H_mask != COORD_SPACE_SIZE or W_mask != COORD_SPACE_SIZE:
        prostate_mask = torch.nn.functional.interpolate(
            prostate_mask.float(),
            size=(COORD_SPACE_SIZE, COORD_SPACE_SIZE),
            mode='nearest'
        )
    
    _, _, H, W = prostate_mask.shape
    
    total_inside = 0
    total_outside = 0
    
    for i in range(B):
        mask_i = prostate_mask[i, 0]
        coords_i = rl_coords[i]
        
        for j in range(num_points):
            x, y = coords_i[j]
            px = int(torch.clamp(x, 0, W - 1).item())
            py = int(torch.clamp(y, 0, H - 1).item())
            
            if mask_i[py, px] > 0.5:
                total_inside += 1
            else:
                total_outside += 1
    
    total = total_inside + total_outside
    pct_inside = 100.0 * total_inside / total if total > 0 else 0.0
    
    return {
        'coords_inside_prostate_pct': pct_inside,
        'coords_outside_prostate_pct': 100.0 - pct_inside,
        'total_coords': total,
    }


def run_train_epoch(
    args, model, loader, criterion, optimizer, scheduler, scaler, epoch, desc="Train"
):
    """Standard training epoch (non-RL)"""
    model.train()
    evaluator = Evaluator(**args.evaluator)

    for train_iter, data in enumerate(tqdm(loader, desc=desc)):

        if args.debug and train_iter > 10:
            break

        with torch.amp.autocast('cuda', enabled=args.use_amp):
            data = model(data)

            if torch.any(torch.isnan(data["cancer_logits"])):
                logging.warning("NaNs in heatmap logits")

            loss = criterion(data)

        loss = loss / args.accumulate_grad_steps
        
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (train_iter + 1) % args.accumulate_grad_steps == 0:
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        step_metrics = {f"train/{k}": v for k, v in evaluator(data).items()}
        step_metrics.update({"train_loss": loss.item() * args.accumulate_grad_steps})
        
        # Log learning rates
        if hasattr(model, 'get_params_groups'):
            encoder_lr = optimizer.param_groups[0]["lr"]
            main_lr = optimizer.param_groups[1]["lr"]
            cnn_lr = optimizer.param_groups[2]["lr"]
            step_metrics["encoder_lr"] = encoder_lr
            step_metrics["main_lr"] = main_lr
            step_metrics["cnn_lr"] = cnn_lr

        wandb.log(step_metrics)

    metrics = evaluator.aggregate_metrics()
    metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
    metrics["epoch"] = epoch
    wandb.log(metrics)


def run_rl_train_epoch(
    args, model, loader, criterion, optimizer, scheduler, scaler, grpo, reward_computer, epoch, desc="Train RL"
):
    """
    RL training epoch with GRPO using Within-Image Comparison.
    
    Key improvement: Instead of comparing different images (which have different difficulty),
    we sample multiple rollouts (attention locations) per image and compare them within
    each image. This allows the model to learn which attention strategies work better
    for each specific image.
    """
    model.train()
    evaluator = Evaluator(**args.evaluator)
    
    # Number of samples (rollouts) per image for within-image comparison
    num_samples_per_image = args.get('rl_num_samples_per_image', 4)

    for train_iter, data in enumerate(tqdm(loader, desc=desc)):

        if args.debug and train_iter > 10:
            break

        # Step 1: Collect multiple rollouts per image for within-image comparison
        # This is the key change: instead of one sample per image, we sample multiple
        # attention location configurations and compare them within each image
        with torch.no_grad():
            all_rollout_data = []
            all_rewards = []
            
            for sample_idx in range(num_samples_per_image):
                # Sample from policy (non-deterministic)
                rollout_data = model(data, deterministic=False)
                
                # Store rollout data
                all_rollout_data.append({
                    'rl_log_probs': rollout_data.get('rl_log_probs'),
                    'rl_value': rollout_data.get('rl_value'),
                    'rl_attention_coords': rollout_data.get('rl_attention_coords'),
                    'cancer_logits': rollout_data.get('cancer_logits'),
                })
                
                # Compute reward for this rollout
                reward = reward_computer(rollout_data, data)
                all_rewards.append(reward)
            
            # Compute prostate boundary statistics for logging
            if args.get('rl_prostate_boundary_penalty_weight', 0) > 0:
                coords_stats = compute_coords_inside_prostate_stats(
                    all_rollout_data[0]['rl_attention_coords'],
                    data['prostate_mask'].to(args.device)
                )
            else:
                coords_stats = {}
            
            # Stack all samples: shape becomes (B * num_samples_per_image,)
            # Interleaved: [img0_sample0, img0_sample1, ..., img1_sample0, img1_sample1, ...]
            B = data['bmode'].shape[0]
            
            # Concatenate log_probs: (B, k) -> (B * num_samples, k)
            old_log_probs_list = [rd['rl_log_probs'] for rd in all_rollout_data]
            old_log_probs = torch.stack(old_log_probs_list, dim=1).view(B * num_samples_per_image, -1)
            
            # Concatenate values: (B, 1) -> (B * num_samples, 1)
            old_values_list = [rd['rl_value'] for rd in all_rollout_data]
            old_values = torch.stack(old_values_list, dim=1).view(B * num_samples_per_image, -1)
            
            # Concatenate rewards: (B,) -> (B * num_samples,)
            rewards = torch.stack(all_rewards, dim=1).view(B * num_samples_per_image)

        # Step 2: GRPO updates with within-image advantage normalization
        num_rl_updates = args.get('rl_num_update_epochs', 4)
        rl_metrics_list = []
        
        for rl_epoch in range(num_rl_updates):
            with torch.amp.autocast('cuda', enabled=args.use_amp):
                # Re-run model for each sample to get current policy
                # We need to sample the same number of times
                current_log_probs_list = []
                current_values_list = []
                
                for sample_idx in range(num_samples_per_image):
                    current_data = model(data, deterministic=False)
                    current_log_probs_list.append(current_data.get('rl_log_probs'))
                    current_values_list.append(current_data.get('rl_value'))
                
                # Stack current samples
                current_log_probs = torch.stack(current_log_probs_list, dim=1).view(B * num_samples_per_image, -1)
                current_values = torch.stack(current_values_list, dim=1).view(B * num_samples_per_image, -1)
                
                # Supervised loss (use last sample's data for supervised loss)
                supervised_loss = criterion(current_data)
                
                # RL loss (GRPO) with within-image comparison
                if old_log_probs is not None:
                    rl_loss, rl_info = grpo.compute_loss(
                        current_log_probs,
                        old_log_probs.detach(),
                        current_values,
                        rewards.detach(),
                        num_samples_per_image=num_samples_per_image,
                    )
                    
                    # Combined loss
                    total_loss = supervised_loss + rl_loss
                    rl_metrics_list.append(rl_info)
                else:
                    total_loss = supervised_loss
                    rl_info = {}

            total_loss = total_loss / args.accumulate_grad_steps
            
            if args.use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if (train_iter + 1) % args.accumulate_grad_steps == 0:
                if args.use_amp:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.get('rl_max_grad_norm', 0.5)
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.get('rl_max_grad_norm', 0.5)
                    )
                    optimizer.step()
                    optimizer.zero_grad()

        scheduler.step()

        # Evaluate and log
        step_metrics = {f"train/{k}": v for k, v in evaluator(current_data).items()}
        step_metrics.update({
            "train_loss": supervised_loss.item(),
            "train_total_loss": total_loss.item() * args.accumulate_grad_steps,
        })
        
        # Add RL metrics
        if rl_metrics_list:
            avg_rl_metrics = {}
            for key in rl_metrics_list[0].keys():
                avg_rl_metrics[f"train_rl/{key}"] = np.mean([m[key] for m in rl_metrics_list])
            step_metrics.update(avg_rl_metrics)
        
        # Add reward stats (now computed within-image)
        step_metrics.update({
            "train_rl/reward_mean": rewards.mean().item(),
            "train_rl/reward_std": rewards.std().item(),
            "train_rl/reward_min": rewards.min().item(),
            "train_rl/reward_max": rewards.max().item(),
            "train_rl/num_samples_per_image": num_samples_per_image,
        })
        
        # Compute within-image reward variance (to monitor diversity)
        rewards_per_image = rewards.view(B, num_samples_per_image)
        within_image_std = rewards_per_image.std(dim=1).mean().item()
        step_metrics["train_rl/within_image_reward_std"] = within_image_std
        
        # Log prostate boundary statistics
        if coords_stats:
            for key, value in coords_stats.items():
                step_metrics[f"train_rl/{key}"] = value
        
        # Log learning rates
        if hasattr(model, 'get_params_groups'):
            step_metrics["encoder_lr"] = optimizer.param_groups[0]["lr"]
            step_metrics["main_lr"] = optimizer.param_groups[1]["lr"]
            step_metrics["cnn_lr"] = optimizer.param_groups[2]["lr"]

        wandb.log(step_metrics)

    metrics = evaluator.aggregate_metrics()
    metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
    metrics["epoch"] = epoch
    wandb.log(metrics)


@torch.no_grad()
def run_eval_epoch(args, model, loader, epoch, desc="eval"):
    model.eval()

    evaluator = Evaluator(**args.evaluator)

    for train_iter, data in enumerate(tqdm(loader, desc=desc)):

        with torch.amp.autocast('cuda', enabled=args.use_amp):
            # Use deterministic policy for evaluation
            data = model(data, deterministic=True)

        step_metrics = {f"{desc}/{k}": v for k, v in evaluator(data).items()}
        wandb.log(step_metrics)

    metrics = evaluator.aggregate_metrics()
    metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
    metrics["epoch"] = epoch
    wandb.log(metrics)

    return metrics


def setup_optimizer(args, model, train_loader):
    from torch.optim import AdamW

    (
        encoder_parameters,
        warmup_parameters,
        cnn_parameters,
    ) = model.get_params_groups()

    total_epochs = args.epochs
    encoder_frozen_epochs = args.get('warmup_epochs', 0)
    warmup_epochs = 5
    niter_per_ep = len(train_loader)
    warmup_lr_factor = args.get('warmup_lr', 0.0001) / args.lr
    params = [
        {"params": encoder_parameters, "lr": args.get('encoder_lr', 1e-5)},
        {"params": warmup_parameters, "lr": args.lr},
        {"params": cnn_parameters, "lr": args.get('cnn_lr', 1e-5)},
    ]

    def compute_lr_multiplier(iter, is_encoder_or_cnn=True):
        schedule = args.get('scheduler', 'cosine')
        if schedule == 'constant': 
            return 1

        if iter < encoder_frozen_epochs * niter_per_ep:
            if is_encoder_or_cnn:
                return 0
            else:
                if iter < warmup_epochs * niter_per_ep:
                    return (iter * warmup_lr_factor) / (warmup_epochs * niter_per_ep)
                else:
                    cur_iter_in_frozen_phase = iter - warmup_epochs * niter_per_ep
                    total_iter_in_frozen_phase = (
                        encoder_frozen_epochs - warmup_epochs
                    ) * niter_per_ep
                    return (
                        0.5
                        * (
                            1
                            + np.cos(
                                np.pi
                                * cur_iter_in_frozen_phase
                                / (total_iter_in_frozen_phase)
                            )
                        )
                        * warmup_lr_factor
                    )
        else:
            iter -= encoder_frozen_epochs * niter_per_ep
            if iter < warmup_epochs * niter_per_ep:
                return iter / (warmup_epochs * niter_per_ep)
            else:
                cur_iter = iter - warmup_epochs * niter_per_ep
                total_iter = (
                    total_epochs - warmup_epochs - encoder_frozen_epochs
                ) * niter_per_ep
                return 0.5 * (1 + np.cos(np.pi * cur_iter / total_iter))

    optimizer = AdamW(params, lr=args.lr, weight_decay=args.get('wd', 0))
    from torch.optim.lr_scheduler import LambdaLR

    lr_scheduler = LambdaLR(
        optimizer,
        [
            lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=True),
            lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=False),
            lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=True),
        ],
    )

    return optimizer, lr_scheduler


class ProstNFoundMeta(nn.Module):
    """Wraps a model to perform forward pass with ProstNFound style training"""

    def __init__(self, model: nn.Module, mask_output_key=None, is_rl=False):
        super().__init__()
        self.model = model
        self.mask_output_key = mask_output_key
        self.is_rl = is_rl

        if isinstance(self.model, ProstNFound):
            logging.info(f"Model ProstNFound with prompts {self.model.prompts}")
        elif isinstance(self.model, ProstNFoundRL):
            logging.info(f"Model ProstNFoundRL with prompts {self.model.prompts}")

        self.register_buffer("temperature", torch.tensor([1.0]))
        self.register_buffer("bias", torch.tensor([0.0]))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, include_postprocessed_heatmaps=False, deterministic=False):
        # extracting relevant data from the batch
        bmode = data["bmode"].to(self.device)
        needle_mask = data["needle_mask"].to(self.device)
        prostate_mask = data["prostate_mask"].to(self.device)
        if "rf" in data:
            rf = data["rf"].to(self.device)
        else:
            rf = None

        B = len(bmode)

        # Wrapped forward pass
        if isinstance(self.model, (ProstNFound, ProstNFoundRL)):
            prompts = {}
            for prompt_name in self.model.prompts:
                prompts[prompt_name] = data[prompt_name].to(
                    device=self.device, dtype=bmode.dtype
                )
                if prompts[prompt_name].ndim == 1:
                    prompts[prompt_name] = prompts[prompt_name][:, None]

            if self.is_rl:
                outputs = self.model(
                    bmode, rf, prostate_mask, needle_mask, 
                    output_mode="all", 
                    deterministic=deterministic,
                    return_rl_info=True,
                    **prompts
                )
            else:
                outputs = self.model(
                    bmode, rf, prostate_mask, needle_mask, output_mode="all", **prompts
                )
            
            cancer_logits = outputs["mask_logits"]
            image_level_classification_outputs = outputs.get("cls_outputs")
            if image_level_classification_outputs is not None:
                data["image_level_classification_outputs"] = image_level_classification_outputs
            
            # Add RL info to data
            if self.is_rl:
                data["rl_attention_coords"] = outputs.get("rl_attention_coords")
                data["rl_log_probs"] = outputs.get("rl_log_probs")
                data["rl_attention_map"] = outputs.get("rl_attention_map")
                data["rl_value"] = outputs.get("rl_value")
        else:
            model_outputs = self.model(bmode)
            if isinstance(model_outputs, dict):
                cancer_logits = model_outputs[self.mask_output_key]
            else:
                cancer_logits = self.model(bmode)

        cancer_logits = (
            cancer_logits / self.temperature[None, None, None, :]
            + self.bias[None, None, None, :]
        )
        data["cancer_logits"] = cancer_logits

        # compute predictions
        masks = (prostate_mask > 0.5) & (needle_mask > 0.5)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)
        mean_predictions_in_needle = []
        for j in range(B):
            mean_predictions_in_needle.append(
                predictions[batch_idx == j].sigmoid().mean()
            )
        mean_predictions_in_needle = torch.stack(mean_predictions_in_needle)
        data["average_needle_heatmap_value"] = mean_predictions_in_needle

        prostate_masks = prostate_mask > 0.5
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, prostate_masks)
        mean_predictions_in_prostate = []
        for j in range(B):
            mean_predictions_in_prostate.append(
                predictions[batch_idx == j].sigmoid().mean()
            )
        mean_predictions_in_prostate = torch.stack(mean_predictions_in_prostate)
        data["average_prostate_heatmap_value"] = mean_predictions_in_prostate

        if include_postprocessed_heatmaps: 
            cancer_logits = data["cancer_logits"]
            heatmap = cancer_logits[0, 0].detach().sigmoid().cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            try: 
                import cv2
                blurred = cv2.GaussianBlur(heatmap, (5, 5), sigmaX=1.5)
                upsampled = cv2.resize(blurred, (256, 256), interpolation=cv2.INTER_LINEAR)
            except ImportError:
                from skimage import filters, transform
                blurred = filters.gaussian(heatmap, sigma=1.5)
                upsampled = transform.resize(blurred, (256, 256), order=1, mode='reflect', anti_aliasing=True)
                upsampled = (upsampled * 255).astype(np.uint8)
            heatmap = upsampled
            data['cancer_probs'] = (torch.tensor(heatmap) / 255.)[None, None, ...]

        return data

    def get_params_groups(self):
        if isinstance(self.model, SETR):
            encoder_parameters = []
            warmup_parameters = []
            cnn_parameters = []
            for name, param in self.model.named_parameters():
                if "head" in name:
                    warmup_parameters.append(param)
                else:
                    encoder_parameters.append(param)
            return encoder_parameters, warmup_parameters, cnn_parameters

        elif hasattr(self.model, "image_encoder"):
            encoder_parameters = []
            warmup_parameters = []
            cnn_parameters = []
            for name, param in self.model.named_parameters():
                if "image_encoder" in name:
                    encoder_parameters.append(param)
                else:
                    warmup_parameters.append(param)
            return encoder_parameters, warmup_parameters, cnn_parameters

        elif hasattr(self.model, "get_params_groups"):
            return self.model.get_params_groups()

        elif isinstance(self.model, (ProstNFound, ProstNFoundRL)):
            return self.model.get_params_groups()

        else:
            from itertools import chain

            encoder_parameters = []
            warmup_parameters = self.model.parameters()
            cnn_parameters = []

            return encoder_parameters, warmup_parameters, cnn_parameters


if __name__ == "__main__":
    p = ArgumentParser(description="Train ProstNFound-RL model")
    p.add_argument('--config', '-c', help='Path to config file (located in cfg/train/...)')
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)

    main(cfg)

