"""
Visualization script for RL attention points

This script helps visualize where the RL policy is looking for suspicious regions.
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from omegaconf import OmegaConf
from tqdm import tqdm

from medAI.modeling.registry import create_model
from medAI.modeling.prostnfound_rl import ProstNFoundRL
import sys
sys.path.append('..')
from src.loaders import get_dataloaders


def visualize_attention_batch(
    images,
    attention_coords,
    attention_maps,
    cancer_logits,
    labels,
    save_dir,
    batch_idx,
    prostate_masks=None,
    needle_masks=None,
):
    """
    Visualize attention points and heatmaps for a batch.
    
    Args:
        images: B-mode images (B, C, H, W)
        attention_coords: Attention coordinates (B, k, 2)
        attention_maps: Attention heatmaps (B, 1, H, W) or None
        cancer_logits: Predicted cancer logits (B, 1, H, W)
        labels: Ground truth labels (B,)
        save_dir: Directory to save visualizations
        batch_idx: Batch index for filename
        prostate_masks: Prostate masks (B, 1, H, W)
        needle_masks: Needle masks (B, 1, H, W)
    """
    B = images.shape[0]
    
    for i in range(B):
        fig, axes = plt.subplots(1, 4 if attention_maps is not None else 3, figsize=(16, 4))
        
        # Original image
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f"Input Image\nLabel: {labels[i].item():.0f}")
        axes[0].axis('off')
        
        # Image with attention points
        axes[1].imshow(img, cmap='gray')
        coords = attention_coords[i].cpu().numpy()
        axes[1].scatter(coords[:, 0], coords[:, 1], c='red', marker='x', s=200, linewidths=3)
        for idx, (x, y) in enumerate(coords):
            axes[1].add_patch(Circle((x, y), radius=10, fill=False, color='red', linewidth=2))
            axes[1].text(x+5, y-5, f'{idx+1}', color='white', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        axes[1].set_title("Attention Points")
        axes[1].axis('off')
        
        # Attention heatmap (if available)
        if attention_maps is not None:
            attn_map = attention_maps[i, 0].cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            axes[2].imshow(attn_map, cmap='hot')
            axes[2].set_title("Attention Heatmap")
            axes[2].axis('off')
            ax_idx = 3
        else:
            ax_idx = 2
        
        # Cancer prediction heatmap
        cancer_pred = cancer_logits[i, 0].sigmoid().cpu().numpy()
        axes[ax_idx].imshow(cancer_pred, cmap='hot', vmin=0, vmax=1)
        
        # Overlay masks if available
        if prostate_masks is not None:
            prostate_contour = prostate_masks[i, 0].cpu().numpy()
            axes[ax_idx].contour(prostate_contour, colors='blue', linewidths=1.5, levels=[0.5])
        if needle_masks is not None:
            needle_contour = needle_masks[i, 0].cpu().numpy()
            axes[ax_idx].contour(needle_contour, colors='green', linewidths=1.5, levels=[0.5])
        
        axes[ax_idx].set_title("Cancer Prediction")
        axes[ax_idx].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'batch{batch_idx:03d}_sample{i:02d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {save_path}")


def main(args):
    # Load config
    cfg = OmegaConf.load(args.config)
    
    # Override with command line args
    if args.checkpoint:
        cfg.model_checkpoint = args.checkpoint
    if args.fold is not None:
        cfg.data.fold = args.fold
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Loading model...")
    model = create_model(cfg.model, **cfg.model_kw)
    
    if not isinstance(model, ProstNFoundRL):
        raise ValueError("This script requires a ProstNFoundRL model")
    
    model = model.to(device)
    model.eval()
    
    # Load checkpoint if provided
    if cfg.model_checkpoint:
        print(f"Loading checkpoint from {cfg.model_checkpoint}")
        state_dict = torch.load(cfg.model_checkpoint, map_location=device)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)
    
    # Load data
    print("Loading data...")
    loaders = get_dataloaders(cfg.data)
    loader = loaders[args.split]
    
    print(f"Visualizing {args.num_batches} batches from {args.split} set")
    
    # Visualize
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader, total=args.num_batches)):
            if batch_idx >= args.num_batches:
                break
            
            # Move to device
            bmode = data['bmode'].to(device)
            prostate_mask = data['prostate_mask'].to(device)
            needle_mask = data['needle_mask'].to(device)
            label = data.get('label', torch.zeros(len(bmode))).to(device)
            
            # Prepare prompts
            prompts = {}
            for prompt_name in model.prompts:
                if prompt_name in data:
                    prompts[prompt_name] = data[prompt_name].to(device, dtype=bmode.dtype)
                    if prompts[prompt_name].ndim == 1:
                        prompts[prompt_name] = prompts[prompt_name][:, None]
            
            # Forward pass
            outputs = model(
                bmode,
                prostate_mask=prostate_mask,
                needle_mask=needle_mask,
                output_mode='all',
                deterministic=True,
                return_rl_info=True,
                **prompts
            )
            
            # Visualize
            visualize_attention_batch(
                images=bmode,
                attention_coords=outputs['rl_attention_coords'],
                attention_maps=outputs.get('rl_attention_map'),
                cancer_logits=outputs['mask_logits'],
                labels=label,
                save_dir=args.output_dir,
                batch_idx=batch_idx,
                prostate_masks=prostate_mask,
                needle_masks=needle_mask,
            )
    
    print(f"\nVisualization complete! Outputs saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize RL attention points')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='visualizations/rl_attention',
                        help='Directory to save visualizations')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to visualize')
    parser.add_argument('--num-batches', type=int, default=5,
                        help='Number of batches to visualize')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold number (if using k-fold)')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    main(args)

