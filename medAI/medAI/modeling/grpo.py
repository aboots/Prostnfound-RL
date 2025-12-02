"""
Group Relative Policy Optimization (GRPO) for ProstNFound-RL

GRPO is a policy gradient method that normalizes advantages within groups
to improve training stability. It's particularly useful for on-policy RL in domains
with high variance rewards.

Key Innovation for Medical Imaging:
- Within-Image Comparison: Different images have different difficulty levels.
  Instead of comparing across images (batch normalization), we sample multiple
  rollouts (attention location selections) per image and normalize advantages
  WITHIN each image. This allows the model to learn which attention locations
  work better for each specific image.

IMPORTANT CHANGES (v2):
1. Removed value function - uses pure GRPO with group mean/std for advantages
   (matches Seg-R1 and DeepSeekMath approach)
2. Batched forward passes - all samples computed in single batched forward
3. Group-based reward computation for proper within-image comparison

Reference: 
DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
https://arxiv.org/abs/2402.03300
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging


class GRPO:
    """
    Group Relative Policy Optimization (Pure GRPO without Value Function).
    
    Following the Seg-R1 and DeepSeekMath approach:
    - No value function (simpler and often works better)
    - Advantages computed as: (reward - group_mean) / (group_std + eps)
    - Within-image comparison: advantages normalized per image, not batch
    
    Args:
        clip_eps: Clipping epsilon for PPO-style clipping (default: 0.2)
        entropy_coef: Coefficient for entropy bonus (default: 0.01)
        max_grad_norm: Maximum gradient norm for clipping (default: 0.5)
        kl_coef: Coefficient for KL penalty (like Seg-R1's beta, default: 0.01)
        normalize_advantages: Whether to normalize advantages (default: True)
        num_samples_per_image: Number of rollouts per image for within-image comparison (default: 4)
    """
    
    def __init__(
        self,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        kl_coef: float = 0.01,
        normalize_advantages: bool = True,
        num_samples_per_image: int = 4,
        # Legacy parameters (ignored)
        value_coef: float = 0.0,  # Ignored - no value function
        gamma: float = 1.0,  # Ignored - single step
    ):
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.kl_coef = kl_coef
        self.normalize_advantages = normalize_advantages
        self.num_samples_per_image = num_samples_per_image
        
        logging.info(
            f"Initialized Pure GRPO (no value function) with clip_eps={clip_eps}, "
            f"entropy_coef={entropy_coef}, kl_coef={kl_coef}, "
            f"num_samples_per_image={num_samples_per_image}"
        )
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        num_samples_per_image: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute advantages using group mean/std (no value function).
        
        This follows Seg-R1/DeepSeekMath approach:
        advantages = (reward - group_mean) / (group_std + eps)
        
        For within-image comparison, we normalize within each image group.
        
        Args:
            rewards: Rewards for each sample (B * num_samples,)
            num_samples_per_image: Number of samples per image
            
        Returns:
            advantages: Computed advantages (B * num_samples,)
        """
        num_samples = num_samples_per_image if num_samples_per_image is not None else self.num_samples_per_image
        
        if num_samples > 1 and self.normalize_advantages:
            total_samples = rewards.shape[0]
            num_images = total_samples // num_samples
            
            if total_samples == num_images * num_samples:
                # Reshape to (num_images, num_samples) for within-image normalization
                rewards_grouped = rewards.view(num_images, num_samples)
                
                # Compute mean and std per image (like Seg-R1)
                group_mean = rewards_grouped.mean(dim=1, keepdim=True)
                group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-4
                
                # Normalize within each image group
                advantages_grouped = (rewards_grouped - group_mean) / group_std
                
                # Flatten back
                advantages = advantages_grouped.view(-1)
            else:
                logging.warning(
                    f"Within-image normalization failed: total_samples={total_samples}, "
                    f"num_images={num_images}, num_samples={num_samples}. "
                    "Falling back to batch normalization."
                )
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        elif self.normalize_advantages:
            # Standard batch normalization
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        else:
            advantages = rewards
        
        return advantages
    
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO-style policy loss with clipping.
        
        Args:
            log_probs: Log probabilities from current policy (B, k)
            old_log_probs: Log probabilities from old policy (B, k)
            advantages: Advantage estimates (B,)
            
        Returns:
            loss: Policy loss
            info: Dictionary with loss components for logging
        """
        # Sum log probs across attention points
        log_probs_sum = log_probs.sum(dim=1)  # B
        old_log_probs_sum = old_log_probs.sum(dim=1)  # B
        
        # Compute ratio
        ratio = torch.exp(log_probs_sum - old_log_probs_sum)  # B
        
        # Clipped surrogate objective
        advantages = advantages.detach()  # Don't backprop through advantages
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute KL divergence penalty (like Seg-R1)
        # KL = exp(old - new) - (old - new) - 1
        per_point_kl = torch.exp(old_log_probs - log_probs) - (old_log_probs - log_probs) - 1
        kl = per_point_kl.sum(dim=1).mean()
        
        # Compute entropy bonus (encourages exploration)
        entropy = -log_probs.mean()
        
        # Total loss (with KL penalty like Seg-R1)
        total_loss = policy_loss + self.kl_coef * kl - self.entropy_coef * entropy
        
        # Logging info
        info = {
            'policy_loss': policy_loss.item(),
            'kl': kl.item(),
            'entropy': entropy.item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_min': ratio.min().item(),
            'ratio_max': ratio.max().item(),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
        }
        
        return total_loss, info
    
    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        num_samples_per_image: Optional[int] = None,
        values: Optional[torch.Tensor] = None,  # Ignored for backward compatibility
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total GRPO loss (pure GRPO without value function).
        
        Args:
            log_probs: Current log probabilities (B * num_samples, k)
            old_log_probs: Old log probabilities (B * num_samples, k)
            rewards: Rewards (B * num_samples,)
            num_samples_per_image: Number of samples per image for within-image normalization
            values: Ignored (kept for backward compatibility)
            
        Returns:
            total_loss: Policy loss
            info: Dictionary with loss components
        """
        # Compute advantages with within-image normalization (no value function)
        advantages = self.compute_advantages(rewards, num_samples_per_image=num_samples_per_image)
        
        # Compute policy loss
        total_loss, info = self.compute_policy_loss(log_probs, old_log_probs, advantages)
        info['total_loss'] = total_loss.item()
        
        # Add within-image comparison info
        if num_samples_per_image is not None and num_samples_per_image > 1:
            info['num_samples_per_image'] = num_samples_per_image
            
            # Compute within-image reward variance
            num_images = rewards.shape[0] // num_samples_per_image
            rewards_per_image = rewards.view(num_images, num_samples_per_image)
            info['within_image_reward_std'] = rewards_per_image.std(dim=1).mean().item()
        
        return total_loss, info


class BatchedGRPOTrainer:
    """
    Optimized GRPO Trainer with batched forward passes.
    
    Key optimization: Instead of running num_samples_per_image separate forward passes,
    we replicate the batch and run ONE batched forward pass.
    
    This dramatically reduces training time on large GPUs.
    
    Args:
        grpo: GRPO algorithm instance
        num_samples_per_image: Number of rollouts per image (default: 4)
        device: Device to use (default: 'cuda')
    """
    
    def __init__(
        self,
        grpo: Optional[GRPO] = None,
        num_samples_per_image: int = 4,
        device: str = 'cuda',
    ):
        self.grpo = grpo if grpo is not None else GRPO(num_samples_per_image=num_samples_per_image)
        self.num_samples_per_image = num_samples_per_image
        self.device = device
        
        logging.info(
            f"Initialized BatchedGRPOTrainer with {num_samples_per_image} samples per image "
            f"(batched forward pass for efficiency)"
        )
    
    @staticmethod
    def replicate_batch(data: Dict[str, torch.Tensor], num_samples: int) -> Dict[str, torch.Tensor]:
        """
        Replicate batch for batched sampling.
        
        Creates a batch of size (B * num_samples) where each image is repeated num_samples times.
        This allows sampling multiple rollouts per image in a single forward pass.
        
        Args:
            data: Original batch with tensors of shape (B, ...)
            num_samples: Number of times to replicate each sample
            
        Returns:
            Replicated data with tensors of shape (B * num_samples, ...)
        """
        replicated = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                # Repeat each sample num_samples times
                # (B, ...) -> (B * num_samples, ...)
                replicated[key] = value.repeat_interleave(num_samples, dim=0)
            elif isinstance(value, list):
                # For lists (like core_ids), repeat elements
                replicated[key] = [v for v in value for _ in range(num_samples)]
            else:
                replicated[key] = value
        return replicated
    
    @staticmethod
    def group_by_image(
        tensor: torch.Tensor, 
        batch_size: int, 
        num_samples: int
    ) -> torch.Tensor:
        """
        Reshape tensor from flat batch to grouped by image.
        
        Args:
            tensor: Tensor of shape (B * num_samples, ...)
            batch_size: Original batch size B
            num_samples: Number of samples per image
            
        Returns:
            Grouped tensor of shape (B, num_samples, ...)
        """
        shape = tensor.shape
        new_shape = (batch_size, num_samples) + shape[1:]
        return tensor.view(*new_shape)


class GRPOTrainer:
    """
    Legacy GRPO Trainer (kept for backward compatibility).
    Use BatchedGRPOTrainer for better performance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        grpo: Optional[GRPO] = None,
        num_update_epochs: int = 4,
        num_samples_per_image: int = 4,
        device: str = 'cuda',
    ):
        self.model = model
        self.optimizer = optimizer
        self.grpo = grpo if grpo is not None else GRPO(num_samples_per_image=num_samples_per_image)
        self.num_update_epochs = num_update_epochs
        self.num_samples_per_image = num_samples_per_image
        self.device = device
        
        logging.info(
            f"Initialized GRPOTrainer with {num_update_epochs} update epochs, "
            f"{num_samples_per_image} samples per image for within-image comparison"
        )
    
    def collect_rollout(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Collect a single rollout (legacy method)."""
        with torch.no_grad():
            outputs = self.model(
                data['bmode'],
                rf_image=data.get('rf'),
                prostate_mask=data['prostate_mask'],
                needle_mask=data['needle_mask'],
                deterministic=False,
                return_rl_info=True,
                **{k: v for k, v in data.items() if k in self.model.prompts}
            )
        
        rollout_data = {
            'log_probs': outputs['rl_log_probs'].detach(),
            'attention_coords': outputs['rl_attention_coords'].detach(),
        }
        
        return outputs, rollout_data


def create_grpo_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """
    Create optimizer for GRPO training.
    
    Args:
        model: The model
        lr: Learning rate
        weight_decay: Weight decay
        
    Returns:
        optimizer: AdamW optimizer
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    return optimizer
