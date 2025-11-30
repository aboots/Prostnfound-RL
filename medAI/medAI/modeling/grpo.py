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

Reference: 
DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
https://arxiv.org/abs/2402.03300
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging


class GRPO:
    """
    Group Relative Policy Optimization with Within-Image Comparison.
    
    This implements GRPO for medical imaging tasks where:
    1. Each image is sampled multiple times (different attention locations)
    2. Rewards are computed for each sample
    3. Advantages are normalized WITHIN each image group (not across batch)
    4. This allows fair comparison of attention strategies per image
    
    Args:
        clip_eps: Clipping epsilon for PPO-style clipping (default: 0.2)
        entropy_coef: Coefficient for entropy bonus (default: 0.01)
        value_coef: Coefficient for value loss (default: 0.5)
        max_grad_norm: Maximum gradient norm for clipping (default: 0.5)
        gamma: Discount factor for future rewards (default: 1.0, no discounting)
        normalize_advantages: Whether to normalize advantages (default: True)
        num_samples_per_image: Number of rollouts per image for within-image comparison (default: 1)
    """
    
    def __init__(
        self,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gamma: float = 1.0,
        normalize_advantages: bool = True,
        num_samples_per_image: int = 1,
    ):
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.normalize_advantages = normalize_advantages
        self.num_samples_per_image = num_samples_per_image
        
        logging.info(
            f"Initialized GRPO with clip_eps={clip_eps}, entropy_coef={entropy_coef}, "
            f"value_coef={value_coef}, gamma={gamma}, num_samples_per_image={num_samples_per_image}"
        )
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        num_samples_per_image: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages with within-image normalization.
        
        For within-image comparison, we have multiple samples per image.
        Advantages are normalized within each image group, not across the batch.
        This accounts for different difficulty levels across images.
        
        Args:
            rewards: Rewards for each sample (B * num_samples,) or (B,)
            values: Value estimates for each sample (B * num_samples, 1) or (B, 1)
            num_samples_per_image: Number of samples per image (overrides self.num_samples_per_image)
            
        Returns:
            advantages: Computed advantages (B * num_samples,) or (B,)
            returns: Target returns for value function (B * num_samples,) or (B,)
        """
        if values.ndim > 1:
            values = values.squeeze(-1)
        
        # For single-step case, return = reward
        returns = rewards
        
        # Advantage = return - value
        advantages = returns - values
        
        num_samples = num_samples_per_image if num_samples_per_image is not None else self.num_samples_per_image
        
        # Normalize advantages
        if self.normalize_advantages:
            if num_samples > 1:
                # Within-image normalization: normalize within each image group
                total_samples = advantages.shape[0]
                num_images = total_samples // num_samples
                
                if total_samples == num_images * num_samples:
                    # Reshape to (num_images, num_samples)
                    advantages_grouped = advantages.view(num_images, num_samples)
                    
                    # Normalize within each image group
                    group_mean = advantages_grouped.mean(dim=1, keepdim=True)
                    group_std = advantages_grouped.std(dim=1, keepdim=True) + 1e-8
                    advantages_grouped = (advantages_grouped - group_mean) / group_std
                    
                    # Flatten back
                    advantages = advantages_grouped.view(-1)
                else:
                    # Fallback to batch normalization if shapes don't match
                    logging.warning(
                        f"Within-image normalization failed: total_samples={total_samples}, "
                        f"num_images={num_images}, num_samples={num_samples}. "
                        "Falling back to batch normalization."
                    )
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                # Standard batch normalization (fallback for num_samples=1)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
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
        
        # Compute entropy bonus (encourages exploration)
        # For categorical policy, entropy is -sum(p * log(p))
        # We approximate using the log_probs
        entropy = -log_probs.mean()
        
        # Total loss
        total_loss = policy_loss - self.entropy_coef * entropy
        
        # Logging info
        info = {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_min': ratio.min().item(),
            'ratio_max': ratio.max().item(),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
        }
        
        return total_loss, info
    
    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute value function loss (MSE).
        
        Args:
            values: Predicted values (B, 1) or (B,)
            returns: Target returns (B,)
            
        Returns:
            loss: Value loss
        """
        if values.ndim > 1:
            values = values.squeeze(-1)
        
        value_loss = F.mse_loss(values, returns)
        return value_loss
    
    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        num_samples_per_image: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total GRPO loss with within-image comparison.
        
        Args:
            log_probs: Current log probabilities (B * num_samples, k)
            old_log_probs: Old log probabilities (B * num_samples, k)
            values: Value estimates (B * num_samples, 1) or (B * num_samples,)
            rewards: Rewards (B * num_samples,)
            num_samples_per_image: Number of samples per image for within-image normalization
            
        Returns:
            total_loss: Combined policy and value loss
            info: Dictionary with loss components
        """
        # Compute advantages with within-image normalization
        advantages, returns = self.compute_advantages(
            rewards, values, num_samples_per_image=num_samples_per_image
        )
        
        # Compute policy loss
        policy_loss, policy_info = self.compute_policy_loss(
            log_probs, old_log_probs, advantages
        )
        
        # Compute value loss
        value_loss = self.compute_value_loss(values, returns)
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss
        
        # Combine info
        info = policy_info
        info['value_loss'] = value_loss.item()
        info['total_loss'] = total_loss.item()
        
        # Add within-image comparison info
        if num_samples_per_image is not None and num_samples_per_image > 1:
            info['num_samples_per_image'] = num_samples_per_image
        
        return total_loss, info
    
    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform a GRPO update step.
        
        Args:
            model: The model being trained
            optimizer: The optimizer
            log_probs: Current log probabilities (B, k)
            old_log_probs: Old log probabilities (B, k)
            values: Value estimates (B, 1)
            rewards: Rewards (B,)
            
        Returns:
            info: Dictionary with training metrics
        """
        # Compute loss
        loss, info = self.compute_loss(log_probs, old_log_probs, values, rewards)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_grad_norm
        )
        info['grad_norm'] = grad_norm.item()
        
        # Optimizer step
        optimizer.step()
        
        return info


class GRPOTrainer:
    """
    Trainer for GRPO that manages rollout collection and multiple update epochs.
    
    Supports within-image comparison: instead of comparing performance across different
    images (which have different difficulty), we sample multiple rollouts per image
    and compare them within each image.
    
    Args:
        model: The RL model (ProstNFoundRL)
        optimizer: Optimizer for model parameters
        grpo: GRPO algorithm instance
        num_update_epochs: Number of epochs to train on each batch (default: 4)
        num_samples_per_image: Number of rollouts per image for within-image comparison (default: 4)
        device: Device to use (default: 'cuda')
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
    
    def collect_rollouts(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Collect multiple rollouts per image for within-image comparison.
        
        Args:
            data: Batch of data
            
        Returns:
            rollout_data: Data needed for GRPO update (log_probs, values, etc.)
                          Shapes are (B * num_samples_per_image, ...)
            outputs_list: List of outputs for reward computation
        """
        B = data['bmode'].shape[0]
        all_log_probs = []
        all_values = []
        all_outputs = []
        
        # Sample multiple rollouts per image
        with torch.no_grad():
            for sample_idx in range(self.num_samples_per_image):
                outputs = self.model(
                    data['bmode'],
                    rf_image=data.get('rf'),
                    prostate_mask=data['prostate_mask'],
                    needle_mask=data['needle_mask'],
                    deterministic=False,  # Sample from policy
                    return_rl_info=True,
                    **{k: v for k, v in data.items() if k in self.model.prompts}
                )
                
                all_log_probs.append(outputs['rl_log_probs'])
                all_values.append(outputs['rl_value'])
                all_outputs.append(outputs)
        
        # Stack and reshape: (B, num_samples, ...) -> (B * num_samples, ...)
        stacked_log_probs = torch.stack(all_log_probs, dim=1)  # (B, num_samples, k)
        stacked_values = torch.stack(all_values, dim=1)  # (B, num_samples, 1)
        
        rollout_data = {
            'log_probs': stacked_log_probs.view(B * self.num_samples_per_image, -1).detach(),
            'values': stacked_values.view(B * self.num_samples_per_image, -1).detach(),
        }
        
        return rollout_data, all_outputs
    
    def collect_rollout(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Collect a single rollout by running the model on data.
        (Legacy method for backward compatibility)
        
        Args:
            data: Batch of data
            
        Returns:
            outputs: Model outputs
            rollout_data: Data needed for GRPO update (log_probs, values, etc.)
        """
        # Run model in sampling mode
        with torch.no_grad():
            outputs = self.model(
                data['bmode'],
                rf_image=data.get('rf'),
                prostate_mask=data['prostate_mask'],
                needle_mask=data['needle_mask'],
                deterministic=False,  # Sample from policy
                return_rl_info=True,
                **{k: v for k, v in data.items() if k in self.model.prompts}
            )
        
        # Extract RL info
        rollout_data = {
            'log_probs': outputs['rl_log_probs'].detach(),
            'values': outputs['rl_value'].detach(),
            'attention_coords': outputs['rl_attention_coords'].detach(),
        }
        
        return outputs, rollout_data
    
    def compute_rewards(
        self,
        outputs: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor],
        criterion: nn.Module,
    ) -> torch.Tensor:
        """
        Compute rewards based on model predictions and ground truth.
        
        The reward is designed to:
        1. Give positive reward for correct predictions (low loss)
        2. Give higher reward for correctly identifying csPCa
        3. Penalize incorrect predictions
        
        Args:
            outputs: Model outputs
            data: Batch data with labels
            criterion: Loss function
            
        Returns:
            rewards: Reward for each sample in batch (B,)
        """
        B = len(data['bmode'])
        device = data['bmode'].device
        
        # Prepare data dict for loss computation
        loss_data = dict(data)
        loss_data['cancer_logits'] = outputs['mask_logits']
        if 'cls_outputs' in outputs:
            loss_data['image_level_classification_outputs'] = outputs['cls_outputs']
        
        # Compute per-sample losses
        # We need to compute loss for each sample individually
        rewards = []
        for i in range(B):
            sample_data = {k: v[i:i+1] for k, v in loss_data.items()}
            
            try:
                loss = criterion(sample_data)
                # Reward is negative loss (higher is better)
                reward = -loss.item()
            except Exception as e:
                logging.warning(f"Error computing loss for reward: {e}")
                reward = 0.0
            
            # Scale reward for numerical stability
            # Typical BCE losses are in range [0, 5], so we scale to [-1, 0]
            reward = reward / 5.0
            
            # Bonus for csPCa cases (grade_group > 2)
            if 'grade_group' in data:
                if data['grade_group'][i] > 2:
                    # Give 2x weight to csPCa cases
                    reward = reward * 2.0
            
            rewards.append(reward)
        
        rewards = torch.tensor(rewards, device=device)
        
        return rewards
    
    def compute_rewards_for_rollouts(
        self,
        outputs_list: list,
        data: Dict[str, torch.Tensor],
        criterion: nn.Module,
    ) -> torch.Tensor:
        """
        Compute rewards for multiple rollouts per image.
        
        Args:
            outputs_list: List of outputs from each rollout
            data: Batch data with labels
            criterion: Loss function
            
        Returns:
            rewards: Rewards shaped (B * num_samples_per_image,)
        """
        B = len(data['bmode'])
        all_rewards = []
        
        for outputs in outputs_list:
            rewards = self.compute_rewards(outputs, data, criterion)
            all_rewards.append(rewards)
        
        # Stack and reshape: (num_samples, B) -> (B * num_samples,)
        stacked_rewards = torch.stack(all_rewards, dim=1)  # (B, num_samples)
        return stacked_rewards.view(B * self.num_samples_per_image)
    
    def train_step(
        self,
        data: Dict[str, torch.Tensor],
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """
        Perform one GRPO training step with within-image comparison.
        
        This involves:
        1. Collecting multiple rollouts per image (sampling different attention locations)
        2. Computing rewards for each rollout
        3. Multiple update epochs with GRPO using within-image advantage normalization
        
        Args:
            data: Batch of data
            criterion: Loss function for computing rewards
            
        Returns:
            info: Dictionary with training metrics
        """
        B = data['bmode'].shape[0]
        
        # Step 1: Collect multiple rollouts per image
        rollout_data, outputs_list = self.collect_rollouts(data)
        
        # Step 2: Compute rewards for all rollouts
        rewards = self.compute_rewards_for_rollouts(outputs_list, data, criterion)
        
        # Step 3: Multiple update epochs with within-image comparison
        update_info_list = []
        for epoch in range(self.num_update_epochs):
            # Re-run model for each sample to get current policy
            self.model.train()
            current_log_probs_list = []
            current_values_list = []
            
            for sample_idx in range(self.num_samples_per_image):
                outputs = self.model(
                    data['bmode'],
                    rf_image=data.get('rf'),
                    prostate_mask=data['prostate_mask'],
                    needle_mask=data['needle_mask'],
                    deterministic=False,
                    return_rl_info=True,
                    **{k: v for k, v in data.items() if k in self.model.prompts}
                )
                current_log_probs_list.append(outputs['rl_log_probs'])
                current_values_list.append(outputs['rl_value'])
            
            # Stack current samples
            current_log_probs = torch.stack(current_log_probs_list, dim=1).view(
                B * self.num_samples_per_image, -1
            )
            current_values = torch.stack(current_values_list, dim=1).view(
                B * self.num_samples_per_image, -1
            )
            
            # Compute GRPO loss with within-image comparison
            loss, info = self.grpo.compute_loss(
                current_log_probs,
                rollout_data['log_probs'],
                current_values,
                rewards,
                num_samples_per_image=self.num_samples_per_image,
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grpo.max_grad_norm
            )
            info['grad_norm'] = grad_norm.item()
            
            # Optimizer step
            self.optimizer.step()
            
            update_info_list.append(info)
        
        # Average metrics across update epochs
        avg_info = {}
        for key in update_info_list[0].keys():
            avg_info[key] = sum(info[key] for info in update_info_list) / len(update_info_list)
        
        # Add reward statistics
        avg_info['reward_mean'] = rewards.mean().item()
        avg_info['reward_std'] = rewards.std().item()
        avg_info['reward_min'] = rewards.min().item()
        avg_info['reward_max'] = rewards.max().item()
        avg_info['num_samples_per_image'] = self.num_samples_per_image
        
        # Compute within-image reward variance
        rewards_per_image = rewards.view(B, self.num_samples_per_image)
        avg_info['within_image_reward_std'] = rewards_per_image.std(dim=1).mean().item()
        
        return avg_info


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
    # Typically we only optimize the policy network during RL training
    # The base model can be frozen or trained with a much lower learning rate
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    return optimizer

