"""
Group Relative Policy Optimization (GRPO) for ProstNFound-RL

GRPO is a policy gradient method that normalizes advantages within groups (batches)
to improve training stability. It's particularly useful for on-policy RL in domains
with high variance rewards.

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
    Group Relative Policy Optimization.
    
    This implements a simplified version of GRPO suitable for our medical imaging task.
    The key idea is to:
    1. Sample multiple rollouts per example
    2. Compute rewards for each rollout
    3. Normalize advantages within each group (batch)
    4. Update policy using policy gradient with the normalized advantages
    
    Args:
        clip_eps: Clipping epsilon for PPO-style clipping (default: 0.2)
        entropy_coef: Coefficient for entropy bonus (default: 0.01)
        value_coef: Coefficient for value loss (default: 0.5)
        max_grad_norm: Maximum gradient norm for clipping (default: 0.5)
        gamma: Discount factor for future rewards (default: 1.0, no discounting)
        normalize_advantages: Whether to normalize advantages (default: True)
    """
    
    def __init__(
        self,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gamma: float = 1.0,
        normalize_advantages: bool = True,
    ):
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.normalize_advantages = normalize_advantages
        
        logging.info(
            f"Initialized GRPO with clip_eps={clip_eps}, entropy_coef={entropy_coef}, "
            f"value_coef={value_coef}, gamma={gamma}"
        )
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using simple advantage estimation.
        
        For our task, we have single-step rewards (no temporal structure),
        so the advantage is simply reward - value_estimate.
        
        Args:
            rewards: Rewards for each sample (B,)
            values: Value estimates for each sample (B, 1)
            
        Returns:
            advantages: Computed advantages (B,)
            returns: Target returns for value function (B,)
        """
        if values.ndim > 1:
            values = values.squeeze(-1)
        
        # For single-step case, return = reward
        returns = rewards
        
        # Advantage = return - value
        advantages = returns - values
        
        # Normalize advantages within the batch (key idea of GRPO)
        if self.normalize_advantages:
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
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total GRPO loss.
        
        Args:
            log_probs: Current log probabilities (B, k)
            old_log_probs: Old log probabilities (B, k)
            values: Value estimates (B, 1) or (B,)
            rewards: Rewards (B,)
            
        Returns:
            total_loss: Combined policy and value loss
            info: Dictionary with loss components
        """
        # Compute advantages
        advantages, returns = self.compute_advantages(rewards, values)
        
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
    
    Args:
        model: The RL model (ProstNFoundRL)
        optimizer: Optimizer for model parameters
        grpo: GRPO algorithm instance
        num_update_epochs: Number of epochs to train on each batch (default: 4)
        device: Device to use (default: 'cuda')
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        grpo: Optional[GRPO] = None,
        num_update_epochs: int = 4,
        device: str = 'cuda',
    ):
        self.model = model
        self.optimizer = optimizer
        self.grpo = grpo if grpo is not None else GRPO()
        self.num_update_epochs = num_update_epochs
        self.device = device
        
        logging.info(f"Initialized GRPOTrainer with {num_update_epochs} update epochs")
    
    def collect_rollout(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Collect a rollout by running the model on data.
        
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
    
    def train_step(
        self,
        data: Dict[str, torch.Tensor],
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """
        Perform one GRPO training step.
        
        This involves:
        1. Collecting rollout (sampling from policy)
        2. Computing rewards
        3. Multiple update epochs with GRPO
        
        Args:
            data: Batch of data
            criterion: Loss function for computing rewards
            
        Returns:
            info: Dictionary with training metrics
        """
        # Step 1: Collect rollout
        outputs, rollout_data = self.collect_rollout(data)
        
        # Step 2: Compute rewards
        rewards = self.compute_rewards(outputs, data, criterion)
        
        # Step 3: Multiple update epochs
        update_info_list = []
        for epoch in range(self.num_update_epochs):
            # Re-run model with same data to get current log_probs
            self.model.train()
            outputs = self.model(
                data['bmode'],
                rf_image=data.get('rf'),
                prostate_mask=data['prostate_mask'],
                needle_mask=data['needle_mask'],
                deterministic=False,
                return_rl_info=True,
                **{k: v for k, v in data.items() if k in self.model.prompts}
            )
            
            # Perform GRPO update
            update_info = self.grpo.update(
                model=self.model,
                optimizer=self.optimizer,
                log_probs=outputs['rl_log_probs'],
                old_log_probs=rollout_data['log_probs'],
                values=outputs['rl_value'],
                rewards=rewards,
            )
            update_info_list.append(update_info)
        
        # Average metrics across update epochs
        avg_info = {}
        for key in update_info_list[0].keys():
            avg_info[key] = sum(info[key] for info in update_info_list) / len(update_info_list)
        
        # Add reward statistics
        avg_info['reward_mean'] = rewards.mean().item()
        avg_info['reward_std'] = rewards.std().item()
        avg_info['reward_min'] = rewards.min().item()
        avg_info['reward_max'] = rewards.max().item()
        
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

