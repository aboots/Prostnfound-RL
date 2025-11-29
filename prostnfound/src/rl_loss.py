"""
RL-specific loss and reward computation for ProstNFound-RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from medAI.layers.masked_prediction_module import MaskedPredictionModule


class RLRewardComputer:
    """
    Computes rewards for RL training based on cancer detection performance.
    
    The reward function is designed to:
    1. Encourage correct cancer detection
    2. Heavily reward correct identification of clinically significant PCa (csPCa)
    3. Penalize false positives and false negatives
    
    Args:
        reward_mode: Type of reward computation ('loss_based', 'accuracy_based', or 'combined')
        cspca_bonus: Bonus multiplier for csPCa cases (default: 2.0)
        normalize_rewards: Whether to normalize rewards within batch (default: True)
    """
    
    def __init__(
        self,
        reward_mode: str = 'loss_based',
        cspca_bonus: float = 2.0,
        normalize_rewards: bool = True,
    ):
        self.reward_mode = reward_mode
        self.cspca_bonus = cspca_bonus
        self.normalize_rewards = normalize_rewards
    
    def compute_loss_based_reward(
        self,
        cancer_logits: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute reward based on BCE loss (lower loss = higher reward).
        
        Args:
            cancer_logits: Predicted cancer logits (B, 1, H, W)
            data: Batch data containing labels and masks
            
        Returns:
            rewards: Reward for each sample (B,)
        """
        B = cancer_logits.shape[0]
        device = cancer_logits.device
        
        prostate_mask = data['prostate_mask'].to(device)
        needle_mask = data['needle_mask'].to(device)
        involvement = data['involvement'].to(device)
        
        rewards = []
        for i in range(B):
            # Get mask for valid region
            mask = torch.ones(prostate_mask[i].shape, device=device).bool()
            mask &= prostate_mask[i] > 0.5
            mask &= needle_mask[i] > 0.5
            
            # Get predictions in valid region
            predictions, _ = MaskedPredictionModule()(
                cancer_logits[i:i+1], mask[None, ...]
            )
            
            if len(predictions) == 0:
                rewards.append(0.0)
                continue
            
            # Compute BCE loss
            target = involvement[i].item()
            pred_prob = predictions.sigmoid().mean()
            
            # BCE: -[y*log(p) + (1-y)*log(1-p)]
            bce = -(target * torch.log(pred_prob + 1e-8) + 
                   (1 - target) * torch.log(1 - pred_prob + 1e-8))
            
            # Reward is negative loss (scaled)
            reward = -bce.item() / 2.0  # Scale to roughly [-1, 0]
            
            rewards.append(reward)
        
        rewards = torch.tensor(rewards, device=device)
        
        # Add csPCa bonus
        if 'grade_group' in data:
            cspca_mask = data['grade_group'] > 2
            rewards[cspca_mask] *= self.cspca_bonus
        
        # Normalize rewards
        if self.normalize_rewards and len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        return rewards
    
    def compute_accuracy_based_reward(
        self,
        cancer_logits: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute reward based on prediction accuracy.
        
        Args:
            cancer_logits: Predicted cancer logits (B, 1, H, W)
            data: Batch data
            
        Returns:
            rewards: Reward for each sample (B,)
        """
        B = cancer_logits.shape[0]
        device = cancer_logits.device
        
        prostate_mask = data['prostate_mask'].to(device)
        needle_mask = data['needle_mask'].to(device)
        label = data['label'].to(device)
        
        rewards = []
        for i in range(B):
            # Get mask for valid region
            mask = torch.ones(prostate_mask[i].shape, device=device).bool()
            mask &= prostate_mask[i] > 0.5
            mask &= needle_mask[i] > 0.5
            
            # Get predictions in valid region
            predictions, _ = MaskedPredictionModule()(
                cancer_logits[i:i+1], mask[None, ...]
            )
            
            if len(predictions) == 0:
                rewards.append(0.0)
                continue
            
            # Get prediction
            pred_prob = predictions.sigmoid().mean()
            pred = (pred_prob > 0.5).float()
            
            # Reward: +1 for correct, -1 for incorrect
            reward = 2.0 * (pred == label[i].float()).float().item() - 1.0
            
            rewards.append(reward)
        
        rewards = torch.tensor(rewards, device=device)
        
        # Add csPCa bonus
        if 'grade_group' in data:
            cspca_mask = data['grade_group'] > 2
            rewards[cspca_mask] *= self.cspca_bonus
        
        return rewards
    
    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute rewards for a batch.
        
        Args:
            outputs: Model outputs (data dict with 'cancer_logits' added)
            data: Original batch data
            
        Returns:
            rewards: Rewards (B,)
        """
        # The outputs dict is actually the data dict with cancer_logits added by ProstNFoundMeta
        cancer_logits = outputs.get('cancer_logits', outputs.get('mask_logits'))
        
        if cancer_logits is None:
            raise KeyError("Could not find 'cancer_logits' or 'mask_logits' in outputs")
        
        if self.reward_mode == 'loss_based':
            return self.compute_loss_based_reward(cancer_logits, data)
        elif self.reward_mode == 'accuracy_based':
            return self.compute_accuracy_based_reward(cancer_logits, data)
        elif self.reward_mode == 'combined':
            loss_reward = self.compute_loss_based_reward(cancer_logits, data)
            acc_reward = self.compute_accuracy_based_reward(cancer_logits, data)
            return 0.5 * loss_reward + 0.5 * acc_reward
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")


class RLLoss(nn.Module):
    """
    Combined loss for RL training.
    
    This combines:
    1. Standard ProstNFound loss (for supervised signal)
    2. RL policy gradient loss (from GRPO)
    
    Args:
        base_criterion: Base loss function (e.g., MIL loss)
        rl_weight: Weight for RL loss component (default: 1.0)
        supervised_weight: Weight for supervised loss (default: 1.0)
    """
    
    def __init__(
        self,
        base_criterion: nn.Module,
        rl_weight: float = 1.0,
        supervised_weight: float = 1.0,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.rl_weight = rl_weight
        self.supervised_weight = supervised_weight
    
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            data: Batch data with model outputs
            
        Returns:
            loss: Combined loss
        """
        # Supervised loss
        supervised_loss = self.base_criterion(data)
        
        # RL loss is computed separately in GRPO trainer
        # Here we just return the supervised component
        loss = self.supervised_weight * supervised_loss
        
        return loss


def build_rl_reward_computer(args) -> RLRewardComputer:
    """
    Build reward computer from config.
    
    Args:
        args: Config with RL settings
        
    Returns:
        reward_computer: RLRewardComputer instance
    """
    reward_mode = args.get('rl_reward_mode', 'loss_based')
    cspca_bonus = args.get('rl_cspca_bonus', 2.0)
    normalize_rewards = args.get('rl_normalize_rewards', True)
    
    return RLRewardComputer(
        reward_mode=reward_mode,
        cspca_bonus=cspca_bonus,
        normalize_rewards=normalize_rewards,
    )

