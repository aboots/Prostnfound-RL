"""
RL-specific loss and reward computation for ProstNFound-RL

Note on Within-Image Comparison:
When using GRPO with within-image comparison (multiple samples per image),
reward normalization should be done in GRPO's compute_advantages(), not here.
Set normalize_rewards=False when using within-image comparison.

Key Improvements:
1. Reduced csPCa bonus (1.25x default instead of 2.0x) for more stable learning
2. Involvement-aware reward smoothing to handle noisy labels
3. Outside prostate penalty to keep attention within valid regions
4. Correct prediction bonus for improved reward signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from medAI.layers.masked_prediction_module import MaskedPredictionModule


class RLRewardComputer:
    """
    Computes rewards for RL training based on cancer detection performance.
    
    The reward function is designed to:
    1. Encourage correct cancer detection
    2. Moderately reward correct identification of clinically significant PCa (csPCa)
    3. Penalize false positives and false negatives
    4. Penalize attention points outside prostate region
    
    Args:
        reward_mode: Type of reward computation ('loss_based', 'accuracy_based', or 'combined')
        cspca_bonus: Bonus multiplier for csPCa cases (default: 1.25, reduced from 2.0)
            Lower values provide more stable learning as involvement labels can be noisy
        normalize_rewards: Whether to normalize rewards within batch (default: False)
            NOTE: When using within-image comparison in GRPO, set this to False
            so that normalization happens within each image group in GRPO.
        outside_prostate_penalty: Penalty for attention points outside prostate (default: 0.3)
            Applied as: reward = reward * (1 - penalty * outside_ratio)
        use_involvement_smoothing: Whether to smooth involvement-based rewards (default: True)
            Helps handle noisy involvement labels by using softer targets
        correct_pred_bonus: Bonus for correct binary predictions (default: 0.2)
            Provides additional reward signal when prediction direction is correct
    """
    
    def __init__(
        self,
        reward_mode: str = 'loss_based',
        cspca_bonus: float = 1.25,  # Reduced from 2.0 for more stable learning
        normalize_rewards: bool = False,  # Changed default to False for within-image comparison
        outside_prostate_penalty: float = 0.3,
        use_involvement_smoothing: bool = True,
        correct_pred_bonus: float = 0.2,
    ):
        self.reward_mode = reward_mode
        self.cspca_bonus = cspca_bonus
        self.normalize_rewards = normalize_rewards
        self.outside_prostate_penalty = outside_prostate_penalty
        self.use_involvement_smoothing = use_involvement_smoothing
        self.correct_pred_bonus = correct_pred_bonus
    
    def compute_loss_based_reward(
        self,
        cancer_logits: torch.Tensor,
        data: Dict[str, torch.Tensor],
        outside_prostate_ratio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reward based on BCE loss (lower loss = higher reward).
        
        Improvements over original:
        - Involvement smoothing to handle noisy labels
        - Correct prediction bonus for better gradient signal
        - Outside prostate penalty
        - Reduced csPCa bonus for stability
        
        Args:
            cancer_logits: Predicted cancer logits (B, 1, H, W)
            data: Batch data containing labels and masks
            outside_prostate_ratio: Fraction of attention points outside prostate (B,)
            
        Returns:
            rewards: Reward for each sample (B,)
        """
        B = cancer_logits.shape[0]
        device = cancer_logits.device
        
        prostate_mask = data['prostate_mask'].to(device)
        needle_mask = data['needle_mask'].to(device)
        involvement = data['involvement'].to(device)
        
        rewards = []
        correct_preds = []
        
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
                correct_preds.append(False)
                continue
            
            pred_prob = predictions.sigmoid().mean()
            target = involvement[i].item()
            
            # Involvement smoothing: soften targets to handle noise
            # Instead of using raw involvement (often 0 or 1), use smoothed version
            if self.use_involvement_smoothing:
                # Smooth towards 0.5 for uncertain cases
                # High involvement (>0.7) stays high, low (<0.3) stays low
                # Middle values get smoothed more
                if target > 0.7:
                    smoothed_target = 0.8 + 0.2 * (target - 0.7) / 0.3
                elif target < 0.3:
                    smoothed_target = 0.2 * target / 0.3
                else:
                    # Middle range: more smoothing towards 0.5
                    smoothed_target = 0.3 + 0.4 * (target - 0.3) / 0.4
                target = smoothed_target
            
            # BCE: -[y*log(p) + (1-y)*log(1-p)]
            bce = -(target * torch.log(pred_prob + 1e-8) + 
                   (1 - target) * torch.log(1 - pred_prob + 1e-8))
            
            # Reward is negative loss (scaled)
            reward = -bce.item() / 2.0  # Scale to roughly [-1, 0]
            
            # Check if prediction is correct (for bonus)
            is_correct = (pred_prob > 0.5) == (involvement[i].item() > 0.5)
            correct_preds.append(is_correct)
            
            rewards.append(reward)
        
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        
        # Add correct prediction bonus
        if self.correct_pred_bonus > 0:
            correct_mask = torch.tensor(correct_preds, device=device, dtype=torch.float32)
            rewards = rewards + self.correct_pred_bonus * correct_mask
        
        # Add csPCa bonus (now reduced to 1.25x default)
        if 'grade_group' in data:
            grade_group = data['grade_group'].to(device)
            cspca_mask = grade_group > 2
            rewards[cspca_mask] = rewards[cspca_mask] * self.cspca_bonus
        
        # Apply outside prostate penalty
        if outside_prostate_ratio is not None and self.outside_prostate_penalty > 0:
            # Penalty increases with ratio of points outside prostate
            penalty_factor = 1.0 - self.outside_prostate_penalty * outside_prostate_ratio.to(device)
            rewards = rewards * penalty_factor
        
        # Normalize rewards
        if self.normalize_rewards and len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        return rewards
    
    def compute_accuracy_based_reward(
        self,
        cancer_logits: torch.Tensor,
        data: Dict[str, torch.Tensor],
        outside_prostate_ratio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reward based on prediction accuracy.
        
        Args:
            cancer_logits: Predicted cancer logits (B, 1, H, W)
            data: Batch data
            outside_prostate_ratio: Fraction of attention points outside prostate (B,)
            
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
        
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        
        # Add csPCa bonus (reduced)
        if 'grade_group' in data:
            grade_group = data['grade_group'].to(device)
            cspca_mask = grade_group > 2
            rewards[cspca_mask] = rewards[cspca_mask] * self.cspca_bonus
        
        # Apply outside prostate penalty
        if outside_prostate_ratio is not None and self.outside_prostate_penalty > 0:
            penalty_factor = 1.0 - self.outside_prostate_penalty * outside_prostate_ratio.to(device)
            rewards = rewards * penalty_factor
        
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
                Can also contain 'rl_outside_prostate_ratio' for penalty
            data: Original batch data
            
        Returns:
            rewards: Rewards (B,)
        """
        # The outputs dict is actually the data dict with cancer_logits added by ProstNFoundMeta
        cancer_logits = outputs.get('cancer_logits', outputs.get('mask_logits'))
        
        if cancer_logits is None:
            raise KeyError("Could not find 'cancer_logits' or 'mask_logits' in outputs")
        
        # Get outside prostate ratio if available
        outside_prostate_ratio = outputs.get('rl_outside_prostate_ratio', None)
        
        if self.reward_mode == 'loss_based':
            return self.compute_loss_based_reward(cancer_logits, data, outside_prostate_ratio)
        elif self.reward_mode == 'accuracy_based':
            return self.compute_accuracy_based_reward(cancer_logits, data, outside_prostate_ratio)
        elif self.reward_mode == 'combined':
            loss_reward = self.compute_loss_based_reward(cancer_logits, data, outside_prostate_ratio)
            acc_reward = self.compute_accuracy_based_reward(cancer_logits, data, outside_prostate_ratio)
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
    
    Note:
        When using within-image comparison (rl_num_samples_per_image > 1),
        reward normalization is handled by GRPO, so we disable it here.
    """
    reward_mode = args.get('rl_reward_mode', 'loss_based')
    cspca_bonus = args.get('rl_cspca_bonus', 1.25)  # Reduced default from 2.0
    outside_prostate_penalty = args.get('rl_outside_prostate_penalty', 0.3)
    use_involvement_smoothing = args.get('rl_use_involvement_smoothing', True)
    correct_pred_bonus = args.get('rl_correct_pred_bonus', 0.2)
    
    # When using within-image comparison, normalization happens in GRPO
    num_samples_per_image = args.get('rl_num_samples_per_image', 4)
    if num_samples_per_image > 1:
        # Within-image comparison: GRPO handles normalization
        normalize_rewards = False
    else:
        # Fallback to batch normalization
        normalize_rewards = args.get('rl_normalize_rewards', True)
    
    return RLRewardComputer(
        reward_mode=reward_mode,
        cspca_bonus=cspca_bonus,
        normalize_rewards=normalize_rewards,
        outside_prostate_penalty=outside_prostate_penalty,
        use_involvement_smoothing=use_involvement_smoothing,
        correct_pred_bonus=correct_pred_bonus,
    )

