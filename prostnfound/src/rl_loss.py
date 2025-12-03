"""
RL-specific loss and reward computation for ProstNFound-RL

Key Features (v2):
1. Pure GRPO rewards (no value function needed)
2. Combined reward: classification + ROI involvement
3. Configurable prostate boundary penalty (can be disabled)
4. Better reward normalization for within-image comparison
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
    1. Reward good classification performance (image-level)
    2. Reward correct ROI involvement prediction (region awareness)
    3. Optionally penalize attention outside prostate
    4. Optionally reward diversity between samples (prevents policy collapse)
    
    Reward Modes:
    - 'combined_v2': Classification + ROI involvement (RECOMMENDED)
    - 'confidence_based': Reward high confidence on correct predictions
    - 'classification_only': Only classification head performance
    - 'loss_based': Negative BCE loss (legacy)
    
    Args:
        reward_mode: Type of reward computation (default: 'combined_v2')
        cspca_bonus: Bonus multiplier for csPCa cases (default: 2.0)
        normalize_rewards: Whether to normalize rewards (default: False for GRPO)
        prostate_boundary_penalty_weight: Weight for outside-prostate penalty (default: 0.0 to disable)
        heatmap_reward_weight: Weight for heatmap/ROI performance (default: 0.5)
        classification_reward_weight: Weight for classification performance (default: 0.5)
        diversity_reward_weight: Weight for within-image diversity bonus (default: 0.0)
    """
    
    def __init__(
        self,
        reward_mode: str = 'combined_v2',
        cspca_bonus: float = 2.0,
        normalize_rewards: bool = False,
        prostate_boundary_penalty_weight: float = 0.0,  # Disabled by default
        prostate_boundary_penalty_scale: float = 10.0,
        heatmap_reward_weight: float = 0.5,
        classification_reward_weight: float = 0.5,
        diversity_reward_weight: float = 0.0,  # Diversity bonus (prevents policy collapse)
    ):
        self.reward_mode = reward_mode
        self.cspca_bonus = cspca_bonus
        self.normalize_rewards = normalize_rewards
        self.prostate_boundary_penalty_weight = prostate_boundary_penalty_weight
        self.prostate_boundary_penalty_scale = prostate_boundary_penalty_scale
        self.heatmap_reward_weight = heatmap_reward_weight
        self.classification_reward_weight = classification_reward_weight
        self.diversity_reward_weight = diversity_reward_weight
    
    def compute_roi_involvement_reward(
        self,
        cancer_logits: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute reward based on ROI involvement prediction.
        
        This rewards the model for:
        - Predicting high involvement in positive cases
        - Predicting low involvement in negative cases
        - Accuracy of involvement level prediction
        
        Args:
            cancer_logits: Predicted cancer logits (B, 1, H, W)
            data: Batch data containing involvement labels and masks
            
        Returns:
            rewards: Reward for each sample (B,)
        """
        B = cancer_logits.shape[0]
        device = cancer_logits.device
        
        prostate_mask = data['prostate_mask'].to(device)
        needle_mask = data['needle_mask'].to(device)
        involvement = data['involvement'].to(device).float()
        
        rewards = []
        for i in range(B):
            # Get mask for valid region (prostate AND needle)
            mask = (prostate_mask[i] > 0.5) & (needle_mask[i] > 0.5)
            
            # Get predictions in valid region
            predictions, _ = MaskedPredictionModule()(
                cancer_logits[i:i+1], mask[None, ...]
            )
            
            if len(predictions) == 0:
                rewards.append(0.0)
                continue
            
            # Get mean prediction probability
            pred_prob = predictions.sigmoid().mean()
            target_involvement = involvement[i].item()
            
            # ROI involvement reward:
            # Reward = 1 - |predicted_involvement - true_involvement|
            # This gives +1 for perfect prediction, 0 for worst case
            involvement_error = torch.abs(pred_prob - target_involvement)
            reward = (1.0 - involvement_error).item()
            
            # Scale to [-1, 1]
            reward = 2.0 * reward - 1.0
            
            rewards.append(reward)
        
        return torch.tensor(rewards, device=device)
    
    def compute_classification_reward(
        self,
        outputs: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute reward based on classification head performance.
        
        Uses confidence on correct class as reward signal.
        
        Args:
            outputs: Model outputs with classification logits
            data: Batch data with labels
            
        Returns:
            rewards: Classification rewards (B,)
        """
        if 'image_level_classification_outputs' not in outputs:
            # No classification head, return zeros
            B = outputs['cancer_logits'].shape[0]
            return torch.zeros(B, device=outputs['cancer_logits'].device)
        
        cls_logits = outputs['image_level_classification_outputs'][0]  # (B, num_classes)
        device = cls_logits.device
        B = cls_logits.shape[0]
        
        # Get labels based on classification mode
        if 'grade_group' in data:
            # csPCa classification (grade_group > 2)
            labels = (data['grade_group'].to(device) > 2).long()
        else:
            # Binary cancer classification
            labels = data['label'].to(device).long()
        
        # Handle label dimension
        if labels.ndim > 1:
            labels = labels.squeeze(-1)
        
        # Compute probabilities
        probs = F.softmax(cls_logits, dim=1)
        
        # Reward based on confidence for correct class
        rewards = []
        for i in range(B):
            label_i = labels[i].item()
            prob_correct = probs[i, int(label_i)].item()
            
            # Reward = 2 * prob - 1 (scales to [-1, 1])
            reward = 2.0 * prob_correct - 1.0
            rewards.append(reward)
        
        return torch.tensor(rewards, device=device)
    
    def compute_combined_v2_reward(
        self,
        outputs: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute combined reward: classification + ROI involvement.
        
        This is the recommended reward for RL training as it:
        1. Provides signal from both heads (classifier and heatmap)
        2. Rewards accurate involvement prediction
        3. Works well with within-image comparison
        
        Args:
            outputs: Model outputs
            data: Batch data
            
        Returns:
            rewards: Combined rewards (B,)
        """
        cancer_logits = outputs.get('cancer_logits', outputs.get('mask_logits'))
        device = cancer_logits.device
        
        # ROI involvement reward
        roi_reward = self.compute_roi_involvement_reward(cancer_logits, data)
        
        # Classification reward
        cls_reward = self.compute_classification_reward(outputs, data)
        
        # Combine rewards
        rewards = (
            self.heatmap_reward_weight * roi_reward + 
            self.classification_reward_weight * cls_reward
        )
        
        # csPCa bonus
        if 'grade_group' in data:
            cspca_mask = data['grade_group'].to(device) > 2
            if cspca_mask.ndim > 1:
                cspca_mask = cspca_mask.squeeze(-1)
            rewards = torch.where(cspca_mask, rewards * self.cspca_bonus, rewards)
        
        return rewards
    
    def compute_confidence_based_reward(
        self,
        cancer_logits: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute reward based on confidence calibration.
        
        Rewards high confidence on correct predictions.
        """
        B = cancer_logits.shape[0]
        device = cancer_logits.device
        
        prostate_mask = data['prostate_mask'].to(device)
        needle_mask = data['needle_mask'].to(device)
        involvement = data['involvement'].to(device)
        
        rewards = []
        for i in range(B):
            mask = (prostate_mask[i] > 0.5) & (needle_mask[i] > 0.5)
            
            predictions, _ = MaskedPredictionModule()(
                cancer_logits[i:i+1], mask[None, ...]
            )
            
            if len(predictions) == 0:
                rewards.append(0.0)
                continue
            
            pred_prob = predictions.sigmoid().mean()
            target = involvement[i].item()
            
            if target > 0.5:
                reward = pred_prob.item()
            else:
                reward = (1.0 - pred_prob).item()
            
            reward = 2.0 * reward - 1.0
            rewards.append(reward)
        
        rewards = torch.tensor(rewards, device=device)
        
        if 'grade_group' in data:
            cspca_mask = data['grade_group'].to(device) > 2
            if cspca_mask.ndim > 1:
                cspca_mask = cspca_mask.squeeze(-1)
            rewards = torch.where(cspca_mask, rewards * self.cspca_bonus, rewards)
        
        return rewards
    
    def compute_diversity_reward(
        self,
        rl_coords: torch.Tensor,
        num_samples_per_image: int,
    ) -> torch.Tensor:
        """
        Compute diversity reward for attention coordinates within each image.
        
        This encourages the policy to explore different regions across samples,
        preventing policy collapse where all samples converge to the same locations.
        
        The reward is based on the average pairwise distance between attention
        points from different samples of the same image.
        
        Args:
            rl_coords: Attention coordinates (B * num_samples, num_points, 2)
            num_samples_per_image: Number of samples per image
            
        Returns:
            diversity_rewards: Diversity bonus for each sample (B * num_samples,)
        """
        if rl_coords is None or num_samples_per_image <= 1:
            return torch.zeros(rl_coords.shape[0] if rl_coords is not None else 1, 
                             device=rl_coords.device if rl_coords is not None else 'cpu')
        
        total_samples = rl_coords.shape[0]
        num_images = total_samples // num_samples_per_image
        num_points = rl_coords.shape[1]
        device = rl_coords.device
        
        # Reshape to (num_images, num_samples_per_image, num_points, 2)
        coords_per_image = rl_coords.view(num_images, num_samples_per_image, num_points, 2)
        
        diversity_rewards = []
        
        for img_idx in range(num_images):
            # Get all samples for this image: (num_samples, num_points, 2)
            img_coords = coords_per_image[img_idx]
            
            # Compute pairwise distances between samples
            # Use mean of all attention points as the "center" of each sample
            sample_centers = img_coords.mean(dim=1)  # (num_samples, 2)
            
            # Compute pairwise distances between sample centers
            # distances[i, j] = ||center_i - center_j||
            pairwise_dists = torch.cdist(sample_centers.unsqueeze(0), sample_centers.unsqueeze(0)).squeeze(0)
            
            # Get upper triangle (avoid counting pairs twice and self-distances)
            mask = torch.triu(torch.ones_like(pairwise_dists), diagonal=1).bool()
            valid_dists = pairwise_dists[mask]
            
            if len(valid_dists) > 0:
                # Average distance, normalized by image size (256)
                avg_dist = valid_dists.mean() / 256.0
                # Scale to [0, 1] range (max theoretical distance is ~sqrt(2) ≈ 1.41 when normalized)
                diversity_score = torch.clamp(avg_dist / 0.5, 0, 1)  # 0.5 is a reasonable target distance
            else:
                diversity_score = torch.tensor(0.0, device=device)
            
            # Same diversity reward for all samples from this image
            for _ in range(num_samples_per_image):
                diversity_rewards.append(diversity_score)
        
        return torch.stack(diversity_rewards).to(device)
    
    def compute_prostate_boundary_penalty(
        self,
        rl_coords: torch.Tensor,
        prostate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute penalty for attention coordinates outside prostate.
        
        Only used if prostate_boundary_penalty_weight > 0.
        """
        if rl_coords is None or self.prostate_boundary_penalty_weight == 0:
            return torch.zeros(prostate_mask.shape[0], device=prostate_mask.device)
        
        B, num_points, _ = rl_coords.shape
        _, _, H_mask, W_mask = prostate_mask.shape
        device = prostate_mask.device
        
        # Resize mask to coordinate space
        COORD_SPACE_SIZE = 256
        if H_mask != COORD_SPACE_SIZE or W_mask != COORD_SPACE_SIZE:
            prostate_mask = F.interpolate(
                prostate_mask.float(),
                size=(COORD_SPACE_SIZE, COORD_SPACE_SIZE),
                mode='nearest'
            )
        
        _, _, H, W = prostate_mask.shape
        
        penalties = []
        for i in range(B):
            mask_i = prostate_mask[i, 0]
            coords_i = rl_coords[i]
            
            outside_count = 0
            for j in range(num_points):
                x, y = coords_i[j]
                px = int(torch.clamp(x, 0, W - 1).item())
                py = int(torch.clamp(y, 0, H - 1).item())
                
                if mask_i[py, px] <= 0.5:
                    outside_count += 1
            
            penalty = outside_count / num_points
            penalties.append(penalty)
        
        penalties = torch.tensor(penalties, device=device)
        return self.prostate_boundary_penalty_weight * penalties
    
    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor],
        num_samples_per_image: int = 1,
    ) -> torch.Tensor:
        """
        Compute rewards for a batch.
        
        Args:
            outputs: Model outputs (data dict with 'cancer_logits' added)
            data: Original batch data
            num_samples_per_image: Number of RL samples per image (for diversity reward)
            
        Returns:
            rewards: Rewards (B,)
        """
        cancer_logits = outputs.get('cancer_logits', outputs.get('mask_logits'))
        
        if cancer_logits is None:
            raise KeyError("Could not find 'cancer_logits' or 'mask_logits' in outputs")
        
        # Compute base reward based on mode
        if self.reward_mode == 'combined_v2':
            rewards = self.compute_combined_v2_reward(outputs, data)
        elif self.reward_mode == 'confidence_based':
            rewards = self.compute_confidence_based_reward(cancer_logits, data)
        elif self.reward_mode == 'classification_only':
            rewards = self.compute_classification_reward(outputs, data)
        elif self.reward_mode == 'roi_only':
            rewards = self.compute_roi_involvement_reward(cancer_logits, data)
        else:
            # Legacy modes
            if self.reward_mode == 'loss_based':
                rewards = self._compute_loss_based_reward(cancer_logits, data)
            else:
                raise ValueError(f"Unknown reward_mode: {self.reward_mode}")
        
        # Apply prostate boundary penalty if enabled
        if self.prostate_boundary_penalty_weight > 0 and 'rl_attention_coords' in outputs:
            rl_coords = outputs['rl_attention_coords']
            prostate_mask = data['prostate_mask'].to(cancer_logits.device)
            boundary_penalty = self.compute_prostate_boundary_penalty(rl_coords, prostate_mask)
            rewards = rewards - boundary_penalty
        
        # Apply diversity reward if enabled (prevents policy collapse)
        if self.diversity_reward_weight > 0 and 'rl_attention_coords' in outputs:
            rl_coords = outputs['rl_attention_coords']
            diversity_bonus = self.compute_diversity_reward(rl_coords, num_samples_per_image)
            rewards = rewards + self.diversity_reward_weight * diversity_bonus
        
        # Normalize if requested (usually disabled for GRPO)
        if self.normalize_rewards and rewards.numel() > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        return rewards
    
    def _compute_loss_based_reward(
        self,
        cancer_logits: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Legacy loss-based reward computation."""
        B = cancer_logits.shape[0]
        device = cancer_logits.device
        
        prostate_mask = data['prostate_mask'].to(device)
        needle_mask = data['needle_mask'].to(device)
        involvement = data['involvement'].to(device)
        
        rewards = []
        for i in range(B):
            mask = (prostate_mask[i] > 0.5) & (needle_mask[i] > 0.5)
            
            predictions, _ = MaskedPredictionModule()(
                cancer_logits[i:i+1], mask[None, ...]
            )
            
            if len(predictions) == 0:
                rewards.append(0.0)
                continue
            
            target = involvement[i].item()
            pred_prob = predictions.sigmoid().mean()
            
            bce = -(target * torch.log(pred_prob + 1e-8) + 
                   (1 - target) * torch.log(1 - pred_prob + 1e-8))
            reward = -bce.item() / 2.0
            rewards.append(reward)
        
        rewards = torch.tensor(rewards, device=device)
        
        if 'grade_group' in data:
            cspca_mask = data['grade_group'].to(device) > 2
            if cspca_mask.ndim > 1:
                cspca_mask = cspca_mask.squeeze(-1)
            rewards = torch.where(cspca_mask, rewards * self.cspca_bonus, rewards)
        
        return rewards


def build_rl_reward_computer(args) -> RLRewardComputer:
    """
    Build reward computer from config.
    
    Args:
        args: Config with RL settings
        
    Returns:
        reward_computer: RLRewardComputer instance
    """
    reward_mode = args.get('rl_reward_mode', 'combined_v2')
    cspca_bonus = args.get('rl_cspca_bonus', 2.0)
    
    # For GRPO, normalization happens in the algorithm
    normalize_rewards = False
    
    # Prostate boundary penalty (disabled by default now)
    prostate_boundary_penalty_weight = args.get('rl_prostate_boundary_penalty_weight', 0.0)
    prostate_boundary_penalty_scale = args.get('rl_prostate_boundary_penalty_scale', 10.0)
    
    # Reward composition weights
    heatmap_reward_weight = args.get('rl_heatmap_reward_weight', 0.5)
    classification_reward_weight = args.get('rl_classification_reward_weight', 0.5)
    
    # Diversity reward (prevents policy collapse)
    diversity_reward_weight = args.get('rl_diversity_reward_weight', 0.0)
    
    return RLRewardComputer(
        reward_mode=reward_mode,
        cspca_bonus=cspca_bonus,
        normalize_rewards=normalize_rewards,
        prostate_boundary_penalty_weight=prostate_boundary_penalty_weight,
        prostate_boundary_penalty_scale=prostate_boundary_penalty_scale,
        heatmap_reward_weight=heatmap_reward_weight,
        classification_reward_weight=classification_reward_weight,
        diversity_reward_weight=diversity_reward_weight,
    )
