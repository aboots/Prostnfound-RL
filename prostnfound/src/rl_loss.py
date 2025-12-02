"""
RL-specific loss and reward computation for ProstNFound-RL

Note on Within-Image Comparison:
When using GRPO with within-image comparison (multiple samples per image),
reward normalization should be done in GRPO's compute_advantages(), not here.
Set normalize_rewards=False when using within-image comparison.
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
    1. Encourage correct cancer detection (heatmap)
    2. Reward good classification performance (image-level)
    3. Heavily reward correct identification of clinically significant PCa (csPCa)
    4. Penalize false positives and false negatives
    5. Softly penalize attention coordinates outside the prostate segmentation
    
    CRITICAL: The RL policy affects BOTH the decoder (heatmap) AND the classifier
    through attention point embeddings. Therefore, the reward MUST include both
    heatmap and classification performance to provide proper learning signal.
    
    Args:
        reward_mode: Type of reward computation:
            - 'loss_based': Negative BCE loss (current, not recommended)
            - 'accuracy_based': Binary accuracy (+1 correct, -1 incorrect)
            - 'confidence_based': Reward high confidence on correct predictions (RECOMMENDED)
            - 'ranking_based': Reward based on ranking quality within batch
            - 'f1_based': F1-score based reward
            - 'combined': Mix of loss and accuracy (legacy)
        cspca_bonus: Bonus multiplier for csPCa cases (default: 2.0)
        normalize_rewards: Whether to normalize rewards within batch (default: False)
            NOTE: When using within-image comparison in GRPO, set this to False
            so that normalization happens within each image group in GRPO.
        prostate_boundary_penalty_weight: Weight for penalizing coordinates outside prostate (default: 0.1)
        prostate_boundary_penalty_scale: Scale factor for distance-based penalty (default: 10.0)
            Higher values make penalty increase faster with distance
        heatmap_reward_weight: Weight for heatmap performance in combined reward (default: 0.7)
        classification_reward_weight: Weight for classification performance (default: 0.3)
    """
    
    def __init__(
        self,
        reward_mode: str = 'loss_based',
        cspca_bonus: float = 2.0,
        normalize_rewards: bool = False,  # Changed default to False for within-image comparison
        prostate_boundary_penalty_weight: float = 0.1,
        prostate_boundary_penalty_scale: float = 10.0,
        heatmap_reward_weight: float = 0.7,
        classification_reward_weight: float = 0.3,
    ):
        self.reward_mode = reward_mode
        self.cspca_bonus = cspca_bonus
        self.normalize_rewards = normalize_rewards
        self.prostate_boundary_penalty_weight = prostate_boundary_penalty_weight
        self.prostate_boundary_penalty_scale = prostate_boundary_penalty_scale
        self.heatmap_reward_weight = heatmap_reward_weight
        self.classification_reward_weight = classification_reward_weight
    
    def compute_prostate_boundary_penalty(
        self,
        rl_coords: torch.Tensor,
        prostate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute soft penalty for attention coordinates outside the prostate mask.
        
        The penalty is:
        - 0 if the coordinate is inside the prostate
        - Increases smoothly with distance from the prostate boundary if outside
        
        This encourages the RL policy to focus on valid prostate regions while
        allowing some exploration near boundaries.
        
        Args:
            rl_coords: Attention coordinates (B, num_points, 2) in [x, y] format, range [0, image_size]
            prostate_mask: Prostate segmentation mask (B, 1, H, W), binary mask
            
        Returns:
            penalties: Penalty for each sample (B,), averaged over all attention points
        """
        if rl_coords is None or self.prostate_boundary_penalty_weight == 0:
            return torch.zeros(prostate_mask.shape[0], device=prostate_mask.device)
        
        B, num_points, _ = rl_coords.shape
        _, _, H_mask, W_mask = prostate_mask.shape
        device = prostate_mask.device
        
        # CRITICAL FIX: Resize mask to match coordinate space
        # RL coordinates are generated in image_size space (256x256 by default)
        # but masks might be downsampled (e.g., 64x64 as per mask_size config)
        # We MUST upsample the mask to match the coordinate resolution
        COORD_SPACE_SIZE = 256  # Standard image size used in the model
        
        if H_mask != COORD_SPACE_SIZE or W_mask != COORD_SPACE_SIZE:
            prostate_mask = F.interpolate(
                prostate_mask.float(),
                size=(COORD_SPACE_SIZE, COORD_SPACE_SIZE),
                mode='nearest'
            )
        
        _, _, H, W = prostate_mask.shape  # Now H=W=256
        
        # Normalize coordinates to [0, 1] range for grid_sample
        coords_normalized = rl_coords.clone()
        coords_normalized[:, :, 0] = coords_normalized[:, :, 0] / W  # x
        coords_normalized[:, :, 1] = coords_normalized[:, :, 1] / H  # y
        
        # Convert to grid_sample format: range [-1, 1]
        coords_grid = coords_normalized * 2.0 - 1.0  # (B, num_points, 2)
        coords_grid = coords_grid.unsqueeze(1)  # (B, 1, num_points, 2)
        
        # Sample prostate mask at attention coordinates
        # grid_sample expects (N, C, H_out, W_out) and grid of shape (N, H_out, W_out, 2)
        prostate_mask_float = prostate_mask.float()
        sampled_mask = F.grid_sample(
            prostate_mask_float,
            coords_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # (B, 1, 1, num_points)
        
        sampled_mask = sampled_mask.squeeze(1).squeeze(1)  # (B, num_points)
        
        # Compute distance-based penalties for points outside the mask
        # Create coordinate grid once for efficiency
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        batch_penalties = []
        
        for i in range(B):
            mask_i = prostate_mask[i, 0]  # (H, W)
            coords_i = rl_coords[i]  # (num_points, 2)
            
            # Find prostate pixels for this sample
            prostate_pixels = mask_i > 0.5
            
            if prostate_pixels.sum() == 0:
                # No prostate mask - apply maximum penalty to all points
                batch_penalties.append(1.0)
                continue
            
            # Get prostate pixel coordinates
            prostate_y = y_grid[prostate_pixels]
            prostate_x = x_grid[prostate_pixels]
            
            point_penalties = []
            for j in range(num_points):
                x, y = coords_i[j]
                
                # Convert to pixel coordinates
                px = int(torch.clamp(x, 0, W - 1).item())
                py = int(torch.clamp(y, 0, H - 1).item())
                
                # Check if inside prostate
                inside = mask_i[py, px] > 0.5
                
                if inside:
                    # No penalty for points inside prostate
                    point_penalties.append(0.0)
                else:
                    # Compute distance to nearest prostate pixel
                    dist_y = prostate_y - y
                    dist_x = prostate_x - x
                    distances = torch.sqrt(dist_x ** 2 + dist_y ** 2)
                    
                    # Minimum distance to prostate boundary
                    min_distance = distances.min()
                    
                    # Soft penalty: increases with distance but saturates
                    # penalty = 1 - exp(-scale * distance / image_size)
                    # Normalize by image size to make penalty scale-invariant
                    normalized_dist = min_distance / max(H, W)
                    penalty = 1.0 - torch.exp(-self.prostate_boundary_penalty_scale * normalized_dist)
                    point_penalties.append(penalty.item())
            
            # Average penalty over all points for this sample
            avg_penalty = sum(point_penalties) / len(point_penalties) if point_penalties else 0.0
            batch_penalties.append(avg_penalty)
        
        penalties = torch.tensor(batch_penalties, device=device)
        
        # Apply weight
        penalties = self.prostate_boundary_penalty_weight * penalties
        
        return penalties
    
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
    
    def compute_confidence_based_reward(
        self,
        cancer_logits: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute reward based on confidence calibration.
        
        This rewards:
        - High confidence (probability close to 1) for positive cases
        - Low confidence (probability close to 0) for negative cases
        - Penalizes overconfidence on wrong predictions
        
        This is better than loss-based because:
        1. Directly optimizes what we care about (confidence calibration)
        2. More interpretable rewards
        3. Better learning signal for ranking tasks
        
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
            
            # Get mean prediction probability
            pred_prob = predictions.sigmoid().mean()
            target = involvement[i].item()
            
            # Confidence-based reward:
            # - For positive cases: reward = pred_prob (higher is better)
            # - For negative cases: reward = 1 - pred_prob (lower pred_prob is better)
            if target > 0.5:
                # Positive case: reward high confidence
                reward = pred_prob.item()
            else:
                # Negative case: reward low confidence
                reward = (1.0 - pred_prob).item()
            
            # Scale to [-1, 1] range: 2 * reward - 1
            reward = 2.0 * reward - 1.0
            
            rewards.append(reward)
        
        rewards = torch.tensor(rewards, device=device)
        
        # Add csPCa bonus
        if 'grade_group' in data:
            cspca_mask = data['grade_group'] > 2
            rewards[cspca_mask] *= self.cspca_bonus
        
        return rewards
    
    def compute_ranking_based_reward(
        self,
        cancer_logits: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute reward based on ranking quality.
        
        This rewards samples that rank higher than other samples in the batch.
        For positive cases: higher rank = higher reward
        For negative cases: lower rank = higher reward
        
        This approximates AUC optimization and is better for ranking tasks.
        
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
        
        # Get predictions for all samples
        sample_scores = []
        labels = []
        
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
                sample_scores.append(0.0)
            else:
                pred_prob = predictions.sigmoid().mean().item()
                sample_scores.append(pred_prob)
            
            labels.append(involvement[i].item())
        
        sample_scores = torch.tensor(sample_scores, device=device)
        labels = torch.tensor(labels, device=device)
        
        # Compute ranking-based rewards
        rewards = torch.zeros(B, device=device)
        
        # For each positive sample, reward based on how many negatives it ranks above
        pos_mask = labels > 0.5
        neg_mask = labels <= 0.5
        
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_scores = sample_scores[pos_mask]
            neg_scores = sample_scores[neg_mask]
            
            # For each positive: count how many negatives it beats
            for i in range(B):
                if labels[i] > 0.5:
                    # Positive case: reward = fraction of negatives ranked below
                    beats_negatives = (neg_scores < sample_scores[i]).float().mean()
                    rewards[i] = beats_negatives * 2.0 - 1.0  # Scale to [-1, 1]
                else:
                    # Negative case: reward = fraction of positives ranked above
                    beaten_by_positives = (pos_scores > sample_scores[i]).float().mean()
                    rewards[i] = (1.0 - beaten_by_positives) * 2.0 - 1.0  # Scale to [-1, 1]
        else:
            # Fallback to confidence-based if no positive/negative mix
            rewards = self.compute_confidence_based_reward(cancer_logits, data)
        
        # Add csPCa bonus
        if 'grade_group' in data:
            cspca_mask = data['grade_group'] > 2
            rewards[cspca_mask] *= self.cspca_bonus
        
        return rewards
    
    def compute_f1_based_reward(
        self,
        cancer_logits: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute reward based on F1-score approximation.
        
        This rewards both precision and recall, which is better for imbalanced datasets.
        However, F1 requires batch-level computation, so we approximate per-sample.
        
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
        
        # Get predictions and labels
        predictions_list = []
        labels_list = []
        
        for i in range(B):
            mask = torch.ones(prostate_mask[i].shape, device=device).bool()
            mask &= prostate_mask[i] > 0.5
            mask &= needle_mask[i] > 0.5
            
            predictions, _ = MaskedPredictionModule()(
                cancer_logits[i:i+1], mask[None, ...]
            )
            
            if len(predictions) == 0:
                pred_prob = 0.5  # Neutral prediction
            else:
                pred_prob = predictions.sigmoid().mean().item()
            
            predictions_list.append(pred_prob)
            labels_list.append(involvement[i].item())
        
        predictions_tensor = torch.tensor(predictions_list, device=device)
        labels_tensor = torch.tensor(labels_list, device=device)
        
        # Compute F1 components at batch level
        pred_binary = (predictions_tensor > 0.5).float()
        
        # True positives, false positives, false negatives
        tp = ((pred_binary == 1) & (labels_tensor > 0.5)).float().sum()
        fp = ((pred_binary == 1) & (labels_tensor <= 0.5)).float().sum()
        fn = ((pred_binary == 0) & (labels_tensor > 0.5)).float().sum()
        
        # Compute precision and recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Per-sample reward: base reward + contribution to F1
        rewards = torch.zeros(B, device=device)
        
        for i in range(B):
            if labels_tensor[i] > 0.5:
                # Positive case: reward based on recall contribution
                if pred_binary[i] > 0.5:
                    # True positive: contributes to recall
                    reward = recall.item()
                else:
                    # False negative: hurts recall
                    reward = -recall.item()
            else:
                # Negative case: reward based on precision contribution
                if pred_binary[i] > 0.5:
                    # False positive: hurts precision
                    reward = -precision.item()
                else:
                    # True negative: contributes to precision
                    reward = precision.item()
            
            # Scale by F1 score (overall quality)
            rewards[i] = reward * f1.item()
        
        # Normalize to [-1, 1]
        if rewards.std() > 1e-8:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards = torch.clamp(rewards, -1.0, 1.0)
        
        # Add csPCa bonus
        if 'grade_group' in data:
            cspca_mask = data['grade_group'] > 2
            rewards[cspca_mask] *= self.cspca_bonus
        
        return rewards
    
    def compute_classification_reward(
        self,
        outputs: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute reward based on classification head performance.
        
        This is crucial because the RL policy affects BOTH the decoder (heatmap)
        and the classifier through attention points. We need to reward good
        classification performance too.
        
        Args:
            outputs: Model outputs with classification logits
            data: Batch data with labels
            
        Returns:
            rewards: Classification rewards (B,)
        """
        cls_logits = outputs['image_level_classification_outputs'][0]  # (B, num_classes)
        device = cls_logits.device
        
        # Get labels based on classification mode
        if 'grade_group' in data:
            # csPCa classification (grade_group > 2)
            labels = (data['grade_group'] > 2).long().to(device)
        else:
            # Binary cancer classification
            labels = data['label'].to(device)
        
        # Compute probabilities
        probs = F.softmax(cls_logits, dim=1)
        pred_probs = probs[torch.arange(len(labels)), labels]  # Probability of correct class
        
        # Reward: log probability of correct class (negative cross-entropy)
        # Scale to similar range as heatmap rewards
        rewards = torch.log(pred_probs + 1e-8) / 2.0  # Scale to roughly [-2, 0]
        
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
        
        # Compute base reward from heatmap (decoder output)
        if self.reward_mode == 'loss_based':
            rewards = self.compute_loss_based_reward(cancer_logits, data)
        elif self.reward_mode == 'accuracy_based':
            rewards = self.compute_accuracy_based_reward(cancer_logits, data)
        elif self.reward_mode == 'confidence_based':
            rewards = self.compute_confidence_based_reward(cancer_logits, data)
        elif self.reward_mode == 'ranking_based':
            rewards = self.compute_ranking_based_reward(cancer_logits, data)
        elif self.reward_mode == 'f1_based':
            rewards = self.compute_f1_based_reward(cancer_logits, data)
        elif self.reward_mode == 'combined':
            loss_reward = self.compute_loss_based_reward(cancer_logits, data)
            acc_reward = self.compute_accuracy_based_reward(cancer_logits, data)
            rewards = 0.5 * loss_reward + 0.5 * acc_reward
        else:
            raise ValueError(
                f"Unknown reward_mode: {self.reward_mode}. "
                f"Must be one of: 'loss_based', 'accuracy_based', 'confidence_based', "
                f"'ranking_based', 'f1_based', 'combined'"
            )
        
        # CRITICAL FIX: Add classification head reward
        # The RL policy affects BOTH decoder and classifier through attention points
        # So we need to reward improvements in BOTH outputs, not just the heatmap
        if 'image_level_classification_outputs' in outputs:
            cls_reward = self.compute_classification_reward(outputs, data)
            # Combine heatmap and classification rewards (weighted average)
            # This ensures RL policy gets signal from both heads
            rewards = self.heatmap_reward_weight * rewards + self.classification_reward_weight * cls_reward
        
        # Apply soft penalty for coordinates outside prostate mask
        if 'rl_attention_coords' in outputs and self.prostate_boundary_penalty_weight > 0:
            rl_coords = outputs['rl_attention_coords']
            prostate_mask = data['prostate_mask'].to(cancer_logits.device)
            
            boundary_penalty = self.compute_prostate_boundary_penalty(rl_coords, prostate_mask)
            
            # Subtract penalty from reward (penalty is positive, so we subtract)
            rewards = rewards - boundary_penalty
        
        return rewards


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
    cspca_bonus = args.get('rl_cspca_bonus', 2.0)
    
    # When using within-image comparison, normalization happens in GRPO
    num_samples_per_image = args.get('rl_num_samples_per_image', 4)
    if num_samples_per_image > 1:
        # Within-image comparison: GRPO handles normalization
        normalize_rewards = False
    else:
        # Fallback to batch normalization
        normalize_rewards = args.get('rl_normalize_rewards', True)
    
    # Prostate boundary penalty parameters
    prostate_boundary_penalty_weight = args.get('rl_prostate_boundary_penalty_weight', 0.1)
    prostate_boundary_penalty_scale = args.get('rl_prostate_boundary_penalty_scale', 10.0)
    
    # Reward composition weights (how much to weight heatmap vs classification)
    heatmap_reward_weight = args.get('rl_heatmap_reward_weight', 0.7)
    classification_reward_weight = args.get('rl_classification_reward_weight', 0.3)
    
    return RLRewardComputer(
        reward_mode=reward_mode,
        cspca_bonus=cspca_bonus,
        normalize_rewards=normalize_rewards,
        prostate_boundary_penalty_weight=prostate_boundary_penalty_weight,
        prostate_boundary_penalty_scale=prostate_boundary_penalty_scale,
        heatmap_reward_weight=heatmap_reward_weight,
        classification_reward_weight=classification_reward_weight,
    )

