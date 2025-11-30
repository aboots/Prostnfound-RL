"""
RL Attention Policy Network for ProstNFound-RL

This module implements a lightweight policy network that learns to identify
suspicious regions in prostate ultrasound images using reinforcement learning.

Key Features:
- Prostate mask-aware sampling: Heavily penalizes attention outside prostate region
- Clinical feature modulation: Uses clinical data to guide attention
- Soft masking: Allows some exploration outside prostate with configurable penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging


class RLAttentionPolicy(nn.Module):
    """
    Lightweight policy network that identifies k suspicious regions for attention.
    
    Args:
        feature_dim: Dimension of input features from encoder (default: 256)
        hidden_dim: Dimension of hidden layers (default: 512)
        num_attention_points: Number of suspicious regions to identify (default: 3)
        image_size: Size of input image (default: 256)
        use_clinical_features: Whether to incorporate clinical data (default: True)
        temperature: Temperature for sampling coordinates (default: 1.0)
        prostate_mask_penalty: Penalty for sampling outside prostate (default: 10.0)
            Higher values = stronger constraint to stay inside prostate
        use_soft_mask: If True, use soft masking; if False, use hard masking (default: True)
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        num_attention_points: int = 3,
        image_size: int = 256,
        use_clinical_features: bool = True,
        temperature: float = 1.0,
        prostate_mask_penalty: float = 10.0,
        use_soft_mask: bool = True,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_attention_points = num_attention_points
        self.image_size = image_size
        self.use_clinical_features = use_clinical_features
        self.temperature = temperature
        self.prostate_mask_penalty = prostate_mask_penalty
        self.use_soft_mask = use_soft_mask
        
        # Feature processing layers
        # Input features are typically B x 256 x H x W (e.g., B x 256 x 64 x 64)
        # Using GroupNorm instead of BatchNorm for stability with small batch sizes (e.g., GRPO sampling)
        self.feature_processor = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=hidden_dim),  # 32 groups works well for hidden_dim=512
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Attention map generator - predicts heatmap of suspicious regions
        self.attention_map_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=hidden_dim // 2),  # 16 groups for hidden_dim//2=256
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )
        
        # Clinical feature embedding (optional)
        if use_clinical_features:
            # Assuming up to 4 clinical features (age, psa, psad, position)
            self.clinical_embedder = nn.Sequential(
                nn.Linear(4, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, hidden_dim),
            )
            self.clinical_modulation = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
        
        # Value head for estimating state value (used in RL training)
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def _apply_prostate_mask(
        self,
        attention_logits: torch.Tensor,
        prostate_mask: Optional[torch.Tensor],
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply prostate mask to attention logits.
        
        Points outside prostate are penalized (soft mask) or zeroed (hard mask).
        
        Args:
            attention_logits: Raw attention logits (B, H*W)
            prostate_mask: Prostate segmentation mask (B, 1, H_mask, W_mask)
            H, W: Feature map dimensions
            
        Returns:
            masked_logits: Attention logits with prostate mask applied
            mask_flat: Flattened mask indicating prostate region
        """
        B = attention_logits.shape[0]
        
        if prostate_mask is None:
            # No mask - return unchanged
            return attention_logits, torch.ones(B, H * W, device=attention_logits.device)
            
        # Check for NaNs in mask
        if torch.isnan(prostate_mask).any():
            logging.warning("NaNs detected in prostate_mask! Ignoring mask.")
            return attention_logits, torch.ones(B, H * W, device=attention_logits.device)
        
        # Resize prostate mask to match feature map size
        if prostate_mask.shape[-2:] != (H, W):
            mask_resized = F.interpolate(
                prostate_mask.float(), 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
        else:
            mask_resized = prostate_mask.float()
        
        # Flatten mask: (B, 1, H, W) -> (B, H*W)
        mask_flat = mask_resized.view(B, -1)
        
        # Clamp attention_logits to prevent extreme values that could cause NaN
        attention_logits = torch.clamp(attention_logits, min=-50.0, max=50.0)
        
        if self.use_soft_mask:
            # Soft masking: subtract penalty from logits outside prostate
            # Points inside prostate (mask > 0.5) get no penalty
            # Points outside (mask < 0.5) get penalty proportional to distance from boundary
            penalty = self.prostate_mask_penalty * (1.0 - mask_flat)
            # Clamp penalty to prevent extreme values
            penalty = torch.clamp(penalty, min=0.0, max=20.0)
            masked_logits = attention_logits - penalty
            
            # Ensure we don't have all -inf values (which would cause NaN in softmax)
            # If all values are too negative, add a constant to shift them up
            max_logits = masked_logits.max(dim=1, keepdim=True)[0]  # Get max (least negative) per batch
            # If max is still very negative, shift up so max is at least -20
            shift_needed = torch.clamp(-20.0 - max_logits, min=0.0)
            masked_logits = masked_logits + shift_needed
            
            # Final clamp for soft masking
            masked_logits = torch.clamp(masked_logits, min=-50.0, max=50.0)
        else:
            # Hard masking: set logits outside prostate to -inf
            masked_logits = attention_logits.clone()
            outside_mask = mask_flat < 0.5
            masked_logits[outside_mask] = float('-inf')
            
            # CRITICAL: Ensure at least one valid (finite) point per batch
            # If all points are -inf, softmax will produce NaN
            all_masked = outside_mask.all(dim=1)
            if all_masked.any():
                center_idx = (H * W) // 2
                # Set center to a reasonable value (use max of original logits)
                for b in range(B):
                    if all_masked[b]:
                        max_val = attention_logits[b].max()
                        masked_logits[b, center_idx] = max_val if torch.isfinite(max_val) else 0.0
            
            # For hard masking, DON'T clamp -inf values, only clamp finite ones
            # This preserves the -inf for softmax to work correctly
            finite_mask = torch.isfinite(masked_logits)
            masked_logits = torch.where(
                finite_mask,
                torch.clamp(masked_logits, min=-50.0, max=50.0),
                masked_logits  # Keep -inf as is
            )
        
        return masked_logits, mask_flat
        
    def forward(
        self,
        image_features: torch.Tensor,
        clinical_features: Optional[torch.Tensor] = None,
        prostate_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        given_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate attention points.
        
        Args:
            image_features: Feature maps from encoder, shape (B, C, H, W)
            clinical_features: Optional clinical data, shape (B, num_features)
            prostate_mask: Optional prostate segmentation mask (B, 1, H_mask, W_mask)
            deterministic: If True, select top-k points; if False, sample from distribution
            given_actions: Optional indices of actions to evaluate log_probs for (B, k). 
                          If provided, skips sampling and uses these actions.
            
        Returns:
            coords: Point coordinates in image space (B, k, 2) in range [0, image_size]
            log_probs: Log probabilities of selected points (B, k)
            attention_map: Raw attention heatmap (B, 1, H, W)
            value: State value estimate (B, 1)
            outside_prostate_ratio: Fraction of attention points outside prostate (B,)
            actions: Action indices used (B, k)
        """
        B, C, H, W = image_features.shape
        
        # Process features
        if torch.isnan(image_features).any():
            logging.error("NaNs detected in image_features input to policy!")
        if torch.isinf(image_features).any():
            logging.error("Infs detected in image_features input to policy!")
        
        # Log feature stats occasionally
        if torch.rand(1) < 0.01:
            logging.info(f"Feature stats: min={image_features.min().item():.4f}, max={image_features.max().item():.4f}, mean={image_features.mean().item():.4f}")

        features = self.feature_processor(image_features)  # B x hidden_dim x H x W
        
        # Modulate with clinical features if available
        if self.use_clinical_features and clinical_features is not None:
            clinical_emb = self.clinical_embedder(clinical_features)  # B x hidden_dim
            modulation = self.clinical_modulation(clinical_emb)  # B x hidden_dim
            modulation = modulation[:, :, None, None]  # B x hidden_dim x 1 x 1
            features = features * modulation
        
        # Generate attention map (B x 1 x H x W)
        attention_map = self.attention_map_head(features)
        
        # Flatten attention map for sampling
        attention_logits = attention_map.view(B, -1)  # B x (H*W)
        
        # Apply temperature
        attention_logits = attention_logits / self.temperature
        
        # Apply prostate mask penalty
        masked_logits, mask_flat = self._apply_prostate_mask(
            attention_logits, prostate_mask, H, W
        )
        
        # Sample or select top-k points
        if given_actions is not None:
            # Use provided actions (indices) to compute log probs
            coords_flat = given_actions
            
            # Compute log probs for these points
            log_probs = F.log_softmax(masked_logits, dim=1)
            selected_log_probs = torch.gather(log_probs, 1, coords_flat)  # B x k
            
        elif deterministic:
            # Select top-k points from masked logits
            # Check for NaN/inf before topk
            if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                # Fallback: use original logits if masked version has issues
                masked_logits = attention_logits
                logging.warning("NaN/Inf detected in masked_logits, using original logits")
            
            top_k_indices = torch.topk(masked_logits, k=self.num_attention_points, dim=1).indices
            coords_flat = top_k_indices  # B x k
            
            # Compute log probs for selected points (from masked distribution)
            log_probs = F.log_softmax(masked_logits, dim=1)
            selected_log_probs = torch.gather(log_probs, 1, coords_flat)  # B x k
        else:
            # Sample from categorical distribution with masked logits
            
            # CRITICAL FIX: Handle NaNs/Infs in masked_logits (same as deterministic path)
            if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                logging.warning("NaN/Inf detected in masked_logits during training, falling back to original logits")
                masked_logits = attention_logits
                
                # If original logits also have NaNs (model instability), fallback to uniform
                if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                    logging.warning("NaN/Inf detected in original logits! Using uniform distribution.")
                    masked_logits = torch.zeros_like(masked_logits)

            # Use log_softmax for better numerical stability, then exp
            # log_softmax handles -inf correctly (gives -inf log prob, which exp gives 0 prob)
            log_probs = F.log_softmax(masked_logits, dim=1)
            attention_probs = torch.exp(log_probs)
            
            # Normalize to ensure sum is exactly 1 (handles any numerical errors)
            prob_sums = attention_probs.sum(dim=1, keepdim=True)
            attention_probs = attention_probs / (prob_sums + 1e-10)  # Small epsilon to avoid div by zero
            
            # Final safety check for probs
            if torch.isnan(attention_probs).any():
                logging.warning("NaNs in attention_probs! Resetting to uniform.")
                attention_probs = torch.ones_like(attention_probs) / attention_probs.shape[1]
            
            # Sample k points without replacement
            sampled_indices_list = []
            log_probs_list = []
            
            for i in range(self.num_attention_points):
                # Check if we have valid probabilities (sum > threshold)
                prob_sums = attention_probs.sum(dim=1)
                invalid_batch = prob_sums < 1e-6
                
                if invalid_batch.any():
                    # If all probabilities are near zero (shouldn't happen, but handle it)
                    # Use uniform distribution over remaining valid points
                    uniform_probs = torch.ones_like(attention_probs) / attention_probs.shape[1]
                    attention_probs = torch.where(
                        invalid_batch.unsqueeze(1),
                        uniform_probs,
                        attention_probs
                    )
                    # Re-normalize
                    prob_sums = attention_probs.sum(dim=1, keepdim=True)
                    attention_probs = attention_probs / (prob_sums.detach() + 1e-10)
                
                # Sample one point
                # Ensure probs are valid for Categorical (no NaNs, no negatives, sum=1)
                # Clamp to be safe
                attention_probs = torch.clamp(attention_probs, min=0.0, max=1.0)
                # Re-normalize one last time
                attention_probs = attention_probs / (attention_probs.sum(dim=1, keepdim=True).detach() + 1e-10)
                
                dist = torch.distributions.Categorical(probs=attention_probs)
                sampled_idx = dist.sample()  # B
                sampled_log_prob = dist.log_prob(sampled_idx)  # B
                
                sampled_indices_list.append(sampled_idx)
                log_probs_list.append(sampled_log_prob)
                
                # Zero out the probability of sampled points to avoid re-sampling
                # (simple approximation of sampling without replacement)
                attention_probs = attention_probs.scatter(1, sampled_idx.unsqueeze(1), 0.0)
                
                # Re-normalize after zeroing out
                prob_sums = attention_probs.sum(dim=1, keepdim=True)
                # CRITICAL: If sum is zero (all points zeroed), we've sampled all points
                # In this case, remaining probabilities should stay zero (we won't sample more)
                # But we need to avoid division by zero
                attention_probs = torch.where(
                    prob_sums > 1e-10,
                    attention_probs / (prob_sums.detach() + 1e-10),
                    attention_probs  # Keep zeros if sum is zero
                )
            
            coords_flat = torch.stack(sampled_indices_list, dim=1)  # B x k
            selected_log_probs = torch.stack(log_probs_list, dim=1)  # B x k
        
        # Compute outside prostate ratio for penalty in reward
        if prostate_mask is not None:
            # Check which selected points are outside prostate
            selected_mask_values = torch.gather(mask_flat, 1, coords_flat)  # B x k
            outside_prostate_ratio = (selected_mask_values < 0.5).float().mean(dim=1)  # B
        else:
            outside_prostate_ratio = torch.zeros(B, device=image_features.device)
        
        # Convert flat indices to (y, x) coordinates
        coords_y = coords_flat // W  # B x k
        coords_x = coords_flat % W   # B x k
        
        # Scale coordinates to image space [0, image_size]
        scale_h = self.image_size / H
        scale_w = self.image_size / W
        
        coords_y = coords_y.float() * scale_h
        coords_x = coords_x.float() * scale_w
        
        # Stack to (B, k, 2) format [x, y] as SAM expects
        coords = torch.stack([coords_x, coords_y], dim=2)  # B x k x 2
        
        # Compute value estimate
        value = self.value_head(features)  # B x 1
        
        return coords, selected_log_probs, attention_map, value, outside_prostate_ratio, coords_flat
    
    def get_attention_map(
        self, 
        image_features: torch.Tensor,
        prostate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the raw attention map without sampling points.
        Useful for visualization.
        
        Args:
            image_features: Feature maps from encoder, shape (B, C, H, W)
            prostate_mask: Optional prostate mask for visualization
            
        Returns:
            attention_map: Attention heatmap (B, 1, H, W)
        """
        features = self.feature_processor(image_features)
        attention_map = self.attention_map_head(features)
        
        # Optionally mask the attention map for visualization
        if prostate_mask is not None:
            B, C, H, W = attention_map.shape
            if prostate_mask.shape[-2:] != (H, W):
                mask_resized = F.interpolate(
                    prostate_mask.float(), 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                mask_resized = prostate_mask.float()
            attention_map = attention_map * mask_resized
        
        return attention_map


class RLAttentionPolicyGaussian(nn.Module):
    """
    Alternative policy that directly predicts coordinates using Gaussian distributions.
    This can be more stable for continuous coordinate prediction.
    
    Args:
        feature_dim: Dimension of input features from encoder (default: 256)
        hidden_dim: Dimension of hidden layers (default: 512)
        num_attention_points: Number of suspicious regions to identify (default: 3)
        image_size: Size of input image (default: 256)
        use_clinical_features: Whether to incorporate clinical data (default: True)
        prostate_mask_penalty: Penalty for sampling outside prostate (default: 0.5)
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        num_attention_points: int = 3,
        image_size: int = 256,
        use_clinical_features: bool = True,
        prostate_mask_penalty: float = 0.5,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_attention_points = num_attention_points
        self.image_size = image_size
        self.use_clinical_features = use_clinical_features
        self.prostate_mask_penalty = prostate_mask_penalty
        
        # Feature aggregation
        # Using GroupNorm instead of BatchNorm for stability with small batch sizes (e.g., GRPO sampling)
        self.feature_aggregator = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        
        flatten_dim = hidden_dim * 4 * 4
        
        # Clinical feature embedding (optional)
        if use_clinical_features:
            self.clinical_embedder = nn.Linear(4, 128)
            flatten_dim += 128
        
        # Coordinate prediction heads (mean and log_std for each point)
        self.coord_mean_head = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_attention_points * 2),  # k points x 2 coords
            nn.Sigmoid(),  # Output in [0, 1], will scale to image_size
        )
        
        self.coord_logstd_head = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_attention_points * 2),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        
    def forward(
        self,
        image_features: torch.Tensor,
        clinical_features: Optional[torch.Tensor] = None,
        prostate_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        given_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate attention points using Gaussian policy.
        
        Args:
            image_features: Feature maps from encoder, shape (B, C, H, W)
            clinical_features: Optional clinical data, shape (B, num_features)
            prostate_mask: Optional prostate mask (B, 1, H, W)
            deterministic: If True, use mean; if False, sample from Gaussian
            given_actions: Optional coordinates (B, k, 2) to evaluate log_probs for.
                          If provided, skips sampling and uses these coordinates.
            
        Returns:
            coords: Point coordinates in image space (B, k, 2)
            log_probs: Log probabilities of selected points (B, k)
            value: State value estimate (B, 1)
            outside_prostate_ratio: Fraction of points outside prostate (B,)
            actions: Point coordinates (same as coords) (B, k, 2)
        """
        B = image_features.shape[0]
        
        # Aggregate features
        features = self.feature_aggregator(image_features)
        
        # Add clinical features if available
        if self.use_clinical_features and clinical_features is not None:
            clinical_emb = self.clinical_embedder(clinical_features)
            features = torch.cat([features, clinical_emb], dim=1)
        
        # Predict mean and std for coordinates
        coord_mean = self.coord_mean_head(features)  # B x (k*2)
        coord_mean = coord_mean.view(B, self.num_attention_points, 2)  # B x k x 2
        
        coord_logstd = self.coord_logstd_head(features)  # B x (k*2)
        coord_logstd = coord_logstd.view(B, self.num_attention_points, 2)  # B x k x 2
        coord_logstd = torch.clamp(coord_logstd, min=-5, max=2)  # Clip for stability
        coord_std = torch.exp(coord_logstd)
        
        if given_actions is not None:
            # Use provided coordinates
            # given_actions are scaled coordinates [0, image_size]
            # We need to normalize back to [0, 1] for log prob computation
            coords_scaled = given_actions
            coords = coords_scaled / self.image_size
            # Clamp to [0, 1] just in case
            coords = torch.clamp(coords, min=0.0, max=1.0)
            
            # Compute log probability
            log_probs = -0.5 * (
                torch.sum(((coords - coord_mean) / coord_std) ** 2, dim=2) +
                torch.sum(2 * coord_logstd, dim=2) +
                2 * torch.log(torch.tensor(2 * 3.14159, device=features.device))
            )
            
        elif deterministic:
            coords = coord_mean
            # Compute log prob at mean
            log_probs = -0.5 * torch.sum(coord_logstd + 0.5 * torch.log(torch.tensor(2 * 3.14159, device=features.device)), dim=2)
        else:
            # Sample from Gaussian
            noise = torch.randn_like(coord_mean)
            coords = coord_mean + noise * coord_std
            
            # Compute log probability
            log_probs = -0.5 * (
                torch.sum(((coords - coord_mean) / coord_std) ** 2, dim=2) +
                torch.sum(2 * coord_logstd, dim=2) +
                2 * torch.log(torch.tensor(2 * 3.14159, device=features.device))
            )
        
        # Clip coordinates to valid range [0, 1] and scale to image size
        if given_actions is None:
            coords = torch.clamp(coords, min=0.0, max=1.0)
            coords_scaled = coords * self.image_size
        
        # Compute outside prostate ratio
        if prostate_mask is not None:
            # Sample prostate mask at coordinate locations
            # coords are in [0, 1] range before scaling
            # Use grid_sample to check if points are inside prostate
            grid = coords * 2 - 1  # Convert to [-1, 1] for grid_sample
            grid = grid.unsqueeze(1)  # B x 1 x k x 2
            
            mask_values = F.grid_sample(
                prostate_mask.float(), 
                grid, 
                mode='bilinear', 
                align_corners=True,
                padding_mode='zeros'
            )  # B x 1 x 1 x k
            mask_values = mask_values.squeeze(1).squeeze(1)  # B x k
            outside_prostate_ratio = (mask_values < 0.5).float().mean(dim=1)  # B
        else:
            outside_prostate_ratio = torch.zeros(B, device=features.device)
        
        # Compute value estimate
        value = self.value_head(features)  # B x 1
        
        return coords_scaled, log_probs, value, outside_prostate_ratio, coords_scaled

