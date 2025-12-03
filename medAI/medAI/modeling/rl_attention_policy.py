"""
RL Attention Policy Network for ProstNFound-RL

This module implements policy networks that learn to identify
suspicious regions in prostate ultrasound images using reinforcement learning.

Key Features:
- Categorical Policy: Samples from attention heatmap (original)
- Patch-Based Policy: Outputs K patch regions instead of K points (NEW)
- Prostate masking can be toggled on/off
- Architecture versions: v1 (legacy) and v2 (deeper, current)
- Optional value head for PPO training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RLAttentionPolicy(nn.Module):
    """
    Lightweight policy network that identifies k suspicious regions for attention.
    
    Now supports:
    - Toggle for prostate mask constraint
    - Improved multi-scale feature processing
    - Architecture version selection (v1=legacy, v2=deeper)
    - Optional value function for PPO training
    
    Args:
        feature_dim: Dimension of input features from encoder (default: 256)
        hidden_dim: Dimension of hidden layers (default: 512)
        num_attention_points: Number of suspicious regions to identify (default: 3)
        image_size: Size of input image (default: 256)
        use_clinical_features: Whether to incorporate clinical data (default: True)
        temperature: Temperature for sampling coordinates (default: 1.0)
        use_prostate_mask_constraint: Whether to constrain sampling to prostate (default: True)
        arch_version: Architecture version ("v1" for legacy, "v2" for deeper) (default: "v2")
        use_value_function: Whether to include value head for PPO training (default: False)
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        num_attention_points: int = 3,
        image_size: int = 256,
        use_clinical_features: bool = True,
        temperature: float = 1.0,
        use_prostate_mask_constraint: bool = True,
        arch_version: str = "v2",
        use_value_function: bool = False,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_attention_points = num_attention_points
        self.image_size = image_size
        self.use_clinical_features = use_clinical_features
        self.temperature = temperature
        self.use_prostate_mask_constraint = use_prostate_mask_constraint
        self.arch_version = arch_version
        self.use_value_function = use_value_function
        
        # Multi-scale feature processing for better spatial understanding
        self.feature_processor = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Attention map generator - architecture depends on version
        if arch_version == "v1":
            # Legacy architecture (simpler, for old checkpoints)
            self.attention_map_head = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),  # Direct to 1 channel
            )
        else:  # v2 (default, deeper)
            # Deeper architecture with more layers
            self.attention_map_head = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 4, 1, kernel_size=1),
            )
        
        # Clinical feature embedding (optional)
        if use_clinical_features:
            self.clinical_embedder = nn.Sequential(
                nn.Linear(4, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, hidden_dim),
            )
            self.clinical_modulation = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
        
        # Value function head for PPO (optional)
        if use_value_function:
            self.value_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(hidden_dim * 16, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
        
    def forward(
        self,
        image_features: torch.Tensor,
        clinical_features: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        prostate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass to generate attention points.
        
        Args:
            image_features: Feature maps from encoder, shape (B, C, H, W)
            clinical_features: Optional clinical data, shape (B, num_features)
            deterministic: If True, select top-k points; if False, sample from distribution
            prostate_mask: Optional prostate mask (B, 1, H_mask, W_mask)
                          Used only if use_prostate_mask_constraint=True
            
        Returns:
            coords: Point coordinates in image space (B, k, 2) in range [0, image_size]
            log_probs: Log probabilities of selected points (B, k)
            attention_map: Raw attention heatmap (B, 1, H, W)
            value: Value estimate if use_value_function=True, else None
        """
        B, C, H, W = image_features.shape
        
        # Process features
        features = self.feature_processor(image_features)  # B x hidden_dim x H x W
        
        # Modulate with clinical features if available
        if self.use_clinical_features and clinical_features is not None:
            clinical_emb = self.clinical_embedder(clinical_features)  # B x hidden_dim
            modulation = self.clinical_modulation(clinical_emb)  # B x hidden_dim
            modulation = modulation[:, :, None, None]  # B x hidden_dim x 1 x 1
            features = features * modulation
        
        # Compute value estimate if using value function (for PPO)
        value = None
        if self.use_value_function:
            value = self.value_head(features).squeeze(-1)  # B
        
        # Generate attention map (B x 1 x H x W)
        attention_map = self.attention_map_head(features)
        
        # Flatten attention map for sampling
        attention_logits = attention_map.view(B, -1)  # B x (H*W)
        
        # Apply temperature
        attention_logits = attention_logits / self.temperature
        
        # CRITICAL: Clamp logits to prevent numerical instability in softmax
        # As training progresses, logits can grow very large causing NaN/underflow
        # This is especially important with AMP (float16)
        attention_logits = torch.clamp(attention_logits, min=-50.0, max=50.0)
        
        # Apply prostate mask constraint if enabled
        if self.use_prostate_mask_constraint and prostate_mask is not None:
            # Resize prostate mask to match attention map spatial dimensions
            prostate_mask_resized = F.interpolate(
                prostate_mask.float(),
                size=(H, W),
                mode='nearest'
            )  # B x 1 x H x W
            
            # Flatten mask
            mask_flat = prostate_mask_resized.view(B, -1)  # B x (H*W)
            
            # Check for empty masks (would cause NaN in softmax)
            valid_mask_per_sample = (mask_flat > 0.5).any(dim=1, keepdim=True)  # B x 1
            
            # Only mask samples that have valid prostate regions
            mask_to_apply = (mask_flat < 0.5) & valid_mask_per_sample  # B x (H*W)
            attention_logits = attention_logits.masked_fill(mask_to_apply, float('-inf'))
        
        # Sample or select top-k points
        if deterministic:
            # Handle all-masked case: if all logits are -inf, use uniform fallback
            all_inf_mask = (attention_logits == float('-inf')).all(dim=1)
            if all_inf_mask.any():
                # Replace with uniform for samples that are completely masked
                attention_logits = attention_logits.clone()
                uniform_logit = 0.0  # All equal logits → uniform distribution
                attention_logits[all_inf_mask] = uniform_logit
            
            # Select top-k points
            top_k_indices = torch.topk(attention_logits, k=self.num_attention_points, dim=1).indices
            coords_flat = top_k_indices  # B x k
            
            # Compute log probs for selected points (handle -inf gracefully)
            # Replace -inf with a very small finite value for log_softmax stability
            safe_logits = attention_logits.clone()
            safe_logits = torch.where(
                safe_logits == float('-inf'),
                torch.full_like(safe_logits, -100.0),
                safe_logits
            )
            log_probs = F.log_softmax(safe_logits, dim=1)
            selected_log_probs = torch.gather(log_probs, 1, coords_flat)  # B x k
        else:
            # Sample from categorical distribution
            # Use log-softmax for numerical stability, then exponentiate
            log_probs_all = F.log_softmax(attention_logits, dim=1)
            attention_probs = log_probs_all.exp().clone()  # Clone to avoid in-place modification issues
            
            # Handle any remaining NaN/invalid rows (should be rare after logit clamping)
            invalid_rows = torch.isnan(attention_probs).any(dim=1) | (attention_probs.sum(dim=1) < 1e-6)
            if invalid_rows.any():
                uniform_probs = torch.ones_like(attention_probs[0]) / attention_probs.shape[1]
                attention_probs[invalid_rows] = uniform_probs
            
            # Sample k points without replacement
            sampled_indices_list = []
            log_probs_list = []
            
            for i in range(self.num_attention_points):
                # Ensure valid probability distribution
                attention_probs = torch.clamp(attention_probs, min=1e-8)
                attention_probs = attention_probs / attention_probs.sum(dim=1, keepdim=True)
                
                dist = torch.distributions.Categorical(probs=attention_probs)
                sampled_idx = dist.sample()  # B
                sampled_log_prob = dist.log_prob(sampled_idx)  # B
                
                sampled_indices_list.append(sampled_idx)
                log_probs_list.append(sampled_log_prob)
                
                # Zero out sampled points (scatter creates new tensor, no in-place issues)
                attention_probs = attention_probs.scatter(1, sampled_idx.unsqueeze(1), 1e-8)
                attention_probs = attention_probs / attention_probs.sum(dim=1, keepdim=True)
            
            coords_flat = torch.stack(sampled_indices_list, dim=1)  # B x k
            selected_log_probs = torch.stack(log_probs_list, dim=1)  # B x k
        
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
        
        return coords, selected_log_probs, attention_map, value
    
    def get_attention_map(self, image_features: torch.Tensor) -> torch.Tensor:
        """Get the raw attention map without sampling points."""
        features = self.feature_processor(image_features)
        attention_map = self.attention_map_head(features)
        return attention_map


class RLPatchPolicy(nn.Module):
    """
    Policy network that outputs K patch regions instead of K individual points.
    
    This is more sensible than individual points because:
    1. Larger output space (patch centers + sizes)
    2. More robust to small coordinate errors
    3. Better matches how radiologists look at regions
    
    Each patch is defined by (center_x, center_y, width, height).
    The decoder then receives multiple points sampled from each patch.
    
    Args:
        feature_dim: Dimension of input features (default: 256)
        hidden_dim: Dimension of hidden layers (default: 512)
        num_patches: Number of patch regions to output (default: 3)
        points_per_patch: Number of points to sample from each patch for SAM (default: 5)
        image_size: Size of input image (default: 256)
        min_patch_size: Minimum patch size as fraction of image (default: 0.1)
        max_patch_size: Maximum patch size as fraction of image (default: 0.3)
        use_clinical_features: Whether to use clinical data (default: True)
        use_prostate_mask_constraint: Whether to constrain patches to prostate (default: True)
        use_value_function: Whether to include value head for PPO training (default: False)
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        num_patches: int = 3,
        points_per_patch: int = 5,
        image_size: int = 256,
        min_patch_size: float = 0.1,
        max_patch_size: float = 0.3,
        use_clinical_features: bool = True,
        use_prostate_mask_constraint: bool = True,
        use_value_function: bool = False,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.points_per_patch = points_per_patch
        self.image_size = image_size
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.use_clinical_features = use_clinical_features
        self.use_prostate_mask_constraint = use_prostate_mask_constraint
        self.use_value_function = use_value_function
        
        # Feature aggregation with attention pooling
        self.feature_processor = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Attention-weighted pooling
        self.attention_weights = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Softmax(dim=-1),  # Softmax over spatial dims
        )
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        flatten_dim = hidden_dim * 4 * 4
        
        # Clinical feature embedding
        if use_clinical_features:
            self.clinical_embedder = nn.Linear(4, 128)
            flatten_dim += 128
        
        # Patch prediction heads
        # Output: num_patches x 4 (center_x, center_y, width, height)
        self.patch_head = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_patches * 4),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        # Log std head for exploration
        self.log_std_head = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_patches * 4),
        )
        
        # Value function head for PPO (optional)
        if use_value_function:
            self.value_head = nn.Sequential(
                nn.Linear(flatten_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
        
    def forward(
        self,
        image_features: torch.Tensor,
        clinical_features: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        prostate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass to generate patch regions and sample points from them.
        
        Args:
            image_features: Feature maps (B, C, H, W)
            clinical_features: Optional clinical data (B, num_features)
            deterministic: If True, use mean; if False, sample
            prostate_mask: Optional prostate mask for constraining patches
            
        Returns:
            coords: Point coordinates sampled from patches (B, k*points_per_patch, 2)
            log_probs: Log probabilities of patch predictions (B, k)
            patches: Patch parameters (B, k, 4) as (cx, cy, w, h) normalized [0, 1]
            value: Value estimate if use_value_function=True, else None
        """
        B, C, H, W = image_features.shape
        
        # Process features
        features = self.feature_processor(image_features)  # B x hidden_dim x H x W
        
        # Global pooling
        pooled = self.global_pool(features)  # B x hidden_dim x 4 x 4
        pooled = pooled.view(B, -1)  # B x (hidden_dim * 16)
        
        # Add clinical features
        if self.use_clinical_features and clinical_features is not None:
            clinical_emb = self.clinical_embedder(clinical_features)  # B x 128
            pooled = torch.cat([pooled, clinical_emb], dim=1)
        
        # Compute value estimate if using value function (for PPO)
        value = None
        if self.use_value_function:
            value = self.value_head(pooled).squeeze(-1)  # B
        
        # Predict patch parameters (mean)
        patch_mean = self.patch_head(pooled)  # B x (num_patches * 4)
        patch_mean = patch_mean.view(B, self.num_patches, 4)  # B x k x 4
        
        # Predict log std for exploration
        log_std = self.log_std_head(pooled)  # B x (num_patches * 4)
        log_std = log_std.view(B, self.num_patches, 4)
        log_std = torch.clamp(log_std, min=-5, max=0)  # Limit exploration
        std = torch.exp(log_std)
        
        if deterministic:
            patches = patch_mean
            # Log prob at mean (Gaussian)
            log_probs = -0.5 * log_std.sum(dim=2)  # B x k
        else:
            # Sample from Gaussian
            noise = torch.randn_like(patch_mean)
            patches = patch_mean + noise * std
            
            # Compute log probability
            log_probs = -0.5 * (
                ((patches - patch_mean) / std) ** 2 +
                2 * log_std
            ).sum(dim=2)  # B x k
        
        # Clamp patches to valid range
        patches = torch.clamp(patches, min=0.0, max=1.0)
        
        # Apply size constraints
        # patches[:, :, 2:4] are width and height
        size_range = self.max_patch_size - self.min_patch_size
        patches[:, :, 2:4] = patches[:, :, 2:4] * size_range + self.min_patch_size
        
        # Sample points from each patch
        coords = self._sample_points_from_patches(patches, prostate_mask)
        
        return coords, log_probs, patches, value
    
    def _sample_points_from_patches(
        self,
        patches: torch.Tensor,
        prostate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample points from patch regions.
        
        Args:
            patches: Patch parameters (B, k, 4) as (cx, cy, w, h) normalized
            prostate_mask: Optional mask for constraining points
            
        Returns:
            coords: Point coordinates (B, k * points_per_patch, 2) in image space
        """
        B, K, _ = patches.shape
        device = patches.device
        
        # Prepare prostate mask if constraint is enabled
        mask_tensor = None
        if self.use_prostate_mask_constraint and prostate_mask is not None:
            # Resize mask to image_size if needed
            B_mask, C_mask, H_mask, W_mask = prostate_mask.shape
            if H_mask != self.image_size or W_mask != self.image_size:
                mask_tensor = F.interpolate(
                    prostate_mask.float(),
                    size=(self.image_size, self.image_size),
                    mode='nearest'
                )  # B x 1 x image_size x image_size
            else:
                mask_tensor = prostate_mask.float()
            # Remove channel dimension: B x image_size x image_size
            mask_tensor = mask_tensor.squeeze(1)
        
        all_points = []
        
        for b in range(B):
            batch_points = []
            for k in range(K):
                cx, cy, w, h = patches[b, k]
                
                # Sample points uniformly within the patch
                # Use rejection sampling if prostate mask constraint is enabled
                points_sampled = 0
                max_attempts = self.points_per_patch * 20  # Safety limit
                attempts = 0
                
                while points_sampled < self.points_per_patch and attempts < max_attempts:
                    # Sample relative positions within patch [-0.5, 0.5]
                    dx = (torch.rand(1, device=device) - 0.5) * w
                    dy = (torch.rand(1, device=device) - 0.5) * h
                    
                    px = (cx + dx) * self.image_size
                    py = (cy + dy) * self.image_size
                    
                    # Clamp to valid range
                    px = torch.clamp(px, 0, self.image_size - 1).squeeze()
                    py = torch.clamp(py, 0, self.image_size - 1).squeeze()
                    
                    # Check if point is within prostate mask (if constraint enabled)
                    if mask_tensor is not None:
                        ix = int(px.item())
                        iy = int(py.item())
                        # Ensure indices are within bounds
                        ix = min(max(0, ix), self.image_size - 1)
                        iy = min(max(0, iy), self.image_size - 1)
                        
                        # Check if this pixel is inside prostate
                        if mask_tensor[b, iy, ix] < 0.5:
                            attempts += 1
                            continue  # Reject this point, try again
                    
                    # Accept this point
                    batch_points.append(torch.stack([px, py]))
                    points_sampled += 1
                    attempts += 1
                
                # If we couldn't sample enough points (e.g., patch is outside prostate),
                # fall back to patch center
                while points_sampled < self.points_per_patch:
                    px = cx * self.image_size
                    py = cy * self.image_size
                    px = torch.clamp(px, 0, self.image_size - 1)
                    py = torch.clamp(py, 0, self.image_size - 1)
                    batch_points.append(torch.stack([px, py]))
                    points_sampled += 1
            
            batch_points = torch.stack(batch_points)  # (k * points_per_patch, 2)
            all_points.append(batch_points)
        
        coords = torch.stack(all_points)  # (B, k * points_per_patch, 2)
        
        return coords


class RLAttentionPolicyGaussian(nn.Module):
    """
    Alternative policy that directly predicts coordinates using Gaussian distributions.
    (Kept for backward compatibility)
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        num_attention_points: int = 3,
        image_size: int = 256,
        use_clinical_features: bool = True,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_attention_points = num_attention_points
        self.image_size = image_size
        self.use_clinical_features = use_clinical_features
        
        # Feature aggregation
        self.feature_aggregator = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        
        flatten_dim = hidden_dim * 4 * 4
        
        if use_clinical_features:
            self.clinical_embedder = nn.Linear(4, 128)
            flatten_dim += 128
        
        self.coord_mean_head = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_attention_points * 2),
            nn.Sigmoid(),
        )
        
        self.coord_logstd_head = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_attention_points * 2),
        )
        
    def forward(
        self,
        image_features: torch.Tensor,
        clinical_features: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        prostate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for Gaussian policy."""
        B = image_features.shape[0]
        
        features = self.feature_aggregator(image_features)
        
        if self.use_clinical_features and clinical_features is not None:
            clinical_emb = self.clinical_embedder(clinical_features)
            features = torch.cat([features, clinical_emb], dim=1)
        
        coord_mean = self.coord_mean_head(features)
        coord_mean = coord_mean.view(B, self.num_attention_points, 2)
        
        coord_logstd = self.coord_logstd_head(features)
        coord_logstd = coord_logstd.view(B, self.num_attention_points, 2)
        coord_logstd = torch.clamp(coord_logstd, min=-5, max=2)
        coord_std = torch.exp(coord_logstd)
        
        if deterministic:
            coords = coord_mean
            log_probs = -0.5 * torch.sum(coord_logstd + 0.5 * torch.log(torch.tensor(2 * 3.14159)), dim=2)
        else:
            noise = torch.randn_like(coord_mean)
            coords = coord_mean + noise * coord_std
            log_probs = -0.5 * (
                torch.sum(((coords - coord_mean) / coord_std) ** 2, dim=2) +
                torch.sum(2 * coord_logstd, dim=2) +
                2 * torch.log(torch.tensor(2 * 3.14159))
            )
        
        coords = torch.clamp(coords, min=0.0, max=1.0)
        coords = coords * self.image_size
        
        return coords, log_probs, None
