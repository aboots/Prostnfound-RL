"""
RL Attention Policy Network for ProstNFound-RL

This module implements a lightweight policy network that learns to identify
suspicious regions in prostate ultrasound images using reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


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
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        num_attention_points: int = 3,
        image_size: int = 256,
        use_clinical_features: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_attention_points = num_attention_points
        self.image_size = image_size
        self.use_clinical_features = use_clinical_features
        self.temperature = temperature
        
        # Feature processing layers
        # Input features are typically B x 256 x H x W (e.g., B x 256 x 64 x 64)
        self.feature_processor = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Attention map generator - predicts heatmap of suspicious regions
        self.attention_map_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
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
        
    def forward(
        self,
        image_features: torch.Tensor,
        clinical_features: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        prostate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate attention points.
        
        Args:
            image_features: Feature maps from encoder, shape (B, C, H, W)
            clinical_features: Optional clinical data, shape (B, num_features)
            deterministic: If True, select top-k points; if False, sample from distribution
            prostate_mask: Optional prostate mask (B, 1, H_mask, W_mask) to constrain sampling
                          If provided, attention is masked to only allow sampling inside prostate
            
        Returns:
            coords: Point coordinates in image space (B, k, 2) in range [0, image_size]
            log_probs: Log probabilities of selected points (B, k)
            attention_map: Raw attention heatmap (B, 1, H, W)
            value: State value estimate (B, 1)
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
        
        # Generate attention map (B x 1 x H x W)
        attention_map = self.attention_map_head(features)
        
        # Flatten attention map for sampling
        attention_logits = attention_map.view(B, -1)  # B x (H*W)
        
        # Apply temperature
        attention_logits = attention_logits / self.temperature
        
        # CRITICAL: Apply prostate mask to constrain sampling INSIDE prostate only
        # This is much more effective than soft penalties because it prevents
        # the model from ever sampling outside the prostate
        if prostate_mask is not None:
            # Resize prostate mask to match attention map spatial dimensions
            prostate_mask_resized = F.interpolate(
                prostate_mask.float(),
                size=(H, W),
                mode='nearest'
            )  # B x 1 x H x W
            
            # Flatten mask
            mask_flat = prostate_mask_resized.view(B, -1)  # B x (H*W)
            
            # Check for empty masks (would cause NaN in softmax)
            # For samples with empty/invalid masks, don't apply masking
            valid_mask_per_sample = (mask_flat > 0.5).any(dim=1, keepdim=True)  # B x 1
            
            # Only mask samples that have valid prostate regions
            # For invalid samples, keep original logits (allow sampling anywhere)
            mask_to_apply = (mask_flat < 0.5) & valid_mask_per_sample  # B x (H*W)
            attention_logits = attention_logits.masked_fill(mask_to_apply, float('-inf'))
        
        # Sample or select top-k points
        if deterministic:
            # Select top-k points (from valid locations only due to masking)
            top_k_indices = torch.topk(attention_logits, k=self.num_attention_points, dim=1).indices
            coords_flat = top_k_indices  # B x k
            
            # Compute log probs for selected points
            log_probs = F.log_softmax(attention_logits, dim=1)
            selected_log_probs = torch.gather(log_probs, 1, coords_flat)  # B x k
        else:
            # Sample from categorical distribution (masked to prostate region)
            attention_probs = F.softmax(attention_logits, dim=1)
            
            # CRITICAL: Handle NaN and invalid probability rows
            # This happens when all logits are -inf (empty prostate mask)
            for batch_idx in range(B):
                row = attention_probs[batch_idx]
                if torch.isnan(row).any() or row.sum() < 1e-6:
                    # Replace with uniform distribution
                    attention_probs[batch_idx] = torch.ones_like(row) / row.size(0)
            
            # Sample k points without replacement
            sampled_indices_list = []
            log_probs_list = []
            
            for i in range(self.num_attention_points):
                # Ensure probs are valid before sampling
                # Clamp to avoid numerical issues
                attention_probs = torch.clamp(attention_probs, min=1e-8)
                attention_probs = attention_probs / attention_probs.sum(dim=1, keepdim=True)
                
                # Sample one point
                dist = torch.distributions.Categorical(probs=attention_probs)
                sampled_idx = dist.sample()  # B
                sampled_log_prob = dist.log_prob(sampled_idx)  # B
                
                sampled_indices_list.append(sampled_idx)
                log_probs_list.append(sampled_log_prob)
                
                # Zero out the probability of sampled points to avoid re-sampling
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
        
        # Compute value estimate
        value = self.value_head(features)  # B x 1
        
        return coords, selected_log_probs, attention_map, value
    
    def get_attention_map(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Get the raw attention map without sampling points.
        Useful for visualization.
        
        Args:
            image_features: Feature maps from encoder, shape (B, C, H, W)
            
        Returns:
            attention_map: Attention heatmap (B, 1, H, W)
        """
        features = self.feature_processor(image_features)
        attention_map = self.attention_map_head(features)
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
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate attention points using Gaussian policy.
        
        Args:
            image_features: Feature maps from encoder, shape (B, C, H, W)
            clinical_features: Optional clinical data, shape (B, num_features)
            deterministic: If True, use mean; if False, sample from Gaussian
            
        Returns:
            coords: Point coordinates in image space (B, k, 2)
            log_probs: Log probabilities of selected points (B, k)
            value: State value estimate (B, 1)
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
        
        if deterministic:
            coords = coord_mean
            # Compute log prob at mean
            log_probs = -0.5 * torch.sum(coord_logstd + 0.5 * torch.log(torch.tensor(2 * 3.14159)), dim=2)
        else:
            # Sample from Gaussian
            noise = torch.randn_like(coord_mean)
            coords = coord_mean + noise * coord_std
            
            # Compute log probability
            log_probs = -0.5 * (
                torch.sum(((coords - coord_mean) / coord_std) ** 2, dim=2) +
                torch.sum(2 * coord_logstd, dim=2) +
                2 * torch.log(torch.tensor(2 * 3.14159))
            )
        
        # Clip coordinates to valid range [0, 1] and scale to image size
        coords = torch.clamp(coords, min=0.0, max=1.0)
        coords = coords * self.image_size
        
        # Compute value estimate
        value = self.value_head(features)  # B x 1
        
        return coords, log_probs, value

