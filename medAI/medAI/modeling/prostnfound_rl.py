"""
ProstNFound-RL: RL-Guided Attention for Prostate Cancer Detection

This module extends ProstNFound with reinforcement learning-based attention
to actively identify suspicious regions for improved cancer detection.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from .prostnfound import ProstNFound
from .rl_attention_policy import RLAttentionPolicy, RLAttentionPolicyGaussian
from .registry import register_model, create_model


class ProstNFoundRL(nn.Module):
    """
    ProstNFound with RL-guided attention mechanism.
    
    This wrapper adds an RL policy network on top of ProstNFound that learns
    to identify suspicious regions. These regions are provided as point prompts
    to guide the decoder's attention.
    
    Key Features:
    - Prostate mask-aware attention: Policy is penalized for selecting points outside prostate
    - Attention gating: RL attention map modulates decoder features for stronger focus
    - Clinical feature integration: Uses clinical data to guide attention
    
    Args:
        prostnfound_model: Base ProstNFound model
        num_attention_points: Number of attention points to generate (default: 3)
        policy_type: Type of policy network ('categorical' or 'gaussian')
        policy_hidden_dim: Hidden dimension for policy network (default: 512)
        use_clinical_in_policy: Whether to use clinical features in policy (default: True)
        freeze_prostnfound: Whether to freeze ProstNFound weights during RL training (default: False)
        temperature: Temperature for sampling (categorical policy only, default: 1.0)
        prostate_mask_penalty: Penalty for attention points outside prostate (default: 10.0)
        use_attention_gating: Whether to use RL attention to gate decoder features (default: True)
        attention_gate_strength: Strength of attention gating (0-1, default: 0.3)
    """
    
    def __init__(
        self,
        prostnfound_model: ProstNFound,
        num_attention_points: int = 3,
        policy_type: str = 'categorical',
        policy_hidden_dim: int = 512,
        use_clinical_in_policy: bool = True,
        freeze_prostnfound: bool = False,
        temperature: float = 1.0,
        prostate_mask_penalty: float = 10.0,
        use_attention_gating: bool = True,
        attention_gate_strength: float = 0.3,
    ):
        super().__init__()
        
        self.prostnfound = prostnfound_model
        self.num_attention_points = num_attention_points
        self.policy_type = policy_type
        self.use_clinical_in_policy = use_clinical_in_policy
        self.freeze_prostnfound = freeze_prostnfound
        self.prostate_mask_penalty = prostate_mask_penalty
        self.use_attention_gating = use_attention_gating
        self.attention_gate_strength = attention_gate_strength
        
        # Freeze ProstNFound if requested
        if freeze_prostnfound:
            logging.info("Freezing ProstNFound model weights")
            for param in self.prostnfound.parameters():
                param.requires_grad = False
        
        # Get feature dimension from encoder
        # MedSAM encoder outputs 256-dimensional features
        feature_dim = 256
        
        # Create policy network
        if policy_type == 'categorical':
            self.policy = RLAttentionPolicy(
                feature_dim=feature_dim,
                hidden_dim=policy_hidden_dim,
                num_attention_points=num_attention_points,
                image_size=256,  # Standard image size
                use_clinical_features=use_clinical_in_policy,
                temperature=temperature,
                prostate_mask_penalty=prostate_mask_penalty,
            )
        elif policy_type == 'gaussian':
            self.policy = RLAttentionPolicyGaussian(
                feature_dim=feature_dim,
                hidden_dim=policy_hidden_dim,
                num_attention_points=num_attention_points,
                image_size=256,
                use_clinical_features=use_clinical_in_policy,
                prostate_mask_penalty=prostate_mask_penalty,
            )
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}. Must be 'categorical' or 'gaussian'")
        
        # Attention gating layer - transforms RL attention map to feature modulation
        if use_attention_gating:
            self.attention_gate = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
                nn.Sigmoid(),
            )
        
        logging.info(f"Created ProstNFoundRL with {policy_type} policy, "
                    f"{num_attention_points} attention points, "
                    f"prostate_mask_penalty={prostate_mask_penalty}, "
                    f"attention_gating={use_attention_gating}")
    
    @property
    def prompts(self):
        """Pass through prompts from base model."""
        return self.prostnfound.prompts
    
    @property
    def device(self):
        """Get device of the model."""
        return next(self.parameters()).device
    
    def forward(
        self,
        image: torch.Tensor,
        rf_image: Optional[torch.Tensor] = None,
        prostate_mask: Optional[torch.Tensor] = None,
        needle_mask: Optional[torch.Tensor] = None,
        output_mode: Optional[str] = None,
        deterministic: bool = False,
        return_rl_info: bool = True,
        rl_actions: Optional[torch.Tensor] = None,
        **prompts,
    ) -> Dict[str, Any]:
        """
        Forward pass with RL-guided attention.
        
        Args:
            image: B-mode ultrasound images (B, C, H, W)
            rf_image: Optional RF ultrasound images
            prostate_mask: Prostate segmentation mask (B, 1, H, W)
            needle_mask: Needle region mask (B, 1, H, W)
            output_mode: Output mode ('heatmaps', 'classifier', or 'all')
            deterministic: If True, use deterministic policy; if False, sample
            return_rl_info: If True, return RL-specific information
            rl_actions: Optional actions to reuse (B, k) or (B, k, 2)
            **prompts: Clinical prompts (age, psa, etc.)
            
        Returns:
            Dictionary containing:
                - mask_logits: Cancer detection heatmap logits
                - cls_outputs: Classification outputs (if applicable)
                - iou: IoU predictions (if applicable)
                - rl_attention_coords: Coordinates of attention points (B, k, 2)
                - rl_log_probs: Log probabilities of actions (B, k)
                - rl_attention_map: Raw attention heatmap (B, 1, H, W)
                - rl_value: State value estimates (B, 1)
                - rl_outside_prostate_ratio: Fraction of points outside prostate (B,)
                - rl_actions: Actions used (B, k) or (B, k, 2)
        """
        output_mode = output_mode or self.prostnfound.output_mode
        B, C, H, W = image.shape
        
        # Step 1: Extract features using frozen MedSAM encoder
        with torch.set_grad_enabled(not self.freeze_prostnfound):
            image_feats = self.prostnfound.medsam_model.image_encoder(image)
            image_feats = self.prostnfound.img_emb_dropout(image_feats)
            if self.prostnfound.upsample is not None:
                image_feats = self.prostnfound.upsample(image_feats)
        
        # Step 2: Generate attention points using RL policy
        # Prepare clinical features for policy if needed
        clinical_features = None
        if self.use_clinical_in_policy:
            # Collect clinical features (pad to 4 features)
            clinical_list = []
            for prompt_name in ['age', 'psa', 'approx_psa_density', 'base_apex_encoding']:
                if prompt_name in prompts and prompts[prompt_name] is not None:
                    feat = prompts[prompt_name]
                    if feat.ndim == 1:
                        feat = feat[:, None]
                    clinical_list.append(feat)
                else:
                    # Use zeros as placeholder
                    clinical_list.append(torch.zeros(B, 1, device=image.device))
            
            clinical_features = torch.cat(clinical_list, dim=1)  # B x 4
        
        # Ensure prostate_mask is in correct format (B, 1, H, W) for policy
        # The policy's _apply_prostate_mask expects this format for F.interpolate
        policy_prostate_mask = None
        if prostate_mask is not None:
            if prostate_mask.ndim == 3:
                # (B, H, W) -> (B, 1, H, W)
                policy_prostate_mask = prostate_mask.unsqueeze(1)
            elif prostate_mask.ndim == 4:
                # Already (B, C, H, W), ensure C=1
                if prostate_mask.shape[1] != 1:
                    # If multiple channels, take first or average
                    policy_prostate_mask = prostate_mask[:, 0:1, :, :]
                else:
                    policy_prostate_mask = prostate_mask
            else:
                logging.warning(f"Unexpected prostate_mask shape: {prostate_mask.shape}, expected (B, H, W) or (B, 1, H, W)")
                policy_prostate_mask = None
        
        # Get attention points from policy with prostate mask constraint
        if self.policy_type == 'categorical':
            rl_coords, rl_log_probs, rl_attention_map, rl_value, outside_prostate_ratio, rl_actions_out = self.policy(
                image_feats,
                clinical_features=clinical_features,
                prostate_mask=policy_prostate_mask,
                deterministic=deterministic,
                given_actions=rl_actions,
            )
        else:  # gaussian
            rl_coords, rl_log_probs, rl_value, outside_prostate_ratio, rl_actions_out = self.policy(
                image_feats,
                clinical_features=clinical_features,
                prostate_mask=policy_prostate_mask,
                deterministic=deterministic,
                given_actions=rl_actions,
            )
            rl_attention_map = None
        
        # Step 3: Encode attention points as sparse prompts using SAM's prompt encoder
        # SAM expects point prompts as (coords, labels) where:
        # - coords: (B, N, 2) in [x, y] format
        # - labels: (B, N) where 1 = foreground point, 0 = background point
        
        # We'll use all points as foreground (suspicious regions)
        point_labels = torch.ones(B, self.num_attention_points, device=rl_coords.device)
        
        # Encode the attention points
        attention_point_embeddings = self.prostnfound.medsam_model.prompt_encoder._embed_points(
            rl_coords,
            point_labels,
            pad=False,
        )  # B x k x 256
        
        # Step 4: Continue with standard ProstNFound forward pass
        # Process prostate mask prompt
        if self.prostnfound.use_prostate_mask_prompt:
            if (
                prostate_mask is None
                or self.prostnfound.prompt_dropout > 0
                and self.prostnfound.training
                and torch.rand(1) < self.prostnfound.prompt_dropout
            ):
                mask = None
            else:
                B_mask, C_mask, H_mask, W_mask = prostate_mask.shape
                if H_mask != 256 or W_mask != 256:
                    prostate_mask = torch.nn.functional.interpolate(
                        prostate_mask, size=(256, 256)
                    )
                mask = prostate_mask
        else:
            mask = None
        
        # Get base sparse and dense embeddings
        sparse_embedding, dense_embedding = self.prostnfound.medsam_model.prompt_encoder.forward(
            None, None, mask
        )
        
        # Resize embeddings if needed
        if (dense_embedding.shape[-2] != image_feats.shape[-2]) or (
            dense_embedding.shape[-1] != image_feats.shape[-1]
        ):
            dense_embedding = torch.nn.functional.interpolate(
                dense_embedding,
                size=image_feats.shape[-2:],
            )
        
        dense_embedding = dense_embedding.repeat_interleave(len(image), 0)
        
        if self.prostnfound.use_cnn_dense_prompt:
            assert self.prostnfound.cnn_for_dense_embedding is not None
            cnn_emb = self.prostnfound.cnn_for_dense_embedding(image).permute(0, 3, 1, 2)
            dense_embedding = dense_embedding + cnn_emb
        
        sparse_embedding = sparse_embedding.repeat_interleave(len(image), 0)
        
        # Add RL attention point embeddings to sparse embeddings
        sparse_embedding = torch.cat([sparse_embedding, attention_point_embeddings], dim=1)
        
        # Process clinical prompts
        for prompt_name, prompt_value in prompts.items():
            if (
                prompt_value is None
                or self.prostnfound.prompt_dropout > 0
                and self.prostnfound.training
                and torch.rand(1) < self.prostnfound.prompt_dropout
            ):
                prompt_embedding = self.prostnfound.null_prompt.repeat_interleave(len(image), 0)
            else:
                if prompt_name in self.prostnfound.floating_point_prompts:
                    prompt_embedding = self.prostnfound.floating_point_prompt_modules[prompt_name](
                        prompt_value
                    )
                elif prompt_name in self.prostnfound.discrete_prompts:
                    prompt_embedding = self.prostnfound.integer_prompt_modules[prompt_name](
                        prompt_value
                    )
                else:
                    # Skip unknown prompts
                    continue
            
            prompt_embedding = prompt_embedding[:, None, :]
            sparse_embedding = torch.cat([sparse_embedding, prompt_embedding], dim=1)
        
        # Add data-independent prompts
        if self.prostnfound.num_data_independent_prompts > 0:
            sparse_embedding = torch.cat(
                [
                    sparse_embedding,
                    self.prostnfound.data_independent_prompts.repeat_interleave(B, 0),
                ],
                dim=1,
            )
        
        # Add CNN patch features if used
        if self.prostnfound.use_sparse_cnn_patch_features:
            patch_cnn_sparse_embeddings = self.prostnfound.get_cnn_patch_embedding_bmode(
                image, needle_mask, prostate_mask
            )
            if patch_cnn_sparse_embeddings is not None:
                sparse_embedding = torch.cat(
                    [sparse_embedding, patch_cnn_sparse_embeddings], dim=1
                )
        
        if self.prostnfound.use_sparse_cnn_patch_features_rf:
            patch_cnn_sparse_embeddings = self.prostnfound.get_cnn_patch_embedding_rf(
                rf_image, needle_mask, prostate_mask
            )
            if patch_cnn_sparse_embeddings is not None:
                sparse_embedding = torch.cat(
                    [sparse_embedding, patch_cnn_sparse_embeddings], dim=1
                )
        
        # Get positional encoding
        pe = self.prostnfound.medsam_model.prompt_encoder.get_dense_pe()
        if (pe.shape[-2] != image_feats.shape[-2]) or (
            pe.shape[-1] != image_feats.shape[-1]
        ):
            pe = torch.nn.functional.interpolate(
                pe,
                size=image_feats.shape[-2:],
            )
        
        # Step 5: Apply attention gating to image features
        # This forces the decoder to focus on RL-highlighted regions
        if self.use_attention_gating and rl_attention_map is not None:
            # Resize attention map to match feature dimensions
            attn_map_resized = torch.nn.functional.interpolate(
                rl_attention_map,
                size=image_feats.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
            
            # Convert attention logits to gate values [0, 1]
            # Use sigmoid to get soft gate, then apply strength modulation
            gate = self.attention_gate(attn_map_resized)  # B x 1 x H x W
            
            # Soft gating: (1 - strength) * original + strength * gated
            # This ensures we don't completely ignore non-attended regions
            # but give more weight to attended regions
            gated_feats = image_feats * (1.0 + self.attention_gate_strength * (gate - 0.5))
        else:
            gated_feats = image_feats
        
        # Pass through mask decoder with gated features
        mask_logits, iou = self.prostnfound.medsam_model.mask_decoder.forward(
            gated_feats,
            pe,
            sparse_embedding,
            dense_embedding,
            multimask_output=False,
        )
        
        # Apply temperature and bias
        mask_logits = (
            mask_logits / self.prostnfound.temperature[None, None, None, :]
            + self.prostnfound.bias[None, None, None, :]
        )
        
        # Class decoder if available
        if self.prostnfound.class_decoder is not None:
            cls_outputs = self.prostnfound.class_decoder.forward(
                gated_feats,  # Use gated features for classifier too
                pe,
                sparse_embedding,
                dense_embedding,
            )
        else:
            cls_outputs = None
        
        # Prepare output
        if output_mode == "heatmaps":
            output = mask_logits
        elif output_mode == "classifier":
            assert cls_outputs is not None
            output = cls_outputs[0]
        else:  # "all"
            output = dict(
                mask_logits=mask_logits,
                cls_outputs=cls_outputs,
                iou=iou,
            )
        
        # Add RL information if requested
        if return_rl_info:
            if isinstance(output, dict):
                output['rl_attention_coords'] = rl_coords
                output['rl_log_probs'] = rl_log_probs
                output['rl_attention_map'] = rl_attention_map
                output['rl_value'] = rl_value
                output['rl_outside_prostate_ratio'] = outside_prostate_ratio
                output['rl_actions'] = rl_actions_out
            else:
                # Convert to dict if not already
                output = {
                    'mask_logits': output,
                    'rl_attention_coords': rl_coords,
                    'rl_log_probs': rl_log_probs,
                    'rl_attention_map': rl_attention_map,
                    'rl_value': rl_value,
                    'rl_outside_prostate_ratio': outside_prostate_ratio,
                    'rl_actions': rl_actions_out,
                }
        
        return output
    
    def get_params_groups(self):
        """
        Get parameter groups for optimizer.
        
        Returns three groups:
        1. Encoder parameters (from ProstNFound)
        2. Warmup parameters (from ProstNFound decoders + policy network)
        3. CNN parameters (from ProstNFound)
        """
        if self.freeze_prostnfound:
            # Only policy parameters are trainable
            encoder_parameters = []
            warmup_parameters = list(self.policy.parameters())
            cnn_parameters = []
        else:
            # Get ProstNFound param groups
            encoder_parameters, warmup_parameters, cnn_parameters = (
                self.prostnfound.get_params_groups()
            )
            # Add policy to warmup group
            warmup_parameters = list(warmup_parameters) + list(self.policy.parameters())
        
        return encoder_parameters, warmup_parameters, cnn_parameters


def create_prostnfound_rl(
    prostnfound_model: ProstNFound,
    num_attention_points: int = 3,
    policy_type: str = 'categorical',
    policy_hidden_dim: int = 512,
    use_clinical_in_policy: bool = True,
    freeze_prostnfound: bool = False,
    temperature: float = 1.0,
    prostate_mask_penalty: float = 10.0,
    use_attention_gating: bool = True,
    attention_gate_strength: float = 0.3,
) -> ProstNFoundRL:
    """
    Factory function to create ProstNFoundRL model.
    
    Args:
        prostnfound_model: Base ProstNFound model
        num_attention_points: Number of attention points (default: 3)
        policy_type: 'categorical' or 'gaussian'
        policy_hidden_dim: Hidden dimension for policy network
        use_clinical_in_policy: Use clinical features in policy
        freeze_prostnfound: Freeze base model weights
        temperature: Sampling temperature
        prostate_mask_penalty: Penalty for points outside prostate (default: 10.0)
        use_attention_gating: Whether to use attention gating (default: True)
        attention_gate_strength: Strength of attention gating (default: 0.3)
        
    Returns:
        ProstNFoundRL model
    """
    return ProstNFoundRL(
        prostnfound_model=prostnfound_model,
        num_attention_points=num_attention_points,
        policy_type=policy_type,
        policy_hidden_dim=policy_hidden_dim,
        use_clinical_in_policy=use_clinical_in_policy,
        freeze_prostnfound=freeze_prostnfound,
        temperature=temperature,
        prostate_mask_penalty=prostate_mask_penalty,
        use_attention_gating=use_attention_gating,
        attention_gate_strength=attention_gate_strength,
    )


@register_model
def prostnfound_rl_adapter_medsam(
    backbone: str = "medsam_adapter",
    backbone_kw: dict = {},
    prompts: list = [],
    num_attention_points: int = 3,
    policy_type: str = 'categorical',
    policy_hidden_dim: int = 512,
    use_clinical_in_policy: bool = True,
    freeze_prostnfound: bool = False,
    temperature: float = 1.0,
    prostate_mask_penalty: float = 10.0,
    use_attention_gating: bool = True,
    attention_gate_strength: float = 0.3,
    prostnfound_kw: dict = {},
    **kwargs,
):
    """
    ProstNFound-RL with adapter MedSAM backbone.
    
    This creates a ProstNFound model with RL-guided attention.
    
    Args:
        backbone: Backbone model name
        backbone_kw: Keyword arguments for backbone
        prompts: List of clinical prompts to use
        num_attention_points: Number of RL attention points
        policy_type: 'categorical' or 'gaussian'
        policy_hidden_dim: Hidden dimension for policy network
        use_clinical_in_policy: Use clinical features in policy
        freeze_prostnfound: Freeze base ProstNFound weights
        temperature: Sampling temperature for policy
        prostate_mask_penalty: Penalty for points outside prostate
        use_attention_gating: Use RL attention to gate decoder features
        attention_gate_strength: Strength of attention gating (0-1)
        prostnfound_kw: Additional kwargs for ProstNFound
        **kwargs: Additional kwargs
        
    Returns:
        ProstNFoundRL model
    """
    # Import here to avoid circular dependency
    from . import prostnfound as pnf_module
    
    # Create base ProstNFound model
    floating_point_prompts = []
    for prompt in prompts:
        if prompt in [
            "age",
            "psa",
            "approx_psa_density",
            "base_apex_encoding",
            "mid_lateral_encoding",
            "family_history",
        ]:
            floating_point_prompts.append(prompt)
        elif prompt == 'pos': 
            floating_point_prompts.append("base_apex_encoding")
            floating_point_prompts.append("mid_lateral_encoding")
        elif prompt == 'psad': 
            floating_point_prompts.append('approx_psa_density')
        else: 
            raise ValueError(f"Unknown prompt {prompt}")
    
    # Create backbone
    backbone_model = create_model(backbone, **backbone_kw)
    
    # Merge prostnfound_kw and kwargs (prostnfound_kw takes precedence)
    pnf_kwargs = {**kwargs, **prostnfound_kw}
    
    # Create ProstNFound
    base_model = ProstNFound(
        backbone_model,
        floating_point_prompts=floating_point_prompts,
        **pnf_kwargs,
    )
    
    # Wrap with RL
    model = ProstNFoundRL(
        prostnfound_model=base_model,
        num_attention_points=num_attention_points,
        policy_type=policy_type,
        policy_hidden_dim=policy_hidden_dim,
        use_clinical_in_policy=use_clinical_in_policy,
        freeze_prostnfound=freeze_prostnfound,
        temperature=temperature,
        prostate_mask_penalty=prostate_mask_penalty,
        use_attention_gating=use_attention_gating,
        attention_gate_strength=attention_gate_strength,
    )
    
    return model


@register_model
def prostnfound_rl_adapter_medsam_legacy(
    adapter_dim: int = 256,
    prompts: list = [],
    num_attention_points: int = 3,
    policy_type: str = 'categorical',
    policy_hidden_dim: int = 512,
    use_clinical_in_policy: bool = True,
    freeze_prostnfound: bool = False,
    temperature: float = 1.0,
    prostate_mask_penalty: float = 10.0,
    use_attention_gating: bool = True,
    attention_gate_strength: float = 0.3,
    upsample: bool = False,
    use_class_decoder: bool = True,
    **kwargs,
):
    """
    ProstNFound-RL with legacy adapter MedSAM backbone (256 resolution).
    Compatible with existing ProstNFound+ checkpoints.
    
    This follows the same pattern as prostnfound_adapter_medsam_legacy.
    
    Key improvements:
    - prostate_mask_penalty: Penalizes attention outside prostate region
    - use_attention_gating: Makes decoder focus on RL-prompted regions
    - attention_gate_strength: Controls how strongly RL attention guides decoder
    """
    from . import sam
    
    if upsample:
        backbone = "medsam_adapter_upsample"
    else:
        backbone = "medsam_adapter"
    
    return prostnfound_rl_adapter_medsam(
        backbone=backbone,
        backbone_kw=dict(adapter_dim=adapter_dim, **kwargs.pop("backbone_kw", {})),
        prompts=prompts,
        num_attention_points=num_attention_points,
        policy_type=policy_type,
        policy_hidden_dim=policy_hidden_dim,
        use_clinical_in_policy=use_clinical_in_policy,
        freeze_prostnfound=freeze_prostnfound,
        temperature=temperature,
        prostate_mask_penalty=prostate_mask_penalty,
        use_attention_gating=use_attention_gating,
        attention_gate_strength=attention_gate_strength,
        prostnfound_kw={'use_class_decoder': use_class_decoder},
        **kwargs,
    )

