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
    
    Args:
        prostnfound_model: Base ProstNFound model
        num_attention_points: Number of attention points to generate (default: 3)
        policy_type: Type of policy network ('categorical' or 'gaussian')
        policy_hidden_dim: Hidden dimension for policy network (default: 512)
        use_clinical_in_policy: Whether to use clinical features in policy (default: True)
        freeze_prostnfound: Whether to freeze ProstNFound weights during RL training (default: False)
        temperature: Temperature for sampling (categorical policy only, default: 1.0)
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
    ):
        super().__init__()
        
        self.prostnfound = prostnfound_model
        self.num_attention_points = num_attention_points
        self.policy_type = policy_type
        self.use_clinical_in_policy = use_clinical_in_policy
        self.freeze_prostnfound = freeze_prostnfound
        
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
            )
        elif policy_type == 'gaussian':
            self.policy = RLAttentionPolicyGaussian(
                feature_dim=feature_dim,
                hidden_dim=policy_hidden_dim,
                num_attention_points=num_attention_points,
                image_size=256,
                use_clinical_features=use_clinical_in_policy,
            )
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}. Must be 'categorical' or 'gaussian'")
        
        logging.info(f"Created ProstNFoundRL with {policy_type} policy, "
                    f"{num_attention_points} attention points")
    
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
        
        # Get attention points from policy
        if self.policy_type == 'categorical':
            rl_coords, rl_log_probs, rl_attention_map, rl_value = self.policy(
                image_feats,
                clinical_features=clinical_features,
                deterministic=deterministic,
            )
        else:  # gaussian
            rl_coords, rl_log_probs, rl_value = self.policy(
                image_feats,
                clinical_features=clinical_features,
                deterministic=deterministic,
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
        
        # Pass through mask decoder
        mask_logits, iou = self.prostnfound.medsam_model.mask_decoder.forward(
            image_feats,
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
                image_feats,
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
            else:
                # Convert to dict if not already
                output = {
                    'mask_logits': output,
                    'rl_attention_coords': rl_coords,
                    'rl_log_probs': rl_log_probs,
                    'rl_attention_map': rl_attention_map,
                    'rl_value': rl_value,
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
    upsample: bool = False,
    use_class_decoder: bool = True,
    **kwargs,
):
    """
    ProstNFound-RL with legacy adapter MedSAM backbone (256 resolution).
    Compatible with existing ProstNFound+ checkpoints.
    
    This follows the same pattern as prostnfound_adapter_medsam_legacy.
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
        prostnfound_kw={'use_class_decoder': use_class_decoder},
        **kwargs,
    )

