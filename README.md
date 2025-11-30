# ProstNFound-RL: Reinforcement Learning-Guided Attention

This extension adds reinforcement learning (RL) guided attention to ProstNFound+ for improved prostate cancer detection.

## Overview

ProstNFound-RL introduces a lightweight RL policy network that learns to identify suspicious regions in ultrasound images. These regions are provided as attention prompts to the MedSAM decoder, guiding it to focus on clinically relevant areas.

### Key Features

- **RL-Guided Attention**: Policy network learns where to look for suspicious regions
- **GRPO Training**: Uses Group Relative Policy Optimization for stable training
- **Configurable**: Can easily switch between baseline and RL modes
- **Backward Compatible**: Existing ProstNFound+ training still works unchanged
- **Flexible Policies**: Supports categorical and Gaussian policy types

## Architecture

```
Input Image → MedSAM Encoder (frozen) → Feature Maps
                                           ↓
                                    RL Policy Network
                                           ↓
                                  k Attention Points
                                           ↓
                          Point Prompts (via SAM encoder)
                                           ↓
Clinical Prompts + Attention Prompts → Sparse Embeddings
                                           ↓
                                    MedSAM Decoder
                                           ↓
                                  Cancer Heatmap
```

### Components

1. **RLAttentionPolicy** (`medAI/modeling/rl_attention_policy.py`)
   - Lightweight CNN that processes encoder features
   - Outputs k point coordinates as attention prompts
   - Two variants: Categorical (sample from heatmap) and Gaussian (direct coordinate prediction)

2. **ProstNFoundRL** (`medAI/modeling/prostnfound_rl.py`)
   - Wrapper that integrates RL policy with ProstNFound
   - Encodes attention points as SAM sparse prompts
   - Can freeze base model weights for policy-only training

3. **GRPO** (`medAI/modeling/grpo.py`)
   - Group Relative Policy Optimization algorithm
   - **Within-Image Comparison**: Samples multiple attention configurations per image
   - Normalizes advantages within each image group (not across batch)
   - This is critical because different images have different difficulty levels
   - Includes value function and entropy regularization

4. **Reward Computer** (`prostnfound/src/rl_loss.py`)
   - Computes rewards based on detection performance
   - Bonus for correctly identifying csPCa
   - Multiple reward modes: loss-based, accuracy-based, or combined

## Installation

The RL components are integrated into the existing codebase. No additional installation is needed beyond the standard ProstNFound requirements.

## Usage

### Training with RL

#### 1. Quick Test (Baseline Configuration)

For initial testing with limited data:

```bash
python train_rl.py -c cfg/train/pnf_plus_rl_baseline.yaml
```

This config:
- Uses smaller batch size and limited data
- Freezes encoder for faster iteration
- Runs for only 5 epochs
- Good for debugging and initial validation

#### 2. Full Training (K-Fold)

For full experiments:

```bash
# Train on fold 0
python train_rl.py -c cfg/train/pnf_plus_rl_kfold.yaml

# Override fold number
python train_rl.py -c cfg/train/pnf_plus_rl_kfold.yaml data.fold=1

# Run all folds
for fold in {0..4}; do
    python train_rl.py -c cfg/train/pnf_plus_rl_kfold.yaml data.fold=$fold
done
```

#### 3. Fine-tuning from ProstNFound+ Checkpoint

To initialize from a trained baseline:

```bash
python train_rl.py -c cfg/train/pnf_plus_rl_kfold.yaml \
    model_checkpoint=checkpoints/best_baseline.pth
```

This will:
- Load the ProstNFound+ weights
- Initialize the RL policy randomly
- Train end-to-end or freeze base model (depending on config)

### Training Without RL (Baseline)

The original training script still works:

```bash
python train.py -c cfg/train/pnf_plus_kfold.yaml
```

## Configuration

### Key Config Parameters

#### Model Configuration

```yaml
model: prostnfound_rl_adapter_medsam_legacy
model_kw:
  prompts: [age, psa]  # Clinical prompts
  use_class_decoder: true
  
  # RL-specific parameters
  num_attention_points: 3  # Number of attention points (k)
  policy_type: categorical  # 'categorical' or 'gaussian'
  policy_hidden_dim: 512  # Hidden dimension for policy network
  use_clinical_in_policy: true  # Use clinical data in policy
  freeze_prostnfound: false  # Set true to only train policy
  temperature: 1.0  # Sampling temperature
```

#### RL Training Parameters

```yaml
use_rl: true  # Enable RL training
rl_mode: grpo  # RL algorithm
rl_reward_mode: loss_based  # 'loss_based', 'accuracy_based', or 'combined'
rl_cspca_bonus: 2.0  # Reward multiplier for csPCa
rl_normalize_rewards: true  # Only used when num_samples_per_image=1

# GRPO hyperparameters
rl_num_update_epochs: 4  # Updates per batch
rl_clip_eps: 0.2  # PPO clipping
rl_entropy_coef: 0.01  # Entropy bonus
rl_value_coef: 0.5  # Value loss weight
rl_max_grad_norm: 0.5  # Gradient clipping
rl_gamma: 1.0  # Discount factor

# Within-Image Comparison (key improvement!)
# Instead of comparing across different images (which have different difficulty),
# we sample multiple attention configurations per image and compare within each.
rl_num_samples_per_image: 4  # Number of rollouts per image (set to 1 to disable)
```

### Within-Image Comparison (Key Feature)

A critical improvement in this implementation is **within-image comparison**. 

**The Problem**: Different ultrasound images have different difficulty levels. Some images have clear cancer regions while others are ambiguous. If we normalize rewards/advantages across the entire batch, we're comparing apples to oranges - an easy image will always get higher rewards than a hard one.

**The Solution**: For each image, we sample multiple rollouts (different attention location configurations) and compare them **within the same image**. This allows the model to learn:
- "For this specific image, attention configuration A works better than B"
- Rather than: "Image 1 with any configuration is better than Image 2"

This is implemented by:
1. Sampling `rl_num_samples_per_image` attention configurations per image
2. Computing rewards for each configuration
3. Normalizing advantages within each image group in GRPO
4. This provides a fair comparison regardless of image difficulty

### Hyperparameter Tuning

Key hyperparameters to tune:

1. **num_attention_points** (3-5): More points = more exploration, but harder to learn
2. **policy_hidden_dim** (256-512): Larger = more capacity, but slower
3. **rl_cspca_bonus** (1.5-3.0): Higher = focus more on csPCa
4. **rl_entropy_coef** (0.001-0.05): Higher = more exploration
5. **temperature** (0.5-2.0): Higher = more stochastic sampling
6. **rl_num_samples_per_image** (4-8): More samples = better within-image comparison, but slower training. Set to 1 to disable within-image comparison.

## Evaluation

### Standard Metrics

All standard ProstNFound metrics are computed:
- Core-level AUROC
- Sensitivity at specificity thresholds
- Heatmap quality metrics

### RL-Specific Metrics

Additional metrics logged during training:
- `train_rl/reward_mean`: Average reward
- `train_rl/policy_loss`: Policy gradient loss
- `train_rl/entropy`: Policy entropy (exploration)
- `train_rl/value_loss`: Value function loss
- `train_rl/ratio_mean`: Policy update ratio
- `train_rl/within_image_reward_std`: Standard deviation of rewards within each image (measures diversity of sampled attention configurations)

### Visualizing Attention Points

To visualize where the RL agent is looking:

```python
from medAI.modeling.prostnfound_rl import ProstNFoundRL
import matplotlib.pyplot as plt

# Load model
model = ProstNFoundRL(...)
model.eval()

# Forward pass
outputs = model(image, ..., deterministic=True, return_rl_info=True)

# Get attention points
coords = outputs['rl_attention_coords']  # (B, k, 2)
attention_map = outputs['rl_attention_map']  # (B, 1, H, W) if categorical

# Visualize
plt.imshow(image[0].permute(1, 2, 0))
plt.scatter(coords[0, :, 0], coords[0, :, 1], c='red', marker='x', s=100)
plt.show()
```

## Expected Results

Based on the proposal, we expect:

1. **Quantitative Improvements**:
   - +1-3% AUROC over baseline ProstNFound+
   - Higher sensitivity for csPCa detection
   - Better calibration

2. **Qualitative Improvements**:
   - Attention points align with suspicious regions
   - Interpretable decision-making process
   - Reduced false positives outside prostate

3. **Ablations**:
   - Increasing k (num attention points) should improve performance
   - Categorical policy may work better than Gaussian initially
   - csPCa bonus is critical for clinical relevance

## Troubleshooting

### Common Issues

**Issue**: NaN losses or rewards

**Solution**: 
- Reduce learning rate (try `lr: 1e-5`)
- Reduce entropy coefficient
- Check data loading (ensure masks are valid)

---

**Issue**: Policy not learning (constant attention points)

**Solution**:
- Increase entropy coefficient for more exploration
- Reduce policy hidden dim to prevent overfitting
- Try Gaussian policy instead of categorical

---

**Issue**: Training unstable

**Solution**:
- Freeze ProstNFound weights initially (`freeze_prostnfound: true`)
- Reduce gradient clipping threshold
- Increase batch size
- Use fewer RL update epochs

---

**Issue**: Out of memory

**Solution**:
- Reduce batch size
- Reduce policy_hidden_dim
- Reduce num_attention_points
- Disable torch.compile

## Code Structure

```
prostnfound/
├── medAI/medAI/modeling/
│   ├── rl_attention_policy.py      # Policy networks
│   ├── prostnfound_rl.py            # RL wrapper
│   ├── grpo.py                      # GRPO algorithm
│   └── __init__.py                  # Model registry
├── prostnfound/
│   ├── train_rl.py                  # RL training script
│   ├── cfg/train/
│   │   ├── pnf_plus_rl_baseline.yaml  # Quick test config
│   │   └── pnf_plus_rl_kfold.yaml     # Full training config
│   └── src/
│       └── rl_loss.py               # Reward computation
└── README_RL.md                     # This file
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{prostnfound2024,
  title={ProstNFound: Integrating Foundation Models with Ultrasound Domain Knowledge and Clinical Context for Robust Prostate Cancer Detection},
  author={Wilson et al.},
  booktitle={MICCAI},
  year={2024}
}

@article{prostnfound_rl2025,
  title={ProstNFound-RL: Guided Attention with Reinforcement Learning for Robust Prostate Cancer Detection},
  author={Abootorabi, Mohammad Mahdi},
  journal={EECE 571S Project},
  year={2025}
}
```

## Future Work

Potential extensions:
- Multi-scale attention (attention at different resolutions)
- Recurrent policy for temporal/video data
- Meta-learning for few-shot adaptation
- Uncertainty-aware attention
- Integration with other RL algorithms (SAC, TD3)

## Contact

For questions or issues, please contact:
- Mohammad Mahdi Abootorabi (mahdi.abootorabi@ece.ubc.ca)

