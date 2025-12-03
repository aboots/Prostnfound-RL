# ProstNFound-RL: Reinforcement Learning-Guided Attention for Prostate Cancer Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project extends **ProstNFound+** (a foundation model for prostate cancer detection in ultrasound) with **Reinforcement Learning** to learn where to focus attention when analyzing medical images. The RL agent learns to identify suspicious regions, which are then used to guide the segmentation decoder for improved cancer detection.

---

## 📋 Table of Contents

- [Motivation: Why Reinforcement Learning?](#-motivation-why-reinforcement-learning)
- [The Intuition](#-the-intuition)
- [Architecture Overview](#-architecture-overview)
- [Reward Function Design](#-reward-function-design)
- [GRPO Algorithm](#-grpo-algorithm)
- [Data Requirements](#-data-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Configuration](#-configuration)
- [Results](#-results)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)

---

## 🎯 Motivation: Why Reinforcement Learning?

### The Problem with Standard Supervised Learning

Traditional cancer detection models learn to predict cancer heatmaps directly from images. While effective, they have limitations:

1. **Passive Processing**: The model processes the entire image uniformly, without actively searching for suspicious regions
2. **No Exploration**: The model can't learn to "look around" - it only sees what it's shown
3. **Limited Interpretability**: It's hard to understand what the model is focusing on

### Why RL is the Right Tool

**Reinforcement Learning** allows the model to:

1. **Actively Explore**: The RL agent learns to sample different attention locations and discover which work best
2. **Learn from Outcomes**: Instead of being told exactly where cancer is pixel-by-pixel, the agent learns from the *outcome* of its decisions (did the prediction improve?)
3. **Develop Intuition**: Over time, the agent develops a policy for where to look based on image features
4. **Provide Interpretability**: We can visualize exactly where the agent decides to focus attention

### The Key Insight

> **Medical imaging experts don't just passively look at images—they actively search for suspicious regions based on experience.**

Our RL agent mimics this process: it learns WHERE to look (attention points) based on image features and clinical context, then uses these attention points to guide the segmentation model.

---

## 💡 The Intuition

### Human Expert vs. Our RL Agent

| Human Radiologist | Our RL Agent |
|-------------------|--------------|
| Looks at ultrasound image | Encodes image with MedSAM encoder |
| Uses experience to identify suspicious regions | RL policy network identifies attention points |
| Focuses on those regions for detailed analysis | Attention points become prompts for decoder |
| Makes diagnosis considering clinical context | Decoder produces cancer heatmap with clinical prompts |

### Why Within-Image Comparison?

Different ultrasound images have different difficulty levels:
- Some images have obvious cancer that's easy to detect
- Some images are ambiguous with subtle findings

If we compare rewards across different images, the model might just learn "easy images are good" rather than "good attention locations are good."

**Solution: Within-Image Comparison (GRPO)**
- For each image, we sample **multiple attention configurations** (e.g., 4-8 different sets of attention points)
- We compare these configurations **within the same image**
- The model learns: "For THIS specific image, attention configuration A works better than B"

This is like a medical student learning: "When I see this pattern, looking at region X is more informative than looking at region Y."

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ProstNFound-RL Architecture                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Input: B-mode Ultrasound Image                                                 │
│           │                                                                     │
│           ▼                                                                     │
│  ┌─────────────────────┐                                                        │
│  │  MedSAM Encoder     │ ──────────────► Image Features (256-dim)               │
│  │  (ViT-B, frozen/    │                      │                                 │
│  │   fine-tuned)       │                      │                                 │
│  └─────────────────────┘                      │                                 │
│                                               ▼                                 │
│                                    ┌──────────────────────┐                     │
│                                    │  RL Policy Network   │                     │
│  Clinical Features ────────────────│  (Categorical/       │                     │
│  (age, PSA, etc.)                  │   Gaussian/Patch)    │                     │
│                                    └──────────────────────┘                     │
│                                               │                                 │
│                                               ▼                                 │
│                                    k Attention Points (x, y coordinates)        │
│                                               │                                 │
│                                               ▼                                 │
│                                    ┌──────────────────────┐                     │
│                                    │  SAM Prompt Encoder  │                     │
│                                    │  (encode as prompts) │                     │
│                                    └──────────────────────┘                     │
│                                               │                                 │
│                                               ▼                                 │
│  ┌─────────────────────┐           ┌──────────────────────┐                     │
│  │  Clinical Prompts   │ ─────────►│   Sparse Embeddings  │                     │
│  │  (age, PSA encoded) │           │   (combined prompts) │                     │
│  └─────────────────────┘           └──────────────────────┘                     │
│                                               │                                 │
│                                               ▼                                 │
│                                    ┌──────────────────────┐                     │
│                                    │  MedSAM Decoder      │                     │
│                                    │  (Mask + Class head) │                     │
│                                    └──────────────────────┘                     │
│                                               │                                 │
│                           ┌───────────────────┼───────────────────┐             │
│                           ▼                                       ▼             │
│                  Cancer Heatmap                          Classification         │
│                  (spatial logits)                        (cancer/benign)        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Location | Description |
|-----------|----------|-------------|
| **RL Policy Network** | `medAI/modeling/rl_attention_policy.py` | Lightweight CNN that processes encoder features and outputs k attention point coordinates |
| **ProstNFoundRL** | `medAI/modeling/prostnfound_rl.py` | Wrapper that integrates RL policy with base ProstNFound model |
| **GRPO Algorithm** | `medAI/modeling/grpo.py` | Group Relative Policy Optimization for stable training |
| **Reward Computer** | `prostnfound/src/rl_loss.py` | Computes rewards based on detection performance |

### Policy Types

1. **Categorical Policy** (`policy_type: categorical`)
   - Generates a spatial heatmap over the image
   - Samples k attention points from this distribution
   - Best for: Learning precise attention locations

2. **Gaussian Policy** (`policy_type: gaussian`)
   - Directly predicts (mean, std) for each attention point
   - Samples from Gaussian distribution
   - Best for: Smooth, continuous attention regions

3. **Patch Policy** (`policy_type: patch`)
   - Selects k patches from the image
   - Samples multiple points within each patch
   - Best for: Regional attention (e.g., suspicious areas)

---

## 🏆 Reward Function Design

The reward function is crucial for RL training. We designed it to capture what matters clinically.

### Combined Reward (Recommended)

```python
reward = (
    heatmap_weight * ROI_involvement_reward +
    classification_weight * classification_reward
) * cspca_bonus_if_applicable
```

### Reward Components

#### 1. ROI Involvement Reward
Measures how well the predicted involvement matches ground truth:

```
reward = 1 - |predicted_involvement - true_involvement|
```
- `predicted_involvement`: Mean sigmoid probability in needle region
- `true_involvement`: Actual cancer involvement (0-1)
- Scaled to [-1, 1] range

**Intuition**: Rewards the model for correctly predicting the extent of cancer in the biopsy region.

#### 2. Classification Reward
Based on confidence for the correct class:

```
reward = 2 * P(correct_class) - 1
```
- High confidence on correct class → positive reward
- Wrong predictions → negative reward
- Scaled to [-1, 1] range

**Intuition**: Rewards the model for being confidently correct.

#### 3. csPCa Bonus
Clinically significant prostate cancer (Gleason Grade Group > 2) gets a bonus multiplier:

```python
if grade_group > 2:
    reward *= cspca_bonus  # e.g., 1.5-2.0x
```

**Intuition**: Missing aggressive cancer is more dangerous than missing indolent cancer.

### Reward Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| `combined_v2` | Classification + ROI involvement | **Recommended** - balanced signal |
| `confidence_based` | Confidence on correct predictions | Good for calibration |
| `classification_only` | Only classification head | When heatmaps aren't critical |
| `roi_only` | Only ROI involvement | When focusing on localization |
| `loss_based` | Negative BCE loss (legacy) | Not recommended |

---

## 🔄 GRPO Algorithm

### What is GRPO?

**Group Relative Policy Optimization** is a policy gradient method that:
1. Samples multiple "rollouts" (attention configurations) per image
2. Computes rewards for each rollout
3. Normalizes advantages **within each image group**
4. Updates policy using PPO-style clipping

### Why GRPO over Standard PPO?

| Standard PPO | GRPO |
|--------------|------|
| Normalizes across batch | Normalizes within image |
| Compares different images | Compares same image with different attention |
| Can be biased by image difficulty | Fair comparison regardless of difficulty |

### GRPO Loss Function

```python
# Advantage computation (within-image)
advantages = (rewards - group_mean) / (group_std + ε)

# Policy loss (PPO-style clipping)
ratio = exp(new_log_prob - old_log_prob)
policy_loss = -min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)

# Entropy bonus (exploration)
entropy_loss = -entropy_coef * mean(log_probs)

# KL penalty (stability)
kl_loss = kl_coef * KL(old_policy || new_policy)

total_loss = policy_loss + kl_loss - entropy_loss
```

### Is This True RL?

**Yes!** Key characteristics that make this true RL:
1. ✅ **Stochastic policy** - samples attention points
2. ✅ **Rewards from outcomes** - not directly differentiable
3. ✅ **Policy gradients** - uses REINFORCE/PPO theorem
4. ✅ **Exploration** - entropy bonus encourages diverse sampling
5. ✅ **No direct supervision** on attention locations

See [GRPO_VS_SUPERVISED_ANALYSIS.md](GRPO_VS_SUPERVISED_ANALYSIS.md) for detailed gradient flow analysis.

---

## 📊 Data Requirements

### Primary Dataset: NCT2013

The main dataset is the **NCT2013 Prostate Cancer Dataset** containing:
- **B-mode ultrasound images** from transrectal ultrasound-guided biopsies
- **Prostate masks** delineating the prostate gland
- **Needle masks** showing the biopsy needle trajectory
- **Pathology labels** (Gleason grades, involvement percentage)
- **Clinical metadata** (age, PSA, family history)

### Data Format

Each sample contains:

| Field | Shape/Type | Description |
|-------|------------|-------------|
| `bmode` | (H, W, 3) uint8 | B-mode ultrasound image |
| `prostate_mask` | (H, W) uint8 | Binary prostate segmentation |
| `needle_mask` | (H, W) uint8 | Binary needle region mask |
| `core_id` | str | Unique identifier |
| `grade` | str | Pathology grade ("Benign", "GG1", etc.) |
| `grade_group` | int | Gleason Grade Group (1-5) |
| `involvement` | float | Cancer involvement percentage |
| `psa` | float | PSA value (ng/mL) |
| `age` | int | Patient age |
| `family_history` | int | Family history of prostate cancer |

### Data Access Setup

1. **Configure data path** in `medAI/datasets/nct2013/data_access.py`:
   ```python
   DATA_ROOT = "/path/to/nct2013/data"
   ```

2. **Required files**:
   ```
   /path/to/nct2013/
   ├── bmode/              # B-mode images (HDF5 or individual files)
   ├── prostate_masks/     # Prostate segmentation masks
   ├── needle_masks/       # Needle trajectory masks
   └── metadata.csv        # Clinical and pathology data
   ```

### Alternative Dataset: OPTIMUM

The codebase also supports the OPTIMUM dataset with different preprocessing. Set `data.dataset: optimum` in config.

---

## 🛠 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 16GB+ GPU memory (RTX 3090, A100, etc.)

### Step-by-Step Installation

#### Option A (Recommended): Conda environment

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/prostnfound.git
cd prostnfound

# 2. Create conda environment (name is "prostnfound" by default)
conda env create -f environment.yml
conda activate prostnfound  # or your preferred env name
```

#### Option B: Pure pip / virtualenv

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/prostnfound.git
cd prostnfound

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install all Python dependencies
pip install -r requirements.txt

# 4. Install local packages in editable mode
pip install -e medAI
pip install -e external_libs

# 5. (Optional) If you prefer manual setup, you can still follow:
#    - Install PyTorch for your CUDA version
#    - pip install -e medAI
#    - pip install -e external_libs
#    - pip install -r external_libs/external_libs/requirements.txt

# 6. Download MedSAM checkpoint
mkdir -p checkpoints
wget -O checkpoints/medsam_vit_b_cpu.pth "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
# Note: You may need to obtain the actual MedSAM weights from the original authors
```

### Verify Installation

```python
import torch
from medAI.modeling.prostnfound_rl import ProstNFoundRL
from medAI.modeling.registry import create_model

# Should print available models including 'prostnfound_rl_adapter_medsam_legacy'
from medAI.modeling import list_models
print(list_models())
```

---

## 🚀 Quick Start

### Training with RL (Recommended Configuration)

```bash
cd prostnfound

# Train on fold 0 with default settings
python train_rl.py -c cfg/train/pnf_plus_rl_kfold.yaml
```

### Testing a Trained Model

```bash
# Evaluate on test set
python test_rl.py -c cfg/test_rl.yaml model_checkpoint=checkpoints_rl/best_rl.pth
```

### Visualizing Attention Points

```python
import torch
from medAI.modeling.registry import create_model

# Load trained model
model = create_model(
    'prostnfound_rl_adapter_medsam_legacy',
    prompts=['psa', 'age'],
    num_attention_points=3,
    policy_type='categorical'
)
checkpoint = torch.load('checkpoints_rl/best_rl.pth')
model.load_state_dict(checkpoint['model'])
model.eval()

# Forward pass with RL info
with torch.no_grad():
    outputs = model(
        image, rf_image=None,
        prostate_mask=prostate_mask,
        needle_mask=needle_mask,
        deterministic=True,  # Use deterministic policy for inference
        return_rl_info=True,
        psa=psa, age=age
    )

# Get attention points
attention_coords = outputs['rl_attention_coords']  # (B, k, 2)
attention_map = outputs['rl_attention_map']  # (B, 1, H, W)
cancer_heatmap = outputs['mask_logits'].sigmoid()  # (B, 1, H, W)
```

---

## 🎓 Training

### Training Modes

#### 1. Full RL Training (K-Fold Cross-Validation)

```bash
# Train all 5 folds
for fold in {0..4}; do
    python train_rl.py -c cfg/train/pnf_plus_rl_kfold.yaml data.fold=$fold
done
```

#### 2. Fine-tuning from ProstNFound+ Checkpoint

```bash
# Start from pretrained baseline
python train_rl.py -c cfg/train/pnf_plus_rl_kfold.yaml \
    model_checkpoint=checkpoints/best_baseline.pth
```

#### 3. Training Only the RL Policy (Freeze Base Model)

```bash
# Freeze encoder and decoder, train only policy
python train_rl.py -c cfg/train/pnf_plus_rl_kfold.yaml \
    model_kw.freeze_prostnfound=true
```

#### 4. Baseline Training (Without RL)

```bash
# Standard supervised training
python train.py -c cfg/train/pnf_plus_kfold.yaml
```

### Key Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rl_num_samples_per_image` | 4-8 | Rollouts per image for within-image comparison |
| `rl_num_update_epochs` | 4 | GRPO updates per batch |
| `rl_clip_eps` | 0.2 | PPO clipping epsilon |
| `rl_entropy_coef` | 0.01 | Exploration bonus |
| `rl_kl_coef` | 0.01 | KL penalty coefficient |
| `rl_cspca_bonus` | 1.5-2.0 | Reward multiplier for csPCa |
| `num_attention_points` | 3-5 | Number of attention points |
| `policy_hidden_dim` | 512 | Policy network hidden dimension |

### Training Tips

1. **Start with frozen base model** to learn a good policy first
2. **Use more samples per image** (8+) for better within-image comparison
3. **Monitor entropy** - too low means policy collapsed, too high means random
4. **Check within-image reward std** - should be positive (different configs give different rewards)

---

## 📈 Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| `core_auc` | Core-level AUROC for cancer detection |
| `core_auc_high_involvement` | AUROC for high-involvement cores (>40%) |
| `sensitivity_at_spec_X` | Sensitivity at X% specificity |
| `train_rl/reward_mean` | Average reward during training |
| `train_rl/within_image_reward_std` | Diversity of attention configurations |

### Running Evaluation

```bash
# Evaluate on validation set
python test_rl.py -c cfg/test_rl.yaml \
    model_checkpoint=checkpoints_rl/best_rl.pth \
    data.fold=0

# Generate heatmaps for visualization
python test_rl.py -c cfg/test_rl.yaml \
    model_checkpoint=checkpoints_rl/best_rl.pth \
    evaluator.log_images=true
```

### Visualizing Results

See notebooks in `prostnfound/notebooks/`:
- `results_analysis.ipynb` - Analyze metrics and generate plots
- `dev_heatmaps.ipynb` - Visualize cancer heatmaps and attention points

---

## ⚙️ Configuration

### Full Configuration Reference

```yaml
# Model Configuration
model: prostnfound_rl_adapter_medsam_legacy
model_kw:
  prompts: [psa, age]           # Clinical prompts
  use_class_decoder: true       # Enable classification head
  num_attention_points: 3       # Number of RL attention points
  policy_type: categorical      # 'categorical', 'gaussian', or 'patch'
  policy_hidden_dim: 512        # Policy network capacity
  use_clinical_in_policy: true  # Use clinical features in policy
  freeze_prostnfound: false     # Freeze base model weights
  temperature: 1.0              # Sampling temperature
  use_prostate_mask_constraint: true  # Constrain attention to prostate

# RL Training Configuration
use_rl: true                    # Enable RL training
rl_mode: grpo                   # Algorithm (pure GRPO)
rl_reward_mode: combined_v2     # Reward function
rl_cspca_bonus: 2.0             # Bonus for csPCa cases
rl_num_samples_per_image: 4     # Rollouts per image
rl_num_update_epochs: 4         # GRPO updates per batch
rl_clip_eps: 0.2                # PPO clipping
rl_entropy_coef: 0.01           # Exploration bonus
rl_kl_coef: 0.01                # KL penalty
rl_max_grad_norm: 0.5           # Gradient clipping
rl_loss_weight: 1.0             # RL vs supervised loss weight

# Data Configuration
data:
  dataset: default              # 'default' (NCT2013) or 'optimum'
  fold: 0                       # K-fold index
  n_folds: 5                    # Total folds
  batch_size: 8                 # Batch size
  image_size: 256               # Input image size
  augmentations: translate      # Data augmentation

# Training Configuration
epochs: 16
lr: 1e-4
encoder_lr: 1e-5
device: cuda
use_amp: true                   # Mixed precision training
```

---

## 📊 Results

### Expected Performance

| Model | Core AUROC | High-Involvement AUROC | Notes |
|-------|------------|------------------------|-------|
| ProstNFound+ (baseline) | ~0.78 | ~0.85 | Supervised only |
| ProstNFound-RL (ours) | ~0.80 | ~0.88 | With RL attention |

### What the RL Agent Learns

1. **Attention focuses on suspicious regions** - not random
2. **Different images get different attention** - context-dependent
3. **Clinical features influence attention** - PSA/age matter
4. **Attention aligns with pathology** - interpretable decisions

---

## 🐛 Troubleshooting

### Common Issues

#### NaN Losses or Rewards
```
Solution:
- Reduce learning rate (lr: 1e-5)
- Reduce entropy coefficient
- Check data loading (ensure masks are valid)
- Increase numerical stability (eps values)
```

#### Policy Not Learning (Constant Attention)
```
Solution:
- Increase entropy coefficient (0.05+)
- Reduce policy hidden dimension
- Try different policy type (gaussian)
- Verify rewards have variance
```

#### Training Unstable
```
Solution:
- Freeze ProstNFound weights initially
- Reduce gradient clipping threshold
- Increase batch size
- Reduce RL update epochs
```

#### Out of Memory
```
Solution:
- Reduce batch size
- Reduce num_samples_per_image
- Reduce policy_hidden_dim
- Disable torch.compile
```

---

## 📚 Project Structure

```
prostnfound/
├── medAI/                          # Core ML library
│   └── medAI/
│       ├── modeling/
│       │   ├── prostnfound.py      # Base ProstNFound model
│       │   ├── prostnfound_rl.py   # RL-extended model
│       │   ├── rl_attention_policy.py  # Policy networks
│       │   ├── grpo.py             # GRPO algorithm
│       │   └── registry.py         # Model registry
│       ├── datasets/
│       │   └── nct2013/            # Dataset utilities
│       └── transforms/             # Data transformations
│
├── prostnfound/                    # Training scripts
│   ├── train_rl.py                 # RL training script
│   ├── train.py                    # Baseline training
│   ├── test_rl.py                  # Evaluation script
│   ├── cfg/
│   │   ├── train/
│   │   │   └── pnf_plus_rl_kfold.yaml  # Training config
│   │   └── test_rl.yaml            # Test config
│   ├── src/
│   │   ├── rl_loss.py              # Reward computation
│   │   ├── loaders.py              # Data loaders
│   │   └── evaluator.py            # Metrics
│   └── notebooks/                  # Analysis notebooks
│
├── external_libs/                  # External dependencies
├── checkpoints/                    # Model checkpoints
└── README.md                       # This file
```

---

## 📖 Citation

If you use this code, please cite:

```bibtex

```

### Related Work

- **MedSAM**: Segment Anything in Medical Images
- **GRPO**: Group Relative Policy Optimization (DeepSeekMath)
- **Seg-R1**: Reinforcement Learning for Segmentation

---

## 📬 Contact

For questions, issues, or collaboration:

**Mohammad Mahdi Abootorabi**  
- Email: mahdi.abootorabi@ece.ubc.ca
- University of British Columbia, ECE Department

---

## 🔮 Future Work

- [ ] Multi-scale attention (attention at different resolutions)
- [ ] Recurrent policy for video/temporal data
- [ ] Uncertainty-aware attention selection
- [ ] Meta-learning for few-shot adaptation
- [ ] Integration with other RL algorithms (SAC, TD3)
- [ ] Multi-task RL (detection + grading)
