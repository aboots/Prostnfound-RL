# ProstNFound-RL: Reinforcement Learning-Guided Attention for Prostate Cancer Detection

ProstNFound-RL extends the **ProstNFound+** ultrasound-based prostate cancer detection model by adding an **RL-guided attention mechanism**. Instead of passively processing the whole image, an RL agent learns *where* the model should look—focusing on suspicious regions that improve both heatmap generation and cancer classification.

This idea is inspired by how radiologists actively search for abnormalities rather than scanning every pixel equally.
<img width="1113" height="389" alt="image" src="https://github.com/user-attachments/assets/db841453-8615-45f7-8b1c-fc810c44bf06" />

---

## 🚀 What This Project Does

* Uses the **MedSAM encoder** from ProstNFound+ to extract image features.
* Trains an **RL policy network** to identify a small set of informative regions in the ultrasound image.
* Passes these regions as **attention prompts** to the decoder, alongside clinical metadata (Age, PSA).
* Optimizes the policy using **GRPO** or **PPO**—both tested and compared in this project.
* Improves sensitivity, heatmap quality, and interpretability by explicitly guiding the model’s focus.

---

## 🧠 Why Reinforcement Learning?

Ultrasound labels are weak: we only know the cancer involvement of the biopsy core, not pixel-level annotations. RL lets the model *discover* helpful attention strategies by trial and error.

* Rewards are based on **outcomes**: Did selecting these regions improve classification or ROI estimation?
* GRPO and PPO allow us to compare multiple attention configurations per image.
* The agent gradually learns to highlight meaningful structures rather than arbitrary regions.

This approach is particularly useful when supervision is limited, noisy, or indirect—exactly the case in prostate ultrasound.

---

## 🏗 High-Level Architecture

1. **MedSAM encoder** → produces feature map
2. **RL policy network** → selects k suspicious locations
3. **Prompt encoder** → turns coordinates into spatial prompts
4. **Mask + Class decoders** → generate heatmap + csPCa likelihood
5. **Reward module** → evaluates how helpful the chosen regions were
6. **GRPO/PPO** → updates the policy

This creates a compact, interpretable attention mechanism between encoder and decoder.

---

## 📊 Key Findings (From Experiments)

Our experiments (on NCT2013 data) show that ProstNFound-RL consistently improves or matches the baseline ProstNFound+ model across most metrics:

* Higher **Core AUROC**
* Higher **csPCa Heatmap AUROC**
* Better **Sensitivity at fixed specificity levels**
* Improved spatial interpretability through meaningful attention points

Among all tested variants:

* **PPO + No Mask + Loss-based reward** gave the strongest results.
* Surprisingly, allowing the agent to explore *outside* the prostate sometimes helped—likely due to contextual cues in surrounding tissue.
* The combined reward (classification + ROI) improved interpretability and heatmap quality.

Full details and metrics appear in the report.

---

## 🧪 Challenges

Training this hybrid RL + multi-task system is computationally heavy and sometimes unstable:

* Occasional **loss divergence** after several epochs
* Attention drifting outside the prostate
* Decoder initially ignoring RL prompts
* Large hyperparameter search space

We implemented solutions such as reward shaping, prostate-boundary constraints, and visualization tools to track attention behavior.

---

## 🔧 Getting Started

### Installation

#### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 16GB+ GPU memory (RTX 3090, A100, etc.)

```bash
git clone https://github.com/aboots/Prostnfound-RL
cd Prostnfound-RL
conda env create -f environment.yml
conda activate prostnfound
```
- Install all Python dependencies
```pip install -r requirements.txt```

- Install local packages in editable mode
```bash
pip install -e medAI
pip install -e external_libs
```

- Download MedSAM checkpoint
```mkdir -p checkpoints
wget -O checkpoints/medsam_vit_b_cpu.pth "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
# Note: You may need to obtain the actual MedSAM weights from the original authors
```

### Train with RL

```bash
python train_rl.py -c cfg/train/pnf_plus_rl_kfold.yaml
```

### Evaluate

```bash
python test_rl.py -c cfg/test_rl.yaml model_checkpoint=path/to/checkpoint.pth
```

---

## 🗂 Project Structure

```
medAI/
   modeling/
      prostnfound_rl.py       # RL-extended model
      rl_attention_policy.py  # Policy network
      grpo.py                 # GRPO implementation
prostnfound/
   train_rl.py                # Training script
   test_rl.py                 # Evaluation
   cfg/                       # YAML configs
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

---

## 📄 Citation

If you use this work, please cite the project report (placeholder):

```bibtex
coming soon !
```

---

## 📬 Contact

**Mohammad Mahdi Abootorabi**
[mahdi.abootorabi@ece.ubc.ca](mailto:mahdi.abootorabi@ece.ubc.ca)
