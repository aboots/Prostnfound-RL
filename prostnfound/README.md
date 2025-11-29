# ProstNFound+

## Training 

The main entrypoint script is `train.py`. For example, running `python train.py -c cfg/train/pnf_plus_kfold.yaml` should give you about 76.5% core-level auc by the end of training (average across folds). Adding the flag `model_kw.prompts=[age,psa,psad,pos]` in the config should improve performance by +1% AUROC. You need to run the experiment 5 times using `data.fold=FOLD_NUM`. 

## ProstNFound-RL (NEW!)

**RL-Guided Attention for Improved Cancer Detection**

ProstNFound-RL extends the base model with reinforcement learning to actively identify suspicious regions. The RL policy learns to guide attention, improving detection accuracy especially for clinically significant prostate cancer (csPCa).

**Quick Start:**

```bash
# Test RL implementation
python train_rl.py -c cfg/train/pnf_plus_rl_baseline.yaml

# Full training
python train_rl.py -c cfg/train/pnf_plus_rl_kfold.yaml
```

**See [README_RL.md](README_RL.md) for detailed documentation.**

## Heatmaps 

To generate the heatmaps for nature reviews urology paper, run: 

```shell
python scripts/generate_heatmaps.py /home/pwilson/projects/aip-medilab/pwilson/medAI/data/checkpoint_store/Paul/medAI/prostnfound/prostnfound_plus_final/checkpoint.pth nature_uro_heatmaps --dataset cfg/data/optimum_UA_full.yaml --core_ids UA-105-004 UA-036-004 --patient_ids UA-105 UA-036 --style miccai --apply_prostate_mask --mode one_image_per_core
```