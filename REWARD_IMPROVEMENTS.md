# RL Reward Function Improvements

## Key Findings

### 1. RL Attention Points Flow Through BOTH Decoders ✅

**Confirmed:** The RL attention points affect **both** the heatmap decoder AND the classifier:

```python
# Line 235: RL attention embeddings added to sparse_embedding
sparse_embedding = torch.cat([sparse_embedding, attention_point_embeddings], dim=1)

# Line 302-308: Mask decoder uses sparse_embedding (includes RL points)
mask_logits, iou = mask_decoder.forward(image_feats, pe, sparse_embedding, dense_embedding)

# Line 317-323: Class decoder ALSO uses the same sparse_embedding (includes RL points)
cls_outputs = class_decoder.forward(image_feats, pe, sparse_embedding, dense_embedding)
```

**Implication:** The RL policy learns to optimize attention locations that improve **both**:
- Heatmap quality (spatial cancer detection)
- Classification accuracy (image-level prediction)

This is why the reward includes both components (70% heatmap + 30% classification).

---

## 2. Problems with Loss-Based Rewards ❌

### Current Approach (`loss_based`):
```python
reward = -BCE_loss / 2.0  # Lower loss = higher reward
```

### Problems:
1. **Doesn't optimize what we care about**: Loss doesn't directly correlate with AUC, accuracy, or clinical utility
2. **Poor scale**: Loss values vary dramatically across cases (easy vs hard cases)
3. **Weak learning signal**: For ranking tasks, loss doesn't reward relative ordering
4. **Not interpretable**: Hard to understand what "good" reward values mean

### Example Problem:
- Case A: Easy negative case, loss=0.01 → reward = -0.005
- Case B: Hard positive case, loss=2.0 → reward = -1.0
- Even if Case B ranks higher than Case A, it gets lower reward!

---

## 3. New Reward Functions ✅

### A. **Confidence-Based Reward** (RECOMMENDED)

```python
# For positive cases: reward = pred_prob (higher confidence = better)
# For negative cases: reward = 1 - pred_prob (lower confidence = better)
reward = 2.0 * confidence - 1.0  # Scale to [-1, 1]
```

**Advantages:**
- ✅ Directly optimizes confidence calibration
- ✅ Interpretable: reward = how confident we are on correct prediction
- ✅ Better learning signal for classification tasks
- ✅ Scales well across different cases

**When to use:** General purpose, good default choice

---

### B. **Ranking-Based Reward**

```python
# For positive cases: reward = fraction of negatives ranked below
# For negative cases: reward = fraction of positives ranked above
```

**Advantages:**
- ✅ Directly optimizes AUC (ranking quality)
- ✅ Better for imbalanced datasets
- ✅ Rewards relative ordering, not absolute values

**When to use:** When AUC is the primary metric, or for ranking tasks

**Limitation:** Requires batch-level computation (less sample-efficient)

---

### C. **F1-Based Reward**

```python
# Rewards based on contribution to F1-score (precision + recall)
```

**Advantages:**
- ✅ Balances precision and recall
- ✅ Good for imbalanced datasets
- ✅ Optimizes both false positives and false negatives

**When to use:** When you care about both precision and recall equally

**Limitation:** Requires batch-level computation

---

### D. **Accuracy-Based Reward** (Simple)

```python
reward = +1 if correct, -1 if incorrect
```

**Advantages:**
- ✅ Simple and interpretable
- ✅ Direct optimization target

**Limitation:** Doesn't reward confidence calibration

---

## 4. Recommendations

### For Your Use Case:

1. **Start with `confidence_based`** (now default in config):
   - Best balance of interpretability and performance
   - Works well with within-image comparison
   - Good learning signal

2. **Try `ranking_based`** if AUC is your primary metric:
   - Better for optimizing ranking quality
   - May need larger batches for stable estimates

3. **Avoid `loss_based`**:
   - Poor scale properties
   - Doesn't optimize what you care about

---

## 5. Implementation Details

### Reward Composition (Still Applies):

```python
# Heatmap reward (70%)
heatmap_reward = compute_confidence_based_reward(cancer_logits, data)

# Classification reward (30%)
cls_reward = compute_classification_reward(cls_outputs, data)

# Combined
total_reward = 0.7 * heatmap_reward + 0.3 * cls_reward

# csPCa bonus
if grade_group > 2:
    total_reward *= 2.0
```

**Why both?** Because RL attention points affect BOTH decoders, so we need to reward improvements in both.

---

## 6. Usage

### In Config File:

```yaml
rl_reward_mode: confidence_based  # Recommended
# or
rl_reward_mode: ranking_based     # For AUC optimization
# or
rl_reward_mode: f1_based          # For balanced precision/recall
```

### Expected Behavior:

- **Confidence-based**: Rewards will be in range [-1, 1], higher for correct high-confidence predictions
- **Ranking-based**: Rewards depend on batch composition, better separation = higher rewards
- **F1-based**: Rewards scale with overall F1 score of the batch

---

## 7. Testing Recommendations

1. **Compare reward modes** on validation set:
   - Monitor: AUC, accuracy, F1-score
   - See which reward mode gives best performance

2. **Monitor reward statistics**:
   - `train_rl/reward_mean`: Should be positive for good policies
   - `train_rl/reward_std`: Lower is better (more stable)
   - `train_rl/within_image_reward_std`: Measures diversity of attention strategies

3. **Check learning curves**:
   - Confidence-based should show steady improvement
   - Ranking-based may be noisier but better for AUC

---

## Summary

✅ **RL attention points affect BOTH decoders** - confirmed  
✅ **New reward functions** - confidence_based, ranking_based, f1_based  
✅ **Better than loss-based** - optimize what we actually care about  
✅ **Config updated** - confidence_based is now default  

**Next Steps:**
1. Try `confidence_based` reward mode (already set in config)
2. Monitor training metrics
3. Experiment with `ranking_based` if AUC is primary concern

