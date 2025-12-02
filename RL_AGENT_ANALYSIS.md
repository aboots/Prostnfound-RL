# RL Agent Architecture Analysis: Two-Head Policy Network

## Overview

Your RL agent uses a **two-head architecture** consisting of:
1. **Policy Head**: Generates attention coordinates (where to look)
2. **Value Head**: Estimates expected future reward (baseline for advantage computation)

Both heads share the same feature extractor but have separate output layers.

---

## Architecture Details

### Two-Head Policy Network (`RLAttentionPolicy`)

Located in: `medAI/medAI/modeling/rl_attention_policy.py`

```
Image Features (B, 256, H, W)
    ↓
Feature Processor (Conv layers)
    ↓
    ├─→ Attention Map Head → Attention Coordinates (B, k, 2)
    │                         Log Probabilities (B, k)
    │
    └─→ Value Head → State Value (B, 1)
```

**Key Components:**

1. **Feature Processor** (shared backbone):
   - Conv2d layers: `256 → 512 → 512`
   - BatchNorm + ReLU
   - Processes encoder features from MedSAM

2. **Policy Head** (`attention_map_head`):
   - Outputs attention heatmap: `(B, 1, H, W)`
   - Samples k points from categorical distribution
   - Returns: coordinates `(B, k, 2)` and log_probs `(B, k)`
   - **Hard masking**: Sets logits outside prostate to `-inf` (guarantees valid sampling)

3. **Value Head** (`value_head`):
   - Global average pooling → Linear layers
   - Outputs: `value (B, 1)` - estimates expected reward

---

## How Gradients Flow

### Forward Pass Flow

```
1. Image → MedSAM Encoder → Features (B, 256, H, W)
2. Features → Policy Network:
   - Policy Head → Attention Coords + Log Probs
   - Value Head → Value Estimate
3. Attention Coords → SAM Prompt Encoder → Point Embeddings
4. Point Embeddings + Features → SAM Decoder → Cancer Heatmap
5. Heatmap → Reward Computation
```

### Backward Pass (Gradient Flow)

**During RL Training (`run_rl_train_epoch`):**

1. **Rollout Collection** (with `torch.no_grad()`):
   - Sample multiple rollouts per image (default: 4)
   - Store: `old_log_probs`, `old_values`, `rewards`
   - **No gradients** computed here

2. **GRPO Update Loop** (4 epochs per batch):
   - Re-run model to get `current_log_probs` and `current_values`
   - **Gradients flow through:**
     ```
     Total Loss = Supervised Loss + RL Loss
     
     RL Loss = Policy Loss + Value Loss
     
     Policy Loss:
       - Clipped PPO objective: min(ratio * advantage, clip(ratio) * advantage)
       - Gradient flows: advantage → ratio → log_probs → policy_head → feature_processor
     
     Value Loss:
       - MSE: (value - return)²
       - Gradient flows: value_loss → value_head → feature_processor
     ```

3. **Gradient Paths:**

   **Policy Gradients:**
   ```
   Advantage (detached)
     ↓
   Policy Loss (PPO clipped objective)
     ↓
   log_probs (from policy head)
     ↓
   attention_map_head (policy head)
     ↓
   feature_processor (shared)
     ↓
   Back to encoder (if not frozen)
   ```

   **Value Gradients:**
   ```
   Value Loss (MSE)
     ↓
   value (from value head)
     ↓
   value_head
     ↓
   feature_processor (shared)
     ↓
   Back to encoder (if not frozen)
   ```

4. **Gradient Clipping:**
   - Applied after backward pass: `clip_grad_norm_(max_norm=0.5)`
   - Prevents exploding gradients

---

## How Rewards Are Computed

Located in: `prostnfound/src/rl_loss.py` (`RLRewardComputer`)

### Reward Components

1. **Heatmap Reward** (70% weight):
   - **Loss-based**: `reward = -BCE_loss / 2.0`
   - Computed on predictions within prostate+needle mask
   - Lower loss → Higher reward

2. **Classification Reward** (30% weight):
   - Log probability of correct class: `log(p_correct) / 2.0`
   - Ensures RL policy improves both heatmap AND classifier

3. **csPCa Bonus**:
   - Multiplies reward by `cspca_bonus` (default: 2.0) for grade_group > 2
   - Prioritizes clinically significant cases

4. **Prostate Boundary Penalty** (optional, currently disabled):
   - Soft penalty for coordinates outside prostate
   - **Note**: Hard masking makes this unnecessary

### Reward Computation Flow

```python
# For each rollout:
1. Model outputs: cancer_logits, cls_outputs, rl_attention_coords
2. Compute heatmap_reward (from cancer_logits)
3. Compute classification_reward (from cls_outputs)
4. Combine: reward = 0.7 * heatmap_reward + 0.3 * classification_reward
5. Apply csPCa bonus if applicable
6. Subtract boundary penalty (if enabled)
```

### Within-Image Comparison

- **Multiple samples per image** (default: 4)
- Rewards computed independently for each sample
- Advantages normalized **within each image group** (not across batch)
- This allows fair comparison: "Which attention strategy works better for THIS image?"

---

## Drawbacks and Limitations

### 1. **Inefficient Rollout Collection**

**Problem:**
- In `run_rl_train_epoch`, rollouts are collected with `torch.no_grad()`
- Then model is re-run 4 times (for 4 update epochs) **with gradients**
- This means: **5 forward passes per batch** (1 rollout + 4 updates)

**Impact:**
- 5x computational cost compared to single forward pass
- Slow training, especially with large models

**Code Location:** `train_rl.py:400-458`

```python
# Step 1: Collect rollouts (no grad)
with torch.no_grad():
    for sample_idx in range(num_samples_per_image):
        rollout_data = model(data, deterministic=False)  # Forward pass 1-4

# Step 2: Update epochs (with grad)
for rl_epoch in range(num_rl_updates):  # 4 epochs
    for sample_idx in range(num_samples_per_image):
        current_data = model(data, deterministic=False)  # Forward pass 5-8
```

### 2. **Value Function Learning Signal**

**Problem:**
- Value head learns from `returns = rewards` (single-step, no discounting)
- With `gamma=1.0`, value estimates are just reward predictions
- No temporal structure or future reward estimation

**Impact:**
- Value function may not provide strong baseline
- Advantage estimates may have high variance
- Limited benefit from value head (could just use reward mean as baseline)

**Code Location:** `grpo.py:96`

```python
# Single-step returns (no discounting)
returns = rewards  # gamma=1.0 means no future rewards
advantages = returns - values
```

### 3. **Policy Gradient Variance**

**Problem:**
- Sampling k points sequentially (without replacement approximation)
- Each point sampled independently, but probabilities modified after each sample
- This creates correlation between samples

**Impact:**
- Higher variance in policy gradients
- May need more samples to get stable estimates

**Code Location:** `rl_attention_policy.py:165-178`

```python
# Sequential sampling with probability modification
for i in range(self.num_attention_points):
    dist = torch.distributions.Categorical(probs=attention_probs)
    sampled_idx = dist.sample()
    # Zero out probability (approximation of without-replacement)
    attention_probs = attention_probs.scatter(1, sampled_idx.unsqueeze(1), 0.0)
```

### 4. **Reward Scale Mismatch**

**Problem:**
- Heatmap reward: roughly `[-1, 0]` (negative BCE loss)
- Classification reward: roughly `[-2, 0]` (log probability)
- Combined with weights: `0.7 * [-1,0] + 0.3 * [-2,0] = [-1.3, 0]`
- But csPCa bonus multiplies by 2.0, so range becomes `[-2.6, 0]`

**Impact:**
- Reward scale varies significantly between cases
- May cause instability in advantage normalization
- Hard to tune hyperparameters

**Code Location:** `rl_loss.py:243, 352`

### 5. **Limited Exploration**

**Problem:**
- Entropy coefficient is low (`entropy_coef=0.01`)
- With hard masking (prostate constraint), exploration is further limited
- Policy may converge to local optima

**Impact:**
- May miss better attention strategies
- Less diverse attention patterns

**Code Location:** `grpo.py:173`

```python
total_loss = policy_loss - self.entropy_coef * entropy  # entropy_coef=0.01
```

### 6. **Supervised Loss on Last Sample Only**

**Problem:**
- Supervised loss computed only on last sample's output
- Other samples' outputs are discarded for supervised learning
- Wastes information from other rollouts

**Code Location:** `train_rl.py:465`

```python
# Supervised loss (use last sample's data for supervised loss)
supervised_loss = criterion(current_data)  # Only uses last sample!
```

### 7. **No Experience Replay**

**Problem:**
- On-policy algorithm (GRPO/PPO)
- Old rollouts discarded after each update
- Cannot reuse past experiences

**Impact:**
- Less sample efficient
- May need more data to learn effectively

### 8. **Fixed Number of Attention Points**

**Problem:**
- Always samples exactly `k=3` points
- Cannot adapt to image complexity
- Some images may need more/less attention points

**Impact:**
- Suboptimal for diverse cases
- May over-attend simple cases or under-attend complex ones

---

## Potential Improvements

### 1. **Optimize Rollout Collection**

**Solution:** Store intermediate features to avoid recomputation

```python
# Collect rollouts with gradients enabled (store features)
# Reuse features for update epochs
with torch.set_grad_enabled(True):
    # Store features after first forward pass
    features = model.get_features(data)
    
    # Use stored features for policy/value heads only
    for epoch in range(num_rl_updates):
        log_probs, values = policy_network(features)  # No encoder forward pass
```

**Benefit:** Reduce from 5x to ~1.5x computational cost

### 2. **Improve Value Function**

**Solution:** Use multi-step returns or better baseline

```python
# Option A: Use reward statistics as baseline
baseline = rewards.mean()  # Simple but effective

# Option B: Learn value function with better architecture
# Add attention pooling or spatial aggregation
```

**Benefit:** Better advantage estimates, lower variance

### 3. **Proper Without-Replacement Sampling**

**Solution:** Use Gumbel-Top-K or proper categorical sampling

```python
# Use Gumbel-Top-K for proper without-replacement
def sample_top_k_without_replacement(logits, k):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    scores = logits + gumbel_noise
    top_k_indices = torch.topk(scores, k=k, dim=1).indices
    return top_k_indices
```

**Benefit:** Lower variance, more stable gradients

### 4. **Normalize Reward Scales**

**Solution:** Standardize reward components before combining

```python
# Normalize each component to [0, 1] or [-1, 1]
heatmap_reward_norm = (heatmap_reward - heatmap_reward.min()) / (heatmap_reward.max() - heatmap_reward.min() + 1e-8)
cls_reward_norm = (cls_reward - cls_reward.min()) / (cls_reward.max() - cls_reward.min() + 1e-8)
```

**Benefit:** More stable training, easier hyperparameter tuning

### 5. **Increase Exploration**

**Solution:** Adaptive entropy coefficient or temperature

```python
# Schedule entropy coefficient
entropy_coef = max(0.01, 0.1 * (1 - epoch / total_epochs))  # Decay over time

# Or use temperature scheduling
temperature = max(0.5, 2.0 * (1 - epoch / total_epochs))
```

**Benefit:** Better exploration early, exploitation later

### 6. **Use All Samples for Supervised Loss**

**Solution:** Average supervised loss across all rollouts

```python
# Compute supervised loss for all samples
supervised_losses = []
for sample_data in all_rollout_data:
    loss = criterion(sample_data)
    supervised_losses.append(loss)
supervised_loss = torch.stack(supervised_losses).mean()
```

**Benefit:** Better supervised signal, more efficient learning

### 7. **Adaptive Attention Points**

**Solution:** Learn number of attention points per image

```python
# Add a "stop" action or learn k dynamically
# Or use attention weights to determine importance
attention_weights = F.softmax(attention_map, dim=-1)
# Use only points above threshold
```

**Benefit:** More flexible, better for diverse cases

### 8. **Add Reward Shaping**

**Solution:** Intermediate rewards for attention quality

```python
# Reward for attention diversity
diversity_reward = -entropy(attention_distribution)  # Encourage diverse attention

# Reward for attention in high-activation regions
activation_reward = attention_map[high_activation_regions].mean()
```

**Benefit:** Better learning signal, faster convergence

---

## Summary

Your RL agent uses a **two-head architecture** (policy + value) with **GRPO** for training. The main strengths are:

✅ **Within-image comparison** for fair reward evaluation  
✅ **Hard masking** to guarantee valid attention locations  
✅ **Combined rewards** from heatmap and classification  

Main limitations:

❌ **Inefficient rollout collection** (5x forward passes)  
❌ **Weak value function** (single-step, no discounting)  
❌ **High variance** from sequential sampling  
❌ **Reward scale mismatch** between components  
❌ **Limited exploration** (low entropy)  
❌ **Wasted supervised signal** (only last sample)  

The most impactful improvements would be:
1. **Optimize rollout collection** (biggest speedup)
2. **Improve value function** (better baselines)
3. **Normalize reward scales** (more stable training)
4. **Use all samples for supervised loss** (better learning)

