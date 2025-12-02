# GRPO vs Supervised Training: Gradient Flow Analysis

## Summary: Is This True RL?

**YES, this is true Reinforcement Learning** because:
1. **Policy makes stochastic decisions** (sampling attention points)
2. **Rewards are computed from outcomes** (not directly differentiable)
3. **Policy gradients** optimize the policy using advantage-weighted updates
4. **Reward signal is detached** - gradients don't flow through rewards directly

---

## Key Differences in Gradient Flow

### 1. Supervised Training (Standard Path)

**Gradient Path:**
```
Input Image
    ↓
Encoder (MedSAM)
    ↓
Decoder (with fixed/clinical prompts)
    ↓
cancer_logits (B, 1, H, W)
    ↓
Supervised Loss (BCE/MIL)
    ↓
Backward: Loss → logits → decoder → encoder
```

**What gets optimized:**
- **All parameters** (encoder + decoder) learn to map inputs → correct predictions
- Direct gradient signal: "make predictions closer to ground truth"
- Single deterministic forward pass per image

**Code location:** `train_rl.py:run_train_epoch()` (lines 355-409)

---

### 2. GRPO Training (RL Path)

**Two-Part Optimization:**

#### Part A: Supervised Loss (Same as above)
```
Input Image
    ↓
Encoder
    ↓
Policy Head → rl_log_probs (attention point selection)
    ↓
Attention Points → Decoder (as prompts)
    ↓
cancer_logits
    ↓
Supervised Loss
    ↓
Backward: Loss → logits → decoder → encoder
```

#### Part B: GRPO Policy Loss (RL-specific)
```
Input Image (replicated N times)
    ↓
Encoder
    ↓
Policy Head → rl_log_probs (current policy)
    ↓
Attention Points (sampled) → Decoder
    ↓
cancer_logits → Rewards (DETACHED - no gradient!)
    ↓
Advantages = (rewards - group_mean) / group_std (DETACHED)
    ↓
GRPO Loss = -min(ratio * advantage, clipped_ratio * advantage)
    ↓
Backward: GRPO Loss → ratio → log_probs → policy_head → encoder
```

**Key Point:** Rewards and advantages are **detached** (lines 448, 483, 156 in grpo.py), so gradients **do NOT flow through the reward computation**. Instead, gradients flow through the **policy gradient method** (PPO-style clipping).

**Code location:** `train_rl.py:run_rl_train_epoch_batched()` (lines 412-570)

---

## Detailed Gradient Flow Comparison

### Supervised Loss Gradient Path

```python
# train_rl.py:477
supervised_loss = criterion(current_outputs)  # BCE/MIL loss

# Gradient flow:
# supervised_loss.backward() →
#   cancer_logits (requires_grad=True) →
#   decoder (requires_grad=True) →
#   encoder (requires_grad=True if not frozen)
```

**What this optimizes:**
- Decoder learns: "given these features, predict cancer correctly"
- Encoder learns: "extract features that help decoder predict correctly"
- **Direct supervision**: loss directly measures prediction quality

---

### GRPO Loss Gradient Path

```python
# train_rl.py:480-485
rl_loss, rl_info = grpo.compute_loss(
    current_log_probs,      # (B*N, k) - requires_grad=True
    old_log_probs.detach(), # (B*N, k) - DETACHED
    all_rewards.detach(),   # (B*N,)   - DETACHED
    num_samples_per_image=num_samples_per_image,
)

# Inside grpo.compute_loss() (grpo.py:209-213):
advantages = self.compute_advantages(rewards, ...)  # rewards are detached
total_loss, info = self.compute_policy_loss(log_probs, old_log_probs, advantages)

# Inside grpo.compute_policy_loss() (grpo.py:153-160):
ratio = torch.exp(log_probs_sum - old_log_probs_sum)  # ratio requires_grad
advantages = advantages.detach()  # DETACHED again (line 156)
policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

# Gradient flow:
# rl_loss.backward() →
#   policy_loss (requires_grad=True) →
#   ratio (requires_grad=True) →
#   log_probs (requires_grad=True) →
#   policy_head (requires_grad=True) →
#   encoder (requires_grad=True if not frozen)
```

**What this optimizes:**
- Policy head learns: "select attention points that lead to higher rewards"
- Encoder learns: "extract features that help policy select good attention points"
- **Indirect optimization**: no direct gradient from reward → policy, uses policy gradient theorem

---

## Why This Is True RL (Not Just Supervised)

### 1. **Non-Differentiable Reward Signal**
```python
# Rewards computed from model outputs but DETACHED
all_rewards = reward_computer(batched_outputs, batched_data)  # No gradient
all_rewards.detach()  # Explicitly detached (line 483)
```

Rewards are based on:
- Classification accuracy
- ROI involvement prediction
- But gradients **don't flow through** reward computation

### 2. **Policy Gradient Method**
```python
# PPO-style clipped objective (grpo.py:157-160)
ratio = exp(log_probs - old_log_probs)  # Importance sampling ratio
policy_loss = -min(ratio * advantage, clip(ratio) * advantage)
```

This is the **REINFORCE/PPO policy gradient**:
- Optimizes policy to increase probability of high-reward actions
- Uses importance sampling to correct for off-policy updates
- Clips to prevent large policy updates

### 3. **Stochastic Policy with Multiple Samples**
```python
# Multiple stochastic samples per image (line 442)
batched_data = replicate_batch_for_sampling(data, num_samples_per_image, ...)
batched_outputs = model(batched_data, deterministic=False)  # Sampling!
```

- Policy **samples** attention points (not deterministic)
- Multiple samples per image for within-image comparison
- GRPO normalizes advantages **within each image group**

### 4. **Advantage-Based Learning**
```python
# Within-image advantage normalization (grpo.py:104-111)
rewards_grouped = rewards.view(num_images, num_samples_per_image)
group_mean = rewards_grouped.mean(dim=1, keepdim=True)
group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-4
advantages = (rewards_grouped - group_mean) / group_std
```

- Compares samples **within the same image** (not across batch)
- Learns: "which attention points work better for THIS specific image"
- This is the key innovation of GRPO for medical imaging

---

## Combined Training: Both Losses Together

```python
# train_rl.py:488-489
total_loss = supervised_loss + rl_weight * rl_loss
total_loss.backward()  # Single backward pass
```

**Gradient contributions:**
1. **Supervised loss gradients:**
   - Flow through: decoder → encoder
   - Signal: "predict correctly"

2. **GRPO loss gradients:**
   - Flow through: policy_head → encoder
   - Signal: "select attention points that lead to better outcomes"

**Both gradients are combined** in the same backward pass, so:
- Encoder gets gradients from **both** supervised and RL objectives
- Decoder gets gradients from supervised loss
- Policy head gets gradients from GRPO loss

---

## What Each Component Learns

### Encoder (MedSAM Image Encoder)
- **Supervised signal**: Extract features useful for cancer detection
- **RL signal**: Extract features useful for attention point selection
- **Combined**: Learns representations that help both tasks

### Decoder (MedSAM Mask Decoder)
- **Supervised signal only**: Given features + prompts, predict cancer
- **No direct RL signal**: But benefits from better attention points (from policy)

### Policy Head (RL Attention Policy)
- **RL signal only**: Learn to select attention points that maximize rewards
- **No direct supervised signal**: But rewards are based on supervised task performance

---

## Why This Hybrid Approach?

1. **Supervised loss**: Ensures model still learns basic cancer detection
2. **GRPO loss**: Learns to actively identify suspicious regions (attention)
3. **Combined**: Best of both worlds - accurate predictions + intelligent attention

---

## Key Code Locations

### Supervised Loss
- **Computation**: `train_rl.py:477` - `criterion(current_outputs)`
- **Gradient flow**: Standard autograd through decoder → encoder

### GRPO Loss
- **Computation**: `train_rl.py:480-485` - `grpo.compute_loss(...)`
- **Policy loss**: `grpo.py:130-185` - `compute_policy_loss()`
- **Advantage computation**: `grpo.py:76-128` - `compute_advantages()`
- **Gradient flow**: Through policy gradient (ratio → log_probs → policy_head)

### Reward Computation
- **Location**: `src/rl_loss.py:313-360` - `RLRewardComputer.__call__()`
- **Key**: Rewards are **detached** - no gradients flow through them

---

## Conclusion

**This IS true Reinforcement Learning** because:

1. ✅ Policy makes **stochastic decisions** (attention point selection)
2. ✅ **Rewards** are computed from outcomes (classification + ROI accuracy)
3. ✅ **Policy gradients** (PPO-style) optimize the policy
4. ✅ **Reward signal is non-differentiable** (detached)
5. ✅ **Multiple samples** per image for exploration
6. ✅ **Advantage-based learning** (within-image comparison)

The key difference from pure supervised learning:
- **Supervised**: Direct gradient from loss → predictions → model
- **RL**: Indirect gradient through policy gradient theorem: loss → policy ratio → log_probs → policy

Both are combined in your code, making it a **hybrid supervised + RL approach**.

