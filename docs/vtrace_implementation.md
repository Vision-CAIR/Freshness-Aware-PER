# V-trace Implementation in ROLL Framework

## Overview

V-trace is an off-policy actor-critic algorithm that uses truncated importance sampling to correct for the difference between the behavior policy (that generated the data) and the target policy (being learned). It was introduced in the IMPALA paper (Espeholt et al., 2018) and is particularly well-suited for distributed reinforcement learning with significant policy lag.

## Mathematical Foundation

### Core V-trace Formula

The V-trace target value is computed as:

```
v_s = V(x_s) + Σ_{t=s}^{s+n-1} γ^{t-s} (Π_{i=s}^{t-1} c_i) δ_t V
```

Where:
- `δ_t V = ρ_t (r_t + γV(x_{t+1}) - V(x_t))` - Importance-weighted TD error
- `ρ_t = min(ρ̄, π(a_t|x_t)/μ(a_t|x_t))` - Truncated importance sampling ratio
- `c_t = min(c̄, π(a_t|x_t)/μ(a_t|x_t))` - Truncated trace coefficient
- `π` - Target policy (current learner policy)
- `μ` - Behavior policy (policy that generated the data)
- `ρ̄, c̄` - Truncation thresholds (typically 1.0)

### Key Properties

1. **Truncation for Stability**: The truncation of ρ and c prevents variance explosion
2. **Off-Policy Correction**: Corrects for policy lag between actors and learner
3. **Convergence Guarantee**: Converges to the optimal policy fixed point

## Implementation Details

### 1. Core Function

Located in `roll/utils/functionals.py`:

```python
@torch.no_grad()
def compute_vtrace_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    old_log_probs: torch.Tensor,      # From behavior policy μ
    current_log_probs: torch.Tensor,   # From target policy π
    response_mask: torch.Tensor,
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute V-trace advantages and returns."""
```

### 2. Integration with compute_advantage

The V-trace estimator is integrated as an option in the `compute_advantage` function:

```python
def compute_advantage(data, gamma, lambd, adv_estimator, ...):
    if adv_estimator == "vtrace":
        # V-trace specific computation
        advantages, returns = compute_vtrace_advantage_return(...)
    elif adv_estimator == "gae":
        # GAE computation
    # ... other estimators
```

### 3. Configuration

V-trace parameters are configured through `VTraceConfig` in `agentic_config.py`:

```python
@dataclass
class VTraceConfig:
    rho_bar: float = 1.0  # Truncation for importance sampling ratio
    c_bar: float = 1.0    # Truncation for trace coefficient
```

## Usage Guide

### Basic Configuration

```yaml
# In your YAML config file
adv_estimator: "vtrace"  # Use V-trace instead of GAE
gamma: 0.99

vtrace:
  rho_bar: 1.0  # Conservative truncation
  c_bar: 1.0
```

### With Replay Buffer (Recommended)

V-trace is particularly effective with replay buffers:

```yaml
adv_estimator: "vtrace"

replay:
  enabled: true
  capacity: 10000
  train_steps_per_env_step: 3  # Leverage off-policy capability

offpolicy_monitor:
  enabled: true
  save_behavior_log_probs: true  # Required for V-trace

vtrace:
  rho_bar: 2.0  # Less conservative for replay buffer
  c_bar: 1.0
```

### Parameter Tuning

#### rho_bar (ρ̄)
- **1.0** (default): Conservative, truncates IS ratio to [0, 1]
- **2.0-5.0**: Allows more off-policy correction
- **∞**: No truncation (not recommended)

#### c_bar (c̄)
- **1.0** (default): Standard choice for stability
- **Higher values**: Faster credit assignment but less stable

### Monitoring Metrics

Key metrics to track when using V-trace:

```python
# In off-policy monitor
metrics["vtrace/avg_rho"]        # Average importance sampling ratio
metrics["vtrace/clipped_ratio"]  # Fraction of clipped ratios
metrics["vtrace/avg_advantage"]  # Average V-trace advantage
```

## When to Use V-trace

### Ideal Scenarios

1. **Large Replay Buffers**: V-trace excels at off-policy learning
2. **Distributed Training**: Handles policy lag between actors and learner
3. **Async Training**: Robust to stale gradients
4. **High Sample Efficiency Needed**: Better utilization of old data

### Comparison with Other Estimators

| Estimator | On-Policy | Off-Policy | Stability | Sample Efficiency |
|-----------|-----------|------------|-----------|-------------------|
| REINFORCE | ✓         | ✗          | Low       | Low               |
| GAE       | ✓         | Limited    | High      | Medium            |
| V-trace   | ✓         | ✓          | High      | High              |

## Example: FrozenLake with V-trace

Complete example configuration for FrozenLake environment:

```yaml
# frozen_lake_vtrace.yaml
adv_estimator: "vtrace"
gamma: 0.99

vtrace:
  rho_bar: 1.5
  c_bar: 1.0

replay:
  enabled: true
  capacity: 5000
  train_steps_per_env_step: 2

offpolicy_monitor:
  enabled: true
  save_behavior_log_probs: true
```

## Testing V-trace

Run the test suite to verify implementation:

```bash
cd roll_dev/ROLL
python tests/test_vtrace.py
```

The test suite includes:
- Basic computation test
- Comparison with GAE (on-policy case)
- Importance sampling ratio tests
- Truncation threshold tests
- DataProto integration test

## Implementation Checklist

- [x] Core V-trace algorithm in `functionals.py`
- [x] Integration with `compute_advantage`
- [x] Configuration classes in `agentic_config.py`
- [x] Support for behavior/target log probs
- [x] Test suite
- [x] Example configuration files
- [ ] Integration with pipeline monitoring (optional)
- [ ] Custom priority function for replay buffer (optional)

## Technical Notes

### Shape Handling

The implementation handles the common shape mismatch between log_probs (seq_len-1) and rewards/values (seq_len) by padding when necessary.

### Numerical Stability

- Uses log-space computation for importance ratios
- Applies masking before exponential to avoid NaN
- Truncation prevents extreme values

### Memory Efficiency

- All computations are done with `@torch.no_grad()`
- In-place operations where possible
- Efficient backward iteration for value targets

## References

1. [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
2. [Original V-trace Implementation in TF](https://github.com/deepmind/scalable_agent)
3. [ROLL Framework Documentation](../README.md)

## Troubleshooting

### Error: "V-trace requires old_log_probs"

Ensure `offpolicy_monitor.save_behavior_log_probs = true` in your config.

### High Variance in Advantages

Try reducing `rho_bar` to 1.0 or lower for more conservative truncation.

### Poor Sample Efficiency

Increase `rho_bar` to allow more off-policy correction (e.g., 2.0-5.0).

### Training Instability

1. Reduce `rho_bar` and `c_bar` to 1.0
2. Enable advantage whitening
3. Reduce learning rate