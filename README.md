<div align="center">

# Freshness-Aware Prioritized Experience Replay for LLM/VLM Reinforcement Learning

<p>
  <a href="https://arxiv.org/abs/2604.16918"><img src="https://img.shields.io/badge/arXiv-2604.16918-b31b1b.svg" alt="arXiv"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://github.com/alibaba/ROLL"><img src="https://img.shields.io/badge/Built%20on-ROLL-ff69b4.svg" alt="Built on ROLL"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white" alt="Python 3.10+">
</p>

<p>
<b>Weiyu Ma</b><sup>1</sup> &nbsp;
Yongcheng Zeng<sup>2</sup> &nbsp;
Yan Song<sup>3</sup> &nbsp;
Xinyu Cui<sup>2</sup> &nbsp;
Jian Zhao<sup>4</sup> &nbsp;
Xuhui Liu<sup>1</sup> &nbsp;
<b>Mohamed Elhoseiny</b><sup>1</sup>
</p>

<sup>1</sup> KAUST &nbsp;&nbsp; <sup>2</sup> CASIA &nbsp;&nbsp; <sup>3</sup> University College London &nbsp;&nbsp; <sup>4</sup> Zhongguancun Institute of AI

</div>

---

## Overview

**FreshPER** is, to the best of our knowledge, the first method to successfully apply Prioritized Experience Replay (PER) to reinforcement learning of Large Language Models (LLMs) and Vision-Language Models (VLMs).

On-policy algorithms that dominate LLM RL today — PPO, GRPO, REINFORCE++ — discard every collected trajectory after a single gradient update, which is especially wasteful in agentic settings where each multi-turn rollout can cost thousands of expensive tool/environment calls. A naïve port of PER does **not** fix this: the rapid policy evolution of billion-parameter models causes stored priorities to go stale, so old "high-priority" trajectories end up dominating sampling long after they have become uninformative.

FreshPER augments **any** PER base priority with a multiplicative exponential **age decay** whose form is directly motivated by the exponential decay of effective sample size (ESS) as the current policy drifts away from the behavior policy. The decay is a modular layer — it can be stacked on top of reward-magnitude, advantage-magnitude, or TD-error priorities without changing the rest of the training stack.

<p align="center"><i>
p<sub>i</sub> = p<sub>i</sub><sup>base</sup> · exp(−τ<sub>i</sub> / τ)
</i></p>

where τ<sub>i</sub> is the age (in gradient steps) of sample *i* since collection, and τ is the **age decay constant**. The half-life of priority is τ · ln 2.

## Key Results

Evaluated on eight environments spanning agentic, reasoning, math-competition, and multi-modal tasks, with Qwen2.5-0.5B/7B-Instruct and Qwen2.5-VL-3B-Instruct. All runs use REINFORCE++ as the policy-gradient backbone.

| Task            | Modality | Metric      | On-Policy | Standard PER | **FreshPER (Ours)** |    Δ    |
| --------------- | :------: | ----------- | :-------: | :----------: | :-----------------: | :-----: |
| NQ Search       |   LLM    | EM          |   0.508   |    0.336     |      **0.742**      |  +46%   |
| AIME            |   LLM    | Success     |   0.205   |    0.168     |      **0.242**      |  +18%   |
| Sokoban Simple  |   LLM    | Score       |   0.493   |    −0.907    |      **2.304**      |  +367%  |
| Sokoban Hard    |   LLM    | Score       |  −0.842   |    −0.847    |     **−0.512**      |    —    |
| FrozenLake      |   LLM    | Success     |   0.297   |    0.281     |      **0.305**      |         |
| FrozenLake      | **VLM**  | Success     |   0.270   |    0.250     |      **0.630**      |  +133%  |
| GeoQA           | **VLM**  | Success     |   0.475   |    0.447     |      **0.481**      |         |

Standard PER (no age decay) consistently under-performs the on-policy baseline — and collapses on Sokoban Simple and NQ Search — empirically confirming the priority-staleness failure mode that FreshPER is designed to fix. See Section 4 of the paper for full learning curves, the τ ablation, and the IS-correction ablation.

## Method at a Glance

1. **Rollout.** The behavior policy μ generates multi-turn trajectories; behavior log-probs and rewards are recorded at collection time.
2. **Replay buffer.** Trajectories are stored with a *base priority* p<sup>base</sup>: `|reward|+ε` for critic-free methods (REINFORCE++, GRPO), `|advantage|+ε` for actor-critic (PPO), or `|TD-error|+ε` for classical PER.
3. **Async age refresh.** A background CPU thread refreshes the age decay `exp(−τᵢ/τ)` for every buffer entry each training iteration — O(N) scan, naturally pipelined with GPU training.
4. **Prioritized sampling.** A sum segment tree enables O(log N) per-sample draws; stratified sampling reduces variance. An importance-sampling correction (β annealed 0.4 → 1.0) compensates for non-uniform sampling bias.
5. **Update.** Policy is updated on both (a) the fresh on-policy batch and (b) K off-policy replay batches per iteration.

See Algorithm 1 in the paper for the full training loop.

## Repository Layout

This repository builds directly on the [ROLL](https://github.com/alibaba/ROLL) framework. The FreshPER-specific additions live under:

```
roll/agentic/replay_buffer/
├── base_buffer.py          # Abstract buffer with priority, age, and IS-correction support
├── trajectory_buffer.py    # Trajectory-level replay buffer
├── step_buffer.py          # Step-level replay buffer
├── priority_functions.py   # Priority functions, including `reward_fresh` (our addition)
├── segment_tree.py         # Sum segment tree for O(log N) prioritized sampling
└── buffer_factory.py       # Buffer construction from config

examples/replay_buffer/     # Example YAML configs for replay + FreshPER
docs/replay_buffer.md       # Design notes for the replay buffer subsystem
docs/age_decay_issue_analysis.md   # Deep-dive on priority staleness
```

## Quick Start

### Installation

Follow the upstream [ROLL installation guide](https://alibaba.github.io/ROLL/docs/English/QuickStart/installation); FreshPER requires no extra dependencies beyond ROLL.

```bash
git clone --recursive https://github.com/Vision-CAIR/Freshness-Aware-PER.git
cd Freshness-Aware-PER
# then follow ROLL's environment setup
```

### Enabling FreshPER in a config

FreshPER is opt-in via a few fields in the agentic pipeline config:

```yaml
replay:
  enabled: true
  capacity: 50000
  sampling_mode: trajectory     # or "step"
  min_size: ${rollout_batch_size}

  # --- Prioritized sampling ---
  priority_function: reward_fresh       # |reward| × exp(-age/τ)
  priority_exponent: 0.6                # α in PER
  importance_sampling_correction: true  # enable IS correction
  importance_beta: 0.4                  # β, annealed → 1.0

  # --- Freshness-aware age decay (this repo's contribution) ---
  enable_age_decay: true
  age_decay: 500                        # τ; paper default 500, FrozenLake uses 1000
  refresh_interval: 1                   # async refresh every training step
```

All knobs are documented in [`roll/pipeline/agentic/agentic_config.py`](roll/pipeline/agentic/agentic_config.py). A good starting point for hands-on experimentation is `examples/replay_buffer/`.

### Running an example

```bash
# Example: FrozenLake with trajectory-level FreshPER
python examples/start_agentic_pipeline.py \
  --config_path examples/replay_buffer \
  --config_name agent_val_frozen_lake_trajrb_step_trainer
```

### Tuning tips from the paper

- **τ = 500** is a solid default. Harder / faster-evolving tasks (Sokoban) benefit from more aggressive decay; slower-evolving tasks (FrozenLake) prefer τ ≈ 1000.
- The benefit of replay scales with task difficulty. On near-saturated tasks (GSM8K, CliffWalking) replay adds little; on hard agentic/VLM tasks it can 2–4× the score.
- Adding the IS correction (β = 0.4) rarely changes peak performance but markedly improves late-stage training stability.

## Citation

If you use this code or build on our work, please cite:

```bibtex
@article{ma2026freshper,
  title   = {Freshness-Aware Prioritized Experience Replay for LLM/VLM Reinforcement Learning},
  author  = {Ma, Weiyu and Zeng, Yongcheng and Song, Yan and Cui, Xinyu and
             Zhao, Jian and Liu, Xuhui and Elhoseiny, Mohamed},
  journal = {arXiv preprint arXiv:2604.16918},
  year    = {2026}
}
```

## Acknowledgements

This project is built on top of the excellent **[ROLL](https://github.com/alibaba/ROLL)** framework by the Alibaba ROLL team. FreshPER is implemented as an additive layer inside ROLL's agentic pipeline — the distributed runtime, inference/training backends (DeepSpeed, Megatron-Core, vLLM, SGLang), multi-turn environment abstractions, and agentic tooling are all inherited from upstream. We are deeply grateful to the ROLL team for releasing such a clean, modular, and high-performance RL library for LLMs; our work would not exist without it. Please also consider citing ROLL:

```bibtex
@article{wang2025roll,
  title   = {Reinforcement Learning Optimization for Large-Scale Learning: An Efficient and User-Friendly Scaling Library},
  author  = {Wang, Weixun and others},
  journal = {arXiv preprint arXiv:2506.06122},
  year    = {2025}
}
```

We also thank the authors of Schaul et al. (2016) for the original PER formulation, and the Qwen team for open-sourcing the Qwen2.5 / Qwen2.5-VL models used in our experiments.

This research was supported by funding from the King Abdullah University of Science and Technology (KAUST) Center of Excellence for Generative AI under Award No. 5940.

## License

Released under the [Apache License 2.0](LICENSE), consistent with upstream ROLL.
