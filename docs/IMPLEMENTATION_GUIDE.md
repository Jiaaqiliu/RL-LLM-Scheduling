# Implementation Guide

This document provides the detailed implementation plan for all 5 project tasks.

---

## Task 1: MDP Formulation and Environment Design

### State Space

```python
import gymnasium
import numpy as np

# Per-instance features (for each of N instances)
INSTANCE_FEATURES = {
    "memory_usage_ratio":     float,  # num_used_gpu_blocks / num_total_gpu_blocks
    "free_blocks_ratio":      float,  # num_free_gpu_blocks / num_total_gpu_blocks
    "num_running_requests":   int,    # Currently executing requests
    "num_waiting_requests":   int,    # Queued requests
    "waiting_demand_ratio":   float,  # num_blocks_all_waiting / num_total_gpu_blocks
    "first_waiting_demand":   float,  # num_blocks_first_waiting / num_total_gpu_blocks
    "avg_seq_length_ratio":   float,  # Average sequence length / max_seq_length
    "has_high_priority":      int,    # 0 or 1
}
# Total: 8 features per instance

# Cluster-level features
CLUSTER_FEATURES = {
    "mean_load":              float,  # Average memory usage across instances
    "load_std":               float,  # Standard deviation (imbalance indicator)
    "max_load":               float,  # Hottest instance
    "total_waiting":          int,    # System-wide queue depth
    "num_active_instances":   int,    # Instances currently serving
}
# Total: 5 features

# Observation space
observation_space = gymnasium.spaces.Dict({
    "instances": gymnasium.spaces.Box(
        low=0.0, high=1.0, shape=(N_INSTANCES, 8), dtype=np.float32
    ),
    "cluster": gymnasium.spaces.Box(
        low=0.0, high=1.0, shape=(5,), dtype=np.float32
    ),
})
```

### Action Space

```python
# Option A (recommended): Continuous virtual usage adjustment per instance
# Each value in [-1, 1], maps to virtual usage multiplier:
#   actual_virtual = physical * (1 + action * SCALE_FACTOR)
# SCALE_FACTOR = 0.5 means range [0.5x, 1.5x] of physical usage
action_space = gymnasium.spaces.Box(
    low=-1.0, high=1.0, shape=(N_INSTANCES,), dtype=np.float32
)

# Option B (fallback): Discrete actions per instance
# 5 levels: {strong_deflate, deflate, keep, inflate, strong_inflate}
action_space = gymnasium.spaces.MultiDiscrete([5] * N_INSTANCES)
```

### Reward Function

```python
def compute_reward(metrics, config):
    """
    Reward at each scheduling step.
    All terms normalized to roughly [-1, 1] range.
    """
    # Latency penalty: normalized P99 decode latency
    r_latency = -config.w_latency * min(metrics.p99_decode_latency / config.baseline_p99, 3.0)

    # Preemption penalty: count of preemptions in this interval
    r_preempt = -config.w_preempt * min(metrics.num_preemptions / config.max_preemptions, 1.0)

    # Fragmentation penalty
    r_frag = -config.w_frag * metrics.fragmentation_ratio

    # SLO compliance bonus (fraction of requests meeting latency target)
    r_slo = config.w_slo * metrics.slo_compliance_rate

    # Throughput bonus (completed requests / expected)
    r_throughput = config.w_throughput * min(metrics.throughput / config.expected_throughput, 1.0)

    return r_latency + r_preempt + r_frag + r_slo + r_throughput

# Default weights
DEFAULT_REWARD_WEIGHTS = {
    "w_latency": 0.30,
    "w_preempt": 0.25,
    "w_frag": 0.15,
    "w_slo": 0.20,
    "w_throughput": 0.10,
}
```

### Simulation Environment

```python
class LlumnixSchedulingEnv(gymnasium.Env):
    """
    Simulated Llumnix scheduling environment for RL training.
    Does NOT require GPUs — uses timing models instead of real inference.
    """

    def __init__(self, config):
        super().__init__()
        self.n_instances = config.n_instances  # 4, 8, or 16
        self.gpu_blocks_per_instance = config.gpu_blocks  # e.g., 2048 for A10 with LLaMA-7B
        self.max_seq_length = config.max_seq_length  # e.g., 2048

        # Timing model: decode time as function of batch size and seq length
        # From Llumnix paper Figure 4 data
        self.timing_model = TimingModel()

        # Request trace
        self.trace = load_trace(config.trace_path)

        # State
        self.instances = [InstanceState(self.gpu_blocks_per_instance) for _ in range(self.n_instances)]
        self.current_time = 0.0
        self.trace_idx = 0
        self.step_interval = config.step_interval  # e.g., 0.1 seconds

        # Spaces
        self.observation_space = ...  # As defined above
        self.action_space = ...       # As defined above

    def reset(self, seed=None, options=None):
        # Reset all instances, reload trace, reset time
        for inst in self.instances:
            inst.reset()
        self.current_time = 0.0
        self.trace_idx = 0
        return self._get_obs(), {}

    def step(self, action):
        # 1. Apply action: set virtual usage multipliers
        for i, inst in enumerate(self.instances):
            inst.virtual_usage_multiplier = 1.0 + action[i] * 0.5

        # 2. Compute freeness for each instance
        freenesses = [inst.compute_freeness() for inst in self.instances]

        # 3. Dispatch arriving requests (in this time interval) to highest-freeness instance
        while self.trace_idx < len(self.trace) and self.trace[self.trace_idx].arrival_time <= self.current_time + self.step_interval:
            req = self.trace[self.trace_idx]
            target = np.argmax(freenesses)
            self.instances[target].add_request(req)
            freenesses[target] = self.instances[target].compute_freeness()
            self.trace_idx += 1

        # 4. Check migration: pair low-freeness ↔ high-freeness instances
        self._check_migration(freenesses)

        # 5. Advance simulation: generate tokens, update KV cache, handle completions/preemptions
        metrics = self._simulate_step()

        # 6. Compute reward
        reward = compute_reward(metrics, self.reward_config)

        # 7. Advance time
        self.current_time += self.step_interval
        terminated = self.trace_idx >= len(self.trace) and all(inst.is_empty() for inst in self.instances)
        truncated = self.current_time > self.max_time

        return self._get_obs(), reward, terminated, truncated, {"metrics": metrics}

    def _simulate_step(self):
        """Simulate one scheduling interval without real GPU inference."""
        metrics = StepMetrics()
        for inst in self.instances:
            # Generate tokens for running requests (using timing model)
            for req in inst.running_requests:
                decode_time = self.timing_model.predict(
                    batch_size=len(inst.running_requests),
                    seq_length=req.current_length
                )
                req.generate_token(decode_time)
                req.kv_blocks = req.current_length // TOKENS_PER_BLOCK

            # Check completions
            completed = [r for r in inst.running_requests if r.is_complete()]
            inst.running_requests = [r for r in inst.running_requests if not r.is_complete()]
            metrics.completed += len(completed)

            # Check preemptions (if used blocks > total blocks)
            while inst.total_used_blocks() > inst.total_blocks:
                preempted = inst.preempt_lowest_priority()
                metrics.preemptions += 1

            # Try to schedule waiting requests
            inst.try_schedule_waiting()

            # Update latency metrics
            metrics.update_latencies(inst, completed)

        return metrics
```

### Timing Model

Based on Llumnix paper Figure 4:

```python
class TimingModel:
    """Predicts decode latency based on batch size and sequence length."""

    def __init__(self):
        # From Figure 4 data: LLaMA-7B on A10 GPU
        # Approximate: decode_time_ms = base + alpha * batch_tokens
        self.base_ms = 8.0  # minimum decode time
        self.alpha = 0.015  # ms per batched token

    def predict(self, batch_size, seq_length):
        """Returns decode time in seconds."""
        total_tokens = batch_size * seq_length
        decode_ms = self.base_ms + self.alpha * total_tokens
        return decode_ms / 1000.0
```

---

## Task 2: RL Agent Implementation and Training

### Policy Network

```python
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LlumnixFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Llumnix scheduling.
    Handles variable-size instance features with attention.
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        # Instance encoder
        instance_dim = observation_space["instances"].shape[-1]  # 8
        self.instance_encoder = nn.Sequential(
            nn.Linear(instance_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Self-attention over instances
        self.attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(64)

        # Cluster encoder
        cluster_dim = observation_space["cluster"].shape[0]  # 5
        self.cluster_encoder = nn.Sequential(
            nn.Linear(cluster_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Combiner
        self.combiner = nn.Sequential(
            nn.Linear(64 + 32, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # Encode per-instance features
        inst = observations["instances"]  # (batch, N, 8)
        inst_enc = self.instance_encoder(inst)  # (batch, N, 64)

        # Self-attention for inter-instance reasoning
        attn_out, _ = self.attention(inst_enc, inst_enc, inst_enc)
        inst_enc = self.attn_norm(inst_enc + attn_out)  # residual
        inst_pooled = inst_enc.mean(dim=1)  # (batch, 64)

        # Encode cluster features
        cluster = observations["cluster"]  # (batch, 5)
        cluster_enc = self.cluster_encoder(cluster)  # (batch, 32)

        # Combine
        combined = torch.cat([inst_pooled, cluster_enc], dim=-1)
        return self.combiner(combined)  # (batch, features_dim)
```

### PPO Training

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(config, seed):
    def _init():
        env = LlumnixSchedulingEnv(config)
        env.reset(seed=seed)
        return env
    return _init

def train(config):
    # Vectorized environments for parallel data collection
    n_envs = 8
    env = SubprocVecEnv([make_env(config, seed=i) for i in range(n_envs)])

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": LlumnixFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": dict(pi=[128, 64], vf=[128, 64]),
        },
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs/ppo/",
        seed=42,
    )

    # Callbacks
    eval_env = LlumnixSchedulingEnv(config)
    callbacks = [
        EvalCallback(eval_env, eval_freq=5000, best_model_save_path="./best_model/"),
        CheckpointCallback(save_freq=10000, save_path="./checkpoints/"),
    ]

    model.learn(total_timesteps=1_000_000, callback=callbacks)
    model.save("./final_model/ppo_llumnix")
```

### Behavior Cloning Pre-training

```python
from stable_baselines3.common.policies import ActorCriticPolicy

def pretrain_from_heuristic(model, trace_path, n_episodes=100):
    """
    Pre-train RL policy to mimic heuristic behavior.
    Collect (state, heuristic_action) pairs, then supervised train the policy.
    """
    env = LlumnixSchedulingEnv(config)
    states, actions = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # Get heuristic action (what the original policy would do)
            heuristic_action = compute_heuristic_action(obs, env)
            states.append(obs)
            actions.append(heuristic_action)
            obs, _, terminated, truncated, _ = env.step(heuristic_action)
            done = terminated or truncated

    # Supervised learning on policy network
    dataset = BCDataset(states, actions)
    train_behavior_cloning(model.policy, dataset, epochs=50, lr=1e-3)
```

### DQN Baseline

```python
from stable_baselines3 import DQN

def train_dqn(config):
    """DQN with discretized action space."""
    env = LlumnixSchedulingEnvDiscrete(config)  # Discrete action variant

    model = DQN(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": LlumnixFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": [128, 64],
        },
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./logs/dqn/",
    )

    model.learn(total_timesteps=500_000)
    model.save("./final_model/dqn_llumnix")
```

---

## Task 3: Llumnix Integration

### Add RLBasedLoad to load_computation.py

```python
# Add to llumnix/load_computation.py

class RLBasedLoad(InstanceLoad):
    """RL-based load metric replacing heuristic virtual usage."""

    BUSY_THRESHOLD = 0.8

    def __init__(self, model_path: str, device: str = "cpu"):
        from stable_baselines3 import PPO
        self.policy = PPO.load(model_path, device=device)
        self._load = 0.0
        self._is_busy = False

    def compute_load(self, instance_info) -> float:
        state = {
            "memory_usage_ratio": instance_info.num_used_gpu_blocks / max(instance_info.num_total_gpu_blocks, 1),
            "free_blocks_ratio": instance_info.num_free_gpu_blocks / max(instance_info.num_total_gpu_blocks, 1),
            "num_running": instance_info.num_running_requests / 100.0,  # normalize
            "num_waiting": instance_info.num_waiting_requests / 50.0,
            "waiting_demand_ratio": instance_info.num_blocks_all_waiting_requests / max(instance_info.num_total_gpu_blocks, 1),
            "first_waiting_demand": instance_info.num_blocks_first_waiting_request / max(instance_info.num_total_gpu_blocks, 1),
        }
        # Note: In production, this would receive cluster-level state too
        # For per-instance load, we use only instance features
        action, _ = self.policy.predict(state, deterministic=True)

        physical_load = state["memory_usage_ratio"]
        self._load = physical_load * (1.0 + float(action[0]) * 0.5)
        self._is_busy = self._load > self.BUSY_THRESHOLD
        return self._load

    def is_busy(self) -> bool:
        return self._is_busy
```

### Register in Config

```python
# Add to llumnix/config/default.py
_C.RL = CN()
_C.RL.ENABLED = False
_C.RL.MODEL_PATH = ""
_C.RL.INFERENCE_DEVICE = "cpu"
```

### Add to Load Metric Factory

Find where load metrics are instantiated (in `instance_info.py` or `global_scheduler.py`) and add:

```python
if config.RL.ENABLED:
    load_metric = RLBasedLoad(config.RL.MODEL_PATH, config.RL.INFERENCE_DEVICE)
```

---

## Task 4 & 5: See EXPERIMENT_GUIDE.md and DELIVERABLES_SPEC.md
