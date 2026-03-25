# Experiment Guide

Complete instructions for running all 5 experiments.

---

## Prerequisites

- Llumnix installed and verified (see `scripts/setup_env.sh`)
- LLaMA-7B model downloaded to `./models/llama-7b/`
- ShareGPT dataset at `./data/sharegpt.json`
- BurstGPT dataset at `./data/burstgpt.json` (if available)
- Trained RL model at `./models/ppo_llumnix/`
- At least 4 NVIDIA GPUs available

---

## General Setup

### Start Llumnix Server (Heuristic Baseline)

```bash
export N_INSTANCES=16  # or 4/8 depending on GPU count
export MODEL_PATH=./models/llama-7b
export PORT=8000

python -m llumnix.entrypoints.vllm.api_server \
    --initial-instances $N_INSTANCES \
    --model $MODEL_PATH \
    --host 0.0.0.0 \
    --port $PORT \
    --dispatch-policy load \
    --launch-ray-cluster \
    --trust-remote-code
```

### Start Llumnix Server (RL Policy)

```bash
# Same as above but with RL enabled
python -m llumnix.entrypoints.vllm.api_server \
    --initial-instances $N_INSTANCES \
    --model $MODEL_PATH \
    --host 0.0.0.0 \
    --port $PORT \
    --dispatch-policy load \
    --launch-ray-cluster \
    --trust-remote-code \
    --rl-enabled \
    --rl-model-path ./models/ppo_llumnix/
```

Note: The `--rl-enabled` and `--rl-model-path` flags need to be added to the CLI argument parser as part of the integration (Task 3).

### Start Round-Robin Baseline

```bash
python -m llumnix.entrypoints.vllm.api_server \
    --initial-instances $N_INSTANCES \
    --model $MODEL_PATH \
    --host 0.0.0.0 \
    --port $PORT \
    --dispatch-policy rr \
    --launch-ray-cluster
```

### Run Benchmark

```bash
python benchmark/benchmark_serving.py \
    --ip_ports "localhost:$PORT" \
    --tokenizer $MODEL_PATH \
    --random_prompt_count 10000 \
    --dataset_type sharegpt \
    --dataset_path ./data/sharegpt.json \
    --qps $QPS \
    --distribution poisson \
    --log_latencies \
    --fail_on_response_failure \
    --results_dir ./results/$EXPERIMENT_NAME/
```

---

## Experiment 1: Baseline Serving Performance

**Goal**: Compare mean & P99 latency on realistic workload traces.

### Configurations to run:

| Policy | Dispatch | Dataset | QPS Range |
|--------|----------|---------|-----------|
| Round-Robin | `rr` | ShareGPT | 7.0, 7.5, 8.0, 8.5 |
| Llumnix-Heuristic | `load` | ShareGPT | 7.0, 7.5, 8.0, 8.5 |
| Llumnix-RL | `load` + RL | ShareGPT | 7.0, 7.5, 8.0, 8.5 |
| Round-Robin | `rr` | BurstGPT | 7.5, 8.0, 8.5 |
| Llumnix-Heuristic | `load` | BurstGPT | 7.5, 8.0, 8.5 |
| Llumnix-RL | `load` + RL | BurstGPT | 7.5, 8.0, 8.5 |

```bash
# Example: ShareGPT with heuristic, QPS=7.5
for qps in 7.0 7.5 8.0 8.5; do
    # Start server with heuristic policy, then run:
    python benchmark/benchmark_serving.py \
        --ip_ports "localhost:8000" \
        --tokenizer $MODEL_PATH \
        --random_prompt_count 10000 \
        --dataset_type sharegpt \
        --dataset_path ./data/sharegpt.json \
        --qps $qps \
        --distribution poisson \
        --log_latencies \
        --results_dir ./results/exp1/sharegpt_heuristic_qps${qps}/
done
```

### Metrics to collect:
- End-to-end request latency: mean, P99
- Prefill latency: mean, P99
- Decode latency: mean, P99
- Preemption loss: total seconds

### Expected output:
Plots matching Figure 11 format from the Llumnix paper (7 columns: Request P99, Request Mean, Prefill P99, Prefill Mean, Decode P99, Decode Mean, Preemption Loss).

---

## Experiment 2: Priority Workloads

**Goal**: Test SLO isolation with mixed-priority requests.

### Setup:
- 10% of requests tagged as high-priority
- Short-Short (S-S) sequence length distribution (input_mean=128, output_mean=128)
- Gamma arrival distribution with varying CV

```bash
for cv in 2 4 6 8; do
    python benchmark/benchmark_serving.py \
        --ip_ports "localhost:8000" \
        --tokenizer $MODEL_PATH \
        --random_prompt_count 10000 \
        --dataset_type synthetic \
        --input_mean 128 --output_mean 128 \
        --high_priority_ratio 0.1 \
        --qps 2.0 \
        --distribution gamma \
        --cv $cv \
        --log_latencies \
        --results_dir ./results/exp2/rl_cv${cv}/
done
```

Note: You may need to modify `benchmark_serving.py` to support `--high_priority_ratio` and synthetic distribution parameters.

### Metrics:
- High-priority request latencies: mean, P99 (end-to-end, prefill, decode)
- Normal request latencies: mean, P99
- Decode execution time comparison

---

## Experiment 3: Bursty Traffic

**Goal**: Test adaptation under load spikes.

### Setup:
- Gamma arrival distribution with high CV (bursty)
- Medium-Medium (M-M) distribution: input_mean=256, output_mean=256

```bash
for cv in 2 4 6 8; do
    for policy in rr heuristic rl; do
        # Start appropriate server, then:
        python benchmark/benchmark_serving.py \
            --ip_ports "localhost:8000" \
            --tokenizer $MODEL_PATH \
            --random_prompt_count 10000 \
            --dataset_type synthetic \
            --input_mean 256 --output_mean 256 \
            --qps 2.5 \
            --distribution gamma \
            --cv $cv \
            --log_latencies \
            --results_dir ./results/exp3/${policy}_cv${cv}/
    done
done
```

### Metrics:
- Same latency metrics as Exp 1
- Focus on how well each policy handles load spikes (compare P99 vs mean gap)

---

## Experiment 4: Long-Context Workloads

**Goal**: Stress memory fragmentation with long sequences.

### Setup:
- Long-Long (L-L) distribution: input_mean=512, output_mean=512
- Poisson arrivals, moderate QPS

```bash
for qps in 1.0 1.5 2.0 2.5; do
    python benchmark/benchmark_serving.py \
        --ip_ports "localhost:8000" \
        --tokenizer $MODEL_PATH \
        --random_prompt_count 5000 \
        --dataset_type synthetic \
        --input_mean 512 --output_mean 512 \
        --qps $qps \
        --distribution poisson \
        --log_latencies \
        --results_dir ./results/exp4/rl_qps${qps}/
done
```

### Metrics:
- Fragmentation ratio over time (if instrumented)
- Queuing delays
- P99 prefill latency (most affected by fragmentation)

---

## Experiment 5: Generalization

**Goal**: Train on ShareGPT, test on BurstGPT (cross-distribution transfer).

### Setup:
1. RL model trained ONLY on ShareGPT traces
2. Test on BurstGPT traces (unseen distribution)
3. Compare with heuristic (which is distribution-agnostic)

```bash
# Test RL (trained on ShareGPT) on BurstGPT
python benchmark/benchmark_serving.py \
    --ip_ports "localhost:8000" \
    --tokenizer $MODEL_PATH \
    --random_prompt_count 10000 \
    --dataset_type burstgpt \
    --dataset_path ./data/burstgpt.json \
    --qps 8.0 \
    --distribution poisson \
    --log_latencies \
    --results_dir ./results/exp5/rl_burstgpt/

# Compare with heuristic on same trace
python benchmark/benchmark_serving.py \
    --ip_ports "localhost:8000" \
    --tokenizer $MODEL_PATH \
    --random_prompt_count 10000 \
    --dataset_type burstgpt \
    --dataset_path ./data/burstgpt.json \
    --qps 8.0 \
    --distribution poisson \
    --log_latencies \
    --results_dir ./results/exp5/heuristic_burstgpt/
```

### Metrics:
- Compare RL vs heuristic performance on unseen distribution
- Report performance gap (does RL generalize or overfit to ShareGPT?)

---

## Ablation Studies

### Reward Weight Sensitivity

Train with different weight configurations:

```python
ABLATION_CONFIGS = [
    {"w_latency": 1.0, "w_preempt": 0.0, "w_frag": 0.0, "w_slo": 0.0, "w_throughput": 0.0},  # Latency only
    {"w_latency": 0.0, "w_preempt": 1.0, "w_frag": 0.0, "w_slo": 0.0, "w_throughput": 0.0},  # Preempt only
    {"w_latency": 0.0, "w_preempt": 0.0, "w_frag": 1.0, "w_slo": 0.0, "w_throughput": 0.0},  # Frag only
    {"w_latency": 0.5, "w_preempt": 0.5, "w_frag": 0.0, "w_slo": 0.0, "w_throughput": 0.0},  # Lat+Preempt
    {"w_latency": 0.3, "w_preempt": 0.25, "w_frag": 0.15, "w_slo": 0.2, "w_throughput": 0.1},  # Full (default)
]
```

### State Feature Ablation

Remove one feature group at a time and retrain:
1. Without waiting request features
2. Without cluster-level features
3. Without priority features
4. Without memory fragmentation indicator

### Training Duration

Record performance at checkpoints:
- 100K, 250K, 500K, 750K, 1M timesteps

---

## Result Collection

All results should be saved in structured format:

```
results/
├── exp1/
│   ├── sharegpt_rr_qps7.0/
│   │   ├── latencies.csv
│   │   └── summary.json
│   ├── sharegpt_heuristic_qps7.0/
│   ├── sharegpt_rl_qps7.0/
│   └── ...
├── exp2/ ...
├── exp3/ ...
├── exp4/ ...
├── exp5/ ...
├── ablation/
│   ├── reward_weights/
│   └── state_features/
└── figures/
    ├── exp1_sharegpt.pdf
    ├── exp1_burstgpt.pdf
    ├── exp2_priority.pdf
    ├── exp3_bursty.pdf
    ├── exp4_longcontext.pdf
    ├── exp5_generalization.pdf
    ├── ablation_rewards.pdf
    ├── ablation_features.pdf
    └── training_curves.pdf
```
