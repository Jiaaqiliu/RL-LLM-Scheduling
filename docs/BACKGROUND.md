# Project Background

## The Problem: LLM Serving Scheduling

Serving large language models (LLMs) at scale has unique scheduling challenges:

1. **Unpredictable output lengths**: Unlike fixed-size DNN inference, LLM requests generate variable-length outputs. The output length is unknown at dispatch time.

2. **Dynamic KV cache growth**: Each token generated adds to the key-value (KV) cache stored in GPU memory. Memory grows linearly with sequence length.

3. **Memory preemptions**: When GPU memory is exhausted, running requests must be preempted (swapped to CPU or recomputed), causing severe tail-latency degradation (up to 3.8x P99 increase).

4. **Memory fragmentation**: Load balancing spreads requests across instances but fragments free memory, preventing large new requests from being scheduled even when total free memory is sufficient.

5. **Priority differentiation**: Interactive applications (chatbots) need low latency while batch tasks (summarization) are latency-tolerant. Existing systems treat all requests equally.

## The Baseline: Llumnix (OSDI 2024)

Llumnix is the first system to support **runtime rescheduling** of LLM requests across GPU instances via **live KV cache migration**. Key innovations:

### Live Migration
- Pipelines KV cache copying with token generation computation
- Achieves **near-zero downtime** (constant, independent of sequence length)
- Uses the append-only property of KV cache to overlap copying and compute

### Virtual Usage Abstraction
Llumnix unifies multiple scheduling goals through a single concept: **virtual usage**.

Each request has a "virtual usage" that may differ from its physical GPU memory usage. The scheduling decisions are driven by a **freeness** metric:

```
Freeness(instance) = (M - sum(VirtualUsage)) / B
```
Where M = total GPU memory blocks, B = batch size.

Llumnix dispatches new requests to the instance with highest freeness, and triggers migration from low-freeness to high-freeness instances.

### Four Heuristic Rules (CalcVirtualUsage)

1. **Normal requests**: virtual = physical usage (standard load balancing)
2. **Queuing requests**: virtual = demand (inflate to attract migration out)
3. **High-priority requests**: virtual = physical + headroom (reserve space)
4. **Terminating instances**: virtual = infinity (drain all requests out)

### Llumnix Architecture

```
┌─────────────────────────────────────────┐
│              Global Scheduler            │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │ New Requests  │  │ Instance Loads   │ │
│  └──────┬───────┘  └───────┬──────────┘ │
│         │ Dispatch         │ Report     │
├─────────┼──────────────────┼────────────┤
│         ▼                  ▲            │
│  ┌─────────────────────────────────┐    │
│  │     Instance (Llumlet)          │    │
│  │  ┌──────────┐ ┌──────────────┐  │    │
│  │  │  Local    │ │  Migration   │  │    │
│  │  │ Scheduler │ │ Coordinator  │  │    │
│  │  └────┬─────┘ └──────────────┘  │    │
│  │       ▼                          │    │
│  │  ┌──────────┐                    │    │
│  │  │ Executor │  (vLLM + GPU)     │    │
│  │  └──────────┘                    │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

### Results from Original Paper
- P99 first-token latency improved by up to 15x over INFaaS++
- P99 decode latency improved by up to 2x
- High-priority requests accelerated by 1.5x
- 36% cost savings with efficient auto-scaling
- Evaluated on 16x A10 GPUs (24GB) with LLaMA-7B and LLaMA-30B

## Our Contribution: RL-based Scheduling

### Why RL?
The 4 heuristic rules above are:
- **Static**: Cannot adapt to workload pattern changes
- **Myopic**: Only see current state, no predictive lookahead
- **Fixed tradeoffs**: Hardcoded balance among competing objectives (latency vs. fragmentation vs. priority)

### Our Approach
Replace `CalcVirtualUsage()` with a **learned RL policy**:
- **State**: Per-instance features (memory, queue) + per-request features (tokens, priority) + cluster metrics
- **Action**: Virtual usage values per instance (continuous)
- **Reward**: Weighted combination of latency, preemptions, fragmentation, SLO compliance

### Key Design Principle
**Minimal modification**: Only the virtual usage calculation changes. All of Llumnix's infrastructure (migration, dispatch, vLLM backend, Ray distributed system) remains untouched. This makes the project tractable and the comparison fair.

### RL Algorithm
- **PPO (Proximal Policy Optimization)**: Stable in continuous action spaces
- **Two-phase training**: (1) Behavior cloning from heuristic traces, (2) Online fine-tuning in simulation
- **Baselines**: Original heuristic, DQN with discretized actions, Round-Robin

## Course Context

- **Course**: 790-199 (Instructor: Sishuai), University of Virginia
- **Team**: Yuxuan Bai (MDP formulation), Jiaqi Liu (RL implementation), Peng Xia (Llumnix integration)
- **Deliverables**: Midterm report, Final presentation (PPT), Final report (LaTeX/PDF)
