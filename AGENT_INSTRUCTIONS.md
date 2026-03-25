# Agent Master Instructions

> **READ THIS FILE FIRST.** This is the single source of truth for what you need to do.

## Your Mission

You are an AI Agent running on a GPU server. Your job is to execute an entire course project:
**Replace Llumnix's heuristic scheduling with RL-based scheduling for LLM serving.**

You must produce ALL deliverables listed below, on schedule.

---

## Phase Overview

| Phase | What to Do | Deliverables |
|-------|-----------|-------------|
| **Phase 0** | Environment setup & verification | Working Llumnix installation |
| **Phase 1** | Implement RL components | Simulation env + RL agent + training pipeline |
| **Phase 2** | Train RL agent | Trained model checkpoints |
| **Phase 3** | Integrate into Llumnix | Modified Llumnix with RL scheduling |
| **Phase 4** | Run experiments (5 total) | Raw results + plots |
| **Phase 5** | Midterm deliverables | Midterm report (PDF) + presentation (if needed) |
| **Phase 6** | Complete all experiments + analysis | Full experimental results + ablation studies |
| **Phase 7** | Final deliverables | Final PPT + Final Report (LaTeX + PDF) |

---

## Phase 0: Environment Setup

Read `scripts/setup_env.sh` and execute it. Then verify:

```bash
# Must all succeed:
python -c "import torch; assert torch.cuda.is_available(); print(f'GPUs: {torch.cuda.device_count()}')"
python -c "import vllm; import ray; import llumnix; print('All imports OK')"
python -c "from stable_baselines3 import PPO; import gymnasium; print('RL libs OK')"
```

Download required models and datasets:
- LLaMA-7B (16-bit) from HuggingFace
- ShareGPT dataset
- BurstGPT dataset (if available)

Quick sanity test — start Llumnix with 1 instance and serve a single request.

---

## Phase 1: Implement RL Components

Read `docs/IMPLEMENTATION_GUIDE.md` for full details. Create directory `llumnix/rl/` in the cloned llumnix-ray repo.

### Files to create:

1. **`llumnix/rl/environment.py`** — Gymnasium environment that simulates Llumnix scheduling
   - State: per-instance features (memory, queue, requests) + cluster features
   - Action: virtual usage adjustment per instance (continuous, Box space)
   - Reward: weighted combination of -latency, -preemptions, -fragmentation, +SLO, +throughput
   - Uses timing model (not real GPU inference) for fast training

2. **`llumnix/rl/policy_network.py`** — Multi-head PPO network
   - Instance encoder (per-instance features → embeddings)
   - Attention over instances (permutation invariant)
   - Cluster encoder
   - Combined output → virtual usage values

3. **`llumnix/rl/train.py`** — Training pipeline
   - Phase A: Behavior cloning from heuristic traces (pre-training)
   - Phase B: PPO fine-tuning in simulation
   - Logging to TensorBoard

4. **`llumnix/rl/state_collector.py`** — Extract RL state from InstanceInfo
5. **`llumnix/rl/reward.py`** — Reward computation
6. **`llumnix/rl/inference.py`** — Low-latency policy inference wrapper (< 1ms)
7. **`llumnix/rl/baselines.py`** — DQN + random policy baselines

---

## Phase 2: Train RL Agent

```bash
# Train on ShareGPT traces, 4-16 simulated instances
python llumnix/rl/train.py \
    --trace_path ./data/sharegpt.json \
    --n_instances 16 \
    --total_timesteps 1000000 \
    --save_path ./models/ppo_llumnix/

# Also train DQN baseline
python llumnix/rl/train.py \
    --algorithm dqn \
    --total_timesteps 500000 \
    --save_path ./models/dqn_llumnix/
```

Verify training converges (reward increases over time).

---

## Phase 3: Integrate into Llumnix

Modify `llumnix/load_computation.py` to add `RLBasedLoad` class that:
1. Loads trained PPO model
2. Collects state from InstanceInfo
3. Runs policy inference to get virtual usage values
4. Returns load metric compatible with existing dispatch/migration code

Add config options in `llumnix/config/default.py` for RL mode.

Validate: run identical workload with heuristic and RL, ensure no crashes and all requests complete.

---

## Phase 4: Run Experiments

Read `docs/EXPERIMENT_GUIDE.md` for exact commands. Run 5 experiments:

| Exp | Description | Traces | Key Metric |
|-----|------------|--------|------------|
| 1 | Baseline serving | ShareGPT + BurstGPT | Mean & P99 latency |
| 2 | Priority workloads | S-S, Gamma, 10% HP | SLO isolation |
| 3 | Bursty traffic | Gamma, CV=2,4,6,8 | Adaptation |
| 4 | Long-context | L-L distribution | Fragmentation |
| 5 | Generalization | Train: ShareGPT, Test: BurstGPT | Transfer |

Compare 4 policies: **Round-Robin**, **INFaaS++**, **Llumnix-Heuristic**, **Llumnix-RL (ours)**.

Generate plots using `scripts/plot_results.py`.

---

## Phase 5: Midterm Deliverables

### Midterm Report (PDF)
- Use `templates/midterm_report.tex` as template
- 4-5 pages, NeurIPS format
- Content: intro, background, progress, preliminary results, remaining plan
- Compile to PDF: `pdflatex midterm_report.tex`

### Midterm Presentation (if needed)
- ~10 slides covering: background, approach, progress, next steps
- **MUST be `.pptx` format** (editable PowerPoint) with speaker notes in the Notes pane
- Also export a `.pdf` backup
- Use `python-pptx` library to generate programmatically

---

## Phase 6: Complete Experiments + Analysis

- Run remaining experiments
- Ablation studies:
  - Reward weight sensitivity (vary each weight independently)
  - State feature importance (remove one feature at a time)
  - Training duration effect (100K, 250K, 500K, 1M steps)
- Generate all result tables and figures

---

## Phase 7: Final Deliverables

### Final Presentation (PPTX — CRITICAL)
- **MUST be `.pptx` format** (editable PowerPoint), also export `.pdf`
- **20 minutes** presentation duration
- **MUST include speaker notes (演讲稿) in the Notes pane of EVERY slide** — complete speaking script, ~150-200 words per slide, totaling ~20 minutes
- 15-20 slides
- Structure: Background → Llumnix → Problem → Our Approach → MDP → Architecture → Training → Experiments (5) → Ablation → Insights → Limitations → Conclusion
- Must include result figures and comparison tables
- Use `python-pptx` library to generate; see `docs/DELIVERABLES_SPEC.md` for detailed requirements and example code

### Final Report (LaTeX + PDF)
- Use `templates/final_report.tex` as template
- 8-10 pages, NeurIPS format
- Sections: Introduction, Background, Method, Experiments, Results, Discussion, Conclusion
- Must include all 5 experiment results with figures
- Compile to PDF

---

## Important Notes

1. **If GPUs are limited** (< 16), scale down instances proportionally. 4 GPUs is the minimum for meaningful migration experiments. Results are still valid — just state the setup clearly.

2. **If full Llumnix integration is too complex**, the fallback plan is:
   - Build a standalone simulator that faithfully models Llumnix scheduling
   - Train and evaluate RL entirely in simulation
   - Report simulated results (clearly labeled)
   - This is a legitimate approach used in systems RL papers

3. **Reference codebase**: Clone from https://github.com/llumnix-project/llumnix-ray (NOT the original AlibabaPAI/llumnix)

4. **All deliverables must be committed to this repo** in an `output/` directory:
   ```
   output/
   ├── midterm/
   │   ├── midterm_report.pdf
   │   ├── midterm_slides.pptx        # Editable PPT with speaker notes
   │   └── midterm_slides.pdf         # PDF backup
   ├── final/
   │   ├── final_presentation.pptx    # MUST: editable PPT with speaker notes (演讲稿)
   │   ├── final_presentation.pdf     # PDF backup
   │   ├── final_report.tex
   │   └── final_report.pdf
   ├── results/
   │   ├── exp1/ ... exp5/
   │   └── figures/
   └── models/
       ├── ppo_llumnix/
       └── dqn_llumnix/
   ```
