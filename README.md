# RL-based Dynamic Scheduling for LLM Serving

> **Course Project** — 790-199 (Instructor: Sishuai), University of Virginia
> **Team**: Yuxuan Bai, Jiaqi Liu, Peng Xia

## What This Repo Is

This repository contains **all context, instructions, and templates** needed for an AI Agent on a GPU server to execute this course project end-to-end. The Agent should read every file in this repo and then carry out the full implementation, experiments, and deliverable generation.

## Project in One Sentence

Replace Llumnix's hand-crafted heuristic scheduling policy (`CalcVirtualUsage()`) with a **Reinforcement Learning (RL)-based policy** for LLM serving, and experimentally demonstrate improvements in latency, preemption reduction, and memory fragmentation.

## Repo Structure

```
.
├── README.md                          # This file — start here
├── AGENT_INSTRUCTIONS.md              # CRITICAL: Step-by-step instructions for the Agent
├── docs/
│   ├── BACKGROUND.md                  # Project background and motivation
│   ├── REFERENCE_PAPER_SUMMARY.md     # Summary of Llumnix (OSDI 2024) paper
│   ├── CODEBASE_ANALYSIS.md           # Analysis of llumnix-ray codebase
│   ├── IMPLEMENTATION_GUIDE.md        # Detailed implementation plan (5 tasks)
│   ├── EXPERIMENT_GUIDE.md            # Experiment design and execution commands
│   └── DELIVERABLES_SPEC.md           # Exact specifications for all deliverables
├── proposal/
│   ├── proposal.tex                   # Our submitted proposal (LaTeX source)
│   └── neurips_2025.sty               # LaTeX style file
├── references/
│   └── REFERENCES.md                  # Links to reference paper, code, datasets
├── templates/
│   ├── midterm_report.tex             # LaTeX template for midterm report
│   └── final_report.tex               # LaTeX template for final report
└── scripts/
    ├── setup_env.sh                   # Server environment setup script
    ├── generate_traces.py             # Synthetic trace generation
    └── plot_results.py                # Result visualization
```

## How to Use This Repo (For the Agent)

1. **Read `AGENT_INSTRUCTIONS.md` first** — it is the master control document
2. Read all files in `docs/` for full context
3. Follow the step-by-step plan in `AGENT_INSTRUCTIONS.md`
4. Generate all deliverables as specified in `docs/DELIVERABLES_SPEC.md`

## Reference Links

- **Reference Paper**: Llumnix: Dynamic Scheduling for Large Language Model Serving (OSDI 2024)
  - arXiv: https://arxiv.org/abs/2406.03243
- **Reference Codebase**: https://github.com/llumnix-project/llumnix-ray
- **ShareGPT Dataset**: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
