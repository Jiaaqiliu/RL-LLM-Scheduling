# Deliverables Specification

This document defines the **exact deliverables** required for this course project. Each deliverable must be generated and committed to the `output/` directory.

---

## Output Directory Structure

```
output/
├── midterm/
│   ├── midterm_report.tex
│   ├── midterm_report.pdf
│   └── midterm_slides.pdf          # Optional if midterm presentation needed
├── final/
│   ├── final_presentation.pptx     # Or .pdf if PPTX generation not possible
│   ├── final_report.tex
│   ├── final_report.pdf
│   └── figures/                    # All figures used in report
├── results/
│   ├── exp1/ ... exp5/             # Raw experiment data
│   ├── ablation/                   # Ablation study data
│   └── figures/                    # Generated plots
├── models/
│   ├── ppo_llumnix/                # Trained PPO model
│   └── dqn_llumnix/                # Trained DQN model
└── code/
    └── llumnix_rl_patch/           # All modified/new code files
```

---

## Deliverable 1: Midterm Report

### Format
- **Template**: `templates/midterm_report.tex`
- **Style**: NeurIPS 2025 format (`neurips_2025.sty`)
- **Length**: 4-5 pages (excluding references)
- **Output**: `output/midterm/midterm_report.pdf`

### Required Sections

1. **Introduction** (0.5 page)
   - Problem statement: scheduling challenges in LLM serving
   - Our approach: RL-based virtual usage replacement
   - Key insight: minimal modification to Llumnix

2. **Background** (1 page)
   - LLM serving and KV cache management
   - Llumnix system overview (virtual usage, freeness, migration)
   - RL for systems scheduling (prior work)

3. **Progress So Far** (1.5-2 pages)
   - MDP formulation (state/action/reward — with math)
   - Simulation environment implementation
   - RL agent architecture (network diagram)
   - Training pipeline and preliminary results
   - Include: training curve plot, preliminary comparison table

4. **Remaining Work** (0.5 page)
   - Integration with real Llumnix
   - Full experiment suite
   - Analysis and ablations

5. **Challenges** (0.5 page)
   - Technical challenges encountered
   - Design decisions and trade-offs made

6. **References**

### Key Figures for Midterm
- Figure 1: System architecture diagram (Llumnix + RL integration)
- Figure 2: Training curves (reward vs. timesteps)
- Table 1: Preliminary results (even if simulation-only)

---

## Deliverable 2: Final Presentation (PPT) — CRITICAL REQUIREMENTS

### Format
- **File format**: **MUST be `.pptx` (editable PowerPoint)**. Also generate a `.pdf` export as backup.
- **Slides**: 15-20 slides
- **Duration**: **20 minutes** presentation (plan pacing accordingly)
- **Speaker notes**: **MUST include speaker notes (演讲稿) in the Notes pane of EVERY slide.** The speaker notes should be a complete, natural-language script that the presenter can read verbatim or use as a guide. Each slide's notes should cover approximately 1-1.5 minutes of speaking time.
- **Output files**:
  - `output/final/final_presentation.pptx` (PRIMARY — editable, with speaker notes)
  - `output/final/final_presentation.pdf` (SECONDARY — for backup/sharing)

### PPTX Generation

Use the `python-pptx` library to generate the `.pptx` file programmatically:

```bash
pip install python-pptx
```

```python
from pptx import Presentation
from pptx.util import Inches, Pt

prs = Presentation()

# For each slide:
slide = prs.slides.add_slide(prs.slide_layouts[1])  # title + content
slide.shapes.title.text = "Slide Title"
slide.placeholders[1].text = "Bullet content"

# ADD SPEAKER NOTES (this is mandatory):
notes_slide = slide.notes_slide
notes_slide.notes_text_frame.text = """
This is the speaker script for this slide.
It should be 1-1.5 minutes of natural speaking content.
Cover the key points, transitions, and what to emphasize.
"""

prs.save("final_presentation.pptx")
```

### Speaker Notes Guidelines

Each slide's speaker notes MUST:
1. Be a **complete speaking script** (not just bullet points)
2. Cover **1-1.5 minutes** of speaking time (~150-200 words per slide)
3. Include **transition phrases** to the next slide (e.g., "Now let's look at...")
4. Highlight what to **emphasize** when presenting
5. Be written in **English** (matching the slides)
6. Total speaking time across all slides: **~20 minutes**

Example speaker notes for a slide:

```
"Good afternoon everyone. Today we're presenting our work on improving
LLM serving scheduling with reinforcement learning.

As you know, serving large language models like GPT and LLaMA at scale
is becoming increasingly important. But there's a fundamental challenge:
unlike traditional DNN inference where request sizes are fixed, LLM
requests have unpredictable output lengths. This means the GPU memory
consumption — specifically the KV cache — grows dynamically during
generation, and we don't know in advance how much memory a request will
need.

This unpredictability creates three major problems that I'll walk you
through on the next slide..."
```

### Required Slide Structure

| Slide # | Title | Content | Notes Duration |
|---------|-------|---------|---------------|
| 1 | Title Slide | Project title, team members, course | 0.5 min (greeting + intro) |
| 2 | Outline | Agenda for the presentation | 0.5 min |
| 3 | Background: LLM Serving | Key challenges (unpredictable lengths, KV cache, preemptions) | 1.5 min |
| 4 | The Problem: Static Scheduling | Why heuristics fail (static, myopic, fixed tradeoffs) | 1 min |
| 5 | Baseline: Llumnix | Virtual usage concept, freeness, 4 heuristic rules | 1.5 min |
| 6 | Our Proposal | RL replaces CalcVirtualUsage(), minimal modification | 1 min |
| 7 | MDP Formulation | State / Action / Reward design (with visuals) | 1.5 min |
| 8 | Architecture | System diagram showing RL integration point | 1 min |
| 9 | Policy Network | Multi-head PPO architecture diagram | 1 min |
| 10 | Training Pipeline | Two-phase: behavior cloning + PPO fine-tuning | 1 min |
| 11 | Exp 1: Baseline Serving | Results plots (latency comparison) | 1.5 min |
| 12 | Exp 2: Priority | SLO isolation results | 1 min |
| 13 | Exp 3: Bursty Traffic | Adaptation under load spikes | 1 min |
| 14 | Exp 4: Long-Context | Memory fragmentation results | 1 min |
| 15 | Exp 5: Generalization | Cross-distribution transfer | 1 min |
| 16 | Ablation Studies | Reward weights, state features, training duration | 1 min |
| 17 | Key Insights | What we learned, when RL helps/hurts | 1 min |
| 18 | Limitations & Future Work | Current gaps, potential extensions | 1 min |
| 19 | Conclusion | Summary of contributions | 0.5 min |
| 20 | Q&A | Thank you + questions | 0.5 min |
| | | **Total** | **~20 min** |

### Design Guidelines
- Use clean, professional design (dark blue headers, orange accents for emphasis)
- Include result plots directly (not tables of numbers)
- Each slide should have a clear takeaway message
- Use code snippets sparingly (only for CalcVirtualUsage comparison)
- Ensure text is large enough to read from the back of a classroom (min 18pt for body text)

---

## Deliverable 3: Final Report (LaTeX + PDF)

### Format
- **Template**: `templates/final_report.tex`
- **Style**: NeurIPS 2025 format
- **Length**: 8-10 pages (excluding references and appendix)
- **Output**: `output/final/final_report.tex` + `output/final/final_report.pdf`

### Required Sections

1. **Abstract** (0.25 page)
   - One paragraph: problem, approach, key results

2. **Introduction** (1 page)
   - LLM serving challenges
   - Llumnix and its limitations
   - Our contribution: RL-based scheduling
   - Results preview

3. **Background and Related Work** (1.5 pages)
   - LLM serving and KV cache management
   - Llumnix: virtual usage and freeness
   - RL for systems (Pensieve, Decima, etc.)
   - Our positioning

4. **Method** (2 pages)
   - 4.1 MDP Formulation
     - State space (with mathematical notation)
     - Action space
     - Reward function (with equation)
   - 4.2 Policy Architecture
     - Network design (with figure)
     - Instance encoder + attention + cluster encoder
   - 4.3 Training Procedure
     - Phase 1: Behavior cloning
     - Phase 2: PPO fine-tuning
     - Hyperparameters table
   - 4.4 Integration with Llumnix
     - What changes, what stays the same
     - RLBasedLoad class
     - Inference latency requirements

5. **Experimental Setup** (1 page)
   - Hardware description
   - Model and dataset details
   - Baselines (Round-Robin, INFaaS++, Llumnix-Heuristic)
   - Metrics (end-to-end, prefill, decode latency; preemption loss; fragmentation)
   - Trace descriptions (Table 1 from Llumnix paper format)

6. **Results** (2 pages)
   - 6.1 Baseline Serving (Exp 1) — with Figure
   - 6.2 Priority Workloads (Exp 2) — with Figure
   - 6.3 Bursty Traffic (Exp 3) — with Figure
   - 6.4 Long-Context (Exp 4) — with Figure
   - 6.5 Generalization (Exp 5) — with Figure
   - 6.6 Ablation Studies — with Figure/Table

7. **Discussion** (0.5 page)
   - When RL outperforms heuristic (and why)
   - When RL underperforms (and why)
   - Training cost vs. deployment benefit
   - Inference overhead analysis

8. **Limitations and Future Work** (0.5 page)
   - Simulation vs. real deployment gap
   - Scalability concerns
   - Multi-objective RL
   - Direct migration actions (Option B from proposal)

9. **Conclusion** (0.25 page)

10. **References**

### Required Figures
- Figure 1: System architecture (Llumnix + RL)
- Figure 2: Policy network architecture
- Figure 3: Training curves
- Figure 4: Exp 1 results (7-column format matching Llumnix paper)
- Figure 5: Exp 2 results (priority comparison)
- Figure 6: Exp 3 results (bursty traffic)
- Figure 7: Exp 4 results (long-context / fragmentation)
- Figure 8: Exp 5 results (generalization)
- Figure 9: Ablation results (reward weights + state features)

### Required Tables
- Table 1: Sequence length distributions used
- Table 2: Hyperparameters
- Table 3: Summary comparison (best results across all experiments)

---

## Quality Checklist

Before finalizing any deliverable, verify:

- [ ] All figures are high-resolution (300 DPI minimum)
- [ ] All axes are labeled with units
- [ ] All baselines are consistently labeled across figures
- [ ] LaTeX compiles without errors or warnings
- [ ] References are complete and correctly formatted
- [ ] No placeholder text (e.g., "TODO", "TBD")
- [ ] Team member names and affiliations are correct
- [ ] Page count is within limits
- [ ] PDF is readable and properly formatted

---

## If Experiments Run in Simulation Only

If full Llumnix integration with real GPUs is not achievable, the deliverables should:

1. **Clearly state** that results are from simulation (not real GPU serving)
2. **Describe the simulator** faithfully (what timing models are used, what is approximated)
3. **Discuss the simulation-to-real gap** as a limitation
4. **Still present all 5 experiments** using the simulator
5. **Provide the integration code** as "ready to deploy" even if not tested end-to-end

This is a valid and common approach in systems RL research. Papers like Pensieve (SIGCOMM 2017) and Decima (SIGCOMM 2019) also use simulation for training and partial real-world validation.
