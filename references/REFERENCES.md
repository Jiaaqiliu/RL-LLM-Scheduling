# References

## Primary Reference

### Paper
- **Title**: Llumnix: Dynamic Scheduling for Large Language Model Serving
- **Authors**: Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, Wei Lin
- **Venue**: OSDI 2024 (18th USENIX Symposium on Operating Systems Design and Implementation)
- **arXiv**: https://arxiv.org/abs/2406.03243
- **Pages**: 173-191

### Code
- **Repository**: https://github.com/llumnix-project/llumnix-ray
- **Branch**: main
- **Language**: Python
- **Key dependency**: vLLM 0.6.3.post1 + Ray 2.45.0-2.47.1

---

## Other References Cited in Our Proposal

### RL for Systems
1. **Resource Management with Deep RL** (Mao et al., HotNets 2016)
   - First work applying RL to cluster resource management
   - Showed RL can outperform hand-tuned heuristics

2. **Neural Adaptive Video Streaming with Pensieve** (Mao et al., SIGCOMM 2017)
   - RL for adaptive bitrate streaming
   - Key precedent: replacing heuristic with learned policy in a systems context
   - arXiv: https://arxiv.org/abs/1720.01955 (see actual published version)

3. **Proximal Policy Optimization (PPO)** (Schulman et al., 2017)
   - Our primary RL algorithm
   - arXiv: https://arxiv.org/abs/1707.06347

### LLM Serving
4. **vLLM: Efficient Memory Management with PagedAttention** (Kwon et al., SOSP 2023)
   - The inference engine underlying Llumnix
   - GitHub: https://github.com/vllm-project/vllm

### RL Libraries
5. **Stable-Baselines3**: https://github.com/DLR-RM/stable-baselines3
   - PPO and DQN implementations we use

6. **Gymnasium**: https://github.com/Farama-Foundation/Gymnasium
   - RL environment interface

---

## Datasets

### ShareGPT
- **URL**: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
- **File**: `ShareGPT_V3_unfiltered_cleaned_split.json`
- **Description**: Real ChatGPT conversation dataset with diverse input/output lengths

### BurstGPT
- **URL**: https://github.com/HPMLL/BurstGPT
- **Description**: GPT-4 conversation traces with bursty arrival patterns
- **Used for**: Cross-distribution generalization experiments

---

## Models

### LLaMA-2-7B
- **HuggingFace**: https://huggingface.co/meta-llama/Llama-2-7b-hf
- **Size**: ~13GB (FP16)
- **GPU requirement**: 1x A10 (24GB) per instance
- **Requires**: HuggingFace access token (request access from Meta)
