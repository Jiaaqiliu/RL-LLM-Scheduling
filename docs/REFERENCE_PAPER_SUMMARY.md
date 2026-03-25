# Reference Paper Summary: Llumnix (OSDI 2024)

**Title**: Llumnix: Dynamic Scheduling for Large Language Model Serving
**Authors**: Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, Wei Lin (Alibaba Group)
**Venue**: 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 2024)
**arXiv**: https://arxiv.org/abs/2406.03243

---

## Key Algorithm: Virtual Usage and Freeness Calculation

This is Algorithm 1 from the paper — the exact logic we are replacing with RL:

```
Function CalcVirtualUsage(req, instance):
    if req.isQueuing then
        if req.isHeadOfLine then
            return req.demand          # Inflate queuing request to its demand
        return 0                       # Non-head queuing requests contribute 0
    if req.isFake then
        return infinity                # Terminating instance: drain all requests
    return req.physicalUsage + GetHeadroom(req.priority, instance)

Function GetHeadroom(p, instance):
    return headroomForPriority[p] / instance.numRequests[p]
    # headroom is obtained from offline profiling
    # For normal priority: headroom = 0
    # For high priority: headroom = memory needed to preserve ideal decode speed

Function CalcFreeness(instance):
    if instance.isTerminating then
        addFakeReq(instance.requests)  # Add fake request with infinite usage
    totalVirtualUsage = 0
    for req in instance.requests do
        totalVirtualUsage += CalcVirtualUsage(req, instance)
    freeness = (instance.M - totalVirtualUsage) / instance.B
    return freeness
```

### How Freeness Drives Decisions

**Dispatching**: New requests go to instance with highest freeness.
- Freeness can be negative (when virtual usage > physical capacity), marking instance as overloaded.

**Migration**: Triggered periodically.
1. Select source instances: freeness < lower_threshold
2. Select destination instances: freeness > upper_threshold
3. Pair them (lowest freeness source ↔ highest freeness destination)
4. Llumlets decide which requests to migrate (prefer low-priority, short sequences)

**Auto-scaling**:
- Scale up: average freeness < x for a period
- Scale down: average freeness > y for a period
- Default range: [10, 60]

---

## Experimental Setup (What We Should Match)

### Hardware
- 16-GPU cluster with 4 GPU VMs on Alibaba Cloud
- Each VM: 4 NVIDIA A10 GPUs (24GB), 128 vCPUs, 752 GB memory, 64 Gb/s network
- VM type: `ecs.gn7i-c32g1.32xlarge`

### Models
- LLaMA-7B: 1 GPU per instance, 16-bit precision
- LLaMA-30B: 4 GPUs per instance (tensor parallel), 16-bit precision
- vLLM with max sequence length 2k

### Traces
Request traces with controlled arrival rates and sequence length distributions:

| Distribution | Input Mean | Input P50 | Input P99 | Output Mean | Output P50 | Output P99 |
|-------------|-----------|----------|----------|------------|-----------|-----------|
| ShareGPT | 306 | 74 | 3388 | 500 | 487 | 1234 |
| BurstGPT | 830 | 582 | 3549 | 243 | 434 | 964 |
| Short (S) | 128 | 38 | 1464 | - | - | - |
| Medium (M) | 256 | 32 | 4208 | - | - | - |
| Long (L) | 512 | 55 | 5166 | - | - | - |

Trace combinations: S-S, M-M, L-L, S-L, L-S (input-output distribution pairs).

Arrival distributions: Poisson (varying request rate) and Gamma (varying CV for burstiness).
Each trace: 10,000 requests.

### Baselines
1. **Round-Robin**: Distribute requests evenly across instances
2. **INFaaS++**: Optimized version of INFaaS with load-balancing dispatch and auto-scaling
3. **Llumnix-base**: Llumnix without priority support (all requests same priority)
4. **Llumnix**: Full Llumnix with virtual usage heuristic

### Key Metrics
- **End-to-end request latency**: Time from request arrival to completion (mean, P99)
- **Prefill latency**: Time to first token / TTFT (mean, P99)
- **Decode latency**: Per-token generation latency (mean, P99)
- **Preemption loss**: Extra queuing + recompute time due to preemptions
- **Fragmentation ratio**: Wasted memory due to fragmentation over time
- **Resource cost**: Average number of instances used

---

## Key Results from Paper

1. **Serving performance**: Llumnix outperforms INFaaS++ in end-to-end latency by up to 2x (mean) and 2.9x (P99). Prefill improvements up to 7.7x (P99/mean) and 14.8x (P99).

2. **Preemption reduction**: 70.4% average reduction across all experiments, translating to 1.3 seconds average latency reduction.

3. **Priority support**: 2.9x to 8.6x improvement for high-priority request mean latency.

4. **Auto-scaling**: Up to 36% cost savings with similar P99 latency.

5. **Migration overhead**: Near-zero downtime (20-30ms, constant regardless of sequence length). Only 1% decode speed difference during migration.

6. **Scalability**: Near-zero scheduling stalls even with 64 instances and high request rates.

---

## Implementation Details

- **3,300 lines** of Python code
- Built as a **standalone scheduling layer** on top of vLLM
- Uses **Ray** for distributed actor coordination
- **Gloo** for KV cache transfer (not NCCL, because concurrent NCCL calls are unsafe)
- **Block fusion**: Copies non-contiguous vLLM KV cache blocks to contiguous CPU buffer before transfer
- **Handshake protocol**: Pre-allocates destination blocks before each migration stage, handles edge cases (OOM, request completion during migration)
