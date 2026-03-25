# Codebase Analysis: llumnix-ray

**Repository**: https://github.com/llumnix-project/llumnix-ray
**This is the Ray-based variant** of the Llumnix project (the version we modify).

---

## Important: CalcVirtualUsage() Does NOT Exist as a Single Function

In the paper, Algorithm 1 describes `CalcVirtualUsage()` as a clean function. In the actual codebase, this logic is **distributed across multiple classes** in the load computation system. Understanding this mapping is critical.

---

## Key Files and Their Roles

### 1. Load Computation — `llumnix/load_computation.py`
**THIS IS THE PRIMARY FILE TO MODIFY.**

Contains load metric classes that serve as the "virtual usage" equivalent:

```python
class RemainingStepsLoad(InstanceLoad):
    """Primary dispatch metric. Equivalent to 'freeness' in the paper."""
    def compute_load(self, instance_info):
        num_available = instance_info.num_available_gpu_blocks
        reserved = num_waiting_requests * instance_info.num_blocks_first_waiting_request
        remaining = (num_available - reserved) / max(num_requests, 1)
        # Higher remaining = more free = lower load
        return remaining

    def is_busy(self):
        return self._load <= BUSY_THRESHOLD  # default: 0.0

class KvBlocksRatioLoad(InstanceLoad):
    """Demand-based metric. Maps to virtual usage inflation for queuing requests."""
    def compute_load(self, instance_info):
        demand = (num_used_blocks + num_blocks_all_waiting) / num_total_blocks
        return demand

    def is_busy(self):
        return self._load >= BUSY_THRESHOLD  # default: 0.8

class AdaptiveDecodeBatchLoad(InstanceLoad):
    """Decode throughput metric."""
    # Based on decode batch size vs compute-bound threshold

class MissWaitingTokensLoad(InstanceLoad):
    """KV cache locality metric for migration decisions."""
    # Counts tokens in waiting requests that miss KV cache
```

### 2. Instance Info — `llumnix/instance_info.py`
**Dataclass containing all observable state per instance.**

Key fields (lines 44-102):
```python
@dataclass
class InstanceInfo:
    # GPU block counts
    num_total_gpu_blocks: int = 0
    num_free_gpu_blocks: int = 0
    num_used_gpu_blocks: int = 0
    num_watermark_blocks: int = 0

    # Derived
    num_available_gpu_blocks: int = 0        # total - used
    num_available_gpu_blocks_waiting: int = 0 # available - waiting_blocks

    # Request counts
    num_running_requests: int = 0
    num_waiting_requests: int = 0

    # Waiting request details
    num_blocks_first_waiting_request: int = 0
    num_blocks_all_waiting_requests: int = 0

    # Load metrics (computed by InstanceLoadCalculator)
    instance_load_dispatch_scale: float = 0
    instance_load_migrate: float = 0

    # Timing
    latency_of_per_token: float = 0

    # Migration tracking
    num_seqs_in_migration: int = 0
```

### 3. Dispatch Policy — `llumnix/global_scheduler/dispatch_policy.py`
**Selects which instance receives a new request.**

Available policies:
- `Flood`: All to one (testing)
- `Balanced`: Min queued requests
- `Load`: **Lowest load metric** (this is what we primarily affect with RL)
- `Queue`: Smallest waiting queue
- `RoundRobin`: Cycle through
- `CacheAware`: KV cache locality

The `Load` policy calls `instance_info.instance_load_dispatch_scale` which is computed by the load metric class.

### 4. Migration Policy — `llumnix/global_scheduler/migration_policy.py`
**Pairs overloaded ↔ underloaded instances for request migration.**

- `BalancedPairMigrationPolicy`: Uses load metric to identify source (overloaded) and destination (underloaded), pairs them greedily
- `DefragPairMigrationPolicy`: Simple pairing without load checks
- `FailoverPairMigrationPolicy`: Routes from unhealthy to healthy

### 5. Migration Scheduler — `llumnix/global_scheduler/migration_scheduler.py`
**Orchestrates migration decisions.**

Migration types:
- `NEUTRAL_LOAD_BALANCE`: Standard load balancing
- `DD_LOAD_BALANCE`: Decode-to-decode
- `PD_MIGRATION`: Prefill-to-decode
- `PRE_STOP_MIGRATION`: Failover
- `DYNAMIC_P_TO_D`: Dynamic prefill to decode

### 6. Global Scheduler — `llumnix/global_scheduler/global_scheduler.py`
**Top-level coordinator.**

Calls dispatch_scheduler and migration_scheduler. Entry point for all scheduling decisions.

### 7. Manager — `llumnix/manager.py` (598 lines)
**Ray actor that manages the cluster.**

Handles:
- New request routing
- Periodic migration triggering
- Instance health monitoring
- Auto-scaling

### 8. vLLM Backend — `llumnix/backends/vllm/scheduler.py`
**Wraps vLLM's scheduler, extracts InstanceInfo.**

`_get_instance_info()` method collects:
- Block allocator state
- Sequence queue information
- Request timing

### 9. Config — `llumnix/config/default.py`
**YACS-based configuration.**

Key config options:
```python
MANAGER.DISPATCH_POLICY = 'load'          # or 'balanced', 'queue', 'rr', etc.
MANAGER.DISPATCH_LOAD_METRIC = 'remaining_steps'  # or 'kv_blocks_ratio', etc.
MANAGER.PAIR_MIGRATION_POLICY = 'balanced'
INSTANCE.MIGRATION_BACKEND = 'rayrpc'     # or 'gloo', 'nccl'
INSTANCE.ENABLE_DEFRAG = False
```

---

## Code Flow: How a Request is Scheduled

### Dispatch (new request arrives)
```
Manager.handle_new_request()
  → GlobalScheduler.dispatch()
    → DispatchScheduler.dispatch()
      → Filter: UnhealthyUnitFilter.filter()
      → Filter: MetricBasedFilter.filter()    # Uses load_metric.is_busy()
      → Select: LoadDispatchPolicy.select()   # Picks instance with lowest load
        → instance_info.instance_load_dispatch_scale  # THE VALUE WE REPLACE
```

### Migration (periodic check)
```
Manager.scale_and_migration() (called periodically)
  → GlobalScheduler.check_migration()
    → MigrationScheduler.check_migration()
      → MigrationPolicy.pair()
        → BalancedPairMigrationPolicy.pair()
          → Sort instances by instance_info.instance_load_migrate
          → Pair overloaded (high load) with underloaded (low load)
      → For each pair: llumlet.migrate_out() → migration_coordinator → KV transfer
```

### Load Update (from instance to manager)
```
Llumlet.step() (after each vLLM iteration)
  → backend._get_instance_info()             # Raw metrics from vLLM
  → InstanceLoadCalculator.compute_instance_load()
    → RemainingStepsLoad.compute_load()       # THE COMPUTATION WE REPLACE
  → Report to Manager (via Ray RPC)
```

---

## RL Integration Point

The cleanest integration is to add a new load metric class:

```python
# In llumnix/load_computation.py

class RLBasedLoad(InstanceLoad):
    def __init__(self, model_path):
        self.policy = load_trained_model(model_path)

    def compute_load(self, instance_info):
        state = extract_state(instance_info)
        action = self.policy.predict(state)
        # action determines the "virtual usage" / load value
        return action_to_load(action)

    def is_busy(self):
        return self._load > THRESHOLD
```

Then register it in the config:
```python
# In llumnix/config/default.py
MANAGER.DISPATCH_LOAD_METRIC = 'rl_based'  # New option
```

---

## Dependencies

From `requirements/requirements_vllm.txt`:
```
vllm == 0.6.3.post1
ray >= 2.45.0, <= 2.47.1
pyarrow
aiohttp
scipy
pandas
matplotlib
func_timeout
pyyaml
yacs
pyzmq
setuptools_scm == 7.1.0
uvloop
```

Python: 3.9 - 3.12.3
