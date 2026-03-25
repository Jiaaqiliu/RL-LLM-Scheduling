"""
Synthetic trace generator for RL-LLM-Scheduling experiments.
Generates request traces with controlled parameters matching Llumnix paper settings.
"""

import numpy as np
import json
import argparse
from pathlib import Path


# Distribution configs matching Llumnix paper Table 1
DISTRIBUTIONS = {
    "S-S": {"input_mean": 128, "output_mean": 128},
    "M-M": {"input_mean": 256, "output_mean": 256},
    "L-L": {"input_mean": 512, "output_mean": 512},
    "S-L": {"input_mean": 128, "output_mean": 512},
    "L-S": {"input_mean": 512, "output_mean": 128},
}


def sample_powerlaw(mean, min_val=4, max_val=6000):
    """Sample from a power-law distribution with given mean."""
    # Use a log-normal approximation for power-law-like distribution
    sigma = 1.0
    mu = np.log(mean) - sigma**2 / 2
    val = np.random.lognormal(mu, sigma)
    return int(np.clip(val, min_val, max_val))


def generate_arrivals(n_requests, qps, distribution="poisson", cv=1.0):
    """Generate request arrival times."""
    if distribution == "poisson":
        inter_arrivals = np.random.exponential(1.0 / qps, n_requests)
    elif distribution == "gamma":
        shape = 1.0 / (cv ** 2)
        scale = 1.0 / (qps * shape)
        inter_arrivals = np.random.gamma(shape, scale, n_requests)
    elif distribution == "uniform":
        inter_arrivals = np.full(n_requests, 1.0 / qps)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return np.cumsum(inter_arrivals)


def generate_trace(
    n_requests=10000,
    qps=7.5,
    distribution="poisson",
    cv=1.0,
    input_mean=256,
    output_mean=256,
    high_priority_ratio=0.0,
    seed=42,
):
    """Generate a complete request trace."""
    np.random.seed(seed)

    arrival_times = generate_arrivals(n_requests, qps, distribution, cv)

    traces = []
    for i in range(n_requests):
        input_len = sample_powerlaw(input_mean)
        output_len = sample_powerlaw(output_mean)
        priority = "high" if np.random.random() < high_priority_ratio else "normal"

        traces.append({
            "request_id": i,
            "arrival_time": float(arrival_times[i]),
            "input_length": int(input_len),
            "output_length": int(output_len),
            "priority": priority,
        })

    return traces


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic request traces")
    parser.add_argument("--n_requests", type=int, default=10000)
    parser.add_argument("--qps", type=float, default=7.5)
    parser.add_argument("--distribution", type=str, default="poisson",
                        choices=["poisson", "gamma", "uniform"])
    parser.add_argument("--cv", type=float, default=1.0,
                        help="Coefficient of variation (for gamma distribution)")
    parser.add_argument("--dist_type", type=str, default="M-M",
                        choices=list(DISTRIBUTIONS.keys()),
                        help="Sequence length distribution type")
    parser.add_argument("--high_priority_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./data/trace.json")
    args = parser.parse_args()

    dist = DISTRIBUTIONS[args.dist_type]

    traces = generate_trace(
        n_requests=args.n_requests,
        qps=args.qps,
        distribution=args.distribution,
        cv=args.cv,
        input_mean=dist["input_mean"],
        output_mean=dist["output_mean"],
        high_priority_ratio=args.high_priority_ratio,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(traces, f, indent=2)

    # Print statistics
    input_lens = [t["input_length"] for t in traces]
    output_lens = [t["output_length"] for t in traces]
    hp_count = sum(1 for t in traces if t["priority"] == "high")

    print(f"Generated {len(traces)} requests")
    print(f"  Distribution: {args.dist_type} ({args.distribution}, CV={args.cv})")
    print(f"  QPS: {args.qps}")
    print(f"  Input length:  mean={np.mean(input_lens):.0f}, P50={np.median(input_lens):.0f}, P99={np.percentile(input_lens, 99):.0f}")
    print(f"  Output length: mean={np.mean(output_lens):.0f}, P50={np.median(output_lens):.0f}, P99={np.percentile(output_lens, 99):.0f}")
    print(f"  High priority: {hp_count} ({hp_count/len(traces)*100:.1f}%)")
    print(f"  Duration: {traces[-1]['arrival_time']:.1f}s")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
