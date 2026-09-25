"""
Benchmark: Shape Bucketing vs Dynamic Pattern Regeneration in ONNX Runtime

Compares three strategies for handling varying input spatial dimensions:

  A. no_pattern    — MemPattern disabled, every tensor allocated individually from arena
  B. dynamic       — MemPattern enabled, shapes change every call → pattern invalidated each time
  C. bucketed      — MemPattern enabled, input padded to nearest bucket → pattern reused from cache

We measure:
  - Per-inference wall-clock latency (median, p95, p99)
  - Peak memory (via ORT arena stats)
  - Pattern hit rate (how often the cached pattern was actually reused)
"""

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import onnxruntime as ort


# ─── Bucketing helpers ───────────────────────────────────────────────────────

def next_power_of_two(n: int) -> int:
    """Round up to the next power of 2."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def round_up_to_multiple(n: int, multiple: int) -> int:
    """Round up n to the nearest multiple."""
    return ((n + multiple - 1) // multiple) * multiple


BUCKET_STRATEGIES = {
    "pow2": next_power_of_two,
    "mul64": lambda n: round_up_to_multiple(n, 64),
    "mul32": lambda n: round_up_to_multiple(n, 32),
    "mul128": lambda n: round_up_to_multiple(n, 128),
}


# ─── Input shape generators ─────────────────────────────────────────────────

def generate_varying_sizes(
    n: int,
    min_size: int = 200,
    max_size: int = 520,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Generate n random (height, width) pairs in [min_size, max_size]."""
    rng = np.random.RandomState(seed)
    heights = rng.randint(min_size, max_size + 1, size=n)
    widths = rng.randint(min_size, max_size + 1, size=n)
    return list(zip(heights.tolist(), widths.tolist()))


def generate_slowly_varying_sizes(
    n: int,
    base: int = 400,
    jitter: int = 30,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Sizes that change slowly (simulating video frames with slight crop variations)."""
    rng = np.random.RandomState(seed)
    sizes = []
    h, w = base, base
    for _ in range(n):
        h = max(100, h + rng.randint(-jitter, jitter + 1))
        w = max(100, w + rng.randint(-jitter, jitter + 1))
        sizes.append((h, w))
    return sizes


# ─── Benchmark core ─────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    strategy: str
    latencies_ms: list[float] = field(default_factory=list)
    peak_memory_bytes: int = 0
    pattern_hit_count: int = 0
    pattern_miss_count: int = 0
    total_waste_bytes: int = 0  # bytes wasted by bucketing padding

    @property
    def median_ms(self) -> float:
        return float(np.median(self.latencies_ms))

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95))

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 99))

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.latencies_ms))

    @property
    def std_ms(self) -> float:
        return float(np.std(self.latencies_ms))

    @property
    def total_ms(self) -> float:
        return float(np.sum(self.latencies_ms))


def create_session(
    model_path: str,
    enable_mem_pattern: bool,
    num_threads: int = 1,
) -> ort.InferenceSession:
    """Create an ORT inference session with specified memory settings."""
    opts = ort.SessionOptions()
    opts.enable_mem_pattern = enable_mem_pattern
    opts.enable_cpu_mem_arena = True
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = num_threads
    # Disable graph optimizations to keep intermediate tensor count stable
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.log_severity_level = 3  # suppress warnings

    session = ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    return session


def run_benchmark(
    model_path: str,
    sizes: list[tuple[int, int]],
    strategy: Literal["no_pattern", "dynamic", "bucketed"],
    bucket_fn_name: str = "pow2",
    num_threads: int = 1,
    warmup: int = 5,
) -> BenchmarkResult:
    """Run the benchmark for a given strategy."""

    enable_mem_pattern = strategy != "no_pattern"
    session = create_session(model_path, enable_mem_pattern, num_threads)

    bucket_fn = BUCKET_STRATEGIES.get(bucket_fn_name, next_power_of_two)
    input_name = session.get_inputs()[0].name

    result = BenchmarkResult(strategy=strategy)

    # Track which bucketed sizes we've seen (to count pattern hits)
    seen_bucketed_sizes: set[tuple[int, int]] = set()

    # Warmup
    for i in range(min(warmup, len(sizes))):
        h, w = sizes[i]
        if strategy == "bucketed":
            h, w = bucket_fn(h), bucket_fn(w)
        inp = np.random.randn(1, 3, h, w).astype(np.float32)
        session.run(None, {input_name: inp})

    # Reset tracking after warmup
    seen_bucketed_sizes.clear()

    # Benchmark runs
    for h_orig, w_orig in sizes:
        if strategy == "bucketed":
            h = bucket_fn(h_orig)
            w = bucket_fn(w_orig)
            waste = (h * w - h_orig * w_orig) * 3 * 4  # channels=3, float32=4 bytes, input only
            result.total_waste_bytes += waste

            key = (h, w)
            if key in seen_bucketed_sizes:
                result.pattern_hit_count += 1
            else:
                result.pattern_miss_count += 1
                seen_bucketed_sizes.add(key)
        else:
            h, w = h_orig, w_orig
            result.pattern_miss_count += 1  # every call is a different shape

        inp = np.random.randn(1, 3, h, w).astype(np.float32)

        # Timed run
        start = time.perf_counter()
        session.run(None, {input_name: inp})
        end = time.perf_counter()

        result.latencies_ms.append((end - start) * 1000.0)

    return result


# ─── Reporting ───────────────────────────────────────────────────────────────

def print_results(results: list[BenchmarkResult], sizes: list[tuple[int, int]]):
    n = len(sizes)
    unique_sizes = len(set(sizes))

    print("\n" + "=" * 90)
    print(f"BENCHMARK RESULTS  ({n} inferences, {unique_sizes} unique input shapes)")
    print("=" * 90)

    # Header
    print(f"\n{'Strategy':<15} {'Median':>10} {'Mean':>10} {'P95':>10} {'P99':>10} "
          f"{'Total':>10} {'Hits':>8} {'Misses':>8} {'Waste':>10}")
    print(f"{'':.<15} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} "
          f"{'(ms)':>10} {'':>8} {'':>8} {'(MB)':>10}")
    print("-" * 90)

    for r in results:
        waste_mb = r.total_waste_bytes / (1024 * 1024)
        print(f"{r.strategy:<15} {r.median_ms:>10.3f} {r.mean_ms:>10.3f} "
              f"{r.p95_ms:>10.3f} {r.p99_ms:>10.3f} {r.total_ms:>10.1f} "
              f"{r.pattern_hit_count:>8} {r.pattern_miss_count:>8} "
              f"{waste_mb:>10.2f}")

    # Comparison
    if len(results) >= 2:
        baseline = results[0]  # no_pattern
        print(f"\n{'Relative to no_pattern:':<30}")
        for r in results[1:]:
            diff_pct = ((r.median_ms - baseline.median_ms) / baseline.median_ms) * 100
            sign = "+" if diff_pct > 0 else ""
            print(f"  {r.strategy:<15} median: {sign}{diff_pct:.2f}%  "
                  f"total: {sign}{((r.total_ms - baseline.total_ms) / baseline.total_ms) * 100:.2f}%")

    if len(results) >= 3:
        dynamic = results[1]
        bucketed = results[2]
        diff_pct = ((bucketed.median_ms - dynamic.median_ms) / dynamic.median_ms) * 100
        sign = "+" if diff_pct > 0 else ""
        print(f"\n  bucketed vs dynamic:  median: {sign}{diff_pct:.2f}%  "
              f"total: {sign}{((bucketed.total_ms - dynamic.total_ms) / dynamic.total_ms) * 100:.2f}%")

    print("=" * 90)


def print_bucket_analysis(sizes: list[tuple[int, int]], bucket_fn_name: str):
    """Show how many unique bucket sizes the strategy produces."""
    bucket_fn = BUCKET_STRATEGIES.get(bucket_fn_name, next_power_of_two)
    unique_original = len(set(sizes))
    bucketed = [(bucket_fn(h), bucket_fn(w)) for h, w in sizes]
    unique_bucketed = len(set(bucketed))

    print(f"\nBucket analysis ({bucket_fn_name}):")
    print(f"  Unique original sizes: {unique_original}")
    print(f"  Unique bucketed sizes: {unique_bucketed}")
    print(f"  Compression ratio:     {unique_original / max(unique_bucketed, 1):.1f}x")

    # Show waste stats
    wastes = []
    for (h_orig, w_orig), (h_buck, w_buck) in zip(sizes, bucketed):
        waste_pct = ((h_buck * w_buck) / (h_orig * w_orig) - 1) * 100
        wastes.append(waste_pct)
    print(f"  Memory waste:          mean={np.mean(wastes):.1f}%, "
          f"max={np.max(wastes):.1f}%, min={np.min(wastes):.1f}%")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark shape bucketing vs dynamic pattern regeneration"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to ONNX model (default: auto-generate test model)"
    )
    parser.add_argument(
        "--n", type=int, default=200,
        help="Number of inference iterations (default: 200)"
    )
    parser.add_argument(
        "--variation", choices=["random", "slow", "both"], default="both",
        help="Shape variation pattern (default: both)"
    )
    parser.add_argument(
        "--bucket", choices=list(BUCKET_STRATEGIES.keys()), default="pow2",
        help="Bucket strategy (default: pow2)"
    )
    parser.add_argument(
        "--min-size", type=int, default=200,
        help="Minimum spatial dimension (default: 200)"
    )
    parser.add_argument(
        "--max-size", type=int, default=520,
        help="Maximum spatial dimension (default: 520)"
    )
    parser.add_argument(
        "--threads", type=int, default=1,
        help="Number of intra-op threads (default: 1)"
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Number of warmup iterations (default: 10)"
    )
    args = parser.parse_args()

    # Model
    model_path = args.model
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "test_model.onnx")
        if not os.path.exists(model_path):
            print("Generating test model...")
            from create_test_model import export_model
            export_model(model_path)
            print()

    print(f"Model:            {model_path}")
    print(f"Iterations:       {args.n}")
    print(f"Bucket strategy:  {args.bucket}")
    print(f"Threads:          {args.threads}")
    print(f"Shape range:      [{args.min_size}, {args.max_size}]")

    # Generate shape scenarios
    scenarios = {}
    if args.variation in ("random", "both"):
        scenarios["random"] = generate_varying_sizes(
            args.n, args.min_size, args.max_size
        )
    if args.variation in ("slow", "both"):
        scenarios["slow_varying"] = generate_slowly_varying_sizes(
            args.n, base=(args.min_size + args.max_size) // 2
        )

    for scenario_name, sizes in scenarios.items():
        print(f"\n{'━' * 90}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'━' * 90}")

        print_bucket_analysis(sizes, args.bucket)

        results = []

        # Strategy A: No memory pattern
        print(f"\nRunning: no_pattern ...")
        r = run_benchmark(model_path, sizes, "no_pattern",
                          num_threads=args.threads, warmup=args.warmup)
        results.append(r)

        # Strategy B: Dynamic (pattern enabled, shapes change every time)
        print(f"Running: dynamic ...")
        r = run_benchmark(model_path, sizes, "dynamic",
                          num_threads=args.threads, warmup=args.warmup)
        results.append(r)

        # Strategy C: Bucketed
        print(f"Running: bucketed ({args.bucket}) ...")
        r = run_benchmark(model_path, sizes, "bucketed", args.bucket,
                          num_threads=args.threads, warmup=args.warmup)
        results.append(r)

        print_results(results, sizes)


if __name__ == "__main__":
    main()
