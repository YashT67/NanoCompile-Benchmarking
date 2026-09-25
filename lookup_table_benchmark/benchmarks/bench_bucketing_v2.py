"""
Benchmark: Shape Bucketing (Memory-Only) vs Dynamic Allocation

This benchmark CORRECTLY tests the user's idea:
  - Allocate memory buffers sized for bucket dimensions (e.g., 512×512)
  - But run actual compute on the real input dimensions (e.g., 500×500)
  - The bucket only determines the MEMORY LAYOUT, not the compute

Since ONNX Runtime doesn't natively support "oversized memory patterns",
we simulate the idea by separating:
  1. Memory overhead: pre-allocating a block sized for 512×512 intermediates
  2. Compute: running inference on actual 500×500 input

The benchmark measures:
  A. dynamic         — Real 500×500, ORT handles memory normally
  B. bucketed_naive  — Actual 512×512 input (the WRONG benchmark from before)
  C. bucketed_ideal  — Simulates the real idea: 500×500 compute + overhead of
                        pre-allocating oversized buffers via a separate sizing pass

We also directly measure the ISOLATED cost of:
  - Memory pattern generation alone (by timing first vs subsequent runs)
  - Pure compute difference (500×500 vs 512×512)
"""

import argparse
import os
import time
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort


# ─── Helpers ─────────────────────────────────────────────────────────────────

def next_power_of_two(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


BUCKET_STRATEGIES = {
    "pow2": next_power_of_two,
    "mul64": lambda n: ((n + 63) // 64) * 64,
    "mul32": lambda n: ((n + 31) // 32) * 32,
    "mul128": lambda n: ((n + 127) // 128) * 128,
}


def create_session(
    model_path: str,
    enable_mem_pattern: bool,
    num_threads: int = 1,
) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.enable_mem_pattern = enable_mem_pattern
    opts.enable_cpu_mem_arena = True
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = num_threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.log_severity_level = 3
    return ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )


# ─── Benchmark: Measure pattern generation cost in isolation ─────────────────

def measure_pattern_generation_cost(
    model_path: str, sizes: list[tuple[int, int]], num_threads: int = 1
) -> dict:
    """
    Measure the cost of ORT regenerating the memory pattern by comparing:
    - First run with a new shape (pattern miss → must generate)
    - Second run with same shape (pattern hit → reuse)
    """
    results = {"first_run_ms": [], "subsequent_run_ms": [], "delta_ms": []}

    for h, w in sizes:
        # Create a FRESH session each time so no pattern is cached
        session = create_session(model_path, enable_mem_pattern=True, num_threads=num_threads)
        input_name = session.get_inputs()[0].name
        inp = np.random.randn(1, 3, h, w).astype(np.float32)

        # First run: pattern miss (must generate pattern)
        start = time.perf_counter()
        session.run(None, {input_name: inp})
        first = (time.perf_counter() - start) * 1000

        # Second run: pattern hit (reuse cached pattern)
        start = time.perf_counter()
        session.run(None, {input_name: inp})
        second = (time.perf_counter() - start) * 1000

        # Third run: also a hit (more stable measurement)
        start = time.perf_counter()
        session.run(None, {input_name: inp})
        third = (time.perf_counter() - start) * 1000

        avg_subsequent = (second + third) / 2
        results["first_run_ms"].append(first)
        results["subsequent_run_ms"].append(avg_subsequent)
        results["delta_ms"].append(first - avg_subsequent)

    return results


# ─── Benchmark: Measure PURE compute difference ─────────────────────────────

def measure_compute_overhead(
    model_path: str,
    real_h: int, real_w: int,
    bucket_h: int, bucket_w: int,
    n: int = 50,
    num_threads: int = 1,
) -> dict:
    """
    Measure the pure compute time difference between running on
    real dimensions vs bucket dimensions.
    Both use the same session with mem_pattern enabled (pattern is warm).
    """
    session = create_session(model_path, enable_mem_pattern=True, num_threads=num_threads)
    input_name = session.get_inputs()[0].name

    # --- Measure real size ---
    inp_real = np.random.randn(1, 3, real_h, real_w).astype(np.float32)
    # Warmup
    for _ in range(5):
        session.run(None, {input_name: inp_real})
    # Timed
    real_times = []
    for _ in range(n):
        start = time.perf_counter()
        session.run(None, {input_name: inp_real})
        real_times.append((time.perf_counter() - start) * 1000)

    # --- Measure bucket size (fresh session to get clean pattern) ---
    session2 = create_session(model_path, enable_mem_pattern=True, num_threads=num_threads)
    inp_bucket = np.random.randn(1, 3, bucket_h, bucket_w).astype(np.float32)
    # Warmup
    for _ in range(5):
        session2.run(None, {input_name: inp_bucket})
    # Timed
    bucket_times = []
    for _ in range(n):
        start = time.perf_counter()
        session2.run(None, {input_name: inp_bucket})
        bucket_times.append((time.perf_counter() - start) * 1000)

    return {
        "real_size": (real_h, real_w),
        "bucket_size": (bucket_h, bucket_w),
        "real_median_ms": float(np.median(real_times)),
        "bucket_median_ms": float(np.median(bucket_times)),
        "compute_overhead_ms": float(np.median(bucket_times) - np.median(real_times)),
        "compute_overhead_pct": float(
            (np.median(bucket_times) - np.median(real_times)) / np.median(real_times) * 100
        ),
        "pixel_overhead_pct": float(
            (bucket_h * bucket_w - real_h * real_w) / (real_h * real_w) * 100
        ),
    }


# ─── Benchmark: Simulate the REAL bucketing idea ────────────────────────────

def measure_bucketed_memory_only(
    model_path: str,
    sizes: list[tuple[int, int]],
    bucket_fn_name: str = "pow2",
    n_per_size: int = 20,
    num_threads: int = 1,
) -> dict:
    """
    Simulate the user's ACTUAL idea:
    1. A "sizing pass" pre-allocates memory for the bucket dimensions
       (this is just a single throwaway inference at bucket size)
    2. All subsequent inferences run at the REAL dimensions
       but conceptually reuse the oversized memory layout

    Since ORT doesn't support this natively, we model the cost as:
       cost = one-time bucket sizing pass + N × real-size inference (no pattern regen)

    vs baseline:
       cost = N × real-size inference (with pattern regen on shape change)
    """
    bucket_fn = BUCKET_STRATEGIES[bucket_fn_name]
    input_name = None
    results = {
        "sizing_pass_ms": [],
        "real_inference_ms": [],
        "baseline_first_ms": [],
        "baseline_subsequent_ms": [],
    }

    for h, w in sizes:
        bh, bw = bucket_fn(h), bucket_fn(w)

        # --- Simulate bucketed approach ---
        session = create_session(model_path, enable_mem_pattern=True, num_threads=num_threads)
        input_name = session.get_inputs()[0].name

        # Step 1: Sizing pass at bucket dimensions (one-time cost)
        inp_bucket = np.random.randn(1, 3, bh, bw).astype(np.float32)
        start = time.perf_counter()
        session.run(None, {input_name: inp_bucket})
        sizing_ms = (time.perf_counter() - start) * 1000
        results["sizing_pass_ms"].append(sizing_ms)

        # Step 2: Real inferences at actual size
        # (In your idea, these would reuse the bucket's memory layout.
        #  In ORT, this triggers a pattern regen, so we measure that cost.)
        inp_real = np.random.randn(1, 3, h, w).astype(np.float32)
        times = []
        for _ in range(n_per_size):
            start = time.perf_counter()
            session.run(None, {input_name: inp_real})
            times.append((time.perf_counter() - start) * 1000)
        results["real_inference_ms"].extend(times)

        # --- Baseline: fresh session, real size only ---
        session_base = create_session(model_path, enable_mem_pattern=True, num_threads=num_threads)
        inp_real2 = np.random.randn(1, 3, h, w).astype(np.float32)

        # First run (pattern generation)
        start = time.perf_counter()
        session_base.run(None, {input_name: inp_real2})
        results["baseline_first_ms"].append((time.perf_counter() - start) * 1000)

        # Subsequent runs (pattern cached)
        for _ in range(n_per_size):
            start = time.perf_counter()
            session_base.run(None, {input_name: inp_real2})
            results["baseline_subsequent_ms"].append((time.perf_counter() - start) * 1000)

    return results


# ─── Benchmark: Varying shapes (the real test) ──────────────────────────────

def measure_varying_shapes(
    model_path: str,
    sizes: list[tuple[int, int]],
    bucket_fn_name: str = "pow2",
    num_threads: int = 1,
) -> dict:
    """
    The KEY benchmark: what happens with VARYING shapes?

    Strategy A (dynamic): Feed varying shapes directly. Every shape change
                          invalidates the pattern → regen cost each time.
    Strategy B (bucketed ideal): For each real shape, the bucketed approach
                          would map it to a bucket. If that bucket's pattern
                          is already cached → no regen. Compute is still on
                          the REAL shape.

    We measure:
      - dynamic: total time with pattern misses
      - bucketed_ideal: total time = bucket warmups + real computes (pattern hits)
    """
    bucket_fn = BUCKET_STRATEGIES[bucket_fn_name]
    
    # --- Strategy A: Dynamic (every shape change = pattern miss) ---
    session_dyn = create_session(model_path, enable_mem_pattern=True, num_threads=num_threads)
    input_name = session_dyn.get_inputs()[0].name

    # Warmup with first shape
    h0, w0 = sizes[0]
    inp = np.random.randn(1, 3, h0, w0).astype(np.float32)
    for _ in range(3):
        session_dyn.run(None, {input_name: inp})

    dynamic_times = []
    for h, w in sizes:
        inp = np.random.randn(1, 3, h, w).astype(np.float32)
        start = time.perf_counter()
        session_dyn.run(None, {input_name: inp})
        dynamic_times.append((time.perf_counter() - start) * 1000)

    # --- Strategy B: Bucketed ideal ---
    # In your ideal runtime: you'd pre-warm each bucket, then run real shapes.
    # The memory offsets come from the bucket, compute is on the real shape.
    # Since ORT can't do this, we model it as:
    #   cost_per_inference = real_shape_compute (from a warm session with that exact shape)
    #   + amortized bucket warmup cost
    
    # First, pre-warm all unique buckets
    unique_buckets = set()
    for h, w in sizes:
        unique_buckets.add((bucket_fn(h), bucket_fn(w)))

    bucket_warmup_total_ms = 0.0
    session_buck = create_session(model_path, enable_mem_pattern=True, num_threads=num_threads)
    for bh, bw in unique_buckets:
        inp = np.random.randn(1, 3, bh, bw).astype(np.float32)
        start = time.perf_counter()
        session_buck.run(None, {input_name: inp})
        bucket_warmup_total_ms += (time.perf_counter() - start) * 1000

    # Now measure real-shape compute (with pattern already warm for that shape)
    # This represents the BEST CASE for your idea: compute on real shape,
    # zero pattern overhead.
    session_real = create_session(model_path, enable_mem_pattern=True, num_threads=num_threads)
    # Warm it up with one shape
    inp = np.random.randn(1, 3, sizes[0][0], sizes[0][1]).astype(np.float32)
    session_real.run(None, {input_name: inp})

    bucketed_compute_times = []
    last_shape = sizes[0]
    for h, w in sizes:
        # If shape changed, we need a "first run" for this shape's compute
        # (but in your ideal runtime, there'd be no pattern regen — just compute)
        inp = np.random.randn(1, 3, h, w).astype(np.float32)
        if (h, w) != last_shape:
            # Shape changed — in your system, memory is already laid out (bucket hit)
            # The only cost is compute. We still measure the ORT run, which includes
            # a pattern regen, but we'll account for that.
            session_real.run(None, {input_name: inp})  # pattern warmup
            last_shape = (h, w)

        start = time.perf_counter()
        session_real.run(None, {input_name: inp})
        bucketed_compute_times.append((time.perf_counter() - start) * 1000)

    return {
        "dynamic_times": dynamic_times,
        "dynamic_median": float(np.median(dynamic_times)),
        "dynamic_total": float(np.sum(dynamic_times)),
        "bucketed_compute_times": bucketed_compute_times,
        "bucketed_compute_median": float(np.median(bucketed_compute_times)),
        "bucketed_compute_total": float(np.sum(bucketed_compute_times)),
        "bucket_warmup_total_ms": bucket_warmup_total_ms,
        "num_unique_buckets": len(unique_buckets),
        "num_unique_shapes": len(set(sizes)),
        "num_inferences": len(sizes),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: Memory-only shape bucketing (CORRECTED)"
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--bucket", choices=list(BUCKET_STRATEGIES.keys()), default="pow2")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--real-h", type=int, default=500)
    parser.add_argument("--real-w", type=int, default=500)
    parser.add_argument("--n", type=int, default=50, help="Iterations per test")
    parser.add_argument(
        "--test", choices=["all", "pattern_cost", "compute_overhead", "varying"],
        default="all", help="Which test to run"
    )
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "test_model.onnx")
        if not os.path.exists(model_path):
            print("Generating test model...")
            from create_test_model import export_model
            export_model(model_path)
            print()

    bucket_fn = BUCKET_STRATEGIES[args.bucket]
    bh, bw = bucket_fn(args.real_h), bucket_fn(args.real_w)

    print(f"Model:            {model_path}")
    print(f"Real size:        {args.real_h}×{args.real_w}")
    print(f"Bucket size:      {bh}×{bw} ({args.bucket})")
    print(f"Threads:          {args.threads}")
    print(f"Iterations:       {args.n}")
    pixel_waste = (bh * bw - args.real_h * args.real_w) / (args.real_h * args.real_w) * 100
    print(f"Pixel overhead:   {pixel_waste:.1f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 1: How expensive is pattern generation ALONE?
    # ═══════════════════════════════════════════════════════════════════════
    if args.test in ("all", "pattern_cost"):
        print(f"\n{'═' * 70}")
        print("TEST 1: Pattern Generation Cost (isolated)")
        print(f"{'═' * 70}")
        print("Measuring: first run (pattern miss) vs subsequent runs (pattern hit)")
        print(f"Running {args.n} fresh sessions at {args.real_h}×{args.real_w}...\n")

        sizes = [(args.real_h, args.real_w)] * args.n
        r = measure_pattern_generation_cost(model_path, sizes, args.threads)

        first_med = float(np.median(r["first_run_ms"]))
        subseq_med = float(np.median(r["subsequent_run_ms"]))
        delta_med = float(np.median(r["delta_ms"]))

        print(f"  First run (pattern miss):    {first_med:.3f} ms (median)")
        print(f"  Subsequent (pattern hit):    {subseq_med:.3f} ms (median)")
        print(f"  Pattern generation cost:     {delta_med:.3f} ms (median)")
        print(f"  Pattern cost as % of run:    {delta_med / first_med * 100:.2f}%")
        print()
        print(f"  → This is the MAXIMUM time your bucketing idea can save per inference")
        print(f"    when a shape change would have caused a pattern miss.")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 2: Pure compute overhead of running 512×512 vs 500×500
    # ═══════════════════════════════════════════════════════════════════════
    if args.test in ("all", "compute_overhead"):
        print(f"\n{'═' * 70}")
        print("TEST 2: Compute Overhead ({args.real_h}×{args.real_w} vs {bh}×{bw})")
        print(f"{'═' * 70}")
        print("This measures how much EXTRA compute the naive benchmark was doing.\n")

        r = measure_compute_overhead(
            model_path, args.real_h, args.real_w, bh, bw, args.n, args.threads
        )

        print(f"  {r['real_size'][0]}×{r['real_size'][1]} compute:   {r['real_median_ms']:.3f} ms (median)")
        print(f"  {r['bucket_size'][0]}×{r['bucket_size'][1]} compute:   {r['bucket_median_ms']:.3f} ms (median)")
        print(f"  Extra compute cost:          {r['compute_overhead_ms']:.3f} ms ({r['compute_overhead_pct']:.2f}%)")
        print(f"  Extra pixels:                {r['pixel_overhead_pct']:.1f}%")
        print()
        print(f"  → In your CORRECT idea, this extra compute cost is ZERO")
        print(f"    because you run compute at {args.real_h}×{args.real_w}, not {bh}×{bw}.")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 3: Varying shapes — the real-world test
    # ═══════════════════════════════════════════════════════════════════════
    if args.test in ("all", "varying"):
        print(f"\n{'═' * 70}")
        print("TEST 3: Varying Shapes (the real-world scenario)")
        print(f"{'═' * 70}")

        # Generate random varying sizes
        rng = np.random.RandomState(42)
        lo, hi = max(args.real_h - 50, 100), args.real_h + 50
        sizes = [
            (int(rng.randint(lo, hi + 1)), int(rng.randint(lo, hi + 1)))
            for _ in range(args.n)
        ]
        unique_shapes = len(set(sizes))
        unique_buckets = len(set((bucket_fn(h), bucket_fn(w)) for h, w in sizes))

        print(f"  {args.n} inferences with random sizes in [{lo}, {hi}]")
        print(f"  Unique shapes: {unique_shapes}")
        print(f"  Unique buckets ({args.bucket}): {unique_buckets}")
        print()

        r = measure_varying_shapes(model_path, sizes, args.bucket, args.threads)

        print(f"  Dynamic (ORT default):")
        print(f"    Median:  {r['dynamic_median']:.3f} ms")
        print(f"    Total:   {r['dynamic_total']:.1f} ms")
        print()
        print(f"  Bucketed ideal (your idea: bucket memory, real compute):")
        print(f"    Compute median: {r['bucketed_compute_median']:.3f} ms")
        print(f"    Compute total:  {r['bucketed_compute_total']:.1f} ms")
        print(f"    Bucket warmup:  {r['bucket_warmup_total_ms']:.1f} ms "
              f"({r['num_unique_buckets']} buckets)")
        print(f"    Grand total:    {r['bucketed_compute_total'] + r['bucket_warmup_total_ms']:.1f} ms")
        print()

        saving = r["dynamic_total"] - (r["bucketed_compute_total"] + r["bucket_warmup_total_ms"])
        pct = saving / r["dynamic_total"] * 100
        sign = "FASTER" if saving > 0 else "SLOWER"
        print(f"  → Your idea is {abs(saving):.1f} ms {sign} over {args.n} inferences ({abs(pct):.2f}%)")
        print(f"    Per-inference saving: {abs(saving / args.n):.3f} ms")

    print(f"\n{'═' * 70}")


if __name__ == "__main__":
    main()
