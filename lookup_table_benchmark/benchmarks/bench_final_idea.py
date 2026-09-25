import argparse
import os
import time
import gc
import psutil
import numpy as np
import onnxruntime as ort

# --- Configuration ---
BUCKETS = [(512, 512), (512, 724), (724, 512), (724, 724)]

def get_bucket(h: int, w: int) -> tuple[int, int]:
    """Maps a real dimension to the smallest fitting bucket."""
    bh = 512 if h <= 512 else 724
    bw = 512 if w <= 512 else 724
    return bh, bw

def get_current_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def create_session(model_path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 3
    return ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])

def measure_peak_memory(model_path: str, h: int, w: int) -> float:
    """Creates a session, runs it to stretch the arena, measures RAM, and destroys it."""
    gc.collect()
    mem_baseline = get_current_memory_mb()
    
    session = create_session(model_path)
    input_name = session.get_inputs()[0].name
    inp = np.random.randn(1, 3, h, w).astype(np.float32)
    
    for _ in range(3):
        session.run(None, {input_name: inp})
        
    mem_peak = get_current_memory_mb()
    
    del session
    del inp
    gc.collect()
    
    return mem_peak - mem_baseline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of iterations")
    parser.add_argument("--min", type=int, default=400, help="Minimum dimension")
    parser.add_argument("--max", type=int, default=600, help="Maximum dimension")
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(__file__), "test_model.onnx")
    
    print("="*70)
    print(" BUCKETING SIMULATION (Double-Run Strategy)")
    print("="*70)
    print(f" Buckets:     {BUCKETS}")
    print(f" Iterations:  {args.n}")
    print(f" Size Range:  [{args.min}, {args.max}]\n")

    # Generate the random sequence of shapes
    rng = np.random.RandomState(42)
    sizes = [(int(rng.randint(args.min, args.max + 1)), 
              int(rng.randint(args.min, args.max + 1))) for _ in range(args.n)]

    # ---------------------------------------------------------
    # PHASE 0: Pre-warm the 4 Buckets (One-time setup)
    # ---------------------------------------------------------
    print("Phase 0: Pre-compiling the 4 memory buckets...")
    bucket_setup_start = time.perf_counter()
    setup_session = create_session(model_path)
    input_name = setup_session.get_inputs()[0].name
    for bh, bw in BUCKETS:
        inp = np.random.randn(1, 3, bh, bw).astype(np.float32)
        setup_session.run(None, {input_name: inp})
    bucket_setup_time = (time.perf_counter() - bucket_setup_start) * 1000
    print(f"  -> Successfully compiled 4 buckets in {bucket_setup_time:.2f} ms\n")

    # ---------------------------------------------------------
    # PHASE 1: Memory
    # ---------------------------------------------------------
    print("Phase 1: Measuring Memory Overheads (this takes a moment)...")
    memory_cache = {}
    extra_mem_list = []
    real_mem_list = []
    
    # Pre-measure all buckets
    for bh, bw in BUCKETS:
        memory_cache[(bh, bw)] = measure_peak_memory(model_path, bh, bw)
        
    for i, (h, w) in enumerate(sizes):
        bh, bw = get_bucket(h, w)
        
        if (h, w) not in memory_cache:
            memory_cache[(h, w)] = measure_peak_memory(model_path, h, w)
            
        real_mem = memory_cache[(h, w)]
        bucket_mem = memory_cache[(bh, bw)]
        
        real_mem_list.append(real_mem)
        extra_mem_list.append(bucket_mem - real_mem)

    # ---------------------------------------------------------
    # PHASE 2: Time (Double-Run Strategy)
    # ---------------------------------------------------------
    print("Phase 2: Measuring Time Saved (Double-run strategy)...\n")
    time_saved_list = []
    dyn_time_list = []
    
    # Single continuous session
    session = create_session(model_path)
    input_name = session.get_inputs()[0].name
    
    # Pre-stretch the memory arena to the max size so OS memory 
    # allocation (malloc) doesn't skew our time measurements!
    inp_max = np.random.randn(1, 3, args.max, args.max).astype(np.float32)
    session.run(None, {input_name: inp_max})

    for i, (h, w) in enumerate(sizes):
        inp = np.random.randn(1, 3, h, w).astype(np.float32)
        
        # 1. RUN 1: Pattern Miss (Simulates standard Dynamic ORT)
        start = time.perf_counter()
        session.run(None, {input_name: inp})
        t_dyn = (time.perf_counter() - start) * 1000
        
        # 2. RUN 2: Pattern Hit (Simulates your Bucketed engine)
        start = time.perf_counter()
        session.run(None, {input_name: inp})
        t_bucket = (time.perf_counter() - start) * 1000
        
        dyn_time_list.append(t_dyn)
        time_saved_list.append(t_dyn - t_bucket)

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------
    mean_mem_extra = np.mean(extra_mem_list)
    mean_mem_real = np.mean(real_mem_list)
    median_mem_extra = np.median(extra_mem_list)
    median_mem_real = np.median(real_mem_list)
    
    mean_time_saved = np.mean(time_saved_list)
    mean_time_dyn = np.mean(dyn_time_list)
    median_time_saved = np.median(time_saved_list)
    median_time_dyn = np.median(dyn_time_list)
    
    # Percentages
    pct_mean_time = (mean_time_saved / mean_time_dyn) * 100
    pct_median_time = (median_time_saved / median_time_dyn) * 100
    pct_mean_mem = (mean_mem_extra / mean_mem_real) * 100
    pct_median_mem = (median_mem_extra / median_mem_real) * 100

    print("="*70)
    print(" FINAL RESULTS (over 100 iterations)")
    print("="*70)
    print(f" ONE-TIME SETUP COST: {bucket_setup_time:.2f} ms")
    print("-" * 70)
    print(" TIME SAVED (per inference):")
    print(f"   Mean:   {mean_time_saved:.3f} ms ({pct_mean_time:.2f}% faster)")
    print(f"   Median: {median_time_saved:.3f} ms ({pct_median_time:.2f}% faster)")
    print("-" * 70)
    print(" EXTRA MEMORY USED (per inference):")
    print(f"   Mean:   {mean_mem_extra:.2f} MB ({pct_mean_mem:.2f}% more RAM)")
    print(f"   Median: {median_mem_extra:.2f} MB ({pct_median_mem:.2f}% more RAM)")
    print("="*70)

if __name__ == "__main__":
    main()
