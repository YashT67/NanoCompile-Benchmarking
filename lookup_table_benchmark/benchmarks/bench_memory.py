import os
import psutil
import gc
import numpy as np
import onnxruntime as ort

def get_current_memory_mb():
    """Returns the current memory usage of the Python process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def measure_peak_memory_for_size(model_path: str, h: int, w: int) -> float:
    """
    Creates a session, runs inference to stretch the memory arena, 
    and returns the total memory consumed (Weights + Activations).
    """
    # Force garbage collection to get a clean baseline
    gc.collect()
    mem_baseline = get_current_memory_mb()

    # Load session
    opts = ort.SessionOptions()
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
    
    input_name = session.get_inputs()[0].name
    inp = np.random.randn(1, 3, h, w).astype(np.float32)

    # Run a few times to ensure the memory arena stretches to its peak
    for _ in range(5):
        session.run(None, {input_name: inp})

    # Measure memory after the arena has expanded
    mem_peak = get_current_memory_mb()
    
    # Clean up so the next test starts fresh
    del session
    del inp
    gc.collect()
    
    return mem_peak - mem_baseline

import argparse

def main():
    parser = argparse.ArgumentParser(description="Measure peak memory for real vs bucket sizes.")
    parser.add_argument("--real", type=int, default=500, help="Real input size (e.g. 500 for 500x500)")
    parser.add_argument("--bucket", type=int, default=512, help="Bucket input size (e.g. 512 for 512x512)")
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(__file__), "test_model.onnx")
    
    print("="*60)
    print(" ONNX RUNTIME PEAK MEMORY BENCHMARK")
    print("="*60)
    
    # 1. Measure Real Size
    print(f"Measuring peak memory for {args.real}x{args.real} (Real Size)...")
    mem_real = measure_peak_memory_for_size(model_path, args.real, args.real)
    print(f"  -> Total RAM used: {mem_real:.2f} MB\n")
    
    # 2. Measure Bucket Size
    print(f"Measuring peak memory for {args.bucket}x{args.bucket} (Bucket Size)...")
    mem_bucket = measure_peak_memory_for_size(model_path, args.bucket, args.bucket)
    print(f"  -> Total RAM used: {mem_bucket:.2f} MB\n")
    
    # 3. Calculate the overhead
    overhead_mb = mem_bucket - mem_real
    overhead_pct = (overhead_mb / mem_real) * 100 if mem_real > 0 else 0
    
    print("="*60)
    print(" RESULTS")
    print("="*60)
    print(f" Real Size ({args.real}x{args.real}):     {mem_real:.2f} MB")
    print(f" Bucket Size ({args.bucket}x{args.bucket}): {mem_bucket:.2f} MB")
    print("-" * 60)
    print(f" MEMORY PENALTY:         +{overhead_mb:.2f} MB (+{overhead_pct:.2f}%)")
    print("="*60)
    print("\nNote: Total RAM includes both the Model Weights (constant) and")
    print("the Activation Arena. The penalty is purely from the Arena expanding")
    print("to fit the larger padded buffers.")

if __name__ == "__main__":
    main()
