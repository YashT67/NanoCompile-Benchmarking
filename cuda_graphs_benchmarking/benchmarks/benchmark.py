import os
import sys

cudnn_path = r"C:\Program Files\NVIDIA\CUDNN\v9.26\bin\13.4\x64"
os.environ['PATH'] = cudnn_path + os.pathsep + os.environ.get('PATH', '')

if hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(cudnn_path)
    except Exception:
        pass

import onnx
import onnxruntime as ort
import numpy as np
import time

def get_dummy_inputs(model_path):
    model = onnx.load(model_path)
    feed = {}
    dtype_map = {1: np.float32, 6: np.int32, 7: np.int64, 9: bool, 11: np.float64}
    
    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            val = dim.dim_value
            if val <= 0: val = 1
            shape.append(val)
        elem_type = inp.type.tensor_type.elem_type
        np_dtype = dtype_map.get(elem_type, np.float32)
        
        if np_dtype == bool:
            feed[inp.name] = np.random.choice([True, False], size=shape)
        elif np_dtype in (np.int32, np.int64):
            feed[inp.name] = np.random.randint(0, 10, size=shape, dtype=np_dtype)
        else:
            feed[inp.name] = np.random.rand(*shape).astype(np_dtype)
    return feed

def load_pipeline(partition_dir, enable_cuda_graph):
    files = sorted([f for f in os.listdir(partition_dir) if f.endswith('.onnx')])
    sessions = []
    
    for f in files:
        path = os.path.join(partition_dir, f)
        is_cuda_partition = "cuda" in f
        
        providers = [
            ("CUDAExecutionProvider", {
                "enable_cuda_graph": "1" if (is_cuda_partition and enable_cuda_graph) else "0"
            }),
            "CPUExecutionProvider"
        ]
        
        sess = ort.InferenceSession(path, providers=providers)
        
        # We must use IOBinding for CUDA Graphs. It pre-allocates memory addresses
        # on the GPU so that they remain strictly stable across iterations!
        io_binding = sess.io_binding()
        input_names = [i.name for i in sess.get_inputs()]
        output_names = [o.name for o in sess.get_outputs()]
        
        # Persistently bind outputs to the GPU
        for name in output_names:
            io_binding.bind_output(name, 'cuda')
            
        sessions.append((sess, input_names, output_names, f, io_binding))
        
    return sessions

def run_pipeline(sessions, input_feed, is_warmup=False):
    # Convert initial CPU inputs to strictly bound GPU memory values
    ort_state = {}
    for k, v in input_feed.items():
        ort_state[k] = ort.OrtValue.ortvalue_from_numpy(v, 'cuda', 0)
        
    for sess, in_names, out_names, partition_name, io_binding in sessions:
        io_binding.clear_binding_inputs()
        
        # Link the required intermediate tensors from the GPU pool
        for name in in_names:
            io_binding.bind_ortvalue_input(name, ort_state[name])
            
        # Execute the partition natively on GPU
        sess.run_with_iobinding(io_binding)
        
        # Store outputs in the GPU pool for the next partition
        outs = io_binding.get_outputs()
        for name, out_val in zip(out_names, outs):
            ort_state[name] = out_val
            
    # Copy final output back to CPU just to force synchronization for the benchmark timer
    if not is_warmup:
        sessions[-1][4].copy_outputs_to_cpu()

def benchmark_pipeline(partition_dir, input_feed, num_warmup=10, num_iters=100):
    print(f"\n--- Benchmarking Orchestration: {partition_dir} ---")
    
    pipe_no_cg = load_pipeline(partition_dir, enable_cuda_graph=False)
    pipe_cg = load_pipeline(partition_dir, enable_cuda_graph=True)
    
    print("Warming up...")
    for _ in range(num_warmup):
        run_pipeline(pipe_no_cg, input_feed, is_warmup=True)
        run_pipeline(pipe_cg, input_feed, is_warmup=True)
        
    print(f"Measuring {num_iters} iterations...")
    
    start = time.perf_counter()
    for _ in range(num_iters):
        run_pipeline(pipe_no_cg, input_feed)
    time_no_cg = (time.perf_counter() - start) / num_iters * 1000
    
    start = time.perf_counter()
    for _ in range(num_iters):
        run_pipeline(pipe_cg, input_feed)
    time_cg = (time.perf_counter() - start) / num_iters * 1000
    
    print(f"Average Latency (Eager fallback):      {time_no_cg:.3f} ms")
    print(f"Average Latency (CUDA Subgraphs):      {time_cg:.3f} ms")
    print(f"-> Kernel Launch Time Saved:           {time_no_cg - time_cg:.3f} ms per inference")

if __name__ == "__main__":
    if os.path.exists("partitions_synthetic"):
        feed_synth = get_dummy_inputs("synthetic_boundary_model.onnx")
        benchmark_pipeline("partitions_synthetic", feed_synth)
    else:
        print("partitions_synthetic not found. Please run partitioner.py first.")
        
    if os.path.exists("partitions_test"):
        feed_test = get_dummy_inputs("test_model.onnx")
        benchmark_pipeline("partitions_test", feed_test)
