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
import os

def get_dummy_inputs(model_path):
    """Dynamically generates dummy input data based on the model's graph inputs."""
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

def benchmark_native(model_path, num_warmup=20, num_iters=200):
    """
    Benchmarks the full model in a single native ONNX Runtime C++ session.
    Uses IOBinding to eliminate Python-to-C++ memory transfer overhead.
    """
    print(f"\n--- Native ORT Benchmark: {model_path} ---")
    feed = get_dummy_inputs(model_path)
    
    def run_with_config(enable_cg):
        providers = [
            ("CUDAExecutionProvider", {
                "enable_cuda_graph": "1" if enable_cg else "0"
            }),
            "CPUExecutionProvider"
        ]
        
        # Suppress the internal C++ Memcpy warnings and error traces
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 4 # Fatal only
        
        sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        io_binding = sess.io_binding()
        
        for name, data in feed.items():
            ort_val = ort.OrtValue.ortvalue_from_numpy(data, 'cuda', 0)
            io_binding.bind_ortvalue_input(name, ort_val)
            
        for out in sess.get_outputs():
            io_binding.bind_output(out.name, 'cuda')
            
        for _ in range(num_warmup):
            sess.run_with_iobinding(io_binding)
            
        io_binding.copy_outputs_to_cpu()
            
        start = time.perf_counter()
        for _ in range(num_iters):
            sess.run_with_iobinding(io_binding)
            
        io_binding.copy_outputs_to_cpu()
        
        return (time.perf_counter() - start) / num_iters * 1000

    try:
        time_no_cg = run_with_config(enable_cg=False)
        print(f"Native Eager Latency (CUDA Graph OFF): {time_no_cg:.3f} ms")
        
        try:
            time_cg = run_with_config(enable_cg=True)
            print(f"Native Latency (CUDA Graph ON):        {time_cg:.3f} ms")
            print(f"-> Launch Time Saved Natively:         {time_no_cg - time_cg:.3f} ms per inference")
        except Exception as e:
            # Expected behavior for the synthetic model!
            print("Native Latency (CUDA Graph ON):        [REJECTED BY ORT]")
            print("-> Note: Native ORT refused to capture this model because it contains dynamic boundaries.")
            print("-> This is expected! Use `benchmark.py` to test the custom subgraph orchestrator.")
            
    except Exception as e:
        print(f"Error benchmarking {model_path}:\n{e}")
        print("\nNote: Make sure ONNX Runtime GPU is installed and CUDA is available.")

if __name__ == "__main__":
    if os.path.exists("synthetic_boundary_model.onnx"):
        benchmark_native("synthetic_boundary_model.onnx")
        
    if os.path.exists("test_model.onnx"):
        benchmark_native("test_model.onnx")
