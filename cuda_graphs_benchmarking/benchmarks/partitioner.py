import onnx
import os

# Rules based on the provided PDF
BOUNDARY_OPS = {
    "If", "Loop", "Scan",                 # Rule 2: Control-Flow
    "NonZero", "Unique", "Compress",      # Rule 3: Value-Dependent Dynamic Output
    "NonMaxSuppression"                   # Rule 3: Value-Dependent Dynamic Output
}
MIN_KERNELS = 3                           # Rule 10: Profitability (min ops for a CUDA graph)

def identify_partitions(model):
    """
    Traverses the model and groups nodes into 'cuda' or 'eager' partitions 
    based on boundary rules.
    """
    partitions = []
    current_cuda_nodes = []
    
    for node in model.graph.node:
        if node.op_type in BOUNDARY_OPS:
            # Hit a boundary! Finalize the current CUDA graph if it exists
            if current_cuda_nodes:
                partitions.append({"type": "cuda", "nodes": current_cuda_nodes})
                current_cuda_nodes = []
            # Add the boundary node itself as an eager partition
            partitions.append({"type": "eager", "nodes": [node]})
        else:
            # Normal node, add to current CUDA graph candidate
            current_cuda_nodes.append(node)
            
    # Add any remaining nodes at the end
    if current_cuda_nodes:
        partitions.append({"type": "cuda", "nodes": current_cuda_nodes})
        
    # Apply Profitability Rule (Rule 10)
    # If a captured region is too small, revert it to eager execution
    for p in partitions:
        if p["type"] == "cuda" and len(p["nodes"]) < MIN_KERNELS:
            p["type"] = "eager"
            
    # Optimization: Merge adjacent eager partitions to reduce overhead
    merged_partitions = []
    for p in partitions:
        if not merged_partitions:
            merged_partitions.append(p)
        elif merged_partitions[-1]["type"] == p["type"]:
            merged_partitions[-1]["nodes"].extend(p["nodes"])
        else:
            merged_partitions.append(p)
            
    return merged_partitions

def extract_subgraph(model, partition_nodes, partition_name):
    """
    Extracts a list of nodes into a standalone, executable ONNX ModelProto.
    """
    node_names = {n.name for n in partition_nodes}
    
    # 1. Dependency Analysis
    produced_tensors = set()
    consumed_tensors = set()
    
    for node in partition_nodes:
        for out in node.output:
            if out: produced_tensors.add(out)
        for inp in node.input:
            if inp: consumed_tensors.add(inp)
                
    # Inputs to this partition are tensors consumed but NOT produced here
    partition_inputs = consumed_tensors - produced_tensors
    
    # Outputs are tensors produced here that are either:
    # a) Consumed by nodes outside this partition
    # b) Part of the main model's final outputs
    global_outputs = {out.name for out in model.graph.output}
    all_other_consumed = set()
    for node in model.graph.node:
        if node.name not in node_names:
            for inp in node.input:
                if inp: all_other_consumed.add(inp)
                
    partition_outputs = produced_tensors & (all_other_consumed | global_outputs)
    
    # 2. Gather IO definitions and Weights (Initializers)
    value_info_map = {vi.name: vi for vi in model.graph.value_info}
    for vi in model.graph.input: value_info_map[vi.name] = vi
    for vi in model.graph.output: value_info_map[vi.name] = vi
        
    init_map = {init.name: init for init in model.graph.initializer}
    
    graph_inputs = []
    graph_outputs = []
    graph_inits = []
    
    for inp in partition_inputs:
        if inp in init_map:
            graph_inits.append(init_map[inp])
        elif inp in value_info_map:
            graph_inputs.append(value_info_map[inp])
        else:
            # Fallback for dynamic/unknown shapes
            graph_inputs.append(onnx.helper.make_tensor_value_info(inp, onnx.TensorProto.FLOAT, None))
            
    for out in partition_outputs:
        if out in value_info_map:
            graph_outputs.append(value_info_map[out])
        else:
            graph_outputs.append(onnx.helper.make_tensor_value_info(out, onnx.TensorProto.FLOAT, None))
            
    # 3. Construct Graph and Model
    graph_def = onnx.helper.make_graph(
        nodes=partition_nodes,
        name=partition_name,
        inputs=graph_inputs,
        outputs=graph_outputs,
        initializer=graph_inits
    )
    
    model_def = onnx.helper.make_model(graph_def, producer_name="cuda_partitioner")
    
    # Ensure opset matches original model
    del model_def.opset_import[:]
    model_def.opset_import.extend(model.opset_import)
    
    return model_def

def run_partitioning(model_path, output_dir="partitions"):
    print(f"\n--- Partitioning {model_path} ---")
    model = onnx.load(model_path)
    
    partitions = identify_partitions(model)
    print(f"Identified {len(partitions)} total partitions.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i, p in enumerate(partitions):
        p_type = p["type"]
        p_nodes = p["nodes"]
        model_name = f"partition_{i}_{p_type}.onnx"
        out_path = os.path.join(output_dir, model_name)
        
        print(f"  - [{p_type.upper()}] Partition {i}: {len(p_nodes)} nodes")
        
        submodel = extract_subgraph(model, p_nodes, model_name)
        onnx.save(submodel, out_path)
        
    print(f"Saved extracted subgraphs to ./{output_dir}/")

if __name__ == "__main__":
    run_partitioning("synthetic_boundary_model.onnx", output_dir="partitions_synthetic")
    run_partitioning("test_model.onnx", output_dir="partitions_test")
