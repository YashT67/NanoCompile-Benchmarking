import onnx
from onnx import helper, TensorProto
import numpy as np

def create_conv_relu(name_prefix, input_name, in_channels, out_channels):
    w_name = f"{name_prefix}_W"
    b_name = f"{name_prefix}_B"
    out_conv = f"{name_prefix}_conv_out"
    out_relu = f"{name_prefix}_out"
    
    # Create initializers (using positive values so the ReLU activations never die completely)
    w_data = (np.random.rand(out_channels, in_channels, 3, 3) + 0.1).astype(np.float32)
    b_data = (np.random.rand(out_channels) + 0.1).astype(np.float32)
    
    init_w = helper.make_tensor(name=w_name, data_type=TensorProto.FLOAT, dims=w_data.shape, vals=w_data.flatten().tolist())
    init_b = helper.make_tensor(name=b_name, data_type=TensorProto.FLOAT, dims=b_data.shape, vals=b_data.flatten().tolist())
    
    # Create nodes
    conv_node = helper.make_node(
        "Conv",
        inputs=[input_name, w_name, b_name],
        outputs=[out_conv],
        name=f"{name_prefix}_Conv",
        pads=[1, 1, 1, 1]
    )
    relu_node = helper.make_node(
        "Relu",
        inputs=[out_conv],
        outputs=[out_relu],
        name=f"{name_prefix}_Relu"
    )
    
    return [conv_node, relu_node], [init_w, init_b], out_relu

def add_boundary(name_prefix, current_input, initializers):
    """
    Injects a value-dependent dynamic output boundary (Unique) that forces a sync.
    Unique guarantees at least 1 element, preventing ReduceSum crashes on empty tensors.
    """
    u_out = f"{name_prefix}_unique_out"
    # Unique outputs 4 tensors, we only need the first one (the unique values).
    u_node = helper.make_node("Unique", [current_input], [u_out, f"{name_prefix}_idx", f"{name_prefix}_inv", f"{name_prefix}_counts"], name=f"{name_prefix}_Unique")
    
    cast_out = f"{name_prefix}_cast_out"
    cast_node = helper.make_node("Cast", [u_out], [cast_out], to=TensorProto.FLOAT, name=f"{name_prefix}_Cast")
    
    sum_out = f"{name_prefix}_sum_out"
    axes_name = f"{name_prefix}_axes_for_reduce"
    axes_data = np.array([0], dtype=np.int64) # Unique output Y is 1D
    init_axes = helper.make_tensor(name=axes_name, data_type=TensorProto.INT64, dims=axes_data.shape, vals=axes_data.tolist())
    initializers.append(init_axes)
    
    reduce_node = helper.make_node("ReduceSum", [cast_out, axes_name], [sum_out], keepdims=0, name=f"{name_prefix}_ReduceSum")
    
    zero_name = f"{name_prefix}_zero_const"
    init_zero = helper.make_tensor(name=zero_name, data_type=TensorProto.FLOAT, dims=[], vals=[0.0])
    initializers.append(init_zero)
    
    mul_out = f"{name_prefix}_mul_out"
    mul_node = helper.make_node("Mul", [sum_out, zero_name], [mul_out], name=f"{name_prefix}_Mul")
    
    add_out = f"{name_prefix}_add_out"
    add_node = helper.make_node("Add", [current_input, mul_out], [add_out], name=f"{name_prefix}_Add")
    
    return [u_node, cast_node, reduce_node, mul_node, add_node], add_out

def build_model():
    nodes = []
    initializers = []
    
    # Input: [batch, channels, height, width]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 16, 32, 32])
    
    current_input = "X"
    
    # We will create 4 Subgraphs separated by 3 Boundaries.
    # Each subgraph will have 8 Conv+Relu blocks.
    num_subgraphs = 4
    blocks_per_subgraph = 8
    
    for sg_idx in range(num_subgraphs):
        # 1. Add Subgraph Nodes
        for b_idx in range(blocks_per_subgraph):
            new_nodes, new_inits, current_input = create_conv_relu(
                f"sg{sg_idx}_block{b_idx}", current_input, 16, 16
            )
            nodes.extend(new_nodes)
            initializers.extend(new_inits)
            
        # 2. Add Boundary (if not the last subgraph)
        if sg_idx < num_subgraphs - 1:
            boundary_nodes, current_input = add_boundary(f"boundary{sg_idx}", current_input, initializers)
            nodes.extend(boundary_nodes)
        
    # Output
    Y = helper.make_tensor_value_info(current_input, TensorProto.FLOAT, ["batch", 16, 32, 32])
    
    # Create Graph
    graph_def = helper.make_graph(
        nodes,
        "SyntheticBoundaryModel",
        [X],
        [Y],
        initializers
    )
    
    # Create Model
    op = onnx.OperatorSetIdProto()
    op.version = 13
    model_def = helper.make_model(graph_def, producer_name="generate_test_model.py", opset_imports=[op])
    
    # Infer shapes to ensure validity
    model_def = onnx.shape_inference.infer_shapes(model_def)
    
    # Save
    onnx.save(model_def, "synthetic_boundary_model.onnx")
    print(f"Generated synthetic_boundary_model.onnx successfully.")
    print(f"Model Structure: {num_subgraphs} subgraphs, separated by {num_subgraphs-1} boundaries.")
    print(f"Total Nodes: {len(nodes)}")

if __name__ == "__main__":
    build_model()
