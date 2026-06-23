import onnx

model = onnx.load("model.onnx")

print("=== INPUTS ===")
for inp in model.graph.input:
    print("name :", inp.name)

    shape = []
    for dim in inp.type.tensor_type.shape.dim:
        if dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append(dim.dim_value)

    print("shape:", shape)
    print()

print("=== OUTPUTS ===")
for out in model.graph.output:
    print("name :", out.name)

    shape = []
    for dim in out.type.tensor_type.shape.dim:
        if dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append(dim.dim_value)

    print("shape:", shape)
    print()