"""Give an FP16 ONNX graph FP32 inputs and outputs with one Cast each, as ModelOpt AutoCast keep_io_types does.

Usage: python io_fp32.py IN.onnx OUT.onnx
"""

import sys

import onnx
from onnx import TensorProto, helper

m = onnx.load(sys.argv[1])
g = m.graph
for t in g.input:
    assert t.type.tensor_type.elem_type == TensorProto.FLOAT16, t.name
    old = t.name
    t.type.tensor_type.elem_type = TensorProto.FLOAT
    for n in g.node:
        n.input[:] = [f"{old}_fp16" if i == old else i for i in n.input]
    g.node.insert(
        0, helper.make_node("Cast", [old], [f"{old}_fp16"], to=TensorProto.FLOAT16, name=f"{old}_cast_to_fp16")
    )
for t in g.output:
    assert t.type.tensor_type.elem_type == TensorProto.FLOAT16, t.name
    old = t.name
    for n in g.node:
        n.output[:] = [f"{old}_fp16" if o == old else o for o in n.output]
        n.input[:] = [f"{old}_fp16" if i == old else i for i in n.input]
    t.type.tensor_type.elem_type = TensorProto.FLOAT
    g.node.append(helper.make_node("Cast", [f"{old}_fp16"], [old], to=TensorProto.FLOAT, name=f"{old}_cast_to_fp32"))
onnx.save(m, sys.argv[2])
print("io fp32:", [i.name for i in g.input], [o.name for o in g.output], len(g.node), "nodes")
