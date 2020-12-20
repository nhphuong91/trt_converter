import onnx_graphsurgeon as gs
import onnx
import numpy as np

graph = gs.import_onnx(onnx.load("model.onnx"))
for inp in graph.inputs:
    inp.dtype = np.float32

onnx.save(gs.export_onnx(graph), "updated_model.onnx")
