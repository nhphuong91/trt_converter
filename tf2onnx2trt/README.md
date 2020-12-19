**Step 1:** use `tf2onnx` tool to convert tf model to onnx format. See [here](https://github.com/onnx/tensorflow-onnx)
* Method 1: saved model
```sh
python -m tf2onnx.convert --saved-model <saved-model-path> --opset 11 --output model.onnx
```
* Method 2: frozen graph

```sh
python -m tf2onnx.convert --graphdef <frozen_inference_graph>.pb --opset 11 --output ssd_mobilenet.onnx --inputs image_tensor:0 --outputs detection_boxes:0,detection_scores:0,num_detections:0,detection_classes:0
```

* Method 3: `checkpoint` format

```sh
python -m tf2onnx.convert --checkpoint <tf-meta-file-path>.meta --opset 11 --output ssd_mobilenet.onnx --inputs image_tensor:0 --outputs detection_boxes:0,detection_scores:0,num_detections:0,detection_classes:0
```

**Step 2:** modify & run `fixUint8_onnx.py` to fix `UNIT8` issue

**Step 3:** 
* Method 1: modify & run `onnx2trt.py` to convert to trt engine file
* Method 2: use `onnx2trt` module. See [here](https://github.com/onnx/onnx-tensorrt)
```
onnx2trt <fixed_model>.onnx -o <model>.trt
```

**NOTE**: the config used here is for `ssd_mobilenet_v2` tf saved model model converting. For converting `ssd_mobilenet_v1` or `ssd_inception` model, pls determine their input/output node (use `saved_model_cli show --dir saved_model/ --tag_set serve  --signature_def serving_default` to view graph structure) & modify cli accordingly.

**Requirements**:
* tensorrt==7.2.1
* onnx_graphsurgeon
* tf2onnx==1.7.2
* onnx==1.6.0
* onnx2trt==7.2.1.6.0
* tensorflow==1.15.4
* sys
* os

***Known issue***:
```
ERROR: Failed to parse the ONNX file.
In node 7 (importResize): UNSUPPORTED_NODE: Assertion failed: (transformationMode == "half_pixel" || transformationMode == "pytorch_half_pixel" || transformationMode == "align_corners") && "TensorRT only supports half_pixel, pytorch_half_pixel, and align_corners transofmration modes for linear resizes when sizes are provided!"
```
**Fix:** in next major release of TRT (see [here](https://github.com/NVIDIA/TensorRT/issues/386#issuecomment-740326000))