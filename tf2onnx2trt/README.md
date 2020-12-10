Firstly, use `tf2onnx` tool to convert tf model to onnx format. See [here](https://github.com/onnx/tensorflow-onnx)
* Method 1: saved model
```sh
python -m tf2onnx.convert --saved-model <saved-model-path> --opset 10 --output model.onnx
```
* Method 2: frozen graph

```sh
python -m tf2onnx.convert --graphdef <frozen_inference_graph>.pb --opset 10 --output ssd_mobilenet.onnx --inputs image_tensor:0 --outputs detection_boxes:0,detection_scores:0,num_detections:0,detection_classes:0
```

* Method 3: `checkpoint` format

```sh
python -m tf2onnx.convert --checkpoint <tf-meta-file-path>.meta --opset 10 --output ssd_mobilenet.onnx --inputs image_tensor:0 --outputs detection_boxes:0,detection_scores:0,num_detections:0,detection_classes:0
```

Then, use modify & run `onnx2trt.py` to convert to trt engine file

**NOTE**: the config used here is for `ssd_mobilenet_v2` tf saved model model converting. For converting `ssd_mobilenet_v1` or `ssd_inception` model, pls determine their input/output node (use `saved_model_cli show --dir saved_model/ --tag_set serve  --signature_def serving_default` to view graph structure) & modify cli accordingly.

**Requirements**:
* tensorrt
* sys
* os
