**Step 1:** download insightface MXNet model from official github repo. See [here](https://github.com/deepinsight/insightface/wiki/Model-Zoo)
Use the model `LResNet100E-IR,ArcFace@ms1m-refine-v2`

Script for converting to onnx model is taken from [here](https://github.com/linghu8812/tensorrt_inference)

**Step 2:** convert model to onnx model `export_onnx.py`
```sh
python export_onnx.py --prefix="<prefix to load model>"
```

Ex: model locates in `./model-r100-ii/`, then <prefix to load model> is `./model-r100-ii/model`

***ATTENTION:*** preprocess the input image to the same size set in `input_shape` before inference

**Step 3:** export to trt engine using `onnx_2_trt.py`
Modify `onnx_file_path` & `engine_file_path` name

**Requirements**:
* tensorrt>=7.1.3
* onnx_graphsurgeon
* tf2onnx==1.7.2
* onnx==1.2.1
* onnx2trt==7.2.1.6.0
* tensorflow==1.15.4
* sys
* os
