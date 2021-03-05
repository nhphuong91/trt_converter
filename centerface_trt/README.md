**Step 1:** download centerface onnx model from official github repo. See [here](https://github.com/Star-Clouds/CenterFace)
The model locates at `/models/onnx`

**Step 2:** re-export the onnx model again using `onnx4trt_export.py`
Modify the `input_size` to targeted resolution,  model path & output model name 
```python
input_size =(<height>,<width>)
```
***ATTENTION:*** preprocess the input image to the same size set in `input_size` before inference

**Step 3:** export to trt engine using `onnx_2_trt.py`
Modify `onnx_file_path` & `engine_file_path` name

**Requirements**:
* tensorrt>=7.1.3
* onnx_graphsurgeon
* tf2onnx==1.7.2
* onnx==1.6.0
* onnx2trt==7.2.1.6.0
* tensorflow==1.15.4
* sys
* os
