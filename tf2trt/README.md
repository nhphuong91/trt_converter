Source: 
1. [Build TF-TRT from TF saved model](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#using-savedmodel)
2. [Load, run & display result ref#1](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/Object_detection_image.py)
3. [Load, run & display result ref#2](https://github.com/tensorflow/tensorrt/blob/r1.14+/tftrt/examples/object_detection/object_detection.py)

Firstly, modify & run `tftrt_savedmodel.py` to convert tf saved model to tftrt engine.

Then, use modify & run `load_n_run.py` to import & run tftrt engine file

**NOTE**: the config used here is for `ssd_mobilenet_v2` tf saved model model converting. For converting `ssd_mobilenet_v1` or `ssd_inception` model, pls determine their input/output node (use `saved_model_cli show --dir saved_model/ --tag_set serve  --signature_def serving_default` to view graph structure) & modify `load_n_run.py` accordingly.

**Requirements**:
* numpy
* opencv-python
* tensorflow-gpu<2.0.0 (note: tensorflow on Jetson platform is tensorflow-gpu & fully compatible with trt)
