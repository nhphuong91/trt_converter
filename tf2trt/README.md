Source: https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#using-savedmodel

Firstly, modify & run `tftrt_savedmodel.py` to convert tf saved model to tftrt engine.

Then, use modify & run `load_n_run.py` to import & run tftrt engine file

**NOTE**: the config used here is for `ssd_mobilenet_v2` tf saved model model converting. For converting `ssd_mobilenet_v1` or `ssd_inception` model, pls determine their input/output node (use `saved_model_cli show --dir saved_model/ --tag_set serve  --signature_def serving_default` to view graph structure) & modify `load_n_run.py` accordingly.

**Requirements**:
* numpy
* opencv-python
* tensorflow-gpu<2.0.0 (note: tensorflow on Jetson platform is tensorflow-gpu & fully compatible with trt)
