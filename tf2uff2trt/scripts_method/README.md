Firstly, modify & run `tf2uff_converter.py` to convert tf frozen graph to uff.

Then, use modify & run `uff2trt_converter.py` to export to trt engine file

**NOTE**: the config used here is for `ssd_mobilenet_v2` tf model converting.
- For converting `ssd_mobilenet_v1` or `ssd_inception` model, pls refer to [here](https://github.com/AastaNV/TRT_object_detection/tree/master/config) for their config & modify the function `ssd_unsupported_nodes_to_plugin_nodes` & class `ModelData` accordingly.
- For converting re-train `ssd_mobilenet_v2` tf model, use [config_retrain.py](../tools_method/config_retrain.py) to modify the function `ssd_unsupported_nodes_to_plugin_nodes`

**Requirements**:
* tensorrt
* numpy
* graphsurgeon
* tensorflow
* uff
