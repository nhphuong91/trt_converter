Use below cmd to convert tf frozen graph model to uff using convert-to-uff tool

```sh
convert-to-uff <frozen graph>.pb -p config.py -o <output file>.uff
```

Then use modify & run `uff2trt_converter.py` to export to trt engine file

**NOTE**: the config file is used for `ssd_mobilenet_v2` tf model converting.
- For converting `ssd_mobilenet_v1` or `ssd_inception` model, pls refer to [here](https://github.com/AastaNV/TRT_object_detection/tree/master/config) for their `config.py` file
- For converting re-train `ssd_mobilenet_v2` tf model, use `config_retrain.py`

**Requirements**:
* tensorrt
* numpy
* graphsurgeon
* tensorflow
