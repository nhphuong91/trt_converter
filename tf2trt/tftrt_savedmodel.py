import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt


import cv2
import numpy as np
converter = trt.TrtGraphConverter(
    input_saved_model_dir="ssd_mobilenet_v1_coco_2018_01_28/saved_model",
    max_workspace_size_bytes=(11<32),
    precision_mode="FP16",
    maximum_cached_engines=100)
converter.convert()
converter.save("ssd_mobilenet_v1_coco_2018_01_28/saved_model_trt")

