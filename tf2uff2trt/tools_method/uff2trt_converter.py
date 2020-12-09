import tensorrt as trt
import numpy as np

# from utils.model import ModelData

# This class contains converted (UFF) model metadata
class ModelData(object):
    # Name of input node
    INPUT_NAME = "Input"
    # CHW format of model input
    INPUT_SHAPE = (3, 300, 300)
    # Name of output node
    OUTPUT_NAME = "NMS"

    @staticmethod
    def get_input_channels():
        return ModelData.INPUT_SHAPE[0]

    @staticmethod
    def get_input_height():
        return ModelData.INPUT_SHAPE[1]

def build_engine(uff_model_path, trt_logger, trt_engine_datatype=trt.DataType.FLOAT, batch_size=1, silent=False):
    with trt.Builder(trt_logger) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 30
        if trt_engine_datatype == trt.DataType.HALF:
            builder.fp16_mode = True
        builder.max_batch_size = batch_size

        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output("MarkOutput_0")
        parser.parse(uff_model_path, network)

        if not silent:
            print("Building TensorRT engine. This may take few minutes.")

        return builder.build_cuda_engine(network)

def save_engine(engine, engine_dest_path):
    buf = engine.serialize()
    with open(engine_dest_path, 'wb') as f:
        f.write(buf)

uff_model_path = 'ssd_mobilenet.uff' #'frozen_inference_graph.uff'
trt_engine_path = 'ssd_trt_engine.trt'
trt_engine_datatype=trt.DataType.FLOAT
batch_size=1

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
trt_runtime = trt.Runtime(TRT_LOGGER)

# Display requested engine settings to stdout
print("TensorRT inference engine settings:")
print("  * Inference precision - {}".format(trt_engine_datatype))
print("  * Max batch size - {}\n".format(batch_size))

trt_engine = build_engine(
                uff_model_path, TRT_LOGGER,
                trt_engine_datatype=trt_engine_datatype,
                batch_size=batch_size)
save_engine(trt_engine, trt_engine_path)