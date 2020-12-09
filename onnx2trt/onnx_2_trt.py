import tensorrt as trt
import sys, os
# sys.path.insert(1, os.path.join(sys.path[0], ".."))

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def build_engine(onnx_file_path, engine_file_path=""):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28 # 28->256MiB - 30->1GiB
        builder.max_batch_size = 1
        # Parse model file
        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please download onnx file first!'.format(onnx_file_path))
            exit(0)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
        # network.get_input(0).shape = [1, 3, 608, 608]
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")
        return engine

def save_engine(engine, engine_dest_path):
    with open(engine_dest_path, "wb") as f:
        f.write(engine.serialize())

def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated network graph in ONNX format:
    onnx_file_path = 'centerface_480_640.onnx'
    engine_file_path = "centerface_480_640.trt"
    trt_engine = build_engine(onnx_file_path, engine_file_path)
    save_engine(trt_engine, engine_file_path)
    print("TRT engine saved!")

if __name__ == '__main__':
    sys.exit(main())
