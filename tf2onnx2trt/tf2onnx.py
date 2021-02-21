# This is a failed attempt!!!


import tensorflow as tf
import tensorrt as trt
import graphsurgeon as gs
import tf2onnx

# UFF conversion functionality

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

    @staticmethod
    def get_input_width():
        return ModelData.INPUT_SHAPE[2]

def ssd_unsupported_nodes_to_plugin_nodes(ssd_graph):
    """Makes ssd_graph TensorRT comparible using graphsurgeon.

    This function takes ssd_graph, which contains graphsurgeon
    DynamicGraph data structure. This structure describes frozen Tensorflow
    graph, that can be modified using graphsurgeon (by deleting, adding,
    replacing certain nodes). The graph is modified by removing
    Tensorflow operations that are not supported by TensorRT's UffParser
    and replacing them with custom layer plugin nodes.

    Note: This specific implementation works only for
    ssd_inception_v2_coco_2017_11_17 & ssd_mobilenet_v2_coco_2018_03_29 network.

    Args:
        ssd_graph (gs.DynamicGraph): graph to convert
    Returns:
        gs.DynamicGraph: UffParser compatible SSD graph
    """
    # Create TRT plugin nodes to replace unsupported ops in Tensorflow graph
    channels = ModelData.get_input_channels()
    height = ModelData.get_input_height()
    width = ModelData.get_input_width()

    all_assert_nodes = ssd_graph.find_nodes_by_op("Assert")
    ssd_graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)
    all_identity_nodes = ssd_graph.find_nodes_by_op("Identity")
    ssd_graph.forward_inputs(all_identity_nodes)

    Squeeze = ssd_graph.find_nodes_by_op('Squeeze')
    ssd_graph.forward_inputs(Squeeze)

    Input = gs.create_plugin_node(name="Input",
        op="Placeholder",
        dtype=tf.float32,
        shape=[1, channels, height, width])

    PriorBox = gs.create_plugin_node(name="GridAnchor", op="GridAnchor_TRT",
        minSize=0.2,
        maxSize=0.95,
        aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
        variance=[0.1,0.1,0.2,0.2],
        featureMapShapes=[19, 10, 5, 3, 2, 1],
        numLayers=6
    )

    NMS = gs.create_plugin_node(
        name="NMS",
        op="NMS_TRT",
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=1e-8,
        nmsThreshold=0.6,
        topK=100,
        keepTopK=100,
        numClasses=91,
        inputOrder=[1, 0, 2],
        confSigmoid=1,
        isNormalized=1
    )

    concat_priorbox = gs.create_node(
        "concat_priorbox",
        op="ConcatV2",
        dtype=tf.float32,
        axis=2
    )

    concat_box_loc = gs.create_plugin_node(
        "concat_box_loc",
        op="FlattenConcat_TRT",
        dtype=tf.float32,
        axis=1,
        ignoreBatch=0
    )

    concat_box_conf = gs.create_plugin_node(
        "concat_box_conf",
        op="FlattenConcat_TRT",
        dtype=tf.float32,
        axis=1,
        ignoreBatch=0
    )

    # Create a mapping of namespace names -> plugin nodes.
    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": PriorBox,
        "Postprocessor": NMS,
        "Preprocessor": Input,
        "Cast": Input,
        "ToFloat": Input,
        "image_tensor": Input,
        "Concatenate": concat_priorbox,
        "MultipleGridAnchorGenerator/Concatenate": concat_priorbox,
        "MultipleGridAnchorGenerator/Identity": concat_priorbox,
        "concat": concat_box_loc,
        "concat_1": concat_box_conf
    }

    # Create a new graph by collapsing namespaces
    ssd_graph.collapse_namespaces(namespace_plugin_map)
    # Remove the outputs, so we just have a single output node (NMS).
    # If remove_exclusive_dependencies is True, the whole graph will be removed!
    ssd_graph.remove(ssd_graph.graph_outputs, remove_exclusive_dependencies=False)

    ssd_graph.find_nodes_by_op("NMS_TRT")[0].input.remove("Input")
    return ssd_graph

"""Takes frozen .pb graph, converts it to .uff and saves it to file.

Args:
    model_path (str): .pb model path
    output_uff_path (str): .uff path where the UFF file will be saved
    silent (bool): if False, writes progress messages to stdout

"""
model_path = 'frozen_inference_graph.pb'
output_onnx_path = 'frozen_inference_graph.onnx'

dynamic_graph = gs.DynamicGraph(model_path)
dynamic_graph = ssd_unsupported_nodes_to_plugin_nodes(dynamic_graph)

tf.compat.v1.graph_util.import_graph_def(dynamic_graph.as_graph_def())
onnx_graph = tf2onnx.tfonnx.process_tf_graph(tf.get_default_graph(), 
                                                input_names=["image_tensor:0"], 
                                                output_names=["detection_boxes:0", "detection_scores:0", "num_detections:0", "detection_classes:0"])
model_proto = onnx_graph.make_model()
with open(output_onnx_path, "wb") as f:
    f.write(model_proto.SerializeToString())
