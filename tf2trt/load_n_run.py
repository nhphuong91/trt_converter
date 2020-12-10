import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

import cv2
import numpy as np
import time

image = cv2.imread("image1.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

timer = time.perf_counter()
with tf.compat.v1.Session() as sess:
    # First load the SavedModel into the session    
    tf.compat.v1.saved_model.loader.load(
        sess, [tf.saved_model.SERVING], "ssd_mobilenet_v1_coco_2018_01_28/saved_model_trt")
    
    timer = time.perf_counter() - timer
    print("Loading time: {}", timer)
    image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
    print("--"*50)
    print(type(detection_boxes))

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
    detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = sess.graph.get_tensor_by_name('num_detections:0')

    print("--"*50)
    timer = time.perf_counter()
    output = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_expanded})
    timer = time.perf_counter() - timer
    print("Inference time: {}", timer)


# input
# image_tensor

# output
# detection_boxes
# detection_scores
# detection_multiclass_scores
# detection_classes
# num_detections
# raw_detection_boxes
# raw_detection_scores
