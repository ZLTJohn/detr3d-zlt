import os
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import tensorflow as tf

if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
from waymo_open_dataset.camera.ops import py_camera_model_ops
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import breakdown_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.protos import submission_pb2
from waymo_open_dataset.utils import box_utils
#more representitive: data/waymo_v131/waymo_format/training/segment-1305342127382455702_3720_000_3740_000_with_camera_labels.tfrecord
FILENAME = 'data/waymo_v131/waymo_format/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord'
#name = '/home/zhengliangtao/pure-detr3d/data/waymo_v131/waymo_format/validation/segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord'
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
dataset_iter = dataset.as_numpy_iterator()

data = next(dataset_iter)
frame = open_dataset.Frame()
frame.ParseFromString(data)