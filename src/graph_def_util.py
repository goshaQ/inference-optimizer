import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saved_model_pb2


def read_def(filepath, object_def, object_name):
    if tf.gfile.Exists(filepath):
        with tf.gfile.Open(filepath, 'rb') as f:
            object_def.ParseFromString(f.read())
    else:
        raise IOError('%s file does not exist at %s!' % (object_name, filepath))


def write_def(filepath, object_def):
    with tf.gfile.Open(filepath, 'wb') as f:
        f.write(object_def.SerializeToString())
