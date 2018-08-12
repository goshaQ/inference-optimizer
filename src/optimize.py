import os
import graph_def_util
import tensorflow as tf

from tensorflow.python.tools import freeze_graph
from tensorflow.tools import graph_transforms

tf.flags.DEFINE_string(
    'savedmodel_dir', '',  # tmp/models/inception_v3
    'The directory where the saved model was written to.')
tf.flags.DEFINE_string(
    'frozen_graph_path', 'tmp/models/pnasnet-5_large/frozen_graph.pb',
    'The path where the frozen graph was written to.')
tf.flags.DEFINE_string(
    'input_names', 'input',
    'Input node names, comma separated.')
tf.flags.DEFINE_string(
    'output_names', 'ToInt64, TopKV2',
    'Output node names, comma separated.')

FLAGS = tf.flags.FLAGS


def main(_):
    input_names = sorted([name for name in FLAGS.input_names.replace(' ', '').split(',')])
    output_names = sorted([name for name in FLAGS.output_names.replace(' ', '').split(',')])
    output_graph_def = None
    optimized_graph_path = None

    if not (FLAGS.frozen_graph_path or FLAGS.savedmodel_dir):
        raise AttributeError('Either path to the frozen graph or directory of the SavedModel must be provided!')
    if FLAGS.frozen_graph_path and not (FLAGS.input_names and FLAGS.output_names):
        raise AttributeError('Input and output tensor names must be provided along with frozen graph path!')

    if FLAGS.savedmodel_dir:
        savedmodel_pb_filename = 'saved_model.pb'
        path_to_pb = os.path.join(FLAGS.savedmodel_dir, savedmodel_pb_filename)

        signature_def = graph_def_util.saved_model_pb2.SavedModel()
        graph_def_util.read_def(path_to_pb, signature_def, 'The SavedModel')
        signature_def = signature_def.meta_graphs[0].signature_def[
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        input_names = sorted([item.name[:item.name.find(':')] for _, item in signature_def.inputs.items()])
        output_names = sorted([item.name[:item.name.find(':')] for _, item in signature_def.outputs.items()])
        frozen_graph_filename = 'frozen_graph.pb'
        frozen_graph_path = os.path.join(FLAGS.savedmodel_dir, frozen_graph_filename)

        output_graph_def = freeze_graph.freeze_graph(
            input_graph=None, input_saver=None, input_checkpoint=None, input_binary=True,
            clear_devices=True, output_node_names=FLAGS.output_names,
            restore_op_name=None, filename_tensor_name=None, output_graph=frozen_graph_path,
            initializer_nodes=None, input_saved_model_dir=FLAGS.savedmodel_dir)
        optimized_graph_path = FLAGS.savedmodel_dir
    elif FLAGS.frozen_graph_path:
        output_graph_def = graph_def_util.graph_pb2.GraphDef()
        graph_def_util.read_def(FLAGS.frozen_graph_path, output_graph_def, 'The frozen graph')
        optimized_graph_path = os.path.dirname(FLAGS.frozen_graph_path)

    # If you want to apply only 'optimize_for_inference' uncomment the following, but
    # don't forget to remove 'graph_transforms' optimization since they are not compatible.
    #
    # output_graph_def = optimize_for_inference_lib.optimize_for_inference(
    #     input_graph_def=output_graph_def, placeholder_type_enum=tf.float32.as_datatype_enum,
    #     input_node_names=input_names, output_node_names=output_names)

    transforms = [
        'strip_unused_nodes(type=float, shape="1,299,299,3")', 'remove_nodes(op=Identity, op=CheckNumerics)',
        'fold_constants(ignore_errors=true)', 'fold_batch_norms', 'fold_old_batch_norms',
        'quantize_weights', 'quantize_nodes']

    output_graph_def = graph_transforms.TransformGraph(
        input_graph_def=output_graph_def, transforms=transforms,
        inputs=input_names, outputs=output_names)

    optimized_graph_filename = 'optimized_graph.pb'
    optimized_graph_path = os.path.join(optimized_graph_path, optimized_graph_filename)
    graph_def_util.write_def(optimized_graph_path, output_graph_def)


if __name__ == '__main__':
    tf.app.run()
