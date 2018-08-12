import os
import time
import graph_def_util
import tensorflow as tf

tf.flags.DEFINE_string(
    'graph_path', '',
    'The path where the original graph was written to.')
tf.flags.DEFINE_string(
    'optimized_graph_path', 'tmp/models/pnasnet-5_large/optimized_graph.pb',
    'The path where the optimized graph was written to.')
tf.flags.DEFINE_string(
    'input_nodes', 'input:0',
    'Input node names, comma separated.')
tf.flags.DEFINE_string(
    'output_nodes', ' ToInt64:0, TopKV2:0',
    'Output node names, comma separated.')
tf.flags.DEFINE_string(
    'dataset_dir', 'tmp/datasets/imagenet',
    'The directory where the dataset files are stored.')
tf.flags.DEFINE_integer(
    'batch_size', '32',
    'The number of samples in each batch.')
tf.flags.DEFINE_integer(
    'max_batches', '30',
    'The maximum number of batches to use for evaluation.')
tf.flags.DEFINE_integer(
    'image_size', '299',
    'The length of the side of an image.')

FLAGS = tf.flags.FLAGS


def input_fn(batch_size, image_size):
    filename_pattern = 'validation-*'
    filenames = tf.data.TFRecordDataset.list_files(os.path.join(FLAGS.dataset_dir, filename_pattern), shuffle=False)
    dataset = tf.data.TFRecordDataset(filenames)

    def parser(record):
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        image = tf.image.decode_jpeg(parsed['image/encoded'], channels=3)
        image = tf.image.resize_images(image, [image_size, image_size]) / 255.0
        label = parsed['image/class/label']

        return image, label

    dataset = dataset.map(parser, num_parallel_calls=32)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels


def eval_once(graph_path, graph_name, input_nodes, output_nodes,
              batch_size, max_batches, image_size):
    tf.reset_default_graph()
    with tf.Session() as sess:
        graph_def = graph_def_util.graph_pb2.GraphDef()
        graph_def_util.read_def(graph_path, graph_def, graph_name)
        graph_size = os.path.getsize(graph_path)

        tf.import_graph_def(graph_def, name='')

        total = 0
        batch_count = 0

        next_batch = input_fn(batch_size, image_size)

        start = time.time()
        for _ in range(max_batches):
            try:
                features, labels = sess.run(next_batch)
                feed_dict = {input_nodes[0]: features}

                predictions,  _ = sess.run(output_nodes, feed_dict=feed_dict)

                correct_predictions = [label == prediction for label, prediction in zip(labels, predictions[:, 0])]
                accuracy = sum(correct_predictions) / len(correct_predictions)

                total += accuracy
                batch_count += 1
            except tf.errors.OutOfRangeError:
                break
        end = time.time()

        return {
            'accuracy': total / batch_count,
            'time': end - start,
            'ips': batch_count * batch_size / (end - start),
            'size': graph_size}


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    input_nodes = FLAGS.input_nodes.replace(' ', '').split(',')
    output_nodes = FLAGS.output_nodes.replace(' ', '').split(',')

    if not (FLAGS.graph_path and FLAGS.optimized_graph_path):
        raise AttributeError('Both path to the frozen graph and path to the optimized frozen graph must be provided!')
    if not (FLAGS.input_names and FLAGS.output_names):
        raise AttributeError('Input and output tensor names must be provided along with frozen graph path!')
    if not FLAGS.dataset_dir:
        raise AttributeError('Path to the directory where the eval dataset files are stored must be provided!')

    stat = eval_once(FLAGS.graph_path, 'The graph',
                     input_nodes, output_nodes, FLAGS.batch_size, FLAGS.max_batches, FLAGS.image_size)
    optimized_stat = eval_once(FLAGS.optimized_graph_path, 'The optimized graph',
                               input_nodes, output_nodes, FLAGS.batch_size, FLAGS.max_batches, FLAGS.image_size)

    improvement = {}
    for key, _ in stat.items():
        improvement[key] = int((stat[key] - optimized_stat[key]) / stat[key] * 100)

    print('''
        Inference time after optimization: %f   ➜   %f
        Inference time improvement: %d%%
        Accuracy after optimization: %f   ➜   %f
        Accuracy improvement: %d%%
        Image per Second after optimization: %f   ➜   %f
        Image per Second improvement: %d%%
        Graph size after optimization: %d   ➜   %d
        Graph size improvement: %d%%
        ''' % (stat['time'], optimized_stat['time'],
               improvement['time'],
               stat['accuracy'], optimized_stat['accuracy'],
               -improvement['accuracy'],
               stat['ips'], optimized_stat['ips'],
               -improvement['ips'],
               stat['size'], optimized_stat['size'],
               improvement['size']))


if __name__ == '__main__':
    tf.app.run()
