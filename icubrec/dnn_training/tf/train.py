#!/usr/bin/env python3

import argparse
import time
import tensorflow as tf

NUM_FEATURES = 60
BATCH_SIZE = 1000
MIN_AFTER_DEQUEUE = 10000


def get_states(phoneme_list_fname):
    file = open(phoneme_list_fname, 'r')
    phonemes = [line.rstrip('\n') for line in file.readlines()]
    states = []
    for p in phonemes:
        for i in range(2, 5):
            label = p + '[' + str(i) + ']'
            states += [label]
    return states


def add_layer(input, nb_units):
    W = tf.Variable(tf.zeros([input.get_shape().as_list()[1], nb_units]))
    b = tf.Variable(tf.zeros([nb_units]))
    return tf.matmul(input, W) + b


def read_csv(filename_queue):
    reader = tf.TextLineReader()
    # _, line = reader.read_up_to(filename_queue, BATCH_SIZE)
    _, line = reader.read(filename_queue)

    # Example and label reading
    defaults = [[1.0]] * NUM_FEATURES + [[1]]
    cols = tf.decode_csv(line, record_defaults=defaults)
    example = tf.stack(cols[0:-1])
    label = cols[-1]
    return example, label


def input_pipeline(filenames, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames)
    example, label = read_csv(filename_queue)
    min_after_dequeue = MIN_AFTER_DEQUEUE
    capacity = min_after_dequeue + 3 * BATCH_SIZE
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=BATCH_SIZE, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


def main(output_size):
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            # File reading
            example_batch, label_batch = input_pipeline(
                ["/DATA/bhigy/corpora/timit/tr.xy.csv"])

        # Defining the network
        input = example_batch
        for i in range(4):
            with tf.name_scope("layer" + str(i + 1)):
                input = tf.nn.relu(add_layer(input, 2000))
        with tf.name_scope("layerout"):
            y = add_layer(input, output_size)
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=label_batch, logits=y))
        gdo = tf.train.GradientDescentOptimizer(0.5)
        train_step = gdo.minimize(cross_entropy)
        correct_prediction = tf.equal(
            tf.argmax(y, 1), tf.cast(label_batch, tf.int64))
        accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

        # TensorBoard summary
        writer = tf.summary.FileWriter('/DATA/bhigy/tf/summaries', sess.graph)
        merged = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer())

        # Starting the filename queue
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        start = time.time()
        for i in range(10):
            # acc, _ = sess.run([accuracy, train_step])
            summary, _ = sess.run([merged, train_step])
            writer.add_summary(summary, i)
        end = time.time()
        print("Elapsed time: " + str(end - start))

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    # Parsing command line
    parser = argparse.ArgumentParser(description='Data loading testing script')
    parser.add_argument(
        'output_size', metavar='output_size', help='Size of the output layer')
    args = parser.parse_args()
    main(args.output_size)
