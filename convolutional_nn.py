import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)


n_classes = 10
batch_size = 128

# height * width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

# dropout can help with global max problems
# 80% of neurons will be kept
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    # strides take the sides, then each time it moves
    # it takes one pixel at a time. Confirms no missing data
    # pads the same pixels to the edge once sides are reached
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    # move the pooling 2*2 pixels at a time when moving
    # the pool window. Not overlapping
    # pads the same pixels to the edge once sides are reached
    #                        size of window    movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    # model neural network with 3 hidden layers
    # create variables for our layers
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               # the fully connected (fc) layer becomes a feature map
               'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
               'b_conv2': tf.Variable(tf.random_normal([64])),
               # the fully connected layer becomes a feature map
               'b_fc': tf.Variable(tf.random_normal([1024])),
               'out': tf.Variable(tf.random_normal([n_classes]))}

    # reshaping a 784 pixel image to a flat 28 * 28 image
    # relu = rectified linear
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    # 80% of neurons will be kept
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    # calculates the difference of the prediction to the known label that we have
    # both are in one_hot format
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # remember: want to minimise the cost function
    # learning_rate = 0.001 natively in tf
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # opochs = how many cycles of feed forward + backprop
    hm_epochs = 20
    # commence the tf session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # trains on the data
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # chunks through the dataset automatically
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # c = cost
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            # track where the session is at
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        # testing
        # run the trained data through the model
        correct = tf.equal(tf.arg_max(prediction, 1), tf.argmax(y, 1))
        # cast changes the variable to a type
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)
