import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10
batch_size = 50
# height * width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(data):
    # model neural network with 3 hidden layers
    # create variables for our layers
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}
    # (input_data * weight) + biases
    # hidden layer 1
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    # hidden layer 2
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    # hidden layer 3
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    # output layer
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
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
