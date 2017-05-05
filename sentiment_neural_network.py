import numpy as np
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import os
from create_sentiment_featuresets import create_feature_sets_and_labels
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 2
batch_size = 100
# height * width
x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')


def neural_network_model(data):
    # model neural network with 3 hidden layers
    # create variables for our layers
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal( [len(train_x[0]), n_nodes_hl1])),
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
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                # c = cost
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            # track where the session is at
            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        # testing
        # run the trained data through the model
        correct = tf.equal(tf.arg_max(prediction, 1), tf.argmax(y, 1))
        # cast changes the variable to a type
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

train_neural_network(x)
