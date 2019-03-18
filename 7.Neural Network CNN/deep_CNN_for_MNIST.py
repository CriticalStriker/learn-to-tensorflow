import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(".\\MNIST_data\\", one_hot=True)

learning_rate = 0.001

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prop = tf.placeholder(tf.float32)

X_img = tf.reshape(X, [-1, 28, 28, 1]) # img 28x28x1 (black/white)

# L1 shape=(?, 28 ,28, 1)
# filter 3x3x1 => #32 of filters
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))

# Conv => (?, 28, 28, 32)
# Pool => (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
print (L1)

L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print (L1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prop)




W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
# Conv => (?, 14, 14, 64)
# Pool => (?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
print (L2)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print (L2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prop)




W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print (L3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prop)

L3 = tf.reshape(L3, [-1, 4 * 4 * 128])
print (L3)

# fully connected 1
W4 = tf.get_variable("W4", shape=[4 * 4 * 128, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prop)

# fully connected 2
W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    print('Learning stared. It takes sometime.')

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prop: 0.7})

            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prop: 1}))