import tensorflow as tf
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(".\\MNIST_data\\", one_hot=True)

use_summary = False
nb_classes = 10
learning_rate = 0.01

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W1 = tf.Variable(tf.random_normal([784, 256]), name='weight1')
b1 = tf.Variable(tf.random_normal([256]), name='bias1')
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]), name='weight2')
b2 = tf.Variable(tf.random_normal([256]), name='bias2')
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, nb_classes]), name='weight3')
b3 = tf.Variable(tf.random_normal([nb_classes]), name='bias3')
hypothesis = tf.matmul(L2, W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

if use_summary:
    W3_hist = tf.summary.histogram("weight3", W3)
    cost_summ = tf.summary.scalar("cost", cost)
    summary = tf.summary.merge_all()

# parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
   
    if use_summary:
        writer = tf.summary.FileWriter('./logs')
        writer.add_graph(sess.graph)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            if use_summary:
                s, c, _ = sess.run([summary, cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
                writer.add_summary(s, global_step=((epoch * total_batch - 1) + i))
            else:
                c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})

            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))


"""
    # Get one and predict
   r = random.randint(0, mnist.test.num_examples - 1)
   print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
   print("Prediction:", sess.run(tf.argmax(hypothesis, 1), 
                        feed_dict={X: mnist.test.images[r:r + 1]}))
                        
   plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
   plt.show()

"""