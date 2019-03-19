import tensorflow as tf
import random
import model_class
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets(".\\MNIST_data\\", one_hot=True)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# initialize
sess = tf.Session()
models = []
num_models = 7
for m in range(num_models):
   models.append(model_class.Model(sess, "model" + str(m), learning_rate))

sess.run(tf.global_variables_initializer())
print('Learning Started!')

# train my model
for epoch in range(training_epochs):
   avg_cost_list = np.zeros(len(models))
   total_batch = int(mnist.train.num_examples / batch_size)
   for i in range(total_batch):
       batch_xs, batch_ys =mnist.train.next_batch(batch_size)

       # train each model
       for m_idx, m in enumerate(models):
           c, _ = m.train(batch_xs, batch_ys)
           avg_cost_list[m_idx] += c / total_batch

   print('Epoch:','%04d'%(epoch + 1),'cost =', avg_cost_list)

print('Learning Finished!')


# Test model and check accuracy
test_size = len(mnist.test.labels)
predictions = np.zeros(test_size * 10).reshape(test_size, 10)


for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy:', m.get_accuracy(mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))