import tensorflow as tf
import numpy as np
import pprint
tf.set_random_seed(777)  # for reproducibility

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

t = np.array([[1.,2.,3.], [4.,5.,6.], [7.,8.,9.], [10.,11.,12.]])
pp.pprint(t)

print(t.ndim)
print(t.shape)

t = tf.constant([1, 2, 3, 4])
tf.shape(t).eval()

t = tf.constant([[1, 2], [3, 4]])
tf.shape(t).eval()

t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
tf.shape(t).eval()

matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[1.], [2.]])
print("Matrix 1 shape", matrix1.shape)
print("Matrix 2 shape", matrix2.shape)
tf.matmul(matrix1, matrix2).eval()

x = [[0, 1, 2],
     [2, 1, 0]]
tf.argmax(x, axis=0).eval()

tf.argmax(x, axis=1).eval()

tf.argmax(x, axis=-1).eval()

sess.close()