import tensorflow as tf


def run_training():
    x = tf.placeholder("float") # Create a placeholder 'x'
    w = tf.Variable(5.0, name="weights")
    y = tf.multiply(w, x)

    with tf.Session() as sess:
        # Add the variable initializer Op.
        tf.global_variables_initializer().run()

        print(sess.run(y, feed_dict={x: 1.0}))
        print(sess.run(y, feed_dict={x: 2.0}))


def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()