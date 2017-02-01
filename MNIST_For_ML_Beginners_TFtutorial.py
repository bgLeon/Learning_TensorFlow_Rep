import tensorflow as tf

# Import MINST data from Tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Preparing a value that I will input when I ask tensorflow to run
# a computation with
x = tf.placeholder(tf.float32, [None, 784])

#Variables: W weights, b bias
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#What I want to minimize
y= tf.nn.softmax(tf.matmul(x, W)+b)
y_= tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#For training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#initializing the variables
init =  tf.global_variables_initializer()


with tf.Session() as sess:
	sess.run(init)

	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("\nAccuracy: ",sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))





