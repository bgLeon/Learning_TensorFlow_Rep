# A nearest neighbor learning algorithm example using TensorFlow library.
# This example is using the MNIST database of handwritten digits
# (http://yann.lecun.com/exdb/mnist/)

from __future__ import print_function

import numpy as np
import tensorflow as tf

# Import MINST data from Tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets ("MNIST_data/", one_hot=True)

# Practicing limiting data
Xtr, Ytr = mnist.train.next_batch(5000) #training samples
Xte, Yte = mnist.test.next_batch(200) #testing samples

#tf Graph Input
xtr = tf.placeholder("float", [None,784])
xte = tf.placeholder("float", [784])

# Nearest Neighbour calculation using L1 Distance

# Calculate L1 distance
distance= tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# Prediction: Get min distance index (Nearest neighbor)
prediction = tf.arg_min( distance, 0)

accuracy = 0.

#initializing the variables
init = tf.global_variables_initializer()

#Launch the Graph
with tf.Session() as sess:
	sess.run(init)

	# Loop over test data
	for i in range(len(Xte)):
		# Get nearest neighbour
		nn_index = sess.run(prediction, feed_dict={xtr: Xtr, xte: Xte[i, :]})
		#get nearest neighbor class label and compare it to its true label
		print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
			"True Class:", np.argmax(Yte[i]))
		# Calculate accuracy
		if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
			accuracy += 1./len(Xte)
	print ("Done!")
	print ("Accuracy: ", accuracy)

