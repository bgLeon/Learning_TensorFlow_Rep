#Author Borja González
#From TensorFlow Tutorials
#2/02/2017
import tensorflow as tf

#Load MNIST Data 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#To be able to be interacting building and runnin the session
sess = tf.InteractiveSession()

#Starting the Graph 
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ =  tf.placeholder(tf.float32, shape=[None, 10])

#Weights and bios
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

# Predicted Class and Loss Function
y = tf.matmul(x,W)+b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_)

#Training the Model
train_step = tf.train.GraientDescentOptimizer(0.5).minimize(cross_entropy) 

for i in range(1000):
	batch = mnist.train.next_batch(100)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#Evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))