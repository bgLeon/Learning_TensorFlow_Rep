#Author Borja González
#30/12/2016
import tensorflow as tf
#Constants
print ('\n')
a = tf.constant(2)
b= tf.constant(3)

with tf.Session() as sess:
	print ("")
	print ("a=2, b=3")
	print ("Addition with constants: %i" % sess.run(a+b))
	print ("Multiplication with constants: %i" % sess.run(a*b))

#variables
a= tf.placeholder(tf.int16)
b= tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.mul(a, b)

with tf.Session() as sess:
	print ("")
	print ("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
	print ("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

#Matrix

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    result = sess.run(product)
    print (result)