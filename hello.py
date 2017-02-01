import tensorflow as tf

hello=tf.constant('Hello,TensorFlow!')
sess= tf.Session()
print ('\n')
print (sess.run(hello))
print ('\n')
bor=tf.constant('Me llamo Borja')
print (sess.run(bor))
