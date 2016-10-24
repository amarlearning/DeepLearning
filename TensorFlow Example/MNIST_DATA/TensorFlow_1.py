import tensorflow as tf

one = tf.constant(5)
two = tf.constant(6)

result = tf.mul(one,two)

with tf.Session() as session:
	print session.run(result)