import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nodes_HL1 = 500
nodes_HL2 = 500
nodes_HL3 = 500

n_Classes = 10
Batch_size = 100

features = tf.placeholder("float", [None, 784])
labels = tf.placeholder("float")

def Neural_Network_Model(data):
	hidden_1_layer = {
						'weight' : tf.Variable(tf.random_normal([784, nodes_HL1])),
						'biases' : tf.Variable(tf.random_normal([nodes_HL1]))
	}
	hidden_2_layer = {
						'weight' : tf.Variable(tf.random_normal([nodes_HL1, nodes_HL2])),
						'biases' : tf.Variable(tf.random_normal([nodes_HL2]))
	}
	hidden_3_layer = {
						'weight' : tf.Variable(tf.random_normal([nodes_HL2, nodes_HL3])),
						'biases' : tf.Variable(tf.random_normal([nodes_HL3]))
	}
	output_layer = {
						'weight' : tf.Variable(tf.random_normal([nodes_HL3, n_Classes])),
						'biases' : tf.Variable(tf.random_normal([n_Classes]))
	}

	# (input* weight) + biases
	layer_1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['biases'])
	layer_1 = tf.nn.relu(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1, hidden_2_layer['weight']), hidden_2_layer['biases'])
	layer_2 = tf.nn.relu(layer_2)

	layer_3 = tf.add(tf.matmul(layer_2, hidden_3_layer['weight']), hidden_3_layer['biases'])
	layer_3 = tf.nn.relu(layer_3)

	output = tf.matmul(layer_3, output_layer['weight']) + output_layer['biases']

	return output


def Train_Neural_Network(features):
	prediction = Neural_Network_Model(features)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, labels))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	n_epochs = 20

	with tf.Session() as session:
		session.run(tf.initialize_all_variables())

		for epochs in range(n_epochs):
			epochs_loss = 0
			for _ in range(int(mnist.train.num_examples/Batch_size)):
				e_features, e_labels = mnist.train.next_batch(Batch_size)
				_, c = session.run([optimizer, cost], feed_dict = {features: e_features, labels: e_labels})
				epochs_loss += c

			print "Epochs : " + str(epochs) + " completed out of " + str(n_epochs) + " Epochs, where loss is : " + str(epochs_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print "Neural Network Model's Accuracy : " + str(accuracy.eval({features : mnist.test.images, labels : mnist.test.labels}))
		print "Done. Your build exited with 0"
Train_Neural_Network(features)