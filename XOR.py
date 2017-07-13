import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''The following code is to produce a two-layer neural network to model the 
non-linear componenets of the XOR gate. The process is as follows:
    1. Set up the data. 
    2. Set up the network.
    3. Feed the data into the network.'''

# Set up data
input_booleans = np.array([[0.,0.], [1.,0.], [0., 1.], [1., 1.]])
target = np.array([[0.], [1.], [1.], [0.]])


# Set up network
x_data = tf.placeholder(tf.float32, shape=[4,2])
y_data = tf.placeholder(tf.float32, shape=[4,1])


# Weights of the layers
first_layered_weights = tf.Variable(tf.random_uniform([2,2], -1, 1))
second_layer_weights = tf.Variable(tf.random_uniform([2,1], -1, 1))

# Bias for each layer
bias_1 = tf.Variable(tf.zeros([2]))
bias_2 = tf.Variable(tf.zeros([1]))


# Multliplication of the layers
Hidden_layer_output = tf.sigmoid(tf.add(tf.matmul(x_data,first_layered_weights),bias_1))
Output = tf.sigmoid(tf.add(tf.matmul(Hidden_layer_output, second_layer_weights),bias_2))

# Define the cost function
cost = cost = - tf.reduce_mean( (target * tf.log(Output)) + (1 - target) * tf.log(1.0 - Output)  )
training_step = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

#3. Run the computational graph.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)




# Feed data into the graph for 10,000 iterations
# Store the cost values
cost_array = []

for i in range(500):
    sess.run(training_step, feed_dict={x_data:input_booleans, y_data:target})


    if i % 10 == 0:
        print("Output value: ", sess.run(Output, feed_dict={x_data:input_booleans, y_data:target}))
        print("Cost: ", sess.run(cost, feed_dict={x_data:input_booleans, y_data:target}))
        print("First layered weights: ")
        print(sess.run(first_layered_weights))
        print("First layered bias: ")
        print(sess.run(bias_1))
        print("Second layered weights: ")
        print(sess.run(first_layered_weights))
        print("Second layered bias: ")
        print(sess.run(bias_2))
    cost_array.append(sess.run(cost, feed_dict={x_data:input_booleans, y_data:target}))
plt.plot(cost_array)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss function over 500 iterations.")
plt.show()
