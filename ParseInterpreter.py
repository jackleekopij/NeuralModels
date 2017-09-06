import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


NUMBER_OF_HIDDEN_NODES_FIRST_LAYER = 5
NUMBER_OF_HIDDEN_NODES_SECOND_LAYER= 5
OUTPUT_NODES = 1

# Create vector of possible inputs.
input_data = np.array(["a", "b","c", "d"])

# Create
first_value = np.random.choice(input_data, 10000)
second_value = np.random.choice(input_data, 10000)

# Create a tuple of output
tuple_input = zip(first_value, second_value)
tuple_output = []
for x in tuple_input:
    if (x[0] == x[1]) or (x[0] == "a" and x[1] == "c" ) or (x[0] == "b" and x[1] == "d"):
        tuple_output.append([1.])
    else:
        tuple_output.append([0.])

# Create two hot encoded input network
input_data = [np.zeros([8]) for _ in range(10000)]

# Create the input data

for index, x in enumerate(tuple_input):
    if x[0] == "a":
        input_data[index][0] = 1.
    elif x[0] == "b":
        input_data[index][1] = 1.
    elif x[0] == "c":
        input_data[index][2] = 1.
    elif x[0] == "d":
        input_data[index][3] = 1.
    if x[1] == "a":
        input_data[index][4] = 1.
    elif x[1] == "b":
        input_data[index][5] = 1.
    elif x[1] == "c":
        input_data[index][6] = 1.
    elif x[1] == "d":
        input_data[index][7] = 1.

print "Input data for classification of {0}".format(input_data[1:5])

print tuple_input[1:10]
print tuple_output[1:10]


# Set up input data
x_data = tf.placeholder(dtype=tf.float32, shape=[None,8])
y_data = tf.placeholder(dtype=tf.float32, shape=[None,1])

# Create the weight layers for the network
weights_first_layer = tf.Variable(tf.random_normal(shape=[8,NUMBER_OF_HIDDEN_NODES_FIRST_LAYER]))
weights_second_layer = tf.Variable(tf.random_normal(shape=[NUMBER_OF_HIDDEN_NODES_FIRST_LAYER,NUMBER_OF_HIDDEN_NODES_SECOND_LAYER]))
weights_third_layer = tf.Variable(tf.random_normal(shape=[NUMBER_OF_HIDDEN_NODES_SECOND_LAYER,OUTPUT_NODES]))


# Create the biases for each of the networks
bias_first_layer = tf.Variable(tf.zeros([NUMBER_OF_HIDDEN_NODES_FIRST_LAYER]))
bias_second_layer = tf.Variable(tf.zeros([NUMBER_OF_HIDDEN_NODES_SECOND_LAYER]))
output_bias_layer = tf.Variable(tf.zeros([OUTPUT_NODES]))

#   Node outpus
First_hidden_layer = tf.nn.relu(tf.add(tf.matmul(x_data,weights_first_layer),bias_first_layer))
Second_hidden_layer = tf.nn.relu(tf.add(tf.matmul(First_hidden_layer,weights_second_layer),bias_second_layer ))
Output = tf.nn.relu(tf.add(tf.matmul(Second_hidden_layer, weights_third_layer), output_bias_layer))

#   Calculate the loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = Output, labels = y_data)
training_step = tf.train.AdamOptimizer(0.05).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#
cost_vector = []
for i in range(200):
    sess.run(training_step, feed_dict={x_data:input_data, y_data:tuple_output})

    if i % 1 == 0:
        cost_vector.append(sess.run(loss, feed_dict={x_data:input_data, y_data:tuple_output})[0])
        print i
#   4. Inspect loss function.
    if i % 199 == 0:
        print "Prediction of output FOR JACK: " + str(sess.run(Output, feed_dict={x_data:input_data})[1])
        print "Input values:  {0}".format(sess.run(y_data, feed_dict={y_data:tuple_output})[1])

print "Length of cost vector is {0}".format(len(cost_vector[0]))

x_array = range(len(cost_vector))
print "Length of the x_array {0}".format(len(x_array))
print "The final cost value " + str(cost_vector[len(cost_vector)-2])
plt.plot(x_array, cost_vector)
plt.show()




