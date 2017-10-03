import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys




'''The following code is to produce a two-layer neural network to model the 
non-linear componenets of the XOR gate. The process is as follows:
    1. Set up the data. 
    2. Set up the network.
    3. Feed the data into the network.'''



#1. Set up data
# Read the csv
# with open('Input.csv', 'rb') as Inputcsv:
#     input = csv.reader(Inputcsv, delimiter=' ', quotechar="'")
#     input_data = [x for x in input]
#
# input_array_for_booleans = []
#
# for index in input_data:
#     input_array_for_booleans.append([float(y) for y in index[0].split(",")])
#
# print input_array_for_booleans
#
# with open('Output.csv', 'rb') as Outputcsv:
#     output = csv.reader(Outputcsv, delimiter=' ', quotechar="'")
#     output_data = [y for x in output]
#
# input_array_for_target = []
#
# for index in output_data:
#     input_array_for_target.append([float(y) for y in index[0].split(",")])
#
# print input_array_for_target
#
# input_booleans = np.array(input_array_for_booleans)
# target = np.array(input_array_for_target)
def training():
    # Set up directory to save the built models to.
    export_directory = os.path.dirname(os.path.realpath(sys.argv[0]))
    builder = tf.saved_model.builder.SavedModelBuilder(export_directory + "Test6")
    print export_directory

    # Create input data for the model.
    input_booleans = np.array([[0.,0.], [1.,0.], [0.,1.], [1.,1.]])
    target = np.array([[0.],[1.],[1.],[0.]])


    # Set up network
    x_data = tf.placeholder(tf.float32, shape=[4,2])
    y_data = tf.placeholder(tf.float32, shape=[4,1])


    # Weights of the layers
    first_layered_weights = tf.Variable(tf.random_uniform([2,5], -2, 2))
    second_layer_weights = tf.Variable(tf.random_uniform([5,1], -2, 2))

    # Bias for each layer
    bias_1 = tf.Variable(tf.zeros([5]))
    bias_2 = tf.Variable(tf.zeros([1]))


    # Multliplication of the layers
    Hidden_layer_output = tf.nn.relu(tf.add(tf.matmul(x_data,first_layered_weights),bias_1))
    Output = tf.sigmoid(tf.add(tf.matmul(Hidden_layer_output, second_layer_weights),bias_2))

    # Define the cost function
    cost = - tf.reduce_mean( (target * tf.log(Output)) + (1 - target) * tf.log(1.0 - Output)  )
    training_step = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

    #3. Run the computational graph.
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)




    # Feed data into the graph for 10,000 iterations
    # Store the cost values
    cost_array = []

    saver = tf.train.Saver()

    for i in range(10):
        sess.run(training_step, feed_dict={x_data:input_booleans, y_data:target})



        if i % 10 == 0:
            print("Output value: ", sess.run(Output, feed_dict={x_data:input_booleans, y_data:target}))
            print("Cost: ", sess.run(cost, feed_dict={x_data:input_booleans, y_data:target}))
            print("First layered weights: ")
            print(sess.run(first_layered_weights))
            print("First layered bias: ")
            print(sess.run(bias_1))
            print("Second layered weights: ")
            print(sess.run(second_layer_weights))
            print("Second layered bias: ")
            print(sess.run(bias_2))
        cost_array.append(sess.run(cost, feed_dict={x_data:input_booleans, y_data:target}))
        if i == 9999:
            print "Output prediction"
            print sess.run(Output, feed_dict={x_data: input_booleans, y_data: target})
    builder.add_meta_graph_and_variables(sess, ["Tag"],signature_def_map = {"TF input":tf.saved_model.signature_def_utils.predict_signature_def(inputs= {"inputs":x_data},outputs={"outputs":cost})})
    builder.save()
    #saver.save(sess, 'saved_model', global_step=9999)
    print "Cost function over iterations:"
    print cost_array
    plt.plot(cost_array)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss function over 500 iterations.")
    plt.show()
    print

def main(_):
    training()

if __name__ == "__main__":
    tf.app.run()

