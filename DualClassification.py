import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn

################################################
    # 1. Create training data using numpy.
    # 2. Define network architecture.
    # 3. Run computation graph.
    # 4. Inspect loss function.
################################################
# 0.

FIRST_HIDDEN_NODES = 5
SECOND_HIDDEN_NODES = 5
THIRD_HIDDEN_NODES = 3



# 1. Create training data using numpy.
'''
Creating training data which will have three inputs. Which all follow a normal distribution each with their own 
mean and standard deviation. Output is a three-dimensional binary array (indicator vector) showing the vectors 
variables are indicated. 
GOAL: see if a network can uncover the rules that we used to create the data.
'''
x1_wex= np.random.normal(400,50, 5000)
x2_conversation = np.random.uniform(0,1, 5000)
x3_search = np.random.normal(100, 30, 5000)


output_data = []
input_data =np.array([])
for index in xrange(5000):
    binary_vector = np.zeros(3)
    if x1_wex[index] > 500:
        binary_vector[0] = int (1)
    elif x2_conversation[index] > 0.75 :
        binary_vector[1] = int(1)

    if 300 < x1_wex[index] < 500 and x2_conversation[index] > 0.15:
        binary_vector[0] = int(1)
        binary_vector[1] = int(1)

    if x3_search[index] < 80:
        binary_vector[2] = int(1)
    output_data.append(binary_vector)
print "[Plot] Plotting x1_wex against binary_vector"



x1_wex_binary = [x[0] for x in output_data]
x2_conversation_binary = [x[1] for x in output_data]

zipped_input = np.array(zip(x1_wex, x2_conversation, x3_search))
print zipped_input[1:5]

union = len([x for index,x in enumerate(x1_wex) if 300 < x1_wex[index] < 500 and x2_conversation_binary[index] < 0.15])

probability = float(union) / float(len([x for index,x in enumerate(x1_wex) if 300 < x1_wex[index] < 500 ]))

print probability
#
# Create a scatter plot
# plt.plot(x1_wex, x1_wex_binary, "o")
# plt.show()
#
# print "[Plot] Plotting x2_conversation against binary_vector"
#
# plt.plot(x2_conversation, x2_conversation_binary)
# plt.show()

# 2. Define network architecture.

x_data = tf.placeholder(dtype=tf.float32, shape=[None,3])
y_data = tf.placeholder(dtype=tf.float32, shape=[None,3])

#    Weights
weights_first_layer = tf.Variable(tf.random_normal(shape=[3,FIRST_HIDDEN_NODES]))
weights_second_layer = tf.Variable(tf.random_normal(shape=[FIRST_HIDDEN_NODES,SECOND_HIDDEN_NODES]))
weights_third_layer = tf.Variable(tf.random_normal(shape=[SECOND_HIDDEN_NODES,THIRD_HIDDEN_NODES]))


#    Bias
bias_first_layer = tf.Variable(tf.zeros([FIRST_HIDDEN_NODES]))
bias_second_layer = tf.Variable(tf.zeros([SECOND_HIDDEN_NODES]))
output_bias_layer = tf.Variable(tf.zeros([THIRD_HIDDEN_NODES]))

#   Node outpus
First_hidden_layer = tf.nn.relu(tf.add(tf.matmul(x_data,weights_first_layer),bias_first_layer))
Second_hidden_layer = tf.nn.relu(tf.add(tf.matmul(First_hidden_layer,weights_second_layer),bias_second_layer ))
Output = tf.nn.relu(tf.add(tf.matmul(Second_hidden_layer, weights_third_layer), output_bias_layer))

#   Calculate the loss
cost = tflearn.objectives.binary_crossentropy(Output, y_data)
training_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
#
#
#
# #  3. Run computation graph.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#
cost_vector = []
for i in range(2000):
    sess.run(training_step, feed_dict={x_data:zipped_input, y_data:output_data})

    if i % 1 == 0:
        cost_vector.append(sess.run(cost, feed_dict={x_data:zipped_input, y_data:output_data}))
        print i
#   4. Inspect loss function.
    if i % 1999 == 0:
        print "Prediction of output: " + str(sess.run(Output, feed_dict={x_data:zipped_input}))


x_array = range(len(cost_vector))
print "The cost vector " + str(cost_vector)
plt.plot(x_array, cost_vector)
plt.show()