import tensorflow as tf
import pandas
import numpy as np
import matplotlib.pyplot as plt

# Set parameters
ARRAY_LENGTH = 500
batch_size = 100

# Create the
oil_production = np.linspace(5,500,ARRAY_LENGTH)
gas_production = np.linspace(10, 5000, ARRAY_LENGTH)

oil_and_gas_production = np.column_stack((oil_production,gas_production))

print "[Zipped data]: " + str(oil_and_gas_production[1:5][1][0])

print "[Oil production]: length"
print len(oil_production)
print "[Gas production]: length"
print len(gas_production)

# Create a function for the
print "[Create company performance linear]"
company_performance_linear = 25.5 * oil_production + 15.2 * gas_production

print "[Create company performance interaction effect]"
company_performance_interaction = 25.5 * oil_production + 15.2 * gas_production + 0.03 * oil_production * gas_production

print "[Company performance linear]: plot"
#plt.plot(company_performance_linear)
#plt.show()

print "[Company perforamnce interaction]: plot"
#plt.plot(company_performance_interaction)
#plt.show()


print "[Normal distribution]: Sample plot"
x_axis = np.linspace(-3,3,10000)


# plt.plot(sorted(np.random.normal(0,1,10000)),x_axis)
# plt.show()

print "[Round function]: " + str(round(len(oil_production)))

####
    # Training and test set
####
#   Create an index of random values.
print "[Type of oil production list]" + str(type(len(oil_production)))
train_indices = np.random.choice(int(len(oil_and_gas_production)), int(round(len(oil_and_gas_production) * 0.8)), replace = False)
test_indices = np.array(list(set(range(len(oil_and_gas_production))) - set(train_indices)))


oil_and_gas_production_training =oil_and_gas_production[train_indices]
oil_and_gas_production_test =  oil_and_gas_production[test_indices]

company_performance_linear_train = company_performance_linear[train_indices]
company_performance_linear_test = company_performance_linear[test_indices]

company_performance_interaction_train = company_performance_interaction[train_indices]
company_performance_interaction_test = company_performance_interaction[test_indices]

print "[Length of oil_and_gas_production]" + str(len(oil_and_gas_production[1]))


####
    # Network
####
def init_weight(shape,st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return weight

def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return bias

oil_and_gas_production_data = tf.placeholder(shape=[None,2], dtype=tf.float32)
company_performance_target = tf.placeholder(shape=[None,1], dtype = tf.float32)

def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return tf.nn.relu(layer)

# Create hidden layer
weight_1 = init_weight(shape=[2,4], st_dev=10.0)
bias_1 = init_bias(shape= [4], st_dev = 10.0)
hidden_layer = fully_connected(oil_and_gas_production_data, weight_1, bias_1)

# Create output layer
weight_output = init_weight(shape=[4,1], st_dev=10.0)
bias_output = init_bias(shape=[1], st_dev=10.0)
output_neuron = fully_connected(hidden_layer, weight_output, bias_output)

# Create a loss function to optimize the regression
loss = tf.reduce_mean(tf.abs(company_performance_target - output_neuron))
optimizer = tf.train.AdamOptimizer(0.001)
train_network = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Storing analysis of the training
loss_vec = []
test_loss = []
for i in range(500000):
    rand_index = np.random.choice(len(oil_and_gas_production_training), size = batch_size)

    # Take out a random batch
    rand_x = oil_and_gas_production_training[rand_index]
    rand_y = np.transpose([company_performance_linear[rand_index]])

    sess.run(train_network, feed_dict = {oil_and_gas_production_data:rand_x,company_performance_target:rand_y})

    temp_loss = sess.run(loss, feed_dict={oil_and_gas_production_data:rand_x, company_performance_target: rand_y })
    loss_vec.append(temp_loss)

    test_temp_loss = sess.run(loss, feed_dict = {oil_and_gas_production_data:oil_and_gas_production_test, company_performance_target: np.transpose([company_performance_linear_test])})
    test_loss.append(test_temp_loss)
    if (i + 1) % 25 == 0:
        print "Generation: " + str(i+1) + " . Loss = " + str(temp_loss)



plt.plot(loss_vec, "k-", label= 'Train Loss')
plt.plot(test_loss, 'r--', label = "Test Loss")
plt.title("Loss per Generation")
plt.xlabel("Generations")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()