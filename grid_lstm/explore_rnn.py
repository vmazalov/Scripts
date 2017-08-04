# Import the relevant components
from __future__ import print_function
import numpy as np
import sys
import os
from grid_lstm import *
import cntk as C

# Select the right target device when this notebook is being tested:
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))

# Define the network
input_dim = 80
num_output_classes = 2

# Ensure that we always get the same results
np.random.seed(0)

#import pdb;pdb.set_trace()

# Helper function to generate a random data sample
def generate_random_data_sample(sample_size, feature_dim, num_classes):
    # Create synthetic data using NumPy.
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable
    X = (np.random.randn(sample_size, feature_dim)+3) * (Y+1)

    # Specify the data type to match the input variable used later in the tutorial
    # (default type is double)
    X = X.astype(np.float32)

    # convert class 0 into the vector "1 0 0",
    # class 1 into the vector "0 1 0", ...
    class_ind = [Y==class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y

mysamplesize = 32
features, labels = generate_random_data_sample(mysamplesize, input_dim, num_output_classes)

feature =  C.sequence.input(input_dim, np.float32)

# Define a dictionary to store the model parameters
mydict = {}

def linear_layer(input_var, output_dim):

    """

    :rtype: object
    """
    input_dim = input_var.shape[0]
    weight_param = C.parameter(shape=(input_dim, output_dim))
    bias_param = C.parameter(shape=(output_dim))

    mydict['w'], mydict['b'] = weight_param, bias_param

    return C.times(input_var, weight_param) + bias_param


def LSTM_func(input_dim, out_dim):
    #print (input_dim, out_dim)
    W = C.parameter(shape=(input_dim, out_dim), init=C.glorot_uniform())
    B = C.parameter(shape=(out_dim), init=0)
    def lstm(dh, dc, input):
        a = C.times(input, W) + B + dh + dc
        ddh = dh + dc
        return (a, ddh)

    return C.BlockFunction("my_lstm", "my_name")(lstm)

def LSTM_layer(input,
               output_dim):
    # we first create placeholders for the hidden state and cell state which we don't have yet
    dh = C.placeholder(shape=(output_dim), dynamic_axes=input.dynamic_axes)
    dc = C.placeholder(shape=(output_dim), dynamic_axes=input.dynamic_axes)

    # we now create an LSTM_cell function and call it with the input and placeholders
    #print ("output_dim ", output_dim, input.shape[0])
    LSTM_cell = LSTM_func(input.shape[0],output_dim) # C.layers.LSTM(output_dim) #
    #print ("output_dim ", output_dim, dh, dc, input.shape)
    f_x_h_c = LSTM_cell(dh, dc, input)
    #print ("output_dim ", output_dim, input.shape)
    h_c = f_x_h_c.outputs

    # we setup the recurrence by specifying the type of recurrence (by default it's `past_value` -- the previous value)
    h = C.sequence.past_value(h_c[0])
    c = C.sequence.past_value(h_c[1])

    replacements = { dh: h.output, dc: c.output }
    f_x_h_c.replace_placeholders(replacements)

    h = f_x_h_c.outputs[0]
    c = f_x_h_c.outputs[1]

    # and finally we return the hidden state and cell state as functions (by using `combine`)
    return C.combine([h]), C.combine([c])

z_lin = linear_layer(feature, 80)
#print(z_lin.shape[0])
#(z_lin, m2, c1, c2) = GLSTM_layer(z_lin, 256)
#(z_lin, output_c) = LSTM_layer(z_lin, 256)
z_lin = freq_grid(z_lin, 256)
z = linear_layer(z_lin, num_output_classes)

label = C.sequence.input(num_output_classes, np.float32)
loss = C.cross_entropy_with_softmax(z, label)

eval_error = C.classification_error(z, label)

# Instantiate the trainer object to drive the model training
learning_rate = 0.5
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
learner = C.sgd(z.parameters, lr_schedule)
trainer = C.Trainer(z, (loss, eval_error), [learner])

# Define a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss, eval_error = "NA", "NA"

    if mb % frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}".format(mb, training_loss, eval_error))

    return mb, training_loss, eval_error

# Initialize the parameters for the trainer
minibatch_size = 25
num_samples_to_train = 20000
num_minibatches_to_train = int(num_samples_to_train  / minibatch_size)

# Run the trainer and perform model training
training_progress_output_freq = 50

for i in range(0, num_minibatches_to_train):
    features, labels = generate_random_data_sample(minibatch_size, input_dim, num_output_classes)

    # Assign the minibatch data to the input variables and train the model on the minibatch
    trainer.train_minibatch({feature : features, label : labels})
    batchsize, loss, error = print_training_progress(trainer, i,
                                                     training_progress_output_freq, verbose=1)