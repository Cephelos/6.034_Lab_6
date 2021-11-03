# MIT 6.034 Lab 6: Neural Nets
# Written by 6.034 Staff

from nn_problems import *
from math import e
INF = float('inf')


#### Part 1: Wiring a Neural Net ###############################################

nn_half = [1]

nn_angle = [2,1]

nn_cross = [2,2,1]

nn_stripe = [3,1]

nn_hexagon = [6,1]

nn_grid = [4,2,1]


#### Part 2: Coding Warmup #####################################################

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    if x < threshold:
        return 0
    else:
        return 1

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return 1/(1+pow(e,(-steepness*(x-midpoint))))

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    return max(0,x)

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return -1.0/2*pow(desired_output-actual_output, 2)


#### Part 3: Forward Propagation ###############################################

def node_value(node, input_values, neuron_outputs):  # PROVIDED BY THE STAFF
    """
    Given
     * a node (as an input or as a neuron),
     * a dictionary mapping input names to their values, and
     * a dictionary mapping neuron names to their outputs
    returns the output value of the node.
    This function does NOT do any computation; it simply looks up
    values in the provided dictionaries.
    """
    if isinstance(node, str):
        # A string node (either an input or a neuron)
        if node in input_values:
            return input_values[node]
        if node in neuron_outputs:
            return neuron_outputs[node]
        raise KeyError("Node '{}' not found in either the input values or neuron outputs dictionary.".format(node))

    if isinstance(node, (int, float)):
        # A constant input, such as -1
        return node

    raise TypeError("Node argument is {}; should be either a string or a number.".format(node))

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""

    neuron_outputs = {}

    for node in net.topological_sort():
        output = 0
        for input in net.get_incoming_neighbors(node):

            if type(input) is str:
                if input in net.inputs:
                    input_val = input_values[input]
                else:
                    input_val = neuron_outputs[input]
            else:
                input_val = input

            wire = net.get_wires(startNode=input, endNode=node)[0]

            output += input_val*wire.get_weight()

        node_output = threshold_fn(output)
        neuron_outputs[node] = node_output

    return node_output, neuron_outputs








#### Part 4: Backward Propagation ##############################################

def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""

    highest = -INF
    final_inputs = None

    for i in (-step_size, 0, step_size):
        for j in (-step_size, 0, step_size):
            for k in (-step_size, 0, step_size):
                result = func(inputs[0]+i, inputs[1]+j, inputs[2]+k)
                if result > highest:
                    highest = result
                    final_inputs = [inputs[0]+i, inputs[1]+j, inputs[2]+k]


    return highest, final_inputs


def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""

    dependencies = set()

    dependencies.add(wire)
    dependencies.add(wire.startNode)
    dependencies.add(wire.endNode)

    for wire in net.get_wires(endNode=wire.startNode):
        dependencies = set.union(dependencies, get_back_prop_dependencies(net, wire))

    return dependencies

def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """

    deltas = {}
    topological_neuron_outputs = {}

    neuron_outputs_list = sorted(neuron_outputs, key=lambda k: net.topological_sort().index(k), reverse=True)

    for node in neuron_outputs_list:
        topological_neuron_outputs[node] = neuron_outputs[node]

    for node in topological_neuron_outputs:
        out = topological_neuron_outputs[node]
        if net.is_output_neuron(node):
            deltas[node] = out * (1-out) * (desired_output-out)

        else:
            weight_sum = 0
            for wire in net.get_wires(startNode=node):
                weight_sum += wire.get_weight() * deltas[wire.endNode]
                deltas[node] = out * (1 - out) * weight_sum

    return deltas

def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""

    deltas = calculate_deltas(net, desired_output, neuron_outputs)

    for node in net.topological_sort():
        for input in net.get_incoming_neighbors(node):

            if type(input) is str:
                if input in net.inputs:
                    input_val = input_values[input]
                else:
                    input_val = neuron_outputs[input]
            else:
                input_val = input

            wire = net.get_wires(startNode=input, endNode=node)[0]

            wire.set_weight(wire.get_weight() + r * deltas[node] * input_val)

    return net

def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""

    num_loops = 0
    while accuracy(desired_output, forward_prop(net, input_values, sigmoid)[0]) <= minimum_accuracy:
        num_loops += 1
        net = update_weights(net, input_values, desired_output, forward_prop(net, input_values, sigmoid)[1], r)

    return net, num_loops


#### Part 5: Training a Neural Net #############################################

ANSWER_1 = 20
ANSWER_2 = 37
ANSWER_3 = 7
ANSWER_4 = 127
ANSWER_5 = 70

ANSWER_6 = 1
ANSWER_7 = "checkerboard"
ANSWER_8 = ['small', 'medium','large']
ANSWER_9 = 'B'

ANSWER_10 = 'D'
ANSWER_11 = ['A','C']
ANSWER_12 = ['A','E']


#### SURVEY ####################################################################

NAME = "Theodore Calabrese"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 10
WHAT_I_FOUND_INTERESTING = "Neural nets are cool"
WHAT_I_FOUND_BORING = "nothing"
SUGGESTIONS = ""
