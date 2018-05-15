# AI for Self Driving Car

# Importing the libraries

# Numpy-> Allows us to play with arrays
import numpy as np
# Random -> For when importing random batch samples when implementing exp replay
import random
# os-> For when we load the model, once model is ready we will 
# implement code to save/load the brain. 
import os
# torch -> our pytorch neural network. Using torch because it can handle dynamic graphs
import torch
# nn module -> Contains all the tools to implement neural networks
# there will be a deep neural network that will takes as inputs the 3 signals of the 3 sensors
# plus orientation and minus orientation and will return as output the action to play or the
# q values of different actions and use a softmax to determine one action
import torch.nn as nn
# Shortcut in nn module-> Contains all of the diff functions used when implementing a neural network
# Typically the loss function, we will be using uberloss
import torch.nn.functional as F
# optimizer -> We will be importing optimizers for stochastic gradient descent
import torch.optim as optim
# Autograd -> to take the variable class from autograd. Have to import the variable class to make
# a conversion from tensors which are like complex arrays to a variable the contains a gradient. You dont
# only want to have a tensor by itself, we want to put the tensor into a variable that will also
# contain a gradient. (Converts tensor into a variable containing and the gradient.)
import torch.autograd as autograd
from torch.autograd import Variable

# architecture of the neural network
# since we want our neural network to be an object, we will make a class. 
# this class will have 2 functions. 
# init function -> defines the variable or your object that is the neural network. In this
# function we define the architecture of the nn. We will 
# define the input layer which will be composed of 5 input neurons because we have 
# 5 dimensions for the encoded vector of input state. 
# then we will define some hidden layers. 
# then we will end up with the output layer that will contain the possible actions that 
# we will play at each time. 
#----
# then we will make another function which will be the forward function.
# this function activates the nuerons in the neural network that will activate the signals
# We will use a rectifier activation function because we're dealing with a none linear problem
# Forward function mostly returns the q values which are the outputs of the neural network.
# We have 1 q value for each action, later on we be taking 1 action by either taking the max
# of the q values or using a soft max method

class Network(nn.Module):
# in the Network class we use inhertence. Our network class is just a child class of the larger
# class nn.Module

    def __init__(self, input_size, nb_action):
    # 3 arguments: self-> refers to the objects that will created from this class. Self specifies that
    # that you are refering to the object. When you want to use a variable for the object. 
    # input_size -> specifies the number of input neurons which is 5 becuse our input vectors have 
    # 5 dimensions with 3 signals plus orientation plus -orientation. That's vectors of encdoed values
    # that describe 1 state of the environment. These 5 values are enough to describe the state of the environment
    # orientation and -orientation keep track of the goal we're trying to reach.
    # nb_action -> output neurons wich correspond to the actions having 3 possibilites: left,
    # right, forward
        super(Network, self).__init__()
        # super() inherits from the nn.Module. Super allows us to use the tools of module.
        # this is a trick to use all the tools of the nn.Module
        self.input_size = input_size
        #now specify the input layer, self.input size contains the number of input neurons
        self.nb_action = nb_action
        #number of output neurons/actions (will be = to 3)
        self.fc1 = nn.Linear(input_size, 30)
        #the full connections between the different layers of our neural network
        #Since our nn only has 1 hidden layer, we will need 2 full connections.
        # 1 for the hidden layer and input layer and 1 between the hidden layer and the output layer
        #full connections -> means all the neurons of the input layer will all be connected to the neurons of the hidden layer.
        # to make this full connection we use the Linear function
        # Linear() -> 3 arguments: in_features -> the num of neurons in the first layer to connect
        # out_features-> num of nuerons in the 2nd layer to connect (hidden layer)
        # bias=true
        self.fc2 = nn.Linear(30, nb_action)
        #in fc2 first argument is num of neurons we want in the first/hidden layer
        # 2nd argu is the num or neurons in the 2nd layer of full connection that is the output layer
        
    def forward(self, state):
    #Forward() -> activates the neurons and returns the q values for each possible action, 
    # function that performs forward propagation.
    # takes 2 argus: self to be able to use the vars of the object
    # 2nd is the input which we will call state because state is exactly the input of our neural networks.
    # then as output we will have the q values of the 3 possible actions: left, foward, right
    # output isn't an argument because that is exactly what this func returns
        x = F.relu(self.fc1(state))
        # first we need to activate the hidden neurons rep by x.
        # to activate them we take our input neurons, use our 1st full connection fc1 to get the hidden neurons
        # then we're going to apply an activation function on them which will be the rectifier function
        # TO do this we will use the torch relu function
        # relu -> is the rectifier function. this activates the hidden neurons.
        # in the relu function we specify the neurons we want to activate which is the hidden neurons fc1
        # inside fc1 we input our input states to go from the input neurons to the hidden neurons
        q_values = self.fc2(x) #output neurons of our nn
        #q_values-> correspondes to the output neurons, the output neurons are the q values.Inside fc2 we input
        # the neurons of the left side of this full connection that is x 
        return q_values
    
    #implement experience replay
    # AI is based on mark decs process. MDP consists of looking at a series of events, like going from 1 state 
    # to the next. Exp replay works by not only considering the current state but a series of events in the past.
    # We will create a memory of the last 100 events, then we will take random batches of these transitions
    # to make our next updat/move. To implement we will use 3 functions. 
    class ReplayMemory(object):
        
        def __init__(self, capacity):
        #arguments: capacity-> will be the number 100 bec exp replay will be the last 100 transition
            self.capacity = capacity
            self.memory = []
            
        def push(self, event):
            #this functions does 2 things:
            # 1 adds a new transition or event in the memory
            # 2 it will make sure that the memory always has 100,000 transitions
            # event is 4 elements: 1 the last state or st 2 New state or st + 1 3 Last action or at 
            # 4 is the last reward or rt. event will be added to the memory.
            #append the new event to the memory
            self.memory.append(event)
            #make sure the memory contains capacity elements less than 100,000
            # if we go over 100,000 delete the oldest
            if len(self.memory) > self.capacity:
                del self.memory[0]
        # get random samples from memory
        def sample(self, batch_size):
            samples = zip(*random.sample(self.memory, batch_size))
            #zip * reshapes our list, we do this because in our memory we have the state, action, and reward.
            # but for our algorhithm we dont want this format we want our format composed of 3 samples
            # 1 for the state, 1 for the action, and 1 for the reward. 
            # we need the 1 batch for each to put each seperately into a pytorch variable.
            # now put the samples into a pytorch variable
            return map(lambda x: Variable(torch.cat(x, 0)), samples)
            #to create a pytorch variable for each sample we use the map function these variables will contain 
            # a tensor and a gradient.
            # lambda x: is a function declaration same as function lambda() { x is the var of the func}. Lambda is just a random name
            # lambda converts our samples into a torch variable.
            # to do this we use the torch Variable function. put x inside because x will be the samples
            #now we have to concate the batches for everything to be well aligned so that each row 
            # with state, action reward corresponds to the same time T.
            # in cat the 0 is the dimension that we want to make that concatinaiton in.
            # no we specefiy param 2 of map to apply this lambda function to all our samples
            
#implement deep Q learning
#Dqn stands for deep q network
class Dqn():
    #arguments, our object self, then since we're creating an object of the netork class, the network class takes
    # the arguments in the init function of input_size and nb_action
    # the last argument wich is the gamma param, in the deep q modal, it is delay coeffcient, that's a parameter of
    # the equation
    def __init__(self, input_size, nb_action, gamma):
        # will need an object of our network, then we will need our memory, then we need variables for the last state
        # last action and the last reward. We will also need an optimizer to perform stochastic gradient descent
        # to update the weights according to how much they will contribute to the error when the ai is making a mistake
