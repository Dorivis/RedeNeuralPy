import numpy as np
import math

def sigmoid(x):
    return 1/(1+math.exp(-x))
def dsigmoid(x):
    return x*(1-x)

class RedeNeural:
    def matRand(self,i,j):
        return (np.random.rand(i,j)*2)-(np.ones((i,j))*-1)
    def __init__(self, i_nodes, h_nodes, o_nodes):
        self.i_nodes = i_nodes
        self.h_nodes = h_nodes
        self.o_nodes = o_nodes

        self.bias_ih = RedeNeural.matRand(self,self.h_nodes,1)
        self.bias_ho = RedeNeural.matRand(self,self.o_nodes,1)

        self.weights_ih = RedeNeural.matRand(self,self.h_nodes,self.i_nodes)
        self.weights_ho = RedeNeural.matRand(self,self.o_nodes,self.h_nodes)

        self.learning_rate = 0.1

    def train(self, input, expected):

        # Feedforward#######################################
        # Input -> Hidden
        inp = (np.asarray(input)[np.newaxis]).T
        hidden = np.dot(self.weights_ih,inp)
        hidden = hidden + self.bias_ih
        hidden = (np.vectorize(sigmoid))(hidden)

        # Hidden -> Output
        output = np.dot(self.weights_ho,hidden)
        output = output + self.bias_ho
        output = (np.vectorize(sigmoid))(output)
        ####################################################

        # Backpropagation###################################

        expec = (np.asarray(expected)[np.newaxis]).T
        output_error = expec - output

        d_output = (np.vectorize(dsigmoid))(output)

        gradient = output_error*d_output
        gradient = gradient*self.learning_rate
        self.bias_ho = self.bias_ho + gradient

        hidden_T = np.transpose(hidden)

        weights_ho_deltas = np.dot(gradient,hidden_T)
        self.weights_ho = weights_ho_deltas + self.weights_ho

        weights_ho_T = np.transpose(self.weights_ho)
        hidden_error = np.dot(weights_ho_T,output_error)

        d_hidden = (np.vectorize(dsigmoid))(hidden)
        input_T = np.transpose(inp)


        gradient_hidden = hidden_error*d_hidden
        gradient_hidden = gradient_hidden*self.learning_rate

        self.bias_ih = self.bias_ih + gradient_hidden

        weights_ih_deltas = np.dot(gradient_hidden,input_T)
        self.weights_ih = self.weights_ih+weights_ih_deltas
        ####################################################

    def predict(self, input):
        # Feedforward#######################################
        # Input -> Hidden
        inp = (np.asarray(input)[np.newaxis]).T
        hidden = np.dot(self.weights_ih, inp)
        hidden = hidden + self.bias_ih
        hidden = (np.vectorize(sigmoid))(hidden)

        # Hidden -> Output
        output = np.dot(self.weights_ho, hidden)
        output = output + self.bias_ho
        output = (np.vectorize(sigmoid))(output)
        ####################################################
        return output.tolist()[0]
