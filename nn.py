import itertools
import numpy
import numpy as np
def v2a(data):
    f  = set(itertools.chain(*(v for v, label in data)))
    f1 = dict((f, i) for i, f in enumerate(f))
    f2 = dict((i, f) for f, i in f1.items())
    y  = [label for v, label in data]
    x  = numpy.zeros((len(data), len(f1)))

    for i, (v, label) in enumerate(data):
        x[i].put(map(f1.__getitem__, v), 1.)
    return x, y, f1, f2

data = [(
    {"meow":1., "purr": 1}, 0.), 
    ({"woof": 1., "bark": 1}, 1.), 
    ({"woof": 1}, 1.),
    ({"purr": 1}, 0.)] * 40

from pattern.db import Datasheet
from pattern.vector import chngrams, shuffled
data = []
for language, tweet in shuffled(Datasheet.load("/Users/tom/Desktop/CLiPS/textgain/data/twitter/language/language.csv")):
    if language in ('ja', 'es'):# 'fr'):
        data.append((chngrams(tweet), language=='es' and 1 or 0))

print len(data)
print data[15]
#print xxx

X, y, f1, f2 = v2a(data)
y = np.array([y]).T

print X.shape
print y.shape


import numpy as np

#X = np.array([ [0,0,1], [0,1,1], [1,0,1], [1,1,1] ])
#y = np.array([ [0,1,1,0] ]).T

def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    #return 1 / (1 + np.exp(-x))
    return 1 / (1 + np.exp(-np.clip(x, -100000, 100000)))

np.random.seed(1)

w0 = 2 * np.random.random(X.shape).T - 1
w1 = 2 * np.random.random(y.shape)   - 1

# Bias
#ones = np.atleast_2d(np.ones(X.shape[0]))
#X = np.concatenate((ones.T, X), axis=1)

r = 1.0 # learning rate

for j in xrange(5):
    # Feed forward
    l0 = X
    l1 = sigmoid(np.dot(l0, w0)) # hidden layer activation
    l2 = sigmoid(np.dot(l1, w1)) # output activation
    # Error
    e2 = y - l2
    d2 = e2 * sigmoid(l2, deriv=True)
    # Backprop
    e1 = d2.dot(w1.T)
    d1 = e1 * sigmoid(l1, deriv=True)

    w1 += r * l1.T.dot(d2) # l0 -> l1
    w0 += r * l0.T.dot(d1) # l1 -> l2
    
    if (j% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(e2)))
    print j
    print "Error:" + str(np.mean(np.abs(e2))) 

def feedfwd(X):
    l0 = X
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))
    print l2


n = 200
e = 0
for i in range(n):
    a = data[i][1]
    #b = model.predict(x[len(x)-201+i])
    #b = nn.predict(np.array(x[len(x)-210+i:len(x)-210+i+2]))[0]
    b = feedfwd(np.array([X[len(x)-201+i]]))
    b = int(round(max(b[0][0],0)))
    print i, a, b
    if a != b:
        e += 1

print e, 1 - float(e) / n


#print l2
print feedfwd(np.array([ [0,1,1] ]))
print xxx

#------------------------------------------------------------------------------------------------------


# https://github.com/dmelcaz/backPropagationNN/blob/master/demo.py

import numpy as np
import time

class NeuralNetwork1(object):

	def __init__(self, inputs, hidden, outputs, activation='tanh', output_act='softmax'):
		
		# Hidden layer activation function
		if activation == 'sigmoid':
			self.activation = sigmoid
			self.activation_prime = sigmoid_prime
		elif activation == 'tanh':
			self.activation = tanh
			self.activation_prime = tanh_prime
		elif activation == 'linear':
			self.activation = linear
			self.activation_prime = linear_prime

		# Output layer activation function
		if output_act == 'sigmoid':
			self.output_act = sigmoid
			self.output_act_prime = sigmoid_prime
		elif output_act == 'tanh':
			self.output_act = tanh
			self.output_act_prime = tanh_prime
		elif output_act == 'linear':
			self.output_act = linear
			self.output_act_prime = linear_prime
		elif output_act == 'softmax':
			self.output_act = softmax
			self.output_act_prime = linear_prime

		# Weights initializarion
		self.wi = np.random.randn(inputs, hidden)/np.sqrt(inputs)
		self.wo = np.random.randn(hidden + 1, outputs)/np.sqrt(hidden)

		# Weights updates initialization
		self.updatei = 0
		self.updateo = 0


	def feedforward(self, X):

		# Hidden layer activation
		ah = self.activation(np.dot(X, self.wi))
			
		# Adding bias to the hidden layer results
		ah = np.concatenate((np.ones(1).T, np.array(ah)))

		# Outputs
		y = self.output_act(np.dot(ah, self.wo))

		# Return the results
		return y


	def fit(self, X, y, epochs=10, learning_rate=0.2, learning_rate_decay = 0 , momentum = 0, verbose = 0):
		
		# Timer start
		startTime = time.time()

		# Epochs loop
		for k in range(epochs):
	
			# Dataset loop
			for i in range(X.shape[0]):
				
				# Hidden layer activation
				ah = self.activation(np.dot(X[i], self.wi))
			
				# Adding bias to the hidden layer
				ah = np.concatenate((np.ones(1).T, np.array(ah))) 

				# Output activation
				ao = self.output_act(np.dot(ah, self.wo))

				# Deltas	
				deltao = np.multiply(self.output_act_prime(ao),y[i] - ao)
				deltai = np.multiply(self.activation_prime(ah),np.dot(self.wo, deltao))

				# Weights update with momentum
				self.updateo = momentum*self.updateo + np.multiply(learning_rate, np.outer(ah,deltao))
				self.updatei = momentum*self.updatei + np.multiply(learning_rate, np.outer(X[i],deltai[1:]))

				# Weights update
				self.wo += self.updateo
				self.wi += self.updatei

			# Print training status
			if verbose == 1:
				print 'EPOCH: {0:4d}/{1:4d}\t\tLearning rate: {2:4f}\t\tElapse time [seconds]: {3:5f}'.format(k,epochs,learning_rate, time.time() - startTime)
				
			# Learning rate update
			learning_rate = learning_rate * (1 - learning_rate_decay)

	def predict(self, X): 

		# Allocate memory for the outputs
		y = np.zeros([X.shape[0],self.wo.shape[1]])

		# Loop the inputs
		for i in range(0,X.shape[0]):
			y[i] = self.feedforward(X[i])

		# Return the results
		return y


# Activation functions
def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
	return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
	return np.tanh(x)

def tanh_prime(x):
	return 1.0 - x**2

def softmax(x):
    return (np.exp(np.array(x)) / np.sum(np.exp(np.array(x))))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def softmax_prime(x):
    return softmax(x)*(1.0-softmax(x))

def linear(x):
	return x

def linear_prime(x):
	return 1

#------------------------------------------------------------------------------------------------------

# http://www.bogotobogo.com/python/python_Neural_Networks_Backpropagation_for_XOR_using_one_hidden_layer.php

import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2


class NeuralNetwork2:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
         
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            # output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0: print 'epochs:', k

    def predict(self, x): 
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)      
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

#------------------------------------------------------------------------------------------------------

#print sigmoid(np.array([0,0,1]))

import itertools
import collections
import numpy
import sys


def frank(vectors=[]):
    """ Returns a (feature, index)-dict ranked by document frequency,
        i.e., the feature that occurs in the most vectors has rank 0.
    """
    r = sum(map(collections.Counter, vectors), collections.Counter())
    r = sorted(r, key=lambda f: (-r[f], f))
    r = map(reversed, enumerate(r))
    r = collections.OrderedDict(r)
    return r

# print frank([{'a': 0}, {'b': 1}, {'b': 1, 'c': 1}, {'c': 1}])

def matrix(vectors=[], features={}):
    """ Returns a 2D numpy.ndarray of the given vectors,
        with the given (feature, index)-dict as columns.
    """
    import numpy # ~5% slower
    f = features or frank(vectors)
    m = numpy.zeros((len(vectors), len(f)))
    for v, a in zip(vectors, m):
        a.put(map(f.__getitem__, v), v.values())
    return m

f = {'meow':0, 'woof': 1, 'chirp': 2, 'blub': 3, 'moo': 4}
#f = {k:v for v, k in enumerate(range(1000))}
#f = list(f)
#print matrix([{'meow':0.5}], f)[0]
#print matrix([{'meow':0.5}, {'woof':0.5}], f)
#print xx



def v2a(data):
    f  = set(itertools.chain(*(v for v, label in data)))
    f1 = dict((f, i) for i, f in enumerate(f))
    f2 = dict((i, f) for f, i in f1.items())
    y  = [label for v, label in data]
    x  = numpy.zeros((len(data), len(f1)))

    for i, (v, label) in enumerate(data):
        x[i].put(map(f1.__getitem__, v), 1.)
    return x, y, f1, f2

data = [(
    {"meow":1., "purr": 1}, 0.), 
    ({"woof": 1., "bark": 1}, 1.), 
    ({"woof": 1}, 1.),
    ({"purr": 1}, 0.)] * 40

from pattern.db import Datasheet
from pattern.vector import chngrams, shuffled
data = []
for language, tweet in shuffled(Datasheet.load("/Users/tom/Desktop/CLiPS/textgain/data/twitter/language/language.csv")):
#for id, type, product, _, score, title, author, review, language in shuffled(Datasheet.load("/Users/tom/Desktop/CLiPS/textgain/data/amazon/reviews-de.csv"))[:4000]:
#for score, tweet, author, date, id in shuffled(Datasheet.load("/Users/tom/Desktop/CLiPS/textgain/data/twitter/sentiment/sentiment-en.csv")):
    if language in ('ja', 'es'):# 'fr'):
        data.append((chngrams(tweet), language=='es' and [1,0] or [0,1]))

print len(data)
print data[15]
#print xxx

x, y, f1, f2 = v2a(data)
#print "X:", x
#print "y:", y
#print "f1:", f1

#nn = NeuralNetwork(len(f1), 2, len(set(y)), 'sigmoid', 'linear')
#nn.fit(x, np.array(y), epochs=50, learning_rate=.1, learning_rate_decay=.01, verbose=1)
#print "ok"
#
##labels = nn.predict(np.array([[1,1,0,0], [0,0,1,1]]))
#labels = nn.predict(x[0:2])
#print labels

#nn = NeuralNetwork2([2,2,1])
#X = np.array([[0, 0],
#                 [0, 1],
#                 [1, 0],
#                 [1, 1]])
#y = np.array([0, 1, 1, 0])
#nn.fit(X, y, epochs=10000, learning_rate=0.1)
#for e in X:
#    print(e,nn.predict(e))


#print numpy.__version__
#from nn.mlnn import *
#model = Model([len(f1), 2, 2])
#model.train(x[:len(x)-300], np.array(y[:len(x)-300]), num_passes=30, epsilon=0.2, reg_lambda=0.01, print_loss=True)

#nn = NeuralNetwork1(len(f1),2,2, 'tanh', 'softmax')
#nn.fit(x[:len(x)-200], np.array(y), epochs=100, learning_rate=0.1, verbose=True)

nn = NeuralNetwork2([len(f1),2,2])
nn.fit(x[:len(x)-200], np.array(y), epochs=100000, learning_rate=0.2)

n = 200
e = 0
for i in range(n):
    a = data[i][1]
    #b = model.predict(x[len(x)-201+i])
    #b = nn.predict(np.array(x[len(x)-210+i:len(x)-210+i+2]))[0]
    b = nn.predict(x[len(x)-201+i])
    b = [int(round(max(b[0],0))), int(round(max(b[1],0)))]
    print i, a, b
    if a != b:
        e += 1

print e, 1 - float(e) / n