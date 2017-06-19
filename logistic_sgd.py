"""
Created on Sun Jul 5 13:04:13 2015

@author: neerajkumar
"""


import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T


class LogisticRegression(object):


    def __init__(self, input, n_in, n_out, parameters):

        # If parameters are not supplied, initialize with 0 the weight matrix W 
        if parameters is None:
            W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                         dtype=theano.config.floatX),
                                        name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        
            b = theano.shared(value=numpy.zeros((n_out,),
                                                         dtype=theano.config.floatX),
                                       name='b', borrow=True)
       
            self.W = W
            self.b = b
        else:
            self.W = parameters[0]
            self.b = parameters[1]

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """ zero one loss over the size of the minibatch

        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
