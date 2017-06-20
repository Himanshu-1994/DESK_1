# -*- coding: utf-8 -*-


import os
import sys
import time
import numpy as np
import tensorflow as tf
from scipy import linalg
from sklearn.utils import array2d, as_float_array
from sklearn.base import TransformerMixin, BaseEstimator

srng = RandomStreams()

rng =np.random.RandomState(23455)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')

def conv3d(x, w):
    return tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def make_variable(name, shape,scope):   
    v = tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.1))
    return v

def make_constant(name, shape,scope): 
    v = tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    return v

class shared_net:

    # Create model
    def __init__(self,nkerns):
        self.x1 = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.x2 = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.Y = tf.placeholder(tf.int64, [None, ])
        self.nkerns = nkerns

        with tf.variable_scope("shared_conv") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)

        self.f1 = tf.reshape(self.o1,[-1,self.nkerns[1]*5*5])
        self.f2 = tf.reshape(self.o2,[-1,self.nkerns[1]*5*5])        
        self.full_in = tf.concat(self.f1,self.f2,1)
    
        self.w_f1 = make_variable('wf1',[self.nkerns[1]*5*5*2,2000])
        self.b_f1 = make_variable('bf1',[2000])
        self.fc = tf.nn.bias_add(tf.matmul(self.full_in, self.w_f1), self.b_f1 )
        self.full_out = tf.nn.tanh(self.fc)    

        self.w_f2 = make_variable('wf2',[2000,2])
        self.b_f2 = make_variable('bf2',[2])
        self.logits = tf.nn.bias_add(tf.matmul(self.full_out, self.w_f2), self.b_f2 )
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.Y))

    def network(self, x): 

        w_conv1 = make_variable('w_conv1', [5, 5, 3, 1, self.nkerns[0]],scope)
        b_conv1 = make_constant('b_conv1', self.nkerns[0],scope)
        conv1 = conv3d(x, w_conv1)
        h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1,b_conv1))
        h_pool1 = max_pool_2x2(h_conv1)
        h_fc1_drop = tf.nn.dropout(h_pool1, 0.15)

        w_conv2 = make_variable('w_conv1', [5, 5, self.nkerns[0], self.nkerns[1],scope)
        b_conv2 = make_constant('b_conv1', self.nkerns[1],scope)
        conv2 = conv2d(h_fc1_drop, w_conv2)
        h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2,b_conv2))
        h_pool2 = max_pool_2x2(h_conv2)
        h_fc2_drop = tf.nn.dropout(h_pool2, 0.25)
        
        return h_fc2_drop



######################### BUILD ACTUAL MODEL ################################
## Model specifications


learning_rate=0.01
n_epochs=1200
#Number of filters in each layer
nkerns=[96, 128]
batch_size=500
    
    

rng = np.random.RandomState(23455)

#loading the dataset
from random import randint
import load_CIFAR10
from sklearn.utils import shuffle
x_train, y_train, x_test, y_test = load_CIFAR10.cifar10()
trainX = np.concatenate((x_train,x_test), axis=0)
trainY = np.concatenate((y_train,y_test), axis=0)

flength=len(trainY)

ntrdiffX1 = np.zeros(shape=(flength/2,32*32*3))
ntrdiffX2 = np.zeros(shape=(flength/2,32*32*3))
ntrdiffY = np.zeros(shape=(flength/2,))

print "Randomly picking 30,000 pairs of dissimilar data (label = 0)"
#producing diff data i.e labels are different

i=0

#for i in xrange(0,flength/2):
while i < (flength/2):
    q=randint(0,flength-1)
    r=randint(0,flength-1)
        
    d=trainY[q]
    e=trainY[r]
        
    if d==e:
        i= i-1
        continue
    else:
        #print d,e
        ntrdiffY[i]=0
        a=trainX[q]
        b=trainX[r]
        ntrdiffX1[i]=a
        ntrdiffX2[i]=b
    i+=1

                
print "Randomly picking 30,000 pairs of similar data (label = 1)"
#print ntrdiffY
#producing equal data i.e labels are same
ntrsameX1 = np.zeros(shape=(flength/2,32*32*3))
ntrsameX2 = np.zeros(shape=(flength/2,32*32*3))
ntrsameY = np.zeros(shape=(flength/2,))
k=0

#
while k < (flength/2):
    q=randint(0,flength-1)
    r=randint(0,flength-1)
        
    d=trainY[q]
    e=trainY[r]
    if d == e:
        #print d,e
        ntrsameY[k]=1
        a=trainX[q]
        b=trainX[r]
        ntrsameX1[k]=a
        ntrsameX2[k]=b
    else:
        k= k-1
#        continue
    k=k+1
    
x_train_1 = np.concatenate((ntrsameX1[:25000],ntrdiffX1[:25000]),axis =0)
x_train_2 = np.concatenate((ntrsameX2[:25000],ntrdiffX2[:25000]),axis =0)
y_train = np.concatenate((ntrsameY[:25000],ntrdiffY[:25000]),axis =0)

x_test_1 = np.concatenate((ntrsameX1[25000:],ntrdiffX1[25000:]),axis =0)
x_test_2 = np.concatenate((ntrsameX2[25000:],ntrdiffX2[25000:]),axis =0)
y_test = np.concatenate((ntrsameY[25000:],ntrdiffY[25000:]),axis =0)

x_train_1= x_train_1.astype('float32',copy=True)
x_train_2= x_train_2.astype('float32',copy=True)

x_test_1= x_test_1.astype('float32',copy=True)
x_test_1= x_test_1.astype('float32',copy=True)

y_train= y_train.astype('int64',copy=True)
y_test= y_test.astype('int64',copy=True)

x_train_1, x_train_2, y_train = shuffle(x_train_1, x_train_2, y_train)

x_test_1, x_test_2, y_test = shuffle(x_test_1, x_test_2, y_test)

print 'Data loading done!'

# compute number of minibatches for training, validation and testing
n_train_batches = x_train_1.shape[0]
n_test_batches = x_test_1.shape[0]
n_train_batches /= batch_size
n_test_batches /= batch_size

  
trainX1,trainX2,train_y = x_train_1,x_train_2,y_train
testX1,testX2,test_y = x_test_1,x_test_2,y_test

    ######################
    # BUILD ACTUAL MODEL #
    ######################

print '... building the model'
#
#    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
#    # to a 4D tensor, compatible with our LeNetConvPoolLayer
#    # (28, 28) is the size of MNIST images.


trainX1 = trainX1.reshape(-1, 32, 32, 3) 
trainX2 = trainX2.reshape(-1, 32, 32, 3) 

testX1 = testX1.reshape(-1, 32, 32, 3) 
testX2 = testX2.reshape(-1, 32, 32, 3) 

   
    # create a function to compute the mistakes that are made by the model
test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x_1: x_test_1[index * batch_size: (index + 1) * batch_size],
            x_2: x_test_2[index * batch_size: (index + 1) * batch_size],
            y: y_test[index * batch_size: (index + 1) * batch_size]
        }
    )

#validate_model = theano.function(
#        [index],
#        layer3.errors(y),
#        givens={
#            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
#            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
#        }
#    )

    # create a list of all model parameters to be fit by gradient descent
params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x_1: x_train_1[index * batch_size: (index + 1) * batch_size],
            x_2: x_train_2[index * batch_size: (index + 1) * batch_size],
            y: y_train[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #epoch
    ###############
print '... training'
    # early-stopping parameters
#patience = 10000  # look as this many examples regardless
#patience_increase = 2  # wait this much longer when a new best is
#                           # found
#improvement_threshold = 0.995  # a relative improvement of this much is
#                                   # considered significant
#validation_frequency = min(n_train_batches, patience / 2)
#                                  # go through this many
#                                  # minibatche before checking the network
#                                  # on the validation set; in this case we
#                                  # check every epoch
#
#best_validation_loss = numpy.inf
#best_iter = 0
test_score = 0.
start_time = time.clock()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    
    for minibatch_index in xrange(n_train_batches):
        iter = (epoch-1)*n_train_batches + minibatch_index
        if iter % 100 == 0:
            print 'Training @ iter= ', iter 
        #if minibatch_index % 10==0:
        print 'Processing minibatch # %d of epoch # %d.' % (minibatch_index, epoch)
        cost_ij= train_model(minibatch_index)
        
        ## Testing the model
        if (iter + 1) % 100 == 0:
#            validation_losses = [validate_model(i) for i
#                                     in xrange(n_valid_batches)]
#            this_validation_loss = numpy.mean(validation_losses)
#            print('epoch %i, minibatch %i/%i, validation error %f %%' %
#                      (epoch, minibatch_index + 1, n_train_batches,
#                       this_validation_loss * 100.))
#            if this_validation_loss < best_validation_loss:
#                if this_validation_loss < best_validation_loss*improvement_threshold:
#                    patience = max(patience, iter * patience_increase)
#                best_validation_loss = this_validation_loss
#                best_iter = iter
            test_losses = [test_model(i)for i in xrange(n_test_batches)]
            test_score = numpy.mean(test_losses)
            print(('epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
   # learning_rate= learning_rate-0.001
#        if patience <= iter:
#            done_looping = True
#            break
     
end_time = time.clock()
print('Optimization complete.')
#print('Best validation score of %f %% obtained at iteration %i, '
 #         'with test performance %f %%' %
  #        (best_validation_loss * 100., best_iter + 1, test_score * 100.))
print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.)) 













