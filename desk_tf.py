# -*- coding: utf-8 -*-


import os
import sys
import time
import numpy as np
import tensorflow as tf
from scipy import linalg

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')

def conv3d(x, w):
    return tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def make_variable(name, shape):   
    v = tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.1))
    return v

def make_constant(name, shape): 
    v = tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    return v

class shared_net:

    # Create model
    def __init__(self,nkerns,trainX1,trainX2):
        self.x1 = tf.placeholder(tf.float32, [None, 3, 32, 32, 1])
        self.x2 = tf.placeholder(tf.float32, [None, 3, 32, 32, 1])
        self.Y = tf.placeholder(tf.int64, [None, ])
        self.nkerns = nkerns

        with tf.variable_scope("sharedconv") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)
            
 
        self.f1 = tf.reshape(self.o1,[tf.shape(self.o1)[0],tf.shape(self.o1)[1]*tf.shape(self.o1)[2]*tf.shape(self.o1)[3]])
        self.f2 = tf.reshape(self.o2,[tf.shape(self.o2)[0],tf.shape(self.o2)[1]*tf.shape(self.o2)[2]*tf.shape(self.o2)[3]])      
        self.full_in = tf.concat([self.f1, self.f2], 1)
   
  
        self.w_f1 = make_variable('wf1',[self.nkerns[1]*5*5*2,2000])
        self.b_f1 = make_variable('bf1',[2000])
        self.fc = tf.nn.bias_add(tf.matmul(self.full_in, self.w_f1), self.b_f1 )
        self.full_out = tf.nn.tanh(self.fc)    

        self.w_f2 = make_variable('wf2',[2000,2])
        self.b_f2 = make_variable('bf2',[2])
        self.logits = tf.nn.bias_add(tf.matmul(self.full_out, self.w_f2), self.b_f2 )
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits))

    def network(self, x): 

        w_conv1 = make_variable('wconv1', [3, 5, 5, 1, self.nkerns[0]])
        b_conv1 = make_constant('bconv1', self.nkerns[0])
        conv1 = conv3d(x, w_conv1)
        h_conv1a = tf.nn.relu(tf.nn.bias_add(conv1,b_conv1))
        h_conv1 = tf.squeeze(h_conv1a)
        h_pool1 = max_pool_2x2(h_conv1)
        h_fc1_drop = tf.nn.dropout(h_pool1, 0.15)

        w_conv2 = make_variable('wconv2', [5, 5, self.nkerns[0], self.nkerns[1]])
        b_conv2 = make_constant('bconv2', self.nkerns[1])
        conv2 = conv2d(h_fc1_drop, w_conv2)
        h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2,b_conv2))
        h_pool2 = max_pool_2x2(h_conv2)
        h_fc2_drop = tf.nn.dropout(h_pool2, 0.25)
        
        return h_fc2_drop

######################### BUILD ACTUAL MODEL ################################
## Model specifications


learning_rate=0.01
n_epochs=10
#Number of filters in each layer
nkerns=[96, 128]
batch_size=10


#loading the dataset
from random import randint
import load_CIFAR10
from sklearn.utils import shuffle
x_train, y_train, x_test, y_test = load_CIFAR10.cifar10()
trainX = np.concatenate((x_train,x_test), axis=0)
trainY = np.concatenate((y_train,y_test), axis=0)

flength = len(trainY)
fl = np.int64(flength/2)

ntrdiffX1 = np.zeros(shape=[fl,32*32*3])
ntrdiffX2 = np.zeros(shape=[fl,32*32*3])
ntrdiffY = np.zeros(shape=[fl,])

print ("Randomly picking 30,000 pairs of dissimilar data (label = 0)")
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

                
print ("Randomly picking 30,000 pairs of similar data (label = 1)")
#print ntrdiffY
#producing equal data i.e labels are same
fl = np.int64(flength/2)
ntrsameX1 = np.zeros(shape=(fl,32*32*3))
ntrsameX2 = np.zeros(shape=(fl,32*32*3))
ntrsameY = np.zeros(shape=(fl,))
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
print ('Here')

num_tr = 100
x_train_1 = np.concatenate((ntrsameX1[:num_tr],ntrdiffX1[:num_tr]),axis =0)
x_train_2 = np.concatenate((ntrsameX2[:num_tr],ntrdiffX2[:num_tr]),axis =0)
y_train = np.concatenate((ntrsameY[:num_tr],ntrdiffY[:num_tr]),axis =0)

x_test_1 = np.concatenate((ntrsameX1[num_tr:num_tr+100],ntrdiffX1[num_tr:num_tr+100]),axis =0)
x_test_2 = np.concatenate((ntrsameX2[num_tr:num_tr+100],ntrdiffX2[num_tr:num_tr+100]),axis =0)
y_test = np.concatenate((ntrsameY[num_tr:num_tr+100],ntrdiffY[num_tr:num_tr+100]),axis =0)

x_train_1= x_train_1.astype('float32',copy=True)
x_train_2= x_train_2.astype('float32',copy=True)

x_test_1= x_test_1.astype('float32',copy=True)
x_test_1= x_test_1.astype('float32',copy=True)

y_train= y_train.astype('int64',copy=True)
y_test= y_test.astype('int64',copy=True)

x_train_1, x_train_2, y_train = shuffle(x_train_1, x_train_2, y_train)

x_test_1, x_test_2, y_test = shuffle(x_test_1, x_test_2, y_test)

print ('Data loading done!')

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

print ('... building the model')
#
#    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
#    # to a 4D tensor, compatible with our LeNetConvPoolLayer
#    # (28, 28) is the size of MNIST images.


trainX1 = trainX1.reshape(-1, 3, 32, 32, 1) 
trainX2 = trainX2.reshape(-1, 3, 32, 32, 1) 

testX1 = testX1.reshape(-1, 3, 32, 32, 1) 
testX2 = testX2.reshape(-1, 3, 32, 32, 1) 

net = shared_net(nkerns,trainX1,trainX2)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(net.cost)
predict_op = tf.argmax(net.logits, 1)
iteration = 0


    ###############
    # TRAIN MODEL #epoch
    ###############
print ('... training')
test_score = 0
start_time = time.clock()

init_op = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init_op)
   
    for epoch in range(n_epochs):  # no.of iterations

            mini_batch = 0
            training_batch = zip(range(0, len(trainX1), batch_size),
                     range(batch_size, len(trainX1) + 1, batch_size))  

            for start, end in training_batch:
                mini_batch+=1
                iteration+=1
                print ('Processing minibatch # %d of epoch # %d.,iteration %i' % (mini_batch, epoch,iteration))
                sess.run(train_op, feed_dict={net.x1: trainX1[start:end], net.x2: trainX2[start:end], 
                                              net.Y: train_y[start:end]})

                if iteration % 5 == 0:
                    precision = []
                    test_batch = zip(range(0, len(testX1), 5*batch_size),
                         range(5*batch_size, len(testX1) + 1, 5*batch_size))

                    for start, end in test_batch:
                        p = sess.run(predict_op, feed_dict={net.x1: testX1[start:end], net.x2: testX2[start:end]})

                        precision.append(np.mean(p == test_y[start:end]))
                
                    test_score = np.mean(precision)        
                    print(('epoch %i, minibatch %i/%i, test accuracy of '
                               'best model %f') %
                              (epoch, mini_batch , len(list(training_batch)),
                               test_score * 100.0)) 

end_time = time.clock()
print('Optimization complete.')
print('The code for file ' + 'ran for %.2fm' % ((end_time - start_time) / 60.)) 













