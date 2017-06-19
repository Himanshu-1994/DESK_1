import numpy as np
import os

def one_hot(x, n):
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def _load_batch_cifar10(filename, dtype='float32'):
  
    path = os.path.join('../data/cifar-10-batches-py', filename)
    batch = unpickle(path)
    data = batch[b'data'] / 255.0 
    labels = batch[b'labels'] # convert labels to one-hot representation
    return data.astype(dtype), labels



def cifar10(dtype='float32', grayscale=True):
    # train
    x_train = []
    t_train = []
    for k in range(5):
        x, t = _load_batch_cifar10("data_batch_%d" % (k + 1), dtype=dtype)
        x_train.append(x)
        t_train.append(t)

    x_train = np.concatenate(x_train, axis=0)
    t_train = np.concatenate(t_train, axis=0)

    # test
    x_test, t_test = _load_batch_cifar10("test_batch", dtype=dtype)

    print(len(t_train),'\n')
    print(len(t_test),'\n')
    
    return x_train, t_train, x_test, t_test



