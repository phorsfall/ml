import gzip
import struct
import os.path
import numpy as np
import data

# Download the dataset from http://yann.lecun.com/exdb/mnist/.

TRAIN_INPUTS = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_INPUTS = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

NUM_CLASSES = 10

# Consider memoization here to save me continually commenting out
# calls to this func once the data is in memory.
def load_inputs(path, filename, rescale=True):
    """
    Load MNIST digit images from a gzipped IDX file. Return a
    <num_images> by <num_pixels> matrix.
    """
    print 'Loading MNIST inputs...'
    with gzip.open(os.path.join(path, filename)) as f:
        magic = struct.unpack('>i', f.read(4))
        assert(magic[0] == 2051)
        num_images, num_rows, num_cols = struct.unpack('>3i', f.read(12))
        data = np.fromstring(f.read(), np.dtype('>u1'))
        data = np.reshape(data, (num_images, num_rows * num_cols))
        if rescale:
            data = data / 255.
        return data

def load_labels(path, filename):
    """
    Load MNIST digit labels from a gzipped IDX file.
    """
    print 'Loading MNIST labels...'
    with gzip.open(os.path.join(path, filename)) as f:
        magic = struct.unpack('>i', f.read(4))
        assert(magic[0] == 2049)
        num_labels = struct.unpack('>i', f.read(4))
        data = np.fromstring(f.read(), np.dtype('>u1'))
        return data

# Perhaps it would be better to pass in the desired validation set
# size to make it easy to specify the empty set.
def load_training_dataset(path,
                          inputs_filename=TRAIN_INPUTS,
                          labels_filename=TRAIN_LABELS,
                          rescale=True,
                          training_set_size=50000):
    """
    """
    inputs = load_inputs(path, inputs_filename, rescale)
    labels = load_labels(path, labels_filename)
    targets = data.targets_from_labels(labels, NUM_CLASSES)
    n = training_set_size
    train = data.dataset(inputs[0:n], targets[0:n], labels[0:n])
    valid = data.dataset(inputs[n:], targets[n:], labels[n:])
    return train, valid

def load_test_dataset(path,
                      inputs_filename=TEST_INPUTS,
                      labels_filename=TEST_LABELS,
                      rescale=True):
    """
    """
    inputs = load_inputs(path, inputs_filename, rescale)
    labels = load_labels(path, labels_filename)
    targets = data.targets_from_labels(labels, NUM_CLASSES)
    test = data.dataset(inputs, targets, labels)
    return test
