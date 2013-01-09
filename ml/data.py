import collections
import numpy as np

dataset = collections.namedtuple('dataset', 'inputs targets labels')

def targets_from_labels(labels, num_classes):
    """
    Create a matrix of targets from the given labels. Input labels are
    integers in {0,1,...,num_classes} representing class labels.
    Targets is a <num_cases> by <num_classes> matrix where each row
    contains a single one in the column corresponding to the class
    label.
    """
    return np.eye(num_classes)[labels]

class BatchIterator:
    """
    """
    # The idea here is that I want the data we pass to sgd to be very
    # generic. By passing an iterator sgd can just loop over batches
    # passing one batch at a time to the optimization objective
    # without needing to know anything about the structure of the
    # data. This should be extendable to the supervised learning case
    # where each batch consists of a tuple of inputs and labels.
    def __init__(self, data, batch_size=100):
        self.data = data
        self.batch_size = batch_size        
        self.is_dataset = isinstance(data, dataset)
        if self.is_dataset:
            num_cases = data.inputs.shape[0]
        else:
            num_cases = data.shape[0]
        assert num_cases % batch_size == 0
        self.num_batches = num_cases / batch_size

    def __iter__(self):
        self.cur_batch = 0
        return self

    def __len__(self):
        return self.num_batches

    def next(self):
        if self.cur_batch >= self.num_batches:
            raise StopIteration
        start = self.cur_batch * self.batch_size
        stop = start + self.batch_size
        s = slice(start, stop)
        if self.is_dataset:
            batch = dataset._make(a[s] for a in self.data)
        else:
            batch = self.data[s]
        self.cur_batch += 1
        return batch
