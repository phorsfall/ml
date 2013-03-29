import os
import re
import cPickle as pickle

import numpy as np

def load(output_path, timestamp=None, epoch=None):
    """
    Load parameters from the last epoch of the most recent training
    run. Specify a particular set of parameters to load with the
    timestamp and epoch parameters.
    """
    # Get the most recent timestamp.
    if timestamp is None:
        m = re.compile('^\d+$')
        timestamps = [int(d) for d in os.listdir(output_path)
                      if m.search(d)]
        timestamp = max(timestamps)

    path = os.path.join(output_path, str(timestamp))

    # Get the last epoch.
    if epoch is None:
        m = re.compile('^\d+\.pickle$')
        epochs = [int(p.split('.')[0])
                  for p in os.listdir(path)
                  if m.search(p)]
        epoch = max(epochs)

    filename = os.path.join(path, str(epoch) + '.pickle')
    print 'Loading \'%s\'.' % filename
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Used with optimize.sgd().

# I'm a little nervous about pickling named tuples. If I move/rename
# the namedtuple I think I hit problems unpickling. Is there something
# better I could do?

def save_hook(output_dir):
    def hook(params, epoch, *args):
        filename = '%i.pickle' % epoch
        with(open(os.path.join(output_dir, filename), 'wb')) as f:
            pickle.dump(params, f, -1)
    return hook
