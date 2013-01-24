import os
import time
import cPickle as pickle
import numpy as np
from PIL import Image

def tile(X, grid_shape=None, spacing=1, channel_count=1):
    """
    Arrange each row of X as a square tile on a grid.
    """
    tile_length = X.shape[1] / channel_count
    tile_size = np.sqrt(tile_length)

    # Layout all the cases in a square grid if grid_shape is not
    # specified.
    if grid_shape is None:
        sqrt_num_cases = np.sqrt(X.shape[0])
        if sqrt_num_cases.is_integer() and sqrt_num_cases < 25:
            grid_shape = int(sqrt_num_cases)

    assert grid_shape is not None

    grid_shape = np.array(grid_shape)

    if grid_shape.size == 1:
        # Assume square grid is grid_shape wasn't a list/tuple.
        grid_shape = grid_shape.repeat(2)
    
    assert grid_shape.shape == (2,)

    out_shape = (tile_size + spacing) * grid_shape - spacing
    if channel_count > 1:
        out_shape = tuple(out_shape) + (channel_count,)
    out_image = np.ones(out_shape)

    if channel_count == 1:
        for y in xrange(grid_shape[0]):
            for x in xrange(grid_shape[1]):
                t = np.reshape(X[y*grid_shape[1]+x,:], (tile_size, tile_size))
                out_image[
                    y*(tile_size+spacing):(y*(tile_size+spacing)+tile_size),
                    x*(tile_size+spacing):(x*(tile_size+spacing)+tile_size)
                    ] = t
    else:
        for i in range(channel_count):
            out_image[:,:,i] = tile(X[:,i*tile_length:(i+1)*tile_length],
                                    grid_shape, spacing)

    return out_image

def remove_dc(X):
    """
    Remove the DC bias from each row.
    """
    return X - X.mean(1)[:,np.newaxis]

def rescale(X):
    """
    Rescale data to lie in the unit interval.
    """
    return (X / float(np.abs(X).max()) + 1) * 0.5

def sample_images(images, sample_count=100, patch_size=8):
    """
    Randomly samples image patches from images. Returns the samples
    concatenated together into a design matrix where each row is a
    sample.
    """
    w, h, n = images.shape
    X = np.zeros((sample_count, patch_size**2))
    for i in xrange(sample_count):
        j = np.random.randint(n)
        x = np.random.randint(w - patch_size + 1)
        y = np.random.randint(h - patch_size + 1)
        patch = images[x:x+patch_size,y:y+patch_size,j]
        X[i,:] = patch.reshape(patch_size**2)
    return X

def zca_white(X, epsilon=0.0):
    """
    Perform ZCA whitening on the matrix X.
    X is expected to be a <num_cases> by <num_dims> matrix.
    """
    m, n = X.shape

    # If there aren't more examples than dimensions then the data will
    # always lay in a lower dimensional sub-space. (3 points in 3D
    # always lie on a plance.) The smallest eigenvalues will be close
    # to zero and the covariance matrix will equal the identity
    # matrix. Check this isn't the case to avoid confusion.
    assert m > n

    # Removing the mean from each example (remove_dc) will remove one
    # degree of freedom and cause the data to lay in an n-1
    # dimensional sub-space. The smallest eigenvalue will again be
    # near zero resulting (because of numercial issues I think) in the
    # covariance matrix of the whitened data to not quite be the
    # identity matrix.

    # Compute the co-variance matrix. (Assuming each dimension already
    # has zero mean.)
    Cov = X.T.dot(X) / m
    # Compute eigenvectors (U) and eigenvalues (s).
    U, s, _ = np.linalg.svd(Cov)
    # Compute the whitening matrix.
    W = U.dot(np.diag(1. / np.sqrt(s + epsilon))).dot(U.T)

    return X.dot(W), W

def save_images(iterable, output_path):
    """
    Save each image from iterable to disk.
    """
    # I wonder if I should yield the orignal data here to allow
    # chaining. i.e. You could save images to disk while visualizing
    # them interactively.
    i = 0
    for item in iterable:
        image = Image.fromarray(item * 255)
        image = image.convert('RGB')
        fn = os.path.join(output_path, 'sample%i.png' % i)
        image.save(fn)
        i += 1

def make_output_directory(output_path):
    timestamp = str(int(time.time()))
    output_dir = os.path.join(output_path, timestamp)
    os.mkdir(output_dir)
    print 'Output directory is \'%s\'' % output_dir
    return output_dir

def call_func_hook(f):
    def hook(params, *args):
        val = f(params)
        print val
        hook.history.append(val)
    hook.history = []
    return hook
