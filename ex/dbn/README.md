This directory contains my attempt to recreate [Geoff Hinton's MNIST
deep belief net](http://www.cs.toronto.edu/~hinton/digits.html).

Here's a video of my implementation generating samples from each of
the 10 digit classes.

[MNIST deep belief net
demo](https://dl.dropbox.com/u/501760/ml/mnist_deep_belief_net.mpg)
(3.1MB)

Each layer was pre-trained for 100 epochs which took about 7 hours.
The top layer was trained with PCD. I've no idea if this was sensible.
See pretrain.py for full details.

The net was fine-tuned for a total of 70 epochs. The first 50 took
about 10.5 hours and used a learning rate of 0.1, no momentum and
CD-10. The last 20 took about 5.5 hours and used a learning rate of
0.05, a small amount of momentum (fixed at 0.1) and CD-15. See
finetune.py.

I did try further fine-tuning but it didn't improve the subjective
appearance of samples. I didn't try to measure the net's
discriminative performance.

You'll find my email on my GitHub profile if you want to get in touch.
