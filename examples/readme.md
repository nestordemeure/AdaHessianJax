# Examples

This code is adapted from [JAX's example folder](https://github.com/google/jax/tree/master/examples).
It contains the fit of a MNIST classifier with Adam and an Adahessian equivalent.

Currently Adam epochs are computed in 0.77 seconds while AdaHessian epochs are computed in 2.2 seconds which is about a x3 overhead.
