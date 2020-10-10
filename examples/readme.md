# Examples

This code is directly extracted from [JAX's example folder](https://github.com/google/jax/tree/master/examples).
It contains the fit of a MNIST classifier with Adam and an Adahessian equivalent.

Currently Adam epochs are computed in 0.77 seconds while AdaHessian epochs are computed in 2.2 seconds.
Adahessian with a gradient computation (used as a proxy for the hessian) is computed in 1.1 seconds.
