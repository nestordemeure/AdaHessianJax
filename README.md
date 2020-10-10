# AdaHessian on Jax

[Jax](https://github.com/google/jax) implementation of the [AdaHessian optimizer](https://github.com/amirgholami/adahessian), a second order optimizer for neural networks.

## Usage

You can install this librarie with:

```
pip install git+https://github.com/nestordemeure/AdaHessianJax.git
```

The implementation tries to stay close to [jax.experimental.optimizers](https://jax.readthedocs.io/en/latest/jax.experimental.optimizers.html) but introduces some modifications due to the need for both randomness and access to the gradient computation:

```python
from adahessianJax import adahessian
...
# builds an optimizer triplet, no need to pass a learning rate
opt_init, opt_update, get_params = adahessian()
...
# generates initial state using network parameters AND a Jax random generator key
rng = numpy.random.RandomState(0)
opt_state = opt_init(init_params, rng)
...
# uses the optimizer, note that we pass the loss and its input instead of the gradient
opt_update(i, loss, (params, batch), opt_state)
...
```

The [example folder](https://github.com/nestordemeure/AdaHessianJax/tree/main/examples) contains JAX's MNIST classification example updated to be run with Adam or AdaHessian in order to compare both implementations.

## TODO

- use [this readme](https://github.com/davda54/ada-hessian) as inspiration to improve documentation
- do a [time](https://jax.readthedocs.io/en/latest/profiling.html) and [memory](https://jax.readthedocs.io/en/latest/device_memory_profiling.html) profiling of the code to compare adahessian et adam
- test adahessian on larger benchmarks
