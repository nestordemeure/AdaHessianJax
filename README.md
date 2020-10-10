# AdaHessian on Jax

[Jax](https://github.com/google/jax) implementation of the [AdaHessian optimizer](https://github.com/amirgholami/adahessian), a second order based optimizer for neural networks.

## Usage

You can install this librarie with:

```
pip install git+https://github.com/nestordemeure/AdaHessianJax.git
```

The implementation tries to stay as compatible as possible with [JAX's optimizers module](https://jax.readthedocs.io/en/latest/jax.experimental.optimizers.html):

```python
# builds an optimizer triple
TODO

# uses the optimizer
TODO
```

We recommand browsing the [example folder](https://github.com/nestordemeure/AdaHessianJax/tree/master/examples) to see the optimizer in action.

## TODO

- finish prototype
- get example working with AdaHessian
- test pip instalation
- use [this readme](https://github.com/davda54/ada-hessian) as inspiration for documentation
- make PR to add this implementation to list on [AdaHessian repo](https://github.com/amirgholami/adahessian)
- do a [time](https://jax.readthedocs.io/en/latest/profiling.html) and [memory](https://jax.readthedocs.io/en/latest/device_memory_profiling.html) profiling of the code to compare adahessian et adam
