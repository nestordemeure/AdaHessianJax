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

# builds an optimizer triplet, no need to pass a learning rate
opt_init, opt_update, get_params = adahessian()

# generates initial state using network parameters AND a Jax random generator key
rng = numpy.random.RandomState(0)
opt_state = opt_init(init_params, rng)

# uses the optimizer, note that we pass the loss and its input instead of the gradient to let adahessian do the computation itself
params = get_params(opt_state)
opt_state = opt_update(i, loss, (params, batch), opt_state)
```

The [example folder](https://github.com/nestordemeure/AdaHessianJax/tree/main/examples) contains JAX's MNIST classification example updated to be run with Adam or AdaHessian in order to compare both implementations.

## Documentation

#### `adahessian`

| **Argument** | **Description** |
| :-------------- | :-------------- |
| `step_size` (float, optional) | learning rate *(default: 1e-3)* |
| `b1`(float, optional) | the exponential decay rate for the first moment estimates *(default: 0.9)* |
| `b2`(float, optional) | the exponential decay rate for the squared hessian estimates *(default: 0.999)* |
| `eps` (float, optional) | term added to the denominator to improve numerical stability *(default: 1e-8)* |
| `weight_decay` (float, optional) | weight decay (L2 penalty) *(default: 0.0)* |
| `hessian_power` (float, optional) | hessian power *(default: 1.0)* |

Returns a `(init_fun, update_fun, get_params)` triple of functions modeling the optimizer, similarly to the [jax.experimental.optimizers API](https://jax.readthedocs.io/en/latest/jax.experimental.optimizers.html).

#### `opt_init`

| **Argument** | **Description** |
| :-------------- | :-------------- |
| `params` (pytree) | pytree representing the initial parameters |
| `rng`(ndarray) | a PRNGKey used as the random key |

Returns a pytree representing the initial optimizer state, which includes the initial parameters and auxiliary values like initial momentum and pseudo random number generator state.

#### `opt_update`

| **Argument** | **Description** |
| :-------------- | :-------------- |
| `step` (int) | integer representing the step index |
| `fun`(Callable) | function to be differentiated |
| `fun_input`(Tuple) | value at which the gradient of `fun` should be evaluated |
| `opt_state` (pytree) | a pytree representing the optimizer state to be updated |
| `argnums`(int, optional) | specifies which positional argument(s) to differentiate with respect to *(default: 0)* |

Returns a pytree with the same structure as the `opt_state` argument representing the updated optimizer state.

#### `get_params`

| **Argument** | **Description** |
| :-------------- | :-------------- |
| `opt_state` (pytree) | pytree representing an optimizer state |

Returns a pytree representing the parameters extracted from `opt_state`.
