# AdaHessian optimizer for Jax and Flax

[Jax](https://github.com/google/jax) and [Flax](https://github.com/google/flax) implementations of the [AdaHessian optimizer](https://github.com/amirgholami/adahessian), a second order optimizer for neural networks.

## Installation

You can install this librarie with:

```
pip install git+https://github.com/nestordemeure/AdaHessianJax.git
```

## Using AdaHessian with Jax

The implementation provides both a fast way to evaluate the diagonal of the hessian of a program and an optimizer API that stays close to [jax.experimental.optimizers](https://jax.readthedocs.io/en/latest/jax.experimental.optimizers.html) in the `adahessianJax.jax` namespace:

```python
# gets the jax.experimental.optimizers version of the optimizer
from adahessianJax import grad_and_hessian
from adahessianJax.jaxOptimizer import adahessian

# builds an optimizer triplet, no need to pass a learning rate
opt_init, opt_update, get_params = adahessian()

# initialize the optimizer with the initial value of the parameters to optimize
opt_state = opt_init(init_params)
rng = jax.random.PRNGKey(0)

# uses the optimizer, note that we pass the gradient AND a hessian
params = get_params(opt_state)
rng, rng_step = jax.random.split(rng)
gradient, hessian = grad_and_hessian(loss, (params, batch), rng_step)
opt_state = opt_update(i, gradient, hessian, opt_state)
```

The [example folder](https://github.com/nestordemeure/AdaHessianJax/tree/main/examples) contains JAX's MNIST classification example updated to be run with Adam or AdaHessian in order to compare both implementations.

## Using AdaHessian with Flax

The implementation provides both a fast way to evaluate the diagonal of the hessian of a program and an optimizer API that stays close to [Flax optimizers](https://flax.readthedocs.io/en/latest/flax.optim.html) in the `adahessianJax.flax` namespace:

```python
# gets the flax version of the optimizer
from adahessianJax import grad_and_hessian
from adahessianJax.flaxOptimizer import Adahessian

# defines the optimizer, no need to pass a learning rate
optimizer_def = Adahessian()

# initialize the optimizer with the initial value of the parameters to optimize
optimizer = optimizer_def.create(init_params)
rng = jax.random.PRNGKey(0)

# uses the optimizer, note that we pass the gradient AND a hessian
params = optimizer.target
rng, rng_step = jax.random.split(rng)
gradient, hessian = grad_and_hessian(loss, (params, batch), rng_step)
optimizer = optimizer.apply_gradient(gradient, hessian)
```

## Documentation

#### `jax.adahessian`

| **Argument** | **Description** |
| :-------------- | :-------------- |
| `step_size` (float, optional) | learning rate *(default: 1e-3)* |
| `b1`(float, optional) | the exponential decay rate for the first moment estimates *(default: 0.9)* |
| `b2`(float, optional) | the exponential decay rate for the squared hessian estimates *(default: 0.999)* |
| `eps` (float, optional) | term added to the denominator to improve numerical stability *(default: 1e-8)* |
| `weight_decay` (float, optional) | weight decay (L2 penalty) *(default: 0.0)* |
| `hessian_power` (float, optional) | hessian power *(default: 1.0)* |

Returns a `(init_fun, update_fun, get_params)` triple of functions modeling the optimizer, similarly to the [jax.experimental.optimizers API](https://jax.readthedocs.io/en/latest/jax.experimental.optimizers.html).
The only difference is that `update_fun` takes both a gradient *and* a hessian parameter.

#### `flax.Adahessian`

| **Argument** | **Description** |
| :-------------- | :-------------- |
| `learning_rate` (float, optional) | learning rate *(default: 1e-3)* |
| `beta1`(float, optional) | the exponential decay rate for the first moment estimates *(default: 0.9)* |
| `beta2`(float, optional) | the exponential decay rate for the squared hessian estimates *(default: 0.999)* |
| `eps` (float, optional) | term added to the denominator to improve numerical stability *(default: 1e-8)* |
| `weight_decay` (float, optional) | weight decay (L2 penalty) *(default: 0.0)* |
| `hessian_power` (float, optional) | hessian power *(default: 1.0)* |

Returns an optimizer definition, similarly to the [Flax optimizers API](https://flax.readthedocs.io/en/latest/flax.optim.html).
The only difference is that `apply_gradient` takes both a gradient *and* a hessian parameter.

#### `grad_and_hessian`

| **Argument** | **Description** |
| :-------------- | :-------------- |
| `fun`(Callable) | function to be differentiated |
| `fun_input`(Tuple) | value at which the gradient and hessian of `fun` should be evaluated |
| `rng`(ndarray) | a PRNGKey used as the random key |
| `argnum`(int, optional) | specifies which positional argument to differentiate with respect to *(default: 0)* |

Returns a pair `(gradient, hessian)` where the first element is the gradient of `fun` evaluated in `fun_input` and the second element is the diagonal of its hessian.

This function expects `fun_input` to be a tuple and will fail otherwise.
One can pass `(fun_input,)` if `fun` has a single input that is not already a tuple.

#### `value_grad_and_hessian`

| **Argument** | **Description** |
| :-------------- | :-------------- |
| `fun`(Callable) | function to be differentiated |
| `fun_input`(Tuple) | value at which the gradient and hessian of `fun` should be evaluated |
| `rng`(ndarray) | a PRNGKey used as the random key |
| `argnum`(int, optional) | specifies which positional argument to differentiate with respect to *(default: 0)* |

Returns a triplet `(value, gradient, hessian)` where the first element is the value of `fun` evaluated in `fun_input`, the second element is its gradient and the third element is the diagonal of its hessian.

This function expects `fun_input` to be a tuple and will fail otherwise.
One can pass `(fun_input,)` if `fun` has a single input that is not already a tuple.
