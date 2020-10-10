import jax.numpy as jnp
from jax import jvp, grad
from jax.experimental.optimizers import optimizer

# Hessian vector product
# computes d²f(primal) . tangents
# https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#Hessian-vector-products-using-both-forward--and-reverse-mode
def hessian_vector_product(f, primals, tangents):
    return jvp(grad(f), primals, tangents)[1]

#https://jax.readthedocs.io/en/latest/jax.random.html#jax.random.rademacher

# TODO if we call random twice with the same rng, do we get the same outputs ?
def hutchinson_hessian_trace(f, x, rng):
    random_vector = jnp.random.rademacher(rng, x) # Rademacher distribution {-1.0, 1.0}
    diagonal = hessian_vector_product(f, x, random_vector)
    trace = (random_vector * diagonal) #/ nb_elements # we want one mean per tensor if possible, we can start without mean
    return trace
    
# https://jax.readthedocs.io/en/latest/jax.experimental.optimizers.html
# https://github.com/google/jax/blob/master/jax/experimental/optimizers.py
#
# https://github.com/davda54/ada-hessian/blob/master/ada_hessian.py
# https://github.com/amirgholami/adahessian/blob/master/instruction/adahessian.py
@optimizer
def adahessian(f, rng, step_size=0.15, b1=0.9, b2=0.999, eps=1e-4, weight_decay=0.0, hessian_power=1):
    """Construct optimizer triple for AdaHessian.
        Args:
        f: the function that will be diferentiated.
        rng: a PRNGKey key.
        step_size: positive scalar, or a callable representing a step size schedule
            that maps the iteration index to positive scalar.
        b1: optional, a positive scalar value for beta_1, the exponential decay rate
            for the first moment estimates (default 0.9).
        b2: optional, a positive scalar value for beta_2, the exponential decay rate
            for the second moment estimates (default 0.999).
        eps: optional, a positive scalar value for epsilon, a small constant for
            numerical stability (default 1e-4).
        weight_decay: optional, weight decay (L2 penalty) (default 0).
        hessian_power: optional, Hessian power (default 1)
        Returns:
        An (init_fun, update_fun, get_params) triple.
    """
    step_size = make_schedule(step_size)

    def init(x0):
        m0 = jnp.zeros_like(x0)
        v0 = jnp.zeros_like(x0)
        return x0, m0, v0

    def update(i, g, state):
        x, m, v = state
        hessian_diagonal = hutchinson_hessian_trace(f, x, rng)
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(hessian_diagonal) + b2 * v  # Exponential moving average of Hessian diagonal square values.
        mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
        vhat = v / (1 - b2 ** (i + 1))
        x = x - step_size(i) * (mhat / (jnp.sqrt(vhat) ** hessian_power + eps) + weight_decay * x)
        return x, m, v

    def get_params(state):
        x, _, _ = state
        return x

    return init, update, get_params
