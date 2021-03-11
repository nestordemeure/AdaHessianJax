import jax.numpy as jnp
from jax.experimental.optimizers import make_schedule

from ..hessian import average_magnitude
from .second_order_optimizer_builder import second_order_optimizer

@second_order_optimizer
def adahessian(step_size=1e-1, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0, hessian_power=1):
    """Construct optimizer triple for AdaHessian.
        Args:
        step_size: positive scalar, or a callable representing a step size schedule that maps the iteration index to positive scalar.
        b1: optional, a positive scalar value for beta_1, the exponential decay rate for the first moment estimates (default 0.9).
        b2: optional, a positive scalar value for beta_2, the exponential decay rate for the second moment estimates (default 0.999).
        eps: optional, a positive scalar value for epsilon, a small constant for numerical stability (default 1e-4).
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

    def update(i, g, h, state):
        x, m, v = state
        h = average_magnitude(h)
        m = (1 - b1) * g + b1 * m  # First moment estimate.
        v = (1 - b2) * jnp.square(h) + b2 * v  # Second moment estimate for the Hessian.
        mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
        vhat = v / (1 - b2 ** (i + 1))
        x = x - step_size(i) * (mhat / (jnp.sqrt(vhat) ** hessian_power + eps) + weight_decay * x)
        return x, m, v

    def get_params(state):
        x, _, _ = state
        return x

    return init, update, get_params
