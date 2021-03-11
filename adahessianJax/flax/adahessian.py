import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct
from flax.optim.adam import _AdamParamState

from ..hessian import average_magnitude
from .second_order_optimizer_builder import SecondOrderOptimizerDef

@struct.dataclass
class _AdahessianHyperParams:
    learning_rate: onp.ndarray
    beta1: onp.ndarray
    beta2: onp.ndarray
    eps: onp.ndarray
    weight_decay: onp.ndarray
    hessian_power: onp.array

class Adahessian(SecondOrderOptimizerDef):
    """Adahessian optimizer,
    like Adam but uses a hessian approximation instead of the square of the gradient
    """
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, hessian_power=1):
        """Constructor for the Adahessian optimizer.
        Args:
            learning_rate: the step size used to update the parameters (default: 1e-3).
            beta1: the coefficient used for the moving average of the gradient (default: 0.9).
            beta2: the coefficient used for the moving average of the gradient magnitude (default: 0.999).
            eps: the term added to the gradient magnitude estimate for numerical stability (default: 1e-8).
            weight_decay: AdamW style weight decay rate (relative to learning rate) (default: 0.0).
            hessian_power: hessian power (default: 1).
        """
        hyper_params = _AdahessianHyperParams(learning_rate, beta1, beta2, eps, weight_decay, hessian_power)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        return _AdamParamState(jnp.zeros_like(param), jnp.zeros_like(param))

    def apply_param_gradient(self, step, hyper_params, param, state, grad, hessian):
        """takes an additional hessian parameter"""
        beta1 = hyper_params.beta1
        beta2 = hyper_params.beta2
        weight_decay = hyper_params.weight_decay

        hessian = average_magnitude(hessian)
        hessian_sq = jax.lax.square(hessian)
        grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
        grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * hessian_sq

        # bias correction
        t = step + 1.
        grad_ema_corr = grad_ema / (1 - beta1 ** t)
        grad_sq_ema_corr = grad_sq_ema / (1 - beta2 ** t)

        denom = jnp.sqrt(grad_sq_ema_corr) ** hyper_params.hessian_power + hyper_params.eps
        new_param = param - hyper_params.learning_rate * grad_ema_corr / denom
        new_param -= hyper_params.learning_rate * weight_decay * param
        new_state = _AdamParamState(grad_ema, grad_sq_ema)
        return new_param, new_state
