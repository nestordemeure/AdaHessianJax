import jax.numpy as jnp
import jax
from flax import struct
from flax.optim import OptimizerState, Optimizer
from flax.optim.adam import _AdamParamState, Adam

@struct.dataclass
class SecondOrderOptimizer(Optimizer):
    """Optimizer but forwards a hessian"""
    def apply_gradient(self, grads, hessians, **hyper_param_overrides):
        hyper_params = self.optimizer_def.update_hyper_params(**hyper_param_overrides)
        new_target, new_state = self.optimizer_def.apply_gradient(hyper_params, self.target, self.state, grads, hessians)
        return self.replace(target=new_target, state=new_state)

class AdaHessian(Adam):
    """like adam but uses a hessian approximation instead of the square of the gradient"""
    def apply_param_gradient(self, step, hyper_params, param, state, grad, hessian):
        assert hyper_params.learning_rate is not None, 'no learning rate provided.'
        beta1 = hyper_params.beta1
        beta2 = hyper_params.beta2
        weight_decay = hyper_params.weight_decay
        grad_sq = jax.lax.square(hessian)
        grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
        grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq

        # bias correction
        t = step + 1.
        grad_ema_corr = grad_ema / (1 - beta1 ** t)
        grad_sq_ema_corr = grad_sq_ema / (1 - beta2 ** t)

        denom = jnp.sqrt(grad_sq_ema_corr) + hyper_params.eps
        new_param = param - hyper_params.learning_rate * grad_ema_corr / denom
        new_param -= hyper_params.learning_rate * weight_decay * param
        new_state = _AdamParamState(grad_ema, grad_sq_ema)
        return new_param, new_state

    def apply_gradient(self, hyper_params, params, state, grads, hessians):
        """forwards a hessian"""
        step = state.step
        params_flat, treedef = jax.tree_flatten(params)
        states_flat = treedef.flatten_up_to(state.param_states)
        grads_flat = treedef.flatten_up_to(grads)
        hessians_flat = treedef.flatten_up_to(hessians)
        out = [self.apply_param_gradient(step, hyper_params, param, state, grad, hessian)
               for param, state, grad, hessian in zip(params_flat, states_flat, grads_flat, hessians_flat)]

        new_params_flat, new_states_flat = list(zip(*out)) if out else ((), ())
        new_params = jax.tree_unflatten(treedef, new_params_flat)
        new_param_states = jax.tree_unflatten(treedef, new_states_flat)
        new_state = OptimizerState(step + 1, new_param_states)
        return new_params, new_state

    def create(self, target):
        """produces a second order optimizer instead of an optimizer"""
        opt_def = self
        state = opt_def.init_state(target)
        return SecondOrderOptimizer(opt_def, state, target)
