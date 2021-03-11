import abc
from typing import Any
import jax
from flax import struct, serialization
from flax.optim import OptimizerState, OptimizerDef, Optimizer

class SecondOrderOptimizerDef(OptimizerDef):
    """
    Class used to define an optimizer with an additional `hessian` parameter
    derived from: https://github.com/google/flax/blob/master/flax/optim/base.py
    """

    @abc.abstractmethod
    def apply_param_gradient(self, step, hyper_params, param, state, grad, hessian):
        """Apply a gradient for a single parameter.
        Args:
            step: the current step of the optimizer.
            hyper_params: a named tuple of hyper parameters.
            param: the parameter that should be updated.
            state: a named tuple containing the state for this parameter
            grad: the gradient tensor for the parameter.
            hessian: the hessian tensor for the parameter.
        Returns:
            A tuple containing the new parameter and the new state.
        """
        pass

    def apply_gradient(self, hyper_params, params, state, grads, hessians):
        """Applies a gradient for a set of parameters.
        Args:
            hyper_params: a named tuple of hyper parameters.
            params: the parameters that should be updated.
            state: a named tuple containing the state of the optimizer
            grads: the gradient tensors for the parameters.
            hessians: the hessian tensors for the parameters.
        Returns:
            A tuple containing the new parameters and the new optimizer state.
        """
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
        """Creates a new second order optimizer for the given target.
        Args:
            target: the object to be optimized. This will typically be an instance of `flax.nn.Model`.
            focus: a `flax.traverse_util.Traversal` that selects which subset of the target is optimized.
        Returns:
            An instance of `SecondOrderOptimizer`.
        """
        opt_def = self
        state = opt_def.init_state(target)
        return SecondOrderOptimizer(opt_def, state, target)

class SecondOrderOptimizer(Optimizer):
    """
    Wraps a `SecondOrderOptimizerDef` like the `Optimizer` class but forwards a hessian in the `apply_gradient` method
    derived from: https://github.com/google/flax/blob/748f7a386828b80df5159d8c606a3cfe52c77b0a/flax/optim/base.py#L177
    """

    def apply_gradient(self, grads, hessians, **hyper_param_overrides):
        hyper_params = self.optimizer_def.update_hyper_params(**hyper_param_overrides)
        new_target, new_state = self.optimizer_def.apply_gradient(hyper_params, self.target, self.state, grads, hessians)
        return self.replace(target=new_target, state=new_state)
