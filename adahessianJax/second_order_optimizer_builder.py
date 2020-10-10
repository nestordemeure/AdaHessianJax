"""
    Implements a decorator to build a second order optimizer, taking both gradient and hessian information.
    This is derived from the decorator implemented in https://github.com/google/jax/blob/master/jax/experimental/optimizers.py
"""

from typing import Callable, NamedTuple, Tuple, Any
import functools
import numpy.random as npr
from jax import grad
from jax.util import partial, unzip2, safe_zip, safe_map
from jax.tree_util import tree_flatten, tree_unflatten # , tree_map, register_pytree_node
from jax.experimental.optimizers import Step, Updates, OptimizerState, InitFn, ParamsFn, Params, State, Optimizer

from adahessianJax.hessian_computation import hutchinson_grad_and_hessian

RngKey = Any
SecondOrderUpdateFn = Callable[[Step, Updates, Updates, OptimizerState, RngKey], OptimizerState]

class SecondOrderOptimizer(NamedTuple):
    init_fn: InitFn
    update_fn: SecondOrderUpdateFn
    params_fn: ParamsFn

def second_order_optimizer(opt_maker: Callable[...,
    Tuple[Callable[[Params], State],
          Callable[[Step, Updates, Updates, Params], Params],
          Callable[[State], Params]]]) -> Callable[..., Optimizer]:
    """Decorator to make an optimizer defined for arrays generalize to containers.
    With this decorator, you can write init, update, and get_params functions that
    each operate only on single arrays, and convert them to corresponding
    functions that operate on pytrees of parameters.
    See the optimizer defined in adahessian.py for an example.
    """
    @functools.wraps(opt_maker)
    def tree_opt_maker(*args, **kwargs):
        init, update, get_params = opt_maker(*args, **kwargs)

        @functools.wraps(init)
        def tree_init(x0_tree):
            x0_flat, tree = tree_flatten(x0_tree)
            initial_states = [init(x0) for x0 in x0_flat]
            states_flat, subtrees = unzip2(safe_map(tree_flatten, initial_states))
            return OptimizerState(states_flat, tree, subtrees)

        @functools.wraps(update)
        def tree_update(i, loss, loss_input, opt_state, rng, argnums=0):
            # computes gradient and hessian
            grad_tree, hessian_tree = hutchinson_grad_and_hessian(loss, loss_input, rng, argnums=argnums)
            # flattens trees
            states_flat, tree_opt_state, subtrees = opt_state
            grad_flat, _ = tree_flatten(grad_tree)
            hessian_flat, _ = tree_flatten(hessian_tree)
            # forward information to optimizer
            states = safe_map(tree_unflatten, subtrees, states_flat)
            new_states = safe_map(partial(update, i), grad_flat, hessian_flat, states)
            new_states_flat, subtrees2 = unzip2(safe_map(tree_flatten, new_states))
            for subtree, subtree2 in safe_zip(subtrees, subtrees2):
                if subtree2 != subtree:
                    msg = ("optimizer update function produced an output structure that "
                           "did not match its input structure: input {} and output {}.")
                    raise TypeError(msg.format(subtree, subtree2))
            return OptimizerState(new_states_flat, tree_opt_state, subtrees)

        @functools.wraps(get_params)
        def tree_get_params(opt_state):
            states_flat, tree, subtrees = opt_state
            states = safe_map(tree_unflatten, subtrees, states_flat)
            params = safe_map(get_params, states)
            return tree_unflatten(tree, params)

        return SecondOrderOptimizer(tree_init, tree_update, tree_get_params)

    return tree_opt_maker
