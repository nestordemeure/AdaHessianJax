"""
    Implements a decorator to build a second order optimizer, taking both gradient and hessian information.
    This is derived from the decorator implemented in https://github.com/google/jax/blob/master/jax/experimental/optimizers.py
"""

from typing import Callable, NamedTuple, Tuple, Any
from collections import namedtuple
import functools
import numpy.random as npr
from jax import grad, random
from jax.util import partial, unzip2, safe_zip, safe_map
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node
from jax.experimental.optimizers import Step, Updates, OptimizerState, InitFn, ParamsFn, Params, State, Optimizer

from adahessianJax.hessian_computation import hutchinson_grad_and_hessian

InitFn = Callable[[Params], OptimizerState]

Rng = Any
SecondOrderInitFn = Callable[[Params, Rng], OptimizerState]
SecondOrderUpdateFn = Callable[[Step, Updates, Updates, OptimizerState], OptimizerState]

SecondOrderOptimizerState = namedtuple("OptimizerState", ["packed_state", "tree_def", "subtree_defs", "rng"])
register_pytree_node( # see https://jax.readthedocs.io/en/latest/pytrees.html
    SecondOrderOptimizerState,
    lambda xs: ((xs.packed_state,xs.rng), (xs.tree_def, xs.subtree_defs)), # flatten
    lambda data, xs: SecondOrderOptimizerState(xs[0], data[0], data[1], xs[1])) # unflatten

class SecondOrderOptimizer(NamedTuple):
    init_fn: SecondOrderInitFn
    update_fn: SecondOrderUpdateFn
    params_fn: ParamsFn

def second_order_optimizer(opt_maker: Callable[...,
    Tuple[Callable[[Params, Rng], State],
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
        def tree_init(x0_tree, rng):
            "takes the network paramaters plus a Jax random generator key"
            x0_flat, tree = tree_flatten(x0_tree)
            initial_states = [init(x0) for x0 in x0_flat]
            states_flat, subtrees = unzip2(safe_map(tree_flatten, initial_states))
            return SecondOrderOptimizerState(states_flat, tree, subtrees, rng)

        @functools.wraps(update)
        def tree_update(i, loss, loss_input, opt_state, argnums=0):
            "takes the step number, the loss function, the input for the loss function, the optimizer state and the argnum information for the gradient"
            states_flat, tree_opt_state, subtrees, rng = opt_state
            rng, rng_hessian = random.split(rng)
            # computes gradient and hessian
            grad_tree, hessian_tree = hutchinson_grad_and_hessian(loss, loss_input, rng_hessian, argnums=argnums)
            # flattens trees
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
            return SecondOrderOptimizerState(new_states_flat, tree_opt_state, subtrees, rng)

        @functools.wraps(get_params)
        def tree_get_params(opt_state):
            states_flat, tree, subtrees, rng = opt_state
            states = safe_map(tree_unflatten, subtrees, states_flat)
            params = safe_map(get_params, states)
            return tree_unflatten(tree, params)

        return SecondOrderOptimizer(tree_init, tree_update, tree_get_params)

    return tree_opt_maker
