import numpy
import jax.numpy as jnp
from jax import jvp, grad, value_and_grad, random, dtypes
from jax.tree_util import tree_map, tree_multimap

__all__ = ['grad_and_hessian', 'value_grad_and_hessian']

#--------------------------------------------------------------------------------------------------
# TREE MANIPULATION

def _tree_product(tree1, tree2):
    """returns tree1*tree2"""
    def leaf_product(leaf1, leaf2):
        return leaf1*leaf2
    return tree_multimap(leaf_product, tree1, tree2)

def _tree_average_magnitude(tree):
    """
    computes mean(abs(leaf)) for all leafs of the tree
    NOTE: see the folowing code for the choice of dimenssions along which the mean is computed
    https://github.com/amirgholami/adahessian/blob/master/instruction/adahessian.py
    """
    def abs_mean(x):
        result = jnp.abs(x)
        if x.ndim <= 1: return result # 1D tensor, no averaging
        if x.ndim <= 3: return jnp.mean(result, axis=[-1], keepdims=True)
        if x.ndim == 4: return jnp.mean(result, axis=[-2, -1], keepdims=True)
        return jnp.mean(result) # average over all dimenssions
    return tree_map(abs_mean, tree)

def _make_random_tree(tree, rng):
    """
    produces a tree of the same shape as the input tree but were the leafs are sampled from a rademacher distribution
    NOTE: we use the same rng for all random generations but it should not degrade perf as those are independent tensors
    """
    def make_random_leaf(leaf):
        # waiting for fix on https://github.com/google/jax/issues/4433 as jvp will fail on bool/int types
        if not jnp.issubdtype(leaf.dtype, jnp.floating): return numpy.zeros(shape=leaf.shape, dtype=dtypes.float0)
        return random.rademacher(rng, shape=leaf.shape, dtype=leaf.dtype)
    return tree_map(make_random_leaf, tree)

#--------------------------------------------------------------------------------------------------
# GRADIENT AND HESSIAN

def _gradient_and_hessian_vector_product(f, primals, tangents, argnums=0):
    """
    Computes the gradient and the hessian vector product which is: d²f(primal) * tangents
    See the [autodiff cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#Hessian-vector-products-using-both-forward--and-reverse-mode) for more information.
    `primals` and `tangent` are expected to be tuples with one element per input of f, if f has only one input, you can pass something of the form (x,) to signify a one element tuple.
    """
    (gradient, hessian_vector_prod) = jvp(grad(f, argnums=argnums), primals, tangents)
    return gradient, hessian_vector_prod

def grad_and_hessian(f, x, rng, argnum=0, average_magnitude=True):
    """
    Uses Hutchinson's randomized algorithm to estimate the absolute value of the diagonal of the hessian of f in x where x is expected to be a tuple
    (you can pass something of the form (x,) to signify a one element tuple).
    The key idea of the estimator is that Expectation(v^t * H * v) = trace(H) with v a random vector of mean 0.
    This is then combined with a hessian-vector-product to estimate the trace of the Hessian.

    Returns the gradient and an estimation of the absolute value of the diagonal of the hessian averaged over tensors.
    If `average_magnitude` is set to False, returns a raw estimation of the diagonal of the hessian.
    """
    if not isinstance(x, tuple): raise ValueError(f'Function input must be a tuple but is a {type(x)}, you might want to wrap your input as `(input,)` instead of `x`.')
    random_vector = _make_random_tree(x, rng)
    gradient, hessian_vector_prod = _gradient_and_hessian_vector_product(f, x, random_vector, argnums=argnum)
    # as abs(+-1*x) = abs(x), we do not multiply by random_vector when computing average_magnitude
    hessian = _tree_average_magnitude(hessian_vector_prod) if average_magnitude else _tree_product(random_vector[argnum],hessian_vector_prod)
    return gradient, hessian

#--------------------------------------------------------------------------------------------------
# VALUE, GRADIENT AND HESSIAN

def _value_gradient_and_hessian_vector_product(f, primals, tangents, argnums=0):
    """
    Computes the gradient and the hessian vector product which is: d²f(primal) * tangents
    See the [autodiff cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#Hessian-vector-products-using-both-forward--and-reverse-mode) for more information.
    `primals` and `tangent` are expected to be tuples with one element per input of f, if f has only one input, you can pass something of the form (x,) to signify a one element tuple.
    """
    ((value, gradient), (_,hessian_vector_prod)) = jvp(value_and_grad(f, argnums=argnums), primals, tangents)
    return value, gradient, hessian_vector_prod

def value_grad_and_hessian(f, x, rng, argnum=0, average_magnitude=True):
    """
    Uses Hutchinson's randomized algorithm to estimate the absolute value of the diagonal of the hessian of f in x where x is expected to be a tuple
    (you can pass something of the form (x,) to signify a one element tuple).
    The key idea of the estimator is that Expectation(v^t * H * v) = trace(H) with v a random vector of mean 0.
    This is then combined with a hessian-vector-product to estimate the trace of the Hessian.

    Returns the value, the gradient and an estimation of the absolute value of the diagonal of the hessian averaged over tensors.
    If `average_magnitude` is set to False, returns a raw estimation of the diagonal of the hessian.
    """
    if not isinstance(x, tuple): raise ValueError(f'Function input must be a tuple but is a {type(x)}, you might want to wrap your input as `(input,)` instead of `x`.')
    random_vector = _make_random_tree(x, rng)
    value, gradient, hessian_vector_prod = _value_gradient_and_hessian_vector_product(f, x, random_vector, argnums=argnum)
    # as abs(+-1*x) = abs(x), we do not multiply by random_vector when computing average_magnitude
    hessian = _tree_average_magnitude(hessian_vector_prod) if average_magnitude else _tree_product(random_vector[argnum],hessian_vector_prod)
    return value, gradient, hessian
