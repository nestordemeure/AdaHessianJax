import numpy
import jax.numpy as jnp
from jax import jvp, grad, random
from jax.tree_util import tree_map

def tree_rademacher(tree, rng):
    """
    produces a tree of the same shape as the input tree but were the leafs are sampled from a rademacher distribution
    meaning that their value are in {-1.0, 1.0}
    """
    def leaf_radeacher(leaf): return random.rademacher(rng, shape=leaf.shape, dtype=numpy.float32)
    return tree_map(leaf_radeacher, tree)

def tree_weighted_mean(tree, weights):
    """
    applies a weighted mean to all the leafs of the tree
    """
    return tree_map(jnp.mean, tree * weights)

def hessian_vector_product(f, primals, tangents):
    """
    Computes the hessian vector product: d²f(primal) * tangents
    See the [autodiff cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#Hessian-vector-products-using-both-forward--and-reverse-mode) for more information.
    `primals` and `tangent` are expected to be tuples with one element per input of f, if f has only one input, you can pass something of the form (x,) to signify a one element tuple.
    """
    return jvp(grad(f), primals, tangents)[1]

def hutchinson_hessian_trace(f, x, rng):
    """
    Uses Hutchinson's randomized algorithm to estimate the trace of the hessian of f in x where x is a pytree
    divide by number of element to get average diagonal element.
    The key idea of the estimator is that Expectation(v^t * H * v) = trace(H) with v a random vector of mean 0.
    This is then combined with a hessian-vector-product to estimate the trace of the Hessian.
    """
    random_vector = tree_rademacher(x, rng)
    diagonal = hessian_vector_product(f, x, random_vector)
    trace = tree_weighted_mean(random_vector, diagonal)
    return trace
