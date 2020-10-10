import numpy
import jax.numpy as jnp
from jax import jvp, grad, random
from jax.tree_util import tree_map, tree_multimap

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
    def weighted_mean(x, weight): return jnp.mean(weight * x)
    return tree_multimap(weighted_mean, tree, weights)

def hessian_vector_product(f, primals, tangents, argnums=0):
    """
    Computes the gradient and the hessian vector product which is: d²f(primal) * tangents
    See the [autodiff cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#Hessian-vector-products-using-both-forward--and-reverse-mode) for more information.
    `primals` and `tangent` are expected to be tuples with one element per input of f, if f has only one input, you can pass something of the form (x,) to signify a one element tuple.
    """
    (gradient, hessian_vector_prod) = jvp(grad(f, argnums=argnums), primals, tangents)
    return gradient, hessian_vector_prod

def hutchinson_grad_and_hessian(f, x, rng, argnums=0):
    """
    Uses Hutchinson's randomized algorithm to estimate the trace of the hessian of f in x where x is a pytree
    divide by number of element to get average diagonal element.
    The key idea of the estimator is that Expectation(v^t * H * v) = trace(H) with v a random vector of mean 0.
    This is then combined with a hessian-vector-product to estimate the trace of the Hessian.
    """
    random_vector = tree_rademacher(x, rng)
    gradient, hessian_vector_prod = hessian_vector_product(f, x, random_vector, argnums=argnums)
    hessian = tree_weighted_mean(random_vector[argnums], hessian_vector_prod)
    return gradient, hessian
