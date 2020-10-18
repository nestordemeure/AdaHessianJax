import numpy
import jax.numpy as jnp
from jax import jvp, grad, random
from jax.tree_util import tree_map, tree_multimap

def tree_weighted_average_magnitude(tree, weights):
    """
    computes mean(abs(weight * leaf)) for all leafs of the tree
    NOTE: see the folowing code for the choice of dimenssions along which the mean is computed
    https://github.com/amirgholami/adahessian/blob/master/instruction/adahessian.py
    """
    # def weighted_abs_mean(x, weight): return jnp.mean(jnp.abs(weight * x))
    def weighted_abs_mean(x, weight):
        result = jnp.abs(weight * x)
        #result = jnp.abs(x / weight) # proper function when not using rademacher distribution
        if x.ndim <= 1: return result # 1D tensor, no averaging
        if x.ndim <= 3: return jnp.mean(result, axis=[-1], keepdims=True)
        if x.ndim == 4: return jnp.mean(result, axis=[-2, -1], keepdims=True)
        return jnp.mean(result) # average over all dimenssions
    return tree_multimap(weighted_abs_mean, tree, weights)

def make_random_tree(tree, rng):
    """
    produces a tree of the same shape as the input tree but were the leafs are sampled from a random distribution with mean zero
    we use a normal distribution with a standard deviation of 0.5 as it worked best in our tests
    NOTE: we use the same rng for all random generations but it should not degrade perf as those are independent tensors
    """
    def make_random_leaf(leaf): return random.rademacher(rng, shape=leaf.shape, dtype=numpy.float32)
    #def make_random_leaf(leaf): return random.normal(rng, shape=leaf.shape, dtype=numpy.float32)
    return tree_map(make_random_leaf, tree)

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
    random_vector = make_random_tree(x, rng)
    gradient, hessian_vector_prod = hessian_vector_product(f, x, random_vector, argnums=argnums)
    hessian = tree_weighted_average_magnitude(hessian_vector_prod, random_vector[argnums])
    return gradient, hessian
