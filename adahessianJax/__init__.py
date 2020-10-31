# hessian computation
from adahessianJax.hessian_computation import grad_and_hessian, value_grad_and_hessian

# experimental.optimizers
from adahessianJax.second_order_optimizer_builder import second_order_optimizer, SecondOrderOptimizer
from adahessianJax.adahessian import adahessian

# from pkgutil import iter_modules
# modules = set(x[1] for x in iter_modules())
# 'flax' in modules
