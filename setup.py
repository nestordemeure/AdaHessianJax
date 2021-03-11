# https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56
from distutils.core import setup
import setuptools

setup(
    name = 'adahessianJax',
    version = '1.1',
    license = 'apache-2.0',
    description = 'Jax implementation of the AdaHessian optimizer.',
    author = 'NestorDemeure',
    # author_email = 'your.email@domain.com',
    url = 'https://github.com/nestordemeure/AdaHessianJax',
    # download_url = 'https://github.com/nestordemeure/AdaHessianJax/archive/v?.?.tar.gz',
    keywords = ['deep-learning', 'jax', 'optimizer'],
    install_requires=['jax', 'flax'],
    classifiers=[ # https://pypi.org/classifiers/
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    package_dir={"": "adahessianJax"},
    packages=setuptools.find_packages(where="adahessianJax"),
)
