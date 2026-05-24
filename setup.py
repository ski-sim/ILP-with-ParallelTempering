"""Setup."""

from setuptools import find_namespace_packages
from setuptools import setup


setup(
    name='ILP-with-ParallelTempering',
    packages=find_namespace_packages(),
    install_requires=[
        'ml_collections',
        'numpy<2',
        'matplotlib',
        'tqdm',
        'tensorflow',
        'networkx',
        'transformers>=4.6.1',
        'tensorflow_probability',
        'absl-py',
        'clu',
        'flax',
        'optax',
        'python-sat',
        'tensorboard',
        'pickle5',
        'nltk',
        'pyscipopt',
        'wandb',
        'jax[cuda12]<0.5',
    ],
)
