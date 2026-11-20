from setuptools import setup, find_packages

setup(
        name='spookyflows',
        version='0.3.0',
        packages=find_packages(include=['spooky', 'spooky.*']),
        description='Tools and algorithms for fluid flow simulations',
        url='https://github.com/orca-physics/spooky',
        author='Patricio Clark Di Leoni',
        author_email='pclarkdileoni@udesa.edu.ar',
        license='MIT',
        install_requires=['numpy', 'scipy'],
        extras_require={'jax': ['jax', 'optax']},
)
