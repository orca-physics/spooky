Collection of algorithms and tools based on pseudospectral methods for fluid flows

- solvers: Collection of time marching solvers
    - Kuramoto-Sivashinsky 1D
    - Kolmogorov Flow 2D
    - Wrappers for GHOST and [SPECTER](https://github.com/specter-cfd/SPECTER)
- DynSys: 
    - UPOs: calculates unstable periodic orbits using a Newton shooting method.
    - Floquet/Lyapunov analysis: calculate exponents.
    - With the JAX backend: differentiable `ad_evolve`, exact tangent maps (JVP/VJP),
      gradient-based UPO seeding.

If Jax is installed, the backend can be changed by setting the environment
variable
$ export NUMPY_BACKEND='jax'

By default, JAX uses float32, this can be changed by doing
$ export JAX_ENABLE_X64=1
This is mandatory for DynSys routines. Float64 is very slow on inference-class
GPUs (T4, L4: 1/32-1/64 of the float32 rate) but runs at full rate on A100-class GPUs.
    

### Installation
```
pip install spookyflows
```
