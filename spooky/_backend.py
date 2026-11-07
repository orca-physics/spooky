import os

# Read from environment variable, default is 'numpy'
BACKEND = os.environ.get('NUMPY_BACKEND', 'numpy').lower()

# --- Backend Import and Aliasing ---
if BACKEND == 'jax':
    try:
        import jax.numpy as xnp
        print("Using JAX backend.")
        JAX_ENABLED = True
    except ImportError:
        import numpy as xnp
        print("JAX requested but not installed. Falling back to NumPy.")
        JAX_ENABLED = False
elif BACKEND == 'numpy':
    import numpy as xnp
    print("Using NumPy backend.")
    JAX_ENABLED = False
else:
    raise ValueError(f"Unknown backend: {BACKEND}. Must be 'numpy' or 'jax'.")

def _jax_index_update(arr, indices, values=0.0):
    """JAX-specific update (immutable, array.at)."""
    # The 'arr' is guaranteed to be a JAX array here
    # A full-shape boolean mask (e.g. dealias_modes) lowers to a scatter over
    # every True entry, which XLA runs ~1000x slower than a select on GPU.
    if (getattr(indices, 'dtype', None) == bool
            and getattr(indices, 'shape', None) == arr.shape
            and xnp.ndim(values) == 0):
        return xnp.where(indices, values, arr)
    return arr.at[indices].set(values)

def _numpy_index_update(arr, indices, values=0.0):
    """NumPy-specific update (uses copy for functional behavior)."""
    # The 'arr' is guaranteed to be a NumPy array here
    arr_copy = arr.copy()
    arr_copy[indices] = values
    return arr_copy

# Define backend functions
if JAX_ENABLED:
    index_update = _jax_index_update

    from jax import jit, grad, vmap, jvp, vjp, jacfwd, jacrev
    from jax import lax as jax_lax
    from functools import partial

    lax_scan = jax_lax.scan

    def apply_jit(func):
        return partial(jit, static_argnums=(0,))(func)

    def copy_arr(arr):
        return arr
else:
    index_update = _numpy_index_update

    def jit(func):
        return func
    def grad(func):
        raise NotImplementedError("Grad is only available with JAX backend.")
    def apply_jit(func):
        return func

    def copy_arr(arr):
        return xnp.copy(arr)

    def _ad_stub(name):
        def stub(*args, **kwargs):
            raise NotImplementedError(
                f"{name} requires JAX backend (set NUMPY_BACKEND=jax)."
            )
        stub.__name__ = name
        return stub

    vmap     = _ad_stub("vmap")
    jvp      = _ad_stub("jvp")
    vjp      = _ad_stub("vjp")
    jacfwd   = _ad_stub("jacfwd")
    jacrev   = _ad_stub("jacrev")
    lax_scan = _ad_stub("lax_scan")

# --- 3. Random Numbers (Stateless Pattern) ---
if JAX_ENABLED:
    from jax import random
    def get_key(seed=0):
        return random.PRNGKey(seed)

    def split_key(key, num=2):
        return random.split(key, num)

    def random_uniform(key, shape=None, minval=0.0, maxval=1.0):
        if shape is None: shape = ()
        return random.uniform(key, shape, minval=minval, maxval=maxval)
else:
    def get_key(seed=0):
        xnp.random.seed(seed)
        return None

    def split_key(key, num=2):
        return [None] * num

    def random_uniform(key, shape=None, minval=0.0, maxval=1.0):
        # Ignores key, uses global NumPy state
        return xnp.random.uniform(minval, maxval, size=shape)
