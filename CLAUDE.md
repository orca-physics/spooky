# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (editable):**
```bash
pip install -e .
```

**Run tests:**
```bash
pytest tests/                                                             # NumPy backend (AD tests skipped)
NUMPY_BACKEND=jax pytest tests/                                           # JAX backend, incl. AD tests (conftest enables x64)
pytest tests/test_grid.py                                                 # grid only
pytest tests/test_grid.py::TestGrid2D::test_inc_proj_divergence_free     # single test
```

**Run an example:**
```bash
cd examples/kolmogorov && python time_marching.py
```

## Architecture

### Package Layout

```
spooky/
├── __init__.py       # Sets BACKEND='numpy', exports Grid classes
├── _backend.py       # NumPy/JAX abstraction (xnp, index_update, jit, etc.)
├── pseudo.py         # Grid classes: Grid1D, Grid2D, Grid2D_semi, Grid3D
├── solvers/          # PDE solvers (pseudospectral.py, specter.py, ghost.py, ...)
└── methods/          # Dynamical systems analysis (dyn_sys.py, krylov.py, upo_gradient.py)
```

### Backend System (`_backend.py`)

All numerical operations go through aliases imported from `_backend.py`. The backend is selected via the `NUMPY_BACKEND` environment variable (`numpy` or `jax`). Key exports used throughout:

- `xnp` — aliased numpy or jax.numpy
- `index_update(arr, indices, values)` — functional array update (handles JAX immutability)
- `apply_jit` — decorator; no-op for NumPy, applies `jax.jit` for JAX (treats `self` as static: parameters read from `self` inside a jitted step are frozen at first compile)
- `copy_arr`, `get_key`, `split_key`, `random_uniform` — backend-aware utilities
- `jvp`, `vjp`, `vmap`, `lax_scan`, `jacfwd`, `jacrev` — JAX transforms, stubs that raise under NumPy; `JAX_ENABLED` flag

When adding new solvers or grid operations, use `xnp.*` instead of `np.*` and `index_update` instead of direct assignment for array mutation.

### Grid Classes (`pseudo.py`)

All grids use `rfftn`/`irfftn` (real FFT). Inheritance chain: `Grid3D`, `Grid2D`, `Grid2D_semi` all extend `Grid1D`.

Key grid attributes:
- `kx`, `ky`, `kz` — wavenumber arrays (physical units, i.e. `2π/L * mode`)
- `ki`, `kj`, `kl` — integer mode indices (used for dealiasing via `kr`)
- `k2`, `kk` — |k|² and |k|
- `kk2` — k² with zero mode set to 1.0 to avoid division by zero
- `dealias_modes` — boolean mask for 2/3 rule dealiasing (`kr > 1/9`)
- `zero_mode` — index of the k=0 mode (set to 0 after each RK step)
- `norm` — 1/(N²) normalization for spectral averages
- `pxx`, `pyy`, `pxy` (etc.) — incompressibility projector components (Grid2D, Grid3D)

`Grid2D_semi` is a hybrid grid (periodic only in x) used exclusively with the SPECTER wrapper.

### Solver Hierarchy

```
Solver (abstract)
└── PseudoSpectral (abstract, explicit RK, periodic grids)
    ├── KuramotoSivashinsky   (1D, num_fields=1)
    ├── KolmogorovFlow        (2D, num_fields=2)
    └── NSE3D                 (3D, num_fields=3)
SPECTER / GHOST               (external code wrappers, extend Solver directly)
```

**Abstract interface** (`Solver`): `evolve()`, `balance()`, `spectra()`, `outs()`

**`PseudoSpectral.evolve(fields, T, bstep, ostep, sstep, bpath, opath, spath)`**: Steps through `Nt = T/dt` time steps using Adams-Bashforth-style RK. Fields are passed in **physical space** and internally transformed to Fourier space. The `rkstep` method operates entirely in Fourier space.

**`rkstep(fields, prev, oo, dt)`**: Decorated with `@apply_jit`. `oo` counts down from `rkord` to 1 (e.g. for `rkord=2`, `oo` is 2 then 1). The formula is `f_new = f_prev + (dt/oo) * RHS(f_current)`.

Each solver's `rkstep` must:
1. Inverse-transform fields to physical space for nonlinear products
2. Forward-transform products back
3. Apply the incompressibility projector (`grid.inc_proj`) on nonlinear terms
4. Zero the `zero_mode` and apply `dealias_modes` after each step

**`PseudoSpectral.ad_evolve(fields, T, checkpoint_every=None)`**: the same map as `evolve` written as a `lax.scan` with no file I/O, so JAX can differentiate through it (`jax.grad`/`jvp`/`vjp`). It takes the same exact fractional last step as `evolve`. `checkpoint_every` (int or `'auto'` ≈ sqrt(Nt)) wraps chunks of steps in `jax.checkpoint`, so reverse mode stores O(sqrt(Nt)) states instead of the full tape at the cost of one extra forward pass.

### Dynamical Systems Methods (`methods/`)

**`DynSys(solver)`**: Wraps a solver to work with flattened 1D state vectors. Key operations (all act on flat vectors via `@flatten_dec` decorator):
- `flatten(fields)` / `unflatten(U)` — convert between field lists and flat numpy arrays (grids with `remove_boundary = True` carry interior rows only)
- `evolve(U, T)` — evolves flat state
- `translate(U, sx)` — spatial translation in Fourier space
- `floquet_multipliers(fields, T, n, tol)` — Arnoldi algorithm for leading Floquet multipliers
- `lyapunov_exponents(fields, T, n, nsteps)` — Benettin QR algorithm

Floquet, Lyapunov and the Newton solvers all obtain the action of the tangent map through `DynSys.tangent_map(U, dU, T, sx, UT, ep0)` (and the batched `tangent_map_batch`): under JAX it is the exact JVP of `ad_evolve`; under NumPy it falls back to the finite difference `J·δU ≈ (Φ(U + ε·δU) - Φ(U)) / ε` with ε = `ep0 * |U| / |δU|`. `jvp_evolve`/`vjp_evolve` expose forward/reverse mode on field lists.

**`UPOGradientDescent(solver, checkpoint_every='auto')`** (`upo_gradient.py`, JAX only): minimises the periodicity loss `||Φ_T(u) - u||^2/||u||^2` with scipy L-BFGS-B and exact gradients (checkpointed reverse pass) to seed Newton (Page et al. 2024). Prototype: fixed `T`, no shift.

**`UPONewtonSolver(DynSys)`**: Newton-Krylov method for finding unstable periodic orbits (UPOs) and relative periodic orbits (RPOs). Uses GMRES with a hookstep trust-region method (Chandler & Kerswell 2013, Viswanath 2007). State vector `X` may contain `(U, T, sx)` or `(U, T, sx, λ)` for arclength continuation. Writes diagnostic output to configurable directories (`newton_dir`, `gmres_dir`, `hookstep_dir`, `apply_A_dir`, `trust_region_dir`) — set these to `None` to suppress file I/O.

### Typical Usage Pattern

```python
import spooky as sp
from spooky.solvers import KolmogorovFlow

grid = sp.Grid2D(Lx=2*pi, Ly=2*pi, Nx=64, Ny=64, dt=0.01)
solver = KolmogorovFlow(grid, nu=0.01, kf=4)

# Initial condition in physical space
fields = [uu, vv]
# Evolve for time T, writing balance every bstep steps
fields = solver.evolve(fields, T=100, bstep=10, bpath='output/')
```

For dynamical systems analysis, wrap the solver:
```python
from spooky.methods import DynSys
ds = DynSys(solver)
lyap, D_KY = ds.lyapunov_exponents(fields, T=10, n=5, nsteps=100)
```
