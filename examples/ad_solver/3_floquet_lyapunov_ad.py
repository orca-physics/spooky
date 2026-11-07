"""
Floquet multipliers and Lyapunov exponents with AD tangent map
==============================================================
Computes both spectral quantities using the JVP-based tangent map instead of
finite differences.  With JAX enabled the same DynSys API is used — the
backend switches automatically via the JAX_ENABLED flag.

Floquet multipliers:  eigenvalues of the monodromy matrix DΦ_T evaluated at a
periodic orbit.  |λ| > 1 → unstable direction.

Lyapunov exponents:   average exponential growth rates of infinitesimal
perturbations along an arbitrary trajectory (Benettin QR algorithm).

Run:  NUMPY_BACKEND=jax python 3_floquet_lyapunov_ad.py
"""
import os; os.environ.setdefault('NUMPY_BACKEND', 'jax')

import numpy as np
import jax.numpy as jnp

import sys; sys.path.insert(0, '..')
import spooky as sp
from spooky.solvers import KolmogorovFlow
from spooky.methods import DynSys
import params as pm

import jax; jax.config.update("jax_enable_x64", True)

grid   = sp.Grid2D(pm.kf_Lx, pm.kf_Ly, pm.kf_Nx, pm.kf_Ny, pm.kf_dt)
solver = KolmogorovFlow(grid, kf=pm.kf_kf, f0=pm.kf_f0, nu=pm.kf_nu, rkord=2)
ds     = DynSys(solver)

# Build an IC and warm it up slightly so the state is on the attractor
rng = np.random.default_rng(42)
uu  = rng.standard_normal(grid.shape)
vv  = rng.standard_normal(grid.shape)
fu, fv = grid.inc_proj([grid.forward(uu), grid.forward(vv)])
uu, vv = grid.inverse(fu), grid.inverse(fv)
fields_warmup = solver.evolve([uu, vv], T=2.0, write_outputs=False)
uu, vv = fields_warmup

# Convert to JAX arrays for the AD path
fields = [jnp.array(uu), jnp.array(vv)]

# ------------------------------------------------------------------
# Floquet multipliers at this base state over period T_fl
# ------------------------------------------------------------------
print("=== Floquet multipliers (JVP tangent map) ===")
T_fl = 1.0   # nominal period; not a true UPO, but illustrates the API
n_fl = 6

evals, evecs, Q = ds.floquet_multipliers(fields, T_fl, n=n_fl, tol=1e-6)

print(f"  Period:  T = {T_fl}")
print(f"  Leading {n_fl} Floquet multipliers (|λ|):")
for i, ev in enumerate(evals):
    print(f"    λ_{i+1} = {ev:.6f}  |λ| = {abs(ev):.6f}"
          + ("  ← unstable" if abs(ev) > 1 else ""))

# ------------------------------------------------------------------
# Lyapunov exponents along a trajectory
# ------------------------------------------------------------------
print("\n=== Lyapunov exponents (JVP tangent map, Benettin QR) ===")
T_ly   = 0.5    # QR re-orthonormalization interval
nsteps = 20     # total time = nsteps * T_ly
n_ly   = 4      # leading exponents

np.random.seed(0)
lyap, D_KY = ds.lyapunov_exponents(fields, T_ly, n=n_ly, nsteps=nsteps)

print(f"  QR interval:  T = {T_ly},  steps = {nsteps}")
print(f"  Lyapunov exponents: {lyap}")
print(f"  Kaplan–Yorke dimension: D_KY = {D_KY:.4f}")
print(f"  (positive exponents → chaotic attractor)")
