"""
JVP tangent map vs finite differences
======================================
Compares DynSys.jvp_evolve (exact, forward-mode AD) against the classical
finite-difference approximation J·dU ≈ (Φ(U+ε·dU) − Φ(U)) / ε.

The JVP result is exact (to float64 precision) and ε-independent.
The FD result depends on ε:  too large → truncation error; too small → cancellation.

The plot shows relative error of FD as a function of ε for a fixed perturbation.

Run:  NUMPY_BACKEND=jax python 2_tangent_map_accuracy.py
"""
import os; os.environ.setdefault('NUMPY_BACKEND', 'jax')

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import spooky as sp
from spooky.solvers import KolmogorovFlow
from spooky.methods import DynSys
import params as pm

import jax; jax.config.update("jax_enable_x64", True)

grid   = sp.Grid2D(pm.kf_Lx, pm.kf_Ly, pm.kf_Nx, pm.kf_Ny, pm.kf_dt)
solver = KolmogorovFlow(grid, kf=pm.kf_kf, f0=pm.kf_f0, nu=pm.kf_nu, rkord=2)
ds     = DynSys(solver)

rng = np.random.default_rng(7)
uu  = rng.standard_normal(grid.shape)
vv  = rng.standard_normal(grid.shape)
fu, fv = grid.inc_proj([grid.forward(uu), grid.forward(vv)])
uu, vv = grid.inverse(fu), grid.inverse(fv)
du  = rng.standard_normal(grid.shape) * 1e-1
dv  = rng.standard_normal(grid.shape) * 1e-1

T = 0.5   # integration time long enough to amplify errors

u0  = jnp.array(uu);   v0  = jnp.array(vv)
du0 = jnp.array(du);   dv0 = jnp.array(dv)

# --- Exact JVP (reference) ---
base, (Jdu_u, Jdu_v) = ds.jvp_evolve([u0, v0], [du0, dv0], T)
Jdu_ref = np.concatenate([np.array(Jdu_u).ravel(), np.array(Jdu_v).ravel()])
norm_ref = np.linalg.norm(Jdu_ref)

# --- FD sweep over ε ---
U0   = np.concatenate([np.array(u0).ravel(), np.array(v0).ravel()])
dU0  = np.concatenate([np.array(du0).ravel(), np.array(dv0).ravel()])
UT   = ds.evolve(U0, T)

epsilons = np.logspace(-12, 0, 40)
rel_errors = []
for eps in epsilons:
    U_pert = U0 + eps * dU0
    UT_pert = ds.evolve(U_pert, T)
    Jdu_fd = (UT_pert - UT) / eps
    rel_errors.append(np.linalg.norm(Jdu_fd - Jdu_ref) / norm_ref)

print("epsilon range:        1e-12 … 1e0")
print(f"min FD rel error:     {min(rel_errors):.2e}  at ε = {epsilons[np.argmin(rel_errors)]:.2e}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(epsilons, rel_errors, 'b.-', label='FD relative error vs JVP')
ax.axhline(1e-14, color='gray', linestyle='--', label='float64 floor')
ax.set_xlabel('ε')
ax.set_ylabel('||J·dU_FD  −  J·dU_JVP|| / ||J·dU_JVP||')
ax.set_title('Tangent map accuracy: FD vs exact JVP (Kolmogorov, T=0.5)')
ax.legend()
plt.tight_layout()
# plt.savefig('tangent_accuracy.png', dpi=150)
print("Saved tangent_accuracy.png")
plt.show()
plt.close()
