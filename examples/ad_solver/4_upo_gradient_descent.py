"""
UPO seeding via periodicity-loss gradient descent
==================================================
Implements the Page, Norgaard, Brenner & Kerswell (PNAS 2024) workflow:

  1. Choose a candidate period T.
  2. Minimize L(u0, T) = ||Φ(u0,T) − u0||² / ||u0||²  with L-BFGS-B,
     using exact gradients from jax.grad through the PDE solver.
  3. If L < threshold, hand the result to Newton-Krylov-Hookstep as a seed.

The gradient is computed by reverse-mode AD (jax.value_and_grad) through
ad_evolve, which uses lax.scan so the full trajectory is differentiable.

Cost per L-BFGS-B step: ~5–10× a single forward solve (VJP tape).
Memory: O(Nt) (one stored array per time step).

Run:  NUMPY_BACKEND=jax python 4_upo_gradient_descent.py
"""
import os; os.environ.setdefault('NUMPY_BACKEND', 'jax')

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, '..')
import spooky as sp
from spooky.solvers import KolmogorovFlow
from spooky.methods import UPOGradientDescent
import params as pm

import jax; jax.config.update("jax_enable_x64", True)

grid   = sp.Grid2D(pm.kf_Lx, pm.kf_Ly, pm.kf_Nx, pm.kf_Ny, pm.kf_dt)
solver = KolmogorovFlow(grid, kf=pm.kf_kf, f0=pm.kf_f0, nu=pm.kf_nu, rkord=2)
gd     = UPOGradientDescent(solver)

# Build IC: warm up from noise to get a state near the attractor
rng = np.random.default_rng(1)
uu  = rng.standard_normal(grid.shape)
vv  = rng.standard_normal(grid.shape)
fu, fv = grid.inc_proj([grid.forward(uu), grid.forward(vv)])
uu, vv = grid.inverse(fu), grid.inverse(fv)
fields0 = solver.evolve([uu, vv], T=5.0, write_outputs=False)

# ------------------------------------------------------------------
# Sweep over candidate periods — the period that converges best
# is the most likely UPO candidate
# ------------------------------------------------------------------
T_candidates = [1.0, 2.0, 3.0]
n_iter       = 200

print(f"Sweeping over periods T = {T_candidates}\n", file=sys.stdout)
results = {}
for T in T_candidates:
    fields_opt, loss_f = gd.run(
        [jnp.array(f) for f in fields0],
        T=T,
        n_iter=n_iter,
        tol=1e-10,
        verbose=True,
    )
    results[T] = (fields_opt, loss_f)
    print()

# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------
print("=" * 50, file=sys.stdout)
print(f"{'T':>6}  {'final loss':>12}  {'status':>20}")
print("-" * 50)
for T, (_, loss_f) in sorted(results.items(), key=lambda x: x[1][1]):
    status = "*** converged (good seed)" if loss_f < 1e-4 else "not converged"
    print(f"{T:>6.1f}  {loss_f:>12.4e}  {status:>20}", file=sys.stdout)
print("=" * 50)

best_T = min(results, key=lambda T: results[T][1])
best_fields, best_loss = results[best_T]
print(f"\nBest candidate: T = {best_T},  L = {best_loss:.4e}", file=sys.stdout)
print("Pass best_fields to UPONewtonSolver for convergence.", file=sys.stdout)

# ------------------------------------------------------------------
# Visualize the best candidate IC
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, field, label in zip(axes, best_fields, ['u', 'v']):
    im = ax.imshow(field.T, cmap='RdBu_r', origin='lower')
    ax.set_title(f'{label}  (T={best_T}, L={best_loss:.2e})')
    plt.colorbar(im, ax=ax)
plt.suptitle('UPO seed from gradient descent (Kolmogorov flow)')
plt.tight_layout()
plt.savefig('upo_seed.png', dpi=150)
print("Saved upo_seed.png", file=sys.stdout)
plt.close()
