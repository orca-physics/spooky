"""
Gradient through the PDE solver via jax.grad
=============================================
Shows the most basic AD capability: computing d(loss)/d(u0) where loss is
any scalar function of the output fields, and u0 is the initial condition.

This is the foundation for:
  - PePINNs (physics loss gradient)
  - Autoencoder training with differentiable physics
  - UPO periodicity-loss minimization (used in example 4)

Run:  NUMPY_BACKEND=jax python 1_gradient_through_solver.py
"""
import os; os.environ.setdefault('NUMPY_BACKEND', 'jax')

import numpy as np
import jax
import jax.numpy as jnp

import sys; sys.path.insert(0, '..')
import spooky as sp
from spooky.solvers import KuramotoSivashinsky, KolmogorovFlow
import params as pm

jax.config.update("jax_enable_x64", True)

# ------------------------------------------------------------------
# 1D KSE — gradient of total energy at t=T w.r.t. initial condition
# ------------------------------------------------------------------
print("=== 1D KSE ===")
grid_kse = sp.Grid1D(pm.kse_Lx, pm.kse_Nx, pm.kse_dt)
solver_kse = KuramotoSivashinsky(grid_kse, nu=pm.kse_nu, rkord=2)

u0 = jnp.array(
    0.3 * np.cos(2 * np.pi * 3 * grid_kse.xx / pm.kse_Lx) +
    0.4 * np.cos(2 * np.pi * 5 * grid_kse.xx / pm.kse_Lx)
)

T = 0.5

def energy_loss_kse(u):
    """Total kinetic energy of the solution at time T."""
    out = solver_kse.ad_evolve([u], T)
    return jnp.sum(out[0] ** 2)

grad_fn = jax.jit(jax.grad(energy_loss_kse))
g = grad_fn(u0)
print(f"  u0 shape:  {u0.shape}")
print(f"  grad shape: {g.shape}")
print(f"  grad norm:  {float(jnp.linalg.norm(g)):.4e}")
print(f"  all finite: {bool(jnp.all(jnp.isfinite(g)))}")

# ------------------------------------------------------------------
# 2D Kolmogorov — gradient w.r.t. first velocity component only
# ------------------------------------------------------------------
print("\n=== 2D Kolmogorov ===")
grid_kf = sp.Grid2D(pm.kf_Lx, pm.kf_Ly, pm.kf_Nx, pm.kf_Ny, pm.kf_dt)
solver_kf = KolmogorovFlow(grid_kf, kf=pm.kf_kf, f0=pm.kf_f0, nu=pm.kf_nu, rkord=2)

rng = np.random.default_rng(0)
uu = rng.standard_normal(grid_kf.shape)
vv = rng.standard_normal(grid_kf.shape)
fu, fv = grid_kf.inc_proj([grid_kf.forward(uu), grid_kf.forward(vv)])
uu, vv = grid_kf.inverse(fu), grid_kf.inverse(fv)
u0_kf = jnp.array(uu)
v0_kf = jnp.array(vv)

T = 0.2

def energy_loss_kf(u):
    """Total energy of (u,v) at time T."""
    out = solver_kf.ad_evolve([u, v0_kf], T)
    return sum(jnp.sum(f ** 2) for f in out)

grad_fn_kf = jax.jit(jax.grad(energy_loss_kf))
g_kf = grad_fn_kf(u0_kf)
print(f"  u0 shape:  {u0_kf.shape}")
print(f"  grad shape: {g_kf.shape}")
print(f"  grad norm:  {float(jnp.linalg.norm(g_kf)):.4e}")
print(f"  all finite: {bool(jnp.all(jnp.isfinite(g_kf)))}")

print("\nDone. The solver is end-to-end differentiable.")
