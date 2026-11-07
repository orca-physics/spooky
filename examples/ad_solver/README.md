# AD examples (JAX backend)

Four short demonstrations of differentiating through the solvers. They use KSE (64 points) and
Kolmogorov flow (32×32, `nu = 1/40`, forcing wavenumber 4); the parameters are in `params.py`.
Each runs in seconds to about a minute on CPU.

```bash
module load python/3.11.3
export PYTHONPATH=/share/data4/jcullen/spooky_AD
NUMPY_BACKEND=jax python 1_gradient_through_solver.py
```

Scripts 2 and 4 import matplotlib, which is broken under `python/3.11.3`. Run them from
`/share/data4/jcullen/envs/spooky_jax_gpu`, which has both JAX and matplotlib; add
`JAX_PLATFORMS=cpu` on a node without a GPU.

The results quoted below are from 2026-10-01 (JAX 0.4.30 for 1 and 3, 0.10.2 for 2 and 4, CPU).

## `1_gradient_through_solver.py`: `jax.grad` through the time integration

Computes the gradient of `sum(u(T)^2)` with respect to the initial condition `u(0)` (KSE, `T = 0.5`)
and with respect to the first velocity component (Kolmogorov, `T = 0.2`). It prints shapes,
gradient norms and a finiteness check.

**Expected output:**
- KSE: gradient norm `4.1561e+00`.
- Kolmogorov: gradient norm `1.9061e+01`.
- `all finite: True` for both.

It only demonstrates the machinery. The tests check correctness against finite differences
(`tests/test_ad.py`).

## `2_tangent_map_accuracy.py`: exact JVP vs finite differences

Computes the tangent map `J·dU` at `T = 0.5` (Kolmogorov) in two ways:
- **exact**, with `DynSys.jvp_evolve`;
- **finite difference**, `(Phi(U + eps dU) - Phi(U)) / eps`, for 40 values of `eps` between 1e-12
  and 1.

It plots the relative error of the finite difference against `eps`.

**Expected output:** the best finite difference reaches a relative error of **`2.3e-8`** at
`eps ≈ 3.5e-7`.
- Larger `eps`: truncation error (nonlinearity).
- Smaller `eps`: cancellation (roundoff).

The JVP has neither error and needs no `eps`. This is why Newton, Floquet and Lyapunov use it under
JAX.

**Known issue:** the script prints `Saved tangent_accuracy.png`, but its `savefig` line is
commented out, so no file is written (it calls `plt.show()` instead).

## `3_floquet_lyapunov_ad.py`: Floquet multipliers and Lyapunov exponents with exact tangents

Warms up a random state for 2 time units, then:
- **Floquet:** 6 leading multipliers of the map over `T = 1`. This state is not a periodic orbit,
  so the numbers only illustrate the API.
- **Lyapunov:** 4 exponents by Benettin's QR method, re-orthonormalising every `T = 0.5`, for 20
  steps.

**Expected output:**
- Floquet multipliers `0.325±1.176i` (`|mu| = 1.22`), `0.750±0.472i` (0.886), `0.981`, `1.821`.
  The list is **not sorted** by modulus.
- Lyapunov exponents `[0.278, 0.262, 0.222, 0.189]` and `D_KY = 3.0000`.

**Reading the Lyapunov result.**
- Ten time units are far too short for converged exponents.
- `D_KY = n - 1` is a saturation artefact: with all computed exponents positive, the Kaplan–Yorke
  sum never turns negative. Compute more exponents, until some are negative, for a meaningful
  dimension.

## `4_upo_gradient_descent.py`: seeding periodic orbits by minimising a loss

Minimises `L(u0) = ||Phi_T(u0) - u0||^2 / ||u0||^2` with L-BFGS-B (`UPOGradientDescent.run`, exact
reverse-mode gradients) from a state warmed up for 5 time units. It tries the periods
`T = 1, 2, 3` with 200 iterations each, prints the loss per period, and saves the best candidate's
fields to `upo_seed.png`.

**Expected output (73 s on CPU):**

| T | final loss | label |
|---|---|---|
| 1 | 1.5e-6 | "converged (good seed)" |
| 2 | 3.4e-6 | "converged (good seed)" |
| 3 | 3.6e-4 | "not converged" |

All three stop at the 200-iteration limit.

**Reading the result.** "Converged" only means `L < 1e-4`. A small loss at a short period can also
come from slow, nearly steady dynamics, so a candidate is a periodic orbit only once
`UPONewtonSolver` converges from it. `UPOGradientDescent.run_orbit` optimises `T` and the shift too.
The Kolmogorov seeding tests (K4) are documented in `~/kolmog/orb11_ad/README.md`.
