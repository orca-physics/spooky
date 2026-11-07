Examples and use cases. Each folder has a README with how to run it, what it writes, how to read
the output, and the results it should give.

| folder | what | status (2026-10-01) |
|---|---|---|
| `test_derivative` | spectral derivatives in 1D and 2D against the analytic ones (plots) | runs |
| `kse` | forward evolution of the 1D Kuramoto–Sivashinsky equation | runs |
| `kolmogorov` | forward evolution of 2D Kolmogorov flow | runs |
| `swhd_1d` | forward evolution of 1D shallow water equations over topography | new (from main; tests in `tests/test_swhd.py`) |
| `ad_solver` | JAX only: gradients through the solver, JVP vs finite-difference tangents, Floquet/Lyapunov with exact tangents, gradient-descent orbit seeding | runs |
| `benchmarks` | cost per time step and memory of the AD paths (Kolmogorov) | runs |
| `newton_kolmog` | Newton–Krylov–hookstep for a Kolmogorov orbit (orb05) | **out of date** (old API) |
| `specter_wrapper`, `ghost_wrapper` | time marching and Lyapunov runs through the Fortran codes | `run_evol` as before; `run_lyap` **out of date** |

**Environment.**
- `module load python/3.11.3` and `export PYTHONPATH=/share/data4/jcullen/spooky_AD`; the
  editable install points at the main checkout.
- `NUMPY_BACKEND=jax` for the AD paths.
- matplotlib is broken under `python/3.11.3`. Plots need `/share/data4/jcullen/envs/spooky_jax_gpu`
  (JAX + matplotlib) or `specter_env` (matplotlib only).
