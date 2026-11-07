# Newton–Krylov–hookstep for a Kolmogorov orbit (orb05) — out of date

> **Status (2026-10-01): these scripts do not run with the current API.** They predate the move to
> keyword-argument constructors: they import `spooky.methods.Parameters` (removed) and call
> `DynSys(pm, solver)`, `KolmogorovFlow(pm_solver)`, `newt.run_newton` on a `DynSys`, and
> `newt.get_restart_values` / `newt.floq_exp` (not in the current `DynSys`). `readme.txt` also
> names `newton_kolmog.py`; the file is `newton_run.py`.
>
> **For a working equivalent use:**
> - `UPONewtonSolver(solver, T=..., sx=..., ...)`, as in `../rayleigh_benard/newton_po1.py`;
> - for Kolmogorov specifically, the K1 test `~/kolmog/orb11_ad/k1/k1_newton.py`;
> - `DynSys.floquet_multipliers` for Floquet.

## What it does (intended)

- **`newton_run.py`:** reads the solver parameters from `params_kolmog.yaml` (box `2 pi`, 256²,
  `nu = 0.025`, `dt = 1e-3`) and the Newton parameters from `params_newton.yaml` (`T = 5.8`,
  `sx = 0`, up to 50 Newton and 50 GMRES iterations, tolerance 1e-10). It starts from
  `input/uu.0000000.npy`, `input/vv.0000000.npy`: a local minimum of a recurrence analysis, the
  same state as `spooky_lic/orbs_hook/orb05`.
- **Outputs (configured in `params_newton.yaml`):**
  - `output/iNkk/` and `balance/iNkk/`: fields and balance along the orbit at each Newton
    iteration;
  - `reports/newton.txt` (`iN, |F|, T, sx, |U|`) and `reports/{gmres,hookstep,trust_region,apply_A}/`,
    the same formats as described in `../rayleigh_benard/README.md` under `newton_po1.py`.
- **`floq_exp.py`:** 50 Floquet multipliers of the converged orbit (Newton iteration `iN` set inside
  the script), saved to `floq/iNkk/floq_exp_50.npy`, `eigvec_H_50.npy` and `Q_50.npy`.
- **`plot_kol.py`:** plots Newton and GMRES errors, orbit fields, balances and Floquet exponents
  into `Imgs/`.
