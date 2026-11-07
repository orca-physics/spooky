# GHOST wrapper: Kolmogorov flow with the Fortran code

`time_marching_kolmog/` drives the GHOST 2D hydrodynamics binary (`HD`, compiled for 256²,
double precision) through `spooky.solvers.GHOST`. That class writes the parameter file and input
fields, runs `HD` with MPI (`nprocs`) and reads the output back. Parameters are in `params.py`:
box `2 pi x 2 pi`, `nu = 1/40`, `dt = 1e-3`, `nprocs = 10`. `parameter.txt` is the GHOST input
template.

- **`run_evol.py`:** evolves a sum of cosines for `Tevolve = 1` and writes fields to `output/`
  every `ostep` steps and balance files every `bstep` steps (GHOST formats). It saves `uu.png` and
  `vv.png` (initial vs final). Note: it passes the *unprojected* fields to `evolve` (it projects
  them into `fields_i` but uses `fields`).
- **`run_lyap.py`:** 5 Lyapunov exponents over 50 QR steps of `T = 1`, with the finite-difference
  tangent map (GHOST has no AD). **Out of date:** it unpacks three return values and passes `tol=`,
  but `DynSys.lyapunov_exponents` now returns `(exponents, D_KY)` (three values only with
  `return_hist=True`) and takes no `tol`.

**Requirements.** An MPI environment able to run `HD`, from inside the folder. The binary must
match the grid in `params.py`.
