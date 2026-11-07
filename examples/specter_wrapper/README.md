# SPECTER wrapper: Boussinesq (Rayleigh–Bénard) with the Fortran code

`time_marching_bouss/` drives the SPECTER `BOUSS` binary (256×103, double precision) through
`spooky.solvers.SPECTER` with `Grid2D_semi` (periodic in x only). The wrapper writes
`parameter.inp` and the input fields, runs `BOUSS` with MPI (`nprocs`) and reads `vx`, `vz`, `th`
back. Parameters are in `params.py`: box `2 pi x pi`, `Ra = 5e5`, `Pr = 1`, `dt = 5e-4`,
`Tevolve = 1`, `nprocs = 14`.

SPECTER uses free-fall units in the box `[2 pi, pi]`, the same units as the native solver with
`Lz = pi` (see `../rayleigh_benard/README.md`). To move fields between the two codes use
`spooky.rb_convert`.

- **`run_evol.py`:** evolves noise in `theta` (windowed to vanish at the walls) from rest for
  `Tevolve` and saves `vx.png`, `vz.png`, `th.png` (initial vs final). The commented line shows how
  to write intermediate fields and balances.
- **`run_lyap.py`:** meant to compute 5 Lyapunov exponents. **Broken:** it calls
  `lyap.lyapunov_exponents` but never creates `lyap = DynSys(solver)`, and it has the same
  out-of-date call as the GHOST version (three return values, `tol=`).

**Binaries.** The `BOUSS` binaries used for the RB paper's runs (`/share/data4/jcullen/RB/...`)
follow the pySPEC protocol and are **not** compatible with this wrapper. The binary in this folder
has not been re-tested here.
