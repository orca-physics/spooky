# Kuramoto–Sivashinsky time marching

`time_marching.py` evolves the 1D KSE `u_t + u u_x + u_xx + nu u_xxxx = 0` (`nu = 1`, periodic,
`Lx = 22`, 128 points, second-order RK) from a sum of three cosines, for `T = 0.2` with `dt = 1e-5`
(20 000 steps; parameters in `params.py`). About 4 s with NumPy.

```bash
module load python/3.11.3; export PYTHONPATH=/share/data4/jcullen/spooky_AD
python time_marching.py
```

| output (working directory) | content |
|---|---|
| `balance.dat` | every `bstep = 100` steps: time, energy `0.5 <u^2>` |
| `uu.NNNN.npy` | the field every `ostep = 1000` steps (`NNNN` = step number, at least 4 digits, so `uu.10000.npy` sorts before `uu.2000.npy`) |
| `energy.png` | energy against time |

**Expected result (2026-10-01).** The energy decays from `0.1250` at `t = 0` to `0.0945` at
`t = 0.2`. That's the transient of this initial condition. For chaotic dynamics at `Lx = 22`, run
much longer (`Tevolve` of order 100 with a larger `dt`, e.g. `1e-3`).

`matplotlib` is broken under `python/3.11.3`; use `/share/data4/jcullen/envs/spooky_jax_gpu` or
`specter_env` for the plot.
