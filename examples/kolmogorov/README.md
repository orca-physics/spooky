# Kolmogorov flow time marching

`time_marching.py` evolves 2D Kolmogorov flow: forcing `f0 sin(2 pi kf y / Ly)` in x with `kf = 4`
and `f0 = 1` (solver defaults), `nu = 1/40`, box `2 pi x 2 pi`, 256×256, `dt = 1e-4`, `T = 0.2`
(parameters in `params.py`). It starts from a projected (divergence-free) sum of cosines. About
25 s with NumPy.

```bash
module load python/3.11.3; export PYTHONPATH=/share/data4/jcullen/spooky_AD
python time_marching.py
```

| output (working directory) | content |
|---|---|
| `balance.dat` | every `bstep = 100` steps: time, energy `0.5 <|u|^2>`, dissipation `-2 nu <enstrophy>`, injection `<u . f>` |
| `fields.png` | final `u` and `v` |

No field files are written: the script calls `evolve` without `ostep`.

**Expected result (2026-10-01).** In this short run:
- energy goes `0.5025 → 0.5065`;
- dissipation goes `-0.0255 → -0.0324`;
- injection grows from `0` to `0.092`, as the flow picks up the forcing mode.

Long runs reach the chaotic regime used for the orbit searches (see `../newton_kolmog`,
`../ad_solver` and `~/kolmog/orb11_ad`).
