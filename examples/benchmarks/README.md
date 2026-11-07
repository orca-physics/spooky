Benchmarks of the AD paths (JAX backend).

- `bench_kolmogorov.py` — per-step cost of `KolmogorovFlow`: under NumPy `solver.evolve`;
  under JAX the jitted `ad_evolve` forward map, one JVP and one gradient through it, and
  `solver.evolve`. `--dtype float32` for the single-precision comparison.
