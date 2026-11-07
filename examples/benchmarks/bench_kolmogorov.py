"""Per-step cost of KolmogorovFlow (explicit RK2, fully periodic) under either backend.

NumPy backend: times ``solver.evolve`` (the path used so far by DynSys/Newton).
JAX backend:   times the jitted ``ad_evolve`` forward map, one JVP and one gradient
               (VJP) through it, and ``solver.evolve`` (jitted rkstep in a Python loop).

Examples
--------
    NUMPY_BACKEND=numpy python bench_kolmogorov.py
    NUMPY_BACKEND=jax   python bench_kolmogorov.py --grids 64,128,256,512 --steps 200
"""
import argparse
import os
import time

os.environ.setdefault('NUMPY_BACKEND', 'numpy')
BACKEND = os.environ['NUMPY_BACKEND']

parser = argparse.ArgumentParser()
parser.add_argument('--grids', default='64,128,256,512', help='comma-separated N (N x N grid)')
parser.add_argument('--steps', type=int, default=100, help='timesteps per timing')
parser.add_argument('--repeat', type=int, default=3, help='timed runs; the minimum is reported')
parser.add_argument('--dtype', default='float64', choices=['float64', 'float32'])
args = parser.parse_args()

import numpy as np

if BACKEND == 'jax':
    import jax
    jax.config.update('jax_enable_x64', args.dtype == 'float64')
    import jax.numpy as jnp
    sync = jax.block_until_ready
    print(f'jax {jax.__version__}  devices {jax.devices()}  dtype {args.dtype}')
else:
    sync = lambda out: out
    print(f'numpy {np.__version__}')

import spooky as sp
from spooky.solvers import KolmogorovFlow


def timed(fn, *a):
    out = fn(*a); sync(out)                           # compile + first run
    best = float('inf')
    for _ in range(args.repeat):
        t0 = time.perf_counter(); out = fn(*a); sync(out)
        best = min(best, time.perf_counter() - t0)
    return best


def bench(N, nsteps):
    dt = 1e-4
    grid = sp.Grid2D(Lx=2*np.pi, Ly=2*np.pi, Nx=N, Ny=N, dt=dt)
    solver = KolmogorovFlow(grid, kf=4, nu=1./40)
    rng = np.random.default_rng(0)
    xx, yy = np.asarray(grid.xx), np.asarray(grid.yy)
    uu = np.sin(yy) * np.cos(2*xx) + 0.1 * np.cos(3*yy + 1.0)
    vv = -0.5 * np.cos(2*yy) * np.sin(xx)
    T = (nsteps + 0.5) * dt                           # unambiguous step count
    ms = 1e3 / nsteps

    if BACKEND == 'jax':
        f0 = [jnp.array(uu), jnp.array(vv)]
        d0 = [jnp.array(rng.standard_normal((N, N))) for _ in range(2)]
        fwd = jax.jit(lambda f: solver.ad_evolve(f, T))
        jvp = jax.jit(lambda f, d: jax.jvp(lambda g: solver.ad_evolve(g, T), (f,), (d,))[1])
        grad = jax.jit(jax.grad(lambda f: sum(jnp.sum(o ** 2) for o in solver.ad_evolve(f, T))))
        t_f = timed(fwd, f0); t_j = timed(jvp, f0, d0)
        try:
            t_g = timed(grad, f0)
        except Exception as e:                        # un-checkpointed tape does not fit (GPU)
            t_g = float('nan')
            print(f'           grad skipped: {type(e).__name__} (tape of {nsteps} steps does not fit)')
        t_e = timed(lambda f: solver.evolve(f, T, write_outputs=False), f0)
        print(f'{N:4d}x{N:<4d} | fwd {ms*t_f:8.3f} ms/step | jvp {ms*t_j:8.3f} | '
              f'grad {ms*t_g:8.3f} | evolve {ms*t_e:8.3f}', flush=True)
    else:
        f0 = [uu, vv]
        t_e = timed(lambda f: solver.evolve([g.copy() for g in f], T, write_outputs=False), f0)
        print(f'{N:4d}x{N:<4d} | evolve {ms*t_e:8.3f} ms/step', flush=True)


for g in args.grids.split(','):
    bench(int(g), args.steps)
