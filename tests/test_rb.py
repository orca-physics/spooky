"""Tests for the 2D Rayleigh-Benard solver (Fourier-x / Chebyshev-z, IMEX).

All times, timesteps and growth rates are in free-fall units.  Physics tests
(boundary conditions, incompressibility, critical Rayleigh number) run under
the default NumPy backend.  The AD tests at the bottom are gated on
NUMPY_BACKEND=jax, like tests/test_ad.py.

Run physics tests:   pytest tests/test_rb.py
Run AD tests:        NUMPY_BACKEND=jax pytest tests/test_rb.py
"""
import os
import numpy as np
import pytest

import spooky as sp
from spooky.solvers import RayleighBenard2D
from spooky import rb_convert as rc

JAX = os.environ.get('NUMPY_BACKEND', 'numpy').lower() == 'jax'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed(grid, kc_mode=1, amp=1e-2):
    """Single-roll temperature perturbation that vanishes at the walls."""
    z = np.asarray(grid.zi) / grid.Lz
    x = np.linspace(0, grid.Lx, grid.Nx, endpoint=False)[:, None]
    th = amp * np.sin(np.pi * z)[None, :] * np.cos(2 * np.pi * kc_mode * x / grid.Lx)
    zero = np.zeros((grid.Nx, grid.Nz))
    return [zero.copy(), zero.copy(), th]


def _leading_sigma(bc, ac, ra, Nx=16, Nz=33, dt=8e-3, T=0.8, niter=30, amp=1e-8):
    """Leading linear growth rate (free-fall units) via tiny-amplitude power
    iteration.

    The box is sized to the critical wavenumber (Lx = 2*pi/ac) so the
    fundamental x-mode is the marginal one.
    """
    grid = sp.Grid2D_wall(2 * np.pi / ac, 1.0, Nx, Nz, dt)
    solver = RayleighBenard2D(grid, ra=ra, pr=1.0, bc=bc)
    f = _seed(grid, amp=1.0)
    f[1] = f[2].copy()                       # excite w as well
    sig = []
    for _ in range(niter):
        n0 = np.sqrt(grid.average(f[0] ** 2 + f[1] ** 2 + f[2] ** 2))
        f = [a / n0 * amp for a in f]
        f = solver.evolve([a.copy() for a in f], T=T, write_outputs=False)
        n1 = np.sqrt(grid.average(f[0] ** 2 + f[1] ** 2 + f[2] ** 2))
        sig.append(np.log(n1 / amp) / T)
    return float(np.mean(sig[-8:]))


# ---------------------------------------------------------------------------
# Grid (Chebyshev) sanity
# ---------------------------------------------------------------------------

def test_chebyshev_derivative_and_quadrature():
    grid = sp.Grid2D_wall(2 * np.pi, 1.0, 16, 24, 0.01)
    z = np.asarray(grid.zi)
    f = np.sin(3 * z)
    np.testing.assert_allclose(f @ np.asarray(grid.Dz).T, 3 * np.cos(3 * z), atol=1e-9)
    np.testing.assert_allclose(f @ np.asarray(grid.Dz2).T, -9 * np.sin(3 * z), atol=1e-7)
    # integral of z^2 over [0,1] = 1/3
    fld = np.tile((z ** 2)[None, :], (grid.Nx, 1))
    assert abs(float(grid.average(fld)) - 1.0 / 3.0) < 1e-10


@pytest.mark.parametrize('Nx', [48, 50, 64, 96])
def test_dealiasing_no_alias_onto_kept_modes(Nx):
    """u = cos(k x) sin(z)^2 at the highest kept mode k: u u_x = -k/2 sin(2kx) sin(z)^4
    has no content at any kept mode, so the dealiased advection must vanish there
    (with k <= Nx/3 kept, Nx = 48 and 96 aliased mode Nx/3 onto itself)."""
    grid = sp.Grid2D_wall(2 * np.pi, np.pi, Nx, 17, 1e-3)
    solver = RayleighBenard2D(grid, ra=1e6)
    kept = ~np.asarray(grid.dealias_modes)[:, 0]
    k = int(np.nonzero(kept)[0].max())
    assert 3 * k < Nx
    x = np.linspace(0, 2 * np.pi, Nx, endpoint=False)[:, None]
    z = np.asarray(grid.zi)[None, :]
    u = np.cos(k * x) * np.sin(z) ** 2
    a_u, _, _ = solver._advection(grid.forward(u), grid.forward(np.zeros_like(u)),
                                  grid.forward(np.zeros_like(u)))
    assert np.abs(np.asarray(a_u)[kept]).max() < 1e-10 * k * Nx


# ---------------------------------------------------------------------------
# Boundary conditions + incompressibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('Lz', [1.0, np.pi])
@pytest.mark.parametrize('scheme', ['sbdf2', 'be'])
@pytest.mark.parametrize('bc', ['no-slip', 'free-slip'])
def test_boundary_conditions_and_divergence(bc, scheme, Lz):
    grid = sp.Grid2D_wall(2.0 * Lz, Lz, 32, 33, 5e-3)
    solver = RayleighBenard2D(grid, ra=2000.0, pr=1.0, bc=bc, scheme=scheme)
    u, w, t = solver.evolve(_seed(grid), T=1.0, write_outputs=False)

    # no penetration and fixed temperature at both walls (all BC types)
    assert np.max(np.abs(w[:, [0, -1]])) < 1e-10
    assert np.max(np.abs(t[:, [0, -1]])) < 1e-10

    uh = grid.forward(u)
    if bc == 'no-slip':
        assert np.max(np.abs(u[:, [0, -1]])) < 1e-10          # u = 0
    else:
        duz = (uh @ np.asarray(grid.Dz).T)[:, [0, -1]]        # du/dz = 0
        assert np.max(np.abs(duz)) < 1e-8

    wh = grid.forward(w)
    div = 1j * grid.kx_col * uh + wh @ np.asarray(grid.Dz).T
    assert np.max(np.abs(div)) < 1e-9


# ---------------------------------------------------------------------------
# Units: the layer height Lz is a pure length rescaling
# ---------------------------------------------------------------------------

def test_length_scale_invariance():
    """Free-fall units with layer height Lz: the Lz = pi (SPECTER) solution is
    the Lz = 1 one with x, z, u, w and theta scaled by pi, at the same times;
    Nu is the same."""
    dt, T, s = 1e-2, 0.5, np.pi
    g1 = sp.Grid2D_wall(2.0, 1.0, 16, 25, dt)
    gs = sp.Grid2D_wall(2.0 * s, s, 16, 25, dt)
    s1 = RayleighBenard2D(g1, ra=3000.0, pr=1.0, bc='no-slip')
    ss = RayleighBenard2D(gs, ra=3000.0, pr=1.0, bc='no-slip')
    f1 = _seed(g1)
    f1[1] = 0.5 * f1[2]
    a = s1.evolve([np.array(x) for x in f1], T=T, write_outputs=False)
    b = ss.evolve([s * np.array(x) for x in f1], T=T, write_outputs=False)
    for x, y in zip(a, b):
        np.testing.assert_allclose(np.asarray(y), s * np.asarray(x),
                                   rtol=0, atol=1e-11 * np.max(np.abs(s * np.asarray(x))))
    np.testing.assert_allclose(float(ss.nusselt(b)), float(s1.nusselt(a)), rtol=1e-11)


# ---------------------------------------------------------------------------
# Temporal convergence order
# ---------------------------------------------------------------------------

def _evolve_at_dt(scheme, dt, T=1.0, Nx=16, Nz=25, ra=3e3):
    grid = sp.Grid2D_wall(2.0, 1.0, Nx, Nz, dt)
    solver = RayleighBenard2D(grid, ra=ra, pr=1.0, bc='no-slip', scheme=scheme)
    return solver.evolve(_seed(grid), T=T, write_outputs=False)


def _l2_diff(f, g):
    return np.sqrt(sum(np.sum((np.asarray(a) - np.asarray(b)) ** 2)
                       for a, b in zip(f, g)))


@pytest.mark.parametrize('scheme, expected', [('be', 1.0), ('sbdf2', 2.0)])
def test_temporal_order(scheme, expected):
    """Global temporal order: 1 for backward Euler, 2 for SBDF2.

    Errors against a fine-dt reference of the same scheme, measured over a
    factor-4 dt refinement.  Also checks that the SBDF2 backward-Euler
    startup step does not degrade the global order.
    """
    ref = _evolve_at_dt(scheme, dt=6.25e-4)
    e_coarse = _l2_diff(_evolve_at_dt(scheme, dt=2e-2), ref)
    e_fine = _l2_diff(_evolve_at_dt(scheme, dt=5e-3), ref)
    order = np.log2(e_coarse / e_fine) / 2.0     # refinement factor 4
    assert abs(order - expected) < 0.3, (
        f"{scheme}: measured order {order:.2f}, expected {expected}")


def test_sbdf2_more_accurate_than_be():
    """At equal dt, SBDF2 must beat backward Euler by a wide margin."""
    ref = _evolve_at_dt('sbdf2', dt=6.25e-4)
    e_be = _l2_diff(_evolve_at_dt('be', dt=1e-2), ref)
    e_sbdf2 = _l2_diff(_evolve_at_dt('sbdf2', dt=1e-2), ref)
    assert e_sbdf2 < 0.1 * e_be


# ---------------------------------------------------------------------------
# Critical Rayleigh number (the gold-standard physics check)
# ---------------------------------------------------------------------------

def test_critical_rayleigh_no_slip():
    # Ra_c = 1707.76 at a_c = 3.117 (no-slip).  Bracket it.
    assert _leading_sigma('no-slip', 3.117, 1650.0) < 0.0
    assert _leading_sigma('no-slip', 3.117, 1770.0) > 0.0
    # marginal: |sigma| < 0.05 diffusive rates = 0.05/sqrt(Ra) in free-fall units
    assert abs(_leading_sigma('no-slip', 3.117, 1707.76)) < 0.05 / np.sqrt(1707.76)


def test_critical_rayleigh_free_slip():
    # Ra_c = 657.5 = 27*pi^4/4 at a_c = 2.221 (free-slip).  Bracket it.
    assert _leading_sigma('free-slip', 2.221, 600.0) < 0.0
    assert _leading_sigma('free-slip', 2.221, 720.0) > 0.0
    assert abs(_leading_sigma('free-slip', 2.221, 657.5)) < 0.05 / np.sqrt(657.5)


# ---------------------------------------------------------------------------
# Exact realisation of T (dt rescaled to T/round(T/dt), operators cached per dt)
# ---------------------------------------------------------------------------

def _norm(fields):
    return np.sqrt(sum(np.sum(np.asarray(f) ** 2) for f in fields))


def test_evolve_realises_exact_T():
    """A period that is not a multiple of grid.dt is integrated with
    round(T/dt) steps of T/round(T/dt): identical to a solver built with that
    dt, and the map is continuous in T across a change of the step count
    (the Newton solvers rely on both)."""
    dt = 1e-2
    grid = sp.Grid2D_wall(2.0, 1.0, 16, 25, dt)
    solver = RayleighBenard2D(grid, ra=3000.0, pr=1.0, bc='no-slip')
    f0 = _seed(grid)
    run = lambda slv, T: slv.evolve([np.array(x) for x in f0], T=T, write_outputs=False)

    T = 10.3 * dt
    a = run(solver, T)
    solver2 = RayleighBenard2D(sp.Grid2D_wall(2.0, 1.0, 16, 25, T / 10), ra=3000.0, pr=1.0, bc='no-slip')
    b = run(solver2, T)
    for x, y in zip(a, b):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=0, atol=1e-13)
    # cache keys (dt, ra, pr), dt rounded to 13 digits
    assert {k[0] for k in solver._ops_cache} == {dt, float(f'{T / 10:.12e}')}

    eps = 1e-6 * dt                                     # step count 10 -> 11 at 10.5 dt
    lo, hi, ref = run(solver, 10.5 * dt - eps), run(solver, 10.5 * dt + eps), run(solver, 10 * dt)
    jump = _norm([np.asarray(x) - np.asarray(y) for x, y in zip(hi, lo)])
    change = _norm([np.asarray(x) - np.asarray(y) for x, y in zip(hi, ref)])
    assert jump < 1e-2 * change, (jump, change)
    assert len(solver._ops_cache) <= solver.ops_cache_size


def test_mean_mode_and_operator_conditioning():
    """The k=0 mode has w = 0 exactly (continuity + walls) for any right-hand
    side, and the implicit operators are well conditioned: sets rebuilt for
    coefficients a few ulp apart give the same step.  (The k=0 influence matrix
    is singular; inverting it anyway gave a spurious k=0 w and ~1e-5 rebuild
    noise.)"""
    grid = sp.Grid2D_wall(2 * np.pi, np.pi, 32, 33, 1e-2)
    solver = RayleighBenard2D(grid, ra=1e6, pr=1.0, bc='no-slip')
    rng = np.random.default_rng(2)
    r = [rng.standard_normal((grid.Nkx, grid.Nz)) + 1j * rng.standard_normal((grid.Nkx, grid.Nz))
         for _ in range(3)]
    lam = 1.5 / grid.dt
    a = solver._imex_solve(solver._build_stokes_ops(lam), *r)
    b = solver._imex_solve(solver._build_stokes_ops(lam * (1 + 4 * np.finfo(float).eps)), *r)
    assert np.max(np.abs(np.asarray(a[1])[0])) == 0.0
    for x, y in zip(a, b):
        x, y = np.asarray(x), np.asarray(y)
        assert np.max(np.abs(x - y)) < 1e-12 * np.max(np.abs(x))


def test_evolve_follows_ra_update():
    """Changing solver.ra (as arclength continuation does) must rebuild the
    implicit operators (cache keyed on ra), not reuse stale ones -- under JAX
    the cached jitted step receives them as traced arguments."""
    grid = sp.Grid2D_wall(2.0, 1.0, 16, 25, 1e-2)
    solver = RayleighBenard2D(grid, ra=3000.0, pr=1.0, bc='no-slip')
    f0 = _seed(grid)
    a = solver.evolve([np.array(x) for x in f0], T=0.5, write_outputs=False)
    solver.ra = 6000.0
    b = solver.evolve([np.array(x) for x in f0], T=0.5, write_outputs=False)
    ref = RayleighBenard2D(grid, ra=6000.0, pr=1.0, bc='no-slip').evolve(
        [np.array(x) for x in f0], T=0.5, write_outputs=False)
    solver.ra = 3000.0
    c = solver.evolve([np.array(x) for x in f0], T=0.5, write_outputs=False)
    assert np.max(np.abs(np.array(a[2]) - np.array(b[2]))) > 1e-6
    for x, y in zip(b, ref):
        np.testing.assert_allclose(np.array(x), np.array(y), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.array(c[2]), np.array(a[2]), atol=1e-13)


@pytest.mark.skipif(not JAX, reason="AD tests require NUMPY_BACKEND=jax")
def test_cached_jvp_follows_ra_update():
    """DynSys caches its compiled JVP per (T, solver.ad_cache_key()); changing
    solver.ra must compile a new map with the new operators, not reuse the
    old one."""
    import jax.numpy as jnp
    from spooky.methods import DynSys
    grid = sp.Grid2D_wall(2.0, 1.0, 16, 25, 1e-2)
    solver = RayleighBenard2D(grid, ra=3000.0, pr=1.0, bc='no-slip')
    ds = DynSys(solver)
    f0 = [jnp.array(a) for a in _seed(grid)]
    df = [jnp.array(a) for a in _seed(grid, kc_mode=2, amp=1.0)]
    a = ds.jvp_evolve(f0, df, 0.5)[1]
    solver.ra = 6000.0
    b = ds.jvp_evolve(f0, df, 0.5)[1]
    ref = DynSys(RayleighBenard2D(grid, ra=6000.0, pr=1.0, bc='no-slip')).jvp_evolve(f0, df, 0.5)[1]
    assert np.max(np.abs(np.array(a[2]) - np.array(b[2]))) > 1e-6
    for x, y in zip(b, ref):
        np.testing.assert_allclose(np.array(x), np.array(y), rtol=0, atol=1e-13)
    assert len(ds._jvp_cache) == 2


# DynSys integration (wall-bounded remove_boundary path)
# ---------------------------------------------------------------------------

def test_dynsys_flatten_roundtrip():
    from spooky.methods import DynSys
    grid = sp.Grid2D_wall(2.0, 1.0, 16, 21, 1e-2)
    solver = RayleighBenard2D(grid, ra=3000.0, pr=1.0, bc='no-slip')
    ds = DynSys(solver)
    assert ds.remove_boundary is True

    fields = _seed(grid)
    U = ds.flatten(fields)
    assert U.shape[0] == grid.Nx * (grid.Nz - 2) * solver.num_fields

    f2 = ds.unflatten(U)
    for orig, back in zip(fields, f2):
        assert np.max(np.abs(np.asarray(back)[:, [0, -1]])) == 0.0     # walls zeroed
        np.testing.assert_allclose(np.asarray(back)[:, 1:-1], orig[:, 1:-1], atol=1e-12)


def test_dynsys_weights():
    """Weighted flat state: round-trips, |U|^2 is the physical mean square,
    evolve is unchanged, and (JAX) the tangent map is conjugated by the
    weights: tangent_map_w(W U, W dU) = W tangent_map(U, dU)."""
    from spooky.methods import DynSys
    grid = sp.Grid2D_wall(2.0, 1.0, 16, 21, 1e-2)
    solver = RayleighBenard2D(grid, ra=3000.0, pr=1.0, bc='no-slip')
    ds = DynSys(solver, weights=rc.quadrature_weights(grid))
    fields = _seed(grid)
    fields[0] = 3.0 * _seed(grid, kc_mode=2, amp=1.0)[2]     # nonzero u, w (zero at walls)
    fields[1] = 0.5 * fields[2]
    U = ds.flatten(fields)
    for back, orig in zip(ds.unflatten(U), fields):
        assert np.max(np.abs(back[:, [0, -1]])) == 0.0
        np.testing.assert_allclose(back[:, 1:-1], orig[:, 1:-1], rtol=1e-12)
    ms = sum(float(grid.average(f ** 2)) for f in fields)
    np.testing.assert_allclose(np.dot(U, U), ms, rtol=1e-12)

    out = ds.unflatten(ds.evolve(U, 0.5))
    ref = solver.evolve([np.array(f) for f in fields], T=0.5, write_outputs=False)
    for a, b in zip(out, ref):
        np.testing.assert_allclose(a[:, 1:-1], np.asarray(b)[:, 1:-1], rtol=1e-12, atol=1e-15)

    if JAX:
        ds0 = DynSys(solver)
        U0 = ds0.flatten(fields)
        pert = _seed(grid, kc_mode=2, amp=1.0)
        pert[1] = pert[2] * 3.0
        dU0 = ds0.flatten(pert)
        W = ds.flatten(ds0.unflatten(np.ones_like(U0)))         # the diagonal weights
        lhs = ds.tangent_map(U, W * dU0, 0.5)
        rhs = W * ds0.tangent_map(U0, dU0, 0.5)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-11 * np.abs(rhs).max())


def test_newton_steady_roll_weighted_and_plain():
    """UPONewtonSolver on the steady roll at Ra = 3000 (fixed point of the
    T = 1 map, x-shift free): converges with and without quadrature weights
    to the same state up to translation; run_newton returns the final X and
    calls on_iteration after every accepted step."""
    from spooky.methods import UPONewtonSolver
    grid = sp.Grid2D_wall(2.0, 1.0, 16, 17, 0.05)
    solver = RayleighBenard2D(grid, ra=3000.0, pr=1.0, bc='no-slip')
    f0 = solver.evolve(_seed(grid, amp=0.05), T=15.0, write_outputs=False)
    no_io = {k: None for k in ('output_dir', 'balance_dir', 'newton_dir', 'gmres_dir',
                               'hookstep_dir', 'trust_region_dir', 'apply_A_dir', 'converged_dir')}
    res = {}
    for name, w in (('plain', None), ('weighted', rc.quadrature_weights(grid))):
        ns = UPONewtonSolver(solver, T=None, sx=0.0, Tconst=1.0, N_newt=8, N_gmres=60,
                             tol_newt=1e-9, save_outputs='none', save_balance='none',
                             weights=w, **no_io)
        log = []
        X = ns.run_newton(ns.form_X(ns.flatten(f0), None, 0.0),
                          on_iteration=lambda iN, X, F: log.append((iN, F, X.copy())))
        U, T, sx = ns.unpack_X(X)
        UT = ns.translate(ns.evolve(U, T), sx)
        assert np.linalg.norm(UT - U) < 1e-12 * np.linalg.norm(U)
        assert [it[0] for it in log] == list(range(1, len(log) + 1))
        assert log[0][1] > 1e3 * log[3][1]                       # fast (quadratic) decrease
        res[name] = ns.unflatten(U)
    nu = [float(solver.nusselt(res[k])) for k in ('plain', 'weighted')]
    np.testing.assert_allclose(nu[1], nu[0], rtol=1e-10)
    for a, b in zip(res['plain'], res['weighted']):              # x-translation invariant
        A, B = np.abs(grid.forward(a)), np.abs(grid.forward(b))
        np.testing.assert_allclose(B, A, rtol=0, atol=1e-9 * A.max())


def test_incompressibility_projection():
    grid = sp.Grid2D_wall(2.0, 1.0, 16, 25, 1e-2)
    solver = RayleighBenard2D(grid, ra=2000.0, pr=1.0, bc='no-slip')
    rng = np.random.default_rng(0)
    u = rng.standard_normal((grid.Nx, grid.Nz))
    w = rng.standard_normal((grid.Nx, grid.Nz))
    t = rng.standard_normal((grid.Nx, grid.Nz))
    up, wp, tp = solver.inc_proj([u, w, t])
    uh = grid.forward(up); wh = grid.forward(wp)
    div = 1j * grid.kx_col * uh + wh @ np.asarray(grid.Dz).T
    # Leray projection: divergence-free at the interior collocation points and
    # no penetration at the walls (the wall-row divergence is not enforced by
    # the Neumann-Poisson collocation -- a standard Chebyshev artifact).
    assert np.max(np.abs(div[:, 1:-1])) < 1e-9        # divergence-free interior
    assert np.max(np.abs(wp[:, [0, -1]])) < 1e-9      # no penetration
    np.testing.assert_allclose(tp, t)                 # temperature passed through


# ---------------------------------------------------------------------------
# AD (JAX backend only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not JAX, reason="AD tests require NUMPY_BACKEND=jax")
@pytest.mark.parametrize('scheme, T', [('sbdf2', 0.5), ('be', 0.5), ('sbdf2', 0.513)])
def test_ad_evolve_matches_evolve(scheme, T):
    import jax.numpy as jnp
    grid = sp.Grid2D_wall(2.0, 1.0, 16, 25, 1e-2)
    solver = RayleighBenard2D(grid, ra=3000.0, pr=1.0, bc='no-slip', scheme=scheme)
    f0 = _seed(grid)
    ref = solver.evolve([np.array(a) for a in f0], T=T, write_outputs=False)
    out = solver.ad_evolve([jnp.array(a) for a in f0], T=T)
    for i in range(3):
        np.testing.assert_allclose(np.array(out[i]), ref[i], atol=1e-6)


@pytest.mark.skip(reason="d/dRa by AD needs the implicit operators built in jnp "
                         "(ra enters them through nu and kappa); future work")
def test_dnu_dra_matches_fd():
    """Parameter gradient d(Nu)/d(Ra) through the solver vs central FD.

    Kept as the specification for a traced-ra ad_evolve.
    """
    import jax
    import jax.numpy as jnp
    grid = sp.Grid2D_wall(2.0, 1.0, 16, 25, 1e-2)
    solver = RayleighBenard2D(grid, ra=3000.0, pr=1.0, bc='no-slip')
    f0 = [jnp.array(a) for a in _seed(grid)]

    def nu_of_ra(ra):
        out = solver.ad_evolve(f0, T=0.5, ra=ra)
        return solver.nusselt(out)

    g = float(jax.grad(nu_of_ra)(jnp.array(3000.0)))
    dra = 1.0
    fd = float((nu_of_ra(jnp.array(3000.0 + dra)) -
                nu_of_ra(jnp.array(3000.0 - dra))) / (2 * dra))
    assert abs(g - fd) / (abs(fd) + 1e-30) < 1e-4


@pytest.mark.skipif(not JAX, reason="AD tests require NUMPY_BACKEND=jax")
def test_grad_through_solver_is_finite():
    import jax, jax.numpy as jnp
    grid = sp.Grid2D_wall(2.0, 1.0, 16, 25, 1e-2)
    solver = RayleighBenard2D(grid, ra=3000.0, pr=1.0, bc='no-slip')
    f0 = [jnp.array(a) for a in _seed(grid)]

    def loss(th):
        out = solver.ad_evolve([f0[0], f0[1], th], T=0.5)
        return jnp.sum(out[2] ** 2)

    g = jax.grad(loss)(f0[2])
    assert g.shape == f0[2].shape
    assert bool(jnp.all(jnp.isfinite(g)))


@pytest.mark.skipif(not JAX, reason="AD tests require NUMPY_BACKEND=jax")
def test_jvp_matches_fd():
    import jax
    import jax.numpy as jnp
    grid = sp.Grid2D_wall(2.0, 1.0, 16, 25, 1e-2)
    solver = RayleighBenard2D(grid, ra=3000.0, pr=1.0, bc='no-slip')
    f0 = [jnp.array(a) for a in _seed(grid)]
    rng = np.random.default_rng(1)
    eps = 1e-6
    du = [jnp.array(rng.standard_normal((grid.Nx, grid.Nz)) * eps) for _ in range(3)]
    _, tan = jax.jvp(lambda f: solver.ad_evolve(f, 0.5), (f0,), (du,))
    base = solver.ad_evolve(f0, 0.5)
    pert = solver.ad_evolve([f0[i] + du[i] for i in range(3)], 0.5)
    for i in range(3):
        fd = np.array(pert[i] - base[i])
        ad = np.array(tan[i])
        denom = np.max(np.abs(fd)) + 1e-30
        assert np.max(np.abs(ad - fd)) / denom < 0.05


@pytest.mark.skipif(not JAX, reason="AD tests require NUMPY_BACKEND=jax")
def test_checkpointed_ad_evolve_matches_plain():
    """SBDF2 carry through a checkpointed scan: same fields and gradient."""
    import jax
    import jax.numpy as jnp
    grid = sp.Grid2D_wall(2.0, 1.0, 16, 25, 1e-2)
    solver = RayleighBenard2D(grid, ra=3000.0, pr=1.0, bc='no-slip')
    f0 = [jnp.array(a) for a in _seed(grid)]
    T = 23 * grid.dt
    plain = solver.ad_evolve(f0, T)
    ck = solver.ad_evolve(f0, T, checkpoint_every=4)
    for a, b in zip(plain, ck):
        np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-12)

    def loss(th, ck):
        out = solver.ad_evolve([f0[0], f0[1], th], T, checkpoint_every=ck)
        return jnp.sum(out[2] ** 2)

    g0 = jax.grad(lambda th: loss(th, None))(f0[2])
    g1 = jax.grad(lambda th: loss(th, 4))(f0[2])
    np.testing.assert_allclose(np.array(g1), np.array(g0), rtol=1e-10, atol=1e-14)
