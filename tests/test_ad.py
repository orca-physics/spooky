"""Tests for the AD (auto-differentiable) solver pipeline.

Run with:  NUMPY_BACKEND=jax pytest tests/test_ad.py
"""
import os
import pytest
import numpy as np

# Skip entire module when not running under JAX backend
pytestmark = pytest.mark.skipif(
    os.environ.get('NUMPY_BACKEND', 'numpy').lower() != 'jax',
    reason="AD tests require NUMPY_BACKEND=jax",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def kse_setup():
    import spooky as sp
    from spooky.solvers import KuramotoSivashinsky
    grid   = sp.Grid1D(22.0, 64, 1e-3)
    solver = KuramotoSivashinsky(grid, nu=1.0, rkord=2)
    return grid, solver


@pytest.fixture(scope='module')
def kf_setup():
    import spooky as sp
    from spooky.solvers import KolmogorovFlow
    grid   = sp.Grid2D(2 * np.pi, 2 * np.pi, 32, 32, 0.01)
    solver = KolmogorovFlow(grid, kf=4, f0=1.0, nu=0.1, rkord=2)
    return grid, solver


def _kse_ic(grid):
    Lx = grid.Lx
    return (0.3 * np.cos(2 * np.pi * 3 * grid.xx / Lx) +
            0.4 * np.cos(2 * np.pi * 5 * grid.xx / Lx) +
            0.5 * np.cos(2 * np.pi * 4 * grid.xx / Lx))


def _kf_ic(grid, seed=42):
    rng = np.random.default_rng(seed)
    uu  = rng.standard_normal(grid.shape)
    vv  = rng.standard_normal(grid.shape)
    fu, fv = grid.inc_proj([grid.forward(uu), grid.forward(vv)])
    return grid.inverse(fu), grid.inverse(fv)


# ---------------------------------------------------------------------------
# Trajectory correctness
# ---------------------------------------------------------------------------

def test_ad_evolve_matches_evolve_kse(kse_setup):
    """ad_evolve and evolve must produce identical trajectories (KSE, 1D)."""
    import jax.numpy as jnp
    grid, solver = kse_setup
    ic = _kse_ic(grid)

    ref = solver.evolve([ic], T=0.1, write_outputs=False)
    out = solver.ad_evolve([jnp.array(ic)], T=0.1)

    np.testing.assert_allclose(np.array(out[0]), ref[0], atol=1e-10,
                               err_msg="KSE: ad_evolve differs from evolve")


def test_ad_evolve_matches_evolve_kolmogorov(kf_setup):
    """ad_evolve and evolve must produce identical trajectories (Kolmogorov, 2D)."""
    import jax.numpy as jnp
    grid, solver = kf_setup
    uu, vv = _kf_ic(grid)

    ref = solver.evolve([uu, vv], T=0.1, write_outputs=False)
    out = solver.ad_evolve([jnp.array(uu), jnp.array(vv)], T=0.1)

    np.testing.assert_allclose(np.array(out[0]), ref[0], atol=1e-10)
    np.testing.assert_allclose(np.array(out[1]), ref[1], atol=1e-10)


def test_ad_evolve_fractional_final_step_kse(kse_setup):
    """ad_evolve must take the same exact fractional final step as evolve.

    Covers two regimes: T a genuine non-multiple of dt (100.5*dt), and T whose
    float division T/dt lands just below an integer (0.3/1e-3 = 299.999...),
    where a bare int() truncation would silently drop a full step.
    """
    import jax
    import jax.numpy as jnp
    grid, solver = kse_setup
    ic = _kse_ic(grid)

    for T in (100.5 * grid.dt, 0.3):
        ref = solver.evolve([ic], T=T, write_outputs=False)
        out = solver.ad_evolve([jnp.array(ic)], T=T)
        np.testing.assert_allclose(
            np.array(out[0]), ref[0], atol=1e-10,
            err_msg=f"KSE: ad_evolve differs from evolve at T={T}")

    # gradient through the fractional-step branch must be finite
    def loss(u):
        return jnp.sum(solver.ad_evolve([u], T=100.5 * grid.dt)[0] ** 2)

    grad = jax.grad(loss)(jnp.array(ic))
    assert jnp.all(jnp.isfinite(grad))


def test_ad_evolve_fractional_final_step_kolmogorov(kf_setup):
    """Same as the KSE fractional-step test, on the 2D Kolmogorov solver."""
    import jax.numpy as jnp
    grid, solver = kf_setup
    uu, vv = _kf_ic(grid)

    T = 10.5 * grid.dt
    ref = solver.evolve([uu, vv], T=T, write_outputs=False)
    out = solver.ad_evolve([jnp.array(uu), jnp.array(vv)], T=T)
    np.testing.assert_allclose(np.array(out[0]), ref[0], atol=1e-10)
    np.testing.assert_allclose(np.array(out[1]), ref[1], atol=1e-10)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_grad_compiles_and_is_finite_kse(kse_setup):
    """jax.grad through ad_evolve must compile and return a finite gradient."""
    import jax
    import jax.numpy as jnp
    grid, solver = kse_setup
    u0 = jnp.array(_kse_ic(grid))

    def loss(u):
        out = solver.ad_evolve([u], T=0.1)
        return jnp.sum(out[0] ** 2)

    grad = jax.grad(loss)(u0)
    assert grad.shape == u0.shape
    assert jnp.all(jnp.isfinite(grad)), "Gradient contains non-finite values"


def test_grad_compiles_and_is_finite_kolmogorov(kf_setup):
    """jax.grad through ad_evolve must compile and return a finite gradient."""
    import jax
    import jax.numpy as jnp
    grid, solver = kf_setup
    uu, vv = _kf_ic(grid)
    u0 = jnp.array(uu)
    v0 = jnp.array(vv)

    def loss(u):
        out = solver.ad_evolve([u, v0], T=0.1)
        return jnp.sum(out[0] ** 2) + jnp.sum(out[1] ** 2)

    grad = jax.grad(loss)(u0)
    assert grad.shape == u0.shape
    assert jnp.all(jnp.isfinite(grad)), "Gradient contains non-finite values"


# ---------------------------------------------------------------------------
# JVP accuracy: AD tangent vs finite-difference tangent
# ---------------------------------------------------------------------------

def test_jvp_matches_fd_kse(kse_setup):
    """DynSys.jvp_evolve tangent must agree with finite-difference tangent (KSE)."""
    import jax.numpy as jnp
    from spooky.methods import DynSys

    grid, solver = kse_setup
    ds = DynSys(solver)

    u0   = jnp.array(_kse_ic(grid))
    du0  = jnp.array(np.random.default_rng(0).standard_normal(u0.shape) * 1e-5)
    T    = 0.1

    _, tangent_ad = ds.jvp_evolve([u0], [du0], T)

    base  = solver.ad_evolve([u0], T)
    pert  = solver.ad_evolve([u0 + du0], T)
    tangent_fd = [pert[0] - base[0]]

    np.testing.assert_allclose(
        np.array(tangent_ad[0]), np.array(tangent_fd[0]),
        rtol=1e-4, atol=1e-8,
        err_msg="KSE: AD and FD tangents disagree",
    )


def test_jvp_evolve_compiles_once_per_key_kse(kse_setup, monkeypatch):
    """jvp_evolve jits the JVP once per (T, solver.ad_cache_key()): repeated
    calls (GMRES/Arnoldi) do not retrace ad_evolve, the result equals the
    eager jax.jvp, a new T or solver key compiles a new entry, and the cache
    is bounded by jvp_cache_size."""
    import jax
    import jax.numpy as jnp
    from spooky.methods import DynSys

    grid, solver = kse_setup
    traces = []
    ad_evolve = solver.ad_evolve
    monkeypatch.setattr(solver, 'ad_evolve',
                        lambda f, T, **kw: traces.append(T) or ad_evolve(f, T, **kw))
    ds = DynSys(solver)
    u0 = [jnp.array(_kse_ic(grid))]
    rng = np.random.default_rng(1)
    dus = [[jnp.array(rng.standard_normal(u0[0].shape))] for _ in range(3)]

    outs = [ds.jvp_evolve(u0, du, 0.1)[1] for du in dus]
    assert traces == [0.1]                                   # traced once, reused twice
    for du, out in zip(dus, outs):
        _, ref = jax.jvp(lambda f: ad_evolve(f, 0.1), (u0,), (du,))
        np.testing.assert_allclose(np.array(out[0]), np.array(ref[0]), rtol=1e-12, atol=1e-14)

    ds.jvp_evolve(u0, dus[0], 0.2)
    monkeypatch.setattr(solver, 'ad_cache_key', lambda: ('changed',))
    ds.jvp_evolve(u0, dus[0], 0.1)
    assert traces == [0.1, 0.2, 0.1] and len(ds._jvp_cache) == 3

    ds.jvp_cache_size = 2
    ds.jvp_evolve(u0, dus[0], 0.3)
    assert len(ds._jvp_cache) == 2


def test_tangent_map_batch_matches_columns_kse(kse_setup, monkeypatch):
    """tangent_map_batch (vmapped JVP, Lyapunov) equals tangent_map column by
    column and Phi_T(U), with and without a shift, and is compiled once per T:
    calls at new base points and tangents do not retrace ad_evolve."""
    from spooky.methods import DynSys

    grid, solver = kse_setup
    traces = []
    ad_evolve = solver.ad_evolve
    monkeypatch.setattr(solver, 'ad_evolve',
                        lambda f, T, **kw: traces.append(T) or ad_evolve(f, T, **kw))
    ds = DynSys(solver)
    rng = np.random.default_rng(3)
    T = 0.1
    for k in range(2):
        U = ds.flatten([_kse_ic(grid) * (1 + 0.1 * k)])
        Q = rng.standard_normal((len(U), 4))
        for sx in (None, 0.7):
            Uevol, V = ds.tangent_map_batch(U, Q, T, sx=sx)
            np.testing.assert_allclose(Uevol, ds.evolve(U, T), rtol=1e-12, atol=1e-13)
            for i in range(Q.shape[1]):
                ref = ds.tangent_map(U, Q[:, i], T, sx=sx)
                np.testing.assert_allclose(V[:, i], ref, rtol=1e-11, atol=1e-12 * np.abs(ref).max())
    assert traces.count(T) == 2                              # one batched + one single-tangent trace


def test_jvp_matches_fd_kolmogorov(kf_setup):
    """DynSys.jvp_evolve tangent must agree with finite-difference tangent (Kolmogorov)."""
    import jax.numpy as jnp
    from spooky.methods import DynSys

    grid, solver = kf_setup
    ds = DynSys(solver)

    uu, vv = _kf_ic(grid)
    rng    = np.random.default_rng(7)
    u0     = jnp.array(uu)
    v0     = jnp.array(vv)
    du0    = jnp.array(rng.standard_normal(uu.shape) * 1e-5)
    dv0    = jnp.array(rng.standard_normal(vv.shape) * 1e-5)
    T      = 0.1

    _, tangents_ad = ds.jvp_evolve([u0, v0], [du0, dv0], T)

    base = solver.ad_evolve([u0, v0], T)
    pert = solver.ad_evolve([u0 + du0, v0 + dv0], T)
    tangents_fd = [pert[i] - base[i] for i in range(2)]

    for i, (t_ad, t_fd) in enumerate(zip(tangents_ad, tangents_fd)):
        np.testing.assert_allclose(
            np.array(t_ad), np.array(t_fd),
            rtol=1e-4, atol=1e-8,
            err_msg=f"Kolmogorov field {i}: AD and FD tangents disagree",
        )


# ---------------------------------------------------------------------------
# VJP: adjoint consistency check (J^T·v, J·w inner product test)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Floquet multipliers: JVP-based tangent vs FD tangent
# ---------------------------------------------------------------------------

def test_floquet_jvp_matches_fd_kolmogorov(kf_setup):
    """floquet_multipliers apply_J (JVP) must agree with FD apply_J."""
    import jax.numpy as jnp
    from spooky.methods import DynSys

    grid, solver = kf_setup
    ds = DynSys(solver)

    uu, vv = _kf_ic(grid)
    U   = ds.flatten([jnp.array(uu), jnp.array(vv)])
    T   = 0.1

    # JVP path (JAX_ENABLED=True here)
    evals_jvp, _, _ = ds.floquet_multipliers([jnp.array(uu), jnp.array(vv)], T, n=3, tol=1e-6)

    # FD path — temporarily patch JAX_ENABLED in the module
    import spooky.methods.dyn_sys as dyn_mod
    orig = dyn_mod.JAX_ENABLED
    dyn_mod.JAX_ENABLED = False
    try:
        evals_fd, _, _ = ds.floquet_multipliers([uu, vv], T, n=3, tol=1e-6)
    finally:
        dyn_mod.JAX_ENABLED = orig

    np.testing.assert_allclose(
        np.sort(np.abs(evals_jvp)), np.sort(np.abs(evals_fd)),
        rtol=1e-4,
        err_msg="Floquet multipliers: JVP and FD Arnoldi disagree",
    )


# ---------------------------------------------------------------------------
# UPOGradientDescent
# ---------------------------------------------------------------------------

def test_gradient_descent_reduces_loss_kse(kse_setup):
    """UPOGradientDescent must reduce the periodicity loss from a random IC."""
    import jax.numpy as jnp
    from spooky.methods import UPOGradientDescent

    grid, solver = kse_setup
    gd = UPOGradientDescent(solver)

    u0 = jnp.array(_kse_ic(grid))
    T  = 0.5

    loss0 = float(gd.loss([u0], T))
    _, loss_f = gd.run([u0], T, n_iter=50, tol=1e-12, verbose=False)

    assert loss_f < loss0, f"Loss did not decrease: {loss0:.4e} -> {loss_f:.4e}"


def test_gradient_descent_reduces_loss_kolmogorov(kf_setup):
    """UPOGradientDescent must reduce the periodicity loss (Kolmogorov)."""
    import jax.numpy as jnp
    from spooky.methods import UPOGradientDescent

    grid, solver = kf_setup
    gd = UPOGradientDescent(solver)

    uu, vv = _kf_ic(grid)
    u0, v0 = jnp.array(uu), jnp.array(vv)
    T = 0.5

    loss0 = float(gd.loss([u0, v0], T))
    _, loss_f = gd.run([u0, v0], T, n_iter=50, tol=1e-12, verbose=False)

    assert loss_f < loss0, f"Loss did not decrease: {loss0:.4e} -> {loss_f:.4e}"


def test_ad_evolve_nsteps_matches_default_kse(kse_setup):
    """nsteps uniform steps of T/nsteps equal the default map when T = nsteps*dt."""
    import jax.numpy as jnp
    grid, solver = kse_setup
    u0 = jnp.array(_kse_ic(grid))
    n = 50
    T = n * grid.dt
    a = solver.ad_evolve([u0], T)[0]
    b = solver.ad_evolve([u0], T, nsteps=n)[0]
    np.testing.assert_allclose(np.array(b), np.array(a), atol=1e-12)


def test_ad_evolve_grad_wrt_T_matches_fd_kse(kse_setup):
    """With nsteps fixed, Phi is differentiable in T: jax.grad matches a central FD."""
    import jax
    import jax.numpy as jnp
    grid, solver = kse_setup
    u0 = jnp.array(_kse_ic(grid))
    n = 200
    f = lambda T: jnp.sum(solver.ad_evolve([u0], T, nsteps=n)[0] ** 2)
    T0, h = 0.2, 1e-6
    g_ad = float(jax.grad(f)(T0))
    g_fd = (float(f(T0 + h)) - float(f(T0 - h))) / (2 * h)
    np.testing.assert_allclose(g_ad, g_fd, rtol=1e-6)


def test_run_orbit_frees_T_and_shift_kse(kse_setup):
    """run_orbit with T and sx free reduces the shift-corrected loss, moves T and sx,
    keeps T within its bounds, and records one history entry per iteration."""
    import jax.numpy as jnp
    from spooky.methods import UPOGradientDescent
    grid, solver = kse_setup
    gd = UPOGradientDescent(solver)
    u0 = jnp.array(_kse_ic(grid))
    T0, sx0 = 0.5, 0.1

    loss0 = float(gd.loss([u0], T0, sx=sx0, nsteps=int(round(T0 / grid.dt))))
    fields, T, sx, loss_f, info = gd.run_orbit([u0], T0, sx0=sx0, n_iter=30, verbose=False)

    assert loss_f < loss0, f"Loss did not decrease: {loss0:.4e} -> {loss_f:.4e}"
    assert T != T0 and sx != sx0
    assert 0.75 * T0 <= T <= 1.25 * T0
    assert info['history'][0][0] == pytest.approx(loss0, rel=1e-10)
    assert len(info['history']) == info['nit'] + 1


def test_run_orbit_stays_solenoidal_kolmogorov(kf_setup):
    """With the default projection run_orbit returns a divergence-free state;
    without it, gradient descent from the same divergence-free start picks up
    divergent content (the failure mode the projection removes)."""
    import jax.numpy as jnp
    from spooky.methods import UPOGradientDescent
    grid, solver = kf_setup
    f0 = [np.array(f) for f in solver.inc_proj([jnp.array(f) for f in _kf_ic(grid)])]

    def div_rel(fields):
        # spectral divergence on the modes the solver keeps (the Nyquist column,
        # removed by dealiasing, is where inc_proj is not idempotent through irfft)
        uh, vh = [np.asarray(grid.forward(jnp.array(f))) for f in fields]
        kx, ky, keep = np.asarray(grid.kx), np.asarray(grid.ky), ~np.asarray(grid.dealias_modes)
        div = (kx * uh + ky * vh) * keep
        ref = np.sqrt(kx ** 2 + ky ** 2) * np.sqrt(np.abs(uh) ** 2 + np.abs(vh) ** 2) * keep
        return np.linalg.norm(div) / np.linalg.norm(ref)

    gd = UPOGradientDescent(solver)
    assert gd.project
    f_p, *_ = gd.run_orbit(f0, 0.5, sx0=0.0, n_iter=10, verbose=False)
    assert div_rel(f_p) < 1e-12

    gd_raw = UPOGradientDescent(solver, project=False)
    f_r, *_ = gd_raw.run_orbit(f0, 0.5, sx0=0.0, n_iter=10, verbose=False)
    assert div_rel(f_r) > 1e-8


# ---------------------------------------------------------------------------
# Lyapunov exponents: JVP-based tangent vs FD tangent
# ---------------------------------------------------------------------------

def test_lyapunov_jvp_matches_fd_kolmogorov(kf_setup):
    """lyapunov_exponents apply_J (JVP) must agree with FD apply_J (Kolmogorov).

    Both runs start with the same random Q (fixed seed) so the only difference
    is how apply_J evaluates J·dU.  After enough QR steps the finite-time
    exponents should agree within a few percent.
    """
    import jax.numpy as jnp
    from spooky.methods import DynSys

    grid, solver = kf_setup
    ds = DynSys(solver)

    uu, vv = _kf_ic(grid)
    T = 0.1
    n, nsteps = 3, 15

    np.random.seed(42)
    lyap_jvp, _ = ds.lyapunov_exponents([jnp.array(uu), jnp.array(vv)],
                                        T, n=n, nsteps=nsteps)

    import spooky.methods.dyn_sys as dyn_mod
    orig = dyn_mod.JAX_ENABLED
    dyn_mod.JAX_ENABLED = False
    try:
        np.random.seed(42)
        lyap_fd, _ = ds.lyapunov_exponents([uu, vv], T, n=n, nsteps=nsteps)
    finally:
        dyn_mod.JAX_ENABLED = orig

    np.testing.assert_allclose(
        lyap_jvp, lyap_fd,
        rtol=0.05,
        err_msg="Lyapunov exponents: JVP and FD Benettin disagree",
    )


def test_vjp_adjoint_consistency_kolmogorov(kf_setup):
    """<J·w, v> == <w, J^T·v> to machine precision (Kolmogorov)."""
    import jax.numpy as jnp
    from spooky.methods import DynSys

    grid, solver = kf_setup
    ds = DynSys(solver)

    uu, vv  = _kf_ic(grid)
    rng     = np.random.default_rng(13)
    u0, v0  = jnp.array(uu), jnp.array(vv)
    w       = [jnp.array(rng.standard_normal(uu.shape)),
               jnp.array(rng.standard_normal(vv.shape))]
    z       = [jnp.array(rng.standard_normal(uu.shape)),
               jnp.array(rng.standard_normal(vv.shape))]
    T       = 0.1

    _, Jw   = ds.jvp_evolve([u0, v0], w, T)
    _, JTz  = ds.vjp_evolve([u0, v0], z, T)

    # <J·w, z>
    lhs = sum(float(jnp.sum(Jw[i] * z[i]))  for i in range(2))
    # <w, J^T·z>
    rhs = sum(float(jnp.sum(w[i] * JTz[0][i])) for i in range(2))

    np.testing.assert_allclose(lhs, rhs, rtol=1e-6,
                               err_msg="Adjoint consistency <Jw,z> != <w,J^Tz>")


# ---------------------------------------------------------------------------
# Checkpointed scan: same values and gradients as the plain scan
# ---------------------------------------------------------------------------

def test_checkpointed_ad_evolve_matches_plain_kse(kse_setup):
    """checkpoint_every must not change Phi_T or its gradient (KSE)."""
    import jax
    import jax.numpy as jnp
    grid, solver = kse_setup
    u0 = jnp.array(_kse_ic(grid))
    T = 37 * grid.dt          # 37 steps: chunks of 5 leave a remainder of 2

    plain = solver.ad_evolve([u0], T)[0]
    for ck in (5, 'auto'):
        out = solver.ad_evolve([u0], T, checkpoint_every=ck)[0]
        np.testing.assert_allclose(np.array(out), np.array(plain), atol=1e-12)

    g_plain = jax.grad(lambda u: jnp.sum(solver.ad_evolve([u], T)[0] ** 2))(u0)
    g_ck = jax.grad(lambda u: jnp.sum(solver.ad_evolve([u], T, checkpoint_every=5)[0] ** 2))(u0)
    np.testing.assert_allclose(np.array(g_ck), np.array(g_plain), rtol=1e-10, atol=1e-12)


def test_checkpointed_grad_uses_less_memory_kolmogorov(kf_setup):
    """Compiled temp memory of the checkpointed gradient must be well below
    the plain tape for a long scan."""
    import jax
    import jax.numpy as jnp
    grid, solver = kf_setup
    uu, vv = _kf_ic(grid)
    f0 = [jnp.array(uu), jnp.array(vv)]
    T = 400 * grid.dt

    def loss(ck):
        return jax.jit(jax.grad(lambda f: sum(jnp.sum(o ** 2) for o in solver.ad_evolve(f, T, checkpoint_every=ck))))

    m_plain = loss(None).lower(f0).compile().memory_analysis().temp_size_in_bytes
    m_ck = loss(20).lower(f0).compile().memory_analysis().temp_size_in_bytes
    assert m_ck < 0.5 * m_plain, f"checkpointing did not reduce memory: {m_ck} vs {m_plain}"


def test_index_update_mask_lowers_to_select(kf_setup):
    """A full-shape boolean mask must become a select, not a scatter over every
    True entry (on GPU that scatter cost ~100 ms per call at 128x128, 1000x the
    rest of the Kolmogorov rkstep), with the same values and gradient."""
    import jax
    import jax.numpy as jnp
    from spooky._backend import index_update
    grid, _ = kf_setup
    mask = grid.dealias_modes
    rng = np.random.default_rng(0)
    a = jnp.array(rng.standard_normal(mask.shape) + 1j * rng.standard_normal(mask.shape))

    np.testing.assert_array_equal(np.array(index_update(a, mask, 0.0)),
                                  np.array(a.at[mask].set(0.0)))
    assert 'scatter' not in str(jax.make_jaxpr(lambda x: index_update(x, mask, 0.0))(a))
    g = jax.grad(lambda x: jnp.sum(jnp.abs(index_update(x, mask, 0.0)) ** 2))(a)
    np.testing.assert_allclose(np.array(g)[np.array(mask)], 0.0)
    # index tuples (zero_mode) still take the .at[] path
    assert index_update(a, grid.zero_mode, 0.0)[grid.zero_mode] == 0.0
