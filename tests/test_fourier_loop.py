"""Tests for FourierLoop (methods/fourier_loop.py). The residual is checked
against an analytic band-limited loop (modes |m| <= 2, |kx|, |ky| <= 2, so
the products stay within N/3), and against a reference computed by an
independent implementation on a random loop
(fixtures/fourier_loop_reference.npz).

Run with:  NUMPY_BACKEND=jax pytest tests/test_fourier_loop.py
"""
import os
import pytest
import numpy as np

pytestmark = pytest.mark.skipif(
    os.environ.get('NUMPY_BACKEND', 'numpy').lower() != 'jax',
    reason="FourierLoop tests require NUMPY_BACKEND=jax",
)

nu = 1.0/40
Nt, N = 8, 24
FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


def make_solver(n, nu=nu):
    import spooky as sp
    from spooky.solvers import KolmogorovFlow
    grid = sp.Grid2D(2*np.pi, 2*np.pi, n, n, 1e-3)
    return KolmogorovFlow(grid, kf=4, f0=1.0, nu=nu)


@pytest.fixture(scope='module')
def solver():
    return make_solver(N)


def make_loop(solver, Nt=Nt, solenoidal=False):
    from spooky.methods import FourierLoop
    return FourierLoop(solver, Nt, solenoidal=solenoidal)


class Series:
    """(u, v, p) as sums of Re(C exp(i ph)), ph = 2*pi*m*tau + kx*x + ky*y in
    the comoving coordinates, with exact derivatives"""
    def __init__(self, seed=0, p_seed=None):
        self.modes = np.array([(m, kx, ky) for m in range(3)
                               for kx in range(-2, 3) for ky in range(-2, 3)
                               if (kx, ky) != (0, 0)], float)
        rng = np.random.default_rng(seed)
        A, B = 0.3*rng.standard_normal((2, 3, len(self.modes)))
        if p_seed is not None:
            A[2], B[2] = 0.3*np.random.default_rng(p_seed).standard_normal(
                (2, len(self.modes)))
        self.C = A - 1j*B
        tau = np.arange(Nt)/Nt
        x = 2*np.pi*np.arange(N)/N
        TT, XX, YY = np.meshgrid(tau, x, x, indexing='ij')
        m, kx, ky = self.modes.T
        self.E = np.exp(1j*(2*np.pi*m*TT[..., None] + kx*XX[..., None]
                            + ky*YY[..., None]))

    def __call__(self, dtau=0, dx=0, dy=0):
        """(3, Nt, N, N) values of d^(dtau, dx, dy) of (u, v, p)"""
        m, kx, ky = self.modes.T
        f = (2j*np.pi*m)**dtau*(1j*kx)**dx*(1j*ky)**dy
        return np.einsum('cj,...j->c...', self.C*f, self.E).real

    def residual(self, T, s):
        """The momentum residual with p, (2, Nt, N, N), and continuity"""
        U, Ut = self(), self(dtau=1)
        Ux, Uy = self(dx=1), self(dy=1)
        lap = self(dx=2) + self(dy=2)
        Dt = (Ut - s*Ux)/T
        adv = U[0]*Ux + U[1]*Uy
        y = 2*np.pi*np.arange(N)/N
        f = np.zeros((2, 1, N, N))
        f[0] = np.sin(4*y)
        R = Dt[:2] + adv[:2] - nu*lap[:2] - f + np.stack([Ux[2], Uy[2]])
        return R, Ux[0] + Uy[1]


def leray(F):
    """F (2, ..., N, N) without its component along k, k != 0 (numpy, full
    fft2, independent of Grid2D.inc_proj)"""
    k = np.fft.fftfreq(N, 1.0/N)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    K2 = np.where(KX**2 + KY**2 > 0, KX**2 + KY**2, 1.0)
    G1, G2 = np.fft.fft2(F[0]), np.fft.fft2(F[1])
    kG = (KX*G1 + KY*G2)/K2
    return np.stack([np.fft.ifft2(G1 - KX*kG).real,
                     np.fft.ifft2(G2 - KY*kG).real])


def test_residual_matches_analytic(solver):
    """Continuity exact, momentum the analytic one projected, whatever p"""
    loop = make_loop(solver)
    for p_seed in (None, 10):
        S = Series(p_seed=p_seed)
        params = loop.make(S()[:2], 7.7, 0.3)
        R = np.asarray(loop.residual(params))
        R_ref, div = S.residual(7.7, 0.3)
        scale = np.abs(R_ref).max()
        assert scale > 1.0
        np.testing.assert_allclose(R[0], div, atol=1e-12*scale)
        np.testing.assert_allclose(R[1:], leray(R_ref), atol=1e-12*scale)
    # Two p give different residuals and the same reduced one
    diff = S.residual(7.7, 0.3)[0] - Series().residual(7.7, 0.3)[0]
    assert np.abs(diff).max() > 0.1


def test_reduced_is_min_over_p(solver):
    """The reduced loss is the one with any p minus the momentum residual's
    longitudinal part, which grad p can cancel"""
    S = Series(seed=6)
    R_ref, div = S.residual(7.7, 0.3)
    loop = make_loop(solver)
    R = np.asarray(loop.residual(loop.make(S()[:2], 7.7, 0.3)))
    full = np.mean(div**2) + np.sum(np.mean(R_ref**2, (1, 2, 3)))
    reduced = np.sum(np.mean(R**2, (1, 2, 3)))
    long = np.sum(np.mean((R_ref - leray(R_ref))**2, (1, 2, 3)))
    assert long > 1.0
    np.testing.assert_allclose(full - reduced, long, rtol=1e-10)


def test_matches_reference():
    """The reference's residual, loss terms and fields on a random loop (not
    solenoidal, nonzero means, modes above N/3 and at the tau Nyquist),
    free and solenoidal"""
    ref = np.load(os.path.join(FIXTURES, 'fourier_loop_reference.npz'))
    U = ref['U']
    solver = make_solver(U.shape[-1], nu=float(ref['nu']))
    for pre, solenoidal in [('', False), ('sol_', True)]:
        loop = make_loop(solver, U.shape[1], solenoidal)
        params = loop.make(U, float(ref['T']), float(ref['s']))
        for got, key in [(loop.fields(params), 'fields'),
                         (loop.residual(params), 'residual')]:
            want = ref[pre + key]
            np.testing.assert_allclose(np.asarray(got), want,
                                       atol=1e-12*np.abs(want).max())
        _, diag = loop.loss(params)
        for key in ('phys', 'phys_mom'):
            np.testing.assert_allclose(float(diag[key]), float(ref[pre + key]),
                                       rtol=1e-12)
        np.testing.assert_allclose(float(diag['phys_div']),
                                   float(ref[pre + 'phys_div']),
                                   rtol=1e-12, atol=1e-25)
        np.testing.assert_allclose(float(diag['A']), float(ref[pre + 'A']),
                                   rtol=1e-12)


def test_fields_and_fluct(solver):
    loop = make_loop(solver)
    Y = Series(seed=1)()[:2]
    params = loop.make(Y, 7.7, 0.3)
    # Band-limited, zero spatial mean: fields() returns it unchanged
    np.testing.assert_allclose(np.asarray(loop.fields(params)), Y, atol=1e-12)
    A = np.mean(np.sum((Y - Y.mean(1, keepdims=True))**2, 0))
    np.testing.assert_allclose(float(loop.fluct(params)), A, rtol=1e-12)
    # A mean flow and modes above N/3 are removed
    U = Y.copy()
    U[0] += 0.5
    U[1] += np.cos(10*2*np.pi*np.arange(N)/N)[None, :, None]
    np.testing.assert_allclose(np.asarray(loop.fields(loop.make(U, 7.7, 0.3))),
                               Y, atol=1e-12)


def test_metric_is_l2(solver):
    """|a|^2 is the integral of u^2 + v^2 over (2*pi)^3"""
    loop = make_loop(solver)
    params = loop.make(Series(seed=2)()[:2], 7.7, 0.3)
    U = np.asarray(loop.fields(params))
    integral = (2*np.pi)**3*np.mean(np.sum(U**2, 0))
    np.testing.assert_allclose(float(np.sum(np.asarray(params['a'])**2)),
                               integral, rtol=1e-12)


def test_loss(solver):
    import jax
    loop = make_loop(solver)
    params = loop.make(Series(seed=3)()[:2], 7.7, 0.3)
    R = np.asarray(loop.residual(params))
    total, diag = loop.loss(params, lambda_phys=2.0)
    phys = 2.0*np.sum(np.mean(R**2, (1, 2, 3)))
    np.testing.assert_allclose(float(total), phys, rtol=1e-12)
    np.testing.assert_allclose(float(diag['phys']), phys, rtol=1e-12)
    np.testing.assert_allclose(float(diag['phys_div'] + diag['phys_mom']),
                               phys, rtol=1e-12)
    np.testing.assert_allclose(float(diag['A']), float(loop.fluct(params)),
                               rtol=1e-12)
    g = jax.grad(lambda p: loop.loss(p)[0])(params)
    assert np.all(np.isfinite(np.asarray(g['a'])))
    assert np.isfinite(float(g['T'])) and np.isfinite(float(g['s']))


def test_resample(solver):
    from spooky.methods.fourier_loop import resample
    loop = make_loop(solver)
    F = np.asarray(loop.fields(loop.make(Series(seed=4)()[:2], 7.7, 0.3)))
    up = resample(F, 64)
    np.testing.assert_allclose(resample(up, N), F, atol=1e-12)
    # The export's path: one snapshot, padded to 64
    params = loop.make(F, 7.7, 0.3)
    np.testing.assert_allclose(loop.snapshot(params, (64, 64)), up[:, 0],
                               atol=1e-12)


def test_resample_against_analytic():
    """Up and down against the field evaluated on each grid, square and not"""
    from spooky.methods.fourier_loop import resample

    def field(nx, ny):
        x = 2*np.pi*np.arange(nx)/nx
        y = 2*np.pi*np.arange(ny)/ny
        X, Y = np.meshgrid(x, y, indexing='ij')
        return np.cos(3*X + 2*Y) + np.sin(5*Y) - 0.4*np.cos(X - 7*Y)
    np.testing.assert_allclose(resample(field(N, N), 64), field(64, 64), atol=1e-12)
    np.testing.assert_allclose(resample(field(64, 64), N), field(N, N), atol=1e-12)
    np.testing.assert_allclose(resample(field(N, 32), (48, 20)), field(48, 20),
                               atol=1e-12)


def test_solenoidal(solver):
    """solenoidal: the fields are the Leray projection of the free ones, f0
    vanishes, and the momentum residual is that of the projected fields"""
    U = Series(seed=7)()[:2]
    free, sol = make_loop(solver), make_loop(solver, solenoidal=True)
    Uf = np.asarray(free.fields(free.make(U, 7.7, 0.3)))
    Us = np.asarray(sol.fields(sol.make(U, 7.7, 0.3)))
    np.testing.assert_allclose(leray(Uf), Us, atol=1e-12)
    assert np.abs(Uf - Us).max() > 0.1
    R = np.asarray(sol.residual(sol.make(U, 7.7, 0.3)))
    assert np.abs(R[0]).max() < 1e-12
    R_free = np.asarray(free.residual(free.make(Us, 7.7, 0.3)))
    np.testing.assert_allclose(R, R_free, atol=1e-12*np.abs(R).max())


def test_from_window(tmp_path, solver):
    """The solver's snapshots at the given steps, resampled; T and s as given;
    a wrong number of steps is refused"""
    rng = np.random.default_rng(5)
    steps = [0, 10, 20, 30]
    for st in steps:
        for name in ('uu', 'vv'):
            np.save(tmp_path/f'{name}.{st:04}.npy', rng.standard_normal((32, 32)))
    loop = make_loop(solver, Nt=4)
    params = loop.from_window(str(tmp_path), steps, 7.4, 0.1)
    assert float(params['T']) == 7.4 and float(params['s']) == 0.1
    from spooky.methods.fourier_loop import resample
    U = np.asarray(params['a'])/np.sqrt(loop.w)
    u = resample(np.load(tmp_path/'uu.0030.npy'), N)
    v = resample(np.load(tmp_path/'vv.0030.npy'), N)
    np.testing.assert_allclose(U[:, 3], np.stack([u, v]), atol=1e-12)
    with pytest.raises(ValueError):
        loop.from_window(str(tmp_path), steps[:3], 7.4, 0.1)


def test_laminar_is_exact(solver):
    """The laminar state u = sin(4y)/(16 nu), v = 0 (steady, any s) has zero
    residual"""
    loop = make_loop(solver)
    y = 2*np.pi*np.arange(N)/N
    U = np.zeros((2, Nt, N, N))
    U[0] = np.sin(4*y)/(16*nu)
    for s in (0.0, 0.3):
        R = np.asarray(loop.residual(loop.make(U, 7.7, s)))
        assert np.abs(R).max() < 1e-12


def test_gradient_finite_differences(solver):
    """jax.grad of the loss against central differences, for T, s and a few
    grid values"""
    import jax
    loop = make_loop(solver)
    params = loop.make(Series(seed=8)()[:2], 7.7, 0.3)
    f = lambda p: loop.loss(p)[0]
    g = jax.grad(f)(params)
    h = 1e-6

    def bump(key, idx=()):
        def b(e):
            p = dict(params)
            p[key] = params[key].at[idx].add(e)
            return p
        return b
    checks = [(bump('T'), g['T']), (bump('s'), g['s'])]
    for idx in [(0, 1, 2, 3), (1, 5, 10, 7), (0, 7, 23, 0)]:
        checks.append((bump('a', idx), g['a'][idx]))
    for b, gj in checks:
        fd = (float(f(b(h))) - float(f(b(-h))))/(2*h)
        # atol: the differences' roundoff, ~1e-16*f/h
        np.testing.assert_allclose(float(gj), fd, rtol=1e-6,
                                   atol=1e-8*float(f(params)))


def test_tau_nyquist_dropped(solver):
    """The tau Nyquist mode (even Nt) does not reach the fields"""
    loop = make_loop(solver)
    x = 2*np.pi*np.arange(N)/N
    U = np.zeros((2, Nt, N, N))
    U[0] = ((-1.0)**np.arange(Nt))[:, None, None]*np.cos(x)[None, :, None]
    assert np.abs(np.asarray(loop.fields(loop.make(U, 7.7, 0.3)))).max() < 1e-13
