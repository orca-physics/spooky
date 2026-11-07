"""Space-time Fourier loop: a (relative) periodic orbit guess as grid values
over one period, its residual minimized directly.

Reference: Parker & Schneider, J. Fluid Mech. 2022 (the PV method, here with
the pressure eliminated exactly).
"""
import os

import numpy as np

from .. import pseudo as ps
from .._backend import JAX_ENABLED, index_update

if JAX_ENABLED:
    import jax
    import jax.numpy as jnp

LOG_COLS = ['it', 'wall', 'loss', 'phys', 'phys_div', 'phys_mom', 'A', 'T', 's',
            'grad_fields', 'grad_Ts', 'step', 'lr', 'ls_steps', 'decrease_err',
            'curvature_err']


class FourierLoop:
    """The loop of a periodic solver: the fields U as grid values on Nt x
    Nx x Ny points of the comoving coordinates (tau, x, y), tau = k/Nt in
    [0, 1), periodic in all three, plus the period T and the shift per
    period s. The lab-frame field is u(t, x) = U(t/T, x - s*t/T): the
    pattern moves by s in x over a period, so translate(u(T), s) = u(0), s
    being UPONewtonSolver's sx. Its time derivative is
    Dt U = (1/T)(dU/dtau - s dU/dx), and the momentum residual
    is P(Dt U - rhs(U)), dealiased, with P the solver's solenoidal
    projection: the pressure is eliminated exactly (P is what grad p cancels
    per mode k != 0; k = 0 untouched), never a descent variable.

    U itself is not projected (unless ``solenoidal``): continuity is a
    penalty, the residual's first component. Every field is truncated to
    the spatially dealiased modes, without its spatial mean, and without
    the tau Nyquist mode (even Nt), whose d/dtau is 0; there is no
    dealiasing in time.

    The unknowns are a pytree ``params = {'a', 'T', 's'}`` with a = U*sqrt(w),
    w = Lx*Ly*2*pi/(Nt*Nx*Ny) the quadrature weight of a grid point (tau
    counted as 2*pi, as P&S), so |a|^2 is the L2 norm of U over the loop.
    Only the dealiased modes reach the fields; the other directions of a
    have zero gradient. Arrays are (field, tau, x, y).

    Requires the JAX backend, in float64 (JAX_ENABLE_X64=1); optimize also
    needs optax.

    Parameters
    ----------
    solver : PseudoSpectral instance on a Grid2D
        Provides ``rhs`` (with the projected advection), ``grid.inc_proj``,
        ``load_fields`` and ``num_fields``.
    Nt : int
        Number of points in tau.
    solenoidal : bool, optional
        Project U too (f0 = 0 to roundoff). Default False.

    Example
    -------
    >>> loop = FourierLoop(solver, Nt=32)
    >>> params = loop.from_window('output', steps, T=7.4, s=0.0)
    >>> params, it, reason = loop.optimize(params, 'odir_loop', max_iters=10**6)
    >>> fields, T, sx = loop.to_fields(params, newt.grid)
    >>> newt.run_newton(newt.form_X(newt.flatten(fields), T, sx))
    """

    def __init__(self, solver, Nt, solenoidal=False):
        if not JAX_ENABLED:
            raise RuntimeError('FourierLoop requires NUMPY_BACKEND=jax')
        if not isinstance(solver.grid, ps.Grid2D):
            raise NotImplementedError('FourierLoop: Grid2D only')
        if not jax.config.jax_enable_x64:
            raise RuntimeError('FourierLoop requires float64 (JAX_ENABLE_X64=1)')
        self.solver = solver
        self.grid = solver.grid
        self.Nt = Nt
        self.solenoidal = solenoidal
        self.num_fields = solver.num_fields
        self.shape = (self.num_fields, Nt) + self.grid.shape

        grid = self.grid
        self.w = grid.Lx*grid.Ly*2*np.pi/(Nt*grid.N)

        # d/dtau on fft over tau, the Nyquist mode (even Nt) zeroed and dropped
        m = np.fft.fftfreq(Nt, 1.0/Nt)
        nyq = np.zeros(Nt, bool)
        if Nt % 2 == 0:
            m[Nt//2] = 0.0
            nyq[Nt//2] = True
        self.dtau = jnp.asarray(2j*np.pi*m)[:, None, None]
        self.keep_tau = jnp.asarray(~nyq)[:, None, None]

        # Spatial masks: dealiased modes, and the fields' (without k = 0)
        dealias = ~np.asarray(grid.dealias_modes)
        nonzero = (np.asarray(grid.ki) != 0) | (np.asarray(grid.kj) != 0)
        self.dealias = jnp.asarray(dealias)
        self.keep = jnp.asarray(dealias & nonzero)

        self._forward = jax.vmap(grid.forward)
        self._inverse = jax.vmap(grid.inverse)
        rhs = lambda Fh: jnp.stack(solver.rhs(list(Fh)))
        self._rhs = jax.vmap(rhs, in_axes=1, out_axes=1)

    def make(self, U, T, s):
        """params from grid values U (num_fields, Nt, Nx, Ny), T and s"""
        U = np.asarray(U)
        if U.shape != self.shape:
            raise ValueError(f'U.shape = {U.shape}, expected {self.shape}')
        return {'a': jnp.asarray(np.sqrt(self.w)*U, jnp.float64),
                'T': jnp.asarray(T, jnp.float64),
                's': jnp.asarray(s, jnp.float64)}

    def forward(self, U):
        """Spatial transform of (num_fields, Nt, Nx, Ny) fields"""
        return jnp.stack([self._forward(F) for F in U])

    def inverse(self, Uh):
        """Inverse of forward"""
        return jnp.stack([self._inverse(Fh) for Fh in Uh])

    def ddtau(self, Uh):
        """d/dtau of spatial spectra (num_fields, Nt, ...), spectral in tau"""
        return jnp.fft.ifft(self.dtau*jnp.fft.fft(Uh, axis=1), axis=1)

    def spectra(self, params):
        """Spatial spectra of the fields: U truncated to the dealiased modes,
        spatial means and the tau Nyquist mode removed (and projected with
        solenoidal)"""
        Uh = self.forward(params['a']/np.sqrt(self.w))
        Uh = jnp.fft.ifft(self.keep_tau*jnp.fft.fft(Uh, axis=1), axis=1)
        Uh = self.keep*Uh
        if self.solenoidal:
            Uh = jnp.stack(self.grid.inc_proj(list(Uh)))
        return Uh

    def fields(self, params):
        """The fields U on the grid, (num_fields, Nt, Nx, Ny)"""
        return self.inverse(self.spectra(params))

    def residual(self, params):
        """(1 + num_fields, Nt, Nx, Ny): f0 the divergence, then the
        momentum residual P(Dt U - rhs(U)), dealiased"""
        grid = self.grid
        T, s = params['T'], params['s']
        Uh = self.spectra(params)
        DtUh = (self.ddtau(Uh) - s*grid.deriv(Uh, grid.kx))/T
        Rh = DtUh - self._rhs(Uh)
        Rh = self.dealias*jnp.stack(grid.inc_proj(list(Rh)))
        f0h = grid.deriv(Uh[0], grid.kx) + grid.deriv(Uh[1], grid.ky)
        return jnp.concatenate([self.inverse(f0h[None]), self.inverse(Rh)])

    def fluct(self, params):
        """A: the grid mean of |U - <U>_tau|^2 over the fields (comoving, so
        a traveling wave has A = 0)"""
        F = self.fields(params)
        return jnp.mean(jnp.sum((F - F.mean(1, keepdims=True))**2, 0))

    def loss(self, params):
        """The sum over the residual's components of the grid mean of f^2.
        Returns (total, diagnostics), A logged only."""
        R = self.residual(params)
        per_eq = jnp.mean(R**2, (1, 2, 3))
        phys = jnp.sum(per_eq)
        diag = {'phys': phys, 'phys_div': per_eq[0],
                'phys_mom': jnp.sum(per_eq[1:]), 'A': self.fluct(params)}
        return phys, diag

    def from_window(self, path, steps, T, s):
        """The loop of a DNS segment, as P&S build theirs: the solver's
        output at the Nt given steps (one period's worth, evenly spaced),
        resampled to the grid and taken to the comoving frame (the k-th
        translated by s*k/Nt, so U(tau_k, x) = u(t_k, x + s*tau_k)), closed
        with whatever jump the window has"""
        if len(steps) != self.Nt:
            raise ValueError(f'{len(steps)} steps for Nt = {self.Nt}')
        snaps = [resample(self.solver.load_fields(path, st), self.grid.shape)
                 for st in steps]
        snaps = [[np.asarray(f) for f in self.grid.translate(list(u), s*k/self.Nt)]
                 for k, u in enumerate(snaps)]
        U = np.array(snaps).transpose(1, 0, 2, 3)
        return self.make(U, T, s)

    def snapshot(self, params, shape=None):
        """The fields at tau = 0 (comoving and lab frames coincide there),
        resampled to shape (default: the grid's)"""
        U0 = np.asarray(self.fields(params)[:, 0])
        return resample(U0, shape or self.grid.shape)

    def to_fields(self, params, grid=None):
        """(fields, T, sx): the fields at tau = 0 on grid (default: the
        loop's), cleaned there as the solver's own (solenoidal projection,
        zero mean, dealiased), as a Newton search starts from them:
        X = newt.form_X(newt.flatten(fields), T, sx). grid must span the
        loop's domain (Lx, Ly)."""
        grid = grid or self.grid
        if not np.allclose((grid.Lx, grid.Ly), (self.grid.Lx, self.grid.Ly)):
            raise ValueError(f'grid domain {(grid.Lx, grid.Ly)}, the loop\'s is '
                             f'{(self.grid.Lx, self.grid.Ly)}')
        f = [grid.forward(jnp.asarray(u)) for u in self.snapshot(params, grid.shape)]
        f = grid.inc_proj(f)
        f = [index_update(ff, grid.zero_mode, 0.0) for ff in f]
        f = [index_update(ff, grid.dealias_modes, 0.0) for ff in f]
        fields = [np.asarray(grid.inverse(ff)) for ff in f]
        return fields, float(params['T']), float(params['s'])

    def save(self, path, params):
        """A state (.npz): a, T, s and the loop's shape"""
        np.savez(path, a=np.asarray(params['a']), T=float(params['T']),
                 s=float(params['s']), Nt=self.Nt, Nx=self.grid.Nx,
                 Ny=self.grid.Ny, solenoidal=self.solenoidal)

    def load(self, path):
        """params of a saved state, checked against this loop's shape"""
        with np.load(path) as z:
            saved = (int(z['Nt']), int(z['Nx']), int(z['Ny']), bool(z['solenoidal']))
            mine = (self.Nt, self.grid.Nx, self.grid.Ny, self.solenoidal)
            if saved != mine:
                raise ValueError(f'{path}: (Nt, Nx, Ny, solenoidal) {saved}, '
                                 f'the loop is {mine}')
            return {'a': jnp.asarray(z['a'], jnp.float64),
                    'T': jnp.asarray(z['T'][()], jnp.float64),
                    's': jnp.asarray(z['s'][()], jnp.float64)}

    def optimize(self, params, odir, max_iters, wall_hours=None,
                 memory_size=10, stall_iters=20, loss_floor=1e-16,
                 plateau_iters=20000, plateau_drop=0.01, log_every=1000,
                 save_every=50000, on_save=None, verbose=True):
        """L-BFGS (optax) on the loss over the flat vector of params (the L2
        metric on the fields, weight 1 on T and s). T and s are not clipped:
        clipping after an update would corrupt the L-BFGS memory.

        Stops at max_iters, after wall_hours, after stall_iters consecutive
        zero steps (optax's zoom line search takes a zero step only when the
        loss is not finite along the direction and no safe step was found;
        an ordinary failed search takes a nonzero step), when T <= 0 or the
        loss is not finite, when the loss is below loss_floor (roundoff), or
        at a plateau: at a log row, the loss fell by less than the fraction
        plateau_drop since the row plateau_iters earlier (a positive
        multiple of log_every). The stops test the loss at the new iterate,
        the one logged. Returns (params, it, reason).

        Writes in odir (a run with saved states is refused: restart from
        one with load() into a new odir):
            fourier.dat  a row every log_every iterations (columns in its
                         header): the loss and its terms, A, T, s, the
                         gradient's norm on the fields and on (T, s), and
                         the last step's norm, line search step size,
                         steps, decrease and curvature errors
            states/loop-IIIIIII.npz  every save_every iterations and at
                         the end (save(); 0 is the start)
            inverse.dat  it, T, s at each saved state
            info.txt, done (iterations and the reason)
        on_save(it, params), if given, runs at each saved state.
        """
        import time
        import optax
        from jax.flatten_util import ravel_pytree

        if plateau_iters < log_every or plateau_iters % log_every:
            raise ValueError(f'plateau_iters {plateau_iters} must be a positive '
                             f'multiple of log_every {log_every}')
        t0 = time.time()
        states = os.path.join(odir, 'states')
        if os.path.isdir(states) and os.listdir(states):
            raise ValueError(f'{states} is not empty: a new run needs a new odir')
        os.makedirs(states, exist_ok=True)
        for f in ('fourier.dat', 'inverse.dat', 'done'):
            if os.path.exists(os.path.join(odir, f)):
                os.remove(os.path.join(odir, f))

        y, unravel = ravel_pytree(params)
        marks = ravel_pytree({'a': jnp.zeros_like(params['a']),
                              'T': jnp.asarray(1.0), 's': jnp.asarray(2.0)})[0]
        iT, i_s = int(jnp.argmax(marks == 1.0)), int(jnp.argmax(marks == 2.0))
        terms = lambda y: self.loss(unravel(y))
        loss = lambda y: terms(y)[0]

        @jax.jit
        def diagnostics(y):
            (total, diag), g = jax.value_and_grad(terms, has_aux=True)(y)
            gTs = jnp.sqrt(g[iT]**2 + g[i_s]**2)
            gF = jnp.sqrt(jnp.maximum(jnp.sum(g**2) - gTs**2, 0.0))
            return dict(diag, loss=total, grad_fields=gF, grad_Ts=gTs)

        opt = optax.lbfgs(memory_size=memory_size)

        @jax.jit
        def step(y, state):
            value, grad = optax.value_and_grad_from_state(loss)(y, state=state)
            updates, state = opt.update(grad, state, y, value=value, grad=grad,
                                        value_fn=loss)
            info = optax.tree_utils.tree_get(state, 'info')
            # value above is the loss at y (cached by the last line search);
            # the state now holds the loss at the new iterate
            ls = dict(step=optax.tree_utils.tree_norm(updates),
                      lr=optax.tree_utils.tree_get(state, 'learning_rate'),
                      ls_steps=info.num_linesearch_steps,
                      decrease_err=info.decrease_error,
                      curvature_err=info.curvature_error,
                      value=optax.tree_utils.tree_get(state, 'value'))
            return optax.apply_updates(y, updates), state, ls

        log = os.path.join(odir, 'fourier.dat')
        with open(log, 'w') as f:
            print('# ' + ' '.join(LOG_COLS), file=f)

        def row(it, y, ls=None):
            vals = {k: float(v) for k, v in diagnostics(y).items()}
            vals.update(it=it, wall=time.time() - t0, T=float(y[iT]), s=float(y[i_s]))
            vals.update({k: float(v) for k, v in (ls or {}).items()})
            with open(log, 'a') as f:
                print(' '.join(f'{vals.get(c, np.nan):.10g}' for c in LOG_COLS),
                      file=f)
            if verbose:
                print(f'it {it} loss {vals["loss"]:.4e} div {vals["phys_div"]:.3e} '
                      f'mom {vals["phys_mom"]:.3e} A {vals["A"]:.4g} '
                      f'T {vals["T"]:.6f} s {vals["s"]:+.6f} '
                      f'|g| {vals["grad_fields"]:.3e} {vals["grad_Ts"]:.3e} '
                      f'wall {vals["wall"]:.1f} s', flush=True)

        def save(it, y):
            p = unravel(y)
            self.save(os.path.join(states, f'loop-{it:07d}.npz'), p)
            with open(os.path.join(odir, 'inverse.dat'), 'a') as f:
                print(it, float(p['T']), float(p['s']), file=f)
            if on_save is not None:
                on_save(it, p)

        with open(os.path.join(odir, 'info.txt'), 'w') as f:
            print(f'Nt {self.Nt} grid {self.grid.shape} solenoidal {self.solenoidal} '
                  f'unknowns {y.size} memory_size {memory_size}', file=f)
            print(f'jax {jax.__version__} {jax.devices()} optax {optax.__version__}',
                  file=f)
        row(0, y)
        save(0, y)

        state = opt.init(y)
        it, stalled, reason, ls = 0, 0, 'max_iters', None
        lag = plateau_iters//log_every
        hist = []                    # the loss at each log row
        while it < max_iters:
            if wall_hours is not None and time.time() - t0 > 3600*wall_hours:
                reason = 'wall time'
                break
            it += 1
            y, state, ls = step(y, state)
            stalled = stalled + 1 if float(ls['step']) == 0.0 else 0
            value = float(ls.pop('value'))
            plateau = False
            if it % log_every == 0:
                row(it, y, ls)
                hist.append(value)
                plateau = (len(hist) > lag
                           and value > (1.0 - plateau_drop)*hist[-1 - lag])
            if it % save_every == 0:
                save(it, y)
            if stalled >= stall_iters:
                reason = f'{stalled} zero steps'
            elif not np.isfinite(value):
                reason = 'loss not finite'
            elif float(y[iT]) <= 0.0:
                reason = 'T <= 0'
            elif value < loss_floor:
                reason = 'loss floor'
            elif plateau:
                reason = 'plateau'
            else:
                continue
            break
        if it % log_every:
            row(it, y, ls)
        if it % save_every:
            save(it, y)
        with open(os.path.join(odir, 'done'), 'w') as f:
            print(it, reason, file=f)
        return unravel(y), it, reason


def resample(F, shape):
    """Fields on a periodic grid (..., Mx, My) on shape = (Nx, Ny) points
    (or N for N x N), by truncating or zero-padding the spectrum (the
    Nyquist modes of even sizes dropped); exact for fields with
    |ki| < min(Mi, Ni)/2"""
    F = np.asarray(F)
    Nx, Ny = (shape, shape) if np.isscalar(shape) else shape
    Mx, My = F.shape[-2:]
    Fh = np.fft.fft2(F, axes=(-2, -1))
    nx, ny = (min(Mx, Nx) + 1)//2, (min(My, Ny) + 1)//2
    ix, iy = np.r_[0:nx, -nx+1:0], np.r_[0:ny, -ny+1:0]
    out = np.zeros(F.shape[:-2] + (Nx, Ny), complex)
    out[..., ix[:, None], iy[None, :]] = Fh[..., ix[:, None], iy[None, :]]
    return np.fft.ifft2(out*(Nx*Ny)/(Mx*My), axes=(-2, -1)).real
