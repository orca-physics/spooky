"""Space-time Fourier loop: a (relative) periodic orbit guess as grid values
over one period, its residual minimized directly.

Reference: Parker & Schneider, J. Fluid Mech. 2022 (the PV method, here with
the pressure eliminated exactly).
"""
import numpy as np

from .. import pseudo as ps
from .._backend import JAX_ENABLED

if JAX_ENABLED:
    import jax
    import jax.numpy as jnp


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

    Requires the JAX backend, in float64 (JAX_ENABLE_X64=1).

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
    >>> total, diag = loop.loss(params)
    """

    def __init__(self, solver, Nt, solenoidal=False):
        if not JAX_ENABLED:
            raise RuntimeError('FourierLoop requires NUMPY_BACKEND=jax')
        if not isinstance(solver.grid, ps.Grid2D):
            raise NotImplementedError('FourierLoop: Grid2D only')
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

    def loss(self, params, lambda_phys=1.0):
        """The sum over the residual's components of the grid mean of
        lambda_phys*f^2. Returns (total, diagnostics), A logged only."""
        R = self.residual(params)
        per_eq = jnp.mean(lambda_phys*R**2, (1, 2, 3))
        phys = jnp.sum(per_eq)
        diag = {'phys': phys, 'phys_div': per_eq[0],
                'phys_mom': jnp.sum(per_eq[1:]), 'A': self.fluct(params)}
        return phys, diag

    def from_window(self, path, steps, T, s):
        """The loop of a DNS segment, as P&S build theirs: the solver's
        output at the Nt given steps (one period's worth, evenly spaced),
        resampled to the grid, closed with whatever jump the window has"""
        if len(steps) != self.Nt:
            raise ValueError(f'{len(steps)} steps for Nt = {self.Nt}')
        snaps = [self.solver.load_fields(path, st) for st in steps]
        U = np.array(snaps).transpose(1, 0, 2, 3)
        return self.make(resample(U, self.grid.shape), T, s)

    def snapshot(self, params, shape=None):
        """The fields at tau = 0 (comoving and lab frames coincide there),
        resampled to shape (default: the grid's)"""
        U0 = np.asarray(self.fields(params)[:, 0])
        return resample(U0, shape or self.grid.shape)


def resample(F, shape):
    """Fields on a periodic grid (..., Mx, My) on shape = (Nx, Ny) points
    (or N for N x N), by truncating or zero-padding the spectrum (the
    Nyquist modes dropped); exact for fields with |ki| < min(Mi, Ni)/2"""
    F = np.asarray(F)
    Nx, Ny = (shape, shape) if np.isscalar(shape) else shape
    Mx, My = F.shape[-2:]
    Fh = np.fft.fft2(F, axes=(-2, -1))
    nx, ny = min(Mx, Nx)//2, min(My, Ny)//2
    ix, iy = np.r_[0:nx, -nx+1:0], np.r_[0:ny, -ny+1:0]
    out = np.zeros(F.shape[:-2] + (Nx, Ny), complex)
    out[..., ix[:, None], iy[None, :]] = Fh[..., ix[:, None], iy[None, :]]
    return np.fft.ifft2(out*(Nx*Ny)/(Mx*My), axes=(-2, -1)).real
