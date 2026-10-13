''' Definitions of 1-, 2- and 3-D periodic grids '''
from ._backend import xnp, index_update
from ._backend import split_key, random_uniform

class Grid1D:
    def __init__(self, Lx, Nx, dt):
        xx, dx = xnp.linspace(0, Lx, Nx, endpoint=False, retstep=True)

        ki = xnp.fft.rfftfreq(Nx, 1/Nx).astype(int)
        kx = 2.0*xnp.pi*xnp.fft.rfftfreq(Nx, dx)
        k2 = kx**2
        kk = xnp.sqrt(k2)
        
        self.Lx = Lx
        self.xx = xx
        self.dx = dx
        self.dt = dt

        self.kx = kx
        self.k2 = k2
        self.kk = kk
        self.kr = ki

        self.N = Nx
        self.Nx = Nx
        self.shape = (Nx,)

        # Norm and de-aliasing
        self.norm = 1.0/(Nx**2)
        self.zero_mode = 0
        self.dealias_modes = (self.kr > Nx/3)

    @staticmethod
    def forward(ui):
        ''' Forward Fourier transform '''
        return xnp.fft.rfftn(ui)

    @staticmethod
    def inverse(ui):
        ''' Invserse Fourier transform '''
        return xnp.fft.irfftn(ui).real

    @staticmethod
    def deriv(ui, ki):
        ''' First derivative in ki direction '''
        return 1.0j*ki*ui

    def avg(self, ui):
        ''' Mean in Fourier space. rfftn is used, so in last dimension middle modes must be 
        doubled to account for negative frequencies. If n is even the last mode contains +fs/2 and -fs/2'''
        tmp = 1 if ui.shape[-1] % 2 == 0 else 2
        sum_ui = xnp.sum(ui[...,0]) + 2.0*xnp.sum(ui[...,1:-1]) + tmp* xnp.sum(ui[...,-1])
        return self.norm * sum_ui

    @staticmethod
    def inner(a, b):
        ''' Inner product '''
        prod = 0.0
        for ca, cb in zip(a, b):
            prod += (ca*cb.conjugate()).real
        return prod

    def energy(self, fields):
        u2  = self.inner(fields, fields)
        eng = 0.5*self.avg(u2)
        return eng

    def enstrophy(self, fields):
        u2  = self.inner(fields, fields)
        ens = 0.5*self.avg(self.k2*u2)
        return ens

    def palinstrophy(self, fields):
        u2  = self.inner(fields, fields)
        ens = 0.5*self.avg((self.k2**4)*u2)
        return ens

    def translate(self, fields, sx):
        # Forward transform
        f = [self.forward(ff) for ff in fields]
        # Translate
        f = [ff*xnp.exp(1.0j*self.kx*sx) for ff in f]
        # Inverse transform
        fields = [self.inverse(ff) for ff in f]
        return fields

    def deriv_fields(self, fields, ki):
        ''' Compute derivatives in Fourier space in i direction '''
        f = [self.forward(ff) for ff in fields]
        f = [self.deriv(ff, ki) for ff in f]
        return [self.inverse(ff) for ff in f]


class Grid2D(Grid1D):
    def __init__(self, Lx, Ly, Nx, Ny, dt):
        super().__init__(Lx, Nx, dt)

        xi, dx = xnp.linspace(0, Lx, Nx, endpoint=False, retstep=True)
        yi, dy = xnp.linspace(0, Ly, Ny, endpoint=False, retstep=True)
        xx, yy = xnp.meshgrid(xi, yi, indexing='ij')

        kx = 2.0*xnp.pi*xnp.fft.fftfreq(Nx, dx)
        ky = 2.0*xnp.pi*xnp.fft.rfftfreq(Ny, dy)
        kx, ky = xnp.meshgrid(kx, ky, indexing='ij')
        k2 = kx**2 + ky**2
        kk = xnp.sqrt(k2)

        ki = xnp.fft.fftfreq(Nx, 1/Nx)
        kj = xnp.fft.rfftfreq(Ny, 1/Ny)
        ki, kj = xnp.meshgrid(ki, kj, indexing='ij')
        kr = (ki/Nx)**2 + (kj/Ny)**2

        self.Lx = Lx
        self.Ly = Ly
        self.xx = xx
        self.yy = yy
        self.dy = dy

        self.kx = kx
        self.ky = ky
        self.ki = ki
        self.kj = kj
        self.k2 = k2
        self.kk = kk
        self.kr = kr

        self.Nx = Nx
        self.Ny = Ny
        self.N  = Nx*Ny
        self.shape = (Nx, Ny)

        # Norm, de-aliasing and solenoidal mode proyector
        self.norm = 1.0/(Nx**2*Ny**2)
        self.zero_mode = (0, 0)
        self.dealias_modes = (kr > 1/9)

        # Incompressiblity projector, careful not to divide by zero
        kk2 = index_update(k2, self.zero_mode, 1.0)
        self.kk2 = kk2
        self.pxx = 1.0 - self.kx**2/kk2
        self.pyy = 1.0 - self.ky**2/kk2
        self.pxy = - self.kx*self.ky/kk2

    def translate2D(self, fields, sx, sy):
        # Forward transform
        f = [self.forward(ff) for ff in fields]
        # Translate
        f = [ff *xnp.exp(1.0j*self.kx*sx) *xnp.exp(1.0j*self.ky*sy) for ff in f]
        # Inverse transform
        fields = [self.inverse(ff) for ff in f]
        return fields

    def inc_proj(self, fields):
        ''' Project onto solenoidal modes '''
        fu = fields[0]
        fv = fields[1]
        return self.pxx*fu + self.pxy*fv, self.pxy*fu + self.pyy*fv


def _cheb(N):
    ''' Chebyshev-Gauss-Lobatto nodes and differentiation matrix on [-1, 1].

    Returns nodes ``x`` (shape N+1, ordered x_0=1 ... x_N=-1) and the dense
    differentiation matrix ``D`` such that ``(D @ f)`` approximates ``df/dx`` at
    the nodes (Trefethen, *Spectral Methods in MATLAB*, cheb.m).
    '''
    if N == 0:
        return xnp.array([1.0]), xnp.zeros((1, 1))
    x = xnp.cos(xnp.pi * xnp.arange(N + 1) / N)
    c = xnp.hstack([2.0, xnp.ones(N - 1), 2.0]) * (-1.0) ** xnp.arange(N + 1)
    X = xnp.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = xnp.outer(c, 1.0 / c) / (dX + xnp.eye(N + 1))
    D = D - xnp.diag(xnp.sum(D, axis=1))
    return x, D


def _clencurt(N):
    ''' Clenshaw-Curtis quadrature weights for the N+1 CGL nodes on [-1, 1]
    (Trefethen, *Spectral Methods in MATLAB*, clencurt.m). '''
    if N == 0:
        return xnp.array([2.0])
    theta = xnp.pi * xnp.arange(N + 1) / N
    w = xnp.zeros(N + 1)
    v = xnp.ones(N - 1)
    ii = xnp.arange(1, N)
    if N % 2 == 0:
        w = index_update(w, 0, 1.0 / (N ** 2 - 1))
        w = index_update(w, N, 1.0 / (N ** 2 - 1))
        for k in range(1, N // 2):
            v = v - 2.0 * xnp.cos(2.0 * k * theta[ii]) / (4.0 * k ** 2 - 1)
        v = v - xnp.cos(N * theta[ii]) / (N ** 2 - 1)
    else:
        w = index_update(w, 0, 1.0 / N ** 2)
        w = index_update(w, N, 1.0 / N ** 2)
        for k in range(1, (N - 1) // 2 + 1):
            v = v - 2.0 * xnp.cos(2.0 * k * theta[ii]) / (4.0 * k ** 2 - 1)
    w = index_update(w, ii, 2.0 * v / N)
    return w


class Grid2D_wall(Grid1D):
    ''' 2D grid periodic in x, wall-bounded (Chebyshev) in z.

    Fourier (rfft) in x, Chebyshev-Gauss-Lobatto collocation in z. The z-grid
    *includes both walls* (z=0 at index 0, z=Lz at index Nz-1) so that Dirichlet
    boundary conditions sit at the first and last collocation rows -- this lines
    up with ``DynSys.remove_boundary`` which strips ``[:, 1:-1]``.

    Mixed-space fields produced by ``forward`` have shape ``(Nx//2+1, Nz)``:
    Fourier in x (axis 0), physical Chebyshev values in z (axis 1). z-derivatives
    are applied as dense matrix products (``dz``/``dz2``); x-derivatives use the
    spectral factor ``i*kx``.
    '''
    remove_boundary = True

    def __init__(self, Lx, Lz, Nx, Nz, dt):
        super().__init__(Lx, Nx, dt)
        N = Nz - 1

        # Chebyshev nodes/matrix on [-1, 1], mapped affinely to z in [0, Lz].
        # x = 1 - 2 z / Lz  =>  z=0 at x=1 (index 0), z=Lz at x=-1 (index N).
        xc, Dc = _cheb(N)
        zi = Lz * (1.0 - xc) / 2.0
        Dz = (-2.0 / Lz) * Dc           # d/dz = (dx/dz) d/dx,  dx/dz = -2/Lz
        Dz2 = Dz @ Dz
        wz = (Lz / 2.0) * _clencurt(N)  # integration weights in z over [0, Lz]

        zz = xnp.tile(zi, (Nx, 1))      # (Nx, Nz) physical z coordinate

        self.Lx = Lx
        self.Lz = Lz
        self.zi = zi
        self.zz = zz
        self.Dz = Dz
        self.Dz2 = Dz2
        self.wz = wz

        # x-direction spectral data over rfft modes
        Nkx = Nx // 2 + 1
        self.Nkx = Nkx
        self.kx = 2.0 * xnp.pi * xnp.fft.rfftfreq(Nx, self.dx)   # (Nkx,)
        self.kx_col = self.kx[:, None]                          # (Nkx, 1) broadcast
        self.k2x = self.kx ** 2
        ki = xnp.fft.rfftfreq(Nx, 1 / Nx)
        # 2/3 rule: keep k < Nx/3, so that p + q of two kept modes never aliases onto a kept mode
        # (with k <= Nx/3, mode Nx/3 was kept when 3 | Nx, e.g. Nx = 384, 768, and aliased onto itself)
        self.dealias_modes = (ki >= Nx / 3)[:, None]           # x-mode mask, (Nkx,1)

        self.Nx = Nx
        self.Nz = Nz
        self.N = Nx * Nz
        self.shape = (Nx, Nz)

    def forward(self, ui):
        ''' Forward Fourier transform in x only '''
        return xnp.fft.rfft(ui, axis=0)

    def inverse(self, ui):
        ''' Inverse Fourier transform in x only '''
        return xnp.fft.irfft(ui, n=self.Nx, axis=0).real

    def ddx(self, fhat):
        ''' x-derivative in mixed space (spectral factor i*kx).
        Named ``ddx`` to avoid clashing with ``self.dx`` (the x grid spacing). '''
        return 1.0j * self.kx_col * fhat

    def dz(self, fhat):
        ''' z-derivative in mixed space (Chebyshev matrix on axis 1) '''
        return fhat @ self.Dz.T

    def dz2(self, fhat):
        ''' second z-derivative in mixed space '''
        return fhat @ self.Dz2.T

    def translate(self, fields, sx):
        ''' Translate fields by sx in x only (z is wall-bounded, not periodic) '''
        f = [self.forward(ff) for ff in fields]
        f = [ff * xnp.exp(1.0j * self.kx_col * sx) for ff in f]
        return [self.inverse(ff) for ff in f]

    def deriv_fields(self, fields, *args):
        ''' x-derivative of physical fields (used by DynSys.deriv_U) '''
        f = [self.forward(ff) for ff in fields]
        f = [self.ddx(ff) for ff in f]
        return [self.inverse(ff) for ff in f]

    def integrate(self, field):
        ''' Domain integral of a real physical field, shape (Nx, Nz), using
        uniform x-spacing and Clenshaw-Curtis weights in z. '''
        return self.dx * xnp.sum(field * self.wz[None, :])

    def average(self, field):
        ''' Domain average of a real physical field '''
        return self.integrate(field) / (self.Lx * self.Lz)


class Grid2D_semi(Grid1D):
    ''' 2D grid periodic only in the horizontal direction. To be used with the SPECTER wrapper '''
    def __init__(self, Lx, Lz, Nx, Nz, dt):
        super().__init__(Lx, Nx, dt)
        zi, dz = xnp.linspace(0, Lz, Nz, endpoint=False, retstep=True)
        kx, zz = xnp.meshgrid(self.kx, zi, indexing='ij')

        self.Lx = Lx
        self.Lz = Lz
        self.zi = zi
        self.dz = dz
        self.zz = zz
        self.kx = kx

        self.Nx = Nx
        self.Nz = Nz
        self.N  = Nx*Nz
        self.shape = (Nx, Nz)

    @staticmethod
    def forward(ui):
        ''' Forward Fourier transform '''
        return xnp.fft.rfft(ui, axis = 0)

    @staticmethod
    def inverse(ui):
        ''' Invserse Fourier transform '''
        ui =  xnp.fft.irfft(ui, axis = 0).real
        return ui


class Grid3D(Grid1D):
    def __init__(self, Lx, Ly, Lz, Nx, Ny, Nz, dt):
        super().__init__(Lx, Nx, dt)

        xi, dx = xnp.linspace(0, Lx, Nx, endpoint=False, retstep=True)
        yi, dy = xnp.linspace(0, Ly, Ny, endpoint=False, retstep=True)
        zi, dz = xnp.linspace(0, Lz, Nz, endpoint=False, retstep=True)
        xx, yy, zz = xnp.meshgrid(xi, yi, zi, indexing='ij')

        kx = 2.0*xnp.pi*xnp.fft.fftfreq(Nx, dx)
        ky = 2.0*xnp.pi*xnp.fft.fftfreq(Ny, dy)
        kz = 2.0*xnp.pi*xnp.fft.rfftfreq(Nz, dz)
        kx, ky, kz = xnp.meshgrid(kx, ky, kz, indexing='ij')
        k2 = kx**2 + ky**2 + kz**2
        kk = xnp.sqrt(k2)

        ki = xnp.fft.fftfreq(Nx, 1/Nx)
        kj = xnp.fft.fftfreq(Ny, 1/Ny)
        kl = xnp.fft.rfftfreq(Nz, 1/Nz)
        ki, kj, kl = xnp.meshgrid(ki, kj, kl, indexing='ij')
        kr = (ki/Nx)**2 + (kj/Ny)**2 + (kl/Nz)**2

        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.xx = xx
        self.yy = yy
        self.zz = zz
        self.dy = dy
        self.dz = dz

        self.kx = kx
        self.ky = ky
        self.kz = kz

        self.ki = ki
        self.kj = kj
        self.kl = kl

        self.k2 = k2
        self.kk = kk
        self.kr = kr

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.N  = Nx*Ny*Nz
        self.shape = (Nx, Ny, Nz)

        # Norm, de-aliasing and solenoidal mode proyector
        self.norm = 1.0/(Nx**2*Ny**2*Nz**2)
        self.zero_mode = (0, 0, 0)
        self.dealias_modes = (self.kr > 1/9)

        kk2 = index_update(k2, self.zero_mode, 1.0)
        self.kk2 = kk2
        self.pxx = 1.0 - self.kx**2/kk2
        self.pyy = 1.0 - self.ky**2/kk2
        self.pzz = 1.0 - self.kz**2/kk2
        self.pxy = - self.kx*self.ky/kk2
        self.pxz = - self.kx*self.kz/kk2
        self.pyz = - self.ky*self.kz/kk2

    def translate3D(self, fields, sx=0., sy=0., sz=0.):
        # Forward transform
        f = [self.forward(ff) for ff in fields]
        # Translate
        f = [ff *xnp.exp(1.0j*self.kx*sx) *xnp.exp(1.0j*self.ky*sy) *xnp.exp(1.0j*self.kz*sz) for ff in f]
        # Inverse transform
        fields = [self.inverse(ff) for ff in f]
        return fields

    def inc_proj(self, fields):
        ''' Project onto solenoidal modes '''
        fu = fields[0]
        fv = fields[1]
        fw = fields[2]
        return (self.pxx*fu + self.pxy*fv + self.pxz*fw,
                self.pxy*fu + self.pyy*fv + self.pyz*fw,
                self.pxz*fu + self.pyz*fv + self.pzz*fw)

    def iso_random(self, kd, ku, key):
        '''Generate an isotropic random field'''
        key, subkey = split_key(key)

        phases = 2.0*xnp.pi*random_uniform(subkey,
                                           shape=(self.Nx, self.Ny, self.Nz//2 + 1))
        phases = xnp.exp(1.0j*phases) * 1/self.kk2
        field = phases * (self.k2 > kd**2) * (self.k2 < ku**2)

        conjg = field[:,:self.Ny//2,0].conjugate()
        field = index_update(field,
                             (slice(None), slice(-1, -self.Ny//2-1, -1), 0),
                              conjg[::-1,:])

        return field, key
