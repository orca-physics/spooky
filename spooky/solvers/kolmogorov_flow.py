''' 2D Kolmogorov flow solver '''

import numpy as np
from .._backend import xnp, index_update, apply_jit
import os

from .pseudospectral import PseudoSpectral
from .. import pseudo as ps


class KolmogorovFlow(PseudoSpectral):
    '''
    Kolmogorov flow: 2D Navier-Stokes with fx = sin(2*pi*kf*y/Ly) forcing.

    nu = 1/Re

    See Eq. (6.143) in Pope's Turbulent flows for details on the Fourier
    decomposition of the NS equations and the pressure proyector.

    Parameters
    ---------
    grid: Grid2D instance
        Defines spatiotemporal grid parameters.
    kf: int, optional
        Forcing wavenumber. Default is 4.
    f0: float, optional
        Forcing amplitude. Default is 1.0.
    nu: float, optional
        Viscosity. Default is 1.0.
    ftypes: list of strings, optional
        Field names used when writing output files. Defaults are 'uu' and 'vv'.
    rkord: int, optional
        Order of the RK integration. Default is 2.
    ext: int, optional
        Length of zero padding in output file name. Default is 4.
    '''

    num_fields = 2
    dim_fields = 2

    def __init__(self,
                 grid: ps.Grid2D,
                 kf=4,
                 f0=1.,
                 nu = 1.0,
                 ftypes=['uu', 'vv'],
                 rkord=2,
                 ext=4):
        super().__init__(grid, rkord)
        self.ftypes = ftypes
        self.solver = 'KolmogorovFlow'
        self.ext = ext

        # viscosity
        self.nu = nu

        # Forcing
        self.kf = kf
        self.f0 = f0
        self.fx = f0 *xnp.sin(2*xnp.pi*kf*self.grid.yy/self.grid.Ly)
        self.fx = self.grid.forward(self.fx)
        self.fy = xnp.zeros_like(self.fx, dtype=complex)
        self.fx, self.fy = self.grid.inc_proj([self.fx, self.fy])

    def rhs(self, fields):
        ''' Time derivative of the Fourier fields [fu, fv], the pressure
        included (projected non-linear term) '''
        # Unpack
        fu, fv = fields

        # Non-linear term
        uu = self.grid.inverse(fu)
        vv = self.grid.inverse(fv)
        
        ux = self.grid.inverse(self.grid.deriv(fu, self.grid.kx))
        uy = self.grid.inverse(self.grid.deriv(fu, self.grid.ky))

        vx = self.grid.inverse(self.grid.deriv(fv, self.grid.kx))
        vy = self.grid.inverse(self.grid.deriv(fv, self.grid.ky))

        gx = self.grid.forward(uu*ux + vv*uy)
        gy = self.grid.forward(uu*vx + vv*vy)
        gx, gy = self.grid.inc_proj([gx, gy])

        # Equations
        du = (- gx
              - self.nu * self.grid.k2 * fu
              + self.fx)

        dv = (- gy
              - self.nu * self.grid.k2 * fv
              + self.fy)

        return [du, dv]

    @apply_jit
    def rkstep(self, fields, prev, oo, dt):
        # Unpack
        fup, fvp = prev
        du, dv = self.rhs(fields)

        fu = fup + (dt/oo) * du
        fv = fvp + (dt/oo) * dv

        # de-aliasing
        fu = index_update(fu, self.grid.zero_mode, 0.0)
        fu = index_update(fu, self.grid.dealias_modes, 0.0)
        fv = index_update(fv, self.grid.zero_mode, 0.0)
        fv = index_update(fv, self.grid.dealias_modes, 0.0)

        return [fu, fv]

    def injection(self, fields, forcing):
        return self.grid.avg(self.grid.inner(fields, forcing))

    def outs(self, fields, step, opath):
        uu = self.grid.inverse(fields[0])
        vv = self.grid.inverse(fields[1])
        np.save(os.path.join(opath,f'uu.{step:0{self.ext}}'), uu)
        np.save(os.path.join(opath,f'vv.{step:0{self.ext}}'), vv)

    def balance(self, fields, step, bpath):
        eng = self.grid.energy(fields)
        ens = self.grid.enstrophy(fields)
        dis = - 2 * self.nu * ens
        inj = self.injection(fields, [self.fx, self.fy])

        bal = [f'{self.grid.dt*step:.4e}', f'{eng:.6e}', f'{dis:.6e}', f'{inj:.6e}']
        with open(os.path.join(bpath, 'balance.dat'), 'a') as output:
            print(*bal, file=output)

    def load_fields(self, path, step, ext = None):
        if not ext:
            ext = self.ext
        uu = np.load(os.path.join(path, f'uu.{step:0{ext}}.npy'))
        vv = np.load(os.path.join(path, f'vv.{step:0{ext}}.npy'))
        return [uu, vv]

    def oz(self, fields):
        ''' Computes vorticity field '''
        fu, fv = [self.grid.forward(ff) for ff in fields]
        uy = self.grid.inverse(self.grid.deriv(fu, self.grid.ky))
        vx = self.grid.inverse(self.grid.deriv(fv, self.grid.kx))
        return uy - vx

    def inc_proj(self, fields):
        fields = [self.grid.forward(ff) for ff in fields]
        inc_f = self.grid.inc_proj(fields)         
        fields = [self.grid.inverse(ff) for ff in inc_f]
        return fields
