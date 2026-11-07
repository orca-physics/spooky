''' 1D Kuramoto Sivashinsky equation '''

import numpy as np
from .._backend import xnp, index_update, apply_jit

from .pseudospectral import PseudoSpectral
from .. import pseudo as ps

class KuramotoSivashinsky(PseudoSpectral):
    ''' 1D Kuramoto Sivashinsky equation with periodic boundary conditions

    Parameters
    ---------
    grid: Grid1D instance
        Defines spatiotemporal grid parameters.
    nu: float, optional
        Viscosity. Default is 1.0.
    rkord: int, optional
        Order of the RK integration. Default is 2.
    ext: int, optional
        Length of zero padding in output file name. Default is 4.
    '''

    num_fields = 1
    dim_fields = 1
    def __init__(self,
                 grid: ps.Grid1D,
                 nu=1.0,
                 rkord=2,
                 ext=4):
        super().__init__(grid, rkord=rkord)
        self.nu = nu
        self.ext = ext

    def rhs(self, fields):
        ''' Time derivative of the Fourier field [fu] '''
        # Unpack
        fu  = fields[0]

        # Non-linear term
        uu  = self.grid.inverse(fu)
        fu2 = self.grid.forward(uu**2)

        du = (- (0.5*1.0j*self.grid.kx*fu2)
              + ((self.grid.k2)*fu)
              - (self.nu*(self.grid.k2**2)*fu))

        return [du]

    @apply_jit
    def rkstep(self, fields, prev, oo, dt):
        # Unpack
        fup = prev[0]
        du, = self.rhs(fields)

        fu = fup + (dt/oo) * du

        # de-aliasing
        fu = index_update(fu, self.grid.zero_mode, 0.0)
        fu = index_update(fu, self.grid.dealias_modes, 0.0)

        return [fu]

    def outs(self, fields, step, opath):
        uu = self.grid.inverse(fields[0])
        np.save(f'{opath}uu.{step:0{self.ext}}', uu)

    def balance(self, fields, step, bpath):
        eng = self.grid.energy(fields)
        bal = [f'{self.grid.dt*step:.4e}', f'{eng:.6e}']
        with open(f'{bpath}balance.dat', 'a') as output:
            print(*bal, file=output)
