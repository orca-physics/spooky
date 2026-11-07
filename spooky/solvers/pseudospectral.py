import abc

from .solver import Solver
from .._backend import xnp, copy_arr, JAX_ENABLED, lax_scan
from .. import pseudo as ps

class PseudoSpectral(Solver, abc.ABC):
    ''' Abstract Pseudospectral solver class

    All solvers must implement the following methods:
        - rkstep: Runge-Kutta step
        - balance: Energy balance
        - spectra: Energy spectra
        - outs: Outputs
    '''
    def __init__(self,
                 grid: ps.Grid1D | ps.Grid2D | ps.Grid2D_semi | ps.Grid3D,
                 rkord: int):
        super().__init__(grid)
        self.rkord = rkord

    def evolve(self,
               fields,
               T,
               bstep=None,
               ostep=None,
               sstep=None,
               bpath='',
               opath='',
               spath='',
               write_outputs=True,
               realfields=True):
        ''' Evolves velocity fields to time T '''

        # Forward transform
        if realfields:
            fields = [self.grid.forward(ff) for ff in fields]

        Nt = int(T/self.grid.dt)
        for step in range(Nt+1):
            # Store previous time step
            prev = copy_arr(fields)

            # Write outputs
            if write_outputs:
                self.write_outputs(fields, step, bstep, ostep, sstep, bpath, opath, spath)

            dt = self.grid.dt
            #Make exact last step
            if step == Nt:
                dt = T - Nt*self.grid.dt
                if xnp.isclose(dt, 0., atol=1e-8):
                    break

            # Time integration
            for oo in range(self.rkord, 0, -1):
                fields = self.rkstep(fields, prev, oo, dt)

        # Write final outputs
        if write_outputs:
            self.write_outputs(fields, Nt, bstep, ostep, sstep, bpath, opath, spath, final=True)

        # Inverse transform
        if realfields:
            fields = [self.grid.inverse(ff) for ff in fields]

        return fields

    def ad_evolve(self, fields, T, checkpoint_every=None, nsteps=None):
        """JAX-traceable time evolution using lax.scan. No file I/O.

        Mirrors evolve() but is differentiable end-to-end with jax.grad/jvp/vjp.
        Takes the same exact fractional final step as evolve(), so both compute
        the same map Phi(u, T) for any T, not only multiples of dt.
        ``checkpoint_every`` (int or 'auto' = ~sqrt(Nt)) wraps chunks of steps
        in jax.checkpoint so reverse mode stores ~sqrt(Nt) carries instead of
        the full per-step tape, at ~2x the forward cost.
        ``nsteps`` (int) instead realises T with ``nsteps`` uniform steps of
        ``T/nsteps`` and no fractional step, so T may be a traced value and
        Phi is differentiable with respect to it (used to optimise the period).
        Requires JAX backend (NUMPY_BACKEND=jax).
        """
        if not JAX_ENABLED:
            raise NotImplementedError(
                "ad_evolve requires JAX backend (set NUMPY_BACKEND=jax)."
            )

        fields = [self.grid.forward(ff) for ff in fields]
        if nsteps is None:
            dt = self.grid.dt
            Nt = int(T / dt)
            dt_last = T - Nt * dt
        else:
            Nt = int(nsteps)
            dt = T / Nt
            dt_last = 0.0
        rkord = self.rkord

        def rk(flds, dt_step):
            prev = flds  # immutable JAX arrays; no copy needed
            for oo in range(rkord, 0, -1):
                flds = self.rkstep(flds, prev, oo, dt_step)
            return flds

        step = lambda carry, _: (rk(carry, dt), None)
        if checkpoint_every == 'auto':
            checkpoint_every = max(1, int(round(Nt ** 0.5)))
        if checkpoint_every and Nt > checkpoint_every:
            import jax
            n_in = int(checkpoint_every)
            n_out, rem = divmod(Nt, n_in)

            def chunk(carry, _):
                carry, _ = lax_scan(step, carry, None, length=n_in)
                return carry, None

            fields, _ = lax_scan(jax.checkpoint(chunk, prevent_cse=False),
                                 fields, None, length=n_out)
            if rem:
                fields, _ = lax_scan(step, fields, None, length=rem)
        else:
            fields, _ = lax_scan(step, fields, None, length=Nt)
        # dt_last is a plain Python float; keep the check in Python (not xnp)
        # so the branch resolves at trace time even under jit
        if abs(dt_last) > 1e-8:
            fields = rk(fields, dt_last)
        return [self.grid.inverse(ff) for ff in fields]

    @abc.abstractmethod
    def rkstep(self, fields, prev, oo, dt):
        return []
