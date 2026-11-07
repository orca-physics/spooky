"""Gradient-based UPO initializer via periodicity-loss minimization.

Reference: Page, Norgaard, Brenner & Kerswell, PNAS 2024.
arXiv:2212.11886
"""
import numpy as np
from scipy.optimize import minimize as scipy_minimize

from .dyn_sys import DynSys
from .._backend import JAX_ENABLED


class UPOGradientDescent(DynSys):
    """Find UPO initial guesses by minimizing the periodicity loss with L-BFGS-B.

    Minimizes L(u0, T) = ||Φ(u0, T) - u0||² / ||u0||² using exact JAX
    gradients, generating high-quality seeds for Newton-Krylov-Hookstep.

    ``run`` keeps T fixed (to search over periods, loop over T). ``run_orbit``
    also optimises the period T and, for relative periodic orbits, the shift sx.

    Parameters
    ----------
    solver : PseudoSpectral instance
        The PDE solver to wrap.
    checkpoint_every : int, 'auto' or None
        Reverse-mode rematerialisation chunk passed to ``ad_evolve``
        (default 'auto': ~sqrt(number of steps)).
    project : bool or None
        Evaluate the loss on the solenoidal projection ``solver.inc_proj(u)``
        (default: whenever the solver has ``inc_proj``). The projection is
        linear, and idempotent and self-adjoint on the modes the solver keeps
        (not on the dealiased Nyquist column), so gradients, and hence the
        iterates, stay divergence-free there. Without it the optimiser
        accumulates divergent content at low k, which the solver only damps
        viscously.

    Example
    -------
    >>> gd = UPOGradientDescent(solver)
    >>> fields_guess, loss = gd.run(fields0, T=10.0, n_iter=500)
    >>> # Hand off to Newton-Krylov:
    >>> U_guess = ds.flatten(fields_guess)
    >>> # period and shift free (relative periodic orbit):
    >>> fields_guess, T, sx, loss, info = gd.run_orbit(fields0, T0=10.0, sx0=0.0)
    """

    def __init__(self, solver, checkpoint_every='auto', project=None):
        super().__init__(solver)
        if not JAX_ENABLED:
            raise RuntimeError(
                "UPOGradientDescent requires JAX backend (set NUMPY_BACKEND=jax)."
            )
        #: rematerialisation chunk for the reverse pass ('auto' ~ sqrt(Nt),
        #: an int, or None for the full tape)
        self.checkpoint_every = checkpoint_every
        self.project = hasattr(solver, 'inc_proj') if project is None else bool(project)

    def _proj(self, fields):
        """Solenoidal projection of physical fields if ``project`` is on."""
        return list(self.solver.inc_proj(fields)) if self.project else list(fields)

    def loss(self, fields, T, sx=None, nsteps=None):
        """Normalized periodicity residual ||translate(Φ(u,T), sx) - u||² / ||u||².

        ``sx=None``: no translation (periodic orbit). ``nsteps``: realise T with
        that many uniform steps (see ``ad_evolve``), needed for a traced T.
        """
        import jax.numpy as jnp
        fields = self._proj(fields)
        kw = {} if nsteps is None else {'nsteps': nsteps}
        out = self.solver.ad_evolve(fields, T, checkpoint_every=self.checkpoint_every, **kw)
        if sx is not None:
            out = self.grid.translate(out, sx)
        res = [o - f for o, f in zip(out, fields)]
        num = sum(jnp.sum(r ** 2) for r in res)
        den = sum(jnp.sum(f ** 2) for f in fields)
        return num / den

    def run(self, fields0, T, n_iter=500, tol=1e-8, verbose=True):
        """Minimize periodicity loss from initial condition fields0 at period T.

        Parameters
        ----------
        fields0 : list of array_like
            Initial condition in physical space (any numeric arrays).
        T : float
            Period (fixed; must be an integer multiple of solver.grid.dt).
        n_iter : int
            Maximum L-BFGS-B iterations. Default 500.
        tol : float
            Gradient norm tolerance for convergence. Default 1e-8.
        verbose : bool
            Print initial/final loss and convergence message. Default True.

        Returns
        -------
        fields_opt : list of np.ndarray
            Optimized initial condition in physical space.
        loss_final : float
            Final periodicity loss value.
        """
        import jax
        import jax.numpy as jnp

        shapes = [np.array(f).shape for f in fields0]
        sizes  = [int(np.prod(s)) for s in shapes]

        def pack(fields):
            return np.concatenate(
                [np.array(f).ravel() for f in fields]
            ).astype(np.float64)

        def unpack(x):
            fields, idx = [], 0
            for s, sz in zip(shapes, sizes):
                fields.append(jnp.array(x[idx:idx + sz].reshape(s)))
                idx += sz
            return fields

        val_and_grad = jax.jit(jax.value_and_grad(lambda f: self.loss(f, T)))

        def fg(x):
            fields = unpack(x)
            val, g_fields = val_and_grad(fields)
            g = np.concatenate(
                [np.array(gi).ravel() for gi in g_fields]
            ).astype(np.float64)
            return float(val), g

        x0 = pack(fields0)

        if verbose:
            print(f"UPO gradient descent  T={T}  n_iter={n_iter}")
            print(f"  initial loss = {fg(x0)[0]:.4e}")

        result = scipy_minimize(
            fg, x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': n_iter, 'gtol': tol},
        )

        fields_opt = [np.array(f) for f in self._proj(unpack(result.x))]

        if verbose:
            status = 'converged' if result.success else result.message
            print(f"  final loss   = {result.fun:.4e}  "
                  f"({result.nit} iters, {status})")

        return fields_opt, float(result.fun)

    def run_orbit(self, fields0, T0, sx0=None, free_T=True, free_sx=None,
                  n_iter=500, tol=1e-10, T_bounds=(0.75, 1.25), verbose=True):
        """Minimize the periodicity loss over (u0, T, sx) with L-BFGS-B.

        The period is realised with ``nsteps = round(T0/dt)`` uniform steps of
        ``T/nsteps``, so T is a smooth optimization variable (the step size
        changes, not the step count) and is kept within ``T_bounds`` (factors
        of T0). ``sx0=None`` searches for periodic orbits; otherwise the loss
        includes the translation and sx is free unless ``free_sx=False``.
        T and sx are rescaled by |dU/dt| and |dU/dx| at u0, so that a unit
        change of either moves the state about as much as a unit change of u0.
        With ``free_T=False`` the evolution still uses the uniform steps
        T0/nsteps, which differs from ``run`` (steps of dt plus a fractional
        last step) at O(dt^2).
        Zero loss is also reached by equilibria (any T, sx=0), travelling waves
        (sx = c T) and the laminar state; check results before calling them
        periodic orbits.

        Returns
        -------
        fields_opt : list of np.ndarray
        T, sx : float (sx is None for periodic orbits)
        loss_final : float
        info : dict
            ``nit``, ``nfev``, ``message``, ``nsteps`` and ``history`` (one
            (loss, T, sx) tuple per iteration, starting with the initial guess).
        """
        import jax
        import jax.numpy as jnp

        use_sx = sx0 is not None
        if free_sx is None:
            free_sx = use_sx
        free_sx = free_sx and use_sx
        nsteps = max(1, int(round(T0 / self.grid.dt)))

        shapes = [np.array(f).shape for f in fields0]
        sizes = [int(np.prod(s)) for s in shapes]
        nu = sum(sizes)
        fields0 = [np.array(f) for f in self._proj(fields0)]
        U0 = np.concatenate([f.ravel() for f in fields0]).astype(np.float64)

        # variable scaling: |dU/dt| (one-step forward difference) and |dU/dx| at u0
        sT = float(np.linalg.norm((self.evolve(U0, self.grid.dt) - U0) / self.grid.dt)) if free_T else 1.0
        ss = float(np.linalg.norm(self.deriv_U(U0, self.grid.kx))) if free_sx else 1.0

        def unpack(x):
            fields, idx = [], 0
            for s, sz in zip(shapes, sizes):
                fields.append(jnp.array(x[idx:idx + sz].reshape(s)))
                idx += sz
            T = x[idx] / sT if free_T else T0
            idx += int(free_T)
            sx = (x[idx] / ss if free_sx else sx0) if use_sx else 0.0
            return fields, float(T), float(sx)

        loss_fn = lambda f, T, sx: self.loss(f, T, sx=sx if use_sx else None, nsteps=nsteps)
        val_and_grad = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2)))

        last = {}
        def fg(x):
            fields, T, sx = unpack(x)
            val, (gf, gT, gs) = val_and_grad(fields, T, sx)
            g = [np.array(gi).ravel() for gi in gf]
            if free_T:
                g.append([float(gT) / sT])
            if free_sx:
                g.append([float(gs) / ss])
            last.update(x=np.array(x), val=float(val), T=T, sx=sx)
            return float(val), np.concatenate(g).astype(np.float64)

        extra, bounds = [], [(None, None)] * nu
        if free_T:
            extra.append(T0 * sT)
            bounds.append((T_bounds[0] * T0 * sT, T_bounds[1] * T0 * sT))
        if free_sx:
            extra.append(sx0 * ss)
            bounds.append((None, None))
        x0 = np.concatenate([U0, np.array(extra, dtype=np.float64)])

        history = []
        def record(xk):
            if 'x' not in last or not np.array_equal(xk, last['x']):
                fg(xk)
            history.append((last['val'], last['T'], last['sx'] if use_sx else None))

        record(x0)
        if verbose:
            print(f"UPO gradient descent  T0={T0}  sx0={sx0}  nsteps={nsteps}  n_iter={n_iter}")
            print(f"  initial loss = {history[0][0]:.4e}")

        result = scipy_minimize(fg, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                                callback=record,
                                options={'maxiter': n_iter, 'gtol': tol, 'ftol': 1e-16})

        fields_opt, T, sx = unpack(result.x)
        fields_opt = [np.array(f) for f in self._proj(fields_opt)]
        if verbose:
            status = 'converged' if result.success else result.message
            print(f"  final loss   = {result.fun:.4e}  T={T:.6f}  sx={sx:.6e}  "
                  f"({result.nit} iters, {status})")
        info = {'nit': int(result.nit), 'nfev': int(result.nfev),
                'message': str(result.message), 'nsteps': nsteps, 'history': history}
        return fields_opt, T, (sx if use_sx else None), float(result.fun), info
