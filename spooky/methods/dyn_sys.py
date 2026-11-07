from .krylov import GMRES, backsub, arnoldi_eig
import os
import numpy as np
import functools
from collections import OrderedDict

from .. import pseudo as ps
from .._backend import JAX_ENABLED

class DynSys():
    """General class for Dynamical Systems-based methods.

    Provides decorated version of several functions from grid and solver that can work with a flattened field.

    Methods implemented:
        - Floquet multipliers
        - Lyapunov exponents

    Parameters
    ----------
    solver: Solver instance
        Solver to be used.
    """
    def __init__(self, solver):
        self.solver = solver
        self.grid = solver.grid

        # Wall-bounded grids carry interior-only state for the UPO/Floquet
        # machinery: the Dirichlet boundary rows are stripped on flatten and
        # re-padded with zeros on unflatten. Grids advertise this via a
        # ``remove_boundary`` attribute (e.g. Grid2D_semi).
        self.remove_boundary = getattr(self.grid, 'remove_boundary', False)
        if isinstance(self.grid, ps.Grid2D_semi):
            self.remove_boundary = True

    def flatten(self, fields):
        '''Flattens fields'''
        if not self.remove_boundary:
            return np.concatenate([f.flatten() for f in fields])
        else:
            return np.concatenate([f[:,1:-1].flatten() for f in fields])

    def unflatten(self, U):
        '''Unflatten fields'''
        ll = len(U)//self.solver.num_fields
        fields = [U[i*ll:(i+1)*ll] for i in range(self.solver.num_fields)]
        if not self.remove_boundary:
            fields = [f.reshape(self.grid.shape) for f in fields]
            return fields
        else:
            trim_shape = (self.grid.Nx, self.grid.Nz-2)
            trim_fields = [f.reshape(trim_shape) for f in fields]
            fields = [np.pad(f, pad_width = ((0,0),(1,1))) for f in trim_fields] #pads second dimension with zeros before and after
            return fields

    def flatten_dec(func):
        """Decorator that allows to work with flattened fields U instead of fields"""
        @functools.wraps(func)
        def wrapper(self, U, *args, **kwargs):
            fields = self.unflatten(U)
            result = func(self, fields, *args, **kwargs)
            return self.flatten(result)
        return wrapper

    @flatten_dec
    def evolve(self, U, *args, **kwargs):
        '''Evolves fields U in time T'''
        return self.solver.evolve(U, *args, **kwargs)

    @flatten_dec
    def translate(self, U, *args):
        '''Translates fields by sx in x direction'''
        return self.grid.translate(U, *args)

    @flatten_dec
    def deriv_U(self, U, *args):
        '''Derivatives in x direction of fields'''
        return self.grid.deriv_fields(U, *args)

    @flatten_dec
    def sol_project(self, U):
        '''Solenoidal projection of fields'''
        return self.solver.inc_proj(U)

    def write_fields(self, U, idx, path):
        ''' Writes fields to binary file'''
        fields = self.unflatten(U)
        return self.solver.write_fields(fields, idx, path)

    # --- AD methods (require JAX backend) ------------------------------------

    def ad_evolve(self, fields, T):
        """AD-compatible evolution on field lists. Requires JAX backend."""
        return self.solver.ad_evolve(fields, T)

    def jvp_evolve(self, fields, tangent_fields, T):
        """Forward-mode AD: exact tangent map J·v on field lists.

        Parameters
        ----------
        fields : list of arrays
            Base point (physical space).
        tangent_fields : list of arrays
            Tangent vector v (same structure as fields).
        T : float
            Integration time.

        Returns
        -------
        primals : list of arrays
            Evolved fields Φ(fields, T).
        tangents : list of arrays
            Tangent output J·v where J = dΦ/d(fields).

        The JVP is jitted once per ``(T, solver.ad_cache_key())`` and cached
        (``jvp_cache_size`` entries), so repeated calls at the same T -- every
        GMRES or Arnoldi iteration -- reuse one compilation instead of
        retracing the whole time loop.
        """
        return self._compiled_jvp(T)(fields, tangent_fields)

    #: number of compiled JVPs kept by jvp_evolve
    jvp_cache_size = 4

    def _compiled_jvp(self, T, batched=False):
        ''' jit((fields, tangents) -> jvp of solver.ad_evolve over T), cached.
        ``batched``: the tangents carry a leading batch axis and the JVP is
        vmapped over it with the base trajectory left unbatched, so it is
        computed once and returned without a batch axis.  The key includes
        ``solver.ad_cache_key()`` (the solver parameters its traced evolution
        reads), so a parameter change gives a new entry instead of a stale
        compiled map. '''
        from .._backend import jit, jvp, vmap
        key = (bool(batched), float(T)) + tuple(self.solver.ad_cache_key())
        cache = self.__dict__.setdefault('_jvp_cache', OrderedDict())
        if key not in cache:
            solver, T_ = self.solver, key[1]
            F = lambda u: solver.ad_evolve(u, T_)
            if batched:
                fn = lambda f, dfs: vmap(lambda df: jvp(F, (f,), (df,)), out_axes=(None, 0))(dfs)
            else:
                fn = lambda f, df: jvp(F, (f,), (df,))
            cache[key] = jit(fn)
            while len(cache) > self.jvp_cache_size:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    def vjp_evolve(self, fields, cotangent_fields, T):
        """Reverse-mode AD: adjoint action J^T·v on field lists.

        Parameters
        ----------
        fields : list of arrays
            Base point (physical space).
        cotangent_fields : list of arrays
            Cotangent vector v (same structure as output fields).
        T : float
            Integration time.

        Returns
        -------
        primals : list of arrays
            Evolved fields Φ(fields, T).
        cotangents : list of arrays
            Adjoint output J^T·v where J = dΦ/d(fields).
        """
        from .._backend import vjp
        primals, vjp_fn = vjp(
            lambda f: self.solver.ad_evolve(f, T),
            fields
        )
        return primals, vjp_fn(cotangent_fields)

    def tangent_map(self, U, dU, T, sx=None, UT=None, ep0=1e-7):
        """Action of the tangent map J = d(translate_sx ∘ Φ_T)/dU on dU.

        Flat-vector interface shared by Floquet, Lyapunov and Newton-Krylov.
        JAX backend: exact JVP through ad_evolve.  NumPy backend: first-order
        finite difference with step ep0*|U|/|dU|, which requires ``UT``.

        Parameters
        ----------
        U, dU : np.ndarray
            Flat base point and perturbation.
        T : float
            Integration time.
        sx : float or None
            Optional x-translation applied after the flow map.
        UT : np.ndarray or None
            Base-point image translate(Φ_T(U), sx) — required for the FD
            (NumPy) path, ignored under JAX.
        ep0 : float
            FD perturbation scale factor.
        """
        if JAX_ENABLED:
            import jax.numpy as jnp
            f_jax  = [jnp.array(f) for f in self.unflatten(U)]
            dU_jax = [jnp.array(d) for d in self.unflatten(dU)]
            _, tangent_list = self.jvp_evolve(f_jax, dU_jax, T)
            J_dU = np.array(self.flatten(tangent_list))
            if sx is not None:
                J_dU = self.translate(J_dU, sx)
        else:
            if UT is None:
                raise ValueError(
                    "tangent_map FD path requires UT = translate(Phi_T(U), sx)")
            epsilon = ep0 * np.linalg.norm(U) / np.linalg.norm(dU)
            J_dU = self.evolve(U + epsilon * dU, T)
            if sx is not None:
                J_dU = self.translate(J_dU, sx)
            J_dU = (J_dU - UT) / epsilon
        return J_dU

    def tangent_map_batch(self, U, dUs, T, sx=None, ep0=1e-7):
        """Batched tangent map: (Uevol, V) with V[:, i] = J · dUs[:, i].

        JAX backend: one jitted forward pass of the JVP vmapped over the n
        tangents (base trajectory computed once, tangents propagated together
        as batched products), compiled once per (T, solver.ad_cache_key()) and
        cached with ``jvp_evolve``.  Memory: O(n x state) -- no linearization
        tape.  NumPy backend: FD loop over the columns.

        Returns
        -------
        Uevol : np.ndarray
            Untranslated image Φ_T(U) (the next base point).
        V : np.ndarray, shape (len(U), n)
            Tangent images (translated by sx if given).
        """
        n = dUs.shape[1]
        V = np.zeros((len(U), n))
        if JAX_ENABLED:
            import jax.numpy as jnp
            f_jax = [jnp.array(f) for f in self.unflatten(U)]
            cols = [self.unflatten(dUs[:, i]) for i in range(n)]
            batched = [jnp.stack([np.asarray(c[j]) for c in cols])
                       for j in range(self.solver.num_fields)]
            primals, tans = self._compiled_jvp(T, batched=True)(f_jax, batched)
            Uevol = np.array(self.flatten(primals))
            for i in range(n):
                V[:, i] = self.flatten([np.array(t[i]) for t in tans])
                if sx is not None:
                    V[:, i] = self.translate(V[:, i], sx)
        else:
            Uevol = self.evolve(U, T)
            UT = self.translate(Uevol, sx) if sx is not None else Uevol
            for i in range(n):
                V[:, i] = self.tangent_map(U, dUs[:, i], T, sx=sx, UT=UT,
                                           ep0=ep0)
        return Uevol, V

    # -------------------------------------------------------------------------

    def phase_shifted_b(self, fields):
        """Generate b vector by modifying phases in Fourier space."""
        def apply_phase_shift(U):
            """Applies a random phase shift to a single field U and projects to solenoidal modes."""
            U_hat = self.grid.forward(U)
            random_phases = np.exp(1j * np.random.uniform(0, 2*np.pi, U_hat.shape))
            U_hat_shifted = np.abs(U_hat) * random_phases  # Preserve magnitudes
            return U_hat_shifted

        # Separate into u and v components
        u, v = fields
        b_vector = []

        u_hat_shifted = apply_phase_shift(u)
        v_hat_shifted = apply_phase_shift(v)

        # Project in Fourier space
        u_proj_hat, v_proj_hat = self.grid.inc_proj((u_hat_shifted, v_hat_shifted))

        # Inverse FFT to physical space
        u_proj = self.grid.inverse(u_proj_hat)
        v_proj = self.grid.inverse(v_proj_hat)
        fields_proj = [u_proj, v_proj]

        b_vector = self.flatten([f.flatten() for f in fields_proj])
        return b_vector

    def floquet_multipliers(self, fields, T, n, tol, ep0=1e-7, sx=None, b='U', test=False):
        ''' Calculates Floquet multipliers using the Arnoldi algorithm.
        Method follows Viswanath (2007), same as Newton method.

        Paramters
        ---------
        fields: list of arrays
            List of fields to be used.
        T: float
            Time to evolve in each arnoldi iteration.
        n: int
            Number of multiplier to calculate
        tol: float
            Tolerance of the Arnoldi iteration
        ep0: float, optional
            Perturbation factor. Default is 1e-7.
        sx: float or None, optional
            Translation in x direction. Default is None.
        b: str or np.array, optional
            Initial guess for Arnoldi. Default is 'U'. Could also take in
            'random' or a np.array of the same size as U
        test: bool, optional
            If True, also returns Aev_r and Aev_i to check if returned eigenvectors satisfy Av=lambda v
            
        Returns
        -------
        eigval_H: np.array of complex
            Floquet multipliers
        eigvec_H: np.arrays of complex
            Floquet vectors
        Q: np.arrays
            Arnoldi basis
        '''

        U = self.flatten(fields)
        UT = self.evolve(U, T)

        # Translate UT by sx
        if sx is not None:
            UT = self.translate(UT, sx)

        def apply_J(dU):
            ''' Applies J (jacobian of poincare map) matrix to vector dU  '''
            return self.tangent_map(U, dU, T, sx=sx, UT=UT, ep0=ep0)

        if isinstance(b, str):
            if b == 'U':
                b = U
            elif b == 'random':
                b = np.random.randn(len(U))
            elif b == 'phases':
                b = self.phase_shifted_b(fields)
        elif isinstance(b, np.ndarray):
            pass
        else:
            raise ValueError('b must be one of the given options')

        eigval_H, eigvec_H, Q = arnoldi_eig(apply_J, b, n, tol)

        if test:
            # Checks if returned eigenvectors satisfy Av=lambda v
            Aev_r = np.zeros((len(U), n))
            Aev_i = np.zeros((len(U), n))
            for i in range(n):
                ev = Q @ eigvec_H[:,i]
                Aev_r[:,i] = apply_J(ev.real)
                Aev_i[:,i] = apply_J(ev.imag)

            return eigval_H, eigvec_H, Q, Aev_r, Aev_i

        return eigval_H, eigvec_H, Q

    def lyapunov_exponents(
        self, fields, T, n, nsteps, ep0=1e-7, sx=None, b='random',
        return_hist=False,
        resume=False,
        checkpoint_file=None,
        checkpoint_every=None
        ):
        ''' Computes Lyapunov exponents and Kaplan–Yorke dimension via QR iteration.

        This method implements the Benettin algorithm for estimating finite-time
        Lyapunov exponents by repeatedly evolving a set of orthonormal perturbations
        over intervals of duration T and reorthonormalizing them using QR decomposition.

        Parameters
        ----------
        fields : list of fields
            Initial fields defining the state U.
        T : float
            Integration time between QR reorthonormalizations.
        n : int
            Number of Lyapunov exponents to compute.
        nsteps : int
            Number of reorthonormalization intervals (total time = nsteps * T)
        ep0 : float, optional
            Perturbation scaling for the finite-difference tangent map.
            Default is 1e-7.
        sx : float, optional
            Translation in x direction (for translationally invariant systems).
        b : str or np.ndarray, optional
            Initial perturbation seed ('random', 'U', 'phases', or user-defined array).
            Ignored if restart=True.
        return_hist : bool, optional
            If True, also returns the history of Lyapunov exponents at each step.
        resume : bool, optional
            If True, attempts to resume from checkpoint_file if it exists. Default is False.
        checkpoint_file : str or None, optional
            Path to checkpoint file for saving and resuming. If None, no checkpointing is done.
        checkpoint_every : int or None, optional
            If not None, saves a checkpoint every this many steps. Ignored if checkpoint_file is None.

        Returns
        -------
        lyap_exponents : np.ndarray, shape (n,)
            Sorted Lyapunov exponents in descending order.
        D_KY : float
            Kaplan–Yorke dimension computed from the cumulative sum of exponents.
        le_hist : np.ndarray, shape (nsteps, n), optional
            History of Lyapunov exponents at each reorthonormalization step.
        '''

        # --------------------------------------------------
        # --- INITIAL STATE OR RESTART ---------------------
        # --------------------------------------------------

        start_step = 0
        times = np.arange(1, nsteps + 1) * T
        if resume:
            # Restarting from checkpoint
            if checkpoint_file is None:
                raise ValueError("resume=True requires checkpoint_file.")

            if os.path.exists(checkpoint_file):
                ckpt = np.load(checkpoint_file, allow_pickle=True)

                start_step = int(ckpt["step"])
                U = ckpt["U"]
                Q = ckpt["Q"]
                le_sum = ckpt["le_sum"]

                if "le_hist" in ckpt.files and return_hist:
                    le_hist = ckpt["le_hist"]
                    if le_hist.shape[1] != n:
                        raise ValueError(f"Checkpoint le_hist n_lyap {le_hist.shape[1]} does not match expected {n}.")    
                    if le_hist.shape[0] < nsteps:
                        # Add padding if checkpoint has fewer steps than requested
                        le_hist = np.pad(le_hist, ((0, nsteps - le_hist.shape[0]), (0, 0)), mode='constant')
                    elif le_hist.shape[0] > nsteps:
                        le_hist = le_hist[:nsteps, :]

            else:
                raise FileNotFoundError(
                    f"resume=True but checkpoint file was not found: {checkpoint_file}"
                )

        else:
            # Initial run
            U = self.flatten(fields)
            le_sum = np.zeros(n)
            if isinstance(b, str):
                if b == 'U':
                    b = U.copy()
                elif b == 'random':
                    b = np.random.randn(len(U))
                elif b == 'phases':
                    b = self.phase_shifted_b(fields)
            elif not isinstance(b, np.ndarray):
                raise ValueError("b must be 'U', 'random', 'phases', or ndarray.")

            # Initialize Q
            Q = np.zeros((len(U), n))
            Q[:, 0] = b / np.linalg.norm(b)
            for i in range(1, n):
                q = np.random.randn(len(U))
                for j in range(i):
                    q -= np.dot(Q[:, j], q) * Q[:, j]
                Q[:, i] = q / np.linalg.norm(q)

            if return_hist:
                le_hist = np.zeros((nsteps, n))

        # --------------------------------------------------
        # --- MAIN LOOP ------------------------------------
        # --------------------------------------------------

        for step in range(start_step, nsteps):

            # batched tangent map: exact linearized pass under JAX,
            # FD column loop under NumPy (see tangent_map_batch)
            Uevol, V = self.tangent_map_batch(U, Q, T, sx=sx, ep0=ep0)

            Q, R = np.linalg.qr(V)

            diagR = np.abs(np.diag(R))
            le_sum += np.log(diagR + 1e-300)

            t = (step + 1) * T

            if return_hist:
                le_hist[step, :] = le_sum / t

            U = np.copy(Uevol)

            # If using forcing update seed
            if hasattr(self.solver, 'seed'):
                self.solver.seed += 1 

            # --------------------------------------------------
            # --- CHECKPOINT SAVE -----------------------------
            # --------------------------------------------------

            if checkpoint_file is not None and checkpoint_every is not None:
                if (step + 1) % checkpoint_every == 0:
                    save_dict = dict(
                        step=step + 1,
                        U=U,
                        Q=Q,
                        le_sum=le_sum,
                        times=times
                    )

                    if return_hist:
                        save_dict["le_hist"] = le_hist

                    np.savez(checkpoint_file, **save_dict)

        # Final checkpoint save after loop completion
        if checkpoint_file is not None:
            save_dict = dict(
                step=nsteps,
                U=U,
                Q=Q,
                le_sum=le_sum
            )

            if return_hist:
                save_dict["le_hist"] = le_hist

            np.savez(checkpoint_file, **save_dict)

        # --------------------------------------------------
        # --- FINAL RESULTS --------------------------------
        # --------------------------------------------------

        lyap_exponents = le_sum / (nsteps * T)
        lyap_exponents = np.sort(lyap_exponents)[::-1]

        # --- Kaplan–Yorke dimension ---
        S = np.cumsum(lyap_exponents)
        positive = np.where(S >= 0)[0]
        if len(positive) > 0:
            j = positive[-1]
            if j + 1 < len(S):
                D_KY = (j + 1) + S[j] / abs(lyap_exponents[j + 1])
            else:
                D_KY = float(j)
        else:
            D_KY = 0.0

        if not return_hist:
            return lyap_exponents, D_KY

        sort_idx = np.argsort(le_hist[-1])[::-1]
        le_hist = le_hist[:, sort_idx]

        return lyap_exponents, D_KY, le_hist

class UPONewtonSolver(DynSys):
    """
    Implements the standard Newton-Krylov method for computing steady states and
    periodic orbits in PDE systems.
    """
    def __init__(self,
                 solver,
                 T: float | None,
                 sx: float | None,
                 Tconst: float = 1.0,
                 N_newt: int = 200,
                 N_gmres: int = 300,
                 N_hook: int = 25,
                 tol_newt: float = 1e-5,
                 tol_gmres: float = 1e-3,
                 tol_improve: float = 1e-3,
                 restart_iN: int = 0,
                 eps0: float = 1e-7,
                 mu0: float = 1e-3,
                 mu_inc: float = 2.0,
                 c: float = 0.5,
                 reduc_reg: float = 0.5,
                 sp1: bool = False,
                 sp2: bool = False,
                 sp_dU: bool = False,
                 save_outputs: str = 'all',
                 save_balance: str = 'all',
                 ostep: int = 1000,
                 bstep: int = 1000,
                 output_dir: str = 'output',
                 balance_dir: str = 'balance',
                 newton_dir: str = 'newton',
                 gmres_dir: str = 'gmres',
                 hookstep_dir: str = 'hookstep',
                 trust_region_dir: str = 'trust_region',
                 apply_A_dir: str = 'apply_A',
                 converged_dir: str = 'converged'
                 ):

        super().__init__(solver)
        
        # System parameters
        self.T = T
        self.sx = sx
        self.Tconst = Tconst
        self.eps0 = eps0
        
        # Projections and outputs
        self.sp1 = sp1
        self.sp2 = sp2
        self.sp_dU = sp_dU
        self.save_outputs = save_outputs
        self.save_balance = save_balance
        self.ostep = ostep
        self.bstep = bstep
        
        # Solver hyperparameters
        self.restart_iN = restart_iN
        self.N_newt = N_newt
        self.N_gmres = N_gmres
        self.tol_gmres = tol_gmres
        self.N_hook = N_hook
        self.c = c
        self.reduc_reg = reduc_reg
        self.mu0 = mu0
        self.mu_inc = mu_inc
        self.tol_newt = tol_newt
        self.tol_improve = tol_improve
        
        # Directories
        self.output_dir = output_dir
        self.balance_dir = balance_dir
        self.newton_dir = newton_dir
        self.gmres_dir = gmres_dir
        self.apply_A_dir = apply_A_dir
        self.hookstep_dir = hookstep_dir
        self.trust_region_dir = trust_region_dir
        self.converged_dir = converged_dir

    def _evolve_path(self, base_dir, iN):
        """Build the output/balance path for a given base directory and Newton iteration."""
        path = base_dir
        if iN is not None:
            path = os.path.join(path, f"iN{iN:02}")
        os.makedirs(path, exist_ok=True)
        return path

    def _evolve_kwargs(self, save_out, save_bal, iN=None):
        """Build keyword arguments for self.evolve with proper ostep/bstep and paths."""
        kwargs = {}
        if save_out:
            kwargs['ostep'] = self.ostep
            kwargs['opath'] = self._evolve_path(self.output_dir, iN)
        if save_bal:
            kwargs['bstep'] = self.bstep
            kwargs['bpath'] = self._evolve_path(self.balance_dir, iN)
        return kwargs

    def form_X(self, U, T, sx):
        """Pack U, T, sx into a single state vector X."""
        X = np.copy(U)
        if self.T is not None:
            X = np.append(X, T)
        if self.sx is not None:
            X = np.append(X, sx)
        return X

    def unpack_X(self, X):
        """Extract U, T, sx from state vector X."""
        if not self.remove_boundary:
            dim_U = self.grid.N * self.solver.num_fields
        else:
            # Remove boundaries in z for Krylov
            dim_U = self.grid.Nx * (self.grid.Nz-2) * self.solver.num_fields

        U = X[:dim_U]
        idx = dim_U

        if self.T is not None:
            T = X[idx]
            idx += 1
        else:
            T = self.Tconst

        sx = X[idx] if (self.sx is not None) else 0.
        return U, T, sx

    def form_b(self, U, UT):
        """Form Newton residual b = U - U(T), appending zeros for T and sx constraints."""
        b = U - UT
        if self.sx is not None:
            b = np.append(b, 0.)
        if self.T is not None:
            b = np.append(b, 0.)
        return b

    def norm(self, U):
        """Return L2 norm of U."""
        return np.linalg.norm(U)

    def apply_proj(self, U, dU, sp):
        """Return U + dU, optionally applying solenoidal projection to dU or the sum."""
        if not sp:
            return U+dU
        if self.sp_dU:
            dU = self.sol_project(dU)
            return U+dU
        else:
            return self.sol_project(U+dU)

    def update_A(self, X, iN):
        """Evolve trajectory from X and build the linear operator apply_A for GMRES."""
        U, T, sx = self.unpack_X(X)
        save_out = True if self.save_outputs == 'all' else False
        save_bal = True if self.save_balance == 'all' else False
        UT = self.evolve(U, T, **self._evolve_kwargs(save_out, save_bal, iN=iN-1))

        if self.sx is not None:
            UT = self.translate(UT, sx)
            dUT_ds = self.deriv_U(UT, self.grid.kx)
            dU_ds = self.deriv_U(U, self.grid.kx)
        else:
            dUT_ds = dU_ds = np.zeros_like(U)

        if self.T is not None:
            dUT_dT = self.evolve(UT, self.grid.dt)
            dUT_dT = (dUT_dT - UT)/self.grid.dt
            dU_dt = self.evolve(U, self.grid.dt)
            dU_dt = (dU_dt - U)/self.grid.dt
        else:
            dUT_dT = dU_dt = np.zeros_like(U)

        def apply_A(dX):
            ''' Applies A (extended Jacobian) to vector X^t  '''
            dU, dT, ds = self.unpack_X(dX)

            # sp1 projects the perturbation *direction* (same semantics in
            # both backends; the NumPy path previously could project U + dU)
            dU_in = self.sol_project(dU) if self.sp1 else dU
            dUT_dU = self.tangent_map(U, dU_in, T,
                                      sx=sx if self.sx is not None else None,
                                      UT=UT, ep0=self.eps0)

            Tx_proj = np.dot(dU_ds.conj(), dU).real
            t_proj = np.dot(dU_dt.conj(), dU).real
            norms = [self.norm(U_) for U_ in [U, dU, dUT_dU, dU_dt, dUT_dT, dU_ds, dUT_ds]]

            self.write_apply_A(iN, norms, t_proj, Tx_proj)

            LHS = dUT_dU - dU + dUT_ds*ds + dUT_dT*dT
            if self.T is not None:
                LHS = np.append(LHS, t_proj)
            if self.sx is not None:
                LHS = np.append(LHS, Tx_proj)
            return LHS

        return apply_A, UT

    def run_newton(self, X):
        """Run Newton-Krylov iterations starting from state X until convergence."""
        if self.restart_iN == 0:
            self.write_header_newton()

        for iN in range(self.restart_iN+1, self.N_newt):
            U, T, sx = self.unpack_X(X)
            apply_A, UT = self.update_A(X, iN)
            b = self.form_b(U, UT)
            F = self.norm(b)

            self.write_newton(iN, F, U, sx, T)
            self.write_headers(iN)

            H, beta, Q = GMRES(apply_A, b, self.N_gmres, self.tol_gmres, self.gmres_dir, iN)
            X, F_new, UT = self.hookstep(X, H, beta, Q, iN)
            U, T, sx = self.unpack_X(X)

            if (F_new < self.tol_newt) and ((F-F_new)/F < self.tol_improve):
                save_out = True if self.save_outputs in ('all', 'last') else False
                save_bal = True if self.save_balance in ('all', 'last') else False
                UT = self.evolve(U, T, **self._evolve_kwargs(save_out, save_bal, iN=iN))
                if self.sx is not None:
                    UT = self.translate(UT, sx)
                b = self.form_b(U, UT)
                F = self.norm(b)
                self.write_newton(iN+1, F, U, sx, T)
                break

    def hookstep(self, X, H, beta, Q, iN):
        """Perform trust-region hookstep update; shrinks step until residual improvement is linear."""
        U, T, sx = self.unpack_X(X)
        y = backsub(H, beta)
        Delta = self.norm(y)
        trust_region = self.trust_region_function(H, beta, iN, y)

        mu = 0.
        for iH in range(self.N_hook):
            y, mu = trust_region(Delta, mu)
            dx = Q@y
            dU, dT, dsx = self.unpack_X(dx)

            U_new = self.apply_proj(U, dU, self.sp2)
            sx_new = sx + dsx.real if self.sx is not None else 0.
            T_new = T + dT.real if self.T else self.Tconst
            X_new = self.form_X(U_new, T_new, sx_new)

            UT = self.evolve(U_new, T_new)
            if self.sx is not None:
                UT = self.translate(UT, sx_new)

            b = self.form_b(U_new, UT)
            F_new = self.norm(b)
            lin_exp = self.norm(beta - self.c * H @ y)

            self.write_hookstep(iN, iH, F_new, lin_exp)

            if F_new <= lin_exp:
                break
            else:
                Delta *= self.reduc_reg
        return X_new, F_new, UT

    def trust_region_function(self, H, beta, iN, y0, iA: int | None = None):
        """Return a closure that solves the trust-region subproblem (A + mu*I)y = b."""
        R = H
        A_ = R.T @ R
        b = R.T @ beta

        def trust_region(Delta, mu):
            if mu == 0:
                y_norm = self.norm(y0)
                self.write_trust_region(iN, y_norm, mu, y0, A_, iA)
                return y0, self.mu0

            for _ in range(1, 1000):
                A = A_ + mu * np.eye(A_.shape[0])
                y = np.linalg.solve(A, b)
                self.write_trust_region(iN, Delta, mu, y, A, iA)
                if self.norm(y) <= Delta:
                    break
                else:
                    mu *= self.mu_inc
            return y, mu
        return trust_region

    def mkdir(self, path):
        os.makedirs(path, exist_ok=True)

    def _get_path(self, base_dir: str | None, *parts, iA: int | None = None):
        if base_dir is None:
            return None
        if iA is not None:
            base_dir = f"{base_dir}{iA:02}"
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, *parts)

    def write_header_newton(self, iA: int | None = None):
        path = self._get_path(self.newton_dir, "newton.txt", iA=iA)
        if path:
            print("iN, |F|, T, sx, |U|", file=open(path, "a"))

    def write_newton(self, iN, F, U, sx, T, iA: int | None = None):
        path = self._get_path(self.newton_dir, "newton.txt", iA=iA)
        if path:
            print(f"{iN-1:02},{F:.6e},{T},{sx:.8e},{self.norm(U):.6e}", file=open(path, "a"))

    def write_headers(self, iN, iA: int | None = None):
        suffix = f'iN{iN:02}.txt'
        gmres_path = self._get_path(self.gmres_dir, 'gmres_'+suffix, iA=iA)
        apply_A_path = self._get_path(self.apply_A_dir, 'apply_A_'+suffix, iA=iA)
        hook_path = self._get_path(self.hookstep_dir, 'hookstep_'+suffix, iA=iA)
        trust_region_path = self._get_path(self.trust_region_dir, 'trust_region_'+suffix, iA=iA)

        if gmres_path: print('iG, error', file=open(gmres_path, "w"))
        if hook_path: print('iH, |F|, |F(x)+cAdx|', file=open(hook_path, "w"))
        if apply_A_path: print("|U|, |dU|, |dUT/dU|, |dU/dt|, |dUT/dT|, |dU/ds|, |dUT/ds|, t_proj, Tx_proj", file=open(apply_A_path, "w"))
        if trust_region_path: print("Delta, mu, |y|, cond(A)", file=open(trust_region_path, "w"))

    def write_apply_A(self, iN, norms, t_proj, Tx_proj, iA: int | None = None):
        path = self._get_path(self.apply_A_dir, f"apply_A_iN{iN:02}.txt", iA=iA)
        if path:
            content = ",".join([f"{norm:.4e}" for norm in norms]) + f",{t_proj:.4e},{Tx_proj:.4e}"
            print(content, file=open(path, "a"))

    def write_hookstep(self, iN, iH, F_new, lin_exp, iA: int | None = None):
        path = self._get_path(self.hookstep_dir, f"hookstep_iN{iN:02}.txt", iA=iA)
        if path:
            print(f"{iH:02},{F_new:.4e},{lin_exp:.4e}", file=open(path, "a"))

    def write_trust_region(self, iN, Delta, mu, y, A, iA: int | None = None):
        path = self._get_path(self.trust_region_dir, f"trust_region_iN{iN:02}.txt", iA=iA)
        if path:
            print(f"{Delta:.4e},{mu:.4e},{self.norm(y):.4e},{np.linalg.cond(A):.3e}", file=open(path, "a"))

class ArclengthNewtonSolver(UPONewtonSolver):
    """
    Extends the UPONewtonSolver to perform arclength continuation following
    Chandler & Kerswell (2013).
    """
    def __init__(self,
                 solver,
                 lda: float,
                 alpha: float = 1.0,
                 norm_val: float = 1.0,
                 restart_iA: int = 0,
                 N_arc: int = 10,
                 **kwargs):
        
        # Initialize the base UPONewtonSolver with all other kwargs
        super().__init__(solver, **kwargs)
        
        self.lda = lda
        self.alpha = alpha
        self.norm_val = norm_val
        self.restart_iA = restart_iA
        self.N_arc = N_arc
        self._current_iA = None

    def _evolve_path(self, base_dir, iN):
        """Build path with iA prefix for arclength continuation."""
        path = base_dir
        if self._current_iA is not None:
            path = f"{path}{self._current_iA:02}"
        if iN is not None:
            path = os.path.join(path, f"iN{iN:02}")
        os.makedirs(path, exist_ok=True)
        return path

    def form_X(self, U, T=None, sx=None, lda=None):
        """Pack U, T, sx, lda into a single state vector X."""
        X = super().form_X(U, T, sx)
        if lda is not None:
            X = np.append(X, lda)
        return X

    def unpack_X(self, X):
        """Extract U, T, sx, lda from state vector X."""
        U, T, sx = super().unpack_X(X)
        lda = X[-1]
        return U, T, sx, lda

    def hookstep(self, X, H, beta, Q, iN, arclength: dict):
        """Perform hookstep update with the arclength constraint included in the residual."""
        U, T, sx, lda = self.unpack_X(X)
        y = backsub(H, beta)
        Delta = self.norm(y)

        trust_region = self.trust_region_function(H, beta, iN, y, arclength["iA"])
        mu = 0.

        for iH in range(self.N_hook):
            y, mu = trust_region(Delta, mu)
            dx = Q@y
            dU, dT, dsx, dlda = self.unpack_X(dx)

            U_new = self.apply_proj(U, dU, self.sp2)
            sx_new = sx + dsx.real if self.sx is not None else 0.
            T_new = T + dT.real if self.T else self.Tconst
            
            X_new = super().form_X(U_new, T_new, sx_new)
            
            lda_new = lda + dlda.real
            X_new = np.append(X_new, lda_new)
            self.update_lda(lda_new)

            UT = self.evolve(U_new, T_new)
            if self.sx is not None:
                UT = self.translate(UT, sx_new)

            b = self.form_b(U_new, UT)
            N = self.N_constraint(X_new, arclength["dX_dr"], arclength["dr"], arclength["X1"])
            b = np.append(b, -N)

            F_new = self.norm(b)
            lin_exp = self.norm(beta - self.c * H @ y)

            self.write_hookstep(iN, iH, F_new, lin_exp, arclength["iA"])

            if F_new <= lin_exp:
                break
            else:
                Delta *= self.reduc_reg
                
        return X_new, F_new, UT

    def write_header_newton(self, iA: int | None = None):
        path = self._get_path(self.newton_dir, "newton.txt", iA=iA)
        if path:
            print("iN, |F|, T, sx, |U|, lambda, N(X)", file=open(path, "a"))

    def write_newton(self, iN, F, U, sx, T, arclength_info: tuple, iA: int | None = None):
        path = self._get_path(self.newton_dir, "newton.txt", iA=iA)
        if path:
            lda_scaled, N_val = arclength_info
            print(f"{iN-1:02},{F:.6e},{T},{sx:.8e},{self.norm(U):.6e},{lda_scaled},{N_val}", file=open(path, "a"))

    def write_headers(self, iN, iA: int | None = None):
        super().write_headers(iN, iA)
        suffix = f'iN{iN:02}.txt'
        apply_A_path = self._get_path(self.apply_A_dir, 'apply_A_'+suffix, iA=iA)
        if apply_A_path:
            header = "|U|, |dU|, |dUT/dU|, |dU/dt|, |dUT/dT|, |dU/ds|, |dUT/ds|, |dUT/dlda|, t_proj, Tx_proj, lda_proj"
            print(header, file=open(apply_A_path, "w"))

    def write_apply_A(self, iN, norms, t_proj, Tx_proj, lda_proj, iA: int | None = None):
        path = self._get_path(self.apply_A_dir, f"apply_A_iN{iN:02}.txt", iA=iA)
        if path:
            content = ",".join([f"{norm:.4e}" for norm in norms]) + f",{t_proj:.4e},{Tx_proj:.4e},{lda_proj:.4e}"
            print(content, file=open(path, "a"))

    def update_lda(self, lda):
        '''Updates lambda parameter in solver. norm: normalization parameter

        WARNING (JAX backend): mutating a parameter that a jitted ``rkstep``
        reads from ``self`` (e.g. Kolmogorov nu) is NOT picked up by the
        cached compiled step -- jit treats self as a static argument.  Under
        JAX, rebuild the solver instead of mutating it.
        '''
        if self.solver.solver == 'KolmogorovFlow':
            self.solver.pm.nu = 1 / lda  # Assigning directly to solver's pm since self.pm is removed
        elif self.solver.solver == 'BOUSS':
            self.solver.pm.ra = lda * self.norm_val
        else:
            raise Exception('Arclength only implemented for Kolmogorov and BOUSS')

    def N_constraint(self, X, dX_dr, dr, X1):
        """Evaluate arclength constraint N(X) = dX_dr . (X - X1) - dr * alpha^2."""
        return np.dot(dX_dr, (X - X1)) - dr * self.alpha**2

    def update_A_arc(self, X, dX_dr, iN, iA: int | None):
        """Evolve trajectory and build apply_A including the lda column for arclength continuation."""
        U, T, sx, lda = self.unpack_X(X)
        self.update_lda(lda)
        self._current_iA = iA

        save_out = True if self.save_outputs == 'all' else False
        save_bal = True if self.save_balance == 'all' else False
        UT = self.evolve(U, T, **self._evolve_kwargs(save_out, save_bal, iN=iN-1))

        if self.sx is not None:
            UT = self.translate(UT, sx)
            dUT_ds = self.deriv_U(UT, self.grid.kx)
            dU_ds = self.deriv_U(U, self.grid.kx)
        else:
            dUT_ds = dU_ds = np.zeros_like(U)

        if self.T is not None:
            dUT_dT = self.evolve(UT, self.grid.dt)
            dUT_dT = (dUT_dT - UT) / self.grid.dt
            dU_dt = self.evolve(U, self.grid.dt)
            dU_dt = (dU_dt - U) / self.grid.dt
        else:
            dUT_dT = dU_dt = np.zeros_like(U)

        dlda = lda * 1e-3
        self.update_lda(lda + dlda)
        dUT_dlda = self.evolve(U, T)
        if self.sx is not None:
            dUT_dlda = self.translate(dUT_dlda, sx)
        dUT_dlda = (dUT_dlda - UT) / dlda
        self.update_lda(lda)

        def apply_A(dX):
            ''' Applies A (extended Jacobian) to vector X^t  '''
            dU, dT, ds, dlda_val = self.unpack_X(dX)

            # sp1 projects the perturbation *direction* (same semantics in
            # both backends; the NumPy path previously could project U + dU)
            dU_in = self.sol_project(dU) if self.sp1 else dU
            dUT_dU = self.tangent_map(U, dU_in, T,
                                      sx=sx if self.sx is not None else None,
                                      UT=UT, ep0=self.eps0)

            Tx_proj = np.dot(dU_ds.conj(), dU).real
            t_proj = np.dot(dU_dt.conj(), dU).real
            lda_proj = np.dot(dX_dr.conj(), dX).real

            norms = [self.norm(U_) for U_ in [U, dU, dUT_dU, dU_dt, dUT_dT, dU_ds, dUT_ds, dUT_dlda]]
            self.write_apply_A(iN, norms, t_proj, Tx_proj, lda_proj, iA)

            LHS = dUT_dU - dU + dUT_ds*ds + dUT_dT*dT + dUT_dlda*dlda_val
            if self.T is not None:
                LHS = np.append(LHS, t_proj)
            if self.sx is not None:
                LHS = np.append(LHS, Tx_proj)
            LHS = np.append(LHS, lda_proj)
            
            return LHS

        return apply_A, UT

    def run_arclength(self, X0, X1, X_restart=None, iA: int | None = None):
        """Run Newton iterations for one arclength step from X0 to a new solution near X1."""
        dr = np.linalg.norm(X1 - X0)
        dX_dr = (X1 - X0) / dr
        X = X_restart if X_restart is not None else X1

        start_iN = self.restart_iN + 1 if (iA == self.restart_iA) else 1

        for iN in range(start_iN, self.N_newt):
            U, T, sx, lda = self.unpack_X(X)
            apply_A, UT = self.update_A_arc(X, dX_dr, iN, iA)
            b = self.form_b(U, UT)

            N = self.N_constraint(X, dX_dr, dr, X1)
            b = np.append(b, -N)
            F = self.norm(b[:-1])

            self.write_newton(iN, F, U, sx, T, (lda * self.norm_val, N), iA)
            self.write_headers(iN, iA)

            H, beta, Q = GMRES(apply_A, b, self.N_gmres, self.tol_gmres, self.gmres_dir, iN, iA)
            X, F_new, UT = self.hookstep(X, H, beta, Q, iN, arclength={'iA': iA, 'dX_dr': dX_dr, 'dr': dr, 'X1': X1})
            U, T, sx, lda = self.unpack_X(X)

            if (F_new < self.tol_newt) and ((F - F_new) / F < self.tol_improve):
                save_out = True if self.save_outputs in ('all', 'last') else False
                save_bal = True if self.save_balance in ('all', 'last') else False
                UT = self.evolve(U, T, **self._evolve_kwargs(save_out, save_bal, iN=iN))
                if self.sx is not None:
                    UT = self.translate(UT, sx)
                    
                b = self.form_b(U, UT)
                N = self.N_constraint(X, dX_dr, dr, X1)
                b = np.append(b, -N)
                F = self.norm(b[:-1])
                
                self.write_newton(iN + 1, F, U, sx, T, (lda * self.norm_val, N), iA)

                if iA is not None:
                    self.save_converged(X, iA)
                return X

    def run_arc_auto(self, X0, X1, X_restart=None):
        """Automate successive arclength continuation steps, using each converged solution as the next predictor."""
        start_iA = max(2, self.restart_iA)

        for iA in range(start_iA, self.N_arc):
            if (self.restart_iN == 0) or (iA != start_iA):
                self.write_header_newton(iA=iA)

            X = self.run_arclength(X0, X1, X_restart, iA)
            X0 = X1
            X1 = X
            X_restart = None

    def save_converged(self, X, iA):
        """Save converged solution fields and scalar parameters to the converged directory."""
        U, T, sx, lda = self.unpack_X(X)
        path = self._get_path(self.converged_dir, "solver.txt", iA=iA)
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if iA == 0 and not os.path.exists(path):
            print("iA, T, sx, lda, |U|", file=open(path, "a"))

        print(f"{iA:02},{T},{sx:.8e},{lda * self.norm_val},{self.norm(U):.6e}", file=open(path, "a"))
        self.write_fields(U, idx=0, path=os.path.dirname(path))
