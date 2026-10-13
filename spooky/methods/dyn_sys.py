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
    weights: sequence or None, optional
        Per-field weights (scalars or arrays broadcastable to the flattened
        field shape, i.e. the interior ``(Nx, Nz-2)`` on wall-bounded grids)
        applied by ``flatten`` and removed by ``unflatten``.  The flat state is
        ``w_i * f_i``, so Euclidean inner products on flat vectors -- the ones
        Newton/GMRES/hookstep/Arnoldi/arclength use -- become weighted ones:
        e.g. the square root of the quadrature weights makes |U|^2 the
        physical mean-square norm on a Chebyshev grid.  The tangent map is
        conjugated by the same diagonal scaling, which leaves Floquet
        multipliers (and periodic orbits) unchanged.  Default: unweighted.
    sym: callable or None, optional
        A linear involution sigma of the fields (list of physical arrays ->
        list) that commutes with the flow, e.g. a reflection.  When given, the
        map after the flow becomes sigma tau_sx (``post_map``): Newton then
        solves sigma tau_sx Phi_T(U) = U, i.e. pre-periodic orbits with
        u(T) = (sigma tau_sx)^-1 u(0) over half of their period, and Floquet
        returns the eigenvalues of sigma tau_sx DPhi_T, whose squares are the
        multipliers of the full orbit.  Default: None (plain tau_sx).
    """
    def __init__(self, solver, weights=None, sym=None):
        self.solver = solver
        self.grid = solver.grid
        self.weights = None if weights is None else [np.asarray(w, dtype=float) for w in weights]
        self.sym = sym

        # Wall-bounded grids carry interior-only state for the UPO/Floquet
        # machinery: the Dirichlet boundary rows are stripped on flatten and
        # re-padded with zeros on unflatten. Grids advertise this via a
        # ``remove_boundary`` attribute (Grid2D_semi, Grid2D_wall).
        self.remove_boundary = getattr(self.grid, 'remove_boundary', False)
        if isinstance(self.grid, ps.Grid2D_semi):
            self.remove_boundary = True

    def flatten(self, fields):
        '''Flattens fields (interior only on wall-bounded grids; weighted if
        ``weights`` were given)'''
        parts = [np.asarray(f) if not self.remove_boundary else np.asarray(f)[:, 1:-1]
                 for f in fields]
        if self.weights is not None:
            parts = [w * p for w, p in zip(self.weights, parts)]
        return np.concatenate([p.ravel() for p in parts])

    def unflatten(self, U):
        '''Unflatten fields (removes the weights; re-pads the wall rows with zeros)'''
        ll = len(U)//self.solver.num_fields
        fields = [U[i*ll:(i+1)*ll] for i in range(self.solver.num_fields)]
        shape = self.grid.shape if not self.remove_boundary else (self.grid.Nx, self.grid.Nz-2)
        fields = [f.reshape(shape) for f in fields]
        if self.weights is not None:
            fields = [f / w for f, w in zip(fields, self.weights)]
        if self.remove_boundary:
            fields = [np.pad(f, pad_width = ((0,0),(1,1))) for f in fields] #pads second dimension with zeros before and after
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

    def symmetry(self, U):
        '''Flat-vector action of ``sym`` (identity when sym is None)'''
        if self.sym is None:
            return U
        return self.flatten(self.sym(self.unflatten(U)))

    def post_map(self, UT, sx=None):
        '''The map applied after the flow: translate by sx (if not None), then ``sym``'''
        if sx is not None:
            UT = self.translate(UT, sx)
        return self.symmetry(UT)

    @flatten_dec
    def deriv_U(self, U, *args):
        '''Derivatives in x direction of fields'''
        return self.grid.deriv_fields(U, *args)

    @flatten_dec
    def sol_project(self, U):
        '''Solenoidal projection of fields'''
        return self.solver.inc_proj(U)

    def write_fields(self, U, idx, path):
        ''' Writes fields to file'''
        fields = self.unflatten(U)
        if hasattr(self.solver, 'write_fields'):
            return self.solver.write_fields(fields, path=path, idx=idx)
        # Pseudo-spectral solvers save through outs, which expects Fourier-space fields
        fields = [self.grid.forward(f) for f in fields]
        return self.solver.outs(fields, idx, path)

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
        """Action of the tangent map J = d(sym ∘ translate_sx ∘ Φ_T)/dU on dU.

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
            Base-point image post_map(Φ_T(U), sx) — required for the FD
            (NumPy) path, ignored under JAX.
        ep0 : float
            FD perturbation scale factor.
        """
        if JAX_ENABLED:
            import jax.numpy as jnp
            f_jax  = [jnp.array(f) for f in self.unflatten(U)]
            dU_jax = [jnp.array(d) for d in self.unflatten(dU)]
            _, tangent_list = self.jvp_evolve(f_jax, dU_jax, T)
            J_dU = self.post_map(np.array(self.flatten(tangent_list)), sx)
        else:
            if UT is None:
                raise ValueError(
                    "tangent_map FD path requires UT = post_map(Phi_T(U), sx)")
            epsilon = ep0 * np.linalg.norm(U) / np.linalg.norm(dU)
            J_dU = self.post_map(self.evolve(U + epsilon * dU, T), sx)
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
            Tangent images (post_map: translated by sx if given, then sym).
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
                V[:, i] = self.post_map(self.flatten([np.array(t[i]) for t in tans]), sx)
        else:
            Uevol = self.evolve(U, T)
            UT = self.post_map(Uevol, sx)
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

        b_vector = self.flatten(fields_proj)
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

        # Translate UT by sx (and apply sym)
        UT = self.post_map(UT, sx)

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
        checkpoint_every=None,
        store_dir=None,
        store_Q_every=None,
        store_Q_max=None,
        ):
        ''' Computes Lyapunov exponents and Kaplan–Yorke dimension via QR iteration.

        This method implements the Benettin algorithm for estimating finite-time
        Lyapunov exponents by repeatedly evolving a set of orthonormal perturbations
        over intervals of duration T and reorthonormalizing them using QR decomposition.

        With `store_dir` it is also the forward pass of the Ginelli et al. (2007)
        algorithm for covariant Lyapunov vectors: the QR factors of step k
        (J_T Q_{k-1} = Q_k R_k, t_k = k T) are written to disk and
        `covariant_lyapunov_vectors` runs the backward recursion on them.

        Parameters
        ----------
        fields : list of fields
            Initial fields defining the state U. Not used when resuming from a
            checkpoint (the state is read from it), so it may be None then.
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
        store_dir : str or None, optional
            If given, after every QR step k (1-based: k = step + 1, so t_k = k T and
            le_hist[k-1] is the estimate at t_k) the triangular factor R_k is written
            to `<store_dir>/R_{k:05d}.npy`. The basis Q_k and the state U_k at the
            same instant go to `Q_{k:05d}.npy` / `U_{k:05d}.npy` when k is a multiple
            of `store_Q_every` (and k <= `store_Q_max`, if set). Step numbers continue
            across resumes, so a resumed run extends the same store.
        store_Q_every : int or None, optional
            Store Q_k (and U_k) every this many steps; None stores no Q at all.
        store_Q_max : int or None, optional
            Do not store Q_k beyond this step (the backward transient of the CLV
            recursion makes late steps useless, so their Q need not be kept).

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
            np.random.seed(42)
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

            # Ginelli forward pass: keep the QR factors of this step (k = step + 1)
            if store_dir is not None:
                k = step + 1
                os.makedirs(store_dir, exist_ok=True)
                np.save(os.path.join(store_dir, f'R_{k:05d}.npy'), R)
                if (store_Q_every is not None and k % store_Q_every == 0
                        and (store_Q_max is None or k <= store_Q_max)):
                    np.save(os.path.join(store_dir, f'Q_{k:05d}.npy'), Q)
                    np.save(os.path.join(store_dir, f'U_{k:05d}.npy'), Uevol)

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


def covariant_lyapunov_vectors(store_dir, k_lo=None, k_hi=None, n_back=None,
                               load_Q=True, seed=0, verbose=True):
    ''' Covariant Lyapunov vectors (CLVs) by the backward pass of Ginelli et al.,
    PRL 99, 130601 (2007), from the QR factors stored by `DynSys.lyapunov_exponents`
    (`store_dir`). Pure numpy: no solver is needed.

    The forward pass gives J_T Q_{k-1} = Q_k R_k (t_k = k T). Writing the j-th CLV
    in the Gram-Schmidt basis, v_j(t_k) = Q_k c_k^{(j)}, covariance
    J_T v_j(t_{k-1}) ∝ v_j(t_k) becomes R_k c_{k-1} ∝ c_k, i.e.

        c_{k-1} = R_k^{-1} c_k / ||R_k^{-1} c_k||        (backward recursion)

    v_j lies in span(Q_k[:, :j]) (backward Oseledets filtration), so the coefficient
    matrix C_k = [c^{(1)} ... c^{(n)}] is upper triangular and stays so. Starting from
    an arbitrary upper-triangular C at k_hi and iterating down, C_k converges to the
    CLV coefficients after a backward transient: only k <= k_hi - n_back is trusted.
    The log growth of CLV j across step k, g_k^{(j)} = log||R_k c_{k-1}|| =
    -log||R_k^{-1} c_k||, averaged over the trusted window and divided by T, must
    reproduce lambda_j; this is the convergence check.

    Parameters
    ----------
    store_dir : str
        Directory holding R_{k:05d}.npy for every k and Q_{k:05d}.npy for some k.
    k_lo, k_hi : int or None
        Steps to process: R_k is used for k_lo < k <= k_hi and C_k is returned for
        k_lo <= k <= k_hi. Default: every R present.
    n_back : int or None
        Backward transient in steps; default 40% of (k_hi - k_lo).
    load_Q : bool
        Also form V_k = Q_k C_k for every stored Q_k with k_lo <= k <= k_hi - n_back.
    seed : int
        Seed of the random upper-triangular start at k_hi.

    Returns
    -------
    dict with
        steps   : (K+1,) ints k_lo .. k_hi
        C       : (K+1, n, n) unit-column upper-triangular coefficients, C[i] at steps[i]
        growth  : (K, n) g_k for k = steps[1:], row i <-> step steps[i+1]
        angles  : (K+1, n, n) angles between CLVs, arccos|c_i . c_j| (Q is orthonormal)
        k_valid : largest trusted step, k_hi - n_back
        n_back  : the transient used
        vectors : {k: (N, n) array} V_k = Q_k C_k for stored, trusted k (if load_Q)
    '''
    import re
    import glob
    try:
        from scipy.linalg import solve_triangular
        _solve = lambda R, C: solve_triangular(R, C)
    except ImportError:                                   # keep C triangular by hand
        _solve = lambda R, C: np.triu(np.linalg.solve(R, C))

    ks = {}
    for f in glob.glob(os.path.join(store_dir, 'R_*.npy')):
        m = re.search(r'R_(\d+)\.npy$', f)
        if m:
            ks[int(m.group(1))] = f
    if not ks:
        raise FileNotFoundError(f'no R_*.npy in {store_dir}')
    if k_hi is None:
        k_hi = max(ks)
    if k_lo is None:
        k_lo = min(ks) - 1
    missing = [k for k in range(k_lo + 1, k_hi + 1) if k not in ks]
    if missing:
        raise FileNotFoundError(f'{store_dir}: missing R for steps {missing[:5]}...')

    K = k_hi - k_lo
    if n_back is None:
        n_back = int(round(0.4 * K))
    n = np.load(ks[k_hi]).shape[0]

    rng = np.random.default_rng(seed)
    C = np.triu(rng.random((n, n)) + 0.5)                 # positive diagonal: safe start
    C /= np.linalg.norm(C, axis=0)

    C_hist = np.zeros((K + 1, n, n))
    growth = np.zeros((K, n))
    C_hist[K] = C
    for k in range(k_hi, k_lo, -1):
        X = _solve(np.load(ks[k]), C)                      # R_k^{-1} C_k
        norms = np.linalg.norm(X, axis=0)
        growth[k - k_lo - 1] = -np.log(norms)              # growth across step k
        C = X / norms
        C_hist[k - k_lo - 1] = C

    cos = np.abs(np.einsum('kij,kil->kjl', C_hist, C_hist))
    angles = np.arccos(np.clip(cos, 0.0, 1.0))

    k_valid = k_hi - n_back
    out = dict(steps=np.arange(k_lo, k_hi + 1), C=C_hist, growth=growth, angles=angles,
               k_valid=k_valid, n_back=n_back, vectors={})
    if load_Q:
        for f in sorted(glob.glob(os.path.join(store_dir, 'Q_*.npy'))):
            m = re.search(r'Q_(\d+)\.npy$', f)
            k = int(m.group(1)) if m else None
            if k is None or k < k_lo or k > k_valid:
                continue
            out['vectors'][k] = np.load(f) @ C_hist[k - k_lo]
    if verbose:
        lam_clv = growth[:max(k_valid - k_lo, 1)].mean(axis=0)
        print(f'CLV backward pass: steps {k_lo}..{k_hi}, trusted <= {k_valid}, '
              f'mean log-growth per step over trusted window: {np.array2string(lam_clv, precision=4)}; '
              f'{len(out["vectors"])} vector sets formed')
    return out


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
                 mu_search: bool = False,
                 delta_inc: float = 2.0,
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
                 converged_dir: str = 'converged',
                 weights=None,
                 sym=None,
                 ):

        super().__init__(solver, weights=weights, sym=sym)
        
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
        # mu_search: solve the trust-region subproblem for |y| ~ Delta and keep Delta between Newton iterations
        self.mu_search = mu_search
        self.delta_inc = delta_inc
        self._Delta_prev = None
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
            # d/dsx of sym(tau_sx Phi): differentiate before sym (X-type reflections anticommute with d/dx)
            dUT_ds = self.symmetry(self.deriv_U(UT, self.grid.kx))
            dU_ds = self.deriv_U(U, self.grid.kx)
        else:
            dUT_ds = dU_ds = np.zeros_like(U)
        UT = self.symmetry(UT)

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

    def run_newton(self, X, on_iteration=None):
        """Run Newton-Krylov iterations starting from state X until convergence;
        returns the last state X = (U, T, sx).  ``on_iteration(iN, X, F)`` is
        called after each accepted hookstep with the new state and residual
        norm (e.g. to checkpoint X; resume with ``restart_iN=iN``)."""
        if self.restart_iN == 0:
            self.write_header_newton()
        self._Delta_prev = None

        # N_newt is the total number of iterations (restart_iN of them already done)
        for iN in range(self.restart_iN+1, self.N_newt+1):
            U, T, sx = self.unpack_X(X)
            apply_A, UT = self.update_A(X, iN)
            b = self.form_b(U, UT)
            F = self.norm(b)

            self.write_newton(iN, F, U, sx, T)
            self.write_headers(iN)

            H, beta, Q = GMRES(apply_A, b, self.N_gmres, self.tol_gmres, self.gmres_dir, iN)
            X, F_new, UT = self.hookstep(X, H, beta, Q, iN)
            U, T, sx = self.unpack_X(X)
            if on_iteration is not None:
                on_iteration(iN, X, F_new)

            if (F_new < self.tol_newt) and ((F-F_new)/F < self.tol_improve):
                save_out = True if self.save_outputs in ('all', 'last') else False
                save_bal = True if self.save_balance in ('all', 'last') else False
                UT = self.evolve(U, T, **self._evolve_kwargs(save_out, save_bal, iN=iN))
                UT = self.post_map(UT, sx if self.sx is not None else None)
                b = self.form_b(U, UT)
                F = self.norm(b)
                self.write_newton(iN+1, F, U, sx, T)
                break
        return X

    def hookstep(self, X, H, beta, Q, iN):
        """Perform trust-region hookstep update; shrinks step until residual improvement is linear."""
        U, T, sx = self.unpack_X(X)
        y = backsub(H, beta)
        trust_region = self.trust_region_function(H, beta, iN, y)

        Delta, mu = self.hook_start(y)
        for iH in range(self.N_hook):
            y, mu = trust_region(Delta, mu)
            dx = Q@y
            dU, dT, dsx = self.unpack_X(dx)

            U_new = self.apply_proj(U, dU, self.sp2)
            sx_new = sx + dsx.real if self.sx is not None else 0.
            T_new = T + dT.real if self.T else self.Tconst
            X_new = self.form_X(U_new, T_new, sx_new)

            UT = self.post_map(self.evolve(U_new, T_new), sx_new if self.sx is not None else None)

            b = self.form_b(U_new, UT)
            F_new = self.norm(b)
            lin_exp = self.norm(beta - self.c * H @ y)

            self.write_hookstep(iN, iH, F_new, lin_exp)

            if F_new <= lin_exp:
                break
            else:
                Delta = self.shrink_Delta(Delta, y)
        self.hook_done(y, iH, F_new <= lin_exp)
        return X_new, F_new, UT

    def hook_start(self, y0):
        """Initial (Delta, mu) of a hookstep.  mu = 0 means: try the full Newton step y0 first.
        With ``mu_search``, Delta is capped by the radius kept from the previous Newton iteration."""
        Delta = self.norm(y0)
        if self.mu_search and self._Delta_prev is not None and self._Delta_prev < Delta:
            return self._Delta_prev, self.mu0
        return Delta, 0.

    def hook_done(self, y, iH, accepted):
        """With ``mu_search``, keep the accepted |y| as the next radius, enlarged by ``delta_inc`` if the
        first trial was accepted.  If all ``N_hook`` trials were rejected, forget the radius (the next
        iteration starts again from the full Newton step) instead of keeping the last, shrunken one."""
        if self.mu_search:
            self._Delta_prev = self.norm(y) * (self.delta_inc if iH == 0 else 1.) if accepted else None

    def shrink_Delta(self, Delta, y):
        """Trust radius after rejecting the trial step y: ``reduc_reg`` times the smaller of Delta and |y|.
        In the default mode |y| can lie far below Delta (mu only grows), and shrinking Delta alone would
        return the same, already rejected y until Delta drops below |y|."""
        return self.reduc_reg * min(Delta, self.norm(y))

    def search_mu(self, y_of, Delta, s, rtol=0.1, n_bisect=60):
        """mu > 0 with (1 - rtol) Delta <= |y(mu)| <= Delta, by bisection in log(mu) (|y(mu)| decreases
        with mu).  s: singular values of R, setting the scale of mu."""
        lo, hi = 1e-16 * s[0]**2, s[0]**2
        if self.norm(y_of(lo)) <= Delta:
            return lo
        while self.norm(y_of(hi)) > Delta:
            lo, hi = hi, 10 * hi
        for _ in range(n_bisect):
            mid = np.sqrt(lo * hi)
            y_norm = self.norm(y_of(mid))
            if y_norm > Delta:
                lo = mid
            else:
                hi = mid
                if y_norm >= (1 - rtol) * Delta:
                    break
        return hi

    def trust_region_function(self, H, beta, iN, y0, iA: int | None = None):
        """Return a closure that solves the trust-region subproblem (A + mu*I)y = b.

        Default: mu starts at ``mu0`` and grows by ``mu_inc`` until |y| <= Delta, so |y| can end far below
        Delta when A is nearly singular.  With ``mu_search``, mu is chosen so that |y| ~ Delta, using the
        SVD of R (y(mu) = V diag(s / (s^2 + mu)) W^T beta, no linear solves)."""
        R = H
        A_ = R.T @ R
        b = R.T @ beta
        if self.mu_search:
            W, s, Vt = np.linalg.svd(R)
            g = s * (W.conj().T @ beta)

            def y_of(mu):
                return Vt.conj().T @ (g / (s**2 + mu))

        def trust_region(Delta, mu):
            if mu == 0:
                y_norm = self.norm(y0)
                self.write_trust_region(iN, y_norm, mu, y0, A_, iA)
                return y0, self.mu0

            if self.mu_search:
                mu = self.search_mu(y_of, Delta, s)
                y = y_of(mu)
                self.write_trust_region(iN, Delta, mu, y, A_ + mu * np.eye(A_.shape[0]), iA)
                return y, mu

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

        trust_region = self.trust_region_function(H, beta, iN, y, arclength["iA"])
        Delta, mu = self.hook_start(y)
        UT = None

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
            if not lda_new > 0:
                # the trial leaves the parameter range (e.g. Ra <= 0 from a large full Newton step): reject it
                # without evolving (the solver cannot be built there) and shrink the trust region
                F_new, lin_exp = np.inf, self.norm(beta - self.c * H @ y)
                self.write_hookstep(iN, iH, F_new, lin_exp, arclength["iA"])
                Delta = self.shrink_Delta(Delta, y)
                continue
            self.update_lda(lda_new)

            UT = self.post_map(self.evolve(U_new, T_new), sx_new if self.sx is not None else None)

            b = self.form_b(U_new, UT)
            N = self.N_constraint(X_new, arclength["dX_dr"], arclength["dr"], arclength["X1"])
            b = np.append(b, -N)

            F_new = self.norm(b)
            lin_exp = self.norm(beta - self.c * H @ y)

            self.write_hookstep(iN, iH, F_new, lin_exp, arclength["iA"])

            if F_new <= lin_exp:
                break
            else:
                Delta = self.shrink_Delta(Delta, y)
        self.hook_done(y, iH, F_new <= lin_exp)
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
        '''Updates the continuation parameter lambda in the solver.
        lda is Re for KolmogorovFlow and Ra/norm_val for BOUSS and RayleighBenard2D.

        WARNING (JAX backend): mutating a parameter that a jitted ``rkstep``
        reads from ``self`` (e.g. Kolmogorov nu) is NOT picked up by the
        cached compiled step -- jit treats self as a static argument.  Under
        JAX, rebuild the solver instead of mutating it.  The RB path is safe:
        ra enters only the implicit operators, which the solver caches per
        (dt, ra, pr) and passes to its jitted step as traced arguments, and
        compiled JVPs are keyed on ``solver.ad_cache_key() = (ra, pr)``.
        '''
        if self.solver.solver == 'KolmogorovFlow':
            self.solver.nu = 1 / lda
        elif self.solver.solver in ('BOUSS', 'RayleighBenard2D'):
            # RayleighBenard2D: the next evolve builds (or reuses) the operators for the new ra
            self.solver.ra = lda * self.norm_val
        else:
            raise Exception(
                'Arclength only implemented for Kolmogorov, BOUSS and RayleighBenard2D')

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
            # d/dsx of sym(tau_sx Phi): differentiate before sym (X-type reflections anticommute with d/dx)
            dUT_ds = self.symmetry(self.deriv_U(UT, self.grid.kx))
            dU_ds = self.deriv_U(U, self.grid.kx)
        else:
            dUT_ds = dU_ds = np.zeros_like(U)
        UT = self.symmetry(UT)

        if self.T is not None:
            dUT_dT = self.evolve(UT, self.grid.dt)
            dUT_dT = (dUT_dT - UT) / self.grid.dt
            dU_dt = self.evolve(U, self.grid.dt)
            dU_dt = (dU_dt - U) / self.grid.dt
        else:
            dUT_dT = dU_dt = np.zeros_like(U)

        dlda = lda * 1e-3
        self.update_lda(lda + dlda)
        dUT_dlda = self.post_map(self.evolve(U, T), sx if self.sx is not None else None)
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
        """Run Newton iterations for one arclength step: starting from the two
        previous converged solutions X0 and X1, converge to a new solution at
        arclength distance ~dr further along the branch (Chandler & Kerswell 2013)."""
        dr = np.linalg.norm(X1 - X0)
        dX_dr = (X1 - X0) / dr

        if X_restart is not None:
            X = X_restart
        else:
            # Linear extrapolation predictor along the branch: satisfies N(X) = 0
            X = X1 + dr * self.alpha**2 * dX_dr

        start_iN = self.restart_iN + 1 if (iA == self.restart_iA) else 1
        self._Delta_prev = None

        for iN in range(start_iN, self.N_newt+1):
            U, T, sx, lda = self.unpack_X(X)
            apply_A, UT = self.update_A_arc(X, dX_dr, iN, iA)
            b = self.form_b(U, UT)

            N = self.N_constraint(X, dX_dr, dr, X1)
            b = np.append(b, -N)
            # Full residual of the extended system, consistent with F_new from hookstep
            F = self.norm(b)

            self.write_newton(iN, F, U, sx, T, (lda * self.norm_val, N), iA)
            self.write_headers(iN, iA)

            H, beta, Q = GMRES(apply_A, b, self.N_gmres, self.tol_gmres, self.gmres_dir, iN, iA)
            X, F_new, UT = self.hookstep(X, H, beta, Q, iN, arclength={'iA': iA, 'dX_dr': dX_dr, 'dr': dr, 'X1': X1})
            U, T, sx, lda = self.unpack_X(X)

            if (F_new < self.tol_newt) and ((F - F_new) / F < self.tol_improve):
                save_out = True if self.save_outputs in ('all', 'last') else False
                save_bal = True if self.save_balance in ('all', 'last') else False
                UT = self.evolve(U, T, **self._evolve_kwargs(save_out, save_bal, iN=iN))
                UT = self.post_map(UT, sx if self.sx is not None else None)

                b = self.form_b(U, UT)
                N = self.N_constraint(X, dX_dr, dr, X1)
                b = np.append(b, -N)
                F = self.norm(b)

                self.write_newton(iN + 1, F, U, sx, T, (lda * self.norm_val, N), iA)

                if iA is not None:
                    self.save_converged(X, iA)
                return X

        # not converged: return None (callers may retry with a smaller dr; run_arc_auto raises)
        return None

    def run_arc_auto(self, X0, X1, X_restart=None):
        """Automate successive arclength continuation steps, using each converged solution as the next predictor."""
        start_iA = max(2, self.restart_iA)

        for iA in range(start_iA, self.N_arc):
            if (self.restart_iN == 0) or (iA != start_iA):
                self.write_header_newton(iA=iA)

            X = self.run_arclength(X0, X1, X_restart, iA)
            if X is None:
                raise RuntimeError(
                    f'Arclength Newton iteration did not converge in {self.N_newt} iterations (iA={iA})')
            X0 = X1
            X1 = X
            X_restart = None

    def save_converged(self, X, iA):
        """Save converged solution fields and scalar parameters to the converged directory."""
        U, T, sx, lda = self.unpack_X(X)
        path = self._get_path(self.converged_dir, "solver.txt")
        if path is None:
            return

        if not os.path.exists(path):
            print("iA, T, sx, lda, |U|", file=open(path, "a"))

        print(f"{iA:02},{T},{sx:.8e},{lda * self.norm_val},{self.norm(U):.6e}", file=open(path, "a"))
        self.write_fields(U, idx=iA, path=os.path.dirname(path))
