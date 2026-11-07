"""Shared parameters for AD solver examples."""
import numpy as np

# --- KSE (1D, fast) ---
kse_Lx = 22.0
kse_Nx = 64
kse_dt = 1e-3
kse_nu = 1.0

# --- Kolmogorov flow (2D) ---
kf_Lx = 2 * np.pi
kf_Ly = 2 * np.pi
kf_Nx = 32
kf_Ny = 32
kf_dt = 0.01
kf_nu = 1.0 / 40.0
kf_kf = 4
kf_f0 = 1.0
