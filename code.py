
#!/usr/bin/env python3
"""
ma_irs_reproduce.py -- single-file reproduction of Figs. 2, 3, 6 and 8 of

    X. Wei, W. Mei, Q. Wu, Q. Jia, B. Ning, Z. Chen, J. Fang,
    "Movable Antennas Meet Intelligent Reflecting Surface: Friends or Foes?",
    IEEE Trans. Commun., vol. 73, no. 11, pp. 12756-12770, Nov. 2025.

Runs the correctness checks, the Monte-Carlo simulation and the plotting in one
go.  Only numpy and matplotlib are required.

    python3 ma_irs_reproduce.py --quick              fast end-to-end check
    python3 ma_irs_reproduce.py                      100 realisations/point
    python3 ma_irs_reproduce.py --trials 1000        the paper's sample size
    python3 ma_irs_reproduce.py --fig 6 8 --jobs 8   only the multi-user figures
    python3 ma_irs_reproduce.py --trials 500 --t0 1000   extend an earlier run
    python3 ma_irs_reproduce.py --validate-only
    python3 ma_irs_reproduce.py --plot-only          re-draw without simulating
    python3 ma_irs_reproduce.py --no-show            write PNGs only

Figures are both written to disk and opened on screen.  An interactive
matplotlib backend is selected automatically when one is usable; on a headless
machine the script falls back to Agg, says so, and just writes the PNGs.

Results accumulate as raw sums plus a trial count in results/sums_figN*.npz, so
a long run can be split over several invocations with --t0 and still give the
same average as one long run (trial t always uses seed (0, distance_index, t)).

----------------------------------------------------------------------
WHAT IS TAKEN FROM THE PAPER, WHAT IS ASSUMED, AND WHAT THE PAPER GETS WRONG
----------------------------------------------------------------------
Stated parameters (Sec. V):
    fc = 5 GHz, N = 4, My = Mz = 15 (M = 225), d = lam/2, Dmin = lam/2,
    A = 0.6 m, delta_s = lam/10, P = 46 dBm, sigma^2 = -80 dBm,
    qB = [4sqrt2, 4sqrt2, 0], K = 3, Rician factor 3 dB, d_k ~ U[30,50],
    path-loss exponent alpha = 2.8, 1e3 channel realisations.
Fig. 2 overrides (Sec. III-A.1): M = 25^2, P/sigma^2 = 110 dB, N = 1 / N = 4,
    LoS (NUSW) BS-IRS channel, FPA fixed at Eq. (13).
Fig. 3: single user, N = 4, L = 8, d_IU = 30 m.
Fig. 6: multi-user, LoS BS-IRS channel.   Fig. 8: multi-user, multi-path L = 4.

ASSUMPTIONS the paper does not specify (all marked [ASSUMED] in the code):
A1. qB is given only at one distance (|qB| = 8 m); the stated direction is kept
    and the range swept: qB = d_BI * [1/sqrt2, 1/sqrt2, 0].
A2. C_t is parallel to the x-axis (Sec. III-A.1's own worked example), centred
    on qB.
A3. beta_BI,p in Eq. (46) is never given a value.  Since l_p ~ CN(0, 1/(L+1))
    for ALL p = 0..L including the LoS term, the model only normalises if every
    path carries equal average power, so |beta_BI,p| = lam/(4*pi*d_BI).
A4. Scatterer positions s_BI,p are unspecified: uniform in a ball of radius
    d_BI/2 about the BS-IRS midpoint.
A5. User directions are unspecified (only d_k): uniform on the x>0 hemisphere.
A6. d_IU for Fig. 2 is not stated; 30 m is used.  This is a constant offset on
    both curves and does not affect the MA-vs-FPA comparison Fig. 2 is about.
A7. "LoS BS-IRS channel" is taken as the deterministic Eq. (47).
A8. Initialisation of the position search.  This matters a great deal in the
    multi-user case and is the single largest source of divergence from the
    paper -- see the note under --init below.
A9. Convergence: 1e-4..1e-5 relative, with iteration caps.

ERRORS FOUND IN THE PAPER (implemented as corrected):
E1. Eq. (45) prints lam/(4*pi*d_k^{-alpha/2}); as printed the path gain GROWS
    with distance.  Implemented as lam/(4*pi) * d_k^{-alpha/2}.
E2. Sec. II requires My, Mz even, but Sec. V sets My = Mz = 15 (odd) and Fig. 2
    uses 25.  A centred symmetric grid of My x Mz elements is used.
E3. Eq. (35c) sums |chi_k|^2 kappa_k inside the i-sum (should be index i) and
    ends in h_k^H where a column h_k is required.  Standard WMMSE used.
E4. Eq. (43) writes the retraction as unt(alpha*eta), dropping the base point;
    unt(phi + alpha*eta) is used.
E5. G1 = diag(h_IU^H) H_BI and r_{k,i} = diag(h_IU,k^H) H_BI w_i carry a
    CONJUGATED h_IU on the diagonal.  Mixing the conjugated and unconjugated
    forms between the Phi-step and the T-step silently desynchronises the two
    blocks -- the AO still converges monotonically, it just converges to the
    wrong thing.  One convention is used throughout (see `_g2`).

ON --init (assumption A8, and the known divergence from the paper):
    The multi-user position step (Algorithm 3) is a greedy sequential search
    evaluated against a Phi that has just been phase-matched to the current
    layout, so moving any antenna destroys its own alignment and the search
    tends to stay put.  It is therefore strongly initialisation-dependent, and
    the paper does not say how it is initialised:
        compact  start from the FPA layout          -> MA gain ~0.04 bps/Hz
        spread   start spread across C_t            -> MA gain ~1.9 bps/Hz
        best     (default) try both, keep the better
    The paper's reported gain (0.45 bps/Hz for Fig. 6 at 3 m) lies between the
    two.  `best` is the default because the FPA layout is itself a feasible MA
    layout, so a correct solution of (P1) can never come out below the FPA
    benchmark -- `compact` alone occasionally does, which the paper never shows.
    Every gain measured under RANDOM IRS phases matches the paper to within
    2-3%, which is what pins the channel model down; only the optimised-IRS
    gains are affected by this.
    Figs. 2 and 3 are immune: their position step is a globally optimal DP.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")          # safe default; _enable_display() upgrades it
import matplotlib.pyplot as plt

_NON_GUI = ("agg", "pdf", "ps", "svg", "template")


def _enable_display():
    """Switch to a GUI backend if one actually works; return whether it did.

    matplotlib.use() only validates the backend NAME -- an unusable backend is
    not detected until a figure is drawn, which would crash after the whole
    simulation had already run.  switch_backend() imports the backend module
    for real, so failures surface here where they can be caught.
    """
    forced = os.environ.get("MPLBACKEND")
    if forced:
        # pyplot was imported under Agg above so that an unusable MPLBACKEND
        # cannot crash the script at import; honour the request here instead.
        if forced.lower() in _NON_GUI:
            return False
        try:
            plt.switch_backend(forced)
            return True
        except Exception:
            plt.switch_backend("Agg")
            print(f"   requested backend {forced!r} is not usable here")
            return False
    if (sys.platform.startswith("linux")
            and not os.environ.get("DISPLAY")
            and not os.environ.get("WAYLAND_DISPLAY")):
        return False
    for backend in ("QtAgg", "TkAgg", "MacOSX", "Qt5Agg"):
        try:
            plt.switch_backend(backend)
            return True
        except Exception:
            continue
    plt.switch_backend("Agg")
    return False


C_LIGHT = 3e8


# ----------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------
@dataclass
class Params:
    fc: float = 5e9          # Sec. V
    N: int = 4               # Sec. V
    My: int = 15             # Sec. V
    Mz: int = 15             # Sec. V
    A: float = 0.6           # Sec. V  (length of C_t, m)
    K: int = 3               # Sec. V
    P_dbm: float = 46.0      # Sec. V
    sigma2_dbm: float = -80.0  # Sec. V
    rician_db: float = 3.0   # Sec. V
    pl_exp: float = 2.8      # Sec. V
    d_iu: tuple = (30.0, 50.0)   # Sec. V (single user uses 30 m, Fig. 3)
    snr_db: float | None = None  # Fig. 2 override: P/sigma^2 = 110 dB

    @property
    def lam(self) -> float:
        return C_LIGHT / self.fc

    @property
    def d_irs(self) -> float:
        return self.lam / 2

    @property
    def Dmin(self) -> float:
        return self.lam / 2

    @property
    def delta_s(self) -> float:
        return self.lam / 10

    @property
    def M(self) -> int:
        return self.My * self.Mz

    @property
    def P(self) -> float:
        return 10 ** ((self.P_dbm - 30) / 10)

    @property
    def sigma2(self) -> float:
        return 10 ** ((self.sigma2_dbm - 30) / 10)

    @property
    def p_over_sigma2(self) -> float:
        if self.snr_db is not None:
            return 10 ** (self.snr_db / 10)
        return self.P / self.sigma2


# ----------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------
def irs_elements(prm: Params) -> np.ndarray:
    """IRS element coordinates, yOz-plane, centred at the origin. (M,3)."""
    my = (np.arange(prm.My) - (prm.My - 1) / 2) * prm.d_irs
    mz = (np.arange(prm.Mz) - (prm.Mz - 1) / 2) * prm.d_irs
    Y, Z = np.meshgrid(my, mz, indexing="ij")
    return np.stack([np.zeros(Y.size), Y.ravel(), Z.ravel()], axis=1)


def bs_centre(d_bi: float) -> np.ndarray:
    """[ASSUMED A1] keep the stated direction of qB, sweep the range."""
    return d_bi * np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0.0])


AXIS = np.array([1.0, 0.0, 0.0])          # [ASSUMED A2] C_t parallel to x


def offsets_fine(prm: Params) -> np.ndarray:
    """Sampling points of Algorithms 1/3: spacing delta_s over C_t."""
    L = int(round(prm.A / prm.delta_s)) + 1          # 101 points, endpoints incl.
    return np.linspace(-prm.A / 2, prm.A / 2, L)


def offsets_as(prm: Params) -> np.ndarray:
    """AS benchmark: A/Dmin FPAs spaced Dmin, centred in C_t (Sec. V-2)."""
    n = int(round(prm.A / prm.Dmin))                 # 20
    return (np.arange(n) - (n - 1) / 2) * prm.Dmin


def offsets_fpa(prm: Params, N: int) -> np.ndarray:
    """FPA benchmark: N antennas symmetric about qB, spaced Dmin (Sec. V-1)."""
    return (np.arange(N) - (N - 1) / 2) * prm.Dmin


def offset_eq13(prm: Params, d_bi: float) -> float:
    """Eq. (13): t* = argmin_{t in C_t} ||t|| (single-MA optimum)."""
    qB = bs_centre(d_bi)
    u = -float(qB @ AXIS)
    return float(np.clip(u, -prm.A / 2, prm.A / 2))


def points_from_offsets(d_bi: float, off: np.ndarray) -> np.ndarray:
    return bs_centre(d_bi)[None, :] + np.asarray(off)[:, None] * AXIS[None, :]


# ----------------------------------------------------------------------
# channels
# ----------------------------------------------------------------------
def los_bi(points: np.ndarray, E: np.ndarray, lam: float) -> np.ndarray:
    """NUSW LoS BS-IRS channel, Eq. (7)/(47).  Returns (M, n_points)."""
    D = np.linalg.norm(points[None, :, :] - E[:, None, :], axis=2)   # M x Lp
    return (lam / (4 * np.pi * D)) * np.exp(1j * 2 * np.pi * D / lam)


def sample_scatterers(d_bi: float, n: int, rng) -> np.ndarray:
    """[ASSUMED A4] uniform in a ball of radius d_bi/2 about the BS-IRS midpoint."""
    mid = bs_centre(d_bi) / 2.0
    R = d_bi / 2.0
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = R * rng.random(n) ** (1 / 3)
    return mid[None, :] + r[:, None] * v


def bi_channel(prm: Params, E: np.ndarray, points: np.ndarray,
               d_bi: float, L_paths: int, rng) -> np.ndarray:
    """BS-IRS channel Eq. (46) evaluated at every candidate MA position.

    L_paths == 0  ->  deterministic LoS channel Eq. (47)  [ASSUMED A7]
    """
    lam = prm.lam
    H_los = los_bi(points, E, lam)
    if L_paths == 0:
        return H_los

    n_tot = L_paths + 1
    l = (rng.normal(size=n_tot) + 1j * rng.normal(size=n_tot)) / np.sqrt(2 * n_tot)
    beta = lam / (4 * np.pi * d_bi)                      # [ASSUMED A3]
    H = l[0] * H_los
    S = sample_scatterers(d_bi, L_paths, rng)
    for p in range(L_paths):
        a = np.exp(1j * 2 * np.pi * np.linalg.norm(S[p] - E, axis=1) / lam)      # (48a)
        b = np.exp(1j * 2 * np.pi * np.linalg.norm(S[p] - points, axis=1) / lam)  # (48b)
        H += (l[p + 1] * beta) * np.outer(a, b)
    return H


def iu_channel(prm: Params, E: np.ndarray, d_user: float, rng) -> np.ndarray:
    """Rician IRS-user channel, Eq. (45) (with error E1 corrected)."""
    lam = prm.lam
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    v[0] = abs(v[0])                                     # [ASSUMED A5] x>0
    q = d_user * v
    los = np.exp(1j * 2 * np.pi * np.linalg.norm(q - E, axis=1) / lam)
    nlos = (rng.normal(size=E.shape[0]) + 1j * rng.normal(size=E.shape[0])) / np.sqrt(2)
    k = 10 ** (prm.rician_db / 10)
    small = np.sqrt(k / (1 + k)) * los + np.sqrt(1 / (1 + k)) * nlos
    return (lam / (4 * np.pi)) * d_user ** (-prm.pl_exp / 2) * small


# ----------------------------------------------------------------------
# common convention  (see error note E5)
# ----------------------------------------------------------------------
def _g2(h_iu: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Row vector g2^H = h_IU^H Phi, i.e. conj(h_IU) * phi."""
    return np.conj(h_iu) * phi


# ----------------------------------------------------------------------
# Algorithm 1, step 1: element-wise BCD for Phi  (Eqs. 22-23)
# ----------------------------------------------------------------------
def bcd_phi(G1: np.ndarray, phi: np.ndarray, n_sweep: int = 30,
            tol: float = 1e-9) -> np.ndarray:
    """Maximise || sum_m G1[m,:] e^{j phi_m} ||^2 by cyclic exact updates.

    G1[m,:] = conj(h_IU[m]) * H_BI[m,:]   (Sec. III-B.1)
    """
    phi = phi.copy()
    M = G1.shape[0]
    s = G1.T @ phi                                    # (N,)
    prev = float(np.vdot(s, s).real)
    for _ in range(n_sweep):
        for m in range(M):
            g = G1[m]
            a = s - g * phi[m]
            val = a @ np.conj(g)                      # alpha_m g_{1,m}^H
            if val == 0:
                continue
            phi[m] = val / abs(val)                   # Eq. (23)
            s = a + g * phi[m]
        cur = float(np.vdot(s, s).real)
        if cur - prev <= tol * max(prev, 1e-300):
            break
        prev = cur
    return phi


# ----------------------------------------------------------------------
# Algorithm 1, step 2: graph / DP position selection  (P2-3)
# ----------------------------------------------------------------------
def dp_select(w: np.ndarray, N: int, a_min: int):
    """Max sum of N weights at indices spaced >= a_min. Optimal, O(N*L)."""
    L = w.size
    NEG = -np.inf
    f = np.full((N + 1, L), NEG)
    par = np.full((N + 1, L), -1, dtype=int)
    f[1] = w
    for n in range(2, N + 1):
        run, arg = NEG, -1
        rmax = np.empty(L)
        rarg = np.empty(L, dtype=int)
        prev = f[n - 1]
        for l in range(L):
            if prev[l] > run:
                run, arg = prev[l], l
            rmax[l], rarg[l] = run, arg
        for l in range(a_min, L):
            if rmax[l - a_min] > NEG:
                f[n][l] = w[l] + rmax[l - a_min]
                par[n][l] = rarg[l - a_min]
    end = int(np.argmax(f[N]))
    best = float(f[N][end])
    idx = [end]
    cur = end
    for n in range(N, 1, -1):
        cur = int(par[n][cur])
        idx.append(cur)
    return np.array(idx[::-1]), best


# ----------------------------------------------------------------------
# single-user AO  (Algorithm 1) and the four benchmarks
# ----------------------------------------------------------------------
def su_gamma(h_iu, H_BI_sel, phi):
    """gamma = || h_IU^H Phi H_BI ||_2^2  (Eq. 4 with MRT, Eq. 20)."""
    v = _g2(h_iu, phi) @ H_BI_sel
    return float(np.vdot(v, v).real)


def su_solve(prm, h_iu, H_samp, phi0, idx0, a_min, opt_phi, opt_pos,
             n_outer=15, tol=1e-6):
    """AO between BCD (Phi) and DP (T). Skips a block if not optimised."""
    phi = phi0.copy()
    idx = np.asarray(idx0)
    N = len(idx)
    prev = -np.inf
    for _ in range(n_outer):
        if opt_phi:
            phi = bcd_phi(np.conj(h_iu)[:, None] * H_samp[:, idx], phi)
        if opt_pos:
            g2 = _g2(h_iu, phi)
            w = np.abs(g2 @ H_samp) ** 2
            idx, _ = dp_select(w, N, a_min)
        cur = su_gamma(h_iu, H_samp[:, idx], phi)
        if cur - prev <= tol * max(prev, 1e-300):
            break
        prev = cur
        if not (opt_phi and opt_pos):
            break                       # nothing left to alternate over
    return cur, phi, idx


# ----------------------------------------------------------------------
# WMMSE  (Eq. 35, error E3 corrected)
# ----------------------------------------------------------------------
def wmmse(H, P, sigma2, W0=None, iters=60, tol=1e-7):
    """H: (K,N) with rows h_k^H.  Returns W (N,K)."""
    K, N = H.shape
    if W0 is None:
        W = np.conj(H).T.copy()
        W *= np.sqrt(P / K) / np.linalg.norm(W, axis=0, keepdims=True)
    else:
        W = W0.copy()
    hc = np.conj(H)                                    # rows are h_k
    prev = -np.inf
    for _ in range(iters):
        F = H @ W                                      # F[k,i] = h_k^H w_i
        tot = np.sum(np.abs(F) ** 2, axis=1) + sigma2
        chi = np.diag(F) / tot                          # (35a)
        kap = 1.0 / np.maximum(1.0 - np.real(np.conj(chi) * np.diag(F)), 1e-12)  # (35b)
        c = kap * np.abs(chi) ** 2
        A = (hc.T * c) @ np.conj(hc)                    # sum_i c_i h_i h_i^H
        A = 0.5 * (A + A.conj().T)
        ev, U = np.linalg.eigh(A)
        # A = sum_i c_i h_i h_i^H has rank <= K and is singular whenever K < N,
        # so (A + mu I)^-1 blows up at mu = 0; floor the spectrum instead.
        ev = np.maximum(ev, 0.0)
        floor = 1e-15 * max(float(ev.max()), 1e-300)
        B = U.conj().T @ hc.T                           # (N,K)
        coef = kap * np.conj(chi)

        def power(mu):
            X = B / (ev[:, None] + mu + floor)
            return float(np.sum(np.abs(coef) ** 2 * np.sum(np.abs(X) ** 2, axis=0)))

        mu = 0.0
        if power(0.0) > P:
            lo, hi = 0.0, max(1e-12, ev.max())
            while power(hi) > P:
                hi *= 10
                if hi > 1e30:
                    break
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if power(mid) > P:
                    lo = mid
                else:
                    hi = mid
            mu = hi
        W = U @ (B / (ev[:, None] + mu + floor)) * coef[None, :]
        r = sum_rate(H @ W, sigma2)
        if np.isfinite(prev) and r - prev <= tol * max(abs(prev), 1e-12):
            break
        prev = r
    return W


def sum_rate(F, sigma2):
    """F[k,i] = h_k^H w_i  ->  sum_k R_k  (Eq. 2)."""
    p = np.abs(F) ** 2
    sig = np.diag(p)
    intf = p.sum(axis=1) - sig + sigma2
    return float(np.sum(np.log2(1.0 + sig / intf)))


# ----------------------------------------------------------------------
# Algorithm 2: CG on the complex circle manifold  (Eqs. 36-43, E4 corrected)
# ----------------------------------------------------------------------
def _f2_and_grad(phi, D, sigma2, want_grad=True):
    """f2 = -sum_k R_k with h_k^H w_i = D[k,i,:] @ phi.  Returns (f2, grad_E)."""
    F = np.tensordot(D, phi, axes=([2], [0]))          # (K,K)
    p = np.abs(F) ** 2
    sig = np.diag(p)
    T = p.sum(axis=1) + sigma2
    I = T - sig
    f2 = -float(np.sum(np.log2(T / I)))
    if not want_grad:
        return f2, None
    Dc = np.conj(D)
    num_all = np.einsum('ki,kim->m', F / T[:, None], Dc)
    Fi = F.copy()
    np.fill_diagonal(Fi, 0.0)
    num_int = np.einsum('ki,kim->m', Fi / I[:, None], Dc)
    grad = -2.0 / np.log(2.0) * (num_all - num_int)     # 2 * df2/dphi*
    return f2, grad


def manifold_cg(phi0, D, sigma2, iters=150, tol=1e-10):
    phi = phi0 / np.abs(phi0)
    f, g = _f2_and_grad(phi, D, sigma2)
    rg = g - np.real(g * np.conj(phi)) * phi            # Eq. (39)
    eta = -rg
    gn2 = float(np.vdot(rg, rg).real)
    # the gradient scale is arbitrary, so size the very first trial step so that
    # the largest phase perturbation is ~0.5 rad; afterwards reuse the last
    # accepted step (doubled), which is what keeps the line search cheap.
    a_prev = 0.5 / max(np.abs(eta).max(), 1e-300)
    for _ in range(iters):
        if np.sqrt(gn2) <= tol:
            break
        # Armijo backtracking with the retraction of Eq. (43) (base point kept)
        slope = float(np.real(np.vdot(rg, eta)))
        if slope >= 0:
            eta = -rg
            slope = -gn2
        a = 2.0 * a_prev
        ok = False
        for _ in range(40):
            cand = phi + a * eta
            cand = cand / np.abs(cand)
            fn, _ = _f2_and_grad(cand, D, sigma2, want_grad=False)
            if fn <= f + 1e-4 * a * slope:
                ok = True
                break
            a *= 0.5
        if not ok:
            break
        a_prev = a
        phi_new = cand
        f, g_new = _f2_and_grad(phi_new, D, sigma2)
        rg_new = g_new - np.real(g_new * np.conj(phi_new)) * phi_new
        tr = eta - np.real(eta * np.conj(phi_new)) * phi_new          # Eq. (41)
        rg_tr = rg - np.real(rg * np.conj(phi_new)) * phi_new
        gn2_new = float(np.vdot(rg_new, rg_new).real)
        tau = float(np.real(np.vdot(rg_new, rg_new - rg_tr))) / max(gn2, 1e-300)
        tau = max(tau, 0.0)                                           # PR+
        eta = -rg_new + tau * tr                                      # Eq. (42)
        phi, rg, gn2 = phi_new, rg_new, gn2_new
    return phi


# ----------------------------------------------------------------------
# Algorithm 3: discrete-sampling sequential search for T
# ----------------------------------------------------------------------
def discrete_sampling(C, W, sigma2, idx, a_min):
    """C[k,l] = g2_k^H h_BI(p_l);  W (N,K).  Sequential search over P_n."""
    idx = list(idx)
    N = W.shape[0]
    L = C.shape[1]
    for n in range(N):
        others = [idx[m] for m in range(N) if m != n]
        feas = np.ones(L, dtype=bool)
        for o in others:
            lo, hi = max(0, o - a_min + 1), min(L, o + a_min)
            feas[lo:hi] = False
        cand = np.flatnonzero(feas)
        if cand.size == 0:
            continue
        rows = [m for m in range(N) if m != n]
        F0 = C[:, others] @ W[rows, :]
        Fc = F0[None, :, :] + C[:, cand].T[:, :, None] * W[n][None, None, :]
        p = np.abs(Fc) ** 2
        sig = np.diagonal(p, axis1=1, axis2=2)
        intf = p.sum(axis=2) - sig + sigma2
        rates = np.sum(np.log2(1.0 + sig / intf), axis=1)
        idx[n] = int(cand[int(np.argmax(rates))])
    return np.array(idx)


# ----------------------------------------------------------------------
# Algorithm 4: multi-user AO
# ----------------------------------------------------------------------
def mu_solve(prm, H_iu, H_samp, phi0, idx0, a_min, opt_phi, opt_pos,
             n_outer=8, tol=1e-5):
    """H_iu: (K,M) rows h_IU,k.  H_samp: (M,L).  Returns (sum_rate, W, phi, idx)."""
    sigma2 = prm.sigma2
    phi = phi0.copy()
    idx = np.asarray(idx0)
    K = H_iu.shape[0]
    N = len(idx)

    # numerical conditioning: scale channels, scale noise by the same factor
    scale = np.abs(H_iu).mean() * np.abs(H_samp).mean() * prm.M
    W = None
    prev = -np.inf
    rate = 0.0
    for _ in range(n_outer):
        G2 = np.conj(H_iu) * phi[None, :]                 # (K,M)
        C = (G2 @ H_samp) / scale                         # (K,L)
        H = C[:, idx]                                     # (K,N) rows h_k^H
        s2 = sigma2 / scale ** 2
        W = wmmse(H, prm.P, s2, W0=W)
        if opt_phi:
            HBw = H_samp[:, idx] @ W                      # (M,K)
            D = (np.conj(H_iu)[:, None, :] * HBw.T[None, :, :]) / scale
            phi = manifold_cg(phi, D, s2)
            G2 = np.conj(H_iu) * phi[None, :]
            C = (G2 @ H_samp) / scale
        if opt_pos:
            idx = discrete_sampling(C, W, s2, idx, a_min)
        rate = sum_rate(C[:, idx] @ W, s2)
        if np.isfinite(prev) and rate - prev <= tol * max(abs(prev), 1e-12):
            break
        prev = rate
    return rate, W, phi, idx


# ======================================================================
#  correctness checks (independent of the paper's figures)
# ======================================================================
def hdr(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


# ----------------------------------------------------------------- 1. DP
def check_dp():
    hdr("1. Graph/DP position selection (P2-3) vs brute-force enumeration")
    rng = np.random.default_rng(0)
    bad = 0
    for trial in range(60):
        L = int(rng.integers(8, 26))
        N = int(rng.integers(2, 5))
        a = int(rng.integers(1, 4))
        if (N - 1) * a >= L:
            continue
        w = rng.random(L)
        idx, val = dp_select(w, N, a)
        best = -np.inf
        for c in itertools.combinations(range(L), N):
            if all(c[i + 1] - c[i] >= a for i in range(N - 1)):
                best = max(best, sum(w[list(c)]))
        ok = abs(best - val) < 1e-12 and abs(sum(w[idx]) - val) < 1e-12
        ok &= all(idx[i + 1] - idx[i] >= a for i in range(N - 1))
        bad += not ok
    print(f"   60 random configurations, mismatches: {bad}")
    return bad == 0


# ----------------------------------------------------------------- 2. BCD
def check_bcd():
    hdr("2. Element-wise BCD for Phi (Eq. 23)")
    rng = np.random.default_rng(1)
    # (a) N = 1: closed form optimum is (sum_m |g_m|)^2
    M = 200
    g = (rng.normal(size=(M, 1)) + 1j * rng.normal(size=(M, 1)))
    phi0 = np.exp(1j * rng.uniform(0, 2 * np.pi, M))
    phi = bcd_phi(g, phi0)
    got = float(np.abs(g.T @ phi).item() ** 2)
    exact = float(np.abs(g).sum() ** 2)
    print(f"   N=1 closed form : BCD/optimum = {got/exact:.10f}")
    # (b) N = 4: must beat 200 random draws
    N = 4
    G = rng.normal(size=(M, N)) + 1j * rng.normal(size=(M, N))
    phi = bcd_phi(G, np.exp(1j * rng.uniform(0, 2 * np.pi, M)))
    v = G.T @ phi
    val = float(np.vdot(v, v).real)
    def rand_val():
        u = G.T @ np.exp(1j * rng.uniform(0, 2 * np.pi, M))
        return float(np.vdot(u, u).real)
    rnd = max(rand_val() for _ in range(200))
    print(f"   N=4 BCD = {val:.4f}   best of 200 random = {rnd:.4f}")
    return abs(got / exact - 1) < 1e-9 and val > rnd


# ----------------------------------------- 3. Eq. (13)/(17): N=1 -> MA == FPA
def check_eq13():
    hdr("3. Eq. (13)/(17): with N=1 the MA can give no gain over an FPA at t*")
    prm = Params(N=1, My=25, Mz=25, snr_db=110)
    E = irs_elements(prm)
    off = offsets_fine(prm)
    a_min = int(round(prm.Dmin / prm.delta_s))
    ratios = []
    for d in [1.0, 3.0, 6.0]:
        rng = np.random.default_rng(10)
        pts = points_from_offsets(d, off)
        H = bi_channel(prm, E, pts, d, 0, rng)
        h = iu_channel(prm, E, 30.0, rng)
        phi0 = np.exp(1j * rng.uniform(0, 2 * np.pi, prm.M))
        i_fpa = int(np.argmin(np.abs(off - offset_eq13(prm, d))))
        g_ma, _, idx = su_solve(prm, h, H, phi0, [i_fpa], a_min, True, True)
        g_fp, _, _ = su_solve(prm, h, H, phi0, [i_fpa], a_min, True, False)
        ratios.append(g_ma / g_fp)
        print(f"   d_BI={d:>4.1f} m  MA/FPA = {g_ma/g_fp:.8f}   "
              f"MA index {idx[0]} (Eq.13 index {i_fpa})")
    return max(abs(r - 1) for r in ratios) < 1e-9


# --------------------------------------------------- 4. manifold gradient
def check_grad():
    hdr("4. Riemannian/Euclidean gradient of f2 (Eqs. 37-39) vs finite differences")
    rng = np.random.default_rng(3)
    K, M = 3, 60
    D = (rng.normal(size=(K, K, M)) + 1j * rng.normal(size=(K, K, M)))
    phi = np.exp(1j * rng.uniform(0, 2 * np.pi, M))
    s2 = 0.3
    f0, g = _f2_and_grad(phi, D, s2)
    eps = 1e-6
    err = []
    for _ in range(20):
        dv = (rng.normal(size=M) + 1j * rng.normal(size=M))
        fp, _ = _f2_and_grad(phi + eps * dv, D, s2, want_grad=False)
        fm, _ = _f2_and_grad(phi - eps * dv, D, s2, want_grad=False)
        num = (fp - fm) / (2 * eps)
        ana = float(np.real(np.vdot(g, dv)))
        err.append(abs(num - ana) / max(abs(num), 1e-12))
    print(f"   max relative error over 20 random directions: {max(err):.3e}")

    # the CG must beat random unit-modulus draws
    phi_opt = manifold_cg(phi, D, s2)
    f_opt, _ = _f2_and_grad(phi_opt, D, s2, want_grad=False)
    f_rnd = min(_f2_and_grad(np.exp(1j * rng.uniform(0, 2 * np.pi, M)), D, s2,
                                want_grad=False)[0] for _ in range(300))
    print(f"   f2: CG = {f_opt:.5f}   best of 300 random = {f_rnd:.5f}   "
          f"(start {f0:.5f})")
    print(f"   |phi| deviation from 1: {np.abs(np.abs(phi_opt)-1).max():.2e}")
    return max(err) < 1e-6 and f_opt < f_rnd


# --------------------------------------------------------------- 5. WMMSE
def check_wmmse():
    hdr("5. WMMSE (Eq. 35): power constraint + comparison with MRT / random")
    rng = np.random.default_rng(4)
    K, N = 3, 4
    H = (rng.normal(size=(K, N)) + 1j * rng.normal(size=(K, N))) / np.sqrt(2)
    P, s2 = 1.0, 0.05
    W = wmmse(H, P, s2)
    pw = float(np.sum(np.abs(W) ** 2))
    r = sum_rate(H @ W, s2)
    Wm = np.conj(H).T
    Wm *= np.sqrt(P / K) / np.linalg.norm(Wm, axis=0, keepdims=True)
    rm = sum_rate(H @ Wm, s2)
    rr = max(sum_rate(H @ (lambda X: X * np.sqrt(P) / np.linalg.norm(X))(
        rng.normal(size=(N, K)) + 1j * rng.normal(size=(N, K))), s2) for _ in range(300))
    print(f"   power = {pw:.6f} (budget {P})   WMMSE = {r:.4f}   "
          f"MRT = {rm:.4f}   best random = {rr:.4f} bps/Hz")
    return pw <= P * (1 + 1e-6) and r > rm and r > rr


# ------------------------------------------------- 6. Algorithm 4 monotone
def check_monotone():
    hdr("6. Algorithm 4: monotonic non-decreasing sum-rate")
    prm = Params()
    E = irs_elements(prm)
    off = offsets_fine(prm)
    a_min = int(round(prm.Dmin / prm.delta_s))
    rng = np.random.default_rng(7)
    d = 3.0
    pts = points_from_offsets(d, off)
    H_bi = bi_channel(prm, E, pts, d, 4, rng)
    H_iu = np.stack([iu_channel(prm, E, float(rng.uniform(30, 50)), rng)
                     for _ in range(prm.K)])
    phi0 = np.exp(1j * rng.uniform(0, 2 * np.pi, prm.M))
    idx0 = np.array([int(np.argmin(np.abs(off - o)))
                     for o in offsets_fpa(prm, prm.N)])
    seq = []
    for n in range(1, 9):
        r, *_ = mu_solve(prm, H_iu, H_bi, phi0, idx0, a_min, True, True,
                            n_outer=n, tol=-1.0)
        seq.append(r)
    print("   sum-rate per outer iteration: " +
          " ".join(f"{x:.4f}" for x in seq))
    drops = sum(1 for i in range(1, len(seq)) if seq[i] < seq[i - 1] - 1e-9)
    print(f"   decreases: {drops}")
    return drops == 0


# ------------------------------------ 7. Eq. (31): far-field gain vanishes
def check_farfield():
    hdr("7. Eq. (31): MA gain over FPA must vanish in the far-field LoS channel")
    prm = Params()
    E = irs_elements(prm)
    off = offsets_fine(prm)
    a_min = int(round(prm.Dmin / prm.delta_s))
    idx_fpa = np.array([int(np.argmin(np.abs(off - o)))
                        for o in offsets_fpa(prm, prm.N)])
    idx_spr = np.linspace(0, len(off) - 1, prm.N).round().astype(int)
    # d = 1 m is a near-field CONTROL: Eq. (31) only forces the gap to zero in
    # the far field, so a test that samples the far field alone would also pass
    # if the position search were broken and never moved at all.
    print("   d_BI [m]     proposed      FPA        relative gap")
    gaps = []
    for d in [1.0, 5.0, 50.0, 500.0, 5000.0]:
        rng = np.random.default_rng(11)
        pts = points_from_offsets(d, off)
        H_bi = bi_channel(prm, E, pts, d, 0, rng)
        H_iu = np.stack([iu_channel(prm, E, float(rng.uniform(30, 50)), rng)
                         for _ in range(prm.K)])
        phi0 = np.exp(1j * rng.uniform(0, 2 * np.pi, prm.M))
        r_ma = _mu_best(prm, H_iu, H_bi, phi0, [idx_fpa, idx_spr],
                        a_min, True, True)
        r_fp = mu_solve(prm, H_iu, H_bi, phi0, idx_fpa, a_min, True, False)[0]
        g = (r_ma - r_fp) / r_fp
        gaps.append(g)
        tag = "  <- near-field control, must be > 0" if d == 1.0 else ""
        print(f"   {d:>9.0f}   {r_ma:9.5f}  {r_fp:9.5f}   {g:.3e}{tag}")
    return gaps[0] > 1e-6 and max(gaps[-2:]) < 1e-6

def run_validation():
    res = {
        "DP vs brute force": check_dp(),
        "BCD vs closed form / random": check_bcd(),
        "Eq. (13)/(17) N=1 identity": check_eq13(),
        "manifold gradient vs FD": check_grad(),
        "WMMSE": check_wmmse(),
        "Algorithm 4 monotonicity": check_monotone(),
        "Eq. (31) far-field vanishing": check_farfield(),
    }
    hdr("SUMMARY")
    for k, v in res.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
    return all(res.values())


# ======================================================================
#  scheme setup helpers
# ======================================================================
def grids(prm, N, fpa_offsets=None):
    """Sampling grids and starting index sets for each scheme.

    The starting layouts ALWAYS include the FPA benchmark layout: that layout is
    a feasible MA layout, so a correct solution of (P1)/(P2) can never come out
    below the FPA benchmark.  Seeding the search elsewhere alone does not give
    that guarantee -- both the graph-based step (single user) and the sequential
    search (multi-user) are only locally optimal for a GIVEN Phi, and Phi is
    phase-matched to whatever layout it was last optimised for.
    """
    off_f = offsets_fine(prm)
    off_a = offsets_as(prm)
    off_p = offsets_fpa(prm, N) if fpa_offsets is None else np.asarray(fpa_offsets)
    a_fine = int(round(prm.Dmin / prm.delta_s))
    a_as = 1
    i_fpa_f = np.array([int(np.argmin(np.abs(off_f - o))) for o in off_p])
    i_fpa_a = np.array([int(np.argmin(np.abs(off_a - o))) for o in off_p])
    i_spr_f = np.linspace(0, len(off_f) - 1, N).round().astype(int)
    i_spr_a = np.linspace(0, len(off_a) - 1, N).round().astype(int)
    return dict(off_f=off_f, off_a=off_a, off_p=off_p, a_fine=a_fine, a_as=a_as,
                i_fpa_f=i_fpa_f, i_fpa_a=i_fpa_a, i_spr_f=i_spr_f, i_spr_a=i_spr_a)

def draw(prm, E, d, L_paths, off_list, rng, K):
    """One channel realisation, evaluated on every sampling grid."""
    pts = [points_from_offsets(d, o) for o in off_list]
    n = [len(o) for o in off_list]
    allpts = np.concatenate(pts, axis=0)
    H = bi_channel(prm, E, allpts, d, L_paths, rng)
    out, s = [], 0
    for c in n:
        out.append(H[:, s:s + c])
        s += c
    H_iu = np.stack([iu_channel(prm, E, float(rng.uniform(*prm.d_iu)), rng)
                     for _ in range(K)]) if K > 1 else \
        iu_channel(prm, E, prm.d_iu[0], rng)
    return out, H_iu


# ----------------------------------------------------------------------

def _su_best(prm, h_iu, H, phi, starts, a_min, opt_phi, opt_pos):
    best = -np.inf
    for i0 in starts:
        best = max(best, su_solve(prm, h_iu, H, phi, i0, a_min,
                                     opt_phi, opt_pos)[0])
    return best

def _mu_best(prm, H_iu, H, phi, starts, a_min, opt_phi, opt_pos):
    best = -np.inf
    for i0 in starts:
        r = mu_solve(prm, H_iu, H, phi, i0, a_min, opt_phi, opt_pos)[0]
        best = max(best, r)
    return best

def accumulate(path, new, n_new, extra=None):
    """Merge a chunk of trials into a running sum on disk.

    Stored as raw sums plus a trial count so that a long Monte-Carlo run can be
    split over as many invocations as needed and still give the exact same
    average as one long run (trial t always uses seed (seed, di, t)).
    """
    tot, n = {}, 0
    if os.path.exists(path):
        z = np.load(path)
        n = int(z["count"])
        tot = {k: z[k] for k in z.files if k.startswith("sum_")}
    for k, v in new.items():
        tot["sum_" + k] = tot.get("sum_" + k, 0.0) + v * n_new
    n += n_new
    np.savez(path, count=n, **tot, **(extra or {}))
    return n

# ======================================================================
#  Monte-Carlo drivers  (one trial = one independent channel realisation)
# ======================================================================
SCHEMES = ["proposed", "as", "fpa", "ma_rps", "fpa_rps"]


def su_trial(task):
    """One single-user realisation (Figs. 2 and 3). Returns linear SNR gamma."""
    prm, d, L_paths, N, fpa_offsets, seed, di, t = task
    E = irs_elements(prm)
    g = grids(prm, N, fpa_offsets)
    rng = np.random.default_rng((seed, di, t))
    (Hf, Ha, Hp), h_iu = draw(prm, E, d, L_paths,
                              [g["off_f"], g["off_a"], g["off_p"]], rng, 1)
    phi0 = np.exp(1j * rng.uniform(0, 2 * np.pi, prm.M))
    phi_r = np.exp(1j * rng.uniform(0, 2 * np.pi, prm.M))
    # the FPA layout is itself a feasible MA layout, so it is always a start
    sf = [g["i_fpa_f"]] if N == 1 else [g["i_fpa_f"], g["i_spr_f"]]
    sa = [g["i_fpa_a"]] if N == 1 else [g["i_fpa_a"], g["i_spr_a"]]
    i_all = np.arange(N)
    return {
        "proposed": _su_best(prm, h_iu, Hf, phi0, sf, g["a_fine"], True, True),
        "as":       _su_best(prm, h_iu, Ha, phi0, sa, g["a_as"], True, True),
        "fpa":      su_solve(prm, h_iu, Hp, phi0, i_all, 1, True, False)[0],
        "ma_rps":   _su_best(prm, h_iu, Hf, phi_r, sf, g["a_fine"], False, True),
        "fpa_rps":  su_gamma(h_iu, Hp, phi_r),
    }


def mu_trial(task):
    """One multi-user realisation (Figs. 6 and 8). Returns sum-rate in bps/Hz."""
    prm, d, L_paths, init, seed, di, t = task
    E = irs_elements(prm)
    N = prm.N
    g = grids(prm, N)
    rng = np.random.default_rng((seed, di, t))
    (Hf, Ha, Hp), H_iu = draw(prm, E, d, L_paths,
                              [g["off_f"], g["off_a"], g["off_p"]], rng, prm.K)
    phi0 = np.exp(1j * rng.uniform(0, 2 * np.pi, prm.M))
    phi_r = np.exp(1j * rng.uniform(0, 2 * np.pi, prm.M))
    if init == "compact":
        sf, sa = [g["i_fpa_f"]], [g["i_fpa_a"]]
    elif init == "spread":
        sf, sa = [g["i_spr_f"]], [g["i_spr_a"]]
    else:
        sf = [g["i_fpa_f"], g["i_spr_f"]]
        sa = [g["i_fpa_a"], g["i_spr_a"]]
    i_all = np.arange(N)
    return {
        "proposed": _mu_best(prm, H_iu, Hf, phi0, sf, g["a_fine"], True, True),
        "as":       _mu_best(prm, H_iu, Ha, phi0, sa, g["a_as"], True, True),
        "fpa":      mu_solve(prm, H_iu, Hp, phi0, i_all, 1, True, False)[0],
        "ma_rps":   _mu_best(prm, H_iu, Hf, phi_r, sf, g["a_fine"], False, True),
        "fpa_rps":  mu_solve(prm, H_iu, Hp, phi_r, i_all, 1, False, False)[0],
    }


def sweep(fn, tasks_per_d, dists, trials, jobs, label, post=None):
    """Average `fn` over `trials` realisations at every distance.

    `post` maps a single trial's result before accumulation, so that a
    non-linear metric is averaged per realisation -- E[log2(1+g)] is NOT
    log2(1+E[g]), and averaging in the wrong order inflates every rate curve.
    """
    acc = {s: np.zeros(len(dists)) for s in SCHEMES}
    pool = Pool(jobs) if jobs > 1 else None
    try:
        for di, d in enumerate(dists):
            tasks = tasks_per_d(di, d)
            t0 = time.time()
            res = pool.map(fn, tasks) if pool else [fn(x) for x in tasks]
            for r in res:
                if post:
                    r = post(r)
                for s in SCHEMES:
                    acc[s][di] += r[s]
            print(f"    {label}  d_BI = {d:>4.1f} m  ({trials} trials, "
                  f"{time.time()-t0:5.1f} s)", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    return {s: acc[s] / trials for s in SCHEMES}


# ======================================================================
#  figures
# ======================================================================
STYLE = {
    "proposed": dict(label="Proposed", color="#d62728", marker="o", ls="-"),
    "as":       dict(label="AS",       color="#ff7f0e", marker="s", ls="-"),
    "fpa":      dict(label="FPA",      color="#2ca02c", marker="D", ls="--"),
    "ma_rps":   dict(label="MA-RPS",   color="#1f77b4", marker="^", ls="-"),
    "fpa_rps":  dict(label="FPA-RPS",  color="#e377c2", marker="v", ls="--"),
}


def _decorate(ax, ylabel):
    ax.set_xlabel("BS-IRS Distance (m)")
    ax.set_ylabel(ylabel)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.legend(fontsize=8, framealpha=0.95)


def plot_fig2(path, outdir):
    z = load_sums(path)
    d, n = z["dists"], z["count"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, N, tag in zip(axes, (1, 4), ("(a) $N=1$", "(b) $N=4$")):
        ax.plot(d, 10 * np.log10(z[f"N{N}_proposed"]), "-s", color="#d62728",
                ms=5, label="MA")
        ax.plot(d, 10 * np.log10(z[f"N{N}_fpa"]), "--*", color="#1f77b4",
                ms=7, label="FPA")
        _decorate(ax, "User's received SNR (dB)")
        ax.set_title(tag, fontsize=10)
    fig.suptitle(f"Fig. 2  Received SNR at the user, LoS BS-IRS channel "
                 f"($M=25^2$, $P/\\sigma^2=110$ dB, {n} realisations)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(outdir, "fig2_received_snr.png")
    fig.savefig(p, dpi=170)
    e = np.abs(10 * np.log10(z["N1_proposed"] / z["N1_fpa"])).max()
    gap = 10 * np.log10(z["N4_proposed"] / z["N4_fpa"])
    print(f"  Fig. 2  N=1 max |MA-FPA| = {e:.2e} dB  (Eq. 13/17 requires exactly 0)")
    print(f"          N=4 MA gain {gap[0]:.2f} dB at 1 m -> {gap[-1]:.2f} dB at 6 m")
    return p


def plot_curves(path, title, fname, ylabel, outdir, at=3.0):
    z = load_sums(path)
    d, n = z["dists"], z["count"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for s in SCHEMES:
        ax.plot(d, z[s], ms=5, lw=1.4, **STYLE[s])
    i = int(np.argmin(np.abs(d - at)))
    g_opt = z["proposed"][i] - z["fpa"][i]
    g_rnd = z["ma_rps"][i] - z["fpa_rps"][i]
    ax.annotate(f"MA gain, optimised IRS\n{g_opt:.2f} bps/Hz "
                f"({100*g_opt/z['fpa'][i]:.1f}%)",
                xy=(d[i], z["fpa"][i]), xytext=(d[i] + 1.6, z["fpa"][i] + 1.4),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate(f"MA gain, random IRS\n{g_rnd:.2f} bps/Hz "
                f"({100*g_rnd/z['fpa_rps'][i]:.1f}%)",
                xy=(d[i], z["fpa_rps"][i]), xytext=(d[i] + 1.6, z["fpa_rps"][i] + 1.6),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.8))
    _decorate(ax, ylabel)
    ax.set_title(f"{title}  [{n} realisations]", fontsize=10)
    fig.tight_layout()
    p = os.path.join(outdir, fname)
    fig.savefig(p, dpi=170)
    print(f"  {title}\n     at {at:.0f} m: proposed-FPA = {g_opt:.3f} bps/Hz "
          f"({100*g_opt/z['fpa'][i]:.2f}%),  MA-RPS-FPA-RPS = {g_rnd:.3f} bps/Hz "
          f"({100*g_rnd/z['fpa_rps'][i]:.2f}%)")
    return p


def load_sums(path):
    z = np.load(path)
    n = int(z["count"])
    out = {k[4:]: z[k] / n for k in z.files if k.startswith("sum_")}
    out["dists"] = z["dists"]
    out["count"] = n
    return out


# ======================================================================
#  main
# ======================================================================
def build_fig2(trials, t0, jobs, outdir):
    dists = np.arange(1.0, 7.0)                      # Fig. 2 sweeps 1..6 m
    new = {}
    for N in (1, 4):
        # Sec. III-A.1 overrides: M = 25^2, P/sigma^2 = 110 dB, LoS channel
        prm = Params(N=N, My=25, Mz=25, snr_db=110, d_iu=(30.0, 30.0))
        fpa = [offset_eq13(prm, 1.0)] if N == 1 else None
        res = sweep(su_trial,
                    lambda di, d, prm=prm, N=N, fpa=fpa:
                        [(prm, d, 0, N, fpa, 0, di, t) for t in range(t0, t0 + trials)],
                    dists, trials, jobs, f"fig2 N={N}")
        for k in ("proposed", "fpa"):
            new[f"N{N}_{k}"] = prm.p_over_sigma2 * res[k]
    p = os.path.join(outdir, "sums_fig2.npz")
    accumulate(p, new, trials, extra={"dists": dists})
    return p


def build_fig3(trials, t0, jobs, outdir):
    prm = Params(d_iu=(30.0, 30.0))                  # Fig. 3: d_IU = 30 m, L = 8
    dists = np.arange(1.0, 11.0)
    ps = prm.p_over_sigma2
    rate = sweep(su_trial,
                 lambda di, d: [(prm, d, 8, prm.N, None, 0, di, t)
                                for t in range(t0, t0 + trials)],
                 dists, trials, jobs, "fig3",
                 post=lambda r: {s: np.log2(1.0 + ps * r[s]) for s in SCHEMES})
    p = os.path.join(outdir, "sums_fig3.npz")
    accumulate(p, rate, trials, extra={"dists": dists})
    return p


def build_mu(fig, trials, t0, jobs, init, outdir):
    prm = Params()
    dists = np.arange(1.0, 11.0)
    L = 0 if fig == 6 else 4                         # Fig. 6 LoS, Fig. 8 L = 4
    res = sweep(mu_trial,
                lambda di, d: [(prm, d, L, init, 0, di, t)
                               for t in range(t0, t0 + trials)],
                dists, trials, jobs, f"fig{fig}")
    p = os.path.join(outdir, f"sums_fig{fig}_{init}.npz")
    accumulate(p, res, trials, extra={"dists": dists})
    return p


def main():
    ap = argparse.ArgumentParser(
        description="Reproduce Figs. 2, 3, 6 and 8 of Wei et al., IEEE TCOM 2025.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python3 ma_irs_reproduce.py --quick            fast smoke run\n"
               "  python3 ma_irs_reproduce.py --trials 1000      paper's sample size\n"
               "  python3 ma_irs_reproduce.py --fig 6 8 --jobs 8\n"
               "  python3 ma_irs_reproduce.py --trials 500 --t0 1000   resume/extend\n"
               "  python3 ma_irs_reproduce.py --plot-only\n"
               "  python3 ma_irs_reproduce.py --validate-only\n")
    ap.add_argument("--fig", type=int, nargs="+", default=[2, 3, 6, 8],
                    choices=[2, 3, 6, 8], help="which figures to produce")
    ap.add_argument("--trials", type=int, default=100,
                    help="channel realisations per point (paper uses 1000)")
    ap.add_argument("--t0", type=int, default=0,
                    help="first trial index; use to extend an existing run")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 1),
                    help="parallel worker processes")
    ap.add_argument("--init", default="best", choices=["best", "compact", "spread"],
                    help="initialisation of Algorithm 3 (multi-user position search)")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the correctness checks")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="10 trials, for a fast end-to-end check")
    ap.add_argument("--plot-only", action="store_true",
                    help="re-render figures from existing results, no simulation")
    ap.add_argument("--no-show", action="store_true",
                    help="write the PNGs but do not open figure windows")
    a = ap.parse_args()

    if a.quick:
        a.trials = 10
    os.makedirs(a.outdir, exist_ok=True)
    show = (not a.no_show) and _enable_display()
    t_start = time.time()

    if not a.no_validate:
        ok = run_validation()
        print(f"\n   validation: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
        if a.validate_only:
            return
        if not ok:
            print("   refusing to produce figures from code that fails its checks")
            return
    elif a.validate_only:
        return

    made = []
    if a.plot_only:
        for f in a.fig:
            name = f"sums_fig{f}.npz" if f in (2, 3) else f"sums_fig{f}_{a.init}.npz"
            path = os.path.join(a.outdir, name)
            if os.path.exists(path):
                made.append((f"fig{f}", path))
            else:
                print(f"   no results for fig {f} at {path} -- skipping")
    else:
        hdr(f"SIMULATION  ({a.trials} realisations/point, {a.jobs} worker(s))")
        for f in a.fig:
            if f == 2:
                made.append(("fig2", build_fig2(a.trials, a.t0, a.jobs, a.outdir)))
            elif f == 3:
                made.append(("fig3", build_fig3(a.trials, a.t0, a.jobs, a.outdir)))
            else:
                made.append((f"fig{f}",
                             build_mu(f, a.trials, a.t0, a.jobs, a.init, a.outdir)))

    hdr("FIGURES")
    files = []
    for tag, path in made:
        if tag == "fig2":
            files.append(plot_fig2(path, a.outdir))
        elif tag == "fig3":
            files.append(plot_curves(
                path, "Fig. 3  Achievable rate, single user, multi-path ($L=8$)",
                "fig3_single_user_rate.png", "User's achievable rate (bps/Hz)",
                a.outdir))
        else:
            n = tag[3:]
            lab = "LoS BS-IRS channel" if n == "6" else "multi-path BS-IRS channel ($L=4$)"
            files.append(plot_curves(
                path, f"Fig. {n}  Sum-rate, {lab}\n[Algorithm 3 init: {a.init}]",
                f"fig{n}_sum_rate_{a.init}.png", "Sum-rate (bps/Hz)", a.outdir))

    print("\n   written:")
    for f in files:
        print("     " + f)
    print(f"\n   total elapsed {time.time()-t_start:.1f} s")

    if not show:
        plt.close("all")
        if not a.no_show:
            print("   no usable display (headless), so the figures were only "
                  "saved as PNGs;\n   set MPLBACKEND or run with a display to "
                  "see them on screen")
    else:
        print(f"   displaying {len(files)} figure(s) "
              f"[{matplotlib.get_backend()}] -- close the windows to exit")
        try:
            plt.show()
        except Exception as exc:
            # a backend can accept use() and still fail to open a window; the
            # PNGs are already on disk, so never let that lose a long run
            plt.close("all")
            print(f"   could not open a window ({exc.__class__.__name__}: {exc});"
                  f"\n   the PNGs above were still written")



if __name__ == "__main__":
    main()
