#!/usr/bin/env python3

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

matplotlib.use("Agg")      
import matplotlib.pyplot as plt

_NON_GUI = ("agg", "pdf", "ps", "svg", "template")
C_LIGHT = 3e8


def _enable_display():
    forced = os.environ.get("MPLBACKEND")
    if forced:
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



class Params:
    fc: float = 5e9
    N: int = 4
    My: int = 15
    Mz: int = 15
    A: float = 0.6               
    K: int = 3
    P_dbm: float = 46.0
    sigma2_dbm: float = -80.0
    rician_db: float = 3.0
    pl_exp: float = 2.8
    d_iu: tuple = (30.0, 50.0)   
    snr_db: float | None = None  # Fig. 2 fixes P/sigma^2 = 110 dB

    @property
    def lam(self):
        return C_LIGHT / self.fc

    @property
    def d_irs(self):
        return self.lam / 2

    @property
    def Dmin(self):
        return self.lam / 2

    @property
    def delta_s(self):
        return self.lam / 10

    @property
    def M(self):
        return self.My * self.Mz

    @property
    def P(self):
        return 10 ** ((self.P_dbm - 30) / 10)

    @property
    def sigma2(self):
        return 10 ** ((self.sigma2_dbm - 30) / 10)

    @property
    def p_over_sigma2(self):
        if self.snr_db is not None:
            return 10 ** (self.snr_db / 10)
        return self.P / self.sigma2

def irs_elements(prm):
    # UPA in the yOz plane centred at the origin, shape (M, 3)
    my = (np.arange(prm.My) - (prm.My - 1) / 2) * prm.d_irs
    mz = (np.arange(prm.Mz) - (prm.Mz - 1) / 2) * prm.d_irs
    Y, Z = np.meshgrid(my, mz, indexing="ij")
    return np.stack([np.zeros(Y.size), Y.ravel(), Z.ravel()], axis=1)


def bs_centre(d_bi):
    return d_bi * np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0.0])  


AXIS = np.array([1.0, 0.0, 0.0])                                      


def offsets_fine(prm):
    # candidate MA positions, spacing delta_s, endpoints included (101 pts)
    L = int(round(prm.A / prm.delta_s)) + 1
    return np.linspace(-prm.A / 2, prm.A / 2, L)


def offsets_as(prm):
    # antenna selection benchmark: A/Dmin = 20 fixed antennas
    n = int(round(prm.A / prm.Dmin))
    return (np.arange(n) - (n - 1) / 2) * prm.Dmin


def offsets_fpa(prm, N):
    return (np.arange(N) - (N - 1) / 2) * prm.Dmin


def offset_eq13(prm, d_bi):
    # Eq. (13): single MA goes to the point of C_t closest to the IRS
    u = -float(bs_centre(d_bi) @ AXIS)
    return float(np.clip(u, -prm.A / 2, prm.A / 2))


def points_from_offsets(d_bi, off):
    return bs_centre(d_bi)[None, :] + np.asarray(off)[:, None] * AXIS[None, :]


def los_bi(points, E, lam):
    # near-field (NUSW) LoS, Eq. (7)/(47), shape (M, n_points)
    D = np.linalg.norm(points[None, :, :] - E[:, None, :], axis=2)
    return (lam / (4 * np.pi * D)) * np.exp(1j * 2 * np.pi * D / lam)


def sample_scatterers(d_bi, n, rng):
    
    mid = bs_centre(d_bi) / 2.0
    R = d_bi / 2.0
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = R * rng.random(n) ** (1 / 3)
    return mid[None, :] + r[:, None] * v


def bi_channel(prm, E, points, d_bi, L_paths, rng):
    lam = prm.lam
    H_los = los_bi(points, E, lam)
    if L_paths == 0:
        return H_los

    n_tot = L_paths + 1
    l = (rng.normal(size=n_tot) + 1j * rng.normal(size=n_tot)) / np.sqrt(2 * n_tot)
    beta = lam / (4 * np.pi * d_bi)                                  
    H = l[0] * H_los
    S = sample_scatterers(d_bi, L_paths, rng)
    for p in range(L_paths):
        a = np.exp(1j * 2 * np.pi * np.linalg.norm(S[p] - E, axis=1) / lam)
        b = np.exp(1j * 2 * np.pi * np.linalg.norm(S[p] - points, axis=1) / lam)
        H += (l[p + 1] * beta) * np.outer(a, b)
    return H


def iu_channel(prm, E, d_user, rng):
    # Rician IRS-user channel, Eq. (45)
    lam = prm.lam
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    v[0] = abs(v[0])                                                 
    q = d_user * v
    los = np.exp(1j * 2 * np.pi * np.linalg.norm(q - E, axis=1) / lam)
    nlos = (rng.normal(size=E.shape[0]) + 1j * rng.normal(size=E.shape[0])) / np.sqrt(2)
    k = 10 ** (prm.rician_db / 10)
    small = np.sqrt(k / (1 + k)) * los + np.sqrt(1 / (1 + k)) * nlos
    return (lam / (4 * np.pi)) * d_user ** (-prm.pl_exp / 2) * small


def _g2(h_iu, phi):
    # g2^H = h_IU^H Phi
    return np.conj(h_iu) * phi



def bcd_phi(G1, phi, n_sweep=30, tol=1e-9):
    phi = phi.copy()
    M = G1.shape[0]
    s = G1.T @ phi
    prev = float(np.vdot(s, s).real)
    for _ in range(n_sweep):
        for m in range(M):
            g = G1[m]
            a = s - g * phi[m]
            val = a @ np.conj(g)
            if val == 0:
                continue
            phi[m] = val / abs(val)
            s = a + g * phi[m]
        cur = float(np.vdot(s, s).real)
        if cur - prev <= tol * max(prev, 1e-300):
            break
        prev = cur
    return phi


def dp_select(w, N, a_min):
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


def su_gamma(h_iu, H_BI_sel, phi):
    # Eq. (20): ||h_IU^H Phi H_BI||^2 with MRT
    v = _g2(h_iu, phi) @ H_BI_sel
    return float(np.vdot(v, v).real)


def su_solve(prm, h_iu, H_samp, phi0, idx0, a_min, opt_phi, opt_pos,
             n_outer=15, tol=1e-6):
    phi = phi0.copy()
    idx = np.asarray(idx0)
    N = len(idx)
    prev = -np.inf
    for _ in range(n_outer):
        if opt_phi:
            phi = bcd_phi(np.conj(h_iu)[:, None] * H_samp[:, idx], phi)
        if opt_pos:
            w = np.abs(_g2(h_iu, phi) @ H_samp) ** 2
            idx, _ = dp_select(w, N, a_min)
        cur = su_gamma(h_iu, H_samp[:, idx], phi)
        if cur - prev <= tol * max(prev, 1e-300):
            break
        prev = cur
        if not (opt_phi and opt_pos):
            break   # only one block, nothing to alternate
    return cur, phi, idx



def wmmse(H, P, sigma2, W0=None, iters=60, tol=1e-7):
    # H is (K, N), rows h_k^H. returns W (N, K)
    K, N = H.shape
    if W0 is None:
        W = np.conj(H).T.copy()
        W *= np.sqrt(P / K) / np.linalg.norm(W, axis=0, keepdims=True)
    else:
        W = W0.copy()
    hc = np.conj(H)
    prev = -np.inf
    for _ in range(iters):
        F = H @ W
        tot = np.sum(np.abs(F) ** 2, axis=1) + sigma2
        chi = np.diag(F) / tot
        kap = 1.0 / np.maximum(1.0 - np.real(np.conj(chi) * np.diag(F)), 1e-12)
        c = kap * np.abs(chi) ** 2
        A = (hc.T * c) @ np.conj(hc)
        A = 0.5 * (A + A.conj().T)
        ev, U = np.linalg.eigh(A)
        # A has rank <= K, singular when K < N -> floor the eigenvalues
        ev = np.maximum(ev, 0.0)
        floor = 1e-15 * max(float(ev.max()), 1e-300)
        B = U.conj().T @ hc.T
        coef = kap * np.conj(chi)

        def power(mu):
            X = B / (ev[:, None] + mu + floor)
            return float(np.sum(np.abs(coef) ** 2 * np.sum(np.abs(X) ** 2, axis=0)))

        # bisection on the power multiplier
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
    # F[k, i] = h_k^H w_i
    p = np.abs(F) ** 2
    sig = np.diag(p)
    intf = p.sum(axis=1) - sig + sigma2
    return float(np.sum(np.log2(1.0 + sig / intf)))



def _f2_and_grad(phi, D, sigma2, want_grad=True):
    # f2 = -sum rate, with h_k^H w_i = D[k, i, :] @ phi
    F = np.tensordot(D, phi, axes=([2], [0]))
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
    grad = -2.0 / np.log(2.0) * (num_all - num_int)
    return f2, grad


def manifold_cg(phi0, D, sigma2, iters=150, tol=1e-10):
    phi = phi0 / np.abs(phi0)
    f, g = _f2_and_grad(phi, D, sigma2)
    rg = g - np.real(g * np.conj(phi)) * phi          # Riemannian grad (39)
    eta = -rg
    gn2 = float(np.vdot(rg, rg).real)
    # first step: max phase change ~0.5 rad. after that start from 2x the
    # last accepted step
    a_prev = 0.5 / max(np.abs(eta).max(), 1e-300)
    for _ in range(iters):
        if np.sqrt(gn2) <= tol:
            break
        slope = float(np.real(np.vdot(rg, eta)))
        if slope >= 0:
            eta = -rg
            slope = -gn2
        # Armijo backtracking + retraction (43)
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
        tr = eta - np.real(eta * np.conj(phi_new)) * phi_new      
        rg_tr = rg - np.real(rg * np.conj(phi_new)) * phi_new
        gn2_new = float(np.vdot(rg_new, rg_new).real)
        tau = float(np.real(np.vdot(rg_new, rg_new - rg_tr))) / max(gn2, 1e-300)
        tau = max(tau, 0.0)                                      
        eta = -rg_new + tau * tr
        phi, rg, gn2 = phi_new, rg_new, gn2_new
    return phi

def discrete_sampling(C, W, sigma2, idx, a_min):
    # C[k, l] = g2_k^H h_BI(p_l)
    idx = list(idx)
    N = W.shape[0]
    L = C.shape[1]
    for n in range(N):
        others = [idx[m] for m in range(N) if m != n]
        feas = np.ones(L, dtype=bool)
        for o in others:
            feas[max(0, o - a_min + 1):min(L, o + a_min)] = False
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

def mu_solve(prm, H_iu, H_samp, phi0, idx0, a_min, opt_phi, opt_pos,
             n_outer=8, tol=1e-5):
    # H_iu (K, M), H_samp (M, L). returns (sum_rate, W, phi, idx)
    sigma2 = prm.sigma2
    phi = phi0.copy()
    idx = np.asarray(idx0)

    # rescale channels (and noise) so the numbers are O(1)
    scale = np.abs(H_iu).mean() * np.abs(H_samp).mean() * prm.M
    s2 = sigma2 / scale ** 2
    W = None
    prev = -np.inf
    rate = 0.0
    for _ in range(n_outer):
        G2 = np.conj(H_iu) * phi[None, :]
        C = (G2 @ H_samp) / scale
        W = wmmse(C[:, idx], prm.P, s2, W0=W)
        if opt_phi:
            HBw = H_samp[:, idx] @ W
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

def hdr(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


def check_dp():
    hdr("1. DP position selection vs brute force")
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(60):
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
    print(f"   60 random cases, mismatches: {bad}")
    return bad == 0


def check_bcd():
    hdr("2. BCD for Phi (Eq. 23)")
    rng = np.random.default_rng(1)
    M = 200
    # N = 1 has a closed form: (sum |g_m|)^2
    g = rng.normal(size=(M, 1)) + 1j * rng.normal(size=(M, 1))
    phi = bcd_phi(g, np.exp(1j * rng.uniform(0, 2 * np.pi, M)))
    got = float(np.abs(g.T @ phi).item() ** 2)
    exact = float(np.abs(g).sum() ** 2)
    print(f"   N=1 : BCD/optimum = {got/exact:.10f}")
    # N = 4 should beat random phases
    N = 4
    G = rng.normal(size=(M, N)) + 1j * rng.normal(size=(M, N))
    phi = bcd_phi(G, np.exp(1j * rng.uniform(0, 2 * np.pi, M)))
    v = G.T @ phi
    val = float(np.vdot(v, v).real)

    def rand_val():
        u = G.T @ np.exp(1j * rng.uniform(0, 2 * np.pi, M))
        return float(np.vdot(u, u).real)

    rnd = max(rand_val() for _ in range(200))
    print(f"   N=4 : BCD = {val:.4f}   best of 200 random = {rnd:.4f}")
    return abs(got / exact - 1) < 1e-9 and val > rnd

def check_eq13():
    hdr("3. Eq. (13)/(17): N=1 MA should give exactly the FPA result")
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


def check_grad():
    hdr("4. gradient of f2 vs finite differences, and CG vs random")
    rng = np.random.default_rng(3)
    K, M = 3, 60
    D = rng.normal(size=(K, K, M)) + 1j * rng.normal(size=(K, K, M))
    phi = np.exp(1j * rng.uniform(0, 2 * np.pi, M))
    s2 = 0.3
    f0, g = _f2_and_grad(phi, D, s2)
    eps = 1e-6
    err = []
    for _ in range(20):
        dv = rng.normal(size=M) + 1j * rng.normal(size=M)
        fp, _ = _f2_and_grad(phi + eps * dv, D, s2, want_grad=False)
        fm, _ = _f2_and_grad(phi - eps * dv, D, s2, want_grad=False)
        num = (fp - fm) / (2 * eps)
        ana = float(np.real(np.vdot(g, dv)))
        err.append(abs(num - ana) / max(abs(num), 1e-12))
    print(f"   max rel. error over 20 directions: {max(err):.3e}")

    phi_opt = manifold_cg(phi, D, s2)
    f_opt, _ = _f2_and_grad(phi_opt, D, s2, want_grad=False)
    f_rnd = min(_f2_and_grad(np.exp(1j * rng.uniform(0, 2 * np.pi, M)), D, s2,
                             want_grad=False)[0] for _ in range(300))
    print(f"   f2: CG = {f_opt:.5f}   best of 300 random = {f_rnd:.5f}   "
          f"(start {f0:.5f})")
    print(f"   max ||phi|-1| = {np.abs(np.abs(phi_opt) - 1).max():.2e}")
    return max(err) < 1e-6 and f_opt < f_rnd


def check_wmmse():
    hdr("5. WMMSE: power budget, and beats MRT / random")
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

    def rand_rate():
        X = rng.normal(size=(N, K)) + 1j * rng.normal(size=(N, K))
        return sum_rate(H @ (X * np.sqrt(P) / np.linalg.norm(X)), s2)

    rr = max(rand_rate() for _ in range(300))
    print(f"   power = {pw:.6f} (budget {P})   WMMSE = {r:.4f}   "
          f"MRT = {rm:.4f}   best random = {rr:.4f} bps/Hz")
    return pw <= P * (1 + 1e-6) and r > rm and r > rr


def check_monotone():
    hdr("6. Algorithm 4 sum-rate should never decrease")
    prm = Params()
    E = irs_elements(prm)
    off = offsets_fine(prm)
    a_min = int(round(prm.Dmin / prm.delta_s))
    rng = np.random.default_rng(7)
    d = 3.0
    H_bi = bi_channel(prm, E, points_from_offsets(d, off), d, 4, rng)
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
    print("   per iteration: " + " ".join(f"{x:.4f}" for x in seq))
    drops = sum(1 for i in range(1, len(seq)) if seq[i] < seq[i - 1] - 1e-9)
    print(f"   decreases: {drops}")
    return drops == 0


def check_farfield():
    hdr("7. Eq. (31): MA gain should vanish in far-field LoS")
    prm = Params()
    E = irs_elements(prm)
    off = offsets_fine(prm)
    a_min = int(round(prm.Dmin / prm.delta_s))
    idx_fpa = np.array([int(np.argmin(np.abs(off - o)))
                        for o in offsets_fpa(prm, prm.N)])
    idx_spr = np.linspace(0, len(off) - 1, prm.N).round().astype(int)
    # d = 1 m is the near-field control. without it this test would also
    # pass if the position search never moved anything.
    print("   d_BI [m]     proposed      FPA        relative gap")
    gaps = []
    for d in [1.0, 5.0, 50.0, 500.0, 5000.0]:
        rng = np.random.default_rng(11)
        H_bi = bi_channel(prm, E, points_from_offsets(d, off), d, 0, rng)
        H_iu = np.stack([iu_channel(prm, E, float(rng.uniform(30, 50)), rng)
                         for _ in range(prm.K)])
        phi0 = np.exp(1j * rng.uniform(0, 2 * np.pi, prm.M))
        r_ma = _mu_best(prm, H_iu, H_bi, phi0, [idx_fpa, idx_spr],
                        a_min, True, True)
        r_fp = mu_solve(prm, H_iu, H_bi, phi0, idx_fpa, a_min, True, False)[0]
        g = (r_ma - r_fp) / r_fp
        gaps.append(g)
        tag = "  <- near-field control, should be > 0" if d == 1.0 else ""
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

def grids(prm, N, fpa_offsets=None):
    # The FPA layout is always included as a starting point. It's a feasible
    # MA layout, so this guarantees MA >= FPA; the position searches are only
    # locally optimal for a given Phi so other starts alone don't.
    off_f = offsets_fine(prm)
    off_a = offsets_as(prm)
    off_p = offsets_fpa(prm, N) if fpa_offsets is None else np.asarray(fpa_offsets)
    return dict(
        off_f=off_f, off_a=off_a, off_p=off_p,
        a_fine=int(round(prm.Dmin / prm.delta_s)),
        a_as=1,
        i_fpa_f=np.array([int(np.argmin(np.abs(off_f - o))) for o in off_p]),
        i_fpa_a=np.array([int(np.argmin(np.abs(off_a - o))) for o in off_p]),
        i_spr_f=np.linspace(0, len(off_f) - 1, N).round().astype(int),
        i_spr_a=np.linspace(0, len(off_a) - 1, N).round().astype(int),
    )


def draw(prm, E, d, L_paths, off_list, rng, K):
    # one channel realisation, evaluated on all the position grids at once
    pts = [points_from_offsets(d, o) for o in off_list]
    H = bi_channel(prm, E, np.concatenate(pts, axis=0), d, L_paths, rng)
    out, s = [], 0
    for o in off_list:
        out.append(H[:, s:s + len(o)])
        s += len(o)
    if K > 1:
        H_iu = np.stack([iu_channel(prm, E, float(rng.uniform(*prm.d_iu)), rng)
                         for _ in range(K)])
    else:
        H_iu = iu_channel(prm, E, prm.d_iu[0], rng)
    return out, H_iu


def _su_best(prm, h_iu, H, phi, starts, a_min, opt_phi, opt_pos):
    return max(su_solve(prm, h_iu, H, phi, i0, a_min, opt_phi, opt_pos)[0]
               for i0 in starts)


def _mu_best(prm, H_iu, H, phi, starts, a_min, opt_phi, opt_pos):
    return max(mu_solve(prm, H_iu, H, phi, i0, a_min, opt_phi, opt_pos)[0]
               for i0 in starts)


def accumulate(path, new, n_new, extra=None):
    # add a batch of trials to the running sums on disk
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


def load_sums(path):
    z = np.load(path)
    n = int(z["count"])
    out = {k[4:]: z[k] / n for k in z.files if k.startswith("sum_")}
    out["dists"] = z["dists"]
    out["count"] = n
    return out

SCHEMES = ["proposed", "as", "fpa", "ma_rps", "fpa_rps"]


def su_trial(task):
    # single user (Figs 2, 3), returns linear SNR
    prm, d, L_paths, N, fpa_offsets, seed, di, t = task
    E = irs_elements(prm)
    g = grids(prm, N, fpa_offsets)
    rng = np.random.default_rng((seed, di, t))
    (Hf, Ha, Hp), h_iu = draw(prm, E, d, L_paths,
                              [g["off_f"], g["off_a"], g["off_p"]], rng, 1)
    phi0 = np.exp(1j * rng.uniform(0, 2 * np.pi, prm.M))
    phi_r = np.exp(1j * rng.uniform(0, 2 * np.pi, prm.M))
    if N == 1:
        sf, sa = [g["i_fpa_f"]], [g["i_fpa_a"]]
    else:
        sf = [g["i_fpa_f"], g["i_spr_f"]]
        sa = [g["i_fpa_a"], g["i_spr_a"]]
    i_all = np.arange(N)
    return {
        "proposed": _su_best(prm, h_iu, Hf, phi0, sf, g["a_fine"], True, True),
        "as":       _su_best(prm, h_iu, Ha, phi0, sa, g["a_as"], True, True),
        "fpa":      su_solve(prm, h_iu, Hp, phi0, i_all, 1, True, False)[0],
        "ma_rps":   _su_best(prm, h_iu, Hf, phi_r, sf, g["a_fine"], False, True),
        "fpa_rps":  su_gamma(h_iu, Hp, phi_r),
    }


def mu_trial(task):
    # multi user (Figs 6, 8), returns sum rate in bps/Hz
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
    # post() is applied per trial before averaging -- needed for rates since
    # E[log2(1+g)] != log2(1+E[g])
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
    print(f"  Fig. 2  N=1 max |MA-FPA| = {e:.2e} dB  (should be 0 by Eq. 13/17)")
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

def build_fig2(trials, t0, jobs, outdir):
    dists = np.arange(1.0, 7.0)
    new = {}
    for N in (1, 4):
        # Sec. III-A: M = 25^2, P/sigma^2 = 110 dB, LoS only
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
    prm = Params(d_iu=(30.0, 30.0))       # d_IU = 30 m, L = 8
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
    L = 0 if fig == 6 else 4              # Fig 6 LoS, Fig 8 four NLoS paths
    res = sweep(mu_trial,
                lambda di, d: [(prm, d, L, init, 0, di, t)
                               for t in range(t0, t0 + trials)],
                dists, trials, jobs, f"fig{fig}")
    p = os.path.join(outdir, f"sums_fig{fig}_{init}.npz")
    accumulate(p, res, trials, extra={"dists": dists})
    return p

def main():
    ap = argparse.ArgumentParser(
        description="Reproduce Figs. 2, 3, 6, 8 of Wei et al., IEEE TCOM 2025.")
    ap.add_argument("--fig", type=int, nargs="+", default=[2, 3, 6, 8],
                    choices=[2, 3, 6, 8])
    ap.add_argument("--trials", type=int, default=100,
                    help="realisations per point (paper: 1000)")
    ap.add_argument("--t0", type=int, default=0,
                    help="first trial index, for extending a run")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    ap.add_argument("--init", default="best", choices=["best", "compact", "spread"],
                    help="starting layout for Algorithm 3")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="10 trials")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw from saved results")
    ap.add_argument("--no-show", action="store_true", help="only save PNGs")
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
            print("   checks failed, not running the simulation")
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
                print(f"   no results for fig {f} at {path}, skipping")
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
            print("   no display found, figures saved as PNGs only "
                  "(set MPLBACKEND to override)")
        return
    print(f"   showing {len(files)} figure(s) [{matplotlib.get_backend()}], "
          "close the windows to exit")
    try:
        plt.show()
    except Exception as exc:
        # PNGs are already saved, don't lose the run over a window error
        plt.close("all")
        print(f"   could not open a window ({exc.__class__.__name__}: {exc}); "
              "PNGs were still written")


if __name__ == "__main__":
    main()
