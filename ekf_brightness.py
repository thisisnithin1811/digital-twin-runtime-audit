"""
ekf_brightness.py
=================
Extended Kalman Filter for the Vanishing-Rod DT with brightness in the state.

State  :  x = [n_m, T, τ, z, α, B]ᵀ                       (6-dim)
Measure:  z = [E, C, T_sens, z_enc, B_sens]ᵀ               (5-dim)

Measurement model and Jacobian: see physics_brightness.py.

This module is a **drop-in replacement** for the legacy dt/estimator/ekf.py
that ships with the project. The legacy 5-dim variant used
    [n_m, T, τ, z, α] with measurements [E, C, T_sens, z_enc].
To keep your existing routes working, we also expose RefracEKF and Coeffs
symbols (aliases) at the bottom of this file so `from dt.estimator.ekf
import RefracEKF, Coeffs` keeps working — but now carrying brightness.

Usage
-----
    from dt_extension.ekf_brightness import RefracEKF_B, CoeffsB, StateB
    ekf = RefracEKF_B(coeffs)
    ekf.predict(uz=0.0, dt=1.0)
    ekf.update(z_meas=np.array([E, C, T, z_enc, B_nits]))
    print(ekf.n_m, ekf.sigma_n_m)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .physics_brightness import (
    CoeffsB, StateB,
    fresnel_R, dR_dnm,
    C_hat, E_hat,
    jacobian_H, measurement_noise,
    invert_nm_from_C, default_coeffs,
)


# ────────────────────────────────────────────────────────────────────────────
#  EKF
# ────────────────────────────────────────────────────────────────────────────

class RefracEKF_B:
    """Extended Kalman Filter on the 6-dim state with brightness."""

    # Order of state dimensions (do not change without touching H)
    IDX_NM, IDX_T, IDX_TAU, IDX_Z, IDX_ALPHA, IDX_B = range(6)

    def __init__(self,
                 coeffs: CoeffsB,
                 x0: np.ndarray = None,
                 P0: np.ndarray = None,
                 Q:  np.ndarray = None):
        self.coeffs = coeffs

        # Initial state — sensible priors at rest
        if x0 is None:
            x0 = np.array([
                coeffs.n_r * 0.9,   # n_m below rod → non-vanish prior
                coeffs.T0,          # T
                0.0,                # τ
                0.0,                # z
                1.0,                # α (camera scale, set by flat-field)
                coeffs.B0,          # B at calibration brightness
            ])
        if P0 is None:
            # Diagonal priors: nm very uncertain at start, T known to ±0.5°C,
            # τ small, z known after homing, α known after flat-field,
            # B fairly well known from nits sensor at boot.
            P0 = np.diag([1e-2, 0.25, 1e-1, 1.0, 1e-2, (0.1 * coeffs.B0) ** 2])
        if Q is None:
            # Process noise for each state dimension (allows EKF to adapt n_m over time)
            # Increased n_m process noise from 1e-6 to 1e-5 for better convergence
            Q = np.diag([1e-5, 1e-3, 1e-4, 0.01, 1e-4, 1.0])

        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.xi = coeffs.xi0          # backlight fraction (run-time constant)

        self._innov_hist = []         # for chi² / NIS monitoring
        self._step = 0

    # ── Convenience accessors ──────────────────────────────────────────────
    @property
    def state(self) -> StateB:
        return StateB.from_vector(self.x, xi=self.xi)

    @property
    def n_m(self) -> float:
        return float(self.x[self.IDX_NM])

    @property
    def delta_n(self) -> float:
        return float(self.coeffs.n_r - self.x[self.IDX_NM])

    @property
    def sigma_n_m(self) -> float:
        return float(np.sqrt(max(self.P[self.IDX_NM, self.IDX_NM], 0.0)))

    @property
    def B(self) -> float:
        return float(self.x[self.IDX_B])

    # ── Predict ────────────────────────────────────────────────────────────
    def predict(self, uz: float = 0.0, dt: float = 1.0) -> None:
        """Predict step: identity + Z-kinematics."""
        F = np.eye(6)
        self.x[self.IDX_Z] += uz * dt
        self.P = F @ self.P @ F.T + self.Q * dt

    # ── Update ─────────────────────────────────────────────────────────────
    def update(self,
               z_meas: np.ndarray,
               R: np.ndarray = None) -> dict:
        """Measurement update.

        z_meas = [E, C, T_sens, z_enc, B_sens]   (5-dim)
        If the caller supplies only 4 values [E, C, T, z], we pad with the
        current B estimate so legacy callers still work.
        """
        # Back-compat: legacy callers pass length-4 vectors
        if len(z_meas) == 4:
            z_meas = np.array([*z_meas, self.B])
            # Also pad R if it was sized 4×4, adding a loose entry for B
            if R is not None and R.shape == (4, 4):
                R5 = np.zeros((5, 5))
                R5[:4, :4] = R
                R5[4, 4] = (0.1 * self.coeffs.B0) ** 2
                R = R5

        state = self.state
        H = jacobian_H(state, self.coeffs)
        if R is None:
            R = measurement_noise(state, self.coeffs)

        z_pred = np.array([
            E_hat(state, self.coeffs),
            C_hat(state, self.coeffs),
            state.T,
            state.z,
            state.B,
        ])

        y = z_meas - z_pred                           # innovation
        S = H @ self.P @ H.T + R                      # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)           # Kalman gain
        self.x = self.x + K @ y
        # Joseph form for numerical stability
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

        # Clamp n_m to physically plausible window
        self.x[self.IDX_NM] = float(np.clip(self.x[self.IDX_NM], 1.00, 2.00))
        self.x[self.IDX_TAU] = max(0.0, float(self.x[self.IDX_TAU]))
        self.x[self.IDX_B]   = max(1.0, float(self.x[self.IDX_B]))

        # Normalized innovation squared (NIS) — for filter consistency monitoring
        try:
            nis = float(y @ np.linalg.inv(S) @ y)
        except np.linalg.LinAlgError:
            nis = float("nan")
        self._innov_hist.append(nis)
        self._step += 1

        return {
            "innovation": y.tolist(),
            "NIS": nis,
            "n_m": self.n_m,
            "sigma_n_m": self.sigma_n_m,
            "delta_n": self.delta_n,
            "B": self.B,
            "step": self._step,
        }

    # ── One-shot initialiser from a single frame ──────────────────────────
    def seed_from_contrast(self, C_meas: float) -> None:
        """Reset n_m using the analytic inversion of eq. (6)."""
        s = self.state
        nm0 = invert_nm_from_C(C_meas, s, self.coeffs)
        self.x[self.IDX_NM] = nm0
        self.P[self.IDX_NM, self.IDX_NM] = 1e-2    # re-inflate uncertainty

    # ── Consistency diagnostic ─────────────────────────────────────────────
    def consistency_report(self, window: int = 50) -> dict:
        """Chi² check: for a well-tuned 5-measurement EKF, E[NIS] ≈ 5.

        If mean(NIS) >> 5: filter is over-confident — increase Q or R.
        If mean(NIS) << 5: filter is under-confident — decrease Q or R.
        """
        if not self._innov_hist:
            return {"n_samples": 0}
        w = self._innov_hist[-window:]
        arr = np.array(w, dtype=float)
        arr = arr[np.isfinite(arr)]
        return {
            "n_samples": int(arr.size),
            "mean_NIS": float(arr.mean()) if arr.size else float("nan"),
            "expected_NIS": 5.0,     # dim(z)
            "well_tuned": bool(arr.size and 2.0 < arr.mean() < 10.0),
        }


# ────────────────────────────────────────────────────────────────────────────
#  Legacy compatibility shim
#
#  The original code imports:
#      from dt.estimator.ekf import RefracEKF, Coeffs
#  and builds a Coeffs object from a flat list of 4 betas and 4 gammas.
#
#  We provide a Coeffs dataclass and a RefracEKF wrapper that forward to
#  the brightness-aware implementation, so no other file needs to change.
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Coeffs:
    """Legacy 4β, 4γ coefficient holder. Wraps CoeffsB with beta5 = 0."""
    b1: float
    b2: float
    b3: float
    b4: float
    g1: float
    g2: float
    g3: float
    g4: float
    n_r: float = 1.50
    T0:  float = 20.0
    # Optional: give a default brightness so we can run without a nits sensor.
    B0:  float = 300.0


def _legacy_to_B(c: Coeffs) -> CoeffsB:
    return CoeffsB(
        beta1=c.b1, beta2=c.b2, beta3=c.b3, beta4=c.b4, beta5=0.0,
        gamma1=c.g1, gamma2=c.g2, gamma3=c.g3, gamma4=c.g4,
        n_r=c.n_r, T0=c.T0, B0=c.B0, xi0=1.0,
        version="legacy",
    )


class RefracEKF:
    """Legacy-signature wrapper. Uses RefracEKF_B under the hood.

    Old code calls:
        ekf = RefracEKF(Coeffs(b1..b4, g1..g4, n_r, T0))
        ekf.predict(uz=0.0, dt=1.0)
        ekf.update(z_meas=[E, C, T, z], R=np.diag([...]))
        print(ekf.n_m)
    This still works; brightness is simply pinned at the calibration B0.
    """
    def __init__(self, coeffs: Coeffs):
        self._b = RefracEKF_B(_legacy_to_B(coeffs))

    def predict(self, uz: float = 0.0, dt: float = 1.0) -> None:
        self._b.predict(uz=uz, dt=dt)

    def update(self, z_meas, R=None) -> dict:
        return self._b.update(np.asarray(z_meas, dtype=float), R=R)

    @property
    def n_m(self) -> float: return self._b.n_m
    @property
    def delta_n(self) -> float: return self._b.delta_n
    @property
    def sigma_n_m(self) -> float: return self._b.sigma_n_m


# ────────────────────────────────────────────────────────────────────────────
#  Self-test: simulate a run, verify EKF converges to truth
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Ground truth: an "oil" beaker at 22 °C under dim light (B=120 nits).
    truth = StateB(n_m=1.4700, T=22.0, tau=0.0, z=0.0, alpha=1.0, B=120.0)

    coeffs = default_coeffs(n_r=1.50, T0=20.0, B0=300.0)

    # Generate synthetic measurements using the SAME model (ideal case).
    from .physics_brightness import E_hat, C_hat
    def gen_z(s):
        E = E_hat(s, coeffs) + rng.normal(0, 0.02)
        C = C_hat(s, coeffs) + rng.normal(0, 1e-3)
        T = s.T + rng.normal(0, 0.05)
        z = s.z + rng.normal(0, 0.05)
        B = s.B + rng.normal(0, 2.0)
        return np.array([E, C, T, z, B])

    # Production-faithful init: read ONE sensor sample first, set B_0 from it,
    # then seed n_m from the contrast inversion.
    z0 = gen_z(truth)
    x0 = np.array([
        coeffs.n_r * 0.9, z0[2], 0.0, z0[3], 1.0, z0[4],
    ])
    ekf = RefracEKF_B(coeffs, x0=x0)
    ekf.seed_from_contrast(z0[1])

    for k in range(40):
        ekf.predict(uz=0.0, dt=0.1)
        ekf.update(gen_z(truth))

    print(f"Truth n_m  : {truth.n_m:.4f}")
    print(f"EKF   n_m  : {ekf.n_m:.4f}  (σ = {ekf.sigma_n_m:.4f})")
    print(f"Error      : {abs(ekf.n_m - truth.n_m):.4f} RIU")
    print(f"B estimate : {ekf.B:.1f} nits (truth {truth.B:.1f})")
    print(f"Consistency: {ekf.consistency_report()}")
