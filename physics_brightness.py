"""
physics_brightness.py
=====================
Extended photometric / Fresnel model for the Vanishing-Rod Digital Twin,
with ambient brightness B (in nits = cd/m²) as an explicit state.

Why this matters
----------------
Liquid refractive index depends on temperature with dn/dT ≈ O(1e-4) per K
(Tan & Huang, JCED 2015). That is small. What is NOT small is the variation
in ambient / backlight brightness between one lab bench and another: two
identical setups placed in rooms with different lighting will produce
materially different image intensities, different edge-energies, and
different contrast noise — yet the same n_m.

If we ignore B, the calibration coefficients implicitly absorb the brightness
at the calibration site and the DT does not transfer cleanly to another
location. This module derives, and this file implements, a B-aware version
of the measurement model so the same DT works across sites.

Derivation (see PAPER_EXTENDED.md §4 for the full treatment)
-----------------------------------------------------------
Photometric model at a small ROI:

    I_bg(p)  = k · ( L_bl + ρ · B_amb ) + η_bg                      (1)
    I_rod(p) = k · ( L_bl · (1 − κ_R R(n_m) − κ_τ τ)
                     + ρ · B_amb ) + η_rod                          (2)

where
    k         camera-side gain (exposure × sensor sensitivity) [ADU / (cd/m²)],
    L_bl      backlight luminance reaching the ROI             [cd/m² = nit],
    B_amb     ambient luminance coupled into the ROI           [cd/m² = nit],
    ρ         stray-light coupling coefficient                 [dimensionless],
    R(n_m)    Fresnel reflectance at rod-liquid interface      [dimensionless],
    τ         turbidity/scatter proxy                          [dimensionless],
    κ_R, κ_τ  physical constants to be absorbed                [dimensionless],
    η_•       photometric noise.

Define the TOTAL effective luminance observed by the camera at that ROI:

    B  ≡ L_bl + ρ · B_amb                                           (3)

This is exactly what a lux / nits sensor co-located with the camera will
report (up to a known geometric factor). It is the variable we instrument.

With (3), equations (1)–(2) simplify to:

    I_bg  = k·B + η_bg                                              (4)
    I_rod = k·(B − L_bl · (κ_R R + κ_τ τ)) + η_rod                  (5)

Now the two image metrics used by the estimator:

Contrast C = |I_rod − I_bg| / I_bg:

    C  =  (L_bl/B) · (κ_R R + κ_τ τ)  +  O(η/B)

Let ξ = L_bl/B = backlight fraction (close to 1 when the shroud is good,
< 1 when ambient leaks in). Then, adding a temperature correction and an
affine offset absorbed into calibration, we obtain the extended contrast
forward model:

    Ĉ  =  α · [ β₁ R(n_m) + β₂ τ + β₃ (T − T₀) + β₄ ]
          + β₅ · (ξ − 1)                                            (6)

Here β₅ captures the residual coupling of stray light into the contrast.
When ξ = 1 (no stray light) equation (6) reduces to the original model.

Edge energy E = variance of Laplacian. Because the Laplacian is linear in
intensity, E is quadratic in intensity, hence quadratic in B:

    Ê  =  b² · [ γ₁ R(n_m) + γ₂ τ ]  +  γ₃ (T − T₀) + γ₄           (7)

where b = B / B₀ is B normalised by the calibration brightness B₀.

Measurement noise scales too:
    σ_C  ∝  1 / √B        (shot-noise-like, relative contrast)
    σ_E  ∝  1 / B²        (edge-energy variance of a darker image)

So a B-aware EKF can weight measurements correctly AND predict observability:
the Jacobian sensitivity to n_m is proportional to b² in E and to ξ in C,
which means DIMMER setups have LESS observable n_m and need MORE frames.
This is exactly the phenomenon the physical lab was struggling to explain.

What we instrument
------------------
- B at the ROI:    BH1750 / TSL2591 / OPT3001 → lux → nits via geometric factor.
                   See brightness_sensor.py for the driver.
- ξ = L_bl / B:    estimated by turning the backlight off briefly and taking
                   the ratio, OR fixed by calibration when the shroud is good.

Functions provided
------------------
- fresnel_R(n_r, n_m)                 → R
- dR_dnm(n_r, n_m)                    → ∂R/∂n_m
- C_hat(state, coeffs)                → predicted contrast (eq. 6)
- E_hat(state, coeffs)                → predicted edge-energy (eq. 7)
- measurement_noise(state, sensor)    → σ_E², σ_C² scaled by B
- invert_nm_from_C(C, B, coeffs, n_r) → one-shot initialiser
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np


# ────────────────────────────────────────────────────────────────────────────
#  Coefficients (extended)
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CoeffsB:
    """Extended calibration coefficients including brightness terms.

    Legacy 4-vector (beta1..beta4, gamma1..gamma4) is preserved. beta5 and
    gamma5 are the new brightness terms. beta_B0 is the brightness at which
    the calibration was taken (nits). xi0 is the backlight fraction at
    calibration (usually 1.0).
    """
    # Contrast model (eq. 6)
    beta1: float   # R(n_m) coefficient
    beta2: float   # turbidity coefficient
    beta3: float   # (T - T0) coefficient                [1/K]
    beta4: float   # offset
    beta5: float   # brightness (ξ − 1) coefficient       [dimensionless]

    # Edge-energy model (eq. 7)
    gamma1: float  # R(n_m) coefficient (multiplied by b²)
    gamma2: float  # turbidity coefficient (multiplied by b²)
    gamma3: float  # (T - T0) coefficient                [1/K]
    gamma4: float  # offset

    # Reference points
    n_r: float     # rod index (borosilicate ≈ 1.50)
    T0: float      # reference temperature               [°C]
    B0: float      # reference brightness                [nits = cd/m²]
    xi0: float = 1.0  # reference backlight fraction     [dimensionless]

    # Optional coefficient covariance for traceability (filled after
    # weighted-LS calibration).
    cov_beta:  Optional[np.ndarray] = None
    cov_gamma: Optional[np.ndarray] = None

    # Metadata (filled from YAML)
    version: str = "v0"
    calibration_date: str = ""
    device_id: str = ""
    residual_rms_C: float = float("nan")
    residual_rms_E: float = float("nan")


@dataclass
class StateB:
    """Extended state vector. Order is FIXED — do NOT change (EKF depends on it).

        x = [n_m, T, τ, z, α, B]ᵀ

    n_m  medium refractive index       [RIU]
    T    temperature                   [°C]
    τ    turbidity proxy               [dimensionless]
    z    immersion depth               [mm]
    α    photometric scale             [dimensionless]
    B    effective brightness at ROI   [nits]
    """
    n_m: float
    T:   float
    tau: float
    z:   float
    alpha: float
    B:   float

    # Optional: backlight fraction ξ. Default = coeffs.xi0; override per-run
    # if a shroud-off calibration gives you a real measurement.
    xi:  float = 1.0

    def as_vector(self) -> np.ndarray:
        return np.array([self.n_m, self.T, self.tau, self.z, self.alpha, self.B])

    @classmethod
    def from_vector(cls, x: np.ndarray, xi: float = 1.0) -> "StateB":
        return cls(n_m=float(x[0]), T=float(x[1]), tau=float(x[2]),
                   z=float(x[3]), alpha=float(x[4]), B=float(x[5]), xi=xi)


# ────────────────────────────────────────────────────────────────────────────
#  Core Fresnel physics
# ────────────────────────────────────────────────────────────────────────────

def fresnel_R(n_r: float, n_m: float) -> float:
    """Normal-incidence power reflectance R(n_m) = ((n_r − n_m)/(n_r + n_m))²."""
    q = (n_r - n_m) / (n_r + n_m)
    return q * q


def dR_dnm(n_r: float, n_m: float) -> float:
    """dR/dn_m = −4 n_r (n_r − n_m) / (n_r + n_m)³ .

    Follows from R = q² and q = (n_r − n_m)/(n_r + n_m):
        dq/dn_m = −2 n_r / (n_r + n_m)²
        dR/dn_m = 2 q · dq/dn_m
                = −4 n_r (n_r − n_m) / (n_r + n_m)³
    """
    num = -4.0 * n_r * (n_r - n_m)
    den = (n_r + n_m) ** 3
    return num / den


# ────────────────────────────────────────────────────────────────────────────
#  Forward measurement model (eq. 6, 7)
# ────────────────────────────────────────────────────────────────────────────

def _b(state: StateB, coeffs: CoeffsB) -> float:
    """Normalised brightness b = B / B0."""
    return state.B / coeffs.B0 if coeffs.B0 > 0 else 1.0


def C_hat(state: StateB, coeffs: CoeffsB) -> float:
    """Predicted contrast (eq. 6)."""
    R = fresnel_R(coeffs.n_r, state.n_m)
    contrast_core = (coeffs.beta1 * R
                     + coeffs.beta2 * state.tau
                     + coeffs.beta3 * (state.T - coeffs.T0)
                     + coeffs.beta4)
    return state.alpha * contrast_core + coeffs.beta5 * (state.xi - coeffs.xi0)


def E_hat(state: StateB, coeffs: CoeffsB) -> float:
    """Predicted edge-energy (eq. 7)."""
    R = fresnel_R(coeffs.n_r, state.n_m)
    b = _b(state, coeffs)
    return (b * b) * (coeffs.gamma1 * R + coeffs.gamma2 * state.tau) \
           + coeffs.gamma3 * (state.T - coeffs.T0) \
           + coeffs.gamma4


def predict_measurements(state: StateB, coeffs: CoeffsB) -> dict:
    """Convenience: return all four model-predicted sensor values."""
    return {
        "E": E_hat(state, coeffs),
        "C": C_hat(state, coeffs),
        "T": state.T,
        "z": state.z,
        "B": state.B,
    }


# ────────────────────────────────────────────────────────────────────────────
#  Jacobian H = ∂h/∂x at the current state
#
#  z = [E, C, T_sens, z_enc, B_sens]ᵀ      (5-dim with nits sensor)
#  x = [n_m, T, τ, z, α, B]ᵀ                (6-dim)
#
#  Entries derived from (6), (7):
#    ∂Ĉ/∂n_m = α · β₁ · dR/dn_m
#    ∂Ĉ/∂T   = α · β₃
#    ∂Ĉ/∂τ   = α · β₂
#    ∂Ĉ/∂α   = β₁ R + β₂ τ + β₃ (T − T₀) + β₄
#    ∂Ĉ/∂B   = 0        (ξ is treated as a slowly-varying known, not part of x;
#                        if you want ξ in x, add a 7th dim and this entry.)
#    ∂Ê/∂n_m = b² · γ₁ · dR/dn_m
#    ∂Ê/∂T   = γ₃
#    ∂Ê/∂τ   = b² · γ₂
#    ∂Ê/∂B   = 2 b / B₀ · (γ₁ R + γ₂ τ)
# ────────────────────────────────────────────────────────────────────────────

def jacobian_H(state: StateB, coeffs: CoeffsB) -> np.ndarray:
    """5 × 6 Jacobian matrix of h(x) = [E, C, T, z, B]."""
    R = fresnel_R(coeffs.n_r, state.n_m)
    dR = dR_dnm(coeffs.n_r, state.n_m)
    b = _b(state, coeffs)
    H = np.zeros((5, 6))

    # Row 0: Edge energy E
    H[0, 0] = (b * b) * coeffs.gamma1 * dR                       # ∂Ê/∂n_m
    H[0, 1] = coeffs.gamma3                                      # ∂Ê/∂T
    H[0, 2] = (b * b) * coeffs.gamma2                            # ∂Ê/∂τ
    H[0, 5] = (2.0 * b / coeffs.B0) * (coeffs.gamma1 * R
                                        + coeffs.gamma2 * state.tau) \
              if coeffs.B0 > 0 else 0.0                          # ∂Ê/∂B

    # Row 1: Contrast C
    H[1, 0] = state.alpha * coeffs.beta1 * dR                    # ∂Ĉ/∂n_m
    H[1, 1] = state.alpha * coeffs.beta3                         # ∂Ĉ/∂T
    H[1, 2] = state.alpha * coeffs.beta2                         # ∂Ĉ/∂τ
    H[1, 4] = (coeffs.beta1 * R
               + coeffs.beta2 * state.tau
               + coeffs.beta3 * (state.T - coeffs.T0)
               + coeffs.beta4)                                   # ∂Ĉ/∂α

    # Row 2: Temperature sensor
    H[2, 1] = 1.0

    # Row 3: Z encoder
    H[3, 3] = 1.0

    # Row 4: Brightness sensor
    H[4, 5] = 1.0

    return H


# ────────────────────────────────────────────────────────────────────────────
#  Measurement-noise covariance R, scaled by brightness
#
#  σ_E ∝ 1/B²  (variance of Laplacian in a dark image is dominated by noise)
#  σ_C ∝ 1/√B
#
#  The constants are set at calibration; below we return the default scaling.
# ────────────────────────────────────────────────────────────────────────────

def measurement_noise(state: StateB,
                       coeffs: CoeffsB,
                       sigma_E_at_B0: float = 1e-4,
                       sigma_C_at_B0: float = 1e-2,
                       sigma_T:       float = 0.1,
                       sigma_z:       float = 0.05,
                       sigma_B:       float = 1.0) -> np.ndarray:
    """Return 5×5 diagonal R matrix, with E and C variances scaled by B."""
    b = max(_b(state, coeffs), 1e-3)
    var_E = (sigma_E_at_B0 ** 2) / (b ** 4)    # scales as 1/B²  → var as 1/B⁴
    var_C = (sigma_C_at_B0 ** 2) / b           # scales as 1/√B → var as 1/B
    return np.diag([var_E, var_C, sigma_T ** 2, sigma_z ** 2, sigma_B ** 2])


# ────────────────────────────────────────────────────────────────────────────
#  One-shot initialiser: invert contrast to get an n_m seed
# ────────────────────────────────────────────────────────────────────────────

def invert_nm_from_C(C_meas: float,
                      state: StateB,
                      coeffs: CoeffsB) -> float:
    """Given a measured contrast and current (T, τ, α, B) belief,
    solve eq. (6) for R and then for n_m.  Returns an initial guess.

    R_est = ( (C_meas − β₅(ξ−ξ₀)) / α − β₂ τ − β₃ (T−T₀) − β₄ ) / β₁
    n_m   = n_r · (1 − √R) / (1 + √R)
    """
    num = (C_meas - coeffs.beta5 * (state.xi - coeffs.xi0)) / max(state.alpha, 1e-6)
    num -= (coeffs.beta2 * state.tau
            + coeffs.beta3 * (state.T - coeffs.T0)
            + coeffs.beta4)
    R_est = num / coeffs.beta1 if abs(coeffs.beta1) > 1e-12 else 0.0
    R_est = max(0.0, min(R_est, 0.25))   # clamp to physical Fresnel range
    r = math.sqrt(R_est)
    return coeffs.n_r * (1.0 - r) / (1.0 + r)


# ────────────────────────────────────────────────────────────────────────────
#  Weighted least-squares calibration (fills in beta, gamma AND covariances)
#
#  Collects frames for multiple certified liquids at multiple (T, B) pairs.
#  Solves
#      c = X_c · β + ε_c,    e = X_e · γ + ε_e
#  where, for each calibration sample i,
#      X_c[i] = [ R_i , τ_i , (T_i − T₀) , 1 , (ξ_i − ξ₀) / α_i ]
#      X_e[i] = [ b_i² · R_i , b_i² · τ_i , (T_i − T₀) , 1 ]
#      c[i]   = C_i / α_i
#      e[i]   = E_i
# ────────────────────────────────────────────────────────────────────────────

def fit_coefficients(samples: list,
                      n_r: float,
                      T0: float,
                      B0: float,
                      xi0: float = 1.0) -> CoeffsB:
    """Fit β and γ by weighted least squares.

    samples : list of dicts with keys
        n_m, T, tau, B, xi, alpha, C, E, w   (w = per-sample weight, default 1)
    """
    n = len(samples)
    if n < 5:
        raise ValueError(
            f"Need ≥5 calibration samples for a 5-parameter β fit, got {n}.")

    X_c = np.zeros((n, 5))
    X_e = np.zeros((n, 4))
    c   = np.zeros(n)
    e   = np.zeros(n)
    w   = np.zeros(n)

    for i, s in enumerate(samples):
        R_i = fresnel_R(n_r, s["n_m"])
        b_i = s["B"] / B0
        xi_i = s.get("xi", xi0)
        alpha_i = max(s.get("alpha", 1.0), 1e-6)
        tau_i = s.get("tau", 0.0)
        T_i = s["T"]

        X_c[i] = [R_i, tau_i, (T_i - T0), 1.0,
                  (xi_i - xi0) / alpha_i]
        X_e[i] = [(b_i ** 2) * R_i, (b_i ** 2) * tau_i, (T_i - T0), 1.0]
        c[i] = s["C"] / alpha_i
        e[i] = s["E"]
        w[i] = s.get("w", 1.0)

    W = np.diag(w)

    def wls(X, y, label=""):
        XtW = X.T @ W
        M = XtW @ X
        # Use pseudo-inverse to survive rank-deficient design matrices
        # (e.g. if a regressor like ξ was not varied during calibration).
        # Columns that are completely un-identified will receive ~0 and a
        # warning; this is safer than raising a LinAlgError mid-experiment.
        try:
            cov = np.linalg.inv(M)
            kind = "inv"
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(M, rcond=1e-10)
            kind = "pinv"
            import warnings
            warnings.warn(
                f"fit_coefficients: design matrix for {label!r} is "
                f"rank-deficient — using pseudo-inverse. Some coefficients "
                f"may be un-identifiable. Vary the regressor (ξ? τ?) "
                f"across calibration samples to fix this.")
        p = cov @ XtW @ y
        res = y - X @ p
        sigma2 = float(res @ W @ res / max(1, len(y) - X.shape[1]))
        return p, sigma2 * cov, float(np.sqrt(np.mean(res ** 2))), kind

    beta,  cov_b, rms_c, _ = wls(X_c, c, label="beta (contrast)")
    gamma, cov_g, rms_e, _ = wls(X_e, e, label="gamma (edge-energy)")

    return CoeffsB(
        beta1=beta[0],  beta2=beta[1],  beta3=beta[2],
        beta4=beta[3],  beta5=beta[4],
        gamma1=gamma[0], gamma2=gamma[1],
        gamma3=gamma[2], gamma4=gamma[3],
        n_r=n_r, T0=T0, B0=B0, xi0=xi0,
        cov_beta=cov_b, cov_gamma=cov_g,
        residual_rms_C=rms_c, residual_rms_E=rms_e,
    )


# ────────────────────────────────────────────────────────────────────────────
#  YAML I/O (versioned, traceable)
# ────────────────────────────────────────────────────────────────────────────

def save_coeffs_yaml(coeffs: CoeffsB, path: str) -> None:
    import yaml
    d = {
        "version": coeffs.version or "v1",
        "calibration_date": coeffs.calibration_date,
        "device_id": coeffs.device_id,
        "n_r": coeffs.n_r,
        "T0": coeffs.T0,
        "B0": coeffs.B0,
        "xi0": coeffs.xi0,
        "beta":  [coeffs.beta1, coeffs.beta2, coeffs.beta3,
                  coeffs.beta4, coeffs.beta5],
        "gamma": [coeffs.gamma1, coeffs.gamma2,
                  coeffs.gamma3, coeffs.gamma4],
        "residual_rms_C": coeffs.residual_rms_C,
        "residual_rms_E": coeffs.residual_rms_E,
    }
    if coeffs.cov_beta is not None:
        d["cov_beta"] = coeffs.cov_beta.tolist()
    if coeffs.cov_gamma is not None:
        d["cov_gamma"] = coeffs.cov_gamma.tolist()
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)


def load_coeffs_yaml(path: str) -> CoeffsB:
    import yaml
    d = yaml.safe_load(open(path).read())
    beta  = d.get("beta",  [0.0] * 5)
    gamma = d.get("gamma", [0.0] * 4)
    # Back-compat: if only 4 betas, append beta5 = 0.
    if len(beta) == 4:
        beta = list(beta) + [0.0]
    coeffs = CoeffsB(
        beta1=beta[0], beta2=beta[1], beta3=beta[2],
        beta4=beta[3], beta5=beta[4],
        gamma1=gamma[0], gamma2=gamma[1],
        gamma3=gamma[2], gamma4=gamma[3],
        n_r=d.get("n_r", 1.50),
        T0=d.get("T0", 20.0),
        B0=d.get("B0", 300.0),
        xi0=d.get("xi0", 1.0),
        version=d.get("version", "v0"),
        calibration_date=d.get("calibration_date", ""),
        device_id=d.get("device_id", ""),
        residual_rms_C=d.get("residual_rms_C", float("nan")),
        residual_rms_E=d.get("residual_rms_E", float("nan")),
    )
    if "cov_beta" in d:
        coeffs.cov_beta = np.asarray(d["cov_beta"], dtype=float)
    if "cov_gamma" in d:
        coeffs.cov_gamma = np.asarray(d["cov_gamma"], dtype=float)
    return coeffs


# ────────────────────────────────────────────────────────────────────────────
#  Placeholder default coeffs  — REPLACE by running fit_coefficients()
#  These are NOT measured values. They are consistent with the expected
#  signs and orders of magnitude reported in Tan & Huang (2015) and in
#  NithinDTProject_3.pdf, and they exist only so the EKF can start up.
# ────────────────────────────────────────────────────────────────────────────

def default_coeffs(n_r: float = 1.50,
                   T0:  float = 20.0,
                   B0:  float = 300.0) -> CoeffsB:
    """PLACEHOLDER coefficients. Calibrate before trusting any n_m estimate.

    Magnitudes chosen to be consistent with:
      * water (n_m=1.33) → R=3.6e-3 → C≈0.11   (rod clearly visible)
      * oil   (n_m=1.47) → R=1.0e-4 → C≈0.003  (near vanish)
      * match (n_m=1.50) → R=0     → C=0      (vanished)
    and with Tan & Huang (2015) dn/dT ≈ O(1e-4 /K) → β3 ~ 1e-3 on C.
    """
    return CoeffsB(
        beta1=30.0,    beta2=0.10,  beta3=1.0e-3, beta4=0.0, beta5=0.02,
        gamma1=250.0,  gamma2=5.0,  gamma3=-0.05, gamma4=2.0,
        n_r=n_r, T0=T0, B0=B0, xi0=1.0,
        version="v0-placeholder",
        calibration_date="",
        device_id="UNCALIBRATED",
    )


# ────────────────────────────────────────────────────────────────────────────
#  Self-test
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint
    c = default_coeffs()
    s_water = StateB(n_m=1.3330, T=20.0, tau=0.0, z=0.0, alpha=1.0, B=300.0)
    s_oil   = StateB(n_m=1.4700, T=20.0, tau=0.0, z=0.0, alpha=1.0, B=300.0)
    s_match = StateB(n_m=1.5000, T=20.0, tau=0.0, z=0.0, alpha=1.0, B=300.0)

    print("Water (n_m = 1.3330):")
    pprint.pp(predict_measurements(s_water, c))
    print("Oil (n_m = 1.4700):")
    pprint.pp(predict_measurements(s_oil, c))
    print("Perfect match (n_m = n_r = 1.5000):")
    pprint.pp(predict_measurements(s_match, c))

    print("\nJacobian shape:", jacobian_H(s_oil, c).shape)
    print("R noise diag :", np.diag(measurement_noise(s_oil, c)))

    # Same oil under DIM ambient (B = 80 nits)
    s_oil_dim = StateB(n_m=1.4700, T=20.0, tau=0.0, z=0.0, alpha=1.0, B=80.0)
    print("\nSame oil under dim light (B = 80):")
    pprint.pp(predict_measurements(s_oil_dim, c))
    print("R noise under dim light:", np.diag(measurement_noise(s_oil_dim, c)))
    print(" → notice how var_E and var_C are LARGER → EKF will "
          "trust these measurements less, as it should.")
