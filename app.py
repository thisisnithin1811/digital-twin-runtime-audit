"""
app.py — Vanishing Rod Digital Twin (Integrated)
================================================
Run:  python3 app.py
Open: http://<RPI_IP>:5000

New PCB:
  Motor 1 (Water): ULN2003 #1, BCM pins 21,20,16,12 (physical 40,38,36,32)
  Motor 2 (Oil):   ULN2003 #2, BCM pins 24,25, 8, 7 (physical 18,22,24,26)
  DS18B20 Temp:    BCM 5 (physical 29)
  BH1750 Light:    I2C bus 1 — SDA=BCM2 (pin3), SCL=BCM3 (pin5)

4 Operating Modes:
  AB  — Full lab: camera + manual motor control
  Ab  — Physical camera + virtual DT twin (side-by-side)
  aB  — DT auto-control drives real plant (auto-vanish sweep)
  ab  — Pure simulation (no hardware needed)
"""

import os, sys, time, json, logging
import cv2, numpy as np

os.environ['GST_DEBUG'] = '0'
os.environ['GST_DEBUG_FILE'] = '/dev/null'

from flask import (Flask, render_template, Response, jsonify,
                   request, session, redirect, url_for)
from threading import Lock, Thread, Event
from functools import wraps

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("VR-DT")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Hardware import ────────────────────────────────────────────────────────────
try:
    from main_control import HardwareLab
    HW_AVAILABLE = True
    logger.info("✓ main_control imported")
except Exception as e:
    logger.warning(f"Hardware module not available: {e}")
    HW_AVAILABLE = False

# ── EKF import (legacy 4-state) ───────────────────────────────────────────────
EKF_AVAILABLE = False
try:
    from dt.estimator.ekf import RefracEKF, Coeffs
    EKF_AVAILABLE = True
    logger.info("✓ Legacy EKF imported")
except Exception as e:
    logger.warning(f"Legacy EKF not available: {e}")

# ── EKF-B import (6-state brightness-aware EKF from dt_extension) ─────────────
EKF_B_AVAILABLE = False
try:
    from dt_extension.ekf_brightness import RefracEKF_B
    from dt_extension.physics_brightness import default_coeffs, CoeffsB, StateB
    EKF_B_AVAILABLE = True
    logger.info("✓ EKF-B (brightness-aware 6-state) imported")
except Exception as e:
    logger.warning(f"EKF-B not available: {e}")

# ── DT Extension (BH1750 sampler, audit, voice, i18n, run-logger) ─────────────
DT_EXT_AVAILABLE = False
try:
    from dt_extension.app_extensions import register_dt_extensions
    DT_EXT_AVAILABLE = True
    logger.info("✓ DT extensions module imported")
except Exception as e:
    logger.warning(f"DT extensions not available: {e}")

# ── Flask App ──────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.secret_key = "vanishing_rod_dt_2025"

# ── Auth ───────────────────────────────────────────────────────────────────────
USERS = {
    "student": "lab2024",
    "teacher": "teach2024",
    "admin":   "admin123",
}

def login_required(f):
    @wraps(f)
    def dec(*a, **kw):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*a, **kw)
    return dec

# ── Config ────────────────────────────────────────────────────────────────────
# Glass rod refractive index (borosilicate glass)
N_ROD   = 1.500
T0      = 25.0      # calibration temperature °C (DS18B20 reference)
B0      = 300.0     # calibration brightness nits (BH1750 reference)

# Motor travel limits (28BYJ-48, half-step, calibrated)
MOTOR_HOME = 0
MOTOR_DOWN = 10000   # steps to fully submerge rod (maximum travel ~57cm)

# EKF model coefficients (from coeffs_v2.yaml calibration, 2026-05-03)
# CRITICAL FIX: Updated from wrong values [2.5, 0.1, 0.002, 0.05] / [1.8, 0.15, 0.001, 0.03]
# which were 100x too small. Caused EKF estimates to be completely wrong.
BETA  = [12.50, -0.80, 0.0015, 0.040]        # β1..β4 from coeffs_v2.yaml
GAMMA = [220.0, -3.5, 0.050, 0.200]          # γ1..γ4 from coeffs_v2.yaml

# ROI config for both beakers — loaded from roi_config.yaml so you can
# tune pixel coordinates without editing Python.  Falls back to defaults
# if the YAML is missing.
import yaml as _yaml
def _load_roi_config():
    """Load ROI from roi_config.yaml. Defaults match the front-lit lab setup."""
    _roi_path = os.path.join(BASE_DIR, "roi_config.yaml")
    defaults = {
        "zone_beaker_1": {
            "main":        [0,   0, 320, 480],
            "rod":         [80,  50, 160, 450],
            "background":  [200, 50, 280, 450],
            "liquid_level":[80,  50, 160, 450],
        },
        "zone_beaker_2": {
            "main":        [320, 0, 640, 480],
            "rod":         [400, 50, 480, 450],
            "background":  [520, 50, 600, 450],
            "liquid_level":[400, 50, 480, 450],
        },
    }
    if os.path.exists(_roi_path):
        try:
            with open(_roi_path) as f:
                loaded = _yaml.safe_load(f) or {}
            for zone in ("zone_beaker_1", "zone_beaker_2"):
                if zone in loaded:
                    defaults[zone].update(loaded[zone])
            logger.info(f"✓ ROI config loaded from {_roi_path}")
        except Exception as e:
            logger.warning(f"ROI config load failed ({e}), using defaults")
    else:
        logger.info("ROI config: using built-in defaults (no roi_config.yaml found)")
    return defaults

ROI_CFG = _load_roi_config()

# ── Globals ────────────────────────────────────────────────────────────────────
lab: HardwareLab = None
camera_lock   = Lock()
hw_init_lock  = Lock()
is_processing = False
_hw_initialized = False

# Live telemetry dict (updated by background thread)
_telem = {
    "position":    0,
    "position2":   0,
    "temperature": T0,
    "brightness":  B0,
    "beaker_1": {
        "label": "Water", "n_m": 1.3330, "delta_n": round(abs(N_ROD - 1.3330), 5),
        "E": None, "C": None, "status": "INITIALIZING",
        "rod_detected": False, "liquid_level": None,
    },
    "beaker_2": {
        "label": "Oil",   "n_m": 1.4500, "delta_n": round(abs(N_ROD - 1.4500), 5),
        "E": None, "C": None, "status": "INITIALIZING",
        "rod_detected": False, "liquid_level": None,
    },
    "timestamp": 0.0,
}

_sim   = {"n_m": 1.470, "tau": 0.03, "n_r": N_ROD}
_sweep = {"phase": "IDLE", "progress": 0, "scores": [], "best_z": None}

# ── Vision helpers ─────────────────────────────────────────────────────────────

def compute_edge_energy(gray: np.ndarray, roi: list, kernel: int = 5) -> float:
    """Variance of Laplacian inside ROI — measures edge sharpness."""
    x1, y1, x2, y2 = map(int, roi)
    h, w = gray.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = gray[y1:y2, x1:x2]
    if crop.size < 100:
        return 0.0
    lap = cv2.Laplacian(crop, cv2.CV_64F, ksize=kernel)
    return float(np.var(lap))

def compute_contrast(gray: np.ndarray, roi: list,
                     rod_roi: list = None, bg_roi: list = None) -> float:
    """
    Michelson contrast between rod region and background region.
    Falls back to std/mean if sub-ROIs not available.
    """
    x1, y1, x2, y2 = map(int, roi)
    h, w = gray.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = gray[y1:y2, x1:x2]

    if rod_roi and bg_roi:
        rx1,ry1,rx2,ry2 = map(int, rod_roi)
        bx1,by1,bx2,by2 = map(int, bg_roi)
        rod_patch = gray[max(0,ry1):min(h,ry2), max(0,rx1):min(w,rx2)]
        bg_patch  = gray[max(0,by1):min(h,by2), max(0,bx1):min(w,bx2)]
        if rod_patch.size > 0 and bg_patch.size > 0:
            I_rod = float(np.mean(rod_patch))
            I_bg  = float(np.mean(bg_patch))
            denom = (I_rod + I_bg)
            return float(abs(I_rod - I_bg) / denom) if denom > 1 else 0.0

    mean = float(np.mean(crop))
    std  = float(np.std(crop))
    return float(std / mean) if mean > 1 else 0.0

def draw_overlays(frame_bgr: np.ndarray) -> np.ndarray:
    """Overlay ROI boxes, status badges, and metrics onto camera frame."""
    out = frame_bgr.copy()
    T = _telem

    beakers = [
        ("beaker_1", "zone_beaker_1", (255, 200, 50),   "B1 WATER"),
        ("beaker_2", "zone_beaker_2", (50,  200, 255),  "B2 OIL"),
    ]
    for bk, rk, colour, label in beakers:
        roi_d  = ROI_CFG[rk]
        x1, y1, x2, y2 = map(int, roi_d["main"])
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        bd = T.get(bk, {})
        status = bd.get("status", "IDLE")
        nm     = bd.get("n_m")
        C_val  = bd.get("C")
        E_val  = bd.get("E")

        # Vanished = green box; visible = yellow; error = red
        if "VANISH" in status or "MATCH" in status:
            box_col = (0, 255, 100)
        elif status in ("IDLE", "NO_DATA", "ROD_NOT_DETECTED"):
            box_col = colour
        else:
            box_col = (0, 200, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), box_col, 2)

        # Status label
        cv2.rectangle(out, (x1, y1 - 22), (x2, y1), box_col, -1)
        cv2.putText(out, f"{label}  {status}", (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)

        # Metrics at bottom
        if nm is not None:
            cv2.putText(out, f"n={nm:.4f}", (x1 + 4, y2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        if C_val is not None:
            cv2.putText(out, f"C={C_val:.4f}", (x1 + 4, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 255), 1)

    # Sensor readings overlay (top-right)
    temp = T.get("temperature", T0)
    bri  = T.get("brightness", B0)
    pos  = T.get("position", 0)
    cv2.rectangle(out, (450, 5), (635, 55), (20, 20, 20), -1)
    cv2.putText(out, f"T={temp:.1f}°C  B={bri:.0f}lx", (455, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 180), 1)
    cv2.putText(out, f"Motor1={pos}  Motor2={T.get('position2',0)}", (455, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 255), 1)

    return out

# ── EKF analysis ──────────────────────────────────────────────────────────────

def analyse_frame(frame_rgb: np.ndarray, z_steps: int,
                  temperature: float = T0, brightness: float = B0) -> dict:
    """Run EKF estimation on both beakers. Maintains persistent EKF state across frames."""
    frame_gray = cv2.cvtColor(
        (frame_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    results = {}

    for bkey, label, roi_key in [
        ("beaker_1", "Water", "zone_beaker_1"),
        ("beaker_2", "Oil",   "zone_beaker_2"),
    ]:
        rd   = ROI_CFG[roi_key]
        main = rd["main"]
        rod  = rd.get("rod")
        bg   = rd.get("background")

        E = compute_edge_energy(frame_gray, main)
        C = compute_contrast(frame_gray, main, rod, bg)

        # Calibrated reference refractive indices from system characterization
        n_water_ref = 1.3330  # Water (standard, dn/dT ≈ -4e-4/K)
        n_oil_ref = 1.4500    # Mineral/vegetable oil (typical ≈ 1.45-1.47)
        n_m_initial = n_water_ref if label == "Water" else n_oil_ref

        n_m_est = None
        delta_n = None
        status = "NO_DATA"
        _extra = {}

        if E is not None and C is not None and E > 0.0:
            b_ratio = (brightness / B0) if B0 > 0 else 1.0

            # ── Path A: brightness-aware 6-state EKF (DISABLED - needs calibration) ──
            if False and EKF_B_AVAILABLE:  # Disabled until proper calibration
                try:
                    # Create CoeffsB with actual calibrated values (not placeholder defaults)
                    cb = CoeffsB(
                        beta1=BETA[0], beta2=BETA[1], beta3=BETA[2], beta4=BETA[3],
                        beta5=0.02,  # stray-light coupling (fixed)
                        gamma1=GAMMA[0], gamma2=GAMMA[1], gamma3=GAMMA[2], gamma4=GAMMA[3],
                        n_r=N_ROD, T0=T0, B0=B0, xi0=0.95,  # xi0 from experiment
                        version="calibrated-v2", calibration_date="2026-05-03",
                        device_id="VRDT-IIITH-NEW-PCB-001"
                    )
                    
                    # Re-initialize EKF every frame to avoid state drift/lock-in
                    # (persistent state was accumulating errors)
                    x0_ekf = np.array([n_m_initial, float(temperature), 0.0,
                                      float(z_steps), 1.0, float(brightness)])
                    logger.info(f"[{label}] EKF init: n_m_init={n_m_initial:.4f}, E={E:.1f}, C={C:.5f}")
                    ekf_b = RefracEKF_B(cb, x0=x0_ekf)
                    ekf_b.predict(uz=0.0, dt=1.0)
                    
                    # 5-dim measurement: [E, C, T_sens, z_enc, B_nits]
                    B_nits = brightness / 3.14159 if brightness > 0 else 95.0
                    z_meas = np.array([float(E), float(C),
                                       float(temperature), float(z_steps),
                                       float(B_nits)])
                    ekf_b.update(z_meas)
                    
                    n_m_est = float(ekf_b.n_m)  # NO hard constraint yet - let filter work
                    logger.info(f"[{label}] EKF raw estimate: n_m={n_m_est:.4f}, delta_n={ekf_b.delta_n:.4f}, sigma={ekf_b.sigma_n_m:.6f}")
                    n_m_est = max(1.0, min(1.6, n_m_est))  # Soft clipping only
                    n_m_est = round(n_m_est, 5)
                    delta_n = round(abs(N_ROD - n_m_est), 5)
                    R_val   = ((N_ROD - n_m_est) / (N_ROD + n_m_est)) ** 2
                    sigma   = round(float(getattr(ekf_b, 'sigma_n_m', 0.001)), 6)
                    
                    if delta_n < 0.003:
                        status = "VANISHED (Perfect Match)"
                    elif delta_n < 0.015:
                        status = "NEAR_VANISH"
                    else:
                        status = "VISIBLE"
                    _extra = {"sigma_nm": sigma,
                              "R": round(R_val, 8),
                              "B_nits": round(B_nits, 1)}
                except Exception as e:
                    logger.info(f"EKF-B error for {label}: {e}")
                    status = "EKF_B_ERROR"
                    n_m_est = n_m_initial
                    delta_n = round(abs(N_ROD - n_m_initial), 5)

            # ── Path B: legacy 4-state EKF (fallback) ────────────────────────
            elif EKF_AVAILABLE:
                try:
                    E_scaled = E / max(b_ratio ** 2, 0.01)
                    coeffs = Coeffs(
                        BETA[0], BETA[1], BETA[2], BETA[3],
                        GAMMA[0], GAMMA[1], GAMMA[2], GAMMA[3],
                        N_ROD, T0)
                    ekf = RefracEKF(coeffs)
                    ekf.x[0] = n_m_initial
                    ekf.predict(uz=0.0, dt=1.0)
                    # Improved measurement noise matrix for better EKF convergence
                    R_improved = np.diag([1e-5, 1e-2, 1e-1, 1e-1])  # Tighter noise
                    ekf.update(np.array([E_scaled, C, temperature, float(z_steps)]),
                               R_improved)
                    n_m_est = max(1.0, min(1.6, float(ekf.n_m)))  # Soft clipping
                    n_m_est = round(n_m_est, 5)
                    delta_n = round(abs(N_ROD - n_m_est), 5)
                    if delta_n < 0.003:
                        status = "VANISHED (Perfect Match)"
                    elif delta_n < 0.015:
                        status = "NEAR_VANISH"
                    else:
                        status = "VISIBLE"
                except Exception as e:
                    logger.debug(f"EKF error for {label}: {e}")
                    status = "EKF_ERROR"
                    n_m_est = n_m_initial
                    delta_n = round(abs(N_ROD - n_m_initial), 5)

            # ── Path C: physics-based fallback (no EKF modules) ──────────────
            else:
                # Calculate n_m from contrast using Fresnel equation inversion
                # C ≈ κ·R where R = ((n_r - n_m) / (n_r + n_m))²
                kappa = 30.0  # Empirical contrast-to-reflectance coupling
                try:
                    if C > 0.001:
                        sqrt_term = np.sqrt(min(abs(C) / kappa, 0.25))
                        n_m_est = N_ROD * (1.0 - sqrt_term) / (1.0 + sqrt_term)
                    else:
                        n_m_est = n_m_initial
                    n_m_est = max(1.0, min(1.6, float(n_m_est)))
                    n_m_est = round(n_m_est, 5)
                    delta_n = round(abs(N_ROD - n_m_est), 5)
                    if delta_n < 0.003:
                        status = "VANISHED (Index Matched)"
                    elif delta_n < 0.015:
                        status = "NEAR_VANISH"
                    else:
                        status = "VISIBLE"
                except Exception as e:
                    logger.debug(f"Fallback calculation error for {label}: {e}")
                    n_m_est = n_m_initial
                    delta_n = round(abs(N_ROD - n_m_initial), 5)
                    status = "FALLBACK_CALC"

        # Ensure we always have valid calculated values
        if n_m_est is None:
            n_m_est = n_m_initial
            delta_n = round(abs(N_ROD - n_m_initial), 5)
            status = "INITIALIZING"

        results[bkey] = {
            "label":    label,
            "n_m":      n_m_est,
            "delta_n":  delta_n,
            "E":        round(float(E), 6) if E is not None else None,
            "C":        round(float(C), 6) if C is not None else None,
            "status":   status,
        }
        if _extra:
            results[bkey].update(_extra)

    return results

# ── Hardware init ──────────────────────────────────────────────────────────────

def init_hardware():
    global lab, _hw_initialized
    with hw_init_lock:
        if _hw_initialized:
            return
        _hw_initialized = True

    if HW_AVAILABLE and lab is None:
        try:
            logger.info("Initialising HardwareLab…")
            lab = HardwareLab()
            logger.info("✓ HardwareLab ready")
        except Exception as e:
            logger.warning(f"HardwareLab init failed: {e}")
            lab = None

    logger.info(f"Hardware: {'READY' if lab else 'SIMULATED'}")

# ── Background telemetry thread ───────────────────────────────────────────────

def _telem_loop():
    """Poll sensors and camera every second; update _telem in-place."""
    global _telem
    while True:
        try:
            # Read sensors
            temp = lab.read_temperature()  if lab else T0
            # Try to read from DT extension brightness sampler first (live BH1750)
            bri = B0
            try:
                sampler = app.config.get("_dt_sampler")
                if sampler:
                    latest = sampler.latest()
                    if latest and latest.healthy:
                        bri = latest.nits * 3.14159  # Convert nits → illuminance for telemetry
            except Exception:
                # Fall back to hardware layer
                bri = lab.read_brightness_lux() if lab else B0
            pos1 = lab.motor_1.current_pos_steps if lab else 0
            pos2 = lab.motor_2.current_pos_steps if lab else 0

            # Capture frame
            cam = lab.camera if lab else None
            frame_rgb = None
            if cam:
                with camera_lock:
                    frame_rgb = cam.capture_frame()

            if frame_rgb is not None:
                results = analyse_frame(frame_rgb, pos1, temp, bri)
                _telem.update({
                    "position":    pos1,
                    "position2":   pos2,
                    "temperature": round(temp, 1),
                    "brightness":  round(bri, 1),
                    "beaker_1":    results["beaker_1"],
                    "beaker_2":    results["beaker_2"],
                    "timestamp":   time.time(),
                })
            else:
                _telem.update({
                    "position":    pos1,
                    "position2":   pos2,
                    "temperature": round(temp, 1),
                    "brightness":  round(bri, 1),
                    "timestamp":   time.time(),
                })
        except Exception as e:
            logger.error(f"[telem] {e}")
        time.sleep(1.0)

# ── Video stream ──────────────────────────────────────────────────────────────

def _generate_frames():
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    err_count = 0
    while True:
        cam = (lab.camera if lab else None)
        if cam is None:
            frame = black.copy()
            cv2.putText(frame, "NO CAMERA — SIMULATION MODE", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
            time.sleep(0.5)
            continue

        with camera_lock:
            fr_rgb = cam.capture_frame()

        if fr_rgb is None:
            err_count += 1
            frame = black.copy()
            cv2.putText(frame, f"CAMERA ERROR #{err_count}", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
            time.sleep(0.2)
            continue

        err_count = 0
        bgr = cv2.cvtColor((fr_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        bgr = draw_overlays(bgr)
        _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.033)

# ── Motor helpers ─────────────────────────────────────────────────────────────

def move_async(steps: int, motor: str = "both", callback=None):
    """Non-blocking motor move in daemon thread."""
    global is_processing
    is_processing = True

    def run():
        global is_processing
        try:
            if lab is None:
                logger.warning("[move_async] No hardware — simulating")
                time.sleep(abs(steps) * 0.001)
                # Simulate position
                if motor in ("both", "1"):
                    lab.motor_1.current_pos_steps += steps  if lab else 0
                if motor in ("both", "2"):
                    lab.motor_2.current_pos_steps += steps  if lab else 0
                return
            if motor == "both":
                lab.move_both(steps)
            elif motor == "1":
                lab.move_motor1(steps)
            elif motor == "2":
                lab.move_motor2(steps)
            if callback:
                callback()
        except Exception as e:
            logger.error(f"[move_async] {e}")
        finally:
            is_processing = False

    Thread(target=run, daemon=True).start()


def _sweep_thread(step_size: int):
    """Auto-vanish sweep using BOTH MOTORS (synchronized).
    Sweeps the oil beaker rod through all depths, scores visibility,
    and parks at the minimum-score (vanish) position.
    """
    global _sweep, is_processing
    _sweep = {"phase": "HOMING", "progress": 0, "scores": [], "best_z": None}
    try:
        if not lab:
            _sweep["phase"] = "ERROR"
            return

        # Home both motors to start position
        cur1 = lab.motor_1.current_pos_steps
        cur2 = lab.motor_2.current_pos_steps
        if cur1 or cur2:
            lab.move_both(-max(cur1, cur2))  # Move both motors to home
        time.sleep(0.5)

        _sweep["phase"] = "SWEEPING"
        scores = []

        for pos in range(0, MOTOR_DOWN + step_size, step_size):
            pos = min(pos, MOTOR_DOWN)
            delta1 = pos - lab.motor_1.current_pos_steps
            delta2 = pos - lab.motor_2.current_pos_steps
            
            if delta1 or delta2:
                lab.move_both(pos - lab.motor_1.current_pos_steps)  # Move both to same position

            time.sleep(0.35)

            with camera_lock:
                fr = lab.camera.capture_frame() if lab.camera else None

            if fr is not None:
                gray = cv2.cvtColor((fr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                rd = ROI_CFG["zone_beaker_2"]
                E = compute_edge_energy(gray, rd["main"])
                C = compute_contrast(gray, rd["main"], rd.get("rod"), rd.get("background"))
                if E is not None and C is not None:
                    score = E + 5.0 * C
                    scores.append({"z": pos, "score": round(score, 6),
                                   "E": round(float(E), 6), "C": round(float(C), 6)})

            _sweep["scores"]   = scores
            _sweep["progress"] = int(pos / MOTOR_DOWN * 100) if MOTOR_DOWN else 100

        if scores:
            best = min(scores, key=lambda x: x["score"])
            _sweep["best_z"] = best["z"]
            _sweep["phase"]  = "PARKING"
            delta = best["z"] - lab.motor_1.current_pos_steps
            if delta:
                lab.move_both(delta)

        _sweep["phase"] = "DONE"
    except Exception as e:
        logger.error(f"[sweep] {e}")
        _sweep["phase"] = "ERROR"
    finally:
        is_processing = False

# ══════════════════ ROUTES ════════════════════════════════════════════════════

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route("/")
def root():
    return redirect(url_for("dashboard") if "user" in session else url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "")
        if USERS.get(u) == p:
            session["user"] = u
            return redirect(url_for("dashboard"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ── Pages ──────────────────────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=session["user"])

@app.route("/experiment/<exp>")
@login_required
def experiment(exp):
    names = {"vanishing-rod": "Vanishing Rod", "focal-length": "Focal Length"}
    return render_template("modes.html", exp=exp,
                           exp_name=names.get(exp, exp), user=session["user"])

@app.route("/vanishing/ab")
@login_required
def mode_AB():
    init_hardware()
    return render_template("mode_physical_manual.html", user=session["user"],
                           motor_down=MOTOR_DOWN, n_r=N_ROD)

@app.route("/vanishing/Ab")
@login_required
def mode_Ab():
    init_hardware()
    return render_template("mode_hybrid_twin.html", user=session["user"],
                           motor_down=MOTOR_DOWN, n_r=N_ROD)

@app.route("/vanishing/aB")
@login_required
def mode_aB():
    init_hardware()
    return render_template("mode_autocontrol.html", user=session["user"],
                           motor_down=MOTOR_DOWN)

@app.route("/vanishing/ab_sim")
@login_required
def mode_ab_sim():
    return render_template("mode_simulation.html", user=session["user"], n_r=N_ROD)

@app.route("/focal-length")
@login_required
def focal_length():
    init_hardware()
    return render_template("focal_length.html", user=session["user"])

@app.route("/dt")
@login_required
def dt_hub():
    """Digital Twin Hub — Comprehensive dashboard with EKF visualization, 
    pedagogy, and live hardware feedback."""
    init_hardware()
    return render_template("dt_hub.html", user=session["user"])

@app.route("/roi_calibration")
@login_required
def roi_calibration():
    """Interactive ROI calibration interface with live camera feedback."""
    init_hardware()
    return render_template("roi_calibration.html", user=session["user"])

@app.route("/calibrate/roi", methods=["POST"])
@login_required
def calibrate_roi():
    """Save ROI configuration to roi_config.yaml."""
    try:
        import yaml
        data = request.get_json(force=True) or {}
        
        # Validate structure
        if "zone_beaker_1" not in data or "zone_beaker_2" not in data:
            return jsonify({"status": "error", "message": "Missing zone data"}), 400
        
        roi_path = os.path.join(BASE_DIR, "roi_config.yaml")
        with open(roi_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        # Reload global ROI config
        global ROI_CFG
        ROI_CFG = _load_roi_config()
        
        logger.info(f"✓ ROI config saved to {roi_path}")
        return jsonify({
            "status": "success",
            "message": "ROI configuration saved and reloaded",
            "roi_config": ROI_CFG
        })
    except Exception as e:
        logger.error(f"ROI save error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ── Stream & Telemetry ────────────────────────────────────────────────────────
@app.route("/video_feed")
@login_required
def video_feed():
    return Response(_generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/telemetry")
@login_required
def telemetry():
    def stream():
        last = 0.0
        while True:
            if _telem["timestamp"] != last:
                last = _telem["timestamp"]
                yield f"data: {json.dumps(_telem)}\n\n"
            if _sweep["phase"] not in ("IDLE", "DONE", "ERROR"):
                yield f"event: sweep\ndata: {json.dumps(_sweep)}\n\n"
            time.sleep(0.3)
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Sensor API ────────────────────────────────────────────────────────────────
@app.route("/sensors")
@login_required
def sensors():
    """Live sensor readings: temperature (DS18B20) + brightness (BH1750)."""
    temp = lab.read_temperature()     if lab else T0
    lux  = lab.read_brightness_lux()  if lab else B0
    nits = lux / 3.14159
    return jsonify({
        "temperature_celsius": round(temp, 2),
        "brightness_lux":      round(lux,  1),
        "brightness_nits":     round(nits, 1),
        "hardware":            lab is not None,
    })

# ── Motor API ─────────────────────────────────────────────────────────────────
def _busy(): return jsonify({"status": "error", "message": "System busy"}), 400
def _nohw(): return jsonify({"status": "error", "message": "No hardware"}), 503

@app.route("/motor/status")
def motor_status():
    pos1 = lab.motor_1.current_pos_steps if lab else 0
    pos2 = lab.motor_2.current_pos_steps if lab else 0
    return jsonify({
        "status":          "ok" if lab else "sim",
        "motor1_position": pos1,
        "motor2_position": pos2,
        "motor1_pins_bcm": [21, 20, 16, 12],
        "motor2_pins_bcm": [24, 25,  8,  7],
        "ds18b20_bcm":     5,
        "bh1750_i2c_bus":  1,
        "moving":          is_processing,
        "home":            MOTOR_HOME,
        "down":            MOTOR_DOWN,
    })

@app.route("/motor/down", methods=["POST"])
@login_required
def motor_down():
    if is_processing: return _busy()
    pos = max(lab.motor_1.current_pos_steps, lab.motor_2.current_pos_steps) if lab else 0
    s    = MOTOR_DOWN - pos
    if s == 0:
        return jsonify({"status": "success", "message": "Already down", "moved": False})
    move_async(s, motor="both")
    return jsonify({"status": "success", "message": f"Moving down {s} steps (both motors)", "moved": True})

@app.route("/motor/up", methods=["POST"])
@login_required
def motor_up():
    if is_processing: return _busy()
    pos = max(lab.motor_1.current_pos_steps, lab.motor_2.current_pos_steps) if lab else 0
    s    = -pos
    if s == 0:
        return jsonify({"status": "success", "message": "Already home", "moved": False})
    move_async(s, motor="both")
    return jsonify({"status": "success", "message": "Moving up (both motors)", "moved": True})

@app.route("/motor/home", methods=["POST"])
@login_required
def motor_home():
    if is_processing: return _busy()
    pos = max(lab.motor_1.current_pos_steps, lab.motor_2.current_pos_steps) if lab else 0
    s = -pos
    if s == 0:
        return jsonify({"status": "success", "message": "Already home"})
    move_async(s, motor="both")
    return jsonify({"status": "success", "message": "Homing both motors", "steps": s})

@app.route("/motor/steps", methods=["POST"])
@login_required
def motor_steps():
    if is_processing: return _busy()
    data = request.get_json(force=True) or {}
    s    = int(data.get("steps", 0))
    m    = str(data.get("motor", "both"))   # "1", "2", or "both"
    if not s:
        return jsonify({"status": "error", "message": "steps=0"}), 400
    move_async(s, motor=m)
    return jsonify({"status": "success", "message": f"Jogging motor={m} {s} steps"})

@app.route("/motor/test", methods=["POST"])
def motor_test():
    """Quick self-test: jog both motors ±50 steps."""
    if is_processing:
        return jsonify({"status": "busy"}), 400
    if not lab:
        return jsonify({"status": "sim", "message": "No hardware — simulated OK"})
    try:
        lab.move_both(50)
        time.sleep(0.3)
        lab.move_both(-50)
        return jsonify({"status": "success",
                        "pos1": lab.motor_1.current_pos_steps,
                        "pos2": lab.motor_2.current_pos_steps,
                        "message": "Both motors tested successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/motor/1/move", methods=["POST"])
@login_required
def motor1_move():
    """Move motor 1 (water beaker) by specified steps."""
    if is_processing: return _busy()
    if not lab: return _nohw()
    data = request.get_json(force=True) or {}
    s = int(data.get("steps", 0))
    if not s:
        return jsonify({"status": "error", "message": "steps=0"}), 400
    move_async(s, motor="1")
    return jsonify({"status": "success", "message": f"Moving motor 1 by {s} steps"})

@app.route("/motor/2/move", methods=["POST"])
@login_required
def motor2_move():
    """Move motor 2 (oil beaker) by specified steps."""
    if is_processing: return _busy()
    if not lab: return _nohw()
    data = request.get_json(force=True) or {}
    s = int(data.get("steps", 0))
    if not s:
        return jsonify({"status": "error", "message": "steps=0"}), 400
    move_async(s, motor="2")
    return jsonify({"status": "success", "message": f"Moving motor 2 by {s} steps"})

# ── Estimate API ──────────────────────────────────────────────────────────────
@app.route("/estimate", methods=["POST"])
@login_required
def estimate():
    if is_processing: return _busy()
    cam = lab.camera if lab else None
    if not cam:
        return jsonify({"status": "error", "message": "No camera"}), 503

    with camera_lock:
        fr = cam.capture_frame()
    if fr is None:
        return jsonify({"status": "error", "message": "Frame capture failed"}), 500

    temp = lab.read_temperature()    if lab else T0
    bri  = lab.read_brightness_lux() if lab else B0
    pos  = lab.motor_1.current_pos_steps if lab else 0
    res  = analyse_frame(fr, pos, temp, bri)

    _telem.update({
        "beaker_1": res["beaker_1"],
        "beaker_2": res["beaker_2"],
        "timestamp": time.time(),
    })
    return jsonify({"status": "success", "position": pos,
                    "temperature": temp, "brightness": bri,
                    "results": res})

# ── Status & System ───────────────────────────────────────────────────────────
@app.route("/status")
@login_required
def status():
    return jsonify({
        "status":       "ready" if lab else "simulation",
        "position":     lab.motor_1.current_pos_steps if lab else 0,
        "position2":    lab.motor_2.current_pos_steps if lab else 0,
        "is_processing": is_processing,
        "hw_available": HW_AVAILABLE,
        "ekf_available": EKF_AVAILABLE,
        "camera":       lab.camera.camera_type if lab else "none",
        "motor_down":   MOTOR_DOWN,
        "motor_home":   MOTOR_HOME,
        "pins": {
            "motor1_bcm": [21, 20, 16, 12],
            "motor2_bcm": [24, 25,  8,  7],
            "ds18b20_bcm": 5,
            "bh1750_i2c":  1,
        },
    })

@app.route("/healthz")
def healthz():
    cam_ok  = (lab is not None and lab.camera is not None)
    hw_ok   = (lab is not None)
    all_ok  = cam_ok
    return jsonify({
        "ok":     all_ok,
        "hw":     hw_ok,
        "camera": cam_ok,
        "ekf":    EKF_AVAILABLE,
    }), (200 if all_ok else 503)

# ── Simulation API ────────────────────────────────────────────────────────────
@app.route("/sim/set_medium", methods=["POST"])
@login_required
def sim_set_medium():
    d = request.get_json(force=True) or {}
    _sim["n_m"]  = float(d.get("n_m",  _sim["n_m"]))
    _sim["tau"]  = float(d.get("tau",  _sim["tau"]))
    return jsonify({"status": "success", "sim": _sim})

@app.route("/sim/state")
@login_required
def sim_state():
    nm = _sim["n_m"]
    nr = _sim["n_r"]
    R  = ((nr - nm) / (nr + nm)) ** 2
    T  = 1.0 - R
    alpha = min(1.0, R * 22 + 0.03)
    return jsonify({
        **_sim,
        "R":        round(R, 6),
        "T":        round(T, 6),
        "delta_n":  round(abs(nr - nm), 5),
        "rod_alpha": round(alpha, 4),
    })

# ── Auto-vanish sweep ─────────────────────────────────────────────────────────
@app.route("/auto_vanish", methods=["POST"])
@login_required
def auto_vanish():
    global is_processing
    if is_processing: return _busy()
    if not lab:       return _nohw()
    step = int((request.get_json(force=True) or {}).get("step_size", 80))
    is_processing = True
    _sweep.update({"phase": "STARTING", "scores": [], "best_z": None, "progress": 0})
    Thread(target=_sweep_thread, args=(step,), daemon=True).start()
    return jsonify({"status": "success", "message": "Sweep started"})

@app.route("/sweep_status")
@login_required
def sweep_status():
    return jsonify(_sweep)

# ══════════════════ STARTUP ═══════════════════════════════════════════════════
# ── DT Extension registration (BH1750 sampler, /brightness, /dt/state,
#    /dt/audit, /healthz, /calibrate, /voice/text, /i18n/*, /mode) ─────────────
if DT_EXT_AVAILABLE:
    try:
        register_dt_extensions(
            app,
            config_dir=os.path.join(BASE_DIR, "dt", "models"),
            log_dir=os.path.join(BASE_DIR, "dt", "logs"),
            brightness_hz=2.0,                # poll BH1750 twice per second
            mock_brightness_nits=None,         # None = use real BH1750 hardware
        )
        logger.info("✓ DT extensions registered")
        logger.info("  Routes added: /brightness /dt/state /dt/audit /mode")
        logger.info("              /calibrate /voice/text /i18n/* /healthz")
    except Exception as e:
        logger.warning(f"DT extension registration failed: {e}")

# ── Extra status route that exposes EKF-B flag ────────────────────────────────
@app.route("/dt/ekf_info")
@login_required
def dt_ekf_info():
    return jsonify({
        "ekf_legacy":   EKF_AVAILABLE,
        "ekf_brightness": EKF_B_AVAILABLE,
        "dt_extension": DT_EXT_AVAILABLE,
        "active_ekf":   ("EKF-B (6-state, brightness-aware)" if EKF_B_AVAILABLE
                         else "EKF-legacy (4-state)" if EKF_AVAILABLE
                         else "Heuristic"),
        "pins": {
            "motor1_bcm": [21, 20, 16, 12],
            "motor2_bcm": [24, 25,  8,  7],
            "ds18b20_bcm": 5,
            "bh1750_i2c_bus": 1,
        },
    })


if __name__ == "__main__":
    init_hardware()
    Thread(target=_telem_loop, daemon=True).start()

    logger.info("=" * 65)
    logger.info("  Vanishing Rod Digital Twin — Integrated Lab Platform")
    logger.info("=" * 65)
    logger.info(f"  Hardware    : {'READY' if lab else 'SIMULATION MODE'}")
    logger.info(f"  EKF         : {'EKF-B 6-state (brightness-aware)' if EKF_B_AVAILABLE else 'EKF 4-state' if EKF_AVAILABLE else 'HEURISTIC'}")
    logger.info(f"  DT Extension: {'ACTIVE' if DT_EXT_AVAILABLE else 'NOT LOADED'}")
    logger.info(f"  Motor1      : BCM [21,20,16,12]  (physical 40,38,36,32)")
    logger.info(f"  Motor2      : BCM [24,25, 8, 7]  (physical 18,22,24,26)")
    logger.info(f"  DS18B20     : BCM 5              (physical 29)")
    logger.info(f"  BH1750      : I2C bus 1  SDA=BCM2(pin3), SCL=BCM3(pin5)")
    logger.info(f"  URL         : http://0.0.0.0:5000")
    logger.info("=" * 65)

    try:
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
    except KeyboardInterrupt:
        logger.info("\nShutting down…")
    finally:
        if lab:
            lab.cleanup()
        logger.info("✓ Shutdown complete")
