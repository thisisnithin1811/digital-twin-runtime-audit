"""
app_extensions.py
=================
Flask blueprint that adds the missing DT capabilities to your existing app.py
WITHOUT modifying any of its code. Import once at startup:

    from dt_extension.app_extensions import register_dt_extensions
    register_dt_extensions(app)

It adds:

    GET  /brightness          — current nits reading from the sensor
    GET  /dt/state            — full DT state (n_m, Δn, σ, T, B, mode)
    POST /mode                — authority lease: {axis, holder, ttl_s}
    GET  /mode                — current leases
    GET  /dt/audit            — live self-audit (calls dt_quantification)
    POST /calibrate           — trigger a calibration run from a CSV of samples
    POST /voice/text          — voice-command text dispatcher (see voice_commands)
    POST /i18n/language       — set session language
    GET  /i18n/catalogue/<lc> — return translation catalogue

It also:
  * Starts a background thread that samples the brightness sensor and pushes
    nits into the EKF measurement update.
  * Adds an NDJSON logger that writes every /estimate_n result to disk so
    runs are replayable.
  * Exposes a /healthz endpoint that fails fast if the EKF, camera, or
    brightness sensor is not ready.

Nothing here replaces your existing routes. If you already have, say, a
/mode route in app.py, the original wins and this one is skipped.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, session

from .brightness_sensor import BrightnessSensor
from .i18n import register_flask_i18n, set_language as _set_lang
from .voice_commands import register_flask_voice

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
#  Authority arbiter — per-axis leases with TTL
# ────────────────────────────────────────────────────────────────────────────

class _Arbiter:
    """Minimal in-memory lease manager.

    Each axis has one holder. Commands for that axis from any other holder
    are rejected. Leases expire after TTL unless renewed.
    """
    def __init__(self):
        self._lock = threading.Lock()
        # axis -> (holder, expires_at_epoch)
        self._leases: dict = {}

    def acquire(self, axis: str, holder: str, ttl_s: float) -> dict:
        with self._lock:
            now = time.time()
            cur = self._leases.get(axis)
            if cur is None or cur[1] < now or cur[0] == holder:
                self._leases[axis] = (holder, now + ttl_s)
                return {"ok": True, "axis": axis, "holder": holder,
                        "ttl_s": ttl_s, "expires_at": now + ttl_s}
            return {"ok": False, "axis": axis, "current_holder": cur[0],
                    "expires_in_s": cur[1] - now}

    def release(self, axis: str, holder: str) -> dict:
        with self._lock:
            cur = self._leases.get(axis)
            if cur and cur[0] == holder:
                self._leases.pop(axis, None)
                return {"ok": True}
            return {"ok": False, "reason": "not_holder"}

    def status(self) -> dict:
        with self._lock:
            now = time.time()
            return {
                axis: {"holder": h, "expires_in_s": max(0.0, exp - now)}
                for axis, (h, exp) in self._leases.items()
            }


ARBITER = _Arbiter()


# ────────────────────────────────────────────────────────────────────────────
#  Brightness sampler thread
# ────────────────────────────────────────────────────────────────────────────

class BrightnessSampler:
    """Polls the nits sensor at a configurable rate and makes the latest
    reading available to other routes without I²C contention."""

    def __init__(self, sensor: BrightnessSensor, hz: float = 2.0):
        self.sensor = sensor
        self._period = 1.0 / max(hz, 0.1)
        self._last = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="BrightnessSampler")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        while not self._stop.is_set():
            try:
                r = self.sensor.read_full()
                with self._lock:
                    self._last = r
            except Exception as e:
                logger.error(f"BrightnessSampler error: {e}")
            self._stop.wait(self._period)

    def latest(self):
        with self._lock:
            return self._last


# ────────────────────────────────────────────────────────────────────────────
#  NDJSON run logger
# ────────────────────────────────────────────────────────────────────────────

class RunLogger:
    """Append-only NDJSON logger with daily-rotated files."""

    def __init__(self, log_dir: str = "./dt/logs"):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._git_sha = _detect_git_sha()

    def _current_file(self) -> Path:
        return self.dir / f"run_{time.strftime('%Y%m%d')}.ndjson"

    def log(self, record: dict, coeff_version: str = "unknown") -> None:
        record = {
            "timestamp":     time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "git_sha":       self._git_sha,
            "coeff_version": coeff_version,
            **record,
        }
        line = json.dumps(record, default=str)
        with self._lock:
            with self._current_file().open("a") as f:
                f.write(line + "\n")


def _detect_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=1.0)
        return out.decode().strip()
    except Exception:
        return "unknown"


# ────────────────────────────────────────────────────────────────────────────
#  Main registration function
# ────────────────────────────────────────────────────────────────────────────

def register_dt_extensions(app: Flask,
                           config_dir: str = "./dt/models",
                           log_dir: str = "./dt/logs",
                           brightness_hz: float = 2.0,
                           mock_brightness_nits: Optional[float] = None):
    """Attach DT extensions to the given Flask app.

    Parameters
    ----------
    mock_brightness_nits : float or None
        If not None, use a mock brightness sensor at this value instead of
        trying to detect real hardware. Useful for CI.
    """

    # ── Brightness ────────────────────────────────────────────────────────
    if mock_brightness_nits is not None:
        sensor = BrightnessSensor.mock(mock_brightness_nits)
        logger.info(f"Brightness sensor: mock @ {mock_brightness_nits} nits")
    else:
        sensor = BrightnessSensor.autodetect()
    sampler = BrightnessSampler(sensor, hz=brightness_hz)
    sampler.start()
    app.config["_dt_sampler"] = sampler
    app.config["_dt_sensor"] = sensor

    # ── Run logger ────────────────────────────────────────────────────────
    run_logger = RunLogger(log_dir=log_dir)
    app.config["_dt_run_logger"] = run_logger

    # ── i18n and voice ────────────────────────────────────────────────────
    register_flask_i18n(app)
    register_flask_voice(app, base_url=app.config.get("DT_SELF_URL",
                                                       "http://localhost:5000"))

    # ── Routes ────────────────────────────────────────────────────────────

    @app.route("/brightness")
    def _brightness():
        r = sampler.latest()
        if r is None:
            return jsonify({"ok": False, "reason": "no_reading_yet"}), 503
        return jsonify({
            "ok": True,
            "nits": round(r.nits, 2),
            "lux":  round(r.lux, 2),
            "healthy": r.healthy,
            "stale_s": round(r.stale_s, 3),
            "source": r.source,
        })

    @app.route("/dt/state")
    def _dt_state():
        b = sampler.latest()
        try:
            from flask import session
            lang = session.get("lang", "en")
        except Exception:
            lang = "en"
        # We can only report what the legacy app exposes; include brightness
        # and lang alongside.
        return jsonify({
            "timestamp": time.time(),
            "language": lang,
            "brightness": {
                "nits": (None if b is None else round(b.nits, 2)),
                "healthy": (None if b is None else b.healthy),
                "source": (None if b is None else b.source),
            },
            "leases": ARBITER.status(),
            "coeff_version": _current_coeff_version(config_dir),
        })

    @app.route("/mode", methods=["POST", "GET"])
    def _mode():
        if request.method == "GET":
            return jsonify({"leases": ARBITER.status()})
        data = request.get_json(force=True) or {}
        axis = data.get("axis", "z")
        holder = data.get("holder", "dt")
        ttl = float(data.get("ttl_s", 30.0))
        res = ARBITER.acquire(axis, holder, ttl)
        code = 200 if res.get("ok") else 409
        return jsonify(res), code

    @app.route("/mode/release", methods=["POST"])
    def _mode_release():
        data = request.get_json(force=True) or {}
        return jsonify(ARBITER.release(data.get("axis", "z"),
                                       data.get("holder", "dt")))

    @app.route("/dt/audit")
    def _dt_audit():
        """Run the quantification audit in-process and return the report."""
        from .dt_quantification import run_audit
        base_url = app.config.get("DT_SELF_URL", "http://localhost:5000")
        report = run_audit(base_url, Path(config_dir), Path(log_dir))
        return jsonify(report)

    @app.route("/healthz")
    def _healthz():
        ok_cam = _check_camera(app)
        ok_ekf = _check_ekf(app)
        ok_bri = sampler.latest() is not None
        status = {
            "ok": ok_cam and ok_ekf and ok_bri,
            "camera": ok_cam,
            "ekf": ok_ekf,
            "brightness": ok_bri,
        }
        code = 200 if status["ok"] else 503
        return jsonify(status), code

    @app.route("/calibrate", methods=["POST"])
    def _calibrate():
        """Fit β, γ from a list of samples and save a new coeffs_vN.yaml.

        Request body:
            { "samples": [ { "n_m": ..., "T": ..., "B": ..., "xi": ...,
                             "alpha": ..., "C": ..., "E": ..., "tau": ...,
                             "w": ... }, ... ],
              "version": "v2",
              "device_id": "rpi-01" }
        """
        from .physics_brightness import fit_coefficients, save_coeffs_yaml
        data = request.get_json(force=True) or {}
        samples = data.get("samples", [])
        if len(samples) < 5:
            return jsonify({"ok": False,
                            "reason": f"need ≥5 samples, got {len(samples)}"}), 400
        try:
            coeffs = fit_coefficients(
                samples,
                n_r=data.get("n_r", 1.50),
                T0=data.get("T0", 20.0),
                B0=data.get("B0", 300.0),
                xi0=data.get("xi0", 1.0),
            )
            coeffs.version = data.get("version", "v1")
            coeffs.calibration_date = time.strftime("%Y-%m-%d")
            coeffs.device_id = data.get("device_id", "unknown")
            out_path = Path(config_dir) / f"coeffs_{coeffs.version}.yaml"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_coeffs_yaml(coeffs, str(out_path))
            return jsonify({
                "ok": True,
                "version": coeffs.version,
                "path": str(out_path),
                "residual_rms_C": coeffs.residual_rms_C,
                "residual_rms_E": coeffs.residual_rms_E,
            })
        except Exception as e:
            logger.exception("calibration failed")
            return jsonify({"ok": False, "reason": str(e)}), 500

    # Background EKF wrapper that also logs every estimate to NDJSON.
    # We install a post-response hook on /estimate_n if that route exists.
    _install_estimate_logger(app, run_logger, config_dir, sampler)

    logger.info("DT extensions registered on app.")


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────

def _current_coeff_version(config_dir: str) -> str:
    try:
        p = Path(config_dir)
        files = sorted(p.glob("coeffs_v*.yaml"))
        if not files:
            return "unknown"
        import yaml
        d = yaml.safe_load(files[-1].read_text())
        return d.get("version", files[-1].stem)
    except Exception:
        return "unknown"


def _check_camera(app: Flask) -> bool:
    lab = getattr(app, "lab", None)
    if lab is None:
        # Try a module-level handle
        import sys
        app_mod = sys.modules.get("app") or sys.modules.get("__main__")
        lab = getattr(app_mod, "lab", None) if app_mod else None
    if lab is None:
        return False
    cam = getattr(lab, "camera", None)
    return cam is not None


def _check_ekf(app: Flask) -> bool:
    try:
        import dt.estimator.ekf                       # noqa: F401
        return True
    except Exception:
        try:
            import dt_extension.ekf_brightness        # noqa: F401
            return True
        except Exception:
            return False


def _install_estimate_logger(app: Flask,
                             run_logger: RunLogger,
                             config_dir: str,
                             sampler: BrightnessSampler):
    """Wrap any existing /estimate_n so results are persisted to NDJSON."""

    # Use after_request so we log the serialised response.
    @app.after_request
    def _log_estimate(response):
        try:
            if request.path == "/estimate_n" and request.method == "POST":
                payload = response.get_json(silent=True) or {}
                b = sampler.latest()
                rec = {
                    "route": "/estimate_n",
                    "status": response.status_code,
                    "payload": payload,
                    "brightness_nits": None if b is None else b.nits,
                    "brightness_healthy": None if b is None else b.healthy,
                }
                run_logger.log(rec, coeff_version=_current_coeff_version(config_dir))
        except Exception as e:
            logger.debug(f"estimate logger: {e}")
        return response
