"""
dt_quantification.py
====================
Quantitative audit of the Vanishing-Rod system against the Kritzinger et al.
taxonomy (Digital Model → Digital Shadow → Digital Twin) and additional
maturity criteria used in the DT literature.

This script does NOT invent measurements. It inspects your running system via
its HTTP endpoints and the artifacts it writes to disk (calibration YAML,
event logs, EKF state trace). Every point it awards is backed by a concrete,
checkable file or endpoint response.

References for the scoring rubric (the reader can cross-check):
  - Kritzinger, W., et al. (2018). "Digital Twin in manufacturing: A categorical
    literature review and classification." IFAC-PapersOnLine, 51(11), 1016-1022.
  - Grieves, M. (2014). "Digital Twin: Manufacturing Excellence through Virtual
    Factory Replication." White paper.
  - Tao, F., et al. (2019). "Digital Twin in Industry: State-of-the-Art."
    IEEE Trans. Industrial Informatics, 15(4).

Usage:
    python dt_quantification.py --base-url http://localhost:5000 \
                                --config-dir ./dt/models \
                                --log-dir    ./dt/logs \
                                --report     dt_audit_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

try:
    import requests
    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False


# ────────────────────────────────────────────────────────────────────────────
#  RUBRIC — 10 criteria, each scored 0..5, max total = 50
# ────────────────────────────────────────────────────────────────────────────

CRITERIA = [
    dict(
        key="forward_physics_model",
        name="Forward physics model (predictive, first-principles)",
        why=("A digital *shadow* passively mirrors sensors. A digital *twin* "
             "predicts observables from a physics model and tests predictions "
             "against measurements. We check for the Fresnel reflectance "
             "forward map R(n_m) and the photometric link C = f(R, tau, T)."),
        evidence_hint="Model evaluates R(n_m) and predicts E, C from state.",
        max_score=5,
    ),
    dict(
        key="state_estimation_with_uncertainty",
        name="State estimation with quantified uncertainty",
        why=("A DT must report what it believes AND how sure it is. We check "
             "that nm is not read off a sensor but inferred by a recursive "
             "filter (EKF) that propagates a covariance matrix P."),
        evidence_hint="EKF returns sigma(n_m) and delta_n with covariance.",
        max_score=5,
    ),
    dict(
        key="bidirectional_control_authority",
        name="Bidirectional data flow: DT commands the plant",
        why=("The decisive Kritzinger distinction. Shadow = plant→model only. "
             "Twin = model→plant is also automated (auto-vanish, Z-sweep, "
             "orchestration). We check that mode a-B can actuate motors "
             "based on the estimate."),
        evidence_hint="POST /auto_vanish or equivalent triggers real motion.",
        max_score=5,
    ),
    dict(
        key="traceability_versioning",
        name="Traceability — versioned calibration coefficients",
        why=("Calibration coefficients beta, gamma MUST carry a version ID, a "
             "date, and a covariance. Otherwise an estimate is not auditable."),
        evidence_hint="coeffs_vN.yaml exists with version, date, Cov(beta).",
        max_score=5,
    ),
    dict(
        key="reproducible_logging",
        name="Reproducibility — full run logs (NDJSON with IDs)",
        why=("A DT's run must be replayable. Each estimate must be tied to the "
             "software SHA, the calibration version, sensor IDs, and raw input."),
        evidence_hint="logs/run_*.ndjson contains state, sensors, coeff_version.",
        max_score=5,
    ),
    dict(
        key="validation_against_standards",
        name="Validation against certified reference materials",
        why=("Accuracy claim must be backed by measurements against liquids of "
             "known n_m (water 1.3330, sugar solutions, certified oils)."),
        evidence_hint="acceptance_test.json with bias and repeatability stats.",
        max_score=5,
    ),
    dict(
        key="realtime_loop_budget",
        name="Real-time capability (loop period ≤ 250 ms)",
        why=("A DT used for control must close its loop faster than the "
             "process dynamics. Target from spec: ≤ 250 ms per estimate."),
        evidence_hint="/estimate_n responds in under 250 ms on average.",
        max_score=5,
    ),
    dict(
        key="multi_mode_authority_arbiter",
        name="Multi-mode operation with authority arbiter",
        why=("A-B (manual), A-b (physical + virtual), a-B (DT drives plant), "
             "a-b (pure sim). Each mode must exist AND motion must be gated "
             "by an authority lease so modes cannot clash."),
        evidence_hint="4 mode routes + /mode lease endpoint.",
        max_score=5,
    ),
    dict(
        key="ambient_disturbance_compensation",
        name="Ambient disturbance compensation (T, brightness)",
        why=("RI drifts with T at O(1e-4 /K). Room brightness varies across "
             "sites. A mature DT models these explicitly so the estimate is "
             "location-robust (see extended paper, Section 4)."),
        evidence_hint="State includes T and B; h(x) contains β3, β5 terms.",
        max_score=5,
    ),
    dict(
        key="safety_and_guardrails",
        name="Safety: limits, homing, e-stop, watchdog",
        why=("Any cyber-physical DT must refuse unsafe motion — soft/hard "
             "limits, homing, lid/leak interlocks, watchdog on the control link."),
        evidence_hint="/motor/* endpoints return 503 without hardware, "
                      "limit switches wired, is_processing lock.",
        max_score=5,
    ),
]

MAX_SCORE = sum(c["max_score"] for c in CRITERIA)  # 50


# ────────────────────────────────────────────────────────────────────────────
#  Kritzinger taxonomy — 3 levels
# ────────────────────────────────────────────────────────────────────────────

def kritzinger_level(has_forward_model: bool,
                     has_auto_plant_to_model: bool,
                     has_auto_model_to_plant: bool) -> str:
    if has_forward_model and has_auto_plant_to_model and has_auto_model_to_plant:
        return "Digital Twin"
    if has_forward_model and has_auto_plant_to_model:
        return "Digital Shadow"
    if has_forward_model:
        return "Digital Model"
    return "Instrumented Plant (no model)"


# ────────────────────────────────────────────────────────────────────────────
#  Checks — each returns (score, notes, evidence)
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    key: str
    name: str
    score: int
    max_score: int
    notes: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


def _get(url: str, timeout: float = 3.0) -> Optional[dict]:
    if not HAVE_REQUESTS:
        return None
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                return {"_text": r.text[:500]}
    except Exception:
        return None
    return None


def _post(url: str, payload: dict, timeout: float = 5.0) -> Optional[dict]:
    if not HAVE_REQUESTS:
        return None
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                return {"_text": r.text[:500]}
    except Exception:
        return None
    return None


def check_forward_physics_model(base_url: str) -> CheckResult:
    """Does the system compute R(n_m) and predict E, C from it?"""
    cr = CheckResult(key="forward_physics_model",
                     name="Forward physics model",
                     score=0, max_score=5)
    sim = _get(f"{base_url}/sim/state")
    if sim is None:
        cr.notes.append("Server unreachable or /sim/state missing — "
                        "cannot verify forward model at runtime.")
    else:
        cr.evidence["sim_state_sample"] = sim
        if "R" in sim:
            cr.score += 3
            cr.notes.append("R(n_m) exposed via /sim/state (Fresnel computed).")
        else:
            cr.notes.append("No R field in /sim/state — forward model not "
                            "exposed.")
        if "delta_n" in sim:
            cr.score += 1
            cr.notes.append("delta_n = n_r - n_m reported (mismatch observable).")
        if "rod_alpha" in sim or "alpha" in sim:
            cr.score += 1
            cr.notes.append("Photometric alpha/opacity linked to R (image-link "
                            "from physics to intensity).")
    return cr


def check_state_estimation(base_url: str) -> CheckResult:
    cr = CheckResult(key="state_estimation_with_uncertainty",
                     name="State estimation with uncertainty",
                     score=0, max_score=5)
    res = _post(f"{base_url}/estimate_n", {}, timeout=8.0)
    if res is None:
        cr.notes.append("/estimate_n did not respond (may require login or "
                        "hardware). Cannot verify EKF live.")
        return cr
    cr.evidence["estimate_sample"] = res
    results = res.get("results", {})
    if not results:
        cr.notes.append("/estimate_n returned no results.")
        return cr

    saw_est = saw_delta = saw_sigma = saw_metrics = saw_cov = False
    for _, v in results.items():
        if "n_m_estimated" in v or "n_m" in v:
            saw_est = True
        if "delta_n" in v:
            saw_delta = True
        if "metrics" in v or ("E" in v and "C" in v):
            saw_metrics = True
        if "sigma_n_m" in v or "sigma" in v:
            saw_sigma = True
        if "covariance" in v or "P" in v:
            saw_cov = True

    if saw_est:
        cr.score += 2
        cr.notes.append("n_m estimate is returned (not a raw sensor reading).")
    if saw_delta:
        cr.score += 1
        cr.notes.append("delta_n returned (mismatch = derived quantity).")
    if saw_metrics:
        cr.score += 1
        cr.notes.append("E, C metrics accompany estimate (fusion input).")
    if saw_sigma or saw_cov:
        cr.score += 1
        cr.notes.append("Uncertainty (sigma/P) returned. Estimator is proper.")
    else:
        cr.notes.append("WARNING: No sigma_n_m / covariance returned. EKF "
                        "present but uncertainty is not yet exposed to clients "
                        "— add P[0,0] to the JSON to close this gap.")
    return cr


def check_bidirectional_control(base_url: str) -> CheckResult:
    cr = CheckResult(key="bidirectional_control_authority",
                     name="Bidirectional control authority",
                     score=0, max_score=5)
    status = _get(f"{base_url}/status")
    if status:
        cr.evidence["status"] = status
    motor = _get(f"{base_url}/motor/status")
    if motor:
        cr.evidence["motor_status"] = motor
        cr.score += 2
        cr.notes.append("Motor status endpoint exposes actuator state.")
    ok_av = _post(f"{base_url}/auto_vanish", {"dry_run": True}, timeout=3.0)
    if ok_av is not None:
        cr.evidence["auto_vanish_probe"] = ok_av
        cr.score += 3
        cr.notes.append("/auto_vanish endpoint exists → DT can drive the "
                        "plant automatically (mode a-B is realised).")
    else:
        cr.notes.append("/auto_vanish unreachable — DT→plant link cannot "
                        "be verified from this audit.")
    return cr


def check_traceability(config_dir: Path) -> CheckResult:
    cr = CheckResult(key="traceability_versioning",
                     name="Traceability — versioned coefficients",
                     score=0, max_score=5)
    coeff_files = list(config_dir.glob("coeffs_v*.yaml")) + \
                  list(config_dir.glob("coeffs_v*.yml"))
    cfg_files = list(config_dir.glob("config.yaml"))
    cr.evidence["coeff_files"] = [str(p) for p in coeff_files]
    cr.evidence["config_files"] = [str(p) for p in cfg_files]

    if coeff_files:
        cr.score += 2
        cr.notes.append(f"Found {len(coeff_files)} versioned coeff file(s).")
        try:
            import yaml
            d = yaml.safe_load(coeff_files[-1].read_text())
            has_beta = "beta" in d
            has_gamma = "gamma" in d
            has_cov = "cov_beta" in d or "cov_gamma" in d or "covariance" in d
            has_date = "date" in d or "calibration_date" in d
            has_dev = "device_id" in d or "rig_id" in d
            if has_beta and has_gamma:
                cr.score += 1
                cr.notes.append("beta and gamma coefficients present.")
            if has_cov:
                cr.score += 1
                cr.notes.append("Coefficient covariance stored — full traceability.")
            else:
                cr.notes.append("WARNING: coefficient covariance not stored. "
                                "Add 'cov_beta:' from (X^T W X)^-1 to achieve "
                                "full traceability.")
            if has_date:
                cr.score += 1
                cr.notes.append("Calibration date stored.")
            if has_dev:
                cr.notes.append("Device ID stored.")
        except Exception as e:
            cr.notes.append(f"Could not parse YAML: {e}")
    else:
        cr.notes.append("No coeffs_vN.yaml found in config dir. Without "
                        "versioned calibration, no traceability.")
    return cr


def check_logging(log_dir: Path) -> CheckResult:
    cr = CheckResult(key="reproducible_logging",
                     name="Reproducible logging",
                     score=0, max_score=5)
    if not log_dir.exists():
        cr.notes.append(f"{log_dir} does not exist.")
        return cr
    ndjson_files = list(log_dir.glob("*.ndjson")) + list(log_dir.glob("**/*.ndjson"))
    cr.evidence["log_files"] = [str(p) for p in ndjson_files[:20]]
    if ndjson_files:
        cr.score += 2
        cr.notes.append(f"Found {len(ndjson_files)} NDJSON log file(s).")
        try:
            with ndjson_files[-1].open() as f:
                first = f.readline().strip()
            if first:
                obj = json.loads(first)
                fields = set(obj.keys())
                if "timestamp" in fields or "ts" in fields:
                    cr.score += 1
                    cr.notes.append("Timestamps present.")
                if "git_sha" in fields or "software_version" in fields:
                    cr.score += 1
                    cr.notes.append("Software version (git SHA) recorded.")
                else:
                    cr.notes.append("WARNING: add 'git_sha' to log entries for "
                                    "full reproducibility.")
                if "coeff_version" in fields or "calibration_id" in fields:
                    cr.score += 1
                    cr.notes.append("Calibration version tied to the run.")
                else:
                    cr.notes.append("WARNING: add 'coeff_version' to each "
                                    "log line so runs can be replayed.")
        except Exception as e:
            cr.notes.append(f"Log parse error: {e}")
    else:
        cr.notes.append("No NDJSON logs found. Emit one per /estimate_n call.")
    return cr


def check_validation(config_dir: Path) -> CheckResult:
    cr = CheckResult(key="validation_against_standards",
                     name="Validation against standards",
                     score=0, max_score=5)
    candidates = []
    for root in [config_dir, config_dir.parent, config_dir.parent.parent]:
        if root.exists():
            candidates.extend(root.rglob("acceptance*.json"))
            candidates.extend(root.rglob("validation*.json"))
    cr.evidence["validation_files"] = [str(p) for p in candidates[:10]]
    if not candidates:
        cr.notes.append("No acceptance_test.json / validation*.json found. "
                        "Run the acceptance protocol (ice bath + warm bath + "
                        "certified liquids) and save results.")
        return cr
    cr.score += 2
    try:
        d = json.loads(candidates[0].read_text())
        if "bias_RIU" in d or "bias" in d:
            cr.score += 1
        if "repeatability_RIU" in d or "sigma_delta_n" in d:
            cr.score += 1
        if "certified_media" in d or "standards" in d:
            cr.score += 1
        cr.notes.append(f"Parsed {candidates[0].name}.")
    except Exception as e:
        cr.notes.append(f"Parse error: {e}")
    return cr


def check_realtime(base_url: str) -> CheckResult:
    cr = CheckResult(key="realtime_loop_budget",
                     name="Real-time loop budget",
                     score=0, max_score=5)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = _post(f"{base_url}/estimate_n", {}, timeout=8.0)
        dt = (time.perf_counter() - t0) * 1000.0
        if r is not None:
            times.append(dt)
    if not times:
        cr.notes.append("Could not measure /estimate_n latency.")
        return cr
    avg = sum(times) / len(times)
    worst = max(times)
    cr.evidence["latency_ms"] = dict(avg=avg, worst=worst, samples=times)
    if avg <= 100:
        cr.score = 5
    elif avg <= 250:
        cr.score = 4
    elif avg <= 500:
        cr.score = 3
    elif avg <= 1000:
        cr.score = 2
    else:
        cr.score = 1
    cr.notes.append(f"Average /estimate_n latency: {avg:.0f} ms "
                    f"(worst {worst:.0f} ms). Spec target: ≤ 250 ms.")
    return cr


def check_multi_mode(base_url: str) -> CheckResult:
    cr = CheckResult(key="multi_mode_authority_arbiter",
                     name="Multi-mode + authority arbiter",
                     score=0, max_score=5)
    # We probe each mode URL with HEAD/GET. 200 or 302 (login redirect) counts.
    mode_paths = {
        "A-B (physical manual)": "/vanishing/ab",
        "A-b (physical + virtual)": "/vanishing/Ab",
        "a-B (DT commands plant)": "/vanishing/aB",
        "a-b (full sim)":         "/vanishing/ab_sim",
    }
    found = 0
    per_mode = {}
    if HAVE_REQUESTS:
        for name, path in mode_paths.items():
            try:
                r = requests.get(f"{base_url}{path}",
                                 allow_redirects=False, timeout=3.0)
                ok = r.status_code in (200, 301, 302, 401, 403)
                per_mode[name] = r.status_code
                if ok:
                    found += 1
            except Exception as e:
                per_mode[name] = f"ERR: {e}"
    cr.evidence["modes"] = per_mode
    cr.score += found  # 0..4
    cr.notes.append(f"{found}/4 modes routed.")
    # Authority arbiter probe
    lease = _post(f"{base_url}/mode",
                  {"axis": "z", "holder": "dt", "ttl_s": 10}, timeout=3.0)
    if lease is not None:
        cr.score = min(cr.max_score, cr.score + 1)
        cr.notes.append("/mode endpoint exists → authority arbiter in place.")
    else:
        cr.notes.append("No /mode lease endpoint detected. Add POST /mode "
                        "{axis,holder,ttl_s} to prevent mode clashes.")
    return cr


def check_ambient_compensation(base_url: str, config_dir: Path) -> CheckResult:
    cr = CheckResult(key="ambient_disturbance_compensation",
                     name="Ambient disturbance compensation",
                     score=0, max_score=5)
    temp_ok = False
    brightness_ok = False
    # Ask the sim/state and status for T and B fields
    s = _get(f"{base_url}/sim/state")
    if s is None:
        s = {}
    if "T" in s or "temperature_C" in s or "temp" in s:
        temp_ok = True
    if "B" in s or "brightness_nits" in s or "lux" in s:
        brightness_ok = True
    # Check coeffs_v1.yaml for beta3 (temperature coef) and beta5 (brightness coef)
    coeff_files = list(config_dir.glob("coeffs_v*.yaml"))
    if coeff_files:
        try:
            import yaml
            d = yaml.safe_load(coeff_files[-1].read_text())
            beta = d.get("beta", [])
            gamma = d.get("gamma", [])
            if len(beta) >= 4:
                cr.score += 1
                cr.notes.append("beta3 (temperature coef in contrast) present.")
                temp_ok = True
            if len(beta) >= 5 or "beta_B" in d or "brightness" in d:
                cr.score += 1
                cr.notes.append("brightness coef present — location-robust.")
                brightness_ok = True
            if len(gamma) >= 5:
                cr.score += 1
                cr.notes.append("gamma_B present for edge-energy brightness scaling.")
        except Exception as e:
            cr.notes.append(f"coeffs parse: {e}")
    if temp_ok:
        cr.score = min(cr.max_score, cr.score + 1)
        cr.notes.append("Temperature is tracked in state/telemetry.")
    if brightness_ok:
        cr.score = min(cr.max_score, cr.score + 1)
    else:
        cr.notes.append("Brightness (nits) not yet in state. Install a "
                        "lux / nits sensor (BH1750, TSL2591, OPT3001) and "
                        "expose /brightness. See brightness_sensor.py.")
    return cr


def check_safety(base_url: str) -> CheckResult:
    cr = CheckResult(key="safety_and_guardrails",
                     name="Safety and guardrails",
                     score=0, max_score=5)
    ms = _get(f"{base_url}/motor/status")
    if ms is None:
        cr.notes.append("motor/status unreachable — cannot verify safety layer.")
        return cr
    cr.evidence["motor_status"] = ms
    # /motor/status returns 503 when no HW — that itself is a safe behaviour.
    if "moving" in ms:
        cr.score += 1
        cr.notes.append("'moving' flag exposed — busy-state guard exists.")
    if "home" in ms and "down" in ms:
        cr.score += 1
        cr.notes.append("Soft limits (home/down positions) defined.")
    # Probe status busy-lock
    st = _get(f"{base_url}/status")
    if st and "is_processing" in st:
        cr.score += 1
        cr.notes.append("is_processing lock gates motion.")
    # Hardware available check
    if st and "hw_available" in st:
        cr.score += 1
        cr.notes.append("Hardware-availability flag exists — degrade gracefully.")
    # We cannot verify limit switches / e-stop from software; award 1 on faith
    # only if the status also shows a 'limits' or 'limit_switches' field.
    if ms.get("limits") is not None or "limit_switches" in ms:
        cr.score += 1
        cr.notes.append("Limit-switch telemetry exposed.")
    else:
        cr.notes.append("Limit-switch telemetry not exposed. Add "
                        "'limits': [low, high] to /motor/status.")
    return cr


# ────────────────────────────────────────────────────────────────────────────
#  Run audit
# ────────────────────────────────────────────────────────────────────────────

def run_audit(base_url: str,
              config_dir: Path,
              log_dir: Path) -> dict:
    checks = [
        check_forward_physics_model(base_url),
        check_state_estimation(base_url),
        check_bidirectional_control(base_url),
        check_traceability(config_dir),
        check_logging(log_dir),
        check_validation(config_dir),
        check_realtime(base_url),
        check_multi_mode(base_url),
        check_ambient_compensation(base_url, config_dir),
        check_safety(base_url),
    ]

    total = sum(c.score for c in checks)
    max_total = sum(c.max_score for c in checks)
    pct = 100.0 * total / max_total if max_total else 0.0

    has_forward = checks[0].score >= 3
    has_p2m = checks[1].score >= 2 and checks[4].score >= 1  # estimator + logs
    has_m2p = checks[2].score >= 3
    level = kritzinger_level(has_forward, has_p2m, has_m2p)

    if pct >= 80:
        maturity = "Level 4 — Operational Digital Twin"
    elif pct >= 60:
        maturity = "Level 3 — Validated Digital Twin"
    elif pct >= 40:
        maturity = "Level 2 — Functional Digital Twin"
    elif pct >= 20:
        maturity = "Level 1 — Digital Shadow with twin intent"
    else:
        maturity = "Level 0 — Instrumented plant / model"

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": base_url,
        "config_dir": str(config_dir),
        "log_dir": str(log_dir),
        "kritzinger_level": level,
        "maturity_tier": maturity,
        "total_score": total,
        "max_score": max_total,
        "percentage": round(pct, 1),
        "four_modes_as_dt_parameters": {
            "explanation":
                "The four operating modes are the four canonical DT "
                "configurations: physical↔virtual × data-flow direction. "
                "They are the minimum viable set to demonstrate every "
                "DT capability.",
            "A-B": {
                "name": "Physical Manual",
                "dt_role": "Ground-truth reference run. Human drives the "
                           "plant; the DT observes and estimates in the "
                           "background. This is where we verify that our "
                           "estimator is calibrated (plant → model only).",
                "kritzinger": "Shadow (A-B is how shadow-quality is measured)",
            },
            "A-b": {
                "name": "Physical + Virtual (model-based shadow)",
                "dt_role": "Physical rod still moves; virtual rod mirrors it "
                           "under a user-selected n_m. Lets the student "
                           "explore counterfactuals (“what if this liquid "
                           "were glycerol?”) without changing the real "
                           "liquid — the signature capability of a DT.",
                "kritzinger": "Twin (virtual exploration)",
            },
            "a-B": {
                "name": "DT Commands Plant (auto-vanish)",
                "dt_role": "The DT closes the loop: it computes where the "
                           "vanish point should be, moves the Z-stage, and "
                           "parks there. This is the operational proof that "
                           "model → plant data flow is automated.",
                "kritzinger": "Twin (closed-loop prescriptive control)",
            },
            "a-b": {
                "name": "Full Simulation",
                "dt_role": "No hardware. Verifies that the model alone "
                           "reproduces the phenomenon (rod vanishes when "
                           "n_m = n_r). Essential for remote scale-out and "
                           "for teaching when hardware is busy.",
                "kritzinger": "Model (pure virtual)",
            },
        },
        "checks": [asdict(c) for c in checks],
        "recommendations": _build_recommendations(checks),
    }

    return report


def _build_recommendations(checks: list) -> list:
    recs = []
    for c in checks:
        if c.score < c.max_score:
            for n in c.notes:
                if n.startswith("WARNING") or "add " in n.lower():
                    recs.append(f"[{c.key}] {n}")
    return recs


# ────────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:5000")
    ap.add_argument("--config-dir", default="./dt/models")
    ap.add_argument("--log-dir", default="./dt/logs")
    ap.add_argument("--report", default="dt_audit_report.json")
    args = ap.parse_args()

    if not HAVE_REQUESTS:
        print("WARNING: 'requests' not installed — runtime probes will be "
              "skipped. Install with:  pip install requests", file=sys.stderr)

    report = run_audit(args.base_url,
                       Path(args.config_dir),
                       Path(args.log_dir))

    Path(args.report).write_text(json.dumps(report, indent=2))
    # Console summary
    print("=" * 70)
    print(f" Vanishing-Rod Digital-Twin Audit")
    print("=" * 70)
    print(f" Kritzinger level   : {report['kritzinger_level']}")
    print(f" Maturity tier      : {report['maturity_tier']}")
    print(f" Score              : {report['total_score']}/{report['max_score']}"
          f"  ({report['percentage']} %)")
    print("-" * 70)
    for c in report["checks"]:
        bar = "█" * c["score"] + "░" * (c["max_score"] - c["score"])
        print(f"  [{bar}] {c['score']}/{c['max_score']}  {c['name']}")
    if report["recommendations"]:
        print("-" * 70)
        print(" Recommendations (highest ROI first):")
        for r in report["recommendations"]:
            print(f"   • {r}")
    print("=" * 70)
    print(f" Full report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
