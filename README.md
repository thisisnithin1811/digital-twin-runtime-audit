# Vanishing Rod — Digital Twin (Final)

Engineering-grade digital twin of the classical *vanishing rod* refractive-index
experiment, built on a Raspberry Pi with two 28BYJ-48 stepper motors, a
brightness-aware 6-state EKF, multilingual UI (8 languages, voice in Telugu &
Hindi), and a full suite of automated numerical experiments that produce
journal-ready evidence in a single command.

This repository contains everything needed to:
1. Run the live remote-lab on a Pi.
2. Reproduce every quantitative claim in the paper.
3. Audit the system against the standard DT maturity rubrics.

---

## 1. Quick start (3 commands)

```bash
# 1. install
pip install -r requirements.txt

# 2. run the lab server
./start.sh                                  # or: python3 app.py

# 3. (separate terminal) collect every numerical result for the paper
python3 numerical_experiments.py --all
```

When the experiments finish, the file `dt/results/run_YYYYMMDD_HHMMSS/summary_report.md`
contains every number you need. Paste straight into the manuscript.

---

## 2. Hardware — new PCB pinout

The system runs on a hand-soldered PCB with these pin assignments
(physical 1–40 header pin → BCM/GPIO number resolved by `pinmap.py`):

| Component | Physical pins | Resolves to BCM |
|---|---|---|
| ULN2003 #1 IN1..IN4 (Motor 1, Water) | 40, 38, 36, 32 | 21, 20, 16, 12 |
| ULN2003 #2 IN1..IN4 (Motor 2, Oil) | 18, 22, 24, 26 | 24, 25, 8, 7 |
| DS18B20 data | 29 | 5 |
| BH1750 SDA | 3 | 2 (I2C1 SDA) |
| BH1750 SCL | 5 | 3 (I2C1 SCL) |

`pinmap.py` is the single source of truth. Run it once after wiring:

```bash
python3 pinmap.py
```

It performs four sanity checks and tells you exactly what to do if anything
fails:
- All BCM pins distinct (no soldering mistake)
- SPI is **disabled** (motor 2 IN3/IN4 share GPIO 7/8 with SPI CE0/CE1)
- I2C is **enabled** (BH1750)
- `dtoverlay=w1-gpio,gpiopin=5` is in `/boot/firmware/config.txt` (DS18B20)

---

## 3. The four operating modes (DT canonical configurations)

| Mode | URL | What it does | DT role |
|---|---|---|---|
| **A-B** Physical Manual | `/vanishing/ab` | Camera + manual motor control. Human drives the rig. | Shadow / ground-truth |
| **A-b** Physical + Virtual | `/vanishing/Ab` | Real rod moves; virtual rod mirrors it under user-selected n_m. | Twin (counterfactual) |
| **a-B** DT Auto-Control | `/vanishing/aB` | DT computes vanish-point, drives the Z-stage, parks there. | Twin (closed-loop) |
| **a-b** Pure Simulation | `/vanishing/ab_sim` | No hardware needed. | Model |

Together they cover every cell of the *physical × virtual × data-flow direction*
matrix that defines a digital twin in the Kritzinger / DTC frameworks.

---

## 4. Numerical experiments (`numerical_experiments.py`)

Single CLI script. Produces every measurement your IF-12 reviewers will ask for.

### 4.1 What it runs

| Flag | Experiment | What it proves | Time |
|---|---|---|---|
| `--acceptance` | 10 trials × {water, oil} vs reference n_m | Estimator accuracy & repeatability | ~7 min |
| `--latency` | 200 `/estimate` calls | Real-time claim (mean, p50, p95, p99, loss) | ~5 min |
| `--auto-vanish` | 10 closed-loop sweeps | Bidirectional control success rate | ~10 min |
| `--brightness` | Estimates at 4 brightness levels (interactive) | Robustness to ambient light | ~10 min |
| `--nis` | 60 frames → σ stability | EKF tuning evidence | ~2 min |
| `--dt-comp` | Reviewer's C/O/R/S/B/V framework | Architectural DT proof | <1 s |

### 4.2 Run all of them

```bash
python3 numerical_experiments.py --all
```

Output lands in `dt/results/run_YYYYMMDD_HHMMSS/`:

```
acceptance_test.json     -- bias, repeatability vs water (1.3330) and oil (1.4700)
latency_benchmark.json   -- mean, p50, p95, p99, loss
auto_vanish_trials.json  -- success rate, mean iterations, parking depth scatter
brightness_ablation.json -- max bias across brightness levels
nis_consistency.json     -- sigma_n_m time series + drift
dt_comp_score.json       -- C, O, R, S, B, V -> DT_comp in [0,1]
traceability_matrix.csv  -- 21-row component matrix populated from live readings
summary_report.md        -- one-page paper-ready table
audit_aggregate.json     -- everything in one file
```

### 4.3 Run a subset

```bash
python3 numerical_experiments.py --acceptance --latency
python3 numerical_experiments.py --auto-vanish
python3 numerical_experiments.py --list
```

### 4.4 Pass criteria (from the paper)

| Check | Spec |
|---|---|
| Acceptance: |bias| ≤ 0.005 RIU AND σ ≤ 0.002 RIU |
| Latency: p95 ≤ 500 ms (full /estimate including frame + EKF) |
| Auto-vanish: success rate ≥ 0.80 |
| σ_n_m plateau < 0.005 AND drift < 0.003 RIU |
| Brightness: max abs bias ≤ 0.010 RIU across 4 levels |
| DT_comp ≥ 0.60 → "Validated DT" tier |

---

## 5. DT quantification audit (the existing 10-criterion rubric)

In addition to the C/O/R/S/B/V framework, the project ships a mature rubric
(in `dt_extension/dt_quantification.py`) that scores 10 capabilities × 5 pts.

```bash
# while the server is running:
curl http://localhost:5000/dt/audit | python3 -m json.tool
```

Both rubrics give the same conclusion via different decompositions; reporting
both makes the paper bulletproof against either school of thought.

---

## 6. Multilingual UI + voice

8 languages supported throughout the UI (`dt_extension/i18n.py`):
**English, Telugu, Hindi, Gujarati, Tamil, Bengali, Marathi, Kashmiri**.

Voice command is wired for **Telugu and Hindi only** (per the proposal),
using Vosk offline ASR (`dt_extension/voice_commands.py`).

Switch language at runtime:
```bash
curl -X POST http://localhost:5000/i18n/language \
     -H 'Content-Type: application/json' \
     -d '{"lang": "te"}'
```

Native-speaker translations should replace the stub strings in
`dt_extension/i18n.py` (search for `TODO(lang):`) before deployment.

---

## 7. Calibration coefficients

`dt/models/coeffs_v2.yaml` is the authoritative versioned calibration. It
includes:

- β / γ coefficients for the photometric measurement model
- Full **Cov(β, γ) 8×8 covariance block** (closes the audit gap)
- Reference-device provenance
- Calibration history (chain back to v1)

The NDJSON run logger stamps the coeff version into every line, so any past
run can be re-analysed with the exact coefficients used at the time.

---

## 8. File layout

```
vanishing_rod_dt/
├── app.py                       # Flask app + 4-mode router
├── main_control.py              # 28BYJ-48 + ULN2003 driver classes
├── pinmap.py                    # SINGLE source of truth for new-PCB pins
├── numerical_experiments.py     # auto-runner for all paper deliverables
├── start.sh                     # convenience launcher
├── requirements.txt
├── coeffs_identified.yaml       # legacy v1 (kept for reproducibility)
├── config.yaml                  # app config
├── roi_config.yaml              # camera ROI for both beakers
├── README.md                    # this file
│
├── dt/                          # DT scratch space
│   ├── models/
│   │   └── coeffs_v2.yaml       # versioned calibration with Cov block
│   ├── logs/                    # NDJSON run logs (auto-created)
│   └── results/                 # numerical_experiments.py outputs
│
├── dt_extension/                # backend extensions (no separate dashboard page)
│   ├── app_extensions.py        # registers /brightness, /dt/state,
│   │                            #   /dt/audit, /calibrate, /mode, /i18n/*
│   ├── brightness_sensor.py     # BH1750 driver + sampler thread
│   ├── ekf_brightness.py        # 6-state brightness-aware EKF
│   ├── physics_brightness.py    # Fresnel + photometric model + coeffs loader
│   ├── dt_quantification.py     # 10-criterion DT audit
│   ├── i18n.py                  # 8-language catalogue
│   └── voice_commands.py        # Vosk offline ASR (Telugu, Hindi)
│
├── static/
│   └── lab.css
│
└── templates/
    ├── base.html
    ├── login.html
    ├── dashboard.html           # 2-card layout (Vanishing Rod + Focal Length)
    ├── modes.html
    ├── mode_physical_manual.html
    ├── mode_hybrid_twin.html
    ├── mode_autocontrol.html
    ├── mode_simulation.html
    └── focal_length.html
```

The previous "DT Extension Hub" page has been removed. **All backend
endpoints from `dt_extension/` remain and are still registered** (BH1750
sampler, EKF-B, audit, calibration, voice, i18n) — only the separate
dashboard page is gone, per the project decision.

---

## 9. Endpoints reference (cheat sheet)

| Route | Method | Used by |
|---|---|---|
| `/login`, `/logout`, `/dashboard` | UI | session auth |
| `/vanishing/ab`, `/vanishing/Ab`, `/vanishing/aB`, `/vanishing/ab_sim` | UI | the four modes |
| `/video_feed` | GET | live MJPEG stream |
| `/telemetry` | GET | server-sent events for live state |
| `/sensors` | GET | live T (DS18B20) + B (BH1750) |
| `/motor/status`, `/motor/test`, `/motor/home`, `/motor/down`, `/motor/up`, `/motor/steps` | GET/POST | motor control |
| `/estimate` | POST | capture frame, run EKF, return n_m |
| `/auto_vanish`, `/sweep_status` | POST/GET | mode aB control loop |
| `/sim/set_medium`, `/sim/state` | POST/GET | mode ab simulation |
| `/status`, `/healthz`, `/dt/ekf_info` | GET | system status |
| `/brightness`, `/dt/state`, `/dt/audit`, `/calibrate` | GET/POST | DT extension backend |
| `/mode`, `/mode/release` | POST | authority arbiter |
| `/i18n/language`, `/i18n/catalogue/<lang>` | POST/GET | language switch |
| `/voice/text` | POST | text intent (replace ASR for testing) |

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `pinmap.py` says `SPI disabled FAIL` | SPI is on | `sudo raspi-config` → SPI → Disable → reboot |
| `pinmap.py` says `w1-gpio overlay FAIL` | overlay missing | add `dtoverlay=w1-gpio,gpiopin=5` to `/boot/firmware/config.txt`, reboot |
| `pinmap.py` says `I2C enabled FAIL` | I2C is off | `sudo raspi-config` → I2C → Enable → reboot |
| BH1750 reads 0 lux | wrong I2C bus or address | `sudo i2cdetect -y 1` should show `23` |
| DS18B20 reads 85.0 °C | sensor disconnected | check 4.7 kΩ pull-up between data and 3V3 |
| Acceptance bias > 0.01 RIU | ROI misaligned | edit `roi_config.yaml`, re-run |
| Auto-vanish always fails | motor stalls | check 5 V supply current ≥ 1.5 A |
| Latency spikes | thermal throttle | `vcgencmd measure_temp` |
| `cannot log in` from `numerical_experiments.py` | wrong creds | pass `--user` and `--password` matching `USERS` in `app.py` |

---

## 11. Reproducibility checklist (for supplementary materials)

- [ ] `pinmap.py` ran without failures
- [ ] `coeffs_v2.yaml` archived with the run
- [ ] `acceptance_test.json` shows PASS verdict
- [ ] `latency_benchmark.json` p95 within spec
- [ ] `auto_vanish_trials.json` success rate ≥ 0.80
- [ ] `brightness_ablation.json` max bias within spec
- [ ] `nis_consistency.json` σ plateau OK
- [ ] `dt_comp_score.json` DT_comp ≥ 0.60
- [ ] `traceability_matrix.csv` populated from live readings
- [ ] 10-criterion `/dt/audit` report retained alongside

If all 10 boxes are ticked, no reviewer can credibly call the work a digital
shadow.

---

## 12. Default credentials

`app.py` ships with three demo users (in `USERS` dict at top of file):

| Username | Password | Role |
|---|---|---|
| `student` | `lab2024` | basic |
| `teacher` | `teach2024` | basic |
| `admin` | `admin123` | basic (used by `numerical_experiments.py`) |

Change these for production.
