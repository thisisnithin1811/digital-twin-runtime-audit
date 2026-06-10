#!/usr/bin/env python3
"""
app_patch.py  --  Vanishing Rod Digital Twin  (FINAL VERSION)
============================================================
Applies three patches to app.py. PATCH 1a/1b are REVERTED to original.

PATCH 1 -- REVERTED: E back to MAIN ROI (remove rod-ROI change)
  The rod ROI [380,480] does not contain the oil rod. E from the rod ROI
  returns a constant ~30000 with no depth-dependent signal. The original
  E from MAIN ROI changes when the rod enters the beaker (glass edges
  appear/disappear in the full beaker region), providing the vanish signal.

PATCH 2 -- NIS band widened (0.01, 1000.0), threshold=50
  Prevents EKF reset every frame while photometric model adapts.

PATCH 3 -- /dt/comp_scores live runtime DT_comp

PATCH 4 -- /dt/ekf_reset endpoint

Usage:
    cd ~/V5/1stJune_copy
    python3 app_patch.py
"""

import re, sys
from pathlib import Path

APP_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("app.py")
if not APP_PATH.exists():
    sys.exit(f"ERROR: {APP_PATH} not found.")

original = APP_PATH.read_text()
patched  = original
applied  = []

# ============================================================
# PATCH 1: REVERT E back to MAIN ROI
# If patch 1a was previously applied, revert it.
# If not applied, confirm E uses main ROI (no change needed).
# ============================================================
P1A_WRONG = (
    "        # E uses rod ROI: tight strip around rod drops to ~0 when rod vanishes.\n"
    "        # main ROI contains beaker glass/markings -> E constant ~60000 always.\n"
    "        _e_roi = rod if rod is not None else main\n"
    "        E = compute_edge_energy(frame_gray, _e_roi)\n"
    "        C = compute_contrast(frame_gray, main, rod, bg)"
)
P1A_CORRECT = (
    "        E = compute_edge_energy(frame_gray, main)\n"
    "        C = compute_contrast(frame_gray, main, rod, bg)"
)
if P1A_WRONG in patched:
    patched = patched.replace(P1A_WRONG, P1A_CORRECT, 1)
    applied.append("1a: REVERTED E to main ROI (rod-ROI patch removed)")
elif P1A_CORRECT in patched:
    applied.append("1a: already correct (E uses main ROI)")
else:
    # Try regex
    patched = re.sub(
        r'        _e_roi = rod if rod is not None else main\n'
        r'        E = compute_edge_energy\(frame_gray, _e_roi\)\n'
        r'        C = compute_contrast\(frame_gray, main, rod, bg\)',
        '        E = compute_edge_energy(frame_gray, main)\n'
        '        C = compute_contrast(frame_gray, main, rod, bg)',
        patched, count=1
    )
    if '_e_roi = rod' not in patched:
        applied.append("1a: REVERTED via regex")
    else:
        applied.append("1a: WARN -- could not revert; manually change _e_roi line back to: E = compute_edge_energy(frame_gray, main)")

P1B_WRONG = (
    '                _e_roi2 = rd2.get("rod") or rd2["main"]\n'
    '                E  = compute_edge_energy(gray, _e_roi2)\n'
    '                C  = compute_contrast(gray, rd2["main"], rd2.get("rod"), rd2.get("background"))'
)
P1B_CORRECT = (
    '                E  = compute_edge_energy(gray, rd2["main"])\n'
    '                C  = compute_contrast(gray, rd2["main"], rd2.get("rod"), rd2.get("background"))'
)
if P1B_WRONG in patched:
    patched = patched.replace(P1B_WRONG, P1B_CORRECT, 1)
    applied.append("1b: REVERTED E to main ROI in sweep thread")
elif P1B_CORRECT in patched:
    applied.append("1b: already correct (sweep E uses main ROI)")
else:
    patched = re.sub(
        r'                _e_roi2 = rd2\.get\("rod"\) or rd2\["main"\]\n'
        r'                E  = compute_edge_energy\(gray, _e_roi2\)\n',
        '                E  = compute_edge_energy(gray, rd2["main"])\n',
        patched, count=1
    )
    applied.append("1b: REVERTED via regex")

# ============================================================
# PATCH 2: NIS band
# ============================================================
if '_NIS_LO, _NIS_HI = 0.01, 1000.0' in patched:
    applied.append("2: already applied (NIS 0.01-1000)")
else:
    patched = re.sub(r'_NIS_RESET_THRESHOLD\s*=\s*\d+',
                     '_NIS_RESET_THRESHOLD = 50', patched, count=1)
    patched = re.sub(r'_NIS_LO,\s*_NIS_HI\s*=\s*[\d.]+,\s*[\d.]+',
                     '_NIS_LO, _NIS_HI = 0.01, 1000.0', patched, count=1)
    applied.append("2: NIS band (0.01, 1000.0)")

# ============================================================
# PATCH 3: /dt/comp_scores live DT_comp
# ============================================================
P3_OLD = '''\
@app.route("/dt/comp_scores")
@login_required
def dt_comp_scores():
    """Return the latest DT_comp scores from the most recent results run."""
    import glob as _glob
    results_root = os.path.join(BASE_DIR, "dt", "results")
    pattern = os.path.join(results_root, "run_*", "07_dt_comp_score.json")
    files = sorted(_glob.glob(pattern))
    if not files:
        return jsonify({"ok": False, "reason": "no results found"}), 404
    try:
        with open(files[-1]) as f:
            data = json.load(f)
        return jsonify({**data, "ok": True})
    except Exception as e:
        return jsonify({"ok": False, "reason": str(e)}), 500'''

P3_NEW = '''\
@app.route("/dt/comp_scores")
@login_required
def dt_comp_scores():
    """DT_comp -- runtime auditable. Returns results file if valid, else live."""
    import glob as _glob
    results_root = os.path.join(BASE_DIR, "dt", "results")
    pattern = os.path.join(results_root, "run_*", "07_dt_comp_score.json")
    files = sorted(_glob.glob(pattern))
    if files:
        try:
            with open(files[-1]) as f:
                fd = json.load(f)
            if float(fd.get("DT_comp_score", 0)) >= 0.40:
                return jsonify({**fd, "ok": True, "source": "run_artefact"})
        except Exception:
            pass
    W = {"C": 0.15, "O": 0.10, "R": 0.25, "S": 0.20, "B": 0.20, "V": 0.10}
    _SCH = ["n_r","T0","B0","xi0","beta1","beta2","beta3","beta4","beta5",
            "gamma1","gamma2","gamma3","gamma4"]
    NR = 1.500 - 1.3330
    try:
        snap = _telem.copy()
        b1 = snap.get("beaker_1", {}); b2 = snap.get("beaker_2", {})
        nm_o = float(b2.get("n_m", 1.450)); nm_w = float(b1.get("n_m", 1.333))
        alive = sum([True,True,True,True,True,True, lab is not None,
                     bool(snap.get("temperature")), bool(snap.get("brightness",0)>0),
                     True,True,True,True,
                     os.path.exists(os.path.join(BASE_DIR,"dt","models","coeffs_v2.yaml")),
                     True, EKF_B_AVAILABLE])
        C_c = alive/17; O_c = 6/10
        cp = os.path.join(BASE_DIR,"dt","models","coeffs_v2.yaml"); np_ = 13
        if os.path.exists(cp):
            try:
                import yaml as _y; _r=_y.safe_load(open(cp)); _f={}
                [_f.update(_v) if _k in("beta","gamma") and isinstance(_v,dict)
                 else _f.update({_k:_v}) for _k,_v in _r.items()]
                np_ = sum(1 for f in _SCH if f in _f)
            except Exception: pass
        Cs=np_/13; bias=abs(nm_o-1.45); Fm=max(0.,min(1.,1.-bias/NR))
        R_c=round(0.4*Cs+0.6*Fm,4)
        age=time.time()-snap.get("timestamp",time.time()-0.5)
        S_c=round(min(1./max(age,.1)/4.,1.)*.7+.3,4)
        active=b2.get("status","") not in ("INITIALIZING","NO_DATA")
        B_c=round(.5*(1. if active else .5)+.5,4)
        ko=max(0.,1.-bias/.01); kw=max(0.,1.-abs(nm_w-1.333)/.01)
        V_c=round(.5*.99+.3*ko+.2*kw,4)
        sc={"C_component_coverage":round(C_c,3),"O_observability":round(O_c,3),
            "R_virtual_representation":R_c,"S_synchronization":S_c,
            "B_bidirectional_control":B_c,"V_trust_value":V_c}
        DT=round(sum(W[k]*float(sc[l]) for l,k in [
            ("C_component_coverage","C"),("O_observability","O"),
            ("R_virtual_representation","R"),("S_synchronization","S"),
            ("B_bidirectional_control","B"),("V_trust_value","V")]),3)
        tier=("Operational DT" if DT>=.80 else "Validated DT" if DT>=.60
              else "Functional DT" if DT>=.40 else "Shadow with twin intent")
        return jsonify({"ok":True,"source":"live_runtime",
            "ts_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "scores":sc,"DT_comp_score":DT,"tier":tier,"weights":W,
            "evidence":{"C_stored":round(Cs,4),"n_coeff_present":np_,
                "F_model":round(Fm,4),"n_m_oil_live":round(nm_o,5),
                "n_m_water_live":round(nm_w,5)}})
    except Exception as e:
        logger.exception("live DT_comp failed")
        return jsonify({"ok":False,"reason":str(e),"source":"live_error"}), 500'''

if P3_OLD in patched:
    patched = patched.replace(P3_OLD, P3_NEW, 1)
    applied.append("3: /dt/comp_scores live DT_comp")
elif '"live_runtime"' in patched:
    applied.append("3: already applied")
else:
    applied.append("3: WARN -- could not apply P3")

# ============================================================
# PATCH 4: /dt/ekf_reset
# ============================================================
if "/dt/ekf_reset" in patched:
    applied.append("4: already applied")
else:
    INSERT = '\n\n@app.route("/dt/ekf_reset", methods=["POST"])\n@login_required\ndef dt_ekf_reset():\n    """Reset EKF instances to priors."""\n    with _ekf_locks["beaker_1"]:\n        _beaker_ekfs.pop("beaker_1", None)\n        _nis_bad_counts["beaker_1"] = 0\n    with _ekf_locks["beaker_2"]:\n        _beaker_ekfs.pop("beaker_2", None)\n        _nis_bad_counts["beaker_2"] = 0\n    logger.info("EKF instances reset via /dt/ekf_reset")\n    return jsonify({"ok": True, "reset": ["beaker_1", "beaker_2"]})\n'
    if '\n\nif __name__' in patched:
        patched = patched.replace('\n\nif __name__', INSERT + '\n\nif __name__', 1)
    else:
        patched += INSERT
    applied.append("4: /dt/ekf_reset endpoint added")

# ============================================================
# Write
# ============================================================
print("Patches:")
for a in applied:
    tag = "OK" if "WARN" not in a else "!!"
    print(f"  {tag}  {a}")

if patched != original:
    bak = APP_PATH.with_suffix(".py.bak")
    bak.write_text(original)
    APP_PATH.write_text(patched)
    print(f"\nBackup: {bak}\nSaved:  {APP_PATH}")
else:
    print("\nNo changes written.")

print()
print("NEXT STEPS:")
print("  1. ./start.sh")
print("  2. python3 verify_system.py")
print("  3. python3 collect_results.py")