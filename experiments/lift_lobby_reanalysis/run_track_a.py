"""Track A: event-level, CAL-budget re-analysis of R6 and the 6-clip lift-door benchmark (DESIGN.md).

Runs on EC2 CPU. Reads existing maps / cached tokens only; frozen code imported read-only (run with python -B).
Layout (--root):
  in/r6/primary/{cycles.json,thresholds.json,*.npz}
  in/code/scripts/{r6/motion.py,sync_screen/motion.py}
  in/suite/{lift_vjepa_suite.py,lift_vjepa_suite_v2.py,results_v2/summary.json,results_v1/token_cache/*.npy}
  in/lift_dataset/<cat>/<clip>/{vjepa_frames/*.png,ground_truth.json}
  in/finding1/{run_finding1.py,results.json}
  a/{DESIGN.md,DESIGN.sha256}
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np

B_BOOT, SEED, BUDGET_PER_H, MARGIN = 2000, 20261109, 1.0, -0.05


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Log:
    def __init__(self, path):
        self.fh = open(path, "a")

    def __call__(self, msg):
        line = f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ protocol primitives
def persistent(s, k):
    """q(t) = max over length-k windows containing t of the window minimum; -inf if len(s) < k."""
    s = np.asarray(s, float)
    n = len(s)
    q = np.full(n, -np.inf)
    if k <= 1:
        return s.copy()
    for j in range(0, n - k + 1):
        w = s[j:j + k].min()
        q[j:j + k] = np.maximum(q[j:j + k], w)
    return q


def runs(flags):
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def n_episodes(qs, lam):
    return sum(len(runs(q > lam)) for q in qs)


def budget_lambda(qs, hours, budget=BUDGET_PER_H):
    allowed = math.floor(budget * hours)
    vals = np.unique(np.concatenate([np.asarray(q, float) for q in qs]))
    vals = vals[np.isfinite(vals)]
    for lam in np.r_[-np.inf, vals]:
        if n_episodes(qs, lam) <= allowed:
            return float(lam), allowed
    raise RuntimeError("no lambda")


def midrank(x):
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    return ((cum - cnt + 1 + cum) / 2.0)[inv]


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    # -inf (no persistent window) ranks lowest; replace by a finite floor for ranking
    allv = np.concatenate([pos, neg])
    fl = np.nanmin(allv[np.isfinite(allv)]) - 1 if np.isfinite(allv).any() else 0.0
    allv = np.where(np.isfinite(allv), allv, fl)
    r = midrank(allv)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def avg_precision(pos, neg):
    if len(pos) == 0:
        return None
    s = np.concatenate([pos, neg]).astype(float)
    fl = np.nanmin(s[np.isfinite(s)]) - 1 if np.isfinite(s).any() else 0.0
    s = np.where(np.isfinite(s), s, fl)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(-s, kind="mergesort")
    s, y = s[o], y[o]
    thr = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tps = np.cumsum(y)[thr]
    fps = 1 + thr - tps
    prec = tps / (tps + fps)
    rec = tps / tps[-1]
    return float(np.sum((rec - np.r_[0.0, rec[:-1]]) * prec))


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def cp(k, n):
    if n == 0:
        return [None, None]

    def solve(f, lo=0.0, hi=1.0):
        for _ in range(80):
            mid = (lo + hi) / 2
            if f(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) < 0.025)
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) > 0.025)
    return [lower, upper]


def sign_p(d):
    a = np.asarray([x for x in d if x is not None and np.isfinite(x)])
    n = len(a)
    if n == 0:
        return None
    return float(min(1.0, 2 * min(((a <= 0).sum() + 1) / (n + 1), ((a >= 0).sum() + 1) / (n + 1))))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, prev = [0.0] * len(ps), 0.0
    for j, i in enumerate(order):
        prev = max(prev, min(1.0, ps[i] * (len(ps) - j)))
        out[i] = prev
    return out


def ci(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [None, None]


def ni_readout(lo, hi):
    if lo is None:
        return "undefined"
    if lo > 0:
        return "pixel superior (lower bound > 0) and non-inferior at margin 0.05"
    if lo > MARGIN:
        return "pixel non-inferior at margin 0.05"
    if hi < MARGIN:
        return "pixel inferior (upper bound < -0.05)"
    return "inconclusive (interval contains -0.05)"


# ------------------------------------------------------------------ R6
R6_MAP_ARMS = ["D", "pixel", "illum_pixel", "flow_residual"]
R6_ARMS = R6_MAP_ARMS + ["G"]
STEP_R6 = 0.25


def r6_analysis(root, log, res):
    base = root / "in/r6/primary"
    code = root / "in/code/scripts"
    for name, p in [("scripts", code), ("scripts.sync_screen", code / "sync_screen"), ("scripts.r6", code / "r6")]:
        m = types.ModuleType(name)
        m.__path__ = [str(p)]
        sys.modules[name] = m
    stub = types.ModuleType("scripts.sync_screen.common")
    stub.CAMERAS = {}
    sys.modules["scripts.sync_screen.common"] = stub
    load_module("scripts.sync_screen.motion", code / "sync_screen/motion.py")
    motion = load_module("scripts.r6.motion", code / "r6/motion.py")
    inputs = {"cycles.json": sha(base / "cycles.json"), "thresholds.json": sha(base / "thresholds.json"),
              "r6/motion.py": sha(code / "r6/motion.py"), "sync_screen/motion.py": sha(code / "sync_screen/motion.py")}
    npz_sha = [sha(p) for p in sorted(base.glob("*.npz"))]
    inputs["npz_count"] = len(npz_sha)
    inputs["npz_sha_of_sorted_shas"] = hashlib.sha256(("\n".join(
        f"{h}  {p.name}" for h, p in zip(npz_sha, sorted(base.glob("*.npz")))) + "\n").encode()).hexdigest()
    d = json.loads((base / "cycles.json").read_text())
    rows, taus = d["rows"], d["thresholds"]
    log(f"R6 rows {len(rows)}; npz {len(npz_sha)}")
    maps = {}
    for r in rows:
        for a in R6_MAP_ARMS:
            maps[(r["id"], a)] = np.load(base / f"{r['id']}_{a}.npz")["blocks"].astype(np.float64)

    # ---- gate R0
    bad = []
    for r in rows:
        for a in R6_MAP_ARMS:
            m = maps[(r["id"], a)]
            st = r["results"][a]
            if abs(m.max() - st["score"]) > 1e-6 * max(1.0, abs(st["score"])) or bool(m.max() > taus[r["mode"]][a]) != st["alert"]:
                bad.append((r["id"], a, float(m.max()), st["score"], st["alert"]))
        g = r["results"]["G"]
        if bool(g["score"] > taus[r["mode"]]["G"]) != g["alert"]:
            bad.append((r["id"], "G", g["score"], g["alert"]))
    log(f"GATE R0 {'PASS' if not bad else 'FAIL'} mismatches {len(bad)}")
    # ---- intervals + gate R1
    iv, bad1, n1 = {}, [], 0
    for r in rows:
        if r["level"]:
            g = motion.geometry(r, 0, r["faulty_side"] or "left")["intervals"]
            iv[r["id"]] = g
            dt = r["results"]["D"]["dwell_timing"]
            if dt.get("onset_error_s") is not None:
                n1 += 1
                first = r["results"]["D"]["time_support"].index(True)
                if abs(first * 2 / 8 - g["dwell"][0] - dt["onset_error_s"]) > 1e-6:
                    bad1.append(r["id"])
    log(f"GATE R1 {'PASS' if not bad1 else 'FAIL'} checked {n1} mismatches {len(bad1)}")
    res["r6_gates"] = {"R0": {"pass": not bad, "mismatches": bad[:20], "n_mismatch": len(bad)},
                       "R1": {"pass": not bad1, "checked": n1, "mismatches": bad1}}
    res["r6_inputs"] = inputs
    if bad or bad1:
        log("STOP R6: gate failure")
        return

    cal = [r for r in rows if r["split"] == "CAL"]
    test = [r for r in rows if r["split"] == "TEST"]
    modes = sorted({r["mode"] for r in rows})

    def envelopes():
        env = {}
        for mo in modes:
            for a in R6_MAP_ARMS:
                env[(mo, a)] = np.max([maps[(r["id"], a)] for r in cal if r["mode"] == mo], 0) + 1e-4
        return env
    ENV = envelopes()

    def series(r, a, score):
        m = maps[(r["id"], a)]
        if score == "global":
            return m.reshape(16, -1).max(1)
        return (m / ENV[(r["mode"], a)]).reshape(16, -1).max(1)

    def event_window_steps(r):
        a0, b0 = iv[r["id"]]["defect_active"]
        lo, hi = a0 - 0.5, b0 + 0.5
        i = np.arange(16)
        return (STEP_R6 * i < hi) & (STEP_R6 * (i + 1) > lo)

    def run_protocol(score, k):
        out = {"lambda": {}, "rows": {}}
        for a in R6_ARMS:
            for mo in modes:
                crow = [r for r in cal if r["mode"] == mo]
                hours = len(crow) * 4 / 3600
                if a == "G":
                    qs = [np.array([r["results"]["G"]["score"]]) for r in crow]
                else:
                    qs = [persistent(series(r, a, score), k) for r in crow]
                lam, allowed = budget_lambda(qs, hours)
                out["lambda"][f"{a}|{mo}"] = {"lambda": lam, "allowed_episodes": allowed, "cal_clips": len(crow),
                                              "cal_hours": hours, "cal_episodes_at_lambda": n_episodes(qs, lam),
                                              "r6_frozen_tau": taus[mo][a]}
        for r in test:
            per = {}
            for a in R6_ARMS:
                lam = out["lambda"][f"{a}|{r['mode']}"]["lambda"]
                if a == "G":
                    sc = r["results"]["G"]["score"]
                    alert = sc > lam
                    e = {"q": [sc], "alert_steps": [bool(alert)], "episodes": int(alert), "clip_level": True}
                    if r["level"]:
                        e.update({"detected": bool(alert), "peak": sc, "onset": None})
                    e["false_episodes"] = 0 if r["level"] else int(alert)
                    per[a] = e
                    continue
                q = persistent(series(r, a, score), k)
                al = q > lam
                rr = runs(al)
                e = {"q": q.tolist(), "alert_steps": al.tolist(), "episodes": len(rr)}
                if r["level"]:
                    w = event_window_steps(r)
                    hit = [x for x in rr if w[x[0]:x[1] + 1].any()]
                    e["detected"] = bool(hit)
                    e["peak"] = float(q[w].max())
                    e["onset"] = ((hit[0][0] + k) * STEP_R6 - iv[r["id"]]["defect_active"][0]) if hit else None
                    e["false_episodes"] = len(rr) - len(hit)
                else:
                    e["false_episodes"] = len(rr)
                per[a] = e
            out["rows"][r["id"]] = per
        return out

    families = {}
    for i, r in enumerate(test):
        families.setdefault(r["source_family"], []).append(i)
    strata = {}
    for fam, idx in families.items():
        r = test[idx[0]]
        key = f"defect{r['level']}" if r["level"] else ("reopen" if r["mode"] == "reopen" else "healthy")
        strata.setdefault(key, []).append(fam)

    def draws():
        rng = np.random.default_rng(SEED)
        for _ in range(B_BOOT):
            chosen = []
            for key in sorted(strata):
                vals = sorted(strata[key])
                for f in rng.choice(vals, len(vals), replace=True):
                    chosen.extend(families[f])
            yield chosen

    def summarise(P, idx):
        rs = [test[i] for i in idx]
        clean = [r for r in rs if r["cell"] == "clean_defect"]
        refl = [r for r in rs if r["cell"] == "defect_reflection"]
        healthy = [r for r in rs if r["cell"] == "healthy"]
        hard = [r for r in rs if r["cell"] in ("reflection_only", "exposure_only", "reopen")]
        o = {}
        for a in R6_ARMS:
            g = lambda r: P["rows"][r["id"]][a]
            pos = [g(r)["peak"] for r in clean]
            if a == "G":
                neg = [g(r)["q"][0] for r in healthy]
            else:
                neg = [x for r in healthy for x in g(r)["q"]]
            o[a] = {"recall_clean": float(np.mean([g(r)["detected"] for r in clean])) if clean else None,
                    "recall_reflection": float(np.mean([g(r)["detected"] for r in refl])) if refl else None,
                    "auroc": auroc(pos, neg), "ap": avg_precision(pos, neg),
                    "hard_fa_per_clip": float(np.mean([g(r)["false_episodes"] for r in hard])) if hard else None,
                    "onsets": {r["id"]: g(r)["onset"] for r in clean}}
        both = [r["id"] for r in clean if o["pixel"]["onsets"][r["id"]] is not None and o["D"]["onsets"][r["id"]] is not None]
        o["S3_median_onset_diff"] = float(np.median([o["pixel"]["onsets"][i] - o["D"]["onsets"][i] for i in both])) if both else None
        return o

    def full_report(P, label, bootstrap):
        allidx = list(range(len(test)))
        pt = summarise(P, allidx)
        rep = {"lambda": P["lambda"], "arms": {}}
        clean = [r for r in test if r["cell"] == "clean_defect"]
        for a in R6_ARMS:
            g = lambda r: P["rows"][r["id"]][a]
            arm = {}
            for cell in ["clean_defect", "defect_reflection"]:
                sub = [r for r in test if r["cell"] == cell]
                k_ = sum(g(r)["detected"] for r in sub)
                arm[f"{cell}_events_detected"] = [int(k_), len(sub)]
                arm[f"{cell}_recall_CP95"] = cp(int(k_), len(sub))
                arm[f"{cell}_per_level"] = {str(l): [int(sum(g(r)["detected"] for r in sub if r["level"] == l)),
                                                     sum(r["level"] == l for r in sub)] for l in [1, 2, 3, 4]
                                            if any(r["level"] == l for r in sub)}
                fe = sum(g(r)["false_episodes"] for r in sub)
                arm[f"{cell}_false_episodes"] = [int(fe), len(sub)]
            for cell in ["healthy", "reflection_only", "exposure_only", "reopen"]:
                sub = [r for r in test if r["cell"] == cell]
                ep = sum(g(r)["false_episodes"] for r in sub)
                flagged = sum(g(r)["false_episodes"] > 0 for r in sub)
                arm[cell] = {"clips": len(sub), "camera_s": len(sub) * 4, "false_episodes": int(ep),
                             "clips_flagged": int(flagged), "clips_flagged_CP95": cp(int(flagged), len(sub)),
                             "raw_episodes_per_camera_hour": ep * 3600 / (len(sub) * 4)}
            ons = [g(r)["onset"] for r in clean if g(r).get("onset") is not None]
            arm["onset_s_clean"] = {"n": len(ons), "median": float(np.median(ons)) if ons else None,
                                    "p10_p90": [float(np.percentile(ons, 10)), float(np.percentile(ons, 90))] if ons else None}
            arm["event_auroc_clean_vs_healthy_steps"] = pt[a]["auroc"]
            arm["event_ap_clean_vs_healthy_steps"] = pt[a]["ap"]
            arm["unit_note"] = "G: clip-level (positive = clip score, negative = healthy clip score)" if a == "G" else "token-step q"
            rep["arms"][a] = arm
        if not bootstrap:
            return rep
        B = {a: {"recall_clean": [], "recall_reflection": [], "auroc": [], "ap": [], "hard_fa_per_clip": []} for a in R6_ARMS}
        S3 = []
        for idx in draws():
            s = summarise(P, idx)
            for a in R6_ARMS:
                for key in B[a]:
                    B[a][key].append(s[a][key])
            S3.append(s["S3_median_onset_diff"])
        diff = lambda a, key: [None if (x is None or y is None) else x - y for x, y in zip(B[a][key], B["D"][key])]
        boot = {"B": B_BOOT, "seed": SEED, "strata": {k: len(v) for k, v in strata.items()},
                "families_test": len(families), "cal_fixed_not_resampled": True, "arm_ci95": {}, "diff_vs_D_ci95": {}}
        for a in R6_ARMS:
            boot["arm_ci95"][a] = {key: ci(v) for key, v in B[a].items()}
            if a != "D":
                boot["diff_vs_D_ci95"][a] = {key: {"point": (None if pt[a][key] is None or pt["D"][key] is None else pt[a][key] - pt["D"][key]),
                                                   "ci95": ci(diff(a, key))} for key in B[a]}
        prim = diff("pixel", "recall_clean")
        lo, hi = ci(prim)
        rep["PRIMARY"] = {"estimand": "R6 clean-defect event recall, pixel - D, CAL budget 1 episode/camera-hour",
                          "pixel": pt["pixel"]["recall_clean"], "D": pt["D"]["recall_clean"],
                          "delta": pt["pixel"]["recall_clean"] - pt["D"]["recall_clean"], "ci95": [lo, hi],
                          "margin": MARGIN, "readout": ni_readout(lo, hi)}
        sec = {"S1_event_AUROC": diff("pixel", "auroc"), "S2_event_AP": diff("pixel", "ap"),
               "S3_onset_median_diff_s": S3, "S4_hard_negative_FA_per_clip": diff("pixel", "hard_fa_per_clip"),
               "S5_defect_reflection_recall": diff("pixel", "recall_reflection")}
        pts = {"S1_event_AUROC": pt["pixel"]["auroc"] - pt["D"]["auroc"], "S2_event_AP": pt["pixel"]["ap"] - pt["D"]["ap"],
               "S3_onset_median_diff_s": pt["S3_median_onset_diff"],
               "S4_hard_negative_FA_per_clip": pt["pixel"]["hard_fa_per_clip"] - pt["D"]["hard_fa_per_clip"],
               "S5_defect_reflection_recall": pt["pixel"]["recall_reflection"] - pt["D"]["recall_reflection"]}
        raw = {k: sign_p(v) for k, v in sec.items()}
        keys = list(sec)
        adj = holm([raw[k] if raw[k] is not None else 1.0 for k in keys])
        rep["SECONDARY_pixel_minus_D"] = {k: {"point": pts[k], "ci95": ci(sec[k]), "p_raw": raw[k], "p_holm": adj[i],
                                              "n_valid_draws": int(sum(x is not None for x in sec[k]))} for i, k in enumerate(keys)}
        rep["bootstrap"] = boot
        return rep

    # R6 frozen reference (clip-level alert at R6 tau, no persistence)
    ref = {}
    for a in R6_ARMS:
        ref[a] = {cell: [int(sum(r["results"][a]["alert"] for r in test if r["cell"] == cell)),
                         sum(r["cell"] == cell for r in test)]
                  for cell in ["clean_defect", "defect_reflection", "healthy", "reflection_only", "exposure_only", "reopen"]}
    res["r6_frozen_reference_clip_alerts"] = ref
    log("R6 frozen reference clip alerts (flagged/n): " + json.dumps({a: ref[a]["clean_defect"] for a in R6_ARMS}))

    P = run_protocol("global", 2)
    res["r6_primary"] = full_report(P, "global_k2", True)
    log("R6 PRIMARY " + json.dumps(res["r6_primary"]["PRIMARY"]))
    for kk, v in res["r6_primary"]["SECONDARY_pixel_minus_D"].items():
        log(f"R6 SECONDARY {kk}: {json.dumps(v)}")
    res["r6_sensitivity_envelope_k2"] = full_report(run_protocol("envelope", 2), "envelope_k2", False)
    res["r6_sensitivity_global_k4"] = full_report(run_protocol("global", 4), "global_k4", False)
    res["r6_per_row_primary"] = {rid: {a: {k: v for k, v in e.items() if k in ("detected", "onset", "false_episodes", "episodes")}
                                       for a, e in per.items()} for rid, per in P["rows"].items()}
    log("R6 done")


# ------------------------------------------------------------------ lift-door
LD_DETS = ["D_patchKNN", "D_clipNN", "G_pooled", "pixel_patchKNN", "pixel_clipNN", "illum_pixel_patchKNN", "FD_median"]
FAM = {"D_patchKNN": "D", "D_clipNN": "D", "G_pooled": "D", "pixel_patchKNN": "pixel", "pixel_clipNN": "pixel",
       "illum_pixel_patchKNN": "illum", "FD_median": "grey"}
FAULT_TYPES = ("door_desync", "interior_light_flicker")
LD_CONTRASTS = [("pixel_patchKNN", "D_patchKNN"), ("pixel_clipNN", "D_clipNN"), ("illum_pixel_patchKNN", "D_patchKNN"),
                ("FD_median", "D_patchKNN"), ("G_pooled", "D_patchKNN")]


def liftdoor_analysis(root, log, res):
    suite = root / "in/suite"
    sys.path.insert(0, str(suite))
    import lift_vjepa_suite as v1
    import lift_vjepa_suite_v2 as v2
    f1 = load_module("run_finding1_ro", root / "in/finding1/run_finding1.py")
    ds = root / "in/lift_dataset"
    inputs = {p: sha(suite / p) for p in ["lift_vjepa_suite.py", "lift_vjepa_suite_v2.py", "results_v2/summary.json"]}
    inputs["run_finding1.py"] = sha(root / "in/finding1/run_finding1.py")
    inputs["finding1_results.json"] = sha(root / "in/finding1/results.json")
    cases = {p.parent.name: p.parent for p in sorted(ds.glob("*/*/ground_truth.json")) if p.parent.parent.name != "stress"}
    frames, gts, labs, tok, grey = {}, {}, {}, {}, {}
    for n, c in cases.items():
        frames[n] = v1.load_frames(c)
        gts[n] = json.loads((c / "ground_truth.json").read_text())
        labs[n] = v1.sample_labels(gts[n])
        inputs[f"{n}/ground_truth.json"] = sha(c / "ground_truth.json")
        h = hashlib.sha256(frames[n].tobytes()).hexdigest()[:16]
        cp_ = suite / "results_v1/token_cache" / f"{n}_{h}.npy"
        if not cp_.exists():
            log(f"BLOCKED lift-door: token cache miss {cp_.name}")
            res["liftdoor_blocked"] = cp_.name
            return
        inputs[f"token_cache/{cp_.name}"] = sha(cp_)
        tok[n] = v1.unit(np.load(cp_).astype(np.float32))
        grey[n] = frames[n].astype(np.float32).mean(-1) / 255.0
    res["liftdoor_inputs"] = inputs
    log(f"lift-door loaded {len(cases)} clips")
    feats = {"D": tok, "pixel": {n: v1.grey_patches(f, False) for n, f in frames.items()},
             "illum": {n: v1.grey_patches(f, True) for n, f in frames.items()}, "grey": grey}
    bank = list(v1.FROZEN_DESIGN["healthy_bank"])
    anomalies = [n for n in cases if gts[n]["video_health"] == "anomaly"]
    scored = anomalies + ["healthy_ball_reopen"]
    healthy = [n for n in cases if gts[n]["video_health"] == "healthy"]
    twin = {}
    for an in anomalies:
        c = [h for h in healthy if gts[h]["seed"] == gts[an]["seed"]
             and sorted(gts[h]["disturbance_types"]) == sorted(gts[an]["disturbance_types"])]
        assert len(c) == 1
        twin[an] = c[0]

    def mapfn(det, tc, bc, key):
        if det == "FD_median":
            return f1.fd_map(tc, bc, key)
        if det == "pixel_clipNN":
            cands = [v1.blocks(np.abs(tc - b)) for b in bc]
            return cands[int(np.argmin([c.max() for c in cands]))]
        return v1.map_for(det, tc, bc)

    halves = lambda x: [x[:len(x) // 2], x[len(x) // 2:]]
    cache = {}

    def bank_products(det, bk):
        key = (det, tuple(bk))
        if key not in cache:
            F = feats[FAM[det]]
            loo = []
            for b in bk:
                others = [halves(F[o])[k] for o in bk if o != b for k in range(2)]
                okey = tuple(o for o in bk if o != b)
                for c in range(2):
                    loo.append(mapfn(det, halves(F[b])[c], others, okey))
            tau_env = np.max(loo, 0) + 1e-4
            tau_g = max(float(m.max()) for m in loo) + 1e-4
            cache[key] = {"loo": loo, "tau_env": tau_env, "tau_g": tau_g}
        return cache[key]

    def test_series(det, n, bk):
        F = feats[FAM[det]]
        bp = bank_products(det, bk)
        allb = [halves(F[b])[k] for b in bk for k in range(2)]
        m = np.concatenate([mapfn(det, cy, allb, tuple(bk)) for cy in halves(F[n])], 0)
        tt = np.concatenate([bp["tau_env"], bp["tau_env"]], 0)
        return {"global": m.reshape(32, -1).max(1), "envelope": (m / tt).reshape(32, -1).max(1)}

    def cal_series(det, bk, score):
        bp = bank_products(det, bk)
        if score == "global":
            return [m.reshape(16, -1).max(1) for m in bp["loo"]]
        return [(m / bp["tau_env"]).reshape(16, -1).max(1) for m in bp["loo"]]

    conds = {"original": {n: bank for n in scored},
             "twin_out": {n: ([b for b in bank if b != twin[n]] if n in twin else bank) for n in scored}}
    S = {c: {d: {n: test_series(d, n, conds[c][n]) for n in scored} for d in LD_DETS} for c in conds}
    log("lift-door maps computed")

    # ---- gate L0'(b): numeric diagnostics vs Finding 1 (no bar; Amendment 1)
    fr = json.loads((root / "in/finding1/results.json").read_text())["per_sample_scores"]
    diag = []
    for c in conds:
        for d in ["D_patchKNN", "D_clipNN", "G_pooled", "pixel_patchKNN", "illum_pixel_patchKNN", "FD_median"]:
            for n in scored:
                tau_g = bank_products(d, conds[c][n])["tau_g"]
                for kind, mine, theirs in [("envelope", S[c][d][n]["envelope"], fr[c][d][n]["s"]),
                                           ("global", S[c][d][n]["global"] / tau_g, fr[c][d][n]["s_global"])]:
                    t = np.asarray(theirs)[::2]
                    err = np.abs(mine - t)
                    j = int(np.argmax(err))
                    diag.append({"bank": c, "det": d, "clip": n, "score": kind, "max_abs": float(err.max()),
                                 "value_at_max": float(t[j]), "max_rel": float((err / np.maximum(np.abs(t), 1e-12)).max())})
    # ---- gate L0'(a): decision-level reproduction of results_v2/summary.json (the bar)
    ref = json.loads((suite / "results_v2/summary.json").read_text())
    bad, nchk = [], 0
    for d in ["D_patchKNN", "D_clipNN", "G_pooled", "pixel_patchKNN", "illum_pixel_patchKNN"]:
        bp = bank_products(d, bank)
        F = feats[FAM[d]]
        allb = [halves(F[b])[k] for b in bank for k in range(2)]
        for n in scored:
            m = np.concatenate([mapfn(d, cy, allb, tuple(bank)) for cy in halves(F[n])], 0)
            for pol in ("envelope", "global"):
                tt = np.concatenate([bp["tau_env"], bp["tau_env"]], 0) if pol == "envelope" else bp["tau_g"]
                al = np.repeat(v2.persist((m > tt).reshape(32, -1).any(1)), 2)
                ev = v1.evaluate(al, labs[n])
                r = ref["detectors"][d][pol][n]
                nchk += 1
                same = ([e["alerted"] for e in ev["events"]] == [e["alerted"] for e in r["events"]]
                        and ev["false_alert_samples"] == r["false_alert_samples"] and ev["alert_times_s"] == r["alert_times_s"])
                if not same:
                    bad.append({"det": d, "pol": pol, "clip": n, "ours_FA": ev["false_alert_samples"], "ref_FA": r["false_alert_samples"]})
    worst = max(diag, key=lambda x: x["max_abs"])
    log(f"GATE L0'(a) {'PASS' if not bad else 'FAIL'} checked {nchk} mismatches {len(bad)}; "
        f"L0'(b) diagnostic worst abs diff {worst['max_abs']:.3g} ({worst['det']} {worst['bank']} {worst['clip']} {worst['score']}, value {worst['value_at_max']:.4g})")
    res["liftdoor_gates"] = {"L0a_decisions_vs_v2_summary": {"pass": not bad, "checked": nchk, "mismatches": bad},
                             "L0b_numeric_diagnostic_vs_finding1": diag}
    if bad:
        log("STOP lift-door: gate L0'(a) failure")
        return

    times = np.arange(64) / 4.0

    def near_mask(n, types_):
        m = np.zeros(64, bool)
        for e in labs[n]["events"]:
            if (e["type"] in FAULT_TYPES) == (types_ == "fault"):
                m |= (times >= e["start_sec"] - 0.5) & (times < e["end_sec_exclusive"] + 0.5)
        return m

    def protocol(score, k=2):
        out, l1_ok = {}, True
        for c in conds:
            out[c] = {"lambda": {}, "clips": {}}
            for d in LD_DETS:
                lam_by_bank = {}
                for n in scored:
                    bk = tuple(conds[c][n])
                    if bk not in lam_by_bank:
                        qs = [persistent(s, k) for s in cal_series(d, list(bk), score)]
                        lam, allowed = budget_lambda(qs, len(qs) * 8 / 3600)
                        lam_by_bank[bk] = (lam, allowed, len(qs), n_episodes(qs, lam))
                    lam = lam_by_bank[bk][0]
                    s = S[c][d][n][score]
                    q = persistent(s, k)
                    al = q > lam
                    if k == 2 and not np.array_equal(al, v2.persist(s > lam)):
                        l1_ok = False
                    samples = np.repeat(al, 2)
                    ev = v1.evaluate(samples, labs[n])
                    fault = near_mask(n, "fault")
                    dist = near_mask(n, "dist")
                    obs = labs[n]["observable"]
                    rr = runs(al)
                    events, fe, fe_dist = [], 0, 0
                    for e in labs[n]["events"]:
                        if e["type"] not in FAULT_TYPES:
                            continue
                        w = (times >= e["start_sec"] - 0.5) & (times < e["end_sec_exclusive"] + 0.5)
                        wtok = w.reshape(32, 2).any(1)
                        hit = [x for x in rr if wtok[x[0]:x[1] + 1].any()]
                        events.append({"type": e["type"], "start": e["start_sec"], "detected": bool(samples[w].any()),
                                       "peak": float(q[wtok].max()),
                                       "onset": ((hit[0][0] + k) * 0.5 - e["start_sec"]) if hit else None})
                    for x in rr:
                        sm = np.zeros(64, bool)
                        sm[2 * x[0]:2 * x[1] + 2] = True
                        if not (sm & fault).any() and (sm & obs).any():
                            fe += 1
                            fe_dist += int((sm & dist).any())
                    ftok = fault.reshape(32, 2).any(1)
                    out[c]["clips"].setdefault(n, {})[d] = {
                        "events": events, "false_episodes": fe, "false_episodes_overlapping_disturbance": fe_dist,
                        "false_alert_samples_v1": ev["false_alert_samples"], "neg_q": q[~ftok].tolist(),
                        "check_v1_hits": [h["alerted"] for h in ev["events"] if h["type"] in FAULT_TYPES]}
                out[c]["lambda"][d] = {"|".join(bk): {"lambda": v[0], "allowed": v[1], "cal_cycles": v[2], "cal_episodes": v[3]}
                                       for bk, v in lam_by_bank.items()}
        return out, l1_ok

    def summarise(P, c, clips):
        o = {}
        for d in LD_DETS:
            evs = [e for n in clips for e in P[c]["clips"][n][d]["events"]]
            neg = [x for n in clips for x in P[c]["clips"][n][d]["neg_q"]]
            ons = [e["onset"] for e in evs if e["onset"] is not None]
            o[d] = {"events_detected": int(sum(e["detected"] for e in evs)), "events": len(evs),
                    "recall": float(np.mean([e["detected"] for e in evs])) if evs else None,
                    "event_auroc": auroc([e["peak"] for e in evs], neg), "event_ap": avg_precision([e["peak"] for e in evs], neg),
                    "false_episodes": int(sum(P[c]["clips"][n][d]["false_episodes"] for n in clips)),
                    "false_episodes_overlapping_disturbance": int(sum(P[c]["clips"][n][d]["false_episodes_overlapping_disturbance"] for n in clips)),
                    "false_alert_samples_v1": int(sum(P[c]["clips"][n][d]["false_alert_samples_v1"] for n in clips)),
                    "onset_median_s": float(np.median(ons)) if ons else None}
        return o

    out = {}
    for score in ["global", "envelope"]:
        P, l1 = protocol(score)
        if score == "global":
            res["liftdoor_gates"]["L1"] = {"pass": bool(l1)}
            log(f"GATE L1 {'PASS' if l1 else 'FAIL'}")
            if not l1:
                log("STOP lift-door: gate L1 failure")
                return
        rep = {}
        for c in conds:
            pt = summarise(P, c, scored)
            per_clip = {n: {d: {"events": [(e["detected"], e["onset"]) for e in P[c]["clips"][n][d]["events"]],
                                "false_episodes": P[c]["clips"][n][d]["false_episodes"],
                                "false_dist": P[c]["clips"][n][d]["false_episodes_overlapping_disturbance"]}
                            for d in LD_DETS} for n in scored}
            r = {"lambda": P[c]["lambda"], "pooled6": pt, "per_clip": per_clip, "camera_s": len(scored) * 16}
            if score == "global":
                rng = np.random.default_rng(SEED)
                Bv = {f"{x}-{y}": {"recall": [], "event_auroc": [], "false_episodes": []} for x, y in LD_CONTRASTS}
                arm = {d: [] for d in LD_DETS}
                skipped = 0
                for _ in range(B_BOOT):
                    pick = [scored[i] for i in rng.integers(0, len(scored), len(scored))]
                    if not any(n in anomalies for n in pick):
                        skipped += 1
                        continue
                    s = summarise(P, c, pick)
                    for d in LD_DETS:
                        arm[d].append(s[d]["recall"])
                    for x, y in LD_CONTRASTS:
                        for key in Bv[f"{x}-{y}"]:
                            a_, b_ = s[x][key], s[y][key]
                            Bv[f"{x}-{y}"][key].append(None if a_ is None or b_ is None else a_ - b_)
                r["bootstrap_indicative"] = {
                    "B": B_BOOT, "seed": SEED, "skipped_no_event": skipped, "unit": "clip (6 scored clips)",
                    "recall_ci95": {d: ci(v) for d, v in arm.items()},
                    "contrasts": {k: {key: {"point": pt[k.split("-")[0]][key] - pt[k.split("-")[1]][key]
                                            if pt[k.split("-")[0]][key] is not None and pt[k.split("-")[1]][key] is not None else None,
                                            "ci95": ci(v)} for key, v in vv.items()} for k, vv in Bv.items()}}
                pr = r["bootstrap_indicative"]["contrasts"]["pixel_patchKNN-D_patchKNN"]["recall"]
                r["primary_estimand_indicative"] = {"delta": pr["point"], "ci95": pr["ci95"],
                                                    "readout_indicative_only": ni_readout(*pr["ci95"])}
            rep[c] = r
            log(f"LIFT-DOOR {score} {c}: " + json.dumps({d: [pt[d]["events_detected"], pt[d]["false_episodes"]] for d in LD_DETS}))
        out[score] = rep
    res["liftdoor"] = out
    log("lift-door done")


def tables(res):
    L = []
    if "r6_primary" in res:
        p = res["r6_primary"]
        L += ["## R6 (global score, k = 2, CAL budget 1 episode/camera-hour)", "",
              f"PRIMARY: pixel {p['PRIMARY']['pixel']:.3f} vs D {p['PRIMARY']['D']:.3f}; delta {p['PRIMARY']['delta']:+.3f} "
              f"95% CI [{p['PRIMARY']['ci95'][0]:+.3f}, {p['PRIMARY']['ci95'][1]:+.3f}] -> {p['PRIMARY']['readout']}", "",
              "| Arm | clean events | L1 | L2 | L3 | L4 | refl. events | healthy FA ep (40 clips) | refl.-only FA ep (40) | exposure FA ep (10) | reopen FA ep (10) | event AUROC | event AP | onset median s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in R6_ARMS:
            x = p["arms"][a]
            lv = x["clean_defect_per_level"]
            L.append(f"| {a} | {x['clean_defect_events_detected'][0]}/{x['clean_defect_events_detected'][1]} | "
                     + " | ".join(f"{lv[str(l)][0]}/{lv[str(l)][1]}" for l in [1, 2, 3, 4]) +
                     f" | {x['defect_reflection_events_detected'][0]}/{x['defect_reflection_events_detected'][1]} | "
                     f"{x['healthy']['false_episodes']} | {x['reflection_only']['false_episodes']} | {x['exposure_only']['false_episodes']} | "
                     f"{x['reopen']['false_episodes']} | {x['event_auroc_clean_vs_healthy_steps']:.3f} | {x['event_ap_clean_vs_healthy_steps']:.3f} | "
                     f"{x['onset_s_clean']['median'] if x['onset_s_clean']['median'] is None else round(x['onset_s_clean']['median'], 3)} |")
        L += ["", "| Secondary (pixel - D) | point | 95% CI | p raw | p Holm |", "|---|---|---|---|---|"]
        for k, v in p["SECONDARY_pixel_minus_D"].items():
            L.append(f"| {k} | {v['point']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['p_raw']:.4f} | {v['p_holm']:.4f} |")
        for nm in ["r6_sensitivity_envelope_k2", "r6_sensitivity_global_k4"]:
            s = res[nm]
            L += ["", f"### {nm}", "", "| Arm | clean events | refl. events | healthy FA ep | refl.-only FA ep | exposure FA ep | reopen FA ep |", "|---|---|---|---|---|---|---|"]
            for a in R6_ARMS:
                x = s["arms"][a]
                L.append(f"| {a} | {x['clean_defect_events_detected'][0]}/80 | {x['defect_reflection_events_detected'][0]}/40 | "
                         f"{x['healthy']['false_episodes']} | {x['reflection_only']['false_episodes']} | {x['exposure_only']['false_episodes']} | {x['reopen']['false_episodes']} |")
    if "liftdoor" in res:
        for score, rep in res["liftdoor"].items():
            for c, r in rep.items():
                L += ["", f"## Lift-door 6-clip, {score} score, k = 2, bank {c} (n = 1 clip per scenario; indicative)", "",
                      "| Detector | events | event AUROC | event AP | false episodes (6 clips, 96 s) | of which overlap disturbance | v1 FA samples | onset median s |",
                      "|---|---|---|---|---|---|---|---|"]
                for d in LD_DETS:
                    x = r["pooled6"][d]
                    au = "NA" if x["event_auroc"] is None else f"{x['event_auroc']:.3f}"
                    ap = "NA" if x["event_ap"] is None else f"{x['event_ap']:.3f}"
                    L.append(f"| {d} | {x['events_detected']}/{x['events']} | {au} | {ap} | {x['false_episodes']} | "
                             f"{x['false_episodes_overlapping_disturbance']} | {x['false_alert_samples_v1']} | {x['onset_median_s']} |")
                if "bootstrap_indicative" in r:
                    L += ["", "| Contrast | metric | point | 95% interval (clip bootstrap, indicative) |", "|---|---|---|---|"]
                    for k, vv in r["bootstrap_indicative"]["contrasts"].items():
                        for key, v in vv.items():
                            L.append(f"| {k} | {key} | {v['point']} | {v['ci95']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "run.log")
    t0 = time.time()
    want = (root / "a/DESIGN.sha256").read_text().split()[0]
    got = sha(root / "a/DESIGN.md")
    if want != got:
        log(f"STOP: DESIGN sha {got} != {want}")
        return 3
    log(f"design hash OK {got}; python {sys.version.split()[0]} numpy {np.__version__}")
    res = {"design_sha256": got, "script_sha256": sha(Path(__file__)), "python": sys.version.split()[0], "numpy": np.__version__}
    try:
        r6_analysis(root, log, res)
    except Exception as e:  # report, continue to lift-door
        import traceback
        log("R6 ERROR " + traceback.format_exc())
        res["r6_error"] = repr(e)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    try:
        liftdoor_analysis(root, log, res)
    except Exception as e:
        import traceback
        log("LIFT-DOOR ERROR " + traceback.format_exc())
        res["liftdoor_error"] = repr(e)
    res["runtime_s"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    try:
        (out / "tables.md").write_text(tables(res))
    except Exception:
        import traceback
        log("TABLES ERROR " + traceback.format_exc())
    log(f"wrote results.json sha256 {sha(out / 'results.json')}; runtime {res['runtime_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
