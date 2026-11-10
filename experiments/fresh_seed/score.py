"""Track B scorer (pre-registered in DESIGN.md; hash-gated). Runs on EC2 (CPU), reads clips from S3.

Detector maps come ONLY from the frozen suite (lift_vjepa_suite.map_for / grey_patches / unit, imported read-only,
python -B). Thresholds: per-detector envelope tau from leave-one-FAMILY-out BANK maps (+1e-4, suite v2 recipe);
alert multiplier lambda from CAL only (false-alarm budget 1 alert-episode per camera-hour); >= 2 consecutive token
steps (persistent score, finding-1 step 0b equivalence). TEST is touched only after lambda is fixed.

python -B score.py --jobs all_clips.json --s3 s3://.../b --work /opt/b/work --suite /opt/b/code/suite --progress p.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

FAULT_TYPES = ("door_desync", "interior_light_flicker")
DETS = {"D_patchKNN": "D", "G_pooled": "D", "pixel_patchKNN": "pixel", "illum_pixel_patchKNN": "illum"}
VARIANTS = {"A0_aligned": (0, 1.0), "phase_0.25s": (1, 1.0), "phase_0.50s": (2, 1.0), "phase_1.00s": (4, 1.0),
            "speed_0.95": (0, 0.95), "speed_1.05": (0, 1.05)}
BUDGET_PER_H = 1.0
B_BOOT, SEED, MARGIN = 2000, 20261109, -0.05
CLIP_S = 16.0
V1 = None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Prog:
    def __init__(self, p):
        self.p = p

    def __call__(self, **kw):
        kw["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        s = json.dumps(kw)
        print(s, flush=True)
        with open(self.p, "a") as f:
            f.write(s + "\n")


def halves(x):
    return [x[:16], x[16:32]]


def persistent(s):
    n = len(s)
    p = np.empty_like(s)
    for t in range(n):
        c = []
        if t > 0:
            c.append(min(s[t - 1], s[t]))
        if t < n - 1:
            c.append(min(s[t], s[t + 1]))
        p[t] = max(c)
    return p


def clip_dir(work, cid):
    return Path(work) / "data" / cid


def load_tok(work, cid, o):
    return V1.unit(np.load(clip_dir(work, cid) / f"tok_canon_o{o}.npy").astype(np.float32))


def load_frames(work, cid):
    with np.load(clip_dir(work, cid) / "frames.npz") as z:
        return z["frames"]


def feats_for(work, cid, o):
    fr = load_frames(work, cid)[o:o + 64]
    assert len(fr) == 64
    return {"D": load_tok(work, cid, o), "pixel": V1.grey_patches(fr, False), "illum": V1.grey_patches(fr, True)}


def variant_scores(args):
    """For one bank variant: tau per detector from leave-one-family-out bank maps; s_env [32] for CAL+TEST clips."""
    vname, work, suite, bank_clips, eval_clips = args
    global V1
    sys.path.insert(0, suite)
    import lift_vjepa_suite as v1
    V1 = v1
    o, spd = VARIANTS[vname]
    t0 = time.time()
    bank = []
    for c in bank_clips:
        cid = c["clip_id"] if spd == 1.0 else f"{c['clip_id']}_spd{int(round(spd * 100))}"
        f = feats_for(work, cid, o)
        bank.append((c["family"], f))
    out = {"variant": vname, "tau_stats": {}, "s_env": {d: {} for d in DETS}}
    for det, fam in DETS.items():
        cyc = [(fa, halves(f[fam])[k]) for fa, f in bank for k in range(2)]
        loo = [v1.map_for(det, x, [y for fb, y in cyc if fb != fa]) for fa, x in cyc]
        tau = np.max(loo, 0) + 1e-4
        allb = [y for _, y in cyc]
        out["tau_stats"][det] = {"median": float(np.median(tau)), "max": float(tau.max()), "n_bank_cycles": len(allb)}
        for c in eval_clips:
            F = feats_for(work, c["clip_id"], 0)[fam]
            m = np.concatenate([v1.map_for(det, cy, allb) for cy in halves(F)], 0)
            tt = np.concatenate([tau, tau], 0)
            out["s_env"][det][c["clip_id"]] = (m / tt).reshape(32, -1).max(1).astype(float).tolist()
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ------------------------------------------------------------------ metrics
def midrank(x):
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    return ((cum - cnt + 1 + cum) / 2.0)[inv]


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = midrank(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def labels(gt):
    """Per 4 fps sample (64): fault-event windows (+-0.5 s), negatives (observable, outside windows)."""
    times = np.arange(64) / 4.0
    near = np.zeros(64, bool)
    evs = []
    for e in gt["event_intervals"]:
        if e["type"] in FAULT_TYPES and e["start_sec"] < CLIP_S:
            w = (times >= e["start_sec"] - 0.5) & (times < e["end_sec_exclusive"] + 0.5)
            near |= w
            evs.append({"type": e["type"], "start": e["start_sec"], "w": w})
    sf, fr = gt["fps"], gt["frames"]
    obs = np.array([bool(fr[min(len(fr) - 1, int(round(k * sf / 4)))].get("observable", True)) for k in range(64)])
    return {"near": near, "neg": ~near & obs, "events": evs}


def episodes(alert):
    """Contiguous runs of alert samples -> list of index arrays."""
    eps, cur = [], []
    for i, a in enumerate(alert):
        if a:
            cur.append(i)
        elif cur:
            eps.append(cur)
            cur = []
    if cur:
        eps.append(cur)
    return eps


def clip_eval(p_samp, lam, L):
    alert = p_samp > lam
    hits = [bool(alert[e["w"]].any()) for e in L["events"]]
    delays = []
    for e, h in zip(L["events"], hits):
        if h:
            first = np.flatnonzero(alert & e["w"])[0] / 4.0
            delays.append(first - e["start"])
    fa = sum(1 for ep in episodes(alert) if not L["near"][ep].any())
    return hits, fa, delays


def choose_lambda(p_by_clip, cal_ids):
    hours = len(cal_ids) * CLIP_S / 3600
    allowed = int(np.floor(BUDGET_PER_H * hours))
    if allowed == 0:
        return max(float(np.max(p_by_clip[c])) for c in cal_ids), allowed, hours
    cands = np.unique(np.concatenate([p_by_clip[c] for c in cal_ids]))[::-1]
    lam = cands[0]
    for v in cands:  # descending; lowest threshold with <= allowed episodes
        n = sum(len(episodes(p_by_clip[c] > v)) for c in cal_ids)
        if n > allowed:
            break
        lam = v
    return float(lam), allowed, hours


def summarise(p, lam, test_ids, labs, scen):
    hits, fa, dl, per_scen = [], 0, [], {}
    for c in test_ids:
        h, f, d = clip_eval(p[c], lam, labs[c])
        hits += h
        fa += f
        dl += d
        ps = per_scen.setdefault(scen[c], {"events": 0, "hits": 0, "fa_episodes": 0, "clips": 0})
        ps["events"] += len(h)
        ps["hits"] += sum(h)
        ps["fa_episodes"] += f
        ps["clips"] += 1
    hours = len(test_ids) * CLIP_S / 3600
    return {"recall": (sum(hits) / len(hits)) if hits else None, "n_events": len(hits), "n_hit": int(sum(hits)),
            "fa_episodes": int(fa), "test_hours": hours, "fa_per_hour": fa / hours,
            "median_onset_delay_s": float(np.median(dl)) if dl else None, "per_scenario": per_scen}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--progress", required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--boot", type=int, default=B_BOOT, help="dry-run only; the real run uses the default")
    ap.add_argument("--no-sync", action="store_true", help="dry-run only")
    a = ap.parse_args()
    prog = Prog(a.progress)
    work, code = Path(a.work), Path(__file__).resolve().parent
    # ---- design hash gate
    want = (code / "DESIGN.sha256").read_text().split()[0]
    got = sha(code / "DESIGN.md")
    if want != got:
        prog(stage="STOP_design_hash", want=want, got=got)
        return 3
    plan = json.loads(Path(a.jobs).read_text())
    clips = plan["clips"]
    hashes = {"DESIGN.md": got, "score.py": sha(__file__), "all_clips.json": sha(a.jobs),
              "lift_vjepa_suite.py": sha(Path(a.suite) / "lift_vjepa_suite.py")}
    prog(stage="design_ok", **hashes)
    # ---- download
    t0 = time.time()
    if not a.no_sync:
        subprocess.run(["aws", "s3", "sync", "--only-show-errors", "--exclude", "*/clean.npz", "--exclude", "*.mp4",
                        "--exclude", "*.png", f"{a.s3}/data/", str(work / "data")], check=True)
    missing = [c["clip_id"] for c in clips if not (clip_dir(work, c["clip_id"]) / "DONE").exists()]
    prog(stage="downloaded", s=round(time.time() - t0, 1), n=len(clips), missing=len(missing))
    if missing:
        prog(stage="STOP_missing_clips", missing=missing[:50])
        return 4
    # integrity: token window hashes match frames
    for c in clips:
        meta = json.loads((clip_dir(work, c["clip_id"]) / "meta.json").read_text())
        fr = load_frames(work, c["clip_id"])
        for o in c["offsets"]:
            h = hashlib.sha256(fr[o:o + 64].tobytes()).hexdigest()[:16]
            if meta["tokens"][f"tok_canon_o{o}.npy"]["frames_sha16"] != h:
                prog(stage="STOP_integrity", clip=c["clip_id"], offset=o)
                return 5
    # split checks: disjoint families, no TEST background in BANK/CAL
    fam = {s: {c["family"] for c in clips if c["split"] == s} for s in ("bank", "cal", "test")}
    leaf = {s: {c["leaf_seed"] for c in clips if c["split"] == s} for s in ("bank", "cal", "test")}
    assert not (fam["bank"] & fam["test"]) and not (fam["cal"] & fam["test"]) and not (fam["bank"] & fam["cal"])
    assert not (leaf["bank"] & leaf["test"]) and not (leaf["cal"] & leaf["test"])
    bank_clips = [c for c in clips if c["split"] == "bank" and c["speed_scale"] == 1.0]
    cal = [c for c in clips if c["split"] == "cal"]
    test = [c for c in clips if c["split"] == "test"]
    gts = {c["clip_id"]: json.loads((clip_dir(work, c["clip_id"]) / "ground_truth.json").read_text())
           for c in cal + test}
    labs = {k: labels(g) for k, g in gts.items()}
    scen = {c["clip_id"]: c["scenario"] for c in cal + test}
    famof = {c["clip_id"]: c["family"] for c in cal + test}
    prog(stage="integrity_ok", bank_families=sorted(fam["bank"]), cal_families=sorted(fam["cal"]),
         test_families=sorted(fam["test"]))
    # ---- maps (frozen suite) per bank variant, in parallel
    jobs = [(v, str(work), a.suite, bank_clips, cal + test) for v in VARIANTS]
    with mp.get_context("spawn").Pool(min(a.procs, len(jobs))) as pool:
        V = {}
        for r in pool.imap_unordered(variant_scores, jobs):
            V[r["variant"]] = r
            prog(stage="variant_scored", variant=r["variant"], s=r["seconds"])
    # ---- per-sample scores
    S, P = {}, {}
    for v in VARIANTS:
        S[v], P[v] = {}, {}
        for d in DETS:
            S[v][d] = {c: np.repeat(np.array(x), 2) for c, x in V[v]["s_env"][d].items()}
            P[v][d] = {c: np.repeat(persistent(np.array(x)), 2) for c, x in V[v]["s_env"][d].items()}
    cal_ids = [c["clip_id"] for c in cal]
    cal5_ids = [c["clip_id"] for c in cal if c["scenario"] != "healthy_ball_reopen"]
    test_ids = [c["clip_id"] for c in test]
    anom_ids = [c for c in test_ids if gts[c]["video_health"] == "anomaly"]
    pooled6_ids = anom_ids + [c for c in test_ids if scen[c] == "healthy_ball_reopen"]

    def pooled_auroc(sc, ids):
        pos = np.concatenate([sc[c][labs[c]["near"]] for c in ids])
        neg = np.concatenate([sc[c][labs[c]["neg"]] for c in ids])
        return auroc(pos, neg)

    def event_auroc(pc, ids_anom, ids_healthy):
        pos = [float(pc[c][e["w"]].max()) for c in ids_anom for e in labs[c]["events"]]
        neg = [float(pc[c].max()) for c in ids_healthy]
        return auroc(pos, neg)

    healthy_test = [c for c in test_ids if gts[c]["video_health"] == "healthy"]
    res = {"hashes": hashes, "variants": {}, "tau_stats": {v: V[v]["tau_stats"] for v in VARIANTS},
           "n": {"bank_clips": len(bank_clips), "cal_clips": len(cal_ids), "test_clips": len(test_ids),
                 "test_fault_events": sum(len(labs[c]["events"]) for c in anom_ids),
                 "families": {k: len(x) for k, x in fam.items()}}}
    for v in VARIANTS:
        rv = {}
        for d in DETS:
            row = {}
            for calname, cids in (("CAL6", cal_ids), ("CAL5", cal5_ids)):
                lam, allowed, hrs = choose_lambda(P[v][d], cids)
                row[calname] = {"lambda": lam, "cal_allowed_episodes": allowed, "cal_hours": hrs,
                                **summarise(P[v][d], lam, test_ids, labs, scen)}
            row["pooled6_auroc_s"] = pooled_auroc(S[v][d], pooled6_ids)
            row["pooled6_auroc_p"] = pooled_auroc(P[v][d], pooled6_ids)
            row["pooled_all_test_auroc_s"] = pooled_auroc(S[v][d], test_ids)
            row["event_auroc_p"] = event_auroc(P[v][d], anom_ids, healthy_test)
            rv[d] = row
        res["variants"][v] = rv
    prog(stage="point_estimates_done")

    # ---- family-level cluster bootstrap (CAL and TEST families resampled jointly; lambda recomputed)
    cal_f = sorted({famof[c] for c in cal_ids})
    test_f = sorted({famof[c] for c in test_ids})
    by_f = {f: [c for c in cal_ids + test_ids if famof[c] == f] for f in cal_f + test_f}
    rng = np.random.default_rng(SEED)
    contrasts = {"PRIMARY recall pixel-D (A0, CAL6)": ("A0_aligned", "recall", "pixel_patchKNN", "D_patchKNN"),
                 "S1 recall G-D (A0)": ("A0_aligned", "recall", "G_pooled", "D_patchKNN"),
                 "S2 recall illum-D (A0)": ("A0_aligned", "recall", "illum_pixel_patchKNN", "D_patchKNN"),
                 "S3 pooled6 AUROC_s pixel-D (A0)": ("A0_aligned", "auroc", "pixel_patchKNN", "D_patchKNN"),
                 "S4 FA/h pixel-D (A0)": ("A0_aligned", "fa", "pixel_patchKNN", "D_patchKNN")}
    for v in VARIANTS:
        if v != "A0_aligned":
            contrasts[f"AL recall pixel-D ({v})"] = (v, "recall", "pixel_patchKNN", "D_patchKNN")
    draws = {k: [] for k in contrasts}
    per_det = {(v, d, m): [] for v in VARIANTS for d in DETS for m in ("recall", "fa")}
    for b in range(a.boot):
        cf = rng.choice(cal_f, len(cal_f))
        tf = rng.choice(test_f, len(test_f))
        cids = [c for f in cf for c in by_f[f] if c in cal_ids]
        tids = [c for f in tf for c in by_f[f]]
        cache = {}
        for v in VARIANTS:
            for d in DETS:
                lam, _, _ = choose_lambda(P[v][d], cids)
                hits, fa = [], 0
                for c in tids:
                    h, f_, _ = clip_eval(P[v][d][c], lam, labs[c])
                    hits += h
                    fa += f_
                cache[(v, d, "recall")] = np.mean(hits) if hits else np.nan
                cache[(v, d, "fa")] = fa / (len(tids) * CLIP_S / 3600)
                per_det[(v, d, "recall")].append(cache[(v, d, "recall")])
                per_det[(v, d, "fa")].append(cache[(v, d, "fa")])
        p6 = [c for c in tids if c in pooled6_ids]
        for k, (v, m, x, y) in contrasts.items():
            if m == "auroc":
                draws[k].append(pooled_auroc(S[v][x], p6) - pooled_auroc(S[v][y], p6))
            else:
                draws[k].append(cache[(v, x, m)] - cache[(v, y, m)])
        if b % 200 == 0:
            prog(stage="bootstrap", b=b)

    def point(v, m, x, y):
        if m == "auroc":
            return res["variants"][v][x]["pooled6_auroc_s"] - res["variants"][v][y]["pooled6_auroc_s"]
        key = "recall" if m == "recall" else "fa_per_hour"
        return res["variants"][v][x]["CAL6"][key] - res["variants"][v][y]["CAL6"][key]

    inf = {}
    for k, (v, m, x, y) in contrasts.items():
        dd = np.array(draws[k], float)
        dd = dd[np.isfinite(dd)]
        p2 = float(min(1.0, 2 * min(np.mean(dd <= 0), np.mean(dd >= 0))))
        inf[k] = {"point": point(v, m, x, y), "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                  "p_two_sided_boot": p2, "n_draws": int(len(dd))}
        if m == "recall":
            inf[k]["noninferior_margin_-0.05"] = bool(inf[k]["ci95"][0] > MARGIN)
    sec = sorted([k for k in inf if not k.startswith("PRIMARY")], key=lambda k: inf[k]["p_two_sided_boot"])
    mmax = len(sec)
    running = 0.0
    for i, k in enumerate(sec):  # Holm step-down, alpha 0.05, monotone adjusted p
        running = max(running, min(1.0, (mmax - i) * inf[k]["p_two_sided_boot"]))
        inf[k]["holm_p"] = running
        inf[k]["holm_reject_0.05"] = bool(running < 0.05)
    res["inference"] = inf
    res["bootstrap"] = {"B": a.boot, "seed": SEED, "unit": "family (CAL and TEST resampled jointly; BANK fixed)",
                        "per_detector_ci95": {f"{v}|{d}|{m}": [float(np.nanpercentile(x, 2.5)),
                                                                float(np.nanpercentile(x, 97.5))]
                                              for (v, d, m), x in per_det.items()}}
    res["per_sample_scores_s_env_token"] = {v: {d: V[v]["s_env"][d] for d in DETS} for v in VARIANTS}
    res["labels"] = {c: {"near": labs[c]["near"].astype(int).tolist(), "n_events": len(labs[c]["events"])}
                     for c in cal_ids + test_ids}
    res["numpy"] = np.__version__
    out = work / "results"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    if not a.no_sync:
        subprocess.run(["aws", "s3", "cp", "--only-show-errors", str(out / "results.json"),
                        f"{a.s3}/results/results.json"], check=True)
    prog(stage="all_done", results_sha256=sha(out / "results.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
