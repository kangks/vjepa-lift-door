"""Track C: robustness controls for finding 2 (V-JEPA dense patch-kNN under capture perturbations).

EC2 (g6.xlarge, CUDA). Pre-declared design: DESIGN.md (sha256 in DESIGN.sha256, checked first).
Imports the frozen suite (v1, v2) and the R8 harness read-only; run with python -B.

  python -B run_c.py --stage pilot|encode|score --inp /work/in --tok /work/tok --out /work/out
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
TEST_SEEDS = [1001, 1002, 1003, 1004, 1005]
SIGMAS = [1, 2, 3, 6]
BANK_R = [0, 1, 2, 3]
DET_LEVELS = [("gain", 0.8), ("gain", 0.9), ("gain", 0.95), ("gain", 1.05), ("gain", 1.1), ("gain", 1.25),
              ("shift", 1), ("shift", 2), ("shift", 4), ("shift", 8),
              ("jpeg", 90), ("jpeg", 70), ("jpeg", 50), ("jpeg", 30)]
BLURS = {"blur1": 1.0, "blur2": 2.0}
BLUR_SIGMAS = [1, 3, 6]
BLUR_JPEG = [90, 50, 30]
TAVG = {"tavg2": 2, "tavg4": 4}
RULES = ["ENV", "BLKQ99", "BLKQ100", "GLBQ9999", "GLBQ100"]
FAULT_TYPES = ("door_desync", "interior_light_flicker")
BOOT_B, BOOT_SEED = 1000, 20261108
MIN_FREE_GIB = 6.0


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def jdump(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    tmp.replace(path)


class Log:
    def __init__(self, path):
        self.fh = open(path, "a")

    def __call__(self, msg):
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()


def free_gib(p="/"):
    return shutil.disk_usage(p).free / 2 ** 30


# ------------------------------------------------------------------ perturbations
def to_u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def p_gain(f, g):
    return to_u8(f.astype(np.float32) * g)


def p_shift(f, s):
    s = int(s)
    return np.ascontiguousarray(np.pad(f, ((0, 0), (s, 0), (s, 0), (0, 0)), mode="edge")[:, :384, :384])


def p_jpeg(f, q):
    out = []
    for fr in f:
        buf = io.BytesIO()
        Image.fromarray(fr).save(buf, format="JPEG", quality=int(q))
        buf.seek(0)
        out.append(np.asarray(Image.open(buf).convert("RGB")))
    return np.stack(out)


def p_noise(f, sigma, seedseq):
    rng = np.random.default_rng(seedseq)
    return to_u8(f.astype(np.float32) + rng.normal(0.0, float(sigma), size=f.shape).astype(np.float32))


def gauss_blur(f, sb):
    r = int(math.ceil(3 * sb))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sb) ** 2)
    k = (k / k.sum()).astype(np.float32)
    a = np.pad(f.astype(np.float32), ((0, 0), (r, r), (0, 0), (0, 0)), mode="reflect")
    H = f.shape[1]
    y = sum(k[i] * a[:, i:i + H] for i in range(2 * r + 1))
    a = np.pad(y, ((0, 0), (0, 0), (r, r), (0, 0)), mode="reflect")
    W = f.shape[2]
    y = sum(k[i] * a[:, :, i:i + W] for i in range(2 * r + 1))
    return to_u8(y)


# ------------------------------------------------------------------ metrics
def auroc(pos, neg):
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    x = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    r = avg[inv]
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auroc_ref(pos, neg):  # finding-2 definition, used once to check auroc()
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def near_mask(lab):
    times = np.arange(64) / 4.0
    near = np.zeros(64, bool)
    for e in lab["events"]:
        if e["type"] in FAULT_TYPES:
            near |= (times >= e["start_sec"] - 0.5) & (times < e["end_sec_exclusive"] + 0.5)
    return near


def q(x, ps):
    x = np.asarray(x, np.float64).ravel()
    return {f"p{p:g}": float(np.percentile(x, p)) for p in ps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pilot", "encode", "score"])
    ap.add_argument("--inp", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encode-budget-s", type=float, default=3 * 3600)
    ap.add_argument("--smoke", action="store_true", help="score-stage code-path test on pilot data (1 seed, sigma 1)")
    a = ap.parse_args()
    global TEST_SEEDS, SIGMAS, BANK_R
    if a.smoke:
        TEST_SEEDS, SIGMAS, BANK_R = [1001], [1], [0]
    t0 = time.time()
    inp, TOK, OUT = Path(a.inp), Path(a.tok), Path(a.out)
    TOK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log = Log(OUT / "run.log")
    log(f"=== run_c stage={a.stage} python {sys.version.split()[0]} argv {sys.argv}")

    def disk_guard():
        g = free_gib()
        if g < MIN_FREE_GIB:
            log(f"STOP: free disk {g:.2f} GiB < {MIN_FREE_GIB}")
            sys.exit(3)
        return g

    log(f"disk free / {disk_guard():.1f} GiB")
    dsha = sha(HERE / "DESIGN.md")
    if dsha != (HERE / "DESIGN.sha256").read_text().split()[0]:
        log("STOP: DESIGN.md sha256 mismatch")
        return 4
    log(f"DESIGN.md sha256 {dsha} matches")
    suite, r8, llr, ds = inp / "suite", inp / "r8", inp / "llr", inp / "lift_dataset"
    ckpt = inp / "ckpt" / "vitl_ema_encoder_only.pt"
    sys.path.insert(0, str(r8))
    sys.path.insert(0, str(suite))
    import run_robustness as rr
    import lift_vjepa_suite as v1
    import lift_vjepa_suite_v2 as v2
    expect = {suite / "lift_vjepa_suite.py": "45e0e712e0759e5bfb858e16ba11d24433419311dd8b51bfef8d82de951208cb",
              suite / "lift_vjepa_suite_v2.py": "d7f04e56981443527c3e45b7ea49dd5f566308e697551ab4972cf103fc7706fc",
              suite / "results_v2/summary.json": "bb876097593b912d5d6b8ad21ed54c28580e54a78a5d05a0d00105a41c75f716",
              llr / "scripts/vjepa/encoder.py": "87ccc263e7764ad42f2c9ee1f0483d4b614cdfd6bdcaa2cf36a758fe5dc53356",
              r8 / "run_robustness.py": "1fdd865b5183f5c06f28df6b36a9f04c6a4dc6f9513e7fcab2a99e59dfcbaa01",
              ckpt: "06248f29136a5d26c30e466d717d8845c25341e220f500468c0c162a3317cf20"}
    inputs = {}
    for p, e in expect.items():
        s = sha(p)
        inputs[str(p)] = s
        if s != e:
            log(f"STOP: sha mismatch {p}: {s}")
            return 4
    inputs[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    log("frozen input hashes verified")
    DETS, TESTS = rr.DETS, rr.TESTS
    bank = list(v1.FROZEN_DESIGN["healthy_bank"])

    # ---------------- clean frames, labels
    cases = {p.parent.name: p.parent for p in sorted(ds.glob("*/*/ground_truth.json"))}
    frames, labs, fsha = {}, {}, {}
    mps_cache = {}
    g0f = True
    for n in bank + TESTS:
        frames[n] = v1.load_frames(cases[n])
        assert frames[n].shape == (64, 384, 384, 3) and frames[n].dtype == np.uint8
        labs[n] = v1.sample_labels(json.loads((cases[n] / "ground_truth.json").read_text()))
        fsha[n] = hashlib.sha256(frames[n].tobytes()).hexdigest()
        cp = suite / "results_v1/token_cache" / f"{n}_{fsha[n][:16]}.npy"
        g0f &= cp.exists()
        mps_cache[n] = cp
    log(f"G0f frames-key match: {g0f}")
    if not g0f:
        return 1
    near = {t: near_mask(labs[t]) for t in TESTS}
    keep = {t: near[t] | labs[t]["observable"] for t in TESTS}

    # ================================================================ condition registry
    def build_frames(clip, pert, pipeline):
        kind = pert[0]
        f = frames[clip]
        if kind == "clean":
            g = f
        elif kind == "gain":
            g = p_gain(f, pert[1])
        elif kind == "shift":
            g = p_shift(f, pert[1])
        elif kind == "jpeg":
            g = p_jpeg(f, pert[1])
        elif kind == "noise":  # test noise: (noise, sigma, seed)
            g = p_noise(f, pert[1], [pert[2], TESTS.index(clip)])
        elif kind == "bnoise":  # bank noise: (bnoise, sigma, r)
            g = p_noise(f, pert[1], [5000 + pert[2], bank.index(clip)])
        elif kind == "r8noise":
            g = rr.perturb("P5_noise_sigma6", f)
        else:
            raise ValueError(kind)
        if pipeline in BLURS:
            g = gauss_blur(g, BLURS[pipeline])
        assert g.shape == (64, 384, 384, 3) and g.dtype == np.uint8
        return g

    def pid(pert):
        return "_".join(f"{x:g}" if isinstance(x, (int, float)) else str(x) for x in pert)

    def tname(pipeline, clip, pert):
        return f"{pipeline}__{clip}__{pid(pert)}"

    def registry():
        L = []  # (pipeline, clip, pert)
        for c in bank + TESTS:
            L.append(("raw", c, ("clean",)))
        for t in TESTS:
            for lv in DET_LEVELS:
                L.append(("raw", t, lv))
            L.append(("raw", t, ("r8noise", 6, 71)))
            for s in SIGMAS:
                for sd in TEST_SEEDS:
                    L.append(("raw", t, ("noise", s, sd)))
        for b in bank:
            for s in SIGMAS:
                for r in BANK_R:
                    L.append(("raw", b, ("bnoise", s, r)))
            for lv in DET_LEVELS:
                L.append(("raw", b, lv))
        for pl in BLURS:
            for c in bank + TESTS:
                L.append((pl, c, ("clean",)))
            for t in TESTS:
                for s in BLUR_SIGMAS:
                    for sd in TEST_SEEDS:
                        L.append((pl, t, ("noise", s, sd)))
                for qq in BLUR_JPEG:
                    L.append((pl, t, ("jpeg", qq)))
        return L

    manifest_p = OUT / "token_manifest.json"
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {}
    state = {}

    def get_encoder():
        if "enc" not in state:
            import torch
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            mod = v1.encoder_module(llr)
            state["mod"] = mod
            state["enc"] = mod.build_encoder("vitl", ckpt, img_size=384, num_frames=16).to("cuda")
            state["torch"] = torch
        return state["mod"], state["enc"]

    def encode(name, fr, path=None, record=True):
        mod, enc = get_encoder()
        path = path or TOK / f"{name}.npy"
        ts = time.time()
        out = [mod.dense_tokens(enc, fr[s:s + 16], device="cuda") for s in range(0, 64, 16)]
        tok = np.concatenate(out, 0)
        assert tok.shape == (32, 24, 24, 1024)
        np.save(path, tok.astype(np.float16))
        dt = time.time() - ts
        if record:
            manifest[name] = {"frames_sha256": hashlib.sha256(fr.tobytes()).hexdigest(), "file_sha256": sha(path),
                              "encode_s": round(dt, 3)}
        return dt

    def load_tok(name):
        return v1.unit(np.load(TOK / f"{name}.npy").astype(np.float32))

    def have(name):
        return name in manifest and (TOK / f"{name}.npy").exists()

    # ================================================================ scoring helpers
    def loo_maps(det, F, names, src):
        maps = []
        for b in names:
            others = [cyc for o in names if src[o] != src[b] for cyc in v2.cycles(F[o])]
            for c in range(2):
                maps.append(v1.map_for(det, v2.cycles(F[b])[c], others))
        return np.stack(maps)

    def taus_from(loo):
        sh = loo.shape[1:]
        return {"ENV": loo.max(0) + 1e-4,
                "BLKQ99": np.broadcast_to(np.percentile(loo, 99, axis=(0, 1)), sh) + 1e-4,
                "BLKQ100": np.broadcast_to(loo.max((0, 1)), sh) + 1e-4,
                "GLBQ9999": np.full(sh, np.percentile(loo, 99.99)) + 1e-4,
                "GLBQ100": np.full(sh, loo.max()) + 1e-4}

    def make_bankset(det, F, names, src):
        loo = loo_maps(det, F, names, src)
        return {"names": names, "taus": taus_from(loo),
                "cycles": [cyc for n in names for cyc in v2.cycles(F[n])]}

    def score_clip(det, bs, test_feat, lab):
        m = np.concatenate([v1.map_for(det, cyc, bs["cycles"]) for cyc in v2.cycles(test_feat)], 0)
        o = {}
        for rule, tau in bs["taus"].items():
            tt = np.concatenate([tau, tau], 0)
            alerts = np.repeat(v2.persist((m > tt).reshape(32, -1).any(1)), 2)
            ev = v1.evaluate(alerts, lab)
            o[rule] = (int(sum(e["alerted"] for e in ev["events"] if e["type"] in FAULT_TYPES)),
                       int(ev["false_alert_samples"]))
        tt = np.concatenate([bs["taus"]["ENV"], bs["taus"]["ENV"]], 0)
        s = np.repeat((m / tt).reshape(32, -1).max(1), 2)
        return o, s

    def featurise(u, fr, pipeline_tavg=None):
        f = {"D": u, "pixel": v1.grey_patches(fr, False), "illum": v1.grey_patches(fr, True)}
        return f

    def tavg(x, w, renorm):
        cs = np.cumsum(x, 0, dtype=np.float64)
        out = np.empty_like(x)
        for t in range(x.shape[0]):
            lo = max(0, t - w + 1)
            sm = cs[t] - (cs[lo - 1] if lo > 0 else 0.0)
            out[t] = sm / (t - lo + 1)
        return v1.unit(out) if renorm else out

    def boot_stats(FA, H, S, rng):
        """FA,H: [ns,6]; S: [ns,6,64] ENV-normalised scores. Returns point + 95% CIs."""
        ns = FA.shape[0]
        Y = np.stack([near[t] for t in TESTS])
        K = np.stack([keep[t] for t in TESTS])

        def stat(si, ci):
            fa = FA[np.ix_(si, ci)].sum(1).mean()
            h = H[np.ix_(si, ci)].sum(1).mean()
            nf = sum(len([e for e in labs[TESTS[c]]["events"] if e["type"] in FAULT_TYPES]) for c in ci)
            ss = S[np.ix_(si, ci)]
            kk = np.broadcast_to(K[ci], ss.shape)
            yy = np.broadcast_to(Y[ci], ss.shape)
            sv, yv = ss[kk], yy[kk]
            au = auroc(sv[yv], sv[~yv]) if yv.any() and (~yv).any() else np.nan
            return fa, h, (h / nf if nf else np.nan), au

        pt = stat(np.arange(ns), np.arange(6))
        bs = np.array([stat(rng.integers(0, ns, ns), rng.integers(0, 6, 6)) for _ in range(BOOT_B)])
        ci = lambda j: [float(np.nanpercentile(bs[:, j], 2.5)), float(np.nanpercentile(bs[:, j], 97.5))]
        return {"sum_FA": float(pt[0]), "sum_FA_ci95": ci(0), "faults_found": float(pt[1]), "faults_ci95": ci(1),
                "fault_recall": float(pt[2]), "fault_recall_ci95": ci(2), "auroc": float(pt[3]),
                "auroc_ci95": ci(3), "n_seeds": int(ns), "FA_per_seed": FA.sum(1).tolist(),
                "faults_per_seed": H.sum(1).tolist(), "FA_per_clip_mean": dict(zip(TESTS, FA.mean(0).tolist()))}

    # ================================================================ G0 (numpy, MPS cache)
    res = {"design_sha256": dsha, "stage": a.stage}
    mps_tok = {n: v1.unit(np.load(mps_cache[n]).astype(np.float32)) for n in bank + TESTS}
    pix = {n: v1.grey_patches(frames[n], False) for n in bank + TESTS}
    ill = {n: v1.grey_patches(frames[n], True) for n in bank + TESTS}
    featsM = {"D": mps_tok, "pixel": pix, "illum": ill}
    ref = json.loads((suite / "results_v2/summary.json").read_text())
    ok = True
    for det, fam in DETS.items():
        tau = rr.tau_for(v1, v2, det, featsM[fam], bank)
        bs = make_bankset(det, featsM[fam], bank, {b: b for b in bank})
        ok &= bool(np.array_equal(tau, bs["taus"]["ENV"]))
        for t in TESTS:
            r = rr.score(v1, v2, det, featsM[fam], featsM[fam][t], bank, tau, labs[t])
            x = ref["detectors"][det]["envelope"][t]
            ok &= (r["false_alert_samples"] == x["false_alert_samples"] and r["alert_times_s"] == x["alert_times_s"]
                   and [e["alerted"] for e in r["events"]] == [e["alerted"] for e in x["events"]])
            o, _ = score_clip(det, bs, featsM[fam][t], labs[t])
            ok &= o["ENV"] == (r["fault_events_hit"], r["false_alert_samples"])
    log(f"G0 baseline reproduction (MPS tokens, numpy here) + generic scorer == rr.score: {'EXACT' if ok else 'FAILED'}")
    res["G0_exact"] = bool(ok)
    if not ok:
        jdump(OUT / f"results_{a.stage}.json", res)
        return 1
    import numpy
    import PIL
    env = {"python": sys.version.split()[0], "numpy": numpy.__version__, "PIL": PIL.__version__,
           "platform": platform.platform()}
    try:
        import torch, timm, einops
        env.update({"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "timm": timm.__version__, "einops": einops.__version__})
    except Exception as e:  # noqa
        env["torch_error"] = repr(e)
    res["env"] = env
    log(f"env {env}")
    # R8 perturbed-frame arithmetic check (report only)
    r8res = json.loads((r8 / "results.json").read_text())
    r8chk = {}
    pmap = {"P1_gain_0.8": ("gain", 0.8), "P2_gain_1.25": ("gain", 1.25), "P3_shift_8r_8d": ("shift", 8),
            "P4_jpeg_q30": ("jpeg", 30), "P5_noise_sigma6": ("r8noise", 6, 71)}
    for P, pert in pmap.items():
        r8chk[P] = all(hashlib.sha256(build_frames(t, pert, "raw").tobytes()).hexdigest()
                       == r8res["perturbed"][P]["clips"][t]["perturbed_frames_sha256"] for t in TESTS)
    log(f"R8 perturbed-frame sha match (report only): {r8chk}")
    res["r8_perturbed_frames_match"] = r8chk
    assert abs(auroc([3, 1, 2, 2], [2, 0, 1]) - auroc_ref([3, 1, 2, 2], [2, 0, 1])) < 1e-12

    # ================================================================ PILOT
    if a.stage == "pilot":
        times = {}
        for c in bank + TESTS:
            nm = tname("raw", c, ("clean",))
            times[nm] = encode(nm, frames[c])
            disk_guard()
        log(f"clean encodes: {times}")
        jdump(manifest_p, manifest)
        # G1 device + G2 determinism
        g1 = {}
        cuda_tok = {c: load_tok(tname("raw", c, ("clean",))) for c in bank + TESTS}
        for c in bank + TESTS:
            cs = np.sum(cuda_tok[c] * mps_tok[c], -1)
            g1[c] = {"median_cos": float(np.median(cs)), "min_cos": float(cs.min()), "mean_1mcos": float((1 - cs).mean())}
        g1ok = all(v["median_cos"] >= 0.999 for v in g1.values())
        log(f"G1 device CUDA vs MPS: {'PASS' if g1ok else 'FAIL'} {g1}")
        res["G1_device"] = {"pass": g1ok, "per_clip": g1}
        p2 = TOK / "g2_reencode.npy"
        encode("g2", frames["healthy_ball_reopen"], path=p2, record=False)
        a1 = np.load(TOK / f"{tname('raw', 'healthy_ball_reopen', ('clean',))}.npy")
        a2 = np.load(p2)
        ident = bool(np.array_equal(a1, a2))
        cs = np.sum(v1.unit(a1.astype(np.float32)) * v1.unit(a2.astype(np.float32)), -1)
        res["G2_determinism"] = {"byte_identical": ident, "min_cos": float(cs.min())}
        log(f"G2 determinism: {res['G2_determinism']}")
        # CUDA baseline vs MPS baseline decisions
        featsC = {"D": cuda_tok, "pixel": pix, "illum": ill}
        base = {}
        for det, fam in DETS.items():
            for lbl, F in (("cuda", featsC), ("mps", featsM)):
                bs = make_bankset(det, F[fam], bank, {b: b for b in bank})
                per = [score_clip(det, bs, F[fam][t], labs[t]) for t in TESTS]
                base[f"{det}/{lbl}"] = {"faults": sum(p[0]["ENV"][0] for p in per), "FA": sum(p[0]["ENV"][1] for p in per),
                                        "FA_per_clip": [p[0]["ENV"][1] for p in per]}
        log(f"baselines CUDA vs MPS: {base}")
        res["baseline_cuda_vs_mps"] = base
        # one perturbation: noise sigma 1, seed 1001
        tp = []
        for t in TESTS:
            fr = build_frames(t, ("noise", 1, 1001), "raw")
            nm = tname("raw", t, ("noise", 1, 1001))
            tp.append(encode(nm, fr))
        for b in bank:  # smoke-test coverage of the matched-bank path
            nm = tname("raw", b, ("bnoise", 1, 0))
            encode(nm, build_frames(b, ("bnoise", 1, 0), "raw"))
        jdump(manifest_p, manifest)
        ts = time.time()
        bsD = make_bankset("D_patchKNN", featsC["D"], bank, {b: b for b in bank})
        t_loo = time.time() - ts
        ts = time.time()
        pil = {"FA": 0, "faults": 0}
        for t in TESTS:
            u = load_tok(tname("raw", t, ("noise", 1, 1001)))
            o, _ = score_clip("D_patchKNN", bsD, u, labs[t])
            pil["FA"] += o["ENV"][1]
            pil["faults"] += o["ENV"][0]
        t_sc = time.time() - ts
        ts = time.time()
        _ = build_frames(TESTS[0], ("jpeg", 50), "blur2")
        t_cpu = time.time() - ts
        res["pilot"] = {"noise1_s1001_D_ENV": pil, "encode_s_per_clip": tp, "loo_s_R1_D": t_loo,
                        "score_s_6clips_D": t_sc, "jpeg_blur2_frames_s": t_cpu, "clean_encode_s": times,
                        "n_planned_encodes": len(registry()), "runtime_s": round(time.time() - t0, 1)}
        log(f"PILOT {res['pilot']}")
        jdump(OUT / "results_pilot.json", res)
        return 0 if g1ok else 1

    # ================================================================ ENCODE
    if a.stage == "encode":
        L = registry()
        log(f"encode stage: {len(L)} planned, {sum(have(tname(*x)) for x in L)} already present")
        n = 0
        for i, (pl, c, pert) in enumerate(L):
            nm = tname(pl, c, pert)
            if have(nm):
                continue
            if time.time() - t0 > a.encode_budget_s:
                log(f"ENCODE BUDGET reached at {i}/{len(L)}; remaining not encoded")
                break
            if n % 10 == 0:
                disk_guard()
            fr = build_frames(c, pert, pl)
            dt = encode(nm, fr)
            n += 1
            if n % 10 == 0:
                jdump(manifest_p, manifest)
                log(f"encoded {i + 1}/{len(L)} last {nm} {dt:.2f}s elapsed {time.time() - t0:.0f}s free {free_gib():.1f} GiB")
        jdump(manifest_p, manifest)
        log(f"encode stage done: {n} new; total present {sum(have(tname(*x)) for x in L)}/{len(L)}")
        return 0

    # ================================================================ SCORE
    rng = np.random.default_rng(BOOT_SEED)
    L = registry()
    missing = [tname(*x) for x in L if not have(tname(*x))]
    res["missing_encodes"] = missing
    log(f"score stage: {len(missing)} missing encodes")
    clean_name = lambda pl, c: tname(pl, c, ("clean",))

    def feats_for(pl, names_perts):
        """pl in raw/blur1/blur2/tavg2/tavg4 -> {fam: {key: feat}} for (key, clip, pert) list."""
        base_pl = "raw" if pl in TAVG else pl
        out = {"D": {}, "pixel": {}, "illum": {}}
        for key, c, pert in names_perts:
            nm = tname(base_pl, c, pert)
            if not have(nm):
                return None
            u = load_tok(nm)
            fr = build_frames(c, pert, base_pl)
            g, gi = v1.grey_patches(fr, False), v1.grey_patches(fr, True)
            if pl in TAVG:
                w = TAVG[pl]
                u, g, gi = tavg(u, w, True), tavg(g, w, False), tavg(gi, w, False)
            out["D"][key], out["pixel"][key], out["illum"][key] = u, g, gi
        return out

    PIPES = ["raw", "blur1", "blur2", "tavg2", "tavg4"]
    clean = {}
    banksets = {}
    margins = {}
    for pl in PIPES:
        clean[pl] = feats_for(pl, [(c, c, ("clean",)) for c in bank + TESTS])
        if clean[pl] is None:
            log(f"pipeline {pl}: clean encodes missing -> not run")
            continue
        for det, fam in DETS.items():
            banksets[(pl, "clean", det)] = make_bankset(det, clean[pl][fam], bank, {b: b for b in bank})
        # healthy NN margins (finding 2 E4 definition)
        mg = {}
        for det, fam in DETS.items():
            vals = []
            for b in bank:
                others = [cyc for o in bank if o != b for cyc in v2.cycles(clean[pl][fam][o])]
                for cyc in v2.cycles(clean[pl][fam][b]):
                    if fam == "D":
                        d = np.min([1.0 - np.einsum("thwd,thwd->thw", cyc, o_, optimize=True) for o_ in others], 0)
                    else:
                        d = np.min([np.abs(cyc - o_) for o_ in others], 0)
                    vals.append(d.ravel())
            nn = np.concatenate(vals)
            tau = banksets[(pl, "clean", det)]["taus"]["ENV"]
            mg[det] = {"healthy_nn_median": float(np.median(nn)), "healthy_nn_p99": float(np.percentile(nn, 99)),
                       "tau_median": float(np.median(tau)), "tau_min": float(tau.min())}
        margins[pl] = mg
        log(f"margins {pl}: {mg}")
    res["margins"] = margins

    # device drift readout (raw clean CUDA vs MPS)
    if clean.get("raw") is not None:
        dd = []
        for c in bank + TESTS:
            dd.append((1 - np.sum(clean["raw"]["D"][c] * mps_tok[c], -1)).ravel())
        dd = np.concatenate(dd)
        res["device_drift_cuda_vs_mps"] = {"mean_1mcos": float(dd.mean()), "median_1mcos": float(np.median(dd)),
                                           "p99": float(np.percentile(dd, 99)), "max": float(dd.max())}
        log(f"device drift {res['device_drift_cuda_vs_mps']}")

    def run_condition(pl, bankkey, items_by_seed, do_drift=True):
        """items_by_seed: list over seeds of list over TESTS of pert. Returns per det: per rule boot stats + drift."""
        out = {}
        FA = {det: {r: np.zeros((len(items_by_seed), 6), int) for r in RULES} for det in DETS}
        H = {det: {r: np.zeros((len(items_by_seed), 6), int) for r in RULES} for det in DETS}
        S = {det: np.zeros((len(items_by_seed), 6, 64)) for det in DETS}
        drift = {det: [] for det in DETS}
        dfrac = {det: [] for det in DETS}
        dclip = {det: np.zeros((len(items_by_seed), 6)) for det in DETS}
        for si, perts in enumerate(items_by_seed):
            F = feats_for(pl, [(t, t, p) for t, p in zip(TESTS, perts)])
            if F is None:
                return None
            for det, fam in DETS.items():
                bs = banksets[(pl, bankkey, det)]
                tclean = banksets[(pl, "clean", det)]["taus"]["ENV"]
                for ci, t in enumerate(TESTS):
                    o, s = score_clip(det, bs, F[fam][t], labs[t])
                    for r in RULES:
                        H[det][r][si, ci], FA[det][r][si, ci] = o[r]
                    S[det][si, ci] = s
                    if do_drift:
                        if fam == "D":
                            d = 1.0 - np.sum(F["D"][t] * clean[pl]["D"][t], -1)
                        else:
                            d = np.abs(F[fam][t] - clean[pl][fam][t])
                        drift[det].append(d.ravel().astype(np.float32))
                        dclip[det][si, ci] = float(d.mean())
                        dfrac[det].append((v1.blocks(d) > np.concatenate([tclean, tclean], 0)).ravel())
            del F
        for det in DETS:
            out[det] = {"rules": {}}
            for r in RULES:
                st = boot_stats(FA[det][r], H[det][r], S[det], rng)
                if r != "ENV":
                    for k in ("auroc", "auroc_ci95"):
                        st.pop(k)
                out[det]["rules"][r] = st
            if do_drift:
                dv = np.concatenate(drift[det])
                mg = margins[pl][det]
                out[det]["drift"] = {"mean": float(dv.mean()), "median": float(np.median(dv)),
                                     "p95": float(np.percentile(dv, 95)),
                                     "median_over_healthy_nn_median": float(np.median(dv) / mg["healthy_nn_median"]),
                                     "median_over_tau_median": float(np.median(dv) / mg["tau_median"]),
                                     "frac_blocks_gt_ENVtau": float(np.concatenate(dfrac[det]).mean()),
                                     "mean_per_clip_ci95": [float(x) for x in np.percentile(
                                         [dclip[det][rng.integers(0, dclip[det].shape[0], dclip[det].shape[0])][:, rng.integers(0, 6, 6)].mean()
                                          for _ in range(BOOT_B)], [2.5, 97.5])]}
        return out

    R = {}  # R[pl][bankkey][cond] = result

    def put(pl, bk, cond, val):
        R.setdefault(pl, {}).setdefault(bk, {})[cond] = val if val is not None else "NOT RUN (missing encodes)"

    def checkpoint():
        res["results"] = R
        res["runtime_s"] = round(time.time() - t0, 1)
        jdump(OUT / "results.json", res)

    # ---- raw, clean bank: all test conditions
    if clean.get("raw") is not None:
        put("raw", "clean", "clean", run_condition("raw", "clean", [[("clean",)] * 6]))
        for lv in DET_LEVELS:
            put("raw", "clean", pid(lv), run_condition("raw", "clean", [[lv] * 6]))
            log(f"raw/clean {pid(lv)} done")
        put("raw", "clean", "r8noise_6_71", run_condition("raw", "clean", [[("r8noise", 6, 71)] * 6]))
        for s in SIGMAS:
            put("raw", "clean", f"noise_{s}", run_condition("raw", "clean", [[("noise", s, sd)] * 6 for sd in TEST_SEEDS]))
            log(f"raw/clean noise_{s} done")
        checkpoint()
        # ---- matched banks
        for s in SIGMAS:
            for bk, rs in (("R1", [0]), ("R4", BANK_R)):
                items = [(f"{b}#r{r}", b, ("bnoise", s, r)) for b in bank for r in rs]
                Fb = feats_for("raw", items)
                if Fb is None:
                    put("raw", f"matched_{bk}", f"noise_{s}", None)
                    continue
                src = {k: k.split("#")[0] for k, _, _ in items}
                ts = time.time()
                for det, fam in DETS.items():
                    banksets[("raw", f"m_noise_{s}_{bk}", det)] = make_bankset(det, Fb[fam], [k for k, _, _ in items], src)
                del Fb
                v = run_condition("raw", f"m_noise_{s}_{bk}", [[("noise", s, sd)] * 6 for sd in TEST_SEEDS], do_drift=False)
                for det in DETS:
                    if v:
                        v[det]["tau_ENV_median"] = float(np.median(banksets[("raw", f"m_noise_{s}_{bk}", det)]["taus"]["ENV"]))
                put("raw", f"matched_{bk}", f"noise_{s}", v)
                for det in DETS:  # free memory (attempt 3 was OOM-killed holding every matched bank)
                    del banksets[("raw", f"m_noise_{s}_{bk}", det)]
                log(f"matched {bk} noise_{s} done ({time.time() - ts:.0f}s)")
                checkpoint()
        for lv in DET_LEVELS:
            items = [(b, b, lv) for b in bank]
            Fb = feats_for("raw", items)
            if Fb is None:
                put("raw", "matched_R1", pid(lv), None)
                continue
            for det, fam in DETS.items():
                banksets[("raw", f"m_{pid(lv)}", det)] = make_bankset(det, Fb[fam], bank, {b: b for b in bank})
            del Fb
            v = run_condition("raw", f"m_{pid(lv)}", [[lv] * 6], do_drift=False)
            put("raw", "matched_R1", pid(lv), v)
            for det in DETS:
                del banksets[("raw", f"m_{pid(lv)}", det)]
            log(f"matched R1 {pid(lv)} done")
        checkpoint()
    # ---- blur pipelines
    for pl in BLURS:
        if clean.get(pl) is None:
            put(pl, "clean", "ALL", None)
            continue
        put(pl, "clean", "clean", run_condition(pl, "clean", [[("clean",)] * 6]))
        for s in BLUR_SIGMAS:
            put(pl, "clean", f"noise_{s}", run_condition(pl, "clean", [[("noise", s, sd)] * 6 for sd in TEST_SEEDS]))
        for qq in BLUR_JPEG:
            put(pl, "clean", f"jpeg_{qq}", run_condition(pl, "clean", [[("jpeg", qq)] * 6]))
        log(f"{pl} done")
        checkpoint()
    # ---- token temporal averaging
    for pl in TAVG:
        if clean.get(pl) is None:
            put(pl, "clean", "ALL", None)
            continue
        put(pl, "clean", "clean", run_condition(pl, "clean", [[("clean",)] * 6]))
        for s in SIGMAS:
            put(pl, "clean", f"noise_{s}", run_condition(pl, "clean", [[("noise", s, sd)] * 6 for sd in TEST_SEEDS]))
        for qq in (90, 70, 50, 30):
            put(pl, "clean", f"jpeg_{qq}", run_condition(pl, "clean", [[("jpeg", qq)] * 6]))
        log(f"{pl} done")
        checkpoint()

    # ================================================================ decisions (pre-declared)
    def g(pl, bk, cond):
        v = R.get(pl, {}).get(bk, {}).get(cond)
        return v if isinstance(v, dict) else None

    dec = {}
    for pl in PIPES:
        b0 = g(pl, "clean", "clean")
        if b0 is None:
            continue
        dec[pl] = {}
        for det in DETS:
            FA0 = b0[det]["rules"]["ENV"]["sum_FA"]
            dec[pl][det] = {"FA0": FA0, "FA_lim": max(1.5 * FA0, FA0 + 10),
                            "faults0": b0[det]["rules"]["ENV"]["faults_found"], "AUROC0": b0[det]["rules"]["ENV"]["auroc"]}
    raw0 = dec.get("raw", {})
    conds = [pid(lv) for lv in DET_LEVELS] + [f"noise_{s}" for s in SIGMAS]
    decisions = {}
    for det in DETS:
        if det not in raw0:
            continue
        d0 = raw0[det]
        decisions[det] = {}
        for cond in conds:
            c = g("raw", "clean", cond)
            if c is None:
                decisions[det][cond] = {"label": "NOT RUN"}
                continue
            fa_c = c[det]["rules"]["ENV"]["sum_FA"]
            row = {"FA_clean_bank": fa_c, "FA_clean_bank_ci95": c[det]["rules"]["ENV"]["sum_FA_ci95"],
                   "auroc_clean_bank": c[det]["rules"]["ENV"]["auroc"], "degraded": fa_c > d0["FA_lim"]}
            if "drift" in c[det]:
                row["drift_over_healthy_nn"] = c[det]["drift"]["median_over_healthy_nn_median"]
                row["drift_exceeds_healthy_margin"] = row["drift_over_healthy_nn"] >= 1
            mk = "matched_R4" if cond.startswith("noise") else "matched_R1"
            m = g("raw", mk, cond)
            if not row["degraded"]:
                row["label"] = "not degraded"
            elif m is None:
                row["label"] = "NOT TESTED (matched bank missing)"
            else:
                md = m[det]["rules"]
                row.update({"matched_bank": mk, "matched": {r: {"FA": md[r]["sum_FA"], "FA_ci95": md[r]["sum_FA_ci95"],
                                                                 "faults": md[r]["faults_found"]} for r in RULES},
                            "matched_auroc": md["ENV"]["auroc"], "matched_auroc_ci95": md["ENV"]["auroc_ci95"]})

                def restores(r):
                    return (md[r]["sum_FA_ci95"][1] <= d0["FA_lim"] and md[r]["faults_found"] >= d0["faults0"] - 1
                            and md["ENV"]["auroc"] >= d0["AUROC0"] - 0.05)
                if restores("ENV"):
                    row["label"] = "THRESHOLD problem"
                elif any(restores(r) for r in RULES[1:]):
                    row["label"] = "THRESHOLD (secondary rule): " + ",".join(r for r in RULES[1:] if restores(r))
                elif all(md[r]["sum_FA_ci95"][0] > d0["FA_lim"] for r in RULES) or md["ENV"]["auroc_ci95"][1] < d0["AUROC0"] - 0.05:
                    row["label"] = "FEATURE problem"
                else:
                    row["label"] = "MIXED / UNRESOLVED"
            # pre-filter rescue
            resc = {}
            for pl in ("blur1", "blur2", "tavg2", "tavg4"):
                pc = g(pl, "clean", cond)
                if pc is None or pl not in dec:
                    continue
                st = pc[det]["rules"]["ENV"]
                resc[pl] = {"FA": st["sum_FA"], "FA_ci95": st["sum_FA_ci95"], "faults": st["faults_found"],
                            "pipeline_FA0": dec[pl][det]["FA0"], "pipeline_faults0": dec[pl][det]["faults0"],
                            "rescues": bool(row["degraded"] and st["sum_FA_ci95"][1] <= dec[pl][det]["FA_lim"]
                                            and st["faults_found"] >= d0["faults0"] - 1)}
            row["prefilters"] = resc
            decisions[det][cond] = row
    spec = {}
    for cond in conds:
        try:
            dD = decisions["D_patchKNN"][cond]["degraded"]
            spec[cond] = bool(dD and not decisions["pixel_patchKNN"][cond]["degraded"]
                              and not decisions["illum_pixel_patchKNN"][cond]["degraded"])
        except KeyError:
            spec[cond] = None
    ladders = {"darker": ["gain_0.95", "gain_0.9", "gain_0.8"], "brighter": ["gain_1.05", "gain_1.1", "gain_1.25"],
               "shift": ["shift_1", "shift_2", "shift_4", "shift_8"], "jpeg": ["jpeg_90", "jpeg_70", "jpeg_50", "jpeg_30"],
               "noise": [f"noise_{s}" for s in SIGMAS]}
    bp = {}
    for lad, Ls in ladders.items():
        bp[lad] = {}
        for det in DETS:
            fas = [(Lv, g("raw", "clean", Lv)[det]["rules"]["ENV"]["sum_FA"]) for Lv in Ls if g("raw", "clean", Lv)]
            br = next((x for x in fas if x[1] > 150), None)
            bp[lad][det] = {"ladder_FA": fas, "breaking_point": br[0] if br else "not reached"}
    res.update({"baselines": dec, "decisions": decisions, "vjepa_specific": spec, "breaking_points": bp,
                "token_manifest_sha256": sha(manifest_p) if manifest_p.exists() else None, "inputs_sha256": inputs})
    checkpoint()
    for det in decisions:
        for cond, row in decisions[det].items():
            log(f"DECISION {det:21s} {cond:10s} {row.get('label')} FA_c {row.get('FA_clean_bank')} "
                f"prefilter {({k: v['rescues'] for k, v in row.get('prefilters', {}).items()})}")
    log(f"V-JEPA-specific: {spec}")
    log(f"breaking points: {json.dumps(bp)}")
    log(f"wrote results.json sha256 {sha(OUT / 'results.json')} runtime {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
