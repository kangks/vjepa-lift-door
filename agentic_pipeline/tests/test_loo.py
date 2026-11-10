import numpy as np

from lda_core.design import BANK
from lda_core.detector import build_envelope, detect, unit
from lda_core.frames import frames_sha16, load_png_frames


def _tokens(name, frames, cache_dir):
    return unit(np.load(cache_dir / f"{name}_{frames_sha16(frames)}.npy").astype(np.float32))


def test_bank_leave_one_out_never_alerts(eval_dir, cases):
    cache = eval_dir / "results_v1/token_cache"
    frames = {name: load_png_frames(cases[name]) for name in BANK}
    bank = {name: _tokens(name, frames[name], cache) for name in BANK}
    envelope = build_envelope(bank)
    for name in BANK:
        result = detect(bank[name], bank, envelope, exclude=[name])
        assert result["alert"] is False
        assert result["bank_mode"] == "loo"
        assert result["excluded_bank_clips"] == [name]
        assert float(result["excess"].max()) <= -1e-4 + 1e-7
