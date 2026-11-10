import hashlib
import json

from lda_core.design import FROZEN_DESIGN, FROZEN_V2, design_sha256


def test_design_fingerprint_matches_frozen_reference(eval_dir):
    ref = eval_dir / "results_v2/frozen_design_v2.json"
    raw = ref.read_bytes()
    assert hashlib.sha256(raw).hexdigest().startswith("7835561bae3c77ae")
    encoded = json.dumps({"v1": FROZEN_DESIGN, "v2": FROZEN_V2}, indent=1, sort_keys=True).encode()
    assert hashlib.sha256(encoded).hexdigest() == hashlib.sha256(raw).hexdigest()
    assert design_sha256().startswith("7835561bae3c77ae")
