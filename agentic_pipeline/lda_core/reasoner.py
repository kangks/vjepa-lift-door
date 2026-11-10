# Ported from .claude/worktrees/vjepa-talk-draft/workshop/lift-door/eval_suite/reasoner_opus.py (sha256 b8fa3dd3d928ec80) on 2026-11-06; logic unchanged.
# Ported from .claude/worktrees/vjepa-talk-draft/workshop/lift-door/eval_suite/reasoner_opus_v2.py (sha256 b2406cf7eba08f0f) on 2026-11-06; logic unchanged.
from __future__ import annotations

import base64
import hashlib
import io
import json

import numpy as np
from jsonschema import Draft202012Validator
from PIL import Image

MODEL = "anthropic.claude-opus-5-5"
PRICE_IN, PRICE_OUT = 4.4e-6, 22.0e-6
SYSTEM = (
    "You are a lift-maintenance triage assistant reviewing CCTV of a lift lobby. A learned novelty "
    "detector has flagged a time window as unlike the site's normal door cycles. You get the flagged "
    "frames at 4 fps, the same times from a normal cycle, the detector's anomaly map (bright = most "
    "unlike normal), and amplified difference images (bright = pixels that differ from the normal "
    "cycle). Novelty is not the same as a fault: decide from what is VISIBLE whether the flagged "
    "behaviour is the lift working as designed (EXPECTED_BEHAVIOR), a likely equipment problem "
    "(SUSPECTED_MALFUNCTION), or not decidable (UNCERTAIN). Compare the two door leaves carefully "
    "(positions, symmetry, timing) and the cabin light level frame by frame. Only answer "
    "EXPECTED_BEHAVIOR if you can point to a visible, benign cause of the novelty. Cite only frames "
    "you can see."
)
SITE_CONTEXT = {
    "door_safety_design": ("Lift doors automatically stop and reopen if an object or person enters "
                           "the door path while they are closing. This is the designed safety response, "
                           "not a fault."),
    "normal_cycle": "Both door leaves open and close together, symmetrically, at a steady speed.",
    "lighting": "The lift cabin light should be steady while the doors are open.",
}
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["classification", "observed_events", "obstruction_visible", "door_behaviour",
                 "cabin_light_behaviour", "evidence_frame_ids", "rationale", "recommended_action"],
    "properties": {
        "classification": {"type": "string", "enum": ["EXPECTED_BEHAVIOR", "SUSPECTED_MALFUNCTION", "UNCERTAIN"]},
        "observed_events": {"type": "array", "items": {"type": "string"}},
        "obstruction_visible": {"type": "boolean"},
        "door_behaviour": {"type": "string"},
        "cabin_light_behaviour": {"type": "string"},
        "evidence_frame_ids": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
        "recommended_action": {"type": "string", "enum": ["NO_ACTION_LOG_ONLY", "INSPECT_DOOR_MECHANISM",
                                                          "INSPECT_LIGHTING", "HUMAN_REVIEW"]},
    },
}


def img_block(rgb: np.ndarray) -> dict:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(buf.getvalue()).decode()}}


def build_request(
    frames_cur: np.ndarray,
    frames_ref: np.ndarray,
    evidence: dict,
    window_s: list[float] | tuple[float, float],
) -> tuple[dict, int, list[str]]:
    t0, t1 = window_s
    ks = evidence["ks"]
    peak_k = evidence["peak_k"]
    t0 = float(t0)
    t1 = float(t1)
    ctx = {
        "detector": "V-JEPA 2.1 dense embeddings + nearest-neighbour novelty detector",
        "flagged_window_s": [t0, t1], "site_context": SITE_CONTEXT,
        "inputs": "CUR_xx flagged clip (4 fps); REF_xx normal cycle same times; MAP detector map; "
                  "DIFF_xx amplified |CUR-REF|",
    }
    content = [
        {"type": "text", "text": json.dumps(ctx)},
        {"type": "text", "text": f"MAP (detector anomaly map, max over window, on frame t={peak_k / 4:.2f}s)"},
        img_block(np.asarray(Image.open(io.BytesIO(evidence["map_png"])).convert("RGB"))),
    ]
    for i, (k, png) in enumerate(evidence["diffs"]):
        content.extend([
            {"type": "text", "text": f"DIFF_{i:02d} t={k / 4:.2f}s"},
            img_block(np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))),
        ])
    ids = []
    for i, k in enumerate(ks[::2]):
        content.extend([
            {"type": "text", "text": f"REF_{i:02d} t={k / 4:.2f}s"},
            img_block(frames_ref[k]),
        ])
    for i, k in enumerate(ks):
        frame_id = f"CUR_{i:02d}"
        ids.append(frame_id)
        content.extend([
            {"type": "text", "text": f"{frame_id} t={k / 4:.2f}s"},
            img_block(frames_cur[k]),
        ])
    payload = {
        "model": MODEL, "system": SYSTEM, "max_tokens": 4000,
        "messages": [{"role": "user", "content": content}],
        "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "high"},
    }
    return payload, sum(1 for item in content if item["type"] == "image"), ids


def fingerprints() -> dict[str, str]:
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(canonical(SCHEMA)).hexdigest(),
        "site_context_sha256": hashlib.sha256(canonical(SITE_CONTEXT)).hexdigest(),
    }


def validate(output: dict) -> None:
    Draft202012Validator(SCHEMA).validate(output)


def cost_usd(usage: dict) -> float:
    return usage["input_tokens"] * PRICE_IN + usage["output_tokens"] * PRICE_OUT
