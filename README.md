# V-JEPA and pixel patch-kNN on a synthetic lift-door task

Code and study artifacts behind the paper *"Frozen V-JEPA and Pixel Patch-kNN on Synthetic Equipment Faults: A Fresh-Seed Evaluation at Calibration-Fixed Operating Points"* and the accompanying blog post. The paper and the blog are published separately.

## Folder structure

| Folder | Contents |
|--------|----------|
| `experiments/fresh_seed/` | The main study: pre-specified design and amendments, the scorer (`score.py`), the jittered scene generator (`gen/`) and the frozen results |
| `experiments/lift_lobby_reanalysis/` | The separate lift-lobby reanalysis: design, runner and frozen results |
| `experiments/capture_perturbation/` | The capture-perturbation study: design, runner and frozen summary results |
| `scene_generator/` | The Blender lift-door scene generator and its helper scripts |
| `agentic_pipeline/` | The AWS demo from the blog: V-JEPA embedding service, kNN gate, Opus reasoner coordinator, tickets, CDK infrastructure and tests |
| `cost_joiner/` | The per-invocation cost calculator and its summary tables |

## What's not included
- **V-JEPA 2.1 weights and Meta's V-JEPA code.** Download them from https://github.com/facebookresearch/vjepa2 under Meta's licence.
- **Rendered clips.** Regenerate them with `scene_generator/` and `experiments/fresh_seed/gen/`; this needs Blender.
- **Token tensors and score caches.** These are too large.
- **Account-specific outputs and raw billing data.**

## Results status
- **Fresh-seed study and lift-lobby reanalysis:** both were replayed from saved scores with recorded tolerances.
- **Capture-perturbation study:** `results.json` holds the frozen summary values. They were not independently replayed.

## AWS pipeline setup
`agentic_pipeline/` deploys to your own AWS account. Placeholders mark account-specific values:
- `<ACCOUNT_ID>` for the AWS account;
- `<DATA_BUCKET>` for the S3 data bucket;
- `<LOCAL_PATH>/` for local file roots.

Replace them before you deploy. The pipeline also needs:
- Bedrock access to the reasoner model;
- the V-JEPA checkpoint, whose path is read from an environment variable;
- your own credentials, none of which are included here.

## Licence
- **Code** (`*.py`, `*.sh`, `*.yml` and other program files): [MIT License](LICENSE). MIT is GPL-compatible, which matters for the scene generator because it uses Blender's Python API.
- **Study documents and frozen results:** [CC BY 4.0](LICENSE-CC-BY-4.0.md).
- **Not covered:**
  - the V-JEPA 2.1 weights and Meta's V-JEPA code (Meta's terms);
  - Blender, which is a separate GPL-licensed application.
