from services.coordinator.bank_build import build_bank
from lda_core.design import BANK


def test_bank_build_embeds_all_five_clips_and_invokes_knn_once():
    embed_calls = []
    knn_calls = []

    def embed(run_id, clip_key, headers):
        embed_calls.append((run_id, clip_key))
        name = clip_key.rsplit("/", 1)[-1].removesuffix(".mp4")
        return {
            "tokens_key": f"runs/{run_id}/tokens.npy",
            "frames_key": f"runs/{run_id}/frames.npy",
            "mp4_sha256": name + "-sha",
        }

    def knn(event):
        knn_calls.append(event)
        return {"bank_prefix": "bank/v2/test", "design_sha256": "test"}

    result = build_bank("test-id", embed_call=embed, knn_call=knn)
    assert [call[1].rsplit("/", 2)[-2] for call in embed_calls] == BANK
    assert len(embed_calls) == 5
    assert len(knn_calls) == 1
    assert knn_calls[0]["mode"] == "build_bank"
    assert list(knn_calls[0]["clips"]) == BANK
    assert result["bank_prefix"] == "bank/v2/test"
