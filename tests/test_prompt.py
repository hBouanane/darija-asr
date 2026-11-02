from darija_asr.merge.judge import custom_id
from darija_asr.merge.prompt import build_system_prompt, build_user_message


def test_user_message_is_deterministic_and_anonymous():
    cands = {"reference": "واش نتا", "cohere": "واش انت", "voxtral": ""}
    m1 = build_user_message("@chan_1_2", cands)
    assert m1 == build_user_message("@chan_1_2", cands)
    assert "reference" not in m1 and "cohere" not in m1
    assert m1.count("<candidate") == 2  # empty source skipped


def test_reveal_sources_labels_by_name():
    m = build_user_message("x", {"cohere": "a", "voxtral": "b"}, reveal_sources=True)
    assert 'id="cohere"' in m and 'id="voxtral"' in m


def test_system_prompt_includes_style_guide():
    prompt = build_system_prompt("resources/style_guide.md")
    assert "<style_guide>" in prompt and "ڭ" in prompt


def test_custom_id_is_valid_for_batches():
    import re

    cid = custom_id("@eruine4218_23_655")
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cid)
