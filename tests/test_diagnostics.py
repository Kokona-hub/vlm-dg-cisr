from fs_pairgrpo.diagnostics import _target_matches


def test_regex_lora_targets_exclude_patch_embedding_projection():
    names = [
        "model.visual.patch_embed.proj",
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.attn.proj",
        "model.language_model.layers.0.self_attn.q_proj",
    ]
    pattern = (
        r".*(?:visual\.blocks\.[0-9]+\.attn\.(?:qkv|proj)|"
        r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))$"
    )
    matches = _target_matches(names, pattern)[pattern]

    assert "model.visual.patch_embed.proj" not in matches
    assert "model.visual.blocks.0.attn.qkv" in matches
    assert "model.visual.blocks.0.attn.proj" in matches
    assert "model.language_model.layers.0.self_attn.q_proj" in matches
