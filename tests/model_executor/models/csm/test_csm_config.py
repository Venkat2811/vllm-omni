# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU correctness tests for CSM-1B config hoisting + backbone LlamaConfig synthesis.

Covers the fragile spots:
  * ``CsmConfig`` reads the real checkpoint's FLAT top-level backbone fields
    (falling back to the CSM-1B constants only when absent) and honors an
    explicitly passed nested ``backbone_config``, while keeping depth / codec
    params nested. Real config.json values must always beat the constants.
  * ``build_backbone_llama_config`` recovers ``rope_theta`` across the
    transformers >= 5.12 RoPE migration (where ``rope_theta`` is folded INTO
    ``rope_scaling`` and no longer a top-level attribute) and passes the
    checkpoint's rope-scaling dict through verbatim (transformers' rope
    validation only warns -- it never raises -- so nothing may be clamped).
"""

import pytest
from transformers import LlamaConfig

from vllm_omni.model_executor.models.csm.configuration_csm import (
    CsmConfig,
    build_backbone_llama_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_default_config_hoists_backbone_facts():
    cfg = CsmConfig()
    assert cfg.model_type == "csm"
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 16
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 64
    assert cfg.intermediate_size == 8192
    # cb0 logits surface == per-codebook vocab (2051: 2048 codes + 3 reserved).
    assert cfg.vocab_size == 2051
    assert cfg.num_codebooks == 32
    assert cfg.reserved_codebook_ids == (2048, 2049, 2050)
    # vLLM requires this to be absent / None.
    assert cfg.speculative_config is None
    # get_text_config returns self so vLLM reads the hoisted backbone fields.
    assert cfg.get_text_config() is cfg


def test_default_config_keeps_depth_and_codec_facts_nested():
    cfg = CsmConfig()
    assert cfg.depth_hidden_size == 1024
    assert cfg.depth_num_hidden_layers == 4
    assert cfg.depth_head_dim == 128
    assert cfg.depth_num_positions == 33
    assert cfg.codec_sample_rate == 24000
    assert cfg.codec_samples_per_frame == 1920


def test_backbone_llama_config_is_a_real_llama_config():
    llama = build_backbone_llama_config(CsmConfig())
    assert isinstance(llama, LlamaConfig)
    assert llama.hidden_size == 2048
    assert llama.num_hidden_layers == 16
    assert llama.num_attention_heads == 32
    assert llama.num_key_value_heads == 8
    assert llama.head_dim == 64
    assert llama.vocab_size == 2051
    # The real checkpoint ships tie_word_embeddings: false, and the native
    # transformers CsmConfig hard-rejects True -- the default must match.
    assert llama.tie_word_embeddings is False


def test_rope_theta_falls_back_to_csm_default_when_absent():
    # Default config carries no rope_theta inside rope_scaling -> recover from the
    # top-level attribute / CSM-1B default (500000.0) and fold it back in. Under
    # transformers >= 5.12 LlamaConfig has NO top-level ``rope_theta`` attribute,
    # so vLLM's get_rope reads it from rope_scaling/rope_parameters -- that is
    # exactly where build_backbone_llama_config must leave it.
    llama = build_backbone_llama_config(CsmConfig())
    assert llama.rope_scaling["rope_theta"] == 500000.0
    assert llama.rope_parameters["rope_theta"] == 500000.0
    assert not hasattr(llama, "rope_theta")  # confirms the migration shape


def test_rope_theta_recovered_from_nested_rope_scaling():
    # transformers >= 5.12 shape: rope_theta lives INSIDE rope_scaling, no
    # top-level attribute. build_backbone_llama_config must read it from there
    # and keep it there (not drop it back onto a dead top-level attribute).
    cfg = CsmConfig(
        backbone_config={
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 0.125,
                "high_freq_factor": 0.5,
                "original_max_position_embeddings": 1024,
                "rope_theta": 123456.0,
            }
        }
    )
    llama = build_backbone_llama_config(cfg)
    assert llama.rope_scaling["rope_theta"] == 123456.0
    assert llama.rope_parameters["rope_theta"] == 123456.0


def test_default_rope_scaling_matches_checkpoint_values():
    # The defaults are byte-identical to sesame/csm-1b config.json (the
    # "public model facts" contract in the module docstring): the Llama-3.x
    # curve jointly scaled by 1/8, NOT the Llama-3.2 text-model values.
    cfg = CsmConfig()
    assert cfg.rope_scaling["rope_type"] == "llama3"
    assert cfg.rope_scaling["factor"] == 32.0
    assert cfg.rope_scaling["low_freq_factor"] == 0.125
    assert cfg.rope_scaling["high_freq_factor"] == 0.5
    assert cfg.rope_scaling["original_max_position_embeddings"] == 1024


def test_original_max_position_passes_through_unclamped():
    # The checkpoint's real value (1024) must survive verbatim. An earlier
    # revision clamped it below max_position_embeddings citing a transformers
    # validation requirement that does not exist (validation only WARNS on
    # original >= max_position), silently altering the rope curve. A value
    # >= max_position_embeddings must also pass through untouched.
    llama = build_backbone_llama_config(CsmConfig())
    assert llama.rope_scaling["original_max_position_embeddings"] == 1024

    big = CsmConfig(
        backbone_config={"rope_scaling": {**dict(CsmConfig().rope_scaling), "original_max_position_embeddings": 8192}}
    )
    llama_big = build_backbone_llama_config(big)
    assert llama_big.rope_scaling["original_max_position_embeddings"] == 8192


def test_explicit_backbone_overrides_are_honored():
    cfg = CsmConfig(backbone_config={"hidden_size": 1536, "num_hidden_layers": 12, "vocab_size": 4096})
    assert cfg.hidden_size == 1536
    assert cfg.num_hidden_layers == 12
    assert cfg.vocab_size == 4096
    llama = build_backbone_llama_config(cfg)
    assert llama.hidden_size == 1536
    assert llama.num_hidden_layers == 12
    assert llama.vocab_size == 4096


def test_from_pretrained_on_real_transformers_config_layout(tmp_path):
    """Regression: ``from_pretrained`` on the real checkpoint layout.

    The transformers-format CSM-1B config carries ``rope_scaling`` /
    ``rope_theta`` / ``max_position_embeddings`` at the TOP level with no
    nested ``backbone_config``. transformers >= 5.12 standardizes rope
    parameters inside ``PretrainedConfig.__init__`` and reads
    ``max_position_embeddings`` during that pass, so assigning those fields
    only after ``super().__init__`` raised AttributeError on this path."""
    import json

    real_layout = {
        "model_type": "csm",
        "architectures": ["CsmForConditionalGeneration"],
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "intermediate_size": 8192,
        "max_position_embeddings": 2048,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 0.5,
            "low_freq_factor": 0.125,
            "original_max_position_embeddings": 1024,
            "rope_type": "llama3",
        },
        "vocab_size": 2051,
        "num_codebooks": 32,
        "tie_word_embeddings": False,
        "depth_decoder_config": {"hidden_size": 1024, "num_hidden_layers": 4},
        "codec_config": {"sample_rate": 24000, "frame_rate": 12.5},
    }
    (tmp_path / "config.json").write_text(json.dumps(real_layout))

    cfg = CsmConfig.from_pretrained(tmp_path)  # raised AttributeError pre-fix

    assert cfg.max_position_embeddings == 2048
    rope = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
    assert rope is not None
    assert float(rope["factor"]) == 32.0
    assert float(rope["low_freq_factor"]) == 0.125
    assert float(rope["high_freq_factor"]) == 0.5
    assert int(rope["original_max_position_embeddings"]) == 1024
    # The flat top-level backbone fields survive (they coincide with the
    # CSM-1B constants here; test_from_pretrained_flat_variant_layout below
    # proves the checkpoint values, not the constants, are what wins).
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 16
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 64
    assert cfg.intermediate_size == 8192
    assert cfg.vocab_size == 2051
    # tie_word_embeddings: false is the real checkpoint value (the native
    # transformers CsmConfig hard-rejects True).
    assert cfg.tie_word_embeddings is False
    # Nested sections still land where the depth loop / Mimi stage read them.
    assert cfg.depth_hidden_size == 1024
    assert cfg.codec_sample_rate == 24000
    # And the recovered config still synthesizes a valid backbone LlamaConfig.
    llama = build_backbone_llama_config(cfg)
    assert llama.max_position_embeddings == 2048
    assert llama.tie_word_embeddings is False


def test_from_pretrained_flat_variant_layout_beats_the_constants(tmp_path):
    """Regression (review finding): a CSM-family checkpoint whose backbone
    differs from CSM-1B (e.g. a distilled or vocab-extended variant) uses the
    real transformers flat layout with NO nested ``backbone_config``. Its
    values must load verbatim -- pre-fix, every non-rope field was silently
    clobbered with the hardcoded CSM-1B constants after ``super().__init__``.
    Every value here is deliberately different from the module constants."""
    import json

    variant = {
        "model_type": "csm",
        "hidden_size": 1024,
        "num_hidden_layers": 8,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 96,
        "intermediate_size": 4096,
        "max_position_embeddings": 4096,
        "vocab_size": 4096,
        "tie_word_embeddings": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(variant))

    cfg = CsmConfig.from_pretrained(tmp_path)

    assert cfg.hidden_size == 1024
    assert cfg.num_hidden_layers == 8
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 4
    assert cfg.head_dim == 96
    assert cfg.intermediate_size == 4096
    assert cfg.max_position_embeddings == 4096
    assert cfg.vocab_size == 4096
    # The synthesized backbone LlamaConfig is variant-shaped, not 1B-shaped.
    llama = build_backbone_llama_config(cfg)
    assert llama.hidden_size == 1024
    assert llama.num_hidden_layers == 8
    assert llama.num_attention_heads == 16
    assert llama.num_key_value_heads == 4
    assert llama.head_dim == 96
    assert llama.intermediate_size == 4096
    assert llama.vocab_size == 4096


def test_save_pretrained_round_trip_preserves_hoisted_fields(tmp_path):
    """``save_pretrained`` -> ``from_pretrained`` must be lossless for every
    hoisted backbone field and ``tie_word_embeddings`` (the exported-API
    contract for this config class)."""
    fields = {
        "hidden_size": 1024,
        "num_hidden_layers": 8,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 96,
        "intermediate_size": 4096,
        "max_position_embeddings": 4096,
        "vocab_size": 4096,
    }
    cfg = CsmConfig(**fields, tie_word_embeddings=False)
    cfg.save_pretrained(tmp_path)
    reloaded = CsmConfig.from_pretrained(tmp_path)

    for name, value in fields.items():
        assert getattr(reloaded, name) == value, name
    assert reloaded.tie_word_embeddings is False
    assert reloaded.rope_scaling["low_freq_factor"] == cfg.rope_scaling["low_freq_factor"]
    assert (
        reloaded.rope_scaling["original_max_position_embeddings"]
        == (cfg.rope_scaling["original_max_position_embeddings"])
    )
