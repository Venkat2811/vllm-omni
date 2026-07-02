# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for CSM-1B in the vLLM-Omni 2-stage pipeline.

CSM-1B (``sesame/csm-1b``) is a dual-autoregressive speech model: a
Llama-style **backbone** that predicts codebook-0 (cb0) for each 80 ms audio
frame, plus a small **depth decoder** that autoregressively predicts the
remaining RVQ codebook tokens for that frame. The audio is reconstructed from
the 32-codebook frame stream by the Mimi codec (24 kHz / 12.5 Hz).

All fields below are public model facts taken from HF ``transformers``'
``CsmConfig`` and the Mimi model card. This config exposes the backbone
parameters at the top level so vLLM's machinery can read them, and provides a
:func:`build_backbone_llama_config` helper that synthesises a plain
``transformers.LlamaConfig`` describing *only* the backbone, so the backbone
can be rebuilt on vLLM-native ``LlamaDecoderLayer`` + ``PagedAttention``
(rather than wrapping the HF model).
"""

from __future__ import annotations

from transformers import LlamaConfig
from transformers.configuration_utils import PretrainedConfig

# --- Public CSM-1B backbone facts (transformers.CsmConfig) ---
_BACKBONE_HIDDEN_SIZE = 2048
_BACKBONE_NUM_LAYERS = 16
_BACKBONE_NUM_ATTENTION_HEADS = 32
_BACKBONE_NUM_KV_HEADS = 8
_BACKBONE_HEAD_DIM = 64
_BACKBONE_INTERMEDIATE_SIZE = 8192
_BACKBONE_MAX_POSITION_EMBEDDINGS = 2048
_BACKBONE_ROPE_THETA = 500000.0
# llama3 RoPE scaling, byte-identical to the sesame/csm-1b config.json. These
# are the Llama-3.x family curve parameters jointly scaled by 1/8 (original
# context 1024, not 8192); the values below are the checkpoint's own, so a
# default-constructed config matches ``from_pretrained`` on the real layout
# and is self-consistent with ``max_position_embeddings`` (1024 < 2048, so
# transformers' rope validation -- which only ever WARNS on
# original >= max_position, it never raises -- stays silent).
_BACKBONE_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 0.125,
    "high_freq_factor": 0.5,
    "original_max_position_embeddings": 1024,
}

# --- Public CSM-1B depth-decoder facts (transformers.CsmConfig) ---
_DEPTH_HIDDEN_SIZE = 1024
_DEPTH_NUM_LAYERS = 4
_DEPTH_HEAD_DIM = 128
# Depth decoder runs a 31-step inner AR loop per frame (cb1..cb31 after the
# backbone has produced cb0), i.e. 33 positions of dense static KV that reset
# every frame. See A2 design §2.2.
_DEPTH_NUM_POSITIONS = 33

# --- Public Mimi codec facts (Mimi model card) ---
_NUM_CODEBOOKS = 32
_CODEBOOK_VOCAB_SIZE = 2051  # per-codebook vocab; 2048/2049/2050 are reserved
_RESERVED_CODEBOOK_IDS = (2048, 2049, 2050)
_MIMI_SAMPLE_RATE = 24000
_MIMI_FRAME_RATE_HZ = 12.5
_MIMI_SAMPLES_PER_FRAME = 1920  # 80 ms @ 24 kHz


class CsmConfig(PretrainedConfig):
    """Config for CSM-1B (``sesame/csm-1b``) in vLLM-Omni.

    Exposes the backbone parameters at the top level (so vLLM reads them
    directly), keeps the depth-decoder + Mimi codec parameters as nested
    attributes, and registers under ``model_type = "csm"``. The real HF
    ``CsmConfig`` carries the backbone fields FLAT at the top level and nests
    only ``depth_decoder_config`` / ``codec_config``; this class reads the
    flat fields (honoring an explicitly passed nested ``backbone_config``
    first, for callers that build one programmatically) and stores the nested
    sections for the depth loop (C2) and the Mimi stage (C3).
    """

    model_type = "csm"

    def __init__(self, **kwargs):
        backbone_cfg = kwargs.pop("backbone_config", None) or {}
        if hasattr(backbone_cfg, "to_dict"):
            backbone_cfg = backbone_cfg.to_dict()
        depth_cfg = kwargs.pop("depth_decoder_config", None) or {}
        if hasattr(depth_cfg, "to_dict"):
            depth_cfg = depth_cfg.to_dict()
        codec_cfg = kwargs.pop("codec_config", None) or {}
        if hasattr(codec_cfg, "to_dict"):
            codec_cfg = codec_cfg.to_dict()

        # Real checkpoint configs (and the native transformers ``CsmConfig``)
        # carry the backbone fields FLAT at the top level. Pop each top-level
        # value BEFORE ``super().__init__`` and use it as the fallback when a
        # nested ``backbone_config`` (explicit callers only) lacks the key, so
        # real config.json values always win over the hardcoded CSM-1B
        # constants -- the constants are only the no-argument default.
        # Position/rope fields must additionally exist BEFORE
        # ``super().__init__``: transformers >= 5.12 standardizes rope
        # parameters inside ``PretrainedConfig.__init__`` and reads
        # ``max_position_embeddings`` during that pass, so assigning them
        # afterwards breaks ``from_pretrained`` on a real config.json with
        # AttributeError.
        kwargs_mpe = kwargs.pop("max_position_embeddings", _BACKBONE_MAX_POSITION_EMBEDDINGS)
        kwargs_rope_theta = kwargs.pop("rope_theta", _BACKBONE_ROPE_THETA)
        kwargs_rope_scaling = kwargs.pop("rope_scaling", dict(_BACKBONE_ROPE_SCALING))
        kwargs_hidden_size = kwargs.pop("hidden_size", _BACKBONE_HIDDEN_SIZE)
        kwargs_num_layers = kwargs.pop("num_hidden_layers", _BACKBONE_NUM_LAYERS)
        kwargs_num_heads = kwargs.pop("num_attention_heads", _BACKBONE_NUM_ATTENTION_HEADS)
        kwargs_num_kv_heads = kwargs.pop("num_key_value_heads", _BACKBONE_NUM_KV_HEADS)
        kwargs_head_dim = kwargs.pop("head_dim", _BACKBONE_HEAD_DIM)
        kwargs_intermediate = kwargs.pop("intermediate_size", _BACKBONE_INTERMEDIATE_SIZE)
        kwargs_vocab_size = kwargs.pop("vocab_size", None)
        self.max_position_embeddings = backbone_cfg.get("max_position_embeddings", kwargs_mpe)
        self.rope_theta = backbone_cfg.get("rope_theta", kwargs_rope_theta)
        self.rope_scaling = backbone_cfg.get("rope_scaling", kwargs_rope_scaling)

        super().__init__(**kwargs)

        # --- Backbone parameters (hoisted to top level for vLLM) ---
        self.hidden_size = backbone_cfg.get("hidden_size", kwargs_hidden_size)
        self.num_hidden_layers = backbone_cfg.get("num_hidden_layers", kwargs_num_layers)
        self.num_attention_heads = backbone_cfg.get("num_attention_heads", kwargs_num_heads)
        self.num_key_value_heads = backbone_cfg.get("num_key_value_heads", kwargs_num_kv_heads)
        self.head_dim = backbone_cfg.get("head_dim", kwargs_head_dim)
        self.intermediate_size = backbone_cfg.get("intermediate_size", kwargs_intermediate)
        # The real checkpoint ships ``tie_word_embeddings: false`` and the
        # native transformers ``CsmConfig`` hard-rejects True, so a config
        # saved with True is unloadable by the AutoConfig machinery the
        # pipeline itself uses. The backbone here wraps ``LlamaModel`` plus a
        # separate untied ``cb0_head`` (vocab 2051, ~8 MB), so the flag gates
        # no dead weight either way -- default False to match the checkpoint.
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", False)

        # --- Codebook / frame-embedding parameters ---
        self.num_codebooks = getattr(self, "num_codebooks", _NUM_CODEBOOKS)
        self.codebook_vocab_size = getattr(self, "codebook_vocab_size", _CODEBOOK_VOCAB_SIZE)
        # Backbone vocab is the cb0 logits surface (one codebook of audio
        # tokens). Fallback chain: nested backbone_config -> top-level
        # config.json value -> per-codebook vocab.
        if kwargs_vocab_size is None:
            kwargs_vocab_size = self.codebook_vocab_size
        self.vocab_size = backbone_cfg.get("vocab_size", kwargs_vocab_size)
        self.reserved_codebook_ids = _RESERVED_CODEBOOK_IDS

        # --- Depth decoder (nested; consumed by C2 depth loop) ---
        self.depth_hidden_size = depth_cfg.get("hidden_size", _DEPTH_HIDDEN_SIZE)
        self.depth_num_hidden_layers = depth_cfg.get("num_hidden_layers", _DEPTH_NUM_LAYERS)
        self.depth_head_dim = depth_cfg.get("head_dim", _DEPTH_HEAD_DIM)
        self.depth_num_positions = depth_cfg.get("num_positions", _DEPTH_NUM_POSITIONS)
        self.depth_decoder_config = depth_cfg

        # --- Mimi codec (nested; consumed by C3 Stage-1 decoder) ---
        self.codec_config = codec_cfg
        self.codec_sample_rate = codec_cfg.get("sample_rate", _MIMI_SAMPLE_RATE)
        self.codec_frame_rate_hz = codec_cfg.get("frame_rate", _MIMI_FRAME_RATE_HZ)
        self.codec_samples_per_frame = codec_cfg.get("samples_per_frame", _MIMI_SAMPLES_PER_FRAME)
        self.codec_pretrained_name_or_path = getattr(self, "codec_pretrained_name_or_path", "kyutai/mimi")

        # vLLM requires speculative_config to be absent or None.
        self.speculative_config = None

    def get_text_config(self, **kwargs):
        """Return self so vLLM reads our hoisted top-level backbone config."""
        return self


def build_backbone_llama_config(config: CsmConfig) -> LlamaConfig:
    """Synthesise a plain ``LlamaConfig`` describing only the CSM backbone.

    vLLM's native ``LlamaModel`` / ``LlamaDecoderLayer`` consume a standard
    ``LlamaConfig``. CSM's HF config nests its backbone fields; this builds the
    flat ``LlamaConfig`` so the backbone can be rebuilt on vLLM-native layers +
    PagedAttention (A2 §2.1) rather than wrapping the HF model.

    ``vocab_size`` is the cb0 audio-token surface; the depth decoder (C2) and
    Mimi stage (C3) handle the remaining 31 codebooks and waveform synthesis.

    Config-source note: ``config`` here is the **native** ``transformers``
    ``CsmConfig`` (transformers >= 5.12 ships first-class CSM support), which is
    authoritative for the backbone RoPE/attention facts and which carries the
    typed nested ``depth_decoder_config`` / ``codec_config`` the C2/C3 aux
    modules need (see ``CsmForGeneration._build_aux_modules``). We deliberately
    do NOT register the vllm-omni ``CsmConfig`` over ``model_type="csm"`` (that
    would (a) override ``AutoConfig.from_pretrained`` and break the aux-module
    build, and (b) substitute hand-transcribed RoPE-scaling constants for the
    checkpoint's real values). Instead we read ``rope_theta`` defensively: the
    native config (and transformers >= 5.12 ``LlamaConfig``) folds ``rope_theta``
    into ``rope_scaling`` / ``rope_parameters`` and exposes no top-level
    ``rope_theta`` attribute, so we recover it from there.
    """
    # The rope-scaling dict passes through VERBATIM (including
    # original_max_position_embeddings). An earlier revision clamped
    # original_max_position_embeddings below max_position_embeddings citing a
    # transformers validation requirement, but no such hard requirement
    # exists -- transformers only logs a warning for original >= max_position
    # -- and the clamp silently altered the rope frequency curve whenever it
    # fired. The checkpoint's real value (1024 < 2048) needs no adjustment.
    rope_scaling = dict(config.rope_scaling) if config.rope_scaling else None

    # transformers >= 5.12 unified the RoPE config: ``rope_theta`` is nested
    # inside ``rope_scaling`` (key ``"rope_theta"``) and is NOT exposed as a
    # top-level attribute on either the native ``CsmConfig`` or the resulting
    # ``LlamaConfig``. Read it from ``rope_scaling`` first, then any top-level
    # attribute (older configs / the vllm-omni hoisted config), then the public
    # CSM-1B default. Keep it inside ``rope_scaling`` too so vLLM's
    # ``get_rope`` (which reads ``rope_parameters[...]["rope_theta"]``) finds it.
    rope_theta = None
    if rope_scaling is not None:
        rope_theta = rope_scaling.get("rope_theta")
    if rope_theta is None:
        rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = _BACKBONE_ROPE_THETA
    if rope_scaling is not None:
        rope_scaling.setdefault("rope_theta", rope_theta)

    return LlamaConfig(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        vocab_size=config.vocab_size,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        tie_word_embeddings=config.tie_word_embeddings,
        hidden_act="silu",
        attention_bias=False,
        rms_norm_eps=1e-5,
    )
