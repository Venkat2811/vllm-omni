# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TTS serving adapter registry (RFC #4327).

Pure-Python registry/resolution logic; no model or GPU resources are loaded.
"""

import pytest

from vllm_omni.entrypoints.openai.tts_adapters import (
    TTS_ADAPTER_REGISTRY,
    ARTTSAdapter,
    DiffusionTTSAdapter,
    all_tts_model_types,
    resolve_adapter,
)
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import Qwen3TTSAdapter

# Every dedicated TTS model-type must have an adapter so the orchestrator's
# uniform ``self._adapter.build(...)`` dispatch covers it.
EXPECTED_MODEL_TYPES = {
    "csm",
    "qwen3_tts",
    "voxcpm2",
    "voxtral_tts",
    "fish_tts",
    "cosyvoice3",
    "omnivoice",
    "covo_audio",
    "ming_tts",
    "moss_tts_nano",
    "moss_tts",
    "higgs_audio_v2",
    "higgs_audio_v3",
    "glm_tts",
    "step_audio2",
}


def test_all_model_types_registered():
    assert EXPECTED_MODEL_TYPES <= all_tts_model_types()


def test_registry_keyed_by_name():
    for name, cls in TTS_ADAPTER_REGISTRY.items():
        assert cls.name == name


def test_resolve_each_model_type():
    for model_type in EXPECTED_MODEL_TYPES:
        cls = resolve_adapter(model_type)
        assert cls is not None, model_type
        assert cls.name == model_type


def test_resolve_qwen3_tts_class():
    assert resolve_adapter("qwen3_tts") is Qwen3TTSAdapter


def test_resolve_unknown_returns_none():
    assert resolve_adapter("not_a_real_model") is None
    assert resolve_adapter(None) is None


def test_ming_flash_omni_not_migrated():
    """ming_flash_omni is intentionally excluded from the adapter migration in
    this PR; it stays on the legacy inline dispatch in serving_speech.py."""
    assert resolve_adapter("ming_flash_omni_tts") is None


def test_voxcpm2_resolves():
    """VoxCPM2 (the served ``latent_generator`` model) resolves cleanly.

    Detection never returns the legacy ``voxcpm`` type, so there is no shared
    stage-key ambiguity to resolve.
    """
    assert resolve_adapter("voxcpm2") is not None
    assert resolve_adapter("voxcpm") is None


def test_all_adapters_are_ar_or_diffusion():
    for cls in TTS_ADAPTER_REGISTRY.values():
        assert issubclass(cls, (ARTTSAdapter, DiffusionTTSAdapter))
        assert cls.backend in ("ar", "diffusion")


def test_qwen3_tts_metadata():
    assert Qwen3TTSAdapter.backend == "ar"
    assert issubclass(Qwen3TTSAdapter, ARTTSAdapter)


def test_diffusion_adapter_extra_body_params_fallback():
    class _DiffAdapter(DiffusionTTSAdapter):
        name = "diff_probe"

        async def build(self, request, sampling_params_list):  # pragma: no cover
            raise NotImplementedError

    assert _DiffAdapter.extra_body_params() == frozenset()


def _speech_request(**overrides):
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

    fields = {"model": "sesame/csm-1b", "input": "Hello there."}
    fields.update(overrides)
    return OpenAICreateSpeechRequest(**fields)


def test_csm_metadata():
    from vllm_omni.entrypoints.openai.tts_adapters.csm import CsmTTSAdapter

    assert resolve_adapter("csm") is CsmTTSAdapter
    assert CsmTTSAdapter.backend == "ar"
    assert issubclass(CsmTTSAdapter, ARTTSAdapter)


def test_csm_validate_sampling_extras():
    from vllm_omni.entrypoints.openai.tts_adapters.csm import CsmTTSAdapter

    check = CsmTTSAdapter._validate_sampling_extras
    assert check(_speech_request()) is None
    assert check(_speech_request(extra_params={"temperature": 0.7, "top_k": 40})) is None
    assert check(_speech_request(extra_params={"temperature": 0.0, "top_k": 0})) is None
    assert "temperature" in check(_speech_request(extra_params={"temperature": 3.0}))
    assert "temperature" in check(_speech_request(extra_params={"temperature": "hot"}))
    assert "top_k" in check(_speech_request(extra_params={"top_k": -1}))
    assert "top_k" in check(_speech_request(extra_params={"top_k": 1.5}))
    assert "top_k" in check(_speech_request(extra_params={"top_k": True}))


@pytest.mark.asyncio
async def test_csm_build_carries_sampling_and_frame_cap():
    """``build`` mirrors extra_params sampling (HF-reference defaults) and the
    frame cap into ``additional_information``, list-wrapped for ``_pick``."""
    from vllm_omni.entrypoints.openai.tts_adapters.csm import (
        _DEFAULT_CSM_MAX_FRAMES,
        _DEFAULT_TEMPERATURE,
        _DEFAULT_TOP_K,
        CsmTTSAdapter,
    )

    adapter = CsmTTSAdapter.__new__(CsmTTSAdapter)
    adapter._tokenizer = lambda text, add_special_tokens=False: {"input_ids": [1, 2, 3]}

    prepared = await adapter.build(_speech_request(), [], False)
    info = prepared.prompt["additional_information"]
    assert info["temperature"] == [_DEFAULT_TEMPERATURE]
    assert info["top_k"] == [_DEFAULT_TOP_K]
    assert info["max_new_frames"] == [_DEFAULT_CSM_MAX_FRAMES]
    assert info["prompt_token_ids"] == [[1, 2, 3]]

    greedy = _speech_request(extra_params={"temperature": 0.0, "top_k": 0}, max_new_tokens=64)
    prepared = await adapter.build(greedy, [], False)
    info = prepared.prompt["additional_information"]
    assert info["temperature"] == [0.0]
    assert info["top_k"] == [0]
    assert info["max_new_frames"] == [64]


@pytest.mark.asyncio
async def test_csm_build_forwards_request_seed_list_wrapped():
    """``request.seed`` (the API's documented determinism knob) must reach the
    in-model sampler via ``additional_information`` -- CSM samples in-model,
    so the engine-side ``SamplingParams.seed`` alone cannot honor it. Wrapped
    in a list per the ``_pick`` batch-of-1 convention like every other knob."""
    from vllm_omni.entrypoints.openai.tts_adapters.csm import CsmTTSAdapter

    adapter = CsmTTSAdapter.__new__(CsmTTSAdapter)
    adapter._tokenizer = lambda text, add_special_tokens=False: {"input_ids": [1, 2, 3]}

    prepared = await adapter.build(_speech_request(seed=1234), [], False)
    assert prepared.prompt["additional_information"]["seed"] == [1234]

    # No seed on the request -> no key, so the backbone keeps the default RNG.
    prepared = await adapter.build(_speech_request(), [], False)
    assert "seed" not in prepared.prompt["additional_information"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
