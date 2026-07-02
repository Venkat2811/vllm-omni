# SPDX-License-Identifier: Apache-2.0
"""CSM-1B (Sesame) serving adapter (#4330 TTS adapter framework).

CSM-1B is a plain text -> speech model with speaker-id conditioning, served on
the AR engine_client path. The OpenAI ``voice`` field maps to a CSM speaker id
(a non-negative integer string, default ``"0"``); there is no reference-audio
voice cloning on this path. The adapter owns request validation and Stage-0
backbone prompt construction for the two-stage CSM pipeline (backbone AR +
inline 31-step depth -> Mimi code2wav; see ``model_executor/models/csm``).

Drop-in for vllm_omni/entrypoints/openai/tts_adapters/csm.py and add ``csm`` to
the import list in that package's __init__.py.
"""

from typing import TYPE_CHECKING, Any

from vllm.inputs import tokens_input
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    conditioning_cache_salt,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)

# Request bounds (mirror serving_speech._TTS_MAX_NEW_TOKENS_{MIN,MAX}).
_TTS_MAX_NEW_TOKENS_MIN = 1
_TTS_MAX_NEW_TOKENS_MAX = 4096
# Served sampling defaults mirror the HF reference generation config
# (temperature 0.9, top_k 50). Sampling reaches the model's natural all-zero
# EOS frame far more reliably than greedy, which can fall into CSM's
# non-terminating repetition attractor and run to the frame cap with a silent
# tail. Callers override per request via ``extra_params``, e.g.
# ``{"temperature": 0.0, "top_k": 0}`` for deterministic greedy output.
_DEFAULT_TEMPERATURE = 0.9
_DEFAULT_TOP_K = 50
_TEMPERATURE_MAX = 2.0
_TOP_K_MAX = 2048
# Fail-closed frame cap for rollouts that never emit a natural EOS (125 frames
# == 10 s at 80 ms/frame, the CSM reference cap). The backbone forces the frame
# EOS at the cap (csm_backbone _DEFAULT_MAX_FRAMES / GATE-B); ``max_new_tokens``
# overrides it per request.
_DEFAULT_CSM_MAX_FRAMES = 125


@register_tts_adapter
class CsmTTSAdapter(ARTTSAdapter):
    """Adapter for Sesame CSM-1B (AR ``engine_client`` backend)."""

    stage_keys = frozenset({"csm"})
    name = "csm"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._tokenizer = None

    def _get_tokenizer(self):
        """Lazily load + cache the CSM Llama text tokenizer.

        CSM-1B ships a standard Llama tokenizer plus ``<|begin_of_text|>`` /
        ``<|end_of_text|>``. We tokenize the speaker-tagged text here so the
        prompt carries the exact ``prompt_token_ids`` the Stage-0 backbone embeds
        in ``_embed_text_prompt``.
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            model_name = self.ctx.engine_client.model_config.model
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return self._tokenizer

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.voice is not None and not str(request.voice).strip().isdigit():
            return "CSM 'voice' must be a non-negative integer speaker id (e.g. '0')"
        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"
            if request.max_new_tokens > _TTS_MAX_NEW_TOKENS_MAX:
                return f"max_new_tokens cannot exceed {_TTS_MAX_NEW_TOKENS_MAX}"
        return self._validate_sampling_extras(request)

    @staticmethod
    def _validate_sampling_extras(request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate the CSM sampling knobs carried in ``extra_params``."""
        extras = request.extra_params
        if extras is None:
            return None
        if not isinstance(extras, dict):
            return "extra_params must be a JSON object"
        temperature = extras.get("temperature")
        if temperature is not None:
            if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
                return "extra_params.temperature must be a number"
            if not 0.0 <= float(temperature) <= _TEMPERATURE_MAX:
                return f"extra_params.temperature must be in [0, {_TEMPERATURE_MAX}]"
        top_k = extras.get("top_k")
        if top_k is not None:
            if not isinstance(top_k, int) or isinstance(top_k, bool):
                return "extra_params.top_k must be an integer"
            if not 0 <= top_k <= _TOP_K_MAX:
                return f"extra_params.top_k must be in [0, {_TOP_K_MAX}]"
        return None

    async def build(
        self,
        request: "OpenAICreateSpeechRequest",
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        """Build the Stage-0 backbone prompt for a CSM text -> speech request.

        CSM's text conditioning is ``<|begin_of_text|>[<spk>]<text><|end_of_text|>``.
        The backbone reads the ids back out of ``additional_information`` via
        ``_pick`` (csm_backbone), which unwraps the batch-of-1 convention by
        returning ``val[0]`` for list values, so every field is LIST-WRAPPED (a
        bare list would make ``_pick`` return only the first id, embedding one BOS
        token and zero-padding the rest: the 1-2-frame truncated-audio bug).

        Sampling: ``temperature`` / ``top_k`` come from ``request.extra_params``
        (validated above) and drive both cb0 and the inline depth decoder, with
        HF-reference defaults. They travel via ``additional_information`` because
        CSM samples inside the model (``sample_logits`` + ``depth.run``), not
        through the engine ``SamplingParams`` -- and so does ``request.seed``
        (qwen3_tts precedent: an in-model sampler cannot see the engine-side
        ``SamplingParams.seed``), which the backbone turns into a per-request
        ``torch.Generator`` so the API's documented determinism contract holds.
        ``max_new_frames`` bounds a rollout that never reaches natural EOS.
        """
        tokenizer = self._get_tokenizer()
        speaker = str(request.voice).strip() if request.voice is not None else "0"
        text = f"<|begin_of_text|>[{speaker}]{request.input}<|end_of_text|>"
        prompt_token_ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])

        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        temperature = float(extras.get("temperature", _DEFAULT_TEMPERATURE))
        top_k = int(extras.get("top_k", _DEFAULT_TOP_K))
        max_frames = request.max_new_tokens if request.max_new_tokens is not None else _DEFAULT_CSM_MAX_FRAMES
        additional_information: dict[str, Any] = {
            "prompt_token_ids": [prompt_token_ids],
            "temperature": [temperature],
            "top_k": [top_k],
            "max_new_frames": [max_frames],
        }
        if request.seed is not None:
            additional_information["seed"] = [request.seed]

        prompt = tokens_input(prompt_token_ids=prompt_token_ids)
        prompt["additional_information"] = additional_information
        prompt["cache_salt"] = conditioning_cache_salt(request, additional_information)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="csm")
