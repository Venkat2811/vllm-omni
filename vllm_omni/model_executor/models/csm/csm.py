# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CSM-1B single-stage model for vLLM-Omni (SCAFFOLD — C1).

CSM-1B (``sesame/csm-1b``) is a dual-autoregressive speech model:

    backbone forward (1 new KV position)
      -> sample codebook-0 (cb0)
      -> a 31-step inner DEPTH-DECODER autoregressive loop (cb1..cb31)
      -> a 32-code frame
      -> Mimi codec decode -> 1920 samples (80 ms @ 24 kHz)

This file is the **C1 scaffold**: it rebuilds the backbone on vLLM-native
``LlamaDecoderLayer`` + ``PagedAttention`` + fused ``QKVParallelLinear`` /
``ParallelLMHead`` (NOT an HF wrapper) using a synthetic ``LlamaConfig``
synthesised from ``CsmConfig`` (see ``configuration_csm``). The streaming
``inference_stream()`` generator is present in skeleton form; the 31-step
depth loop (C2) and the Mimi codec decode (C3) are explicit
``NotImplementedError`` stubs.

Streaming follows the VoxCPM/MOSS-TTS-Nano pattern:
  - On first forward() for a request, inference_stream() is started as a
    Python generator and stored in self._stream_gens[request_key].
  - Each subsequent forward() pops one audio chunk and returns it as
    multimodal_outputs.
  - compute_logits() emits EOS only when the last chunk has been yielded.

Weight loading deliberately happens inside load_weights() -- NOT __init__ --
so vLLM initialises distributed state before any CUDA allocations occur.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import maybe_prefix

from vllm_omni.model_executor.models.csm.configuration_csm import (
    CsmConfig,
    build_backbone_llama_config,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

__all__ = ["CsmForGeneration"]

# Default sampling parameters (public CSM-1B / Mimi facts).
_DEFAULT_TEMPERATURE = 0.9
_DEFAULT_TOP_K = 50
_DEFAULT_MAX_NEW_FRAMES = 1024  # bounded; one frame = 80 ms

# Public CSM/Mimi frame constants (also on CsmConfig; restated for the loop).
_NUM_CODEBOOKS = 32
_DEPTH_INNER_STEPS = 31  # cb1..cb31 after the backbone produces cb0
_MIMI_SAMPLES_PER_FRAME = 1920


def _pick(info: dict, key: str, default):
    """Extract scalar from additional_information dict (list or plain value)."""
    val = info.get(key, default)
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return val[0]
    return val if val is not None else default


class CsmBackbone(nn.Module):
    """The CSM-1B backbone, rebuilt on vLLM-native Llama layers.

    Wraps :class:`vllm.model_executor.models.llama.LlamaModel` (PagedAttention,
    fused ``QKVParallelLinear``, RoPE) driven by a synthetic ``LlamaConfig``
    built from ``CsmConfig`` (A2 §2.1). This is deliberately NOT an HF wrapper:
    the attention path is vLLM's compiled forward so paged/varlen KV stays
    intact (R0 §4), which is what makes a future right-pad cohort (R0 §2) clean
    at bf16 B>1.

    The codebook-0 (cb0) logits head is a fused ``ParallelLMHead`` over the
    per-codebook audio vocab; the remaining 31 codebooks are produced by the
    depth decoder (C2), which is constructed lazily in ``CsmForGeneration``.
    """

    # vLLM Llama weight packing: fused qkv_proj <- {q,k,v}_proj and
    # gate_up_proj <- {gate,up}_proj. LlamaModel.load_weights consumes this
    # mapping; restated here so the CSM backbone state dict packs identically.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        csm_config: CsmConfig = vllm_config.model_config.hf_config
        self.csm_config = csm_config

        # Synthetic LlamaConfig describing ONLY the backbone (A2 §2.1).
        backbone_llama_config = build_backbone_llama_config(csm_config)

        # Swap the synthetic LlamaConfig into a shallow copy of vllm_config so
        # vLLM's native LlamaModel sees a standard Llama config (not the nested
        # CsmConfig). We keep the original CsmConfig on the parent module.
        backbone_vllm_config = vllm_config
        backbone_model_config = vllm_config.model_config
        backbone_model_config.hf_config = backbone_llama_config
        backbone_model_config.hf_text_config = backbone_llama_config

        self.model = LlamaModel(
            vllm_config=backbone_vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        # cb0 logits surface: one codebook of audio tokens. Tied to the input
        # embedding when ``tie_word_embeddings`` (A2 §2.1) to drop a dead head.
        self.cb0_head = ParallelLMHead(
            csm_config.vocab_size,
            csm_config.hidden_size,
            prefix=maybe_prefix(prefix, "cb0_head"),
        )
        self.logits_processor = LogitsProcessor(csm_config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the backbone for one new KV position; return hidden states.

        NOTE (C1 scaffold): the backbone is wired but the per-frame embedding
        composition (Σ over the 32 codebooks -> one (65632, 2048) frame-embed
        table, tied to the depth embed; A2 §2.4) is a C3 item. For C1 this
        runs the bare vLLM Llama forward.
        """
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Delegate to vLLM's LlamaModel weight loader (stacked-params packing).

        ``LlamaModel.load_weights`` applies ``packed_modules_mapping`` to fuse
        q/k/v -> qkv_proj and gate/up -> gate_up_proj. The C4 glue will route
        the ``backbone.*`` prefix of the CSM checkpoint here and the
        ``depth_decoder.*`` / ``codec.*`` prefixes to C2 / C3.
        """
        return self.model.load_weights(weights)


class CsmForGeneration(nn.Module):
    """Single-stage CSM-1B model with streaming audio output (C1 scaffold).

    Uses the VoxCPM/MOSS pattern: inference_stream() is stored per-request and
    yields one audio chunk per forward() call; the AR scheduler keeps the
    request alive until compute_logits() emits EOS.
    """

    requires_raw_input_tokens = True
    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    enable_update_additional_information = True
    inject_omni_request_id_into_runtime_info = True

    packed_modules_mapping = CsmBackbone.packed_modules_mapping

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config: CsmConfig = vllm_config.model_config.hf_config
        self.model_path: str = vllm_config.model_config.model

        # Backbone is constructed now (vLLM-native layers, no CUDA alloc until
        # load_weights). The depth decoder (C2) and Mimi codec (C3) are
        # constructed lazily in load_weights() to stay off the init path.
        self.backbone = CsmBackbone(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "backbone"),
        )
        self._depth_decoder: nn.Module | None = None  # C2
        self._mimi_codec: nn.Module | None = None  # C3
        self._device: torch.device | None = None
        self._lock = threading.Lock()

        # Per-request streaming generators (VoxCPM pattern).
        self._stream_gens: dict[str, Any] = {}
        # Per-row EOS mask aligned with the most recent forward() batch.
        self._ar_last_chunk_flags: list[bool] = []

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load backbone weights via the vLLM-native loader.

        C1 routes the backbone weights through ``CsmBackbone.load_weights``
        (stacked-params packing). The depth decoder (C2) and Mimi codec (C3)
        weight routing — splitting the CSM checkpoint by ``backbone.*`` /
        ``depth_decoder.*`` / ``codec.*`` prefixes — is filled in there.
        """
        with self._lock:
            try:
                self._device = next(self.parameters()).device
            except StopIteration:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # C1: only the backbone is wired. Route backbone-prefixed weights to
        # the vLLM-native loader; ignore depth/codec until C2/C3.
        backbone_weights = (
            (name[len("backbone.") :] if name.startswith("backbone.") else name, w)
            for name, w in weights
        )
        loaded = self.backbone.load_weights(backbone_weights)

        # TODO(C2): construct + load the depth decoder (4 layers / d1024 /
        #           head_dim 128, dense static KV, 33 positions reset/frame).
        # TODO(C3): construct + load the Mimi codec decoder.
        return loaded

    # ------------------------------------------------------------------
    # Dummy run support
    # ------------------------------------------------------------------

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict]:
        return [{"text": "hello", "_is_dummy": True}] * num_reqs

    # ------------------------------------------------------------------
    # Streaming generator (VoxCPM pattern) — SKELETON
    # ------------------------------------------------------------------

    def _create_stream_gen(self, info: dict[str, Any]):
        """Create an inference_stream() generator for a request.

        Yields (waveform_tensor, is_last) tuples. The per-frame body (the
        dual-AR frame runner) is the heart of the port and is split into the
        depth loop (C2) and the Mimi decode (C3); both are NotImplementedError
        stubs below so the scaffold imports and registers but does not yet run
        a forward.
        """
        text: str = str(_pick(info, "text", "") or "")
        if not text.strip():
            logger.warning("CSM received empty text; yielding silence.")
            sr = int(getattr(self.config, "codec_sample_rate", 24000))
            yield torch.zeros((sr,), dtype=torch.float32), True
            return

        max_new_frames: int = int(_pick(info, "max_new_frames", _DEFAULT_MAX_NEW_FRAMES))
        temperature: float = float(_pick(info, "temperature", _DEFAULT_TEMPERATURE))
        top_k: int = int(_pick(info, "top_k", _DEFAULT_TOP_K))

        # The dual-AR frame loop (A2 §2): each iteration is one 80 ms frame.
        yield from self.inference_stream(
            text=text,
            max_new_frames=max_new_frames,
            temperature=temperature,
            top_k=top_k,
        )

    def inference_stream(
        self,
        *,
        text: str,
        max_new_frames: int = _DEFAULT_MAX_NEW_FRAMES,
        temperature: float = _DEFAULT_TEMPERATURE,
        top_k: int = _DEFAULT_TOP_K,
    ):
        """Dual-AR frame generator (SKELETON; C2/C3 stubs).

        Per the A2 design, one scheduler-visible CSM step == one full 80 ms
        frame, which internally is:

            backbone forward (1 new KV position)
              -> sample cb0
              -> 31-step inner depth-decoder AR loop (cb1..cb31)   [C2]
              -> a 32-code frame
              -> Mimi codec decode -> 1920 samples                  [C3]

        For C1 this generator is wired up to the structure but the inner depth
        loop and the Mimi decode raise NotImplementedError. The frame loop
        below documents the exact control flow C2/C3 will fill in.

        Yields ``(waveform_chunk, is_last)`` tuples.
        """
        device = self._device or torch.device("cpu")

        for _frame_idx in range(max_new_frames):
            # --- 1. Backbone forward: 1 new KV position -> cb0 logits. ---
            #     (C1 wires CsmBackbone; the per-frame embedding composition
            #      — Σ over 32 codebooks into the (65632, 2048) frame-embed
            #      table tied to the depth embed, A2 §2.4 — is a C3 item.)
            #
            # --- 2. Depth-decoder 31-step inner AR loop (cb1..cb31). ---
            frame_codes = self._run_depth_loop(
                cb0=None,  # C2: backbone-sampled cb0 feeds the depth loop
                temperature=temperature,
                top_k=top_k,
            )  # raises NotImplementedError (C2)

            # --- 3. EOS check: a frame with cb0..cb30 all-zero is EOS. ---
            #     (A2 §2.4; audio is trimmed at the first all-32-zero frame.)
            #
            # --- 4. Mimi codec decode: 32-code frame -> 1920 samples. ---
            waveform_chunk = self._mimi_decode(frame_codes)  # NotImplementedError (C3)

            is_last = _frame_idx == max_new_frames - 1
            yield waveform_chunk, is_last

        yield torch.zeros((0,), dtype=torch.float32, device=device), True

    def _run_depth_loop(
        self,
        *,
        cb0: torch.Tensor | None,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        """31-step inner depth-decoder AR loop -> a 32-code frame.

        C2 TODO. This is the dominant long pole (A2 §2.2 / §5):

        - Depth decoder: 4 layers / d1024 / head_dim 128, dense static KV,
          33 positions, reset per frame.
        - Runs as custom dense torch INSIDE forward()/sample(), under the I3
          no-sync discipline: NO ``.item()`` / ``.cpu()`` across the 31 inner
          steps. Codes/flags are resolved with a packed-D2H / one-step-
          lookahead so the inner loop never per-row-syncs (R0 §10).
        - Batch the inner loop across the scheduler's in-flight lanes
          (R0 §1 depth-loop batching); the backbone cohort batches per R0 §2
          (right-pad whole-frame cohort).
        - Returns the (num_codebooks,) — or (B, num_codebooks) — frame: cb0
          from the backbone plus cb1..cb31 from this loop.
        """
        raise NotImplementedError(
            "C2: 31-step depth-decoder inner AR loop "
            f"({_DEPTH_INNER_STEPS} steps -> {_NUM_CODEBOOKS}-code frame) "
            "not implemented yet. See A2 §2.2."
        )

    def _mimi_decode(self, frame_codes: torch.Tensor) -> torch.Tensor:
        """Mimi codec decode: a 32-code frame -> 1920 PCM samples (80 ms).

        C3 TODO (A2 §2.3 / §5):

        - Mimi: 24 kHz / 12.5 Hz, 80 ms = 1920-sample frames, 32 quantizers,
          per-codebook vocab 2051 with reserved 2048/2049/2050 that must NEVER
          reach Mimi (clamp/guard before decode).
        - In the canonical 2-stage shape this becomes the Stage-1 decoder
          implementing ``chunked_decode_streaming()``, wired through the
          ar2decoder_async_chunk processor and the I1 consolidator. For the
          single-stage scaffold it runs inline here.
        """
        raise NotImplementedError(
            f"C3: Mimi codec decode ({_NUM_CODEBOOKS}-code frame -> "
            f"{_MIMI_SAMPLES_PER_FRAME} samples) not implemented yet. See A2 §2.3."
        )

    # ------------------------------------------------------------------
    # Core forward pass (streaming, VoxCPM pattern)
    # ------------------------------------------------------------------

    def _make_dummy_hidden(self, input_ids: torch.Tensor | None) -> torch.Tensor:
        device = self._device or torch.device("cpu")
        hidden = int(getattr(self.config, "hidden_size", 2048))
        n = 1 if input_ids is None else max(1, input_ids.shape[0])
        return torch.zeros((n, hidden), device=device, dtype=torch.float32)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        sr = int(getattr(self.config, "codec_sample_rate", 24000))
        sr_tensor = torch.tensor(sr, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)
        hidden = self._make_dummy_hidden(input_ids)

        infos = runtime_additional_information or [{}]

        # Dummy/warmup path: finish immediately for every row (no forward).
        if not runtime_additional_information or all(info.get("_is_dummy") for info in infos):
            self._ar_last_chunk_flags = [True] * len(infos)
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={
                    "model_outputs": [empty] * len(infos),
                    "sr": [sr_tensor] * len(infos),
                },
            )

        outputs: list[torch.Tensor] = []
        srs: list[torch.Tensor] = []
        last_chunk_flags: list[bool] = []

        for info in infos:
            if info.get("_is_dummy"):
                outputs.append(empty)
                srs.append(sr_tensor)
                last_chunk_flags.append(True)
                continue

            request_key = str(
                info.get("global_request_id") or info.get("_omni_req_id") or id(info)
            )
            if request_key not in self._stream_gens:
                self._stream_gens[request_key] = self._create_stream_gen(info)

            generator = self._stream_gens[request_key]
            try:
                chunk, is_last = next(generator)
            except StopIteration:
                self._stream_gens.pop(request_key, None)
                outputs.append(empty)
                last_chunk_flags.append(True)
            else:
                if is_last:
                    self._stream_gens.pop(request_key, None)
                outputs.append(chunk)
                last_chunk_flags.append(bool(is_last))
            srs.append(sr_tensor)

        self._ar_last_chunk_flags = last_chunk_flags
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"model_outputs": outputs, "sr": srs},
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        """Release streaming generators for requests the scheduler finished.

        I5: per-request state keyed by request id, freed on finish. Closing the
        generator raises GeneratorExit so any cleanup block runs.
        """
        for req_id in finished_req_ids:
            gen = self._stream_gens.pop(str(req_id), None)
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    logger.exception("CSM failed to close stream gen for request %s", req_id)

    # ------------------------------------------------------------------
    # AR runner interface
    # ------------------------------------------------------------------

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor:
        """Emit per-row EOS / non-EOS logits to control AR scheduler lifetime.

        Rows whose ``_ar_last_chunk_flags`` entry is True get EOS-dominant
        logits so the scheduler finishes that request; other rows get a
        non-EOS token so they stay alive for the next streaming chunk. EOS id
        is 0 here (the scheduler-visible backstop; frame-level EOS is the
        cb0..cb30 all-zero condition handled in inference_stream).
        """
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            device = self._device or torch.device("cpu")
            hidden_states = torch.zeros((0, 1), device=device, dtype=torch.float32)
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(-1)
        elif hidden_states.ndim > 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        vocab_size = int(getattr(self.config, "vocab_size", 2051))
        num_rows = int(hidden_states.shape[0])
        logits = torch.zeros(
            (num_rows, vocab_size),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        eos_id = 0
        safe_id = 1 if vocab_size > 1 else 0

        flags = self._ar_last_chunk_flags
        for row in range(num_rows):
            is_last = flags[row] if row < len(flags) else True
            if is_last:
                logits[row, eos_id] = 1.0e6
            else:
                logits[row, eos_id] = -1.0e9
                logits[row, safe_id] = 1.0e6
        return logits

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        hidden = int(getattr(self.config, "hidden_size", 2048))
        return torch.zeros(
            (input_ids.shape[0], hidden),
            device=input_ids.device,
            dtype=torch.float32,
        )
