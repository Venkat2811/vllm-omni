# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CSM-1B single-stage dual-AR speech model for vLLM-Omni.

CSM-1B (``sesame/csm-1b``) is a dual-autoregressive speech model:

    backbone forward (1 new KV position)
      -> sample codebook-0 (cb0)
      -> a 31-step inner DEPTH-DECODER autoregressive loop (cb1..cb31)
      -> a 32-code frame
      -> Mimi codec decode -> 1920 samples (80 ms @ 24 kHz)

The backbone is rebuilt on vLLM-native ``LlamaDecoderLayer`` + ``PagedAttention``
+ fused ``QKVParallelLinear`` (C1) using a synthetic ``LlamaConfig`` synthesised
from ``CsmConfig`` (see ``configuration_csm``). The 31-step depth decoder (C2)
runs as **custom dense torch inside** this model's streaming loop — it is a
second, small AR model nested inside each backbone decode step, with its own
4-layer / d1024 / head_dim128 dense KV that is reset every frame (33 positions).
The Mimi codec (C3) decodes each 32-code frame into 1920 PCM samples.

Per-frame embedding composition (A2 §2.4): the next backbone position embeds the
whole 32-code frame by summing the 32 per-codebook embeddings (with per-codebook
offsets) into one ``(num_codebooks * vocab_size, hidden)`` table, then feeds the
result to the backbone as ``inputs_embeds`` — vLLM's ``LlamaModel.forward``
consumes ``inputs_embeds`` directly and bypasses its own (unused) token embed.

Streaming follows the VoxCPM/MOSS-TTS-Nano pattern:
  - On first forward() for a request, inference_stream() is started as a
    Python generator and stored in self._stream_gens[request_key].
  - Each subsequent forward() pops one audio chunk and returns it as
    multimodal_outputs.
  - compute_logits() emits EOS only when the last chunk has been yielded.

Weight loading deliberately happens inside load_weights() -- NOT __init__ --
so vLLM initialises distributed state before any CUDA allocations occur. The
CSM checkpoint is split by ``backbone_model.*`` / ``lm_head.*`` /
``depth_decoder.*`` / ``codec_model.*`` prefixes (see load_weights()).
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
_CODEBOOK_EOS_ID = 0  # codebook_eos_token_id; a frame with cb0..cb30==0 is EOS.

# Reserved per-codebook ids (Mimi vocab 2051): 2048/2049/2050 must NEVER reach
# the Mimi codec (its real codebook size is 2048). Clamp before decode.
_MIMI_CODEBOOK_SIZE = 2048


def _pick(info: dict, key: str, default):
    """Extract scalar from additional_information dict (list or plain value)."""
    val = info.get(key, default)
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return val[0]
    return val if val is not None else default


def _sample_logits(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """Sample one token id per row from logits.

    PHASE3 §1 numerics hygiene: ``logits`` are cast to fp32 *before* this call
    (no bf16 op flows into the sampler). temperature<=0 is greedy (argmax).
    Returns a ``(B,)`` LongTensor. Keeps the op shapes fixed (top-k via a
    masked fill) so a future CUDA-graph capture sees a stable reduction order.
    """
    logits = logits.float()
    if temperature is None or temperature <= 0.0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature
    if top_k and top_k > 0 and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class CsmBackbone(nn.Module):
    """The CSM-1B backbone, rebuilt on vLLM-native Llama layers.

    Wraps :class:`vllm.model_executor.models.llama.LlamaModel` (PagedAttention,
    fused ``QKVParallelLinear``, RoPE) driven by a synthetic ``LlamaConfig``
    built from ``CsmConfig`` (A2 §2.1). This is deliberately NOT an HF wrapper:
    the attention path is vLLM's compiled forward so paged/varlen KV stays
    intact (R0 §4), which is what makes a future right-pad cohort (R0 §2) clean
    at bf16 B>1.

    The codebook-0 (cb0) logits head is ``cb0_head`` over the per-codebook audio
    vocab; in the ``sesame/csm-1b`` checkpoint this is the (untied) ``lm_head``.

    NOTE on embeddings: the CSM backbone's input embedding is NOT a plain token
    table — it is the Σ-over-32-codebooks frame embed (A2 §2.4), owned by the
    parent ``CsmForGeneration`` as ``frame_embed``. The vLLM ``LlamaModel`` keeps
    its own ``embed_tokens`` (a ``VocabParallelEmbedding`` over the cb0 vocab) but
    it is unused at inference: we always drive the backbone with ``inputs_embeds``
    composed from the frame embed, so no checkpoint weight is routed to it.
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

        # Swap the synthetic LlamaConfig into vllm_config so vLLM's native
        # LlamaModel sees a standard Llama config (not the nested CsmConfig).
        # We keep the original CsmConfig on the parent module.
        backbone_vllm_config = vllm_config
        backbone_model_config = vllm_config.model_config
        backbone_model_config.hf_config = backbone_llama_config
        backbone_model_config.hf_text_config = backbone_llama_config

        self.model = LlamaModel(
            vllm_config=backbone_vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        # cb0 logits surface: one codebook of audio tokens. In the checkpoint
        # this is the untied ``lm_head`` (config sets tie_word_embeddings=False;
        # tie_codebooks_embeddings ties the *audio embeddings*, handled in C3).
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
        """Run the backbone for one or more KV positions; return hidden states.

        The per-frame embedding composition (Σ over the 32 codebooks via the
        parent's ``frame_embed`` table; A2 §2.4) is performed by the caller and
        passed through ``inputs_embeds``; this just forwards to the vLLM Llama.
        """
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Delegate the backbone decoder layers + norm to vLLM's LlamaModel.

        ``LlamaModel.load_weights`` applies ``packed_modules_mapping`` to fuse
        q/k/v -> qkv_proj and gate/up -> gate_up_proj. Only the decoder-layer
        and final-norm weights of the CSM ``backbone_model.*`` prefix are routed
        here; the audio embed table and the cb0/lm_head are handled by the
        parent (see ``CsmForGeneration.load_weights``).
        """
        return self.model.load_weights(weights)


class CsmForGeneration(nn.Module):
    """Single-stage CSM-1B model with streaming audio output.

    Uses the VoxCPM/MOSS pattern: inference_stream() is stored per-request and
    yields one audio chunk per forward() call; the AR scheduler keeps the
    request alive until compute_logits() emits EOS.

    Dual-AR per-frame body (A2 §2):
      backbone forward -> cb0 -> 31-step depth loop (cb1..cb31) -> 32-code frame
      -> Mimi decode -> 1920 samples.
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

        self.num_codebooks = int(getattr(self.config, "num_codebooks", _NUM_CODEBOOKS))
        self.codebook_vocab_size = int(
            getattr(self.config, "codebook_vocab_size", 2051)
        )
        self.hidden_size = int(getattr(self.config, "hidden_size", 2048))

        # Backbone is constructed now (vLLM-native layers, no CUDA alloc until
        # load_weights). Constructing CsmBackbone mutates vllm_config's
        # model_config.hf_config into the synthetic LlamaConfig, so we cache the
        # CsmConfig above first.
        self.backbone = CsmBackbone(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "backbone"),
        )

        # The depth decoder (C2) and Mimi codec (C3) are HF reference modules
        # constructed from the nested CSM config. They are built lazily in
        # load_weights() (after distributed init) to stay off the __init__ path.
        self._depth_decoder: nn.Module | None = None  # CsmDepthDecoderForCausalLM
        self._mimi_codec: nn.Module | None = None  # MimiModel
        # Frame-embed table (A2 §2.4): Σ over the 32 codebooks. Built in
        # load_weights() from the backbone_model.embed_tokens.* checkpoint
        # weight; tied to the depth embed (tie_codebooks_embeddings).
        self._frame_embed: nn.Module | None = None  # CsmBackboneModelEmbeddings
        self._text_embed: nn.Module | None = None  # nn.Embedding (text prompt)
        self._hf_config: Any = None  # authoritative HF CsmConfig (load_weights)
        self._device: torch.device | None = None
        self._dtype: torch.dtype = torch.float32
        self._lock = threading.Lock()

        # Per-request streaming generators (VoxCPM pattern).
        self._stream_gens: dict[str, Any] = {}
        # Per-row EOS mask aligned with the most recent forward() batch.
        self._ar_last_chunk_flags: list[bool] = []

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Split the CSM checkpoint by prefix and route each part.

        The ``sesame/csm-1b`` checkpoint top-level prefixes are:
          - ``backbone_model.layers.* / backbone_model.norm.*`` -> vLLM Llama
            (via ``CsmBackbone.load_weights``; the ``backbone_model.`` prefix is
            stripped to the ``model.`` names vLLM expects).
          - ``backbone_model.embed_tokens.embed_audio_tokens.weight`` -> the
            frame-embed table (A2 §2.4), held by ``self._frame_embed``.
          - ``lm_head.weight`` -> the cb0 head (``backbone.cb0_head``).
          - ``embed_text_tokens.weight`` -> text-prefill embed (kept on the
            frame-embed module; only used for the text prompt prefill).
          - ``depth_decoder.*`` -> the HF ``CsmDepthDecoderForCausalLM`` (C2).
          - ``codec_model.*`` -> the HF Mimi codec (C3).
        """
        with self._lock:
            if self._depth_decoder is not None:
                return set()
            try:
                self._device = next(self.parameters()).device
            except StopIteration:
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            self._dtype = self.config.torch_dtype or torch.float32

            self._build_aux_modules()

        backbone_layer_weights: list[tuple[str, torch.Tensor]] = []
        depth_weights: list[tuple[str, torch.Tensor]] = []
        codec_weights: list[tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()

        frame_embed_state: dict[str, torch.Tensor] = {}
        text_embed_state: dict[str, torch.Tensor] = {}
        cb0_head_state: dict[str, torch.Tensor] = {}

        for name, w in weights:
            if name.startswith("backbone_model.embed_tokens."):
                # backbone_model.embed_tokens.embed_audio_tokens.weight ->
                # the frame-embed table (strip the embed_tokens. prefix so the
                # remaining key matches CsmBackboneModelEmbeddings' param name).
                sub = name[len("backbone_model.embed_tokens.") :]
                frame_embed_state[sub] = w
                loaded.add(name)
            elif name.startswith("backbone_model."):
                # decoder layers + final norm -> vLLM LlamaModel (strip prefix)
                backbone_layer_weights.append((name[len("backbone_model.") :], w))
            elif name == "lm_head.weight":
                cb0_head_state["weight"] = w
                loaded.add(name)
            elif name == "embed_text_tokens.weight":
                text_embed_state["weight"] = w  # text-prompt prefill embed
                loaded.add(name)
            elif name.startswith("depth_decoder."):
                depth_weights.append((name[len("depth_decoder.") :], w))
            elif name.startswith("codec_model."):
                codec_weights.append((name[len("codec_model.") :], w))
            else:
                logger.warning("CSM load_weights: unrouted weight %s", name)

        # Backbone decoder layers + norm through vLLM's stacked-param loader.
        loaded |= {
            f"backbone_model.{n}"
            for n in self.backbone.load_weights(iter(backbone_layer_weights))
        }

        # cb0 head <- lm_head.weight.
        if "weight" in cb0_head_state:
            self._load_into(self.backbone.cb0_head, {"weight": cb0_head_state["weight"]})

        # Frame-embed table + text-prompt embed via direct state load.
        if self._frame_embed is not None and frame_embed_state:
            self._frame_embed.load_state_dict(frame_embed_state, strict=False)
        if self._text_embed is not None and text_embed_state:
            self._text_embed.load_state_dict(text_embed_state, strict=False)

        # Depth decoder + Mimi codec via HF-style state-dict load.
        if self._depth_decoder is not None:
            missing, unexpected = self._depth_decoder.load_state_dict(
                dict(depth_weights), strict=False
            )
            if unexpected:
                logger.warning("CSM depth_decoder unexpected keys: %s", unexpected[:8])
            loaded |= {f"depth_decoder.{n}" for n, _ in depth_weights}
        if self._mimi_codec is not None:
            self._mimi_codec.load_state_dict(dict(codec_weights), strict=False)
            loaded |= {f"codec_model.{n}" for n, _ in codec_weights}

        # tie_codebooks_embeddings: depth embed table == backbone audio embed.
        self._maybe_tie_depth_embed()
        return loaded

    def _build_aux_modules(self) -> None:
        """Construct the depth decoder, Mimi codec, frame-embed + text embed.

        These are HF reference modules (dense torch) — the depth decoder runs
        the 31-step inner loop with its own DynamicCache (NOT PagedAttention),
        which is exactly the A2 §2.2 contract ("custom dense torch inside the
        Stage-0 model"). The Mimi codec is the C3 decoder.

        They are built from the **authoritative HF CsmConfig** loaded from the
        model path (NOT the vllm-omni hoisted CsmConfig), so the strict HF
        attribute names (``codebook_size``, the typed ``CsmDepthDecoderConfig``,
        the ``MimiConfig``) are exactly what the HF module ``__init__`` expects.
        """
        from transformers import AutoConfig, AutoModel
        from transformers.models.csm.modeling_csm import (
            CsmBackboneModelEmbeddings,
            CsmDepthDecoderForCausalLM,
        )

        hf_cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self._hf_config = hf_cfg
        depth_cfg = hf_cfg.depth_decoder_config
        codec_cfg = hf_cfg.codec_config

        self._frame_embed = CsmBackboneModelEmbeddings(hf_cfg)
        self._text_embed = nn.Embedding(
            int(hf_cfg.text_vocab_size), int(hf_cfg.hidden_size)
        )
        self._depth_decoder = CsmDepthDecoderForCausalLM(depth_cfg)
        self._mimi_codec = AutoModel.from_config(codec_cfg)

        for m in (
            self._frame_embed,
            self._text_embed,
            self._depth_decoder,
            self._mimi_codec,
        ):
            m.to(device=self._device, dtype=self._dtype)
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    def _maybe_tie_depth_embed(self) -> None:
        """Tie the depth decoder's audio embed to the backbone audio embed.

        ``tie_codebooks_embeddings=True`` (config) ties
        ``backbone_model.embed_tokens.embed_audio_tokens.weight`` and
        ``depth_decoder.model.embed_tokens.weight`` (both (32*2051, 2048)). The
        checkpoint stores only the backbone copy; mirror it onto the depth embed.
        """
        if self._frame_embed is None or self._depth_decoder is None:
            return
        if not getattr(self.config, "tie_codebooks_embeddings", True):
            return
        try:
            src = self._frame_embed.embed_audio_tokens.weight
            self._depth_decoder.model.embed_tokens.weight = src
        except AttributeError:
            logger.warning("CSM: could not tie depth embed to backbone audio embed")

    @staticmethod
    def _load_into(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
        params = dict(module.named_parameters())
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        for name, w in state.items():
            if name not in params:
                logger.warning("CSM: %s not in %s", name, type(module).__name__)
                continue
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, w)

    # ------------------------------------------------------------------
    # Dummy run support
    # ------------------------------------------------------------------

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict]:
        return [{"text": "hello", "_is_dummy": True}] * num_reqs

    # ------------------------------------------------------------------
    # Streaming generator (VoxCPM pattern)
    # ------------------------------------------------------------------

    def _create_stream_gen(self, info: dict[str, Any]):
        """Create an inference_stream() generator for a request."""
        text: str = str(_pick(info, "text", "") or "")
        if not text.strip():
            logger.warning("CSM received empty text; yielding silence.")
            sr = int(getattr(self.config, "codec_sample_rate", 24000))
            yield torch.zeros((sr,), dtype=torch.float32), True
            return

        max_new_frames: int = int(_pick(info, "max_new_frames", _DEFAULT_MAX_NEW_FRAMES))
        temperature: float = float(_pick(info, "temperature", _DEFAULT_TEMPERATURE))
        top_k: int = int(_pick(info, "top_k", _DEFAULT_TOP_K))

        yield from self.inference_stream(
            info=info,
            text=text,
            max_new_frames=max_new_frames,
            temperature=temperature,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Prefill helpers
    # ------------------------------------------------------------------

    def _embed_text_prompt(self, info: dict[str, Any]) -> torch.Tensor:
        """Embed the text prompt token ids into backbone hidden states.

        The serving layer hands the tokenized prompt via the runtime info under
        the ``prompt_token_ids`` key (CSM uses Llama text-token ids). Each text
        token is embedded with ``embed_text_tokens`` (text_vocab_size, hidden).
        Returns ``(1, T, hidden)``.
        """
        token_ids = _pick(info, "prompt_token_ids", None)
        if token_ids is None:
            token_ids = info.get("prompt_token_ids")
        if token_ids is None:
            # Fall back to a single BOS so prefill is non-empty; the GPU run
            # will route real tokenized ids here (see GPU checklist item).
            bos = int(getattr(self.config, "bos_token_id", 128000) or 128000)
            token_ids = [bos]
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self._device).view(-1)
        return self._text_embed(ids).unsqueeze(0)

    def _compose_frame_embed(self, frame_codes: torch.Tensor) -> torch.Tensor:
        """Σ over the 32 codebooks -> one backbone input embedding (A2 §2.4).

        ``frame_codes`` is ``(B, 32)`` Long. ``CsmBackboneModelEmbeddings``
        expects ``(B, T, num_codebooks)`` and sums over the codebook dim, so we
        add a length-1 frame axis. Returns ``(B, 1, hidden)`` ready to feed the
        backbone as ``inputs_embeds`` for the next position.
        """
        emb = self._frame_embed(frame_codes.unsqueeze(1))  # (B, 1, hidden)
        return emb

    def inference_stream(
        self,
        *,
        info: dict[str, Any],
        text: str,
        max_new_frames: int = _DEFAULT_MAX_NEW_FRAMES,
        temperature: float = _DEFAULT_TEMPERATURE,
        top_k: int = _DEFAULT_TOP_K,
    ):
        """Dual-AR frame generator: yields ``(waveform_chunk, is_last)``.

        Control flow (A2 §2), one iteration == one 80 ms frame:
            (prefill once) backbone(text-prompt inputs_embeds)
            per frame:
              backbone step (1 new KV position) -> cb0 logits + last hidden
              -> sample cb0
              -> 31-step depth loop (cb1..cb31)             [C2]
              -> 32-code frame
              -> EOS if cb0..cb30 all-zero (trim)           [A2 §2.4]
              -> Mimi decode -> 1920 samples                [C3]
              -> yield
        """
        device = self._device or torch.device("cpu")
        backbone = self.backbone

        # --- Prefill: embed the text prompt and run the backbone once. ---
        prompt_embeds = self._embed_text_prompt(info)  # (1, T, hidden)
        prompt_len = prompt_embeds.shape[1]
        positions = torch.arange(prompt_len, device=device)
        hidden = backbone.forward(
            input_ids=None,
            positions=positions,
            inputs_embeds=prompt_embeds.squeeze(0),
        )
        # vLLM LlamaModel returns (sum_tokens, hidden); the last row is the
        # next-token hidden state for our single (B=1) prefill sequence.
        last_hidden = hidden[-1:].to(self._dtype)  # (1, hidden)
        next_pos = prompt_len

        for _frame_idx in range(max_new_frames):
            # --- 1. cb0 from the backbone last hidden state. ---
            cb0_logits = backbone.cb0_head(last_hidden)  # (1, vocab)
            cb0 = _sample_logits(cb0_logits, temperature, top_k)  # (1,)

            # --- 2. 31-step depth loop -> cb1..cb31; assemble 32-code frame. ---
            frame_codes = self._run_depth_loop(
                cb0=cb0,
                backbone_last_hidden_state=last_hidden,
                temperature=temperature,
                top_k=top_k,
            )  # (1, 32) Long

            # --- 3. EOS: a frame with cb0..cb30 all-zero is end-of-stream. ---
            is_eos = bool(
                (frame_codes[0, : self.num_codebooks - 1] == _CODEBOOK_EOS_ID).all().item()
            )
            if is_eos:
                # Trim: do not emit audio for the terminal all-zero frame.
                yield torch.zeros((0,), dtype=torch.float32, device=device), True
                return

            # --- 4. Mimi decode: 32-code frame -> 1920 PCM samples. ---
            waveform_chunk = self._mimi_decode(frame_codes)  # (1920,) fp32

            # --- 5. Feed the frame back as the next backbone position. ---
            frame_embed = self._compose_frame_embed(frame_codes)  # (1, 1, hidden)
            step_pos = torch.arange(next_pos, next_pos + 1, device=device)
            hidden = backbone.forward(
                input_ids=None,
                positions=step_pos,
                inputs_embeds=frame_embed.squeeze(0),
            )
            last_hidden = hidden[-1:].to(self._dtype)
            next_pos += 1

            is_last = _frame_idx == max_new_frames - 1
            yield waveform_chunk, is_last

        # max_new_frames reached without EOS.
        yield torch.zeros((0,), dtype=torch.float32, device=device), True

    def _run_depth_loop(
        self,
        *,
        cb0: torch.Tensor,
        backbone_last_hidden_state: torch.Tensor,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        """31-step inner depth-decoder AR loop -> a 32-code frame (A2 §2.2).

        The depth decoder (4 layers / d1024 / head_dim128) runs as custom dense
        torch with its own DynamicCache, reset every frame (33 positions: the
        backbone hidden at position 0, then cb0..cb31 at positions 1..32). It is
        the dominant per-frame cost and the dominant difficulty.

        Numerics hygiene (PHASE3 §1): depth logits are cast to fp32 before EVERY
        sample (inside ``_sample_logits``); the embed/head dtype is pinned to the
        backbone dtype (set in ``_build_aux_modules``). Codes/flags are collected
        into a per-step packed staging tensor and one D2H copy at the end of the
        loop — NO per-step ``.item()`` / ``.cpu()`` scalar syncs (I3 / R0 §10).

        SAFE path: per-lane B=1 (the correctness anchor). The signature and the
        ``backbone_last_hidden_state`` / ``cb0`` shapes are kept ``(B, *)`` so a
        future slot-indexed depth-batch across lanes (R0 §1) can be flipped on
        by stacking lanes into the batch dim — mirroring sgl-omni's batched-depth
        shape — without changing the loop body.

        Returns ``(B, 32)`` Long: cb0 from the backbone plus cb1..cb31.
        """
        from transformers.cache_utils import DynamicCache

        depth = self._depth_decoder
        device = cb0.device
        bsz = cb0.shape[0]
        n_steps = self.num_codebooks - 1  # 31

        # Packed staging for codes (I3 / R0 §10): one (B, 32) tensor filled on
        # device, with a single D2H at the end — no per-step host sync.
        codes = torch.empty((bsz, self.num_codebooks), dtype=torch.long, device=device)
        codes[:, 0] = cb0

        past = DynamicCache(config=depth.config)
        # Position 0 of the depth sequence is the backbone hidden state; the
        # first depth input token is cb0. Pad a placeholder at position 0 that
        # the depth model overwrites with ``backbone_last_hidden_state``.
        cur_input = cb0.view(bsz, 1)  # (B, 1) = cb0
        backbone_hs = backbone_last_hidden_state.view(bsz, self.hidden_size).to(self._dtype)

        for step in range(n_steps):
            # input_ids carries the previously sampled codebook token; on the
            # first step we also pass backbone_last_hidden_state so the depth
            # model seeds position 0 from the backbone hidden state.
            depth_ids = cur_input
            if step == 0:
                # Prepend the position-0 placeholder (overwritten internally by
                # backbone_last_hidden_state). Sequence is [hidden(pos0), cb0].
                depth_ids = torch.nn.functional.pad(cur_input, (1, 0), value=0)  # (B,2)
                out = depth(
                    input_ids=depth_ids,
                    backbone_last_hidden_state=backbone_hs,
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                )
            else:
                out = depth(
                    input_ids=depth_ids,
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                )
            past = out.past_key_values
            # logits_to_keep=1 -> codebooks_head returns (B, 1, vocab) for the
            # last position, i.e. the logits for codebook (step+1). The fp32
            # cast happens inside _sample_logits (PHASE3 §1).
            step_logits = out.logits[:, -1, :]  # (B, vocab)
            next_code = _sample_logits(step_logits, temperature, top_k)  # (B,)
            codes[:, step + 1] = next_code
            cur_input = next_code.view(bsz, 1)

        return codes

    def _mimi_decode(self, frame_codes: torch.Tensor) -> torch.Tensor:
        """Mimi codec decode: a 32-code frame -> 1920 PCM samples (A2 §2.3).

        ``frame_codes`` is ``(B, 32)`` Long. Mimi.decode expects
        ``(B, num_quantizers, codes_length)``; we decode one frame at a time
        (codes_length == 1) and return a 1-D ``(1920,)`` fp32 waveform for B=1.

        Reserved guard: the per-codebook vocab is 2051 with reserved ids
        2048/2049/2050 that must NEVER reach Mimi (its real codebook size is
        2048). Clamp them to 0 before decode so the codec only ever sees valid
        codebook entries.
        """
        # Clamp on a COPY: the reserved ids 2048/2049/2050 are valid inputs to
        # the (65632-entry) frame-embed table that feeds the next backbone step,
        # so we must NOT mutate frame_codes in place — only the Mimi-bound copy
        # is clamped to the codec's real [0, 2047] codebook range.
        codes = frame_codes.clamp(min=0, max=_MIMI_CODEBOOK_SIZE - 1)
        # (B, 32) -> (B, 32, 1): num_quantizers=32, codes_length=1.
        audio_codes = codes.unsqueeze(-1)
        with torch.no_grad():
            out = self._mimi_codec.decode(audio_codes)
        audio_values = out.audio_values  # (B, channels, samples) or (B, samples)
        wav = audio_values.reshape(audio_values.shape[0], -1)[0]
        return wav.to(torch.float32)

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
