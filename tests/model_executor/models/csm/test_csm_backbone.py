# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU correctness tests for the CSM-1B Stage-0 backbone control logic.

These lock the three places the 2-stage redesign is subtle, all reachable on
CPU by building the model via ``object.__new__`` + stubbing the vLLM-native
backbone / depth wrapper:

  * ``compute_logits`` EOS row-mapping (the "unknown #2" fix): EOS is forced
    POSITIONALLY on ``_eos_flags_by_row`` rows, never by dict insertion order.
  * ``compute_logits`` single sampling authority: every live row is a one-hot
    echo of the cb0 ``forward()`` committed, so the ENGINE sampler (any
    temperature/top_k/seed) reproduces the model's decision and the scheduler
    stop (token id 0) can only fire on a model-latched EOS/cap row.
  * ``forward`` EOS / GATE-B frame-cap emit policy: a natural all-zero EOS frame
    is DROPPED (empty latent) while a cap-forced stop KEEPS its real audio frame;
    both latch the row so the scheduler stops.
  * I5 per-request state: ``preprocess`` decode re-injects the cached Sigma-embed,
    and ``on_requests_finished`` frees every per-request dict.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.csm.csm_backbone import (
    _CB0_ZERO_SHADOW_ID,
    _CODEBOOK_EOS_ID,
    CsmBackboneForConditionalGeneration,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_VOCAB = 8
_HIDDEN = 8


class _FakeLogitsProcessor:
    """Deterministic per-row logits so EOS overrides are checkable. Row i gets a
    strictly-increasing ramp offset by i, so its natural argmax is the last id
    (never id 0), making a forced-EOS (argmax -> 0) unambiguous."""

    def __call__(self, head, hidden):
        n = int(hidden.shape[0])
        return torch.arange(n * _VOCAB, dtype=torch.float32).reshape(n, _VOCAB)


def _make_backbone() -> CsmBackboneForConditionalGeneration:
    m = object.__new__(CsmBackboneForConditionalGeneration)
    nn.Module.__init__(m)
    m.num_codebooks = 32
    m.hidden_size = _HIDDEN
    m._backbone_dtype = torch.float32
    m._dtype = torch.float32
    m._eos_flags_by_row = []
    m._cb0_by_row = []
    m._eos_by_req = {}
    m._cached_sigma_by_req = {}
    m._sampling_by_req = {}
    m._generator_by_req = {}
    m._max_frames_by_req = {}
    m._frames_emitted_by_req = {}
    m.backbone = SimpleNamespace(logits_processor=_FakeLogitsProcessor(), cb0_head=object())
    return m


# --------------------------------------------------------------------------
# compute_logits: positional EOS row mapping
# --------------------------------------------------------------------------


def test_compute_logits_forces_eos_on_flagged_rows_only():
    m = _make_backbone()
    m._eos_flags_by_row = [False, True, False]
    logits = m.compute_logits(torch.randn(3, _HIDDEN))

    assert logits.shape == (3, _VOCAB)
    # Flagged row 1 -> all -inf except the EOS id, set to a huge positive.
    assert int(logits[1].argmax()) == _CODEBOOK_EOS_ID
    assert logits[1, _CODEBOOK_EOS_ID].item() == pytest.approx(1.0e6)
    others = torch.cat([logits[1, :_CODEBOOK_EOS_ID], logits[1, _CODEBOOK_EOS_ID + 1 :]])
    assert torch.isneginf(others).all()
    # Unflagged rows with no latched cb0 keep the raw (finite) ramp except the
    # stop id, which is masked so the engine can never stop on them.
    for row in (0, 2):
        assert torch.isneginf(logits[row, _CODEBOOK_EOS_ID])
        non_stop = torch.cat([logits[row, :_CODEBOOK_EOS_ID], logits[row, _CODEBOOK_EOS_ID + 1 :]])
        assert torch.isfinite(non_stop).all()
        assert int(logits[row].argmax()) != _CODEBOOK_EOS_ID


def test_compute_logits_none_hidden_returns_none():
    assert _make_backbone().compute_logits(None) is None


def test_compute_logits_unsqueezes_1d_hidden():
    out = _make_backbone().compute_logits(torch.randn(_HIDDEN))
    assert out.shape == (1, _VOCAB)


def test_compute_logits_accepts_omni_output():
    m = _make_backbone()
    m._eos_flags_by_row = [False, False]
    oo = OmniOutput(text_hidden_states=torch.randn(2, _HIDDEN), multimodal_outputs=None)
    assert m.compute_logits(oo).shape == (2, _VOCAB)


def test_compute_logits_tolerates_flags_shorter_than_batch():
    # Stale/short flag list must not crash; rows beyond the lists get only
    # the defensive stop-id mask, never a forced stop.
    m = _make_backbone()
    m._eos_flags_by_row = [True]  # only row 0
    logits = m.compute_logits(torch.randn(3, _HIDDEN))
    assert int(logits[0].argmax()) == _CODEBOOK_EOS_ID
    for row in (1, 2):
        assert torch.isneginf(logits[row, _CODEBOOK_EOS_ID])
        non_stop = torch.cat([logits[row, :_CODEBOOK_EOS_ID], logits[row, _CODEBOOK_EOS_ID + 1 :]])
        assert torch.isfinite(non_stop).all()
        assert int(logits[row].argmax()) != _CODEBOOK_EOS_ID


# --------------------------------------------------------------------------
# compute_logits: single sampling authority (one-hot echo of committed cb0)
# --------------------------------------------------------------------------


def _engine_sample(logits: torch.Tensor, temperature: float, top_k: int, seed: int) -> int:
    """Reference engine-side sampler for row 0 (vLLM semantics: temperature
    scaling, top-k mask, softmax, seeded multinomial; temperature 0 is
    argmax). Derived from the sampler contract, not from the model code."""
    row = logits[0:1]
    if temperature <= 0.0:
        return int(row.argmax(dim=-1)[0])
    row = row / temperature
    if top_k and 0 < top_k < row.shape[-1]:
        kth = torch.topk(row, top_k, dim=-1).values[..., -1, None]
        row = torch.where(row < kth, torch.full_like(row, float("-inf")), row)
    probs = torch.softmax(row, dim=-1)
    gen = torch.Generator().manual_seed(seed)
    return int(torch.multinomial(probs, num_samples=1, generator=gen)[0, 0])


def test_compute_logits_echoes_committed_cb0_positionally():
    m = _make_backbone()
    m._eos_flags_by_row = [False, True, False]
    m._cb0_by_row = [5, 4, 0]
    logits = m.compute_logits(torch.randn(3, _HIDDEN))
    # Row 0: one-hot echo of the committed cb0 (5).
    assert int(logits[0].argmax()) == 5
    # Row 1: model-latched EOS wins over the echo -> the stop id.
    assert int(logits[1].argmax()) == _CODEBOOK_EOS_ID
    # Row 2: committed cb0==0 on a NON-EOS frame is remapped to the shadow id
    # so the scheduler stop (token 0) cannot fire on real audio.
    assert int(logits[2].argmax()) == _CB0_ZERO_SHADOW_ID
    assert _CB0_ZERO_SHADOW_ID != _CODEBOOK_EOS_ID
    # Every live row is fully forced: exactly one finite (positive) entry.
    for row in range(3):
        assert int(torch.isfinite(logits[row]).sum()) == 1


def test_engine_sample_reproduces_committed_cb0_at_any_temperature(monkeypatch):
    """The engine's sampled token equals the model-committed cb0 under greedy
    AND under the served stochastic defaults (temperature 0.9 / top_k 50),
    for any seed: the one-hot leaves the sampler no other choice."""
    frame = torch.full((1, 32), 5, dtype=torch.long)  # committed cb0 == 5
    m, out = _drive_forward(monkeypatch, frame, cap=100)
    assert m._cb0_by_row == [5]
    logits = m.compute_logits(out)
    for temperature, top_k, seed in [(0.0, 0, 0), (0.9, 50, 7), (0.9, 50, 8), (2.0, 0, 123)]:
        assert _engine_sample(logits, temperature, top_k, seed) == 5


def test_engine_stop_fires_only_on_model_eos_rows(monkeypatch):
    # (a) Non-EOS frame whose committed cb0 is 0 (legit audio: deeper
    # codebooks nonzero): the engine must NEVER draw the stop id.
    frame = torch.zeros(1, 32, dtype=torch.long)
    frame[0, 5] = 3  # nonzero cb5 -> not the all-zero EOS frame
    m, out = _drive_forward(monkeypatch, frame, cap=100)
    assert m._eos_flags_by_row == [False]
    logits = m.compute_logits(out)
    for seed in range(5):
        assert _engine_sample(logits, 0.9, 50, seed) != _CODEBOOK_EOS_ID

    # (b) Natural all-zero EOS frame: the engine draws the stop id at any
    # temperature/seed.
    eos_frame = torch.zeros(1, 32, dtype=torch.long)
    m, out = _drive_forward(monkeypatch, eos_frame, cap=100)
    logits = m.compute_logits(out)
    for temperature, seed in [(0.0, 0), (0.9, 7), (0.9, 8)]:
        assert _engine_sample(logits, temperature, 50, seed) == _CODEBOOK_EOS_ID

    # (c) Cap-forced stop (real audio frame at the GATE-B cap): stop id too.
    cap_frame = torch.full((1, 32), 6, dtype=torch.long)
    m, out = _drive_forward(monkeypatch, cap_frame, cap=1, emitted=0)
    logits = m.compute_logits(out)
    assert _engine_sample(logits, 0.9, 50, 3) == _CODEBOOK_EOS_ID


# --------------------------------------------------------------------------
# forward: EOS / GATE-B frame-cap emit policy
# --------------------------------------------------------------------------


def _drive_forward(monkeypatch, frame, *, cap, emitted=0, info=None):
    m = _make_backbone()
    m._sampling_by_req = {"r0": (0.0, 0)}  # greedy -> deterministic cb0
    m._max_frames_by_req = {"r0": cap}
    m._frames_emitted_by_req = {"r0": emitted}
    m.backbone.forward = lambda **kw: torch.randn(1, _HIDDEN)
    m.depth = SimpleNamespace(run=lambda **kw: frame)
    m._compose_frame_embed = lambda fc: torch.zeros(1, _HIDDEN)
    monkeypatch.setattr(
        "vllm.forward_context.get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    out = m.forward(
        input_ids=torch.tensor([5], dtype=torch.long),
        positions=torch.tensor([0]),
        inputs_embeds=torch.zeros(1, _HIDDEN),
        runtime_additional_information=[info if info is not None else {"request_id": "r0"}],
    )
    return m, out


def test_forward_natural_eos_drops_frame_and_flags_row(monkeypatch):
    frame = torch.zeros(1, 32, dtype=torch.long)  # cb0..cb30 == 0 -> natural EOS
    m, out = _drive_forward(monkeypatch, frame, cap=100)
    codes = out.multimodal_outputs["codes"]["audio"]
    assert codes[0].shape == (0, 32)  # all-zero EOS frame is dropped
    assert m._eos_flags_by_row == [True]
    assert m._eos_by_req["r0"] is True


def test_forward_eos_ignores_codebook31(monkeypatch):
    # EOS is cb0..cb30 all-zero; a nonzero cb31 must NOT defeat the EOS check.
    frame = torch.zeros(1, 32, dtype=torch.long)
    frame[0, 31] = 9
    m, out = _drive_forward(monkeypatch, frame, cap=100)
    assert m._eos_flags_by_row == [True]
    assert out.multimodal_outputs["codes"]["audio"][0].shape == (0, 32)


def test_forward_cap_forced_keeps_real_frame_and_flags_row(monkeypatch):
    frame = torch.ones(1, 32, dtype=torch.long)  # real audio, no natural EOS
    m, out = _drive_forward(monkeypatch, frame, cap=1, emitted=0)
    codes = out.multimodal_outputs["codes"]["audio"]
    assert codes[0].shape == (1, 32)  # cap-forced stop KEEPS the final frame
    assert torch.equal(codes[0], frame)
    assert m._eos_flags_by_row == [True]  # but still stops the scheduler
    assert m._frames_emitted_by_req["r0"] == 1


def test_forward_normal_frame_is_kept_and_not_flagged(monkeypatch):
    frame = torch.ones(1, 32, dtype=torch.long)
    m, out = _drive_forward(monkeypatch, frame, cap=100, emitted=0)
    codes = out.multimodal_outputs["codes"]["audio"]
    assert codes[0].shape == (1, 32)
    assert m._eos_flags_by_row == [False]
    assert m._eos_by_req["r0"] is False
    # Sigma cached for the next step's feedback (I5), frame counter advanced.
    assert "r0" in m._cached_sigma_by_req
    assert m._frames_emitted_by_req["r0"] == 1


def test_forward_returns_omni_output_with_codes_latent(monkeypatch):
    m, out = _drive_forward(monkeypatch, torch.ones(1, 32, dtype=torch.long), cap=100)
    assert isinstance(out, OmniOutput)
    assert "codes" in out.multimodal_outputs
    assert "audio" in out.multimodal_outputs["codes"]


def test_forward_skips_frame_emission_for_intermediate_prefill_chunks(monkeypatch):
    """Chunked prefill: a span that does NOT complete the prompt must emit no
    frame -- HF runs the depth decoder only on the final prompt position, so
    a mid-prompt chunk's hidden row is not a frame step. The skipped chunk
    must leave the GATE-B counter untouched, cache no Sigma, stay un-latched,
    and be un-stoppable by the engine; the prompt-completing chunk then emits
    frame 0 normally."""
    frame = torch.full((1, 32), 5, dtype=torch.long)

    # Intermediate chunk of a 6-token prompt: computed 0, span 1 -> 1 < 6.
    mid_chunk = {"request_id": "r0", "_omni_num_computed_tokens": 0, "_omni_prompt_len": 6}
    m, out = _drive_forward(monkeypatch, frame, cap=100, info=mid_chunk)
    assert out.multimodal_outputs["codes"]["audio"][0] is None  # no frame shipped
    assert m._frames_emitted_by_req["r0"] == 0  # cap accounting untouched
    assert "r0" not in m._cached_sigma_by_req  # no Sigma overwrite
    assert m._eos_flags_by_row == [False]
    assert m._cb0_by_row == [None]
    logits = m.compute_logits(out)
    # Un-latched row: the stop id is masked so the scheduler can never
    # finish the request on a chunk the model produced no frame for.
    assert torch.isneginf(logits[0, _CODEBOOK_EOS_ID])
    assert int(logits[0].argmax()) != _CODEBOOK_EOS_ID

    # Prompt-completing chunk (computed 5 of 6, span 1): frame 0 is emitted.
    final_chunk = {"request_id": "r0", "_omni_num_computed_tokens": 5, "_omni_prompt_len": 6}
    m, out = _drive_forward(monkeypatch, frame, cap=100, info=final_chunk)
    assert out.multimodal_outputs["codes"]["audio"][0].shape == (1, 32)
    assert m._frames_emitted_by_req["r0"] == 1
    assert "r0" in m._cached_sigma_by_req
    assert m._cb0_by_row == [5]


# --------------------------------------------------------------------------
# I5 per-request state: preprocess Sigma re-inject + cleanup
# --------------------------------------------------------------------------


def test_preprocess_decode_uses_cached_sigma_alone():
    """REGRESSION GUARD (served-path fidelity): the decode-step backbone input is
    the FULL previous-frame Sigma-embed ALONE, not ``base_cb0_embed + cached``.

    HF ``CsmBackboneModelEmbeddings.forward`` builds a position's input as
    ``embed_audio_tokens(codes + audio_tokens_offsets).sum(dim=2)`` over all 32
    codebooks, and during AR generation that position's codes ARE the previously
    generated frame. The cached Sigma is exactly that sum (cb0 is already its
    k=0 term). Adding a separate cb0 base embed on top double-counts codebook 0
    every decode step, compounds through the KV cache, and derails the rollout
    (audio degrades after the first word, runs past EOS, over-loud). So the
    composed decode embedding MUST equal the cached Sigma (5.0), NOT
    base(2.0)+cached(5.0)=7.0."""
    m = _make_backbone()
    m.config = SimpleNamespace(vocab_size=2051)
    m._compose_frame_embed = lambda fc: torch.full((1, _HIDDEN), 2.0)
    m._cached_sigma_by_req = {"r0": torch.full((1, _HIDDEN), 5.0)}
    ids, embeds, upd = m.preprocess(
        input_ids=torch.tensor([7], dtype=torch.long),
        input_embeds=None,
        request_id="r0",
        _omni_is_prefill=False,
    )
    assert torch.equal(ids, torch.tensor([7]))
    # Cached Sigma (5.0) ALONE. If this is 7.0 again, cb0 is being double-counted.
    torch.testing.assert_close(embeds, torch.full((1, _HIDDEN), 5.0))
    assert upd == {}


def test_preprocess_decode_without_cache_uses_base_embed_only():
    m = _make_backbone()
    m.config = SimpleNamespace(vocab_size=2051)
    m._compose_frame_embed = lambda fc: torch.full((1, _HIDDEN), 2.0)
    m._cached_sigma_by_req = {}
    _, embeds, _ = m.preprocess(
        input_ids=torch.tensor([7], dtype=torch.long),
        input_embeds=None,
        request_id="r0",
        _omni_is_prefill=False,
    )
    torch.testing.assert_close(embeds, torch.full((1, _HIDDEN), 2.0))


def test_preprocess_prefill_returns_text_prompt_span():
    m = _make_backbone()
    m.config = SimpleNamespace(vocab_size=2051)
    prompt = torch.arange(3 * _HIDDEN, dtype=torch.float32).reshape(3, _HIDDEN)
    m._embed_text_prompt = lambda info, device: prompt
    ids, embeds, upd = m.preprocess(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        input_embeds=None,
        request_id="rp",
        _omni_is_prefill=True,
        _omni_num_computed_tokens=0,
        _omni_prompt_len=3,
    )
    assert embeds.shape == (3, _HIDDEN)
    torch.testing.assert_close(embeds, prompt)
    assert upd == {}


def test_on_requests_finished_frees_all_per_request_state():
    m = _make_backbone()
    m._cached_sigma_by_req = {"r": torch.zeros(1)}
    m._eos_by_req = {"r": True}
    m._sampling_by_req = {"r": (0.9, 50)}
    m._generator_by_req = {"r": torch.Generator()}
    m._max_frames_by_req = {"r": 64}
    m._frames_emitted_by_req = {"r": 5}
    m.on_requests_finished(["r"])
    assert m._cached_sigma_by_req == {}
    assert m._eos_by_req == {}
    assert m._sampling_by_req == {}
    assert m._generator_by_req == {}
    assert m._max_frames_by_req == {}
    assert m._frames_emitted_by_req == {}


# --------------------------------------------------------------------------
# request.seed -> per-request generator (in-model sampling determinism)
# --------------------------------------------------------------------------


def test_preprocess_seed_creates_per_request_generator_once():
    m = _make_backbone()
    m.config = SimpleNamespace(vocab_size=2051)
    m._compose_frame_embed = lambda fc: torch.zeros(1, _HIDDEN)
    m.preprocess(
        input_ids=torch.tensor([7], dtype=torch.long),
        input_embeds=None,
        request_id="rs",
        _omni_is_prefill=False,
        seed=[1234],
        temperature=[0.9],
        top_k=[50],
    )
    assert "rs" in m._generator_by_req
    gen_first = m._generator_by_req["rs"]
    # Later steps of the same request must keep the SAME generator (its state
    # is the request's RNG stream; recreating it would replay draws).
    m.preprocess(
        input_ids=torch.tensor([7], dtype=torch.long),
        input_embeds=None,
        request_id="rs",
        _omni_is_prefill=False,
        seed=[1234],
        temperature=[0.9],
        top_k=[50],
    )
    assert m._generator_by_req["rs"] is gen_first


def test_preprocess_without_seed_keeps_default_rng_path():
    m = _make_backbone()
    m.config = SimpleNamespace(vocab_size=2051)
    m._compose_frame_embed = lambda fc: torch.zeros(1, _HIDDEN)
    m.preprocess(
        input_ids=torch.tensor([7], dtype=torch.long),
        input_embeds=None,
        request_id="ru",
        _omni_is_prefill=False,
        temperature=[0.9],
        top_k=[50],
    )
    assert "ru" not in m._generator_by_req  # forward() then passes generator=None


def test_preprocess_different_requests_get_independent_generators():
    m = _make_backbone()
    m.config = SimpleNamespace(vocab_size=2051)
    m._compose_frame_embed = lambda fc: torch.zeros(1, _HIDDEN)
    for req, seed in (("a", 1), ("b", 2)):
        m.preprocess(
            input_ids=torch.tensor([7], dtype=torch.long),
            input_embeds=None,
            request_id=req,
            _omni_is_prefill=False,
            seed=[seed],
            temperature=[0.9],
            top_k=[50],
        )
    assert m._generator_by_req["a"] is not m._generator_by_req["b"]
    # Independent streams: seeded differently, the initial states differ.
    assert not torch.equal(m._generator_by_req["a"].get_state(), m._generator_by_req["b"].get_state())


def test_forward_threads_seeded_generator_into_cb0_and_depth(monkeypatch):
    """The per-request generator must govern BOTH sampling sites: the cb0 draw
    (sample_logits) and the 31 depth-step draws (depth.run)."""
    import vllm_omni.model_executor.models.csm.csm_backbone as bb

    m = _make_backbone()
    gen = torch.Generator().manual_seed(77)
    m._sampling_by_req = {"r0": (0.9, 50)}
    m._generator_by_req = {"r0": gen}
    m._max_frames_by_req = {"r0": 100}
    m._frames_emitted_by_req = {"r0": 0}
    m.backbone.forward = lambda **kw: torch.randn(1, _HIDDEN)
    m._compose_frame_embed = lambda fc: torch.zeros(1, _HIDDEN)

    cb0_generators = []
    monkeypatch.setattr(
        bb,
        "sample_logits",
        lambda logits, temperature, top_k, generator=None: (
            cb0_generators.append(generator),
            torch.zeros(1, dtype=torch.long),
        )[1],
    )
    depth_kwargs = {}
    m.depth = SimpleNamespace(
        run=lambda **kw: (depth_kwargs.update(kw), torch.ones(1, 32, dtype=torch.long))[1]
    )
    monkeypatch.setattr(
        "vllm.forward_context.get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    m.forward(
        input_ids=torch.tensor([5], dtype=torch.long),
        positions=torch.tensor([0]),
        inputs_embeds=torch.zeros(1, _HIDDEN),
        runtime_additional_information=[{"request_id": "r0"}],
    )
    assert cb0_generators == [gen]
    assert depth_kwargs["generator"] is gen
