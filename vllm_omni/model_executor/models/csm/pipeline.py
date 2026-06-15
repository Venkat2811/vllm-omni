# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CSM-1B pipeline topology (frozen).

**Single-stage** AR TTS scaffold: text -> speech waveform in one pass. This
mirrors the MOSS-TTS-Nano single-stage shape — the Llama-style backbone, the
nested depth decoder, and the Mimi codec all run inside
``CsmForGeneration.forward()``, driven by the VoxCPM-style generator pattern
(``inference_stream()`` stored per-request; one audio chunk yielded per
forward) through the AR scheduler.

CSM-1B is dual-autoregressive (a backbone that emits codebook-0 plus a 31-step
depth decoder that emits the remaining 31 RVQ codebooks per 80 ms frame); a
later iteration may split the Mimi codec onto its own ``LLM_GENERATION`` stage
(the canonical 2-stage talker -> code2wav shape used by qwen3_tts /
higgs_audio_v2). For C1 we keep it single-stage so the model imports and
registers cleanly while the depth loop (C2) and the Mimi decode (C3) are
filled in.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

CSM_PIPELINE = PipelineConfig(
    model_type="csm",
    model_arch="CsmForCausalLM",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="csm",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            sampling_constraints={
                "detokenize": False,
                # compute_logits() forces EOS (token id 0) when the last
                # streaming chunk is yielded; keep a hard backstop here.
                # CSM frame-level EOS is "cb0..cb30 all-zero" (A2 §2.4); this
                # token-id stop is the scheduler-visible backstop only.
                "stop_token_ids": [0],
            },
        ),
    ),
)
