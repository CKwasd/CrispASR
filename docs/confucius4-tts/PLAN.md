# Confucius4-TTS — port plan (§377)

**Issue:** #377 · **License:** Apache 2.0 · **Languages:** 14 (zh, en, ja, ko, de, fr, es, id, it, th, pt, ru, ms, vi)
**Source:** [netease-youdao/Confucius4-TTS](https://github.com/netease-youdao/Confucius4-TTS) ·
**Weights:** [HF repo](https://huggingface.co/netease-youdao/Confucius4-TTS) (3.06 GB total)

---

## NOW — active work

**Branch:** `feat/confucius4-cfg` (worktree `.claude/worktrees/feat-confucius4-cfg`)
**Status:** four S2A/T2S correctness bugs found by re-reading the Python blueprint
line-by-line; fixes written and compile-clean, end-to-end run pending (the VPS is
heavily contended, the full build is queued).

### Bugs found and fixed on this branch

1. **Timestep embedding was wrong twice** (`s2a_sinusoidal_embed`) — the reference
   `SinusPositionEmbedding.forward` applies `scale=1000` to `t` and concatenates
   **cos then sin**; the port had no scale and sin-then-cos.  Measured against the
   reference module: old cos **0.13–0.18** (essentially uncorrelated), and across
   the 25 ODE steps the old embedding's min pairwise cosine was **0.972** — the
   estimator was seeing an almost constant timestep signal, which by itself
   prevents flow matching from working.  Fixed version is exact (cos 1.0000).
   Unit check: `tools/…/tstep_unit.py` (scratch).

2. **The length regulator was skipped entirely** (`s2a_build_conditioning`).  The
   handover assumed `sampling_ratios=[1,1,1,1]` made it an identity, but the
   ratios only control *how many* conv blocks exist — `InterpolateRegulator` is
   `content_in_proj(1024→512)` → nearest-interpolate → 4×[Conv1d k=3 + GroupNorm(1)
   + Mish] → Conv1d k=1, all of it learned and all of it present in the GGUF
   (`length_regulator.*`, 274 tensors).  Also `encoder_proj` is Linear(2304,
   **1024**), not 512, so the old code truncated its output to the first 512 dims
   and fed that straight to `mu_projection`.  Now ported in full; verified against
   torch's real module at **cos 1.0000000000** with f32 weights (max_abs_diff
   8.6e-08) and cos 0.99898 through the Q4_K GGUF (pure quantization).
   Legacy path kept behind `CRISPASR_CONFUCIUS4_LR_LEGACY=1`.

3. **CFG implemented** (`s2a_flow_matching`) — second pass with mu, reference mel
   and speaker embedding zeroed, blended `v = (1+cfg)·v_cond − cfg·v_uncond`,
   matching `solve_euler`.  `cfg_rate` defaults to 0.7 and `ode_steps` to 25 (the
   reference values); a **negative** `cfg_rate` disables CFG, so a zero-initialised
   params struct still gets the reference default.  Env override
   `CRISPASR_CONFUCIUS4_CFG_RATE` for A/B without recompiling.

4. **The CLI forced greedy T2S decoding** — `crispasr_backend_confucius4_tts.cpp`
   did `cp.temperature = p.temperature`, and `whisper_params.temperature` defaults
   to **0.0**, overwriting the backend's reference default of 0.8.  The reference
   runs `do_sample=True, temperature=0.8`.  This fits the earlier symptom of the
   decode running to the 1520-token cap without ever emitting EOS.  Now only
   overridden when the user actually passes a temperature.

Also corrected: the S2A noise temperature was reusing `params.temperature` (the
T2S *sampling* knob); the reference passes 1.0 to the CFM decoder.  Env override
`CRISPASR_CONFUCIUS4_S2A_TEMP`.

### Parity harness (new)

`CRISPASR_CONFUCIUS4_DUMP_S2A=<dir>` dumps `semantic_codes`, `lm_latent`,
`z_init`, `cond` and `mel` as raw f32/i32 with a `shapes.txt` manifest, so the
real PyTorch S2A can be driven on **identical** inputs and noise and compared
per stage (`s2a_parity.py`).  This is the acceptance gate for the ODE, alongside
the TTS→ASR roundtrip.

### Verified as already correct (read against the Python, no change needed)

DiT attention (RoPE NORMAL/adjacent-pair, base 10000, scale 1/√head_dim), AdaLN
weight/bias split order, FinalLayer's opposite shift/scale order, SwiGLU, U-Net
skip emit/receive sets ({0..5} / {7..12}), `skip_linear(cat[x_res, x_mel])`,
WaveNet res/skip halves and dilation=1/pad=2, InputEmbedding concat order
(x, cond, mu_proj, spks), `T_mel = int(T_sem × 1.72)`, and the `lm_latent`
alignment (the port collects one extra trailing row, which the conditioning
correctly ignores).

### Kaggle run 1 (kernel `chr1s4/crispasr-confucius4-cfg-verify` v1, 8 min)

The whole pipeline now runs on a Kaggle CPU box in ~45 s of compute:

```
TTS rc=0
confucius4: EOS at step 131            <-- was: ran to the 1520 cap, never EOS
confucius4: generated 131 semantic codes
confucius4: S2A flow-matching: T_sem=131 -> T_mel=225, mel_dim=80, 25 ODE steps, cfg=0.70
confucius4: S2A conditioning: embed+project+regulator OK (2304->1024->512 dims)
confucius4: S2A time schedule: linear
confucius4: DiT graph: depth=13 dim=512 heads=8 inter=1536 mel=80 T=225
confucius4: BigVGAN: 225 mel frames -> 57600 PCM samples @ 22050 Hz (2.61s)
```

**EOS at step 131 is the headline.** The decode terminating on its own (131
codes ~= 2.6 s, right for the test sentence) confirms the greedy-decode and
wrong-prompt bugs were real and are fixed. Every fix marker reads as expected.

**But the ASR roundtrip still fails: `(electronic music)`, 0/8 word overlap.**
The audio is not speech yet, so the port is NOT done.

Two harness problems that run 1 exposed, both fixed for run 2:
- `s2a_parity.py` crashed before printing any cosine -- it fed the whole dumped
  `lm_latent` (T_sem + 1 rows) to `encoder_proj`; slice to T_sem.
- The run could not separate "the port is still wrong" from "zero speaker
  conditioning cannot produce speech". Run 2 vocodes BOTH the C++ mel and the
  reference mel with the real BigVGAN and ASRs each.

### Next

- Read run 2's per-stage cosines and the cpp-vs-REF ASR comparison. If the
  REFERENCE mel is also unintelligible on identical inputs, the S2A port is not
  the blocker and speaker conditioning is the whole remaining story.
- Speaker conditioning is still entirely absent: `spks`, the reference mel and
  the T2S `condition_emb` are all zero, and this model is zero-shot -- the
  reference implementation always has a prompt wav. Note `speaker_encoder(0)`
  is NOT zero (the ECAPA-TDNN has biases), so even the "no speaker" path is not
  faithfully reproduced by a literal zero vector.
- Still open from the handover: re-run the converter for the baked BPE vocab,
  and the BigVGAN perf work.
- Note: `max_semantic_tokens` is treated as a *new-token* count, while the
  reference's `max_length=1520` is the **total** sequence length.

## Architecture (read from Python, 2026-08-22)

Two-stage TTS with external conditioning models:

### Pipeline overview

```
text + prompt_wav
  │
  ├── [Wav2Vec2-BERT 2.0] → semantic_features (1, T_feat, 1024)   # layer 17, z-normalised
  ├── [CAMPPlus]           → style_embedding   (1, 192)            # ECAPA-TDNN speaker encoder
  ├── [Tokenizer]          → text_token_ids    (1, T_text)         # SentencePiece, vocab 32000
  │
  ▼
  [T2S: GPT-2 causal LM]  → semantic_codes (1, T_sem) + lm_latent (1, T_sem, 1280)
  │
  ▼
  [S2A: Flow-matching DiT] → mel (1, 80, T_mel)
  │
  ▼
  [BigVGAN vocoder]        → waveform @ 22050 Hz
```

### T2S model (Text-to-Semantic) — `t2s_model.safetensors` (2.64 GB)

- **Architecture:** GPT-2 backbone (HF `GPT2Model`)
  - 24 layers, d_model=1280, 20 heads, vocab=8194 (semantic codebook)
  - **Learned positional embeddings** (NOT RoPE): text pos + semantic pos, each an `nn.Embedding`
  - GPT2's own `wpe` replaced with a zero dummy; `wte` deleted entirely
- **Input concatenation:** `[condition_emb(1,1,1280) | text_emb(1,T,1280) | semantic_emb(1,T',1280)]`
  - `text_emb`: Embedding(32000,4096) → Linear(4096,4096) → SiLU → Linear(4096,1280) — frozen embed + MLP projection
  - `condition_emb`: ECAPA-TDNN speaker encoder (Qwen3TTSSpeakerEncoder) over the Wav2Vec2-BERT features
    - mel_dim=1024 (from w2v-bert), enc_dim=1280 (output)
    - 5-layer: TDNN + 3×SE-Res2Net + MFA + ASP + FC
  - `semantic_emb`: Embedding(8194, 1280)
- **Output head:** LayerNorm → Linear(1280, 8194) → semantic logits
- **Generation:** HF `generate()` with top-p/top-k/beam search, BOS=8192, EOS=8193
- **KV cache:** standard GPT-2 KV caching via HF

### S2A model (Semantic-to-Acoustic) — `s2a_model.pt` (417 MB)

- **Architecture:** Conditional Flow Matching (CFM) with DiT estimator + WaveNet final layer
  - DiT: hidden_dim=512, 8 heads, depth=13, cond_dim=512, style_dim=192
  - WaveNet: hidden_dim=512, kernel=5, dilation=1, 8 layers
  - Long skip connections (U-Net style)
- **Input pipeline:**
  1. Semantic token embedding: Embedding(8192, 8) → Linear(8, 1024)
  2. Concat with lm_latent: cat([lm_latent(1280), semantic_emb(1024)]) → Linear(2304, 1024)
  3. InterpolateRegulator: conv upsampling to target mel length (ratios [1,1,1,1])
  4. Prepend learned prompt condition (1, T_ref, 512)
- **Flow matching:** Euler ODE solver, 25 steps, CFG rate 0.7
- **Output:** mel spectrogram (80 bands), prompt portion stripped

### External models (NOT in the GGUF — must be separate or bundled)

1. **Wav2Vec2-BERT 2.0** (`facebook/w2v-bert-2.0`): ~600M params, extracts layer-17 hidden states
   - SeamlessM4TFeatureExtractor for audio preprocessing (mel filterbank)
   - z-normalised with per-dim mean/var from `wav2vec2bert_stats.pt`
2. **CAMPPlus** (`funasr/campplus`): ECAPA-TDNN speaker encoder
   - Input: 80-mel fbank @ 16kHz, mean-subtracted
   - Output: 192-dim speaker embedding
3. **BigVGAN** (`nvidia/bigvgan_v2_22khz_80band_256x`): mel→waveform vocoder

### Audio parameters

- Target sample rate: 22050 Hz
- Mel: n_fft=1024, hop=256, win=1024, 80 bands, fmin=0, fmax=None
- Prompt audio: resampled to both 16kHz (for w2v-bert + campplus) and 22050 Hz (for ref mel)

### Text formatting

```
formatted = "You are a helpful assistant. {lang_token}:{text}"
```

Where `lang_token` is a per-language string from `LANGUAGE_TOKEN_MAP` (e.g. "请朗读接下来的中文" for zh).

---

## GGUF strategy

This is a **5-model pipeline**. Options:

### Option A: Single mega-GGUF (infeasible)
Wav2Vec2-BERT alone is ~600M params — larger than many ASR models. Bundling all 5 into one GGUF would be >3 GB and force loading everything even when only the vocoder changes.

### Option B: Multi-GGUF (recommended, mirrors qwen3-tts)
- `confucius4-t2s-{quant}.gguf` — the 2.64 GB T2S model (GPT-2 backbone + text projector + speaker encoder + semantic head + position embeddings)
- `confucius4-s2a-{quant}.gguf` — the 417 MB S2A model (DiT + WaveNet + length regulator + token embedding)
- External models resolved via registry/cache:
  - Wav2Vec2-BERT 2.0: could share with any other backend that uses it (or ship a dedicated GGUF)
  - CAMPPlus: ~5 MB, can be baked into the T2S GGUF as extra tensors
  - BigVGAN: could share the existing `hifigan.h` core module or ship as a codec companion

### Option C: T2S+S2A bundled, externals separate
Merge T2S and S2A into one GGUF (they're always used together), keep w2v-bert and bigvgan as companion GGUFs via `--codec-model`. CAMPPlus baked in.

**Recommended: Option B** — matches the qwen3-tts `--codec-model` pattern and keeps file sizes reasonable for quantization.

---

## Existing CrispASR modules to reuse

| Need | Existing module | Notes |
|------|----------------|-------|
| GPT-2 attention | `core/attention.h` `kv_self_attn` | Causal masked self-attention with KV cache |
| Learned position embedding | — | New, but trivial (nn.Embedding lookup) |
| SiLU activation | `core/activation.h` | Already exists |
| LayerNorm | ggml native | `ggml_norm` |
| ECAPA-TDNN speaker enc | — | New module. Conv1d + Res2Net + SE + ASP pooling |
| DiT (S2A) | — | Similar to f5-tts DiT. Could share `core/` |
| WaveNet final layer | `core/hifigan.h` or new | WaveNet is different from HiFi-GAN |
| Flow matching ODE | — | Euler solver, same as f5-tts/chatterbox CFM |
| BigVGAN vocoder | `core/hifigan.h` | BigVGAN is a HiFi-GAN variant |
| Mel spectrogram | `core/mel.h` | Need to match params (n_fft=1024, hop=256, 80 bands) |
| SentencePiece tokenizer | `core/sentencepiece.h` | Already exists |
| Wav2Vec2-BERT | — | Large new model. Possibly share with future backends |

---

## Port steps (following the pipeline from CLAUDE.md)

1. **Converter** — `models/convert-confucius4-to-gguf.py`
   - T2S: extract GPT-2 weights + text projector + speaker encoder + position embeddings + semantic head from `t2s_model.safetensors`
   - S2A: extract DiT + WaveNet + length regulator from `s2a_model.pt`
   - CAMPPlus: bake into T2S GGUF from `funasr/campplus` checkpoint
   - `wav2vec2bert_stats.pt`: bake mean/var as tensors
   - Needs Kaggle for the full run (2.64 GB safetensors + w2v-bert download)

2. **Quantize** — add rules to `crispasr-quantize/main.cpp`
   - Keep embeddings/norms/biases at F16/F32
   - Quantize GPT-2 attention/FFN + DiT/WaveNet weights

3. **Reference dump** — `tools/reference_backends/confucius4_tts.py`
   - Per-stage: text_embed → condition_emb → gpt2_layer_0..23 → semantic_codes → s2a_cond → flow_step_0..24 → mel → wav

4. **C++ runtime** — `src/confucius4_tts.{h,cpp}`
   - T2S: GPT-2 with custom embedding concatenation + KV cache
   - S2A: CFM with DiT estimator
   - Vocoder: BigVGAN (HiFi-GAN variant)

5. **CLI adapter** — `examples/cli/crispasr_backend_confucius4_tts.cpp`

6. **All 12 checklist items** from docs/contributing.md

---

## Blocking questions

1. **Wav2Vec2-BERT**: Do we port this to GGUF too? It's 600M params with a complex conformer-based encoder. Could be its own backend module shared with future models that use w2v-bert, or we could compute the semantic features on-the-fly in the converter/reference-dumper and bake them as fixed conditioning (only works for predefined voices, not zero-shot).
   - **Decision needed:** For zero-shot voice cloning (the main use case), w2v-bert MUST run at inference time on the user's prompt audio. So it needs a GGUF port.
   
2. **BigVGAN**: Is the existing `core/hifigan.h` close enough, or does BigVGAN's architecture differ enough to need new code? BigVGAN uses anti-aliased multi-periodicity composition (AMP) blocks instead of standard HiFi-GAN MRF blocks.
   - **Likely answer:** New module needed, but structurally similar.

3. **External model sizes**: w2v-bert-2.0 is ~1.2 GB at F16. With Q4_K it'd be ~300 MB. CAMPPlus is tiny (~5 MB). BigVGAN v2 22kHz is ~112 MB. Total companion weight budget: ~400-500 MB Q4_K on top of the ~700 MB T2S Q4_K.
