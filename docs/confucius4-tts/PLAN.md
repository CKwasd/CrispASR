# Confucius4-TTS — port plan (§377)

**Issue:** #377 · **License:** Apache 2.0 · **Languages:** 14 (zh, en, ja, ko, de, fr, es, id, it, th, pt, ru, ms, vi)
**Source:** [netease-youdao/Confucius4-TTS](https://github.com/netease-youdao/Confucius4-TTS) ·
**Weights:** [HF repo](https://huggingface.co/netease-youdao/Confucius4-TTS) (3.06 GB total)

---

## NOW — active work

**Status:** Python blueprint read, architecture fully traced, PLAN written.
Converter + C++ backend not started.

---

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
