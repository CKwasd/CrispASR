---
license: mit
base_model: microsoft/VibeVoice-ASR-BitNet
tags:
  - speech
  - asr
  - speech-recognition
  - gguf
  - crispasr
  - bitnet
  - ternary
language:
  - en
  - zh
  - fr
  - it
  - ko
  - pt
  - vi
library_name: crispasr
pipeline_tag: automatic-speech-recognition
---

# VibeVoice-ASR-BitNet GGUF

GGUF conversion of [microsoft/VibeVoice-ASR-BitNet](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

## Available files

All variants use **TQ2_0** (ternary 2-bit) for the LM projection weights. They differ in how aggressively the VAE encoder and LM embedding are quantized. Every variant produces **identical transcription output** on English (JFK) and on three Korean clips — see [Quality verification](#quality-verification), which now covers more than English.

| File | VAE | Embed | Size | Recommended |
|------|-----|-------|------|-------------|
| `vibevoice-asr-bitnet-tq2.gguf` | Q8_0 | F16 | 1.55 GB | default |
| `vibevoice-asr-bitnet-embed-q8.gguf` | Q8_0 | Q8_0 | 1.34 GB | |
| `vibevoice-asr-bitnet-vae-q5.gguf` | Q5_0 | F16 | 1.33 GB | |
| `vibevoice-asr-bitnet-both-q5.gguf` | Q5_0 | Q8_0 | 1.12 GB | |
| `vibevoice-asr-bitnet-vae-q4.gguf` | Q4_0 | F16 | 1.26 GB | |
| `vibevoice-asr-bitnet-aggro.gguf` | Q4_0 | Q8_0 | 1.05 GB | smallest |

**Pick any file** — the variants really are interchangeable, and that now rests on more than one English clip. The `aggro` variant is 35% smaller than the default at identical output.

> ⚠ **That says nothing about how good this checkpoint is.** Variant equivalence
> and model quality are different claims, and an earlier version of this card
> ran them together. The BitNet checkpoint is **materially weaker than the full
> [VibeVoice-ASR](https://huggingface.co/cstr/vibevoice-asr-GGUF) on non-English
> audio** — see [Non-English quality](#non-english-quality). If you need Korean,
> use the full model.

## Model Details

| Property | Value |
|----------|-------|
| Base model | microsoft/VibeVoice-ASR-BitNet (1.5B params) |
| Architecture | Dual VAE encoder + Qwen2 LM decoder |
| LM | 28 layers, 1536 hidden, 12 heads, 2 KV heads |
| VAE | ConvNeXt-style 1D CNN, stride 3200, acoustic (64d) + semantic (128d) |
| Input | 24 kHz mono PCM |
| Languages | English, Chinese, French, Italian, Korean, Portuguese, Vietnamese |
| License | MIT |

## Quantization strategy

The LM projection weights (q/k/v/o/gate/up/down) are BitNet-trained with ternary values {-1, 0, 1}. The converter applies the ternary quantization (`s = 1/mean(|w|)`, round, clamp) and packs into ggml's native **TQ2_0** format (2.06 bpw). No custom SIMD kernels needed -- standard ggml `mul_mat` handles TQ2_0 transparently.

| Component | Quant options | Notes |
|-----------|--------------|-------|
| LM projections | TQ2_0 (all variants) | BitNet ternary, 2 bits/weight |
| VAE encoders | Q8_0 / Q5_0 / Q4_0 | ConvNeXt conv + FFN weights |
| LM embedding | F16 / Q8_0 | Lookup table, Q8_0 is lossless |
| Norms / biases | F32 (all variants) | Numerical stability |
| lm_head | tied to tok_emb (not stored) | Verified byte-identical in the checkpoint, so the fallback is exact, not an approximation |

## Quality verification

All 6 variants were tested on JFK audio (11 s, 24 kHz) and produce identical output:

> And so, my fellow Americans, ask not what your country can do for you—ask what you can do for your country.

**The quantization axis was then pushed much harder** ([#369](https://github.com/CrispStrobe/CrispASR/issues/369)). Three builds from the same checkpoint, storing the *same* ternary values — `ternary_quantize()` has already collapsed them to `{-mean, 0, +mean}`, so only the ggml matmul path changes, and with it whether activations are quantized at all:

| LM stored as | Size | Activations | Output |
|---|---|---|---|
| `TQ2_0` (shipped) | 1.41 GB | block-quantized int8 | baseline |
| `Q8_0` | 2.46 GB | per-32-block int8 | **character-identical** |
| `F16` | 3.69 GB | **not quantized** | **character-identical** |

Identical on all four clips (English + three Korean). F16 activations are strictly more accurate than the per-token int8 of Microsoft's I2_S kernel, so this rules out activation quantization as a source of quality loss: **raising the LM precision does not change a single character, and a larger file buys nothing.**

Sweep ran on Kaggle (CPU). Conversion + transcription took ~20 min for the 6 shipped variants.

## Non-English quality

The model card used to list seven languages and verify one. Measured on Korean
(GT: `내일 오전에 회의 자료를 보내 주세요`), CPU, with CrispASR at the #369 fixes:

| clip | this BitNet checkpoint | full VibeVoice-ASR (7B) |
|---|---|---|
| synthetic TTS, 3.3 s | `네, 오늘은 해외 자료를 보내주세요.` | `내일 오전에 회의 자료를 보내주세요.` ✅ |
| mic recording | `내일 오전에 회의 자료 교육 보내주세요.` | `내일 오전에 회의 자료를 보내주세요.` ✅ |
| mic recording, pitch-shifted | `Nếu ồ trên này, …` (Vietnamese) | `내일 오전에 회의 자료를 보내주세요.` ✅ |

English is unaffected — JFK is exact. This is the ternary checkpoint's own
capability on harder non-English material, not a conversion or runtime defect:
the σ-VAE encoder matches upstream's own modules at cos 0.999926 on identical
24 kHz input, the ternary weights differ from upstream's I2_S in 2 values out of
13.76 M, and the table above shows the LM's numerics are insensitive to storage
precision. The full model gets every one of these right through the *same*
CrispASR pipeline.

Use this checkpoint for its size and for English; reach for the full model when
the language matters.

## Usage

### CLI
```bash
# Auto-download (uses the default tq2 variant)
crispasr -m auto --backend vibevoice-bitnet -f audio.wav

# Or with a specific file
crispasr -m vibevoice-asr-bitnet-aggro.gguf --backend vibevoice -f audio.wav
```

### C API
```c
#include "vibevoice.h"

vibevoice_context_params params = vibevoice_context_default_params();
params.n_threads = 4;
vibevoice_context *ctx = vibevoice_init_from_file("vibevoice-asr-bitnet-aggro.gguf", params);
char *text = vibevoice_transcribe(ctx, samples_24khz, n_samples);
printf("%s\n", text);
free(text);
vibevoice_free(ctx);
```

### Python
```python
import crispasr
s = crispasr.Session("vibevoice-asr-bitnet-aggro.gguf", backend="vibevoice")
result = s.transcribe("audio.wav")
print(result["text"])
```

## Conversion

```bash
# Default (Q8_0 VAE + F16 embed)
python models/convert-vibevoice-bitnet-to-gguf.py \
    --input microsoft/VibeVoice-ASR-BitNet \
    --output vibevoice-asr-bitnet-tq2.gguf

# Aggressive (Q4_0 VAE + Q8_0 embed, 1.05 GB)
python models/convert-vibevoice-bitnet-to-gguf.py \
    --input microsoft/VibeVoice-ASR-BitNet \
    --output vibevoice-asr-bitnet-aggro.gguf \
    --vae-quant q4_0 --embed-quant q8_0

# Research only: store the ternary LM unquantized. Same weights, ~3x the file,
# character-identical output (see Quality verification) -- there is no reason to
# ship this, and it exists so the claim above can be re-checked rather than
# taken on trust.
python models/convert-vibevoice-bitnet-to-gguf.py \
    --input microsoft/VibeVoice-ASR-BitNet \
    --output vibeasr-bitnet-lm-f16.gguf \
    --lm-quant f16 --vae-quant q8_0 --embed-quant q8_0
```

## Acknowledgements

- [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) for the model architecture and training
- [microsoft/VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp) for the BitNet quantization approach

## Provenance and EU AI Act Art. 53 note

- **Upstream model:** [microsoft/VibeVoice-ASR-BitNet](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet) — published by `microsoft`.
- **Upstream licence:** `mit`. This repository redistributes under the same terms; it grants no rights the upstream licence does not.
- **What was done here:** format conversion and/or quantisation only (GGUF). No training, no fine-tuning, no merging, no distillation, no change to architecture, vocabulary or capability. Only the numeric representation of the upstream weights differs.
- **Training data:** documented — where it is documented at all — by the upstream provider; see the upstream model card. No training data was used, added or selected by this repository. No training-content summary was found on the upstream model card at the time of writing; that documentation gap is upstream's and is not filled here.
- **Provider status:** under Regulation (EU) 2024/1689 the upstream authors remain the provider of this model. Converting the serialisation format does not make this repository the provider of a new general-purpose AI model, and no such claim is made. Questions about training content, copyright policy or model capability belong upstream.
