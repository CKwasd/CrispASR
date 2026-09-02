# IndexTTS-2.5 → CrispASR `indextts2` backend — port notes (phase 1)

Companion to `docs/indextts2-feasibility.md` (issue #346). That document decided
*whether*; this one records *what*, at tensor and symbol level, so phase 2 is
transcription rather than archaeology.

**Sources.** Upstream `github.com/index-tts/index-tts` at tag **`v2.5.0`**, cloned
to `/mnt/volume1/tmp-overflow/indextts25-src`; checkpoints from HF
`IndexTeam/IndexTTS-2.5` (revision `c39ce5b`) at
`/mnt/storage/gguf-models/indextts25-src`. Every line citation below is against
that tag. All shapes are read off the actual checkpoints, not inferred.

**License.** `LICENSE` at `v2.5.0` is bilibili's *Model Use License*: §1.4 defines
"Model" to include the **final code**, §1.5(iii) names quantization as a Derivative
Work. GGUFs produced from these checkpoints are Derivative Works and must not be
published to a hub — the converter says so and phase 2's registry entry must point
at a local conversion step, not a `cstr/…-GGUF` URL.

---

## 1. Checkpoint inventory (`IndexTeam/IndexTTS-2.5`, 5.49 GB total)

| file | size | contents | downloaded |
|---|---|---|---|
| `gpt.pth` | 3 259.60 MB | `UnifiedVoice` state dict, 456 tensors, **814.8 M params fp32** | yes |
| `codec.pth` | 607.29 MB | `EnhancedCodec` under key `model`, 243 tensors | yes |
| `s2mel.pth` | 414.91 MB | `{"net": {"cfm": 256, "length_regulator": 22, "gpt_layer": 6}}` | yes |
| `feat1.pt` | 0.06 MB | `spk_matrix`, bare tensor `(73, 192)` — CAMPPlus style prototypes | yes |
| `feat2.pt` | 0.37 MB | `emo_matrix`, bare tensor `(73, 1280)` — emovec prototypes | yes |
| `wav2vec2bert_stats.pt` | 0.01 MB | `{"mean": (1024,), "var": (1024,)}` | yes |
| `multilingual_zh_ja_yue_char_del.tiktoken` | 0.91 MB | 58 836 base64 merge ranks | yes |
| `config.yaml` | 2.9 kB | the hyperparameter source of truth | yes |
| `qwen0.6bemo4-merge/model.safetensors` | 1 192.14 MB | QwenEmotion (Qwen3-0.6B fine-tune) | **no — inventoried only** |
| `qwen0.6bemo4-merge/{tokenizer.json,vocab.json,merges.txt,…}` | 15.9 MB | its tokenizer | **no** |
| `LICENSE`, `README.md`, `.gitattributes` | 15 kB | — | yes |

`qwen0.6bemo4-merge/` is skipped deliberately: it is only reachable via
`use_emo_text=True`, and `use_qwen_emo` defaults to `False`
(`infer_v2_5.py:126-131`). The feasibility doc's §4 decision stands — expose the
8-d vector on the CLI instead.

### External models, **not** in the repo, and already converted in-tree

| model | upstream repo | existing CrispASR GGUF | verified |
|---|---|---|---|
| BigVGAN v2 22 kHz / 80-band / 256× | `nvidia/bigvgan_v2_22khz_80band_256x` (`utils/model_download.py:33`) | `cstr/confucius4-tts-GGUF/confucius4-tts-bigvgan-22k-f16.gguf`, 214.2 MiB | yes, byte-identical upstream repo |
| w2v-BERT 2.0 (encoder, layer-17) | `facebook/w2v-bert-2.0` (`utils/model_download.py:29`) | `cstr/confucius4-tts-GGUF/confucius4-tts-w2v-f16.gguf`, 785.4 MiB | yes, same repo |
| CAMPPlus 192-d | `funasr/campplus` → `campplus_cn_common.bin` (`utils/model_download.py:28`) | baked into the s2mel GGUF under `campplus.*`, same as Confucius4's S2A | yes |

**Do not convert a duplicate BigVGAN or w2v-BERT.** `models/convert-indextts2-to-gguf.py
--check-companions` verifies both are reachable.

### Conversion result (`--outtype f16`, run on the 8 GB VPS, tensor-streamed)

| GGUF | arch | tensors | size |
|---|---|---|---|
| `indextts2-gpt-f16.gguf` | `indextts2.gpt` | 456 (454 from `gpt.pth` + `spk_matrix` + `emo_matrix`) | 1 413.3 MiB |
| `indextts2-codec-f16.gguf` | `indextts2.codec` | 241 | 96.7 MiB |
| `indextts2-s2mel-f16.gguf` | `indextts2.s2mel` | 1 082 (267 s2mel + 815 CAMPPlus) | 234.5 MiB |

Written to `/mnt/storage/gguf-models/indextts25/`. Peak RSS stayed under 1 GB —
`torch.load(..., mmap=True)` plus one-tensor-at-a-time writes. Reproduce with:

```
python models/convert-indextts2-to-gguf.py \
  --model-dir  /mnt/storage/gguf-models/indextts25-src \
  --output-dir /mnt/storage/gguf-models/indextts25 \
  --outtype f16 --check-companions
```

Read back from the shipped files: `indextts2.emo_perceiver.num_layers = 2`,
`num_latents = 1`, `dim = 1024`, `dim_context = 512`, `dim_head = 64`,
`heads = 4`, `ff_mult = 2`; `gpt.{layers,heads} = 24,20`;
`number_text_tokens = 60509`; `mel_pos_size = 1818`; `text_pos_size = 602`;
`w2v_bert.layer = 17`; `sample_rate = 22050`; `emo.num`/`emo.bias`/`emo.labels`
8 entries each; `w2v_bert.mean` 1024; `tokenizer.ggml.tokens` 60 509;
`indextts2.tokenizer.languages` 106. Tensors `spk_matrix (192,73)`,
`emo_matrix (1280,73)`, `lang_embedding.weight (1280,107)` and
`emo_perc.layers.1.*` (the second Perceiver layer) all present. Longest GGUF
tensor name across the three files: 72 chars (s2mel), 41 (codec), 39 (gpt).

---

## 2. Constants (line-cited)

| constant | value | source |
|---|---|---|
| `model_dim` | 1280 | `config.yaml:14` |
| `layers` / `heads` / `head_dim` | 24 / 20 / 64 | `config.yaml:20,17` |
| FFN dim | 5120 | GPT-2 `4*n_embd`; confirmed `gpt.h.N.mlp.c_fc.weight (1280, 5120)` |
| `max_mel_tokens` | 1815 | `config.yaml:15` |
| `max_text_tokens` | 600 | `config.yaml:16` |
| `number_text_tokens` | 60509 | `config.yaml:21` |
| `number_mel_codes` | 8194 | `config.yaml:22` |
| `start/stop_mel_token` | 8192 / 8193 | `config.yaml:23-24` |
| `start/stop_text_token` | 0 / 1 | `config.yaml:25-26` |
| `mel_length_compression` | 1024 | `config.yaml:19` |
| mel pos table | 1818 = 1815+2+1 | `gpt/model_v2.py:398-400`; `mel_pos.weight (1818, 1280)` |
| text pos table | 602 = 600+2 | same; `text_pos.weight (602, 1280)` |
| `spk_cond_mode` | `"campplus"` | `infer_v2_5.py:138` |
| `spk_emb_proj` | Linear(192→1280) | `gpt/model_v2.py:353` |
| `lang_embedding` | Embedding(107, 1280) = `len(LANGUAGE_DICT)+1` | `gpt/model_v2.py:390`; `LANGUAGES` has 106 entries (`utils/tokenizer.py:11-118`) |
| emo Conformer | 4 blocks / 512d / 4 heads / FFN 1024 / `conv2d2` | `config.yaml:36-42` |
| emo Conformer depthwise kernel | 15 | `emo_cond_enc.enc.N.conv.dw.weight (512, 1, 15)` |
| emo Conformer `embed.conv.0` | `Conv2d(1, 512, k=3, stride=2)` + ReLU — subsamples **both** axes | `gpt/conformer/subsampling.py:148-153` |
| emo Conformer `embed.out.0` | `Linear(odim·((idim−1)//2) → 512)` = `Linear(261632 → 512)`, 261632 = 512 × 511 | same; T → (T−1)//2, `subsampling_rate = 2` |
| emo Conformer `after_norm` | LayerNorm(512, eps 1e-5) | `gpt/conformer_encoder.py:395` |
| emo Perceiver | **2 layers** (`depth` defaults to 2 — `perceiver.py:224-231` (`depth=2` at :227); `model_v2.py:381` does not pass it), **1 latent**, dim 1024, ctx 512, ff_mult 2 | `gpt/model_v2.py:381-384`; confirmed by `emo_perc.layers.{0,1}.*` in the checkpoint |
| emo Perceiver head dim | 64 (4 heads × 64 = 256) | `perceiver.py:229`; `emo_perc.layers.0.0.to_q.weight (256, 1024)` |
| emo Perceiver GEGLU | `Linear(1024→2730) → GEGLU → Linear(1365→1024)`, 1365 = ⌊1024·2·2/3⌋ | `perceiver.py:204-218` |
| `emovec_layer` / `emo_layer` | Linear(1024→1280) / Linear(1280→1280) | `gpt/model_v2.py:391-392` |
| codec codebook | 8192 × 8, **1 quantizer**, l2-normalised | `config.yaml:44,46`; `codec/models.py:38` (`num_quantizers` defaults to 1 — the cfg does not set it) |
| codec downsample | 2 | `codec/models.py:39` (cfg does not set it) |
| Vocos backbone | 384d, 12 ConvNeXt layers, intermediate 2048 | `config.yaml:47-49` |
| DiT | 13 layers / 512d / 8 heads, `in_channels` 80 | `config.yaml:76-81` |
| DiT FFN inner | 1536 | `feed_forward.w1.weight (1536, 512)` |
| DiT `cond_x_merge_linear` in | **864** = 512 + 2×80 + 192 | `diffusion_transformer.py:177-179` |
| DiT `skip_linear` in | **592** = 512 + 80 | `diffusion_transformer.py:175` |
| `uvit_skip_connection` | true → emit `i < 6`, receive `i > 6` (13 layers) | `config.yaml:96`; `gpt_fast/model.py:152-158` |
| WaveNet | 8 layers, kernel 5, dilation_rate 1, gin 512 | `config.yaml:99-104` |
| length regulator | channels 512, in 1024, 4 sampling ratios → `model.0…model.12` | `config.yaml:64-73`; `length_regulator.py:47-63` |
| `n_quantizers` at both LR calls | 3 | `infer_v2_5.py:654` (prompt) and `:836` (target) |
| style dim | 192 | `config.yaml:63` |
| sample rate | 22050 | `config.yaml:52` |
| mel | n_fft 1024, win 1024, hop 256, n_mels 80, fmin 0, fmax `None`, **center=False** | `config.yaml:53-59`; `infer_v2_5.py:255-266` (`center` at :263) |
| diffusion steps | 25 | `infer_v2_5.py:829` |
| `inference_cfg_rate` | 0.7 | `infer_v2_5.py:830` |
| duration | `target_len = S_infer.shape[1] * 1.72 * duration_factor` | `infer_v2_5.py:832` |
| ref audio cut | 15 s | `infer_v2_5.py:626, 642, 685`; `_load_and_cut_audio` at :396-408 |
| `emo_num` | `[3,17,2,8,4,5,10,24]` (sums to 73) | `config.yaml:110` |
| emotion order | happy, angry, sad, afraid, disgusted, melancholic, surprised, calm | `infer_v2_5.py:492` (comment), used at :669-679 |
| emotion de-emphasis bias | `[0.9375,0.875,1.0,1.0,0.9375,0.9375,0.6875,0.5625]` | `infer_v2_5.py:493` |
| emotion sum cap | 0.8 | `infer_v2_5.py:496-500` |
| `silent_token` | 52, shrink above 30 consecutive | `infer_v2_5.py:remove_long_silence` |
| tiktoken pattern | `'s\|'t\|'re\|'ve\|'m\|'ll\|'d\| ?\p{L}+\| ?\p{N}+\| ?[^\s\p{L}\p{N}]+\|\s+(?!\S)\|\s+` | `utils/tokenizer.py:215` |
| vocab arithmetic | 58 836 ranks + 1 673 specials = **60 509** = `number_text_tokens`; `text_embedding` has 60 510 rows (one unreachable) | `utils/tokenizer.py:191-210`; `gpt/model_v2.py:388` |
| upstream sampling defaults (**not** used for diffs) | `num_beams=3, repetition_penalty=10.0, top_k=30, top_p=0.8, temperature=0.8` | `infer_v2_5.py:731-739` |
| pinned dependency | `transformers==4.52.1` | `pyproject.toml:61` (see §6) |

### The 1 673 special tokens, in id order (ids start at 58 836)

`<|endoftext|>`, `<|startoftranscript|>`, 99 `<|{lang}|>` (the first 99 of
`LANGUAGES`), 11 `<|{audio_event}|>`, 4 `<|{emotion}|>`, then
`<|translate|> <|transcribe|> <|startoflm|> <|startofprev|> <|nospeech|>
<|notimestamps|>` (**six**, easy to miscount as five), 30
`<|SPECIAL_TOKEN_{1..30}|>`, 20 `<|TTS/…|>`, and 1 501 timestamp tokens
`<|{i*0.02:.2f}|>`. Getting this list wrong shifts every `<|zh|>`/`<|ja|>` id.
`models/convert-indextts2-to-gguf.py:_special_tokens()` reproduces it and the
converter asserts the 60 509 total.

Note `<|{lang}|>` (a *text* token, only the first 99 languages) is a different
thing from `LANGUAGE_DICT[lang]` (a 0..105 index into `lang_embedding`,
`utils/tokenizer.py:121`). Both are used, on the same synthesis
(`infer_v2_5.py:698, 726`). `en=0, zh=1, es=3, ja=7, ar=13, common=105`.

---

## 3. Tensor inventory — checkpoint name → shape → GGUF name

`{N}` is a layer index. Naming rationale: the GPT side mirrors
`models/convert-indextts-to-gguf.py`'s shortened family so `src/indextts.cpp`'s
Conformer/Perceiver binders transfer with only an `emo_` prefix; the s2mel side
mirrors `tools/kaggle/confucius4-tts-convert/confucius4-tts-convert.py` exactly so
`src/confucius4_tts.cpp` binds it unchanged.

The longest emitted name is 72 chars
(`decoder.estimator.transformer_blocks.12.attention_norm.modulation.weight`),
which is over stock ggml's 64-char `GGML_MAX_NAME` and loads only because this
repo raises it: `CMakeLists.txt:172`, `add_compile_definitions(GGML_MAX_NAME=128)`.
Do not "shorten to fit 64" — that would break the Confucius4 mirror. The converter
hard-fails on any name ≥ 128.

### 3.1 `gpt.pth` → `indextts2-gpt-*.gguf` (arch `indextts2.gpt`)

| checkpoint | shape | GGUF |
|---|---|---|
| `spk_emb_proj.weight` / `.bias` | [1280, 192] / [1280] | same |
| `emo_conditioning_encoder.embed.conv.0.weight` / `.bias` | [512, 1, 3, 3] / [512] | `emo_cond_enc.embed.conv.0.*` |
| `emo_conditioning_encoder.embed.out.0.weight` / `.bias` | [512, 261632] / [512] | `emo_cond_enc.embed.out.0.*` |
| `emo_conditioning_encoder.embed.pos_enc.pe` | [1, 5000, 512] | `emo_cond_enc.embed.pos_enc.pe` |
| `emo_conditioning_encoder.after_norm.{weight,bias}` | [512] | `emo_cond_enc.after_norm.*` |
| `emo_conditioning_encoder.encoders.{N}.self_attn.pos_bias_{u,v}` | [4, 128] | `emo_cond_enc.enc.{N}.sa.pos_bias_*` |
| `…self_attn.linear_{q,k,v,out}.{weight,bias}` | [512, 512] / [512] | `emo_cond_enc.enc.{N}.sa.linear_*` |
| `…self_attn.linear_pos.weight` | [512, 512] | `emo_cond_enc.enc.{N}.sa.linear_pos.weight` |
| `…feed_forward.w_1.{weight,bias}` | [1024, 512] / [1024] | `emo_cond_enc.enc.{N}.ff.w_1.*` |
| `…feed_forward.w_2.{weight,bias}` | [512, 1024] / [512] | `emo_cond_enc.enc.{N}.ff.w_2.*` |
| `…conv_module.pointwise_conv1.{weight,bias}` | [1024, 512, 1] / [1024] | `emo_cond_enc.enc.{N}.conv.pw1.*` |
| `…conv_module.depthwise_conv.{weight,bias}` | [512, 1, 15] / [512] | `emo_cond_enc.enc.{N}.conv.dw.*` |
| `…conv_module.norm.{weight,bias}` | [512] | `emo_cond_enc.enc.{N}.conv.norm.*` |
| `…conv_module.pointwise_conv2.{weight,bias}` | [512, 512, 1] / [512] | `emo_cond_enc.enc.{N}.conv.pw2.*` |
| `…norm_{ff,mha,conv,final}.{weight,bias}` | [512] | `emo_cond_enc.enc.{N}.norm_*` |
| `emo_perceiver_encoder.latents` | [1, 1024] | `emo_perc.latents` |
| `emo_perceiver_encoder.proj_context.{weight,bias}` | [1024, 512] / [1024] | `emo_perc.proj_context.*` |
| `emo_perceiver_encoder.layers.{0,1}.0.to_q.weight` | [256, 1024] | `emo_perc.layers.{N}.0.to_q.weight` |
| `emo_perceiver_encoder.layers.{0,1}.0.to_kv.weight` | [512, 1024] | `emo_perc.layers.{N}.0.to_kv.weight` |
| `emo_perceiver_encoder.layers.{0,1}.0.to_out.weight` | [1024, 256] | `emo_perc.layers.{N}.0.to_out.weight` |
| `emo_perceiver_encoder.layers.{0,1}.1.0.{weight,bias}` | [2730, 1024] / [2730] | `emo_perc.layers.{N}.1.0.*` (→ GEGLU) |
| `emo_perceiver_encoder.layers.{0,1}.1.2.{weight,bias}` | [1024, 1365] / [1024] | `emo_perc.layers.{N}.1.2.*` |
| `emo_perceiver_encoder.norm.gamma` | [1024] | `emo_perc.norm.gamma` (RMSNorm) |
| `emo_layer.{weight,bias}` | [1280, 1280] / [1280] | same |
| `emovec_layer.{weight,bias}` | [1280, 1024] / [1280] | same |
| `text_embedding.weight` | [60510, 1280] | same |
| `lang_embedding.weight` | [107, 1280] | same |
| `mel_embedding.weight` | [8194, 1280] | same |
| `gpt.h.{N}.ln_{1,2}.{weight,bias}` | [1280] | same |
| `gpt.h.{N}.attn.c_attn.weight` | [1280, 3840] | same, **transposed** → [3840, 1280] |
| `gpt.h.{N}.attn.c_attn.bias` | [3840] | same |
| `gpt.h.{N}.attn.c_proj.weight` | [1280, 1280] | same, **transposed** |
| `gpt.h.{N}.mlp.c_fc.weight` | [1280, 5120] | same, **transposed** → [5120, 1280] |
| `gpt.h.{N}.mlp.c_proj.weight` | [5120, 1280] | same, **transposed** → [1280, 5120] |
| `gpt.ln_f.{weight,bias}` | [1280] | same |
| `mel_pos_embedding.emb.weight` | [1818, 1280] | `mel_pos.weight` |
| `text_pos_embedding.emb.weight` | [602, 1280] | `text_pos.weight` |
| `final_norm.{weight,bias}` | [1280] | same |
| `mel_head.{weight,bias}` | [8194, 1280] / [8194] | same |
| `text_head.{weight,bias}` | [60510, 1280] / [60510] | **skipped** — mel-AR inference only |
| *(from `feat1.pt`)* | [73, 192] | `spk_matrix` |
| *(from `feat2.pt`)* | [73, 1280] | `emo_matrix` |

GPT-2 `Conv1D` stores `[in, out]`; the four suffixes above are transposed to
`nn.Linear`'s `[out, in]` so `ggml_mul_mat` sees `ne[0] == in_dim`. Identical rule
to `convert-indextts-to-gguf.py:104-113` and `confucius4-tts-convert.py:240-257`.

`gpt.pth` has **no** `conditioning_encoder`, `perceiver_encoder` or `speed_emb`:
`spk_cond_mode="campplus"` takes the other branch of `gpt/model_v2.py:352-373`.
The 1.5-style conformer/perceiver exists only for the *emotion* path.

### 3.2 `s2mel.pth["net"]` → `indextts2-s2mel-*.gguf` (arch `indextts2.s2mel`)

Renames into Confucius4's S2A family. Verified: all 223 keys
`src/confucius4_tts.cpp` binds are present, 0 missing.

| checkpoint (`net["cfm"]`) | shape | GGUF (Confucius4 family) |
|---|---|---|
| `estimator.transformer.layers.{N}.attention.wqkv.weight` | [1536, 512] | `decoder.estimator.transformer_blocks.{N}.attention.wqkv.weight` |
| `…attention.wo.weight` | [512, 512] | `…transformer_blocks.{N}.attention.wo.weight` |
| `…feed_forward.w1.weight` / `w3` | [1536, 512] | `…transformer_blocks.{N}.feed_forward.w1/w3.weight` |
| `…feed_forward.w2.weight` | [512, 1536] | `…transformer_blocks.{N}.feed_forward.w2.weight` |
| `…attention_norm.project_layer.{weight,bias}` | [1024, 512] / [1024] | `…transformer_blocks.{N}.attention_norm.**modulation**.*` |
| `…attention_norm.norm.weight` | [512] | `…transformer_blocks.{N}.attention_norm.norm.weight` |
| `…ffn_norm.project_layer.{weight,bias}` | [1024, 512] / [1024] | `…transformer_blocks.{N}.ffn_norm.**modulation**.*` |
| `…ffn_norm.norm.weight` | [512] | `…transformer_blocks.{N}.ffn_norm.norm.weight` |
| `…skip_in_linear.{weight,bias}` | [512, 1024] / [512] | `…transformer_blocks.{N}.skip_in_linear.*` |
| `estimator.transformer.norm.project_layer.{weight,bias}` | [1024, 512] / [1024] | `decoder.estimator.**transformer_norm.modulation**.*` |
| `estimator.transformer.norm.norm.weight` | [512] | `decoder.estimator.transformer_norm.norm.weight` |
| `estimator.cond_x_merge_linear.{weight,bias}` | [512, 864] / [512] | `decoder.estimator.**input_embed.proj**.*` |
| `estimator.cond_projection.{weight,bias}` | [512, 512] / [512] | `decoder.estimator.**input_embed.mu_projection**.*` |
| `estimator.skip_linear.{weight,bias}` | [512, 592] / [512] | `decoder.estimator.skip_linear.*` |
| `estimator.res_projection.{weight,bias}` | [512, 512] / [512] | `decoder.estimator.res_projection.*` |
| `estimator.conv1.{weight,bias}` | [512, 512] / [512] | `decoder.estimator.conv1.*` (a **Linear**, despite the name) |
| `estimator.conv2.{weight,bias}` | [80, 512, 1] / [80] | `decoder.estimator.conv2.*` (Conv1d k=1) |
| `estimator.t_embedder.mlp.{0,2}.{weight,bias}` | [512, 256] / [512, 512] | `decoder.estimator.t_embedder.**time_mlp**.{0,2}.*` |
| `estimator.t_embedder2.mlp.{0,2}.{weight,bias}` | same | `decoder.estimator.t_embedder2.time_mlp.{0,2}.*` |
| `estimator.final_layer.linear.weight_{g,v}` + `.bias` | [512,1] / [512,512] / [512] | **fused** → `decoder.estimator.final_layer.linear.weight` + `.bias` |
| `estimator.final_layer.adaLN_modulation.1.{weight,bias}` | [1024, 512] / [1024] | same, under `decoder.` |
| `estimator.wavenet.cond_layer.conv.conv.{weight_g,weight_v,bias}` | [8192,1,1] / [8192,512,1] / [8192] | `decoder.estimator.wavenet.cond_layer.conv.{weight_g,weight_v,bias}` — **left unfused** |
| `estimator.wavenet.in_layers.{N}.conv.conv.{weight_g,weight_v,bias}` | [1024,1,1] / [1024,512,5] / [1024] | `decoder.estimator.wavenet.in_layers.{N}.conv.*` — **left unfused** |
| `estimator.wavenet.res_skip_layers.{N}.conv.conv.{weight_g,weight_v,bias}` | [1024,1,1] / [1024,512,1] / [1024] | `decoder.estimator.wavenet.res_skip_layers.{N}.conv.*` — **left unfused** |
| `estimator.input_pos` | [16384] int64 | skipped (buffer) |
| `estimator.t_embedder{,2}.freqs` | [128] | skipped (recomputed) |
| `estimator.x_embedder.{bias,weight_g,weight_v}` | [512] / [512,1] / [512,80] | skipped — **built but never called** in `DiT.forward` |
| `estimator.cond_embedder.weight` | [1024, 512] | skipped — discrete-content path, `cond_in_module = self.cond_projection` is hard-wired (`diffusion_transformer.py:207`) |
| `estimator.content_mask_embedder.weight` | [1, 512] | skipped — class-dropout, training only |

| checkpoint (`net["length_regulator"]`) | shape | GGUF |
|---|---|---|
| `content_in_proj.{weight,bias}` | [512, 1024] / [512] | `length_regulator.content_in_proj.*` |
| `model.{0,3,6,9}.{weight,bias}` | [512, 512, 3] / [512] | `length_regulator.model.{N}.*` (Conv1d k3 p1) |
| `model.{1,4,7,10}.{weight,bias}` | [512] | `length_regulator.model.{N}.*` (GroupNorm(1, 512)) |
| `model.12.{weight,bias}` | [512, 512, 1] / [512] | `length_regulator.model.12.*` (final Conv1d k1) |
| `embedding.weight` | [2048, 512] | skipped — `is_discrete=false` |
| `mask_token` | [1, 512] | skipped — training only |

`net["gpt_layer"]` (Linear 1280→256→128→1024, 6 tensors) is **dead weight**:
`MyModel` is constructed with `use_gpt_latent=False`, so the module is never
instantiated and `load_checkpoint2` never reads that group
(`s2mel/modules/commons.py:409-420, 600-621`).

Plus 815 CAMPPlus tensors under `campplus.*` (from `funasr/campplus`), identical
to Confucius4's S2A bake, so `confucius4_bind_campplus()` binds them as-is.

### 3.3 `codec.pth["model"]` → `indextts2-codec-*.gguf` (arch `indextts2.codec`)

Names kept verbatim; only the RVQ's two weight-normed 1×1 convs are fused.

| checkpoint | shape | GGUF |
|---|---|---|
| `down.{weight,bias}` | [1024, 1024, 3] / [1024] | same (Conv1d k3 **s2** p1) |
| `up.{weight,bias}` | [1024, 1024, 3] / [1024] | same (Conv1d k3 s1 p1) |
| `{encoder,decoder}.0.embed.{weight,bias}` | [384, 1024, 7] / [384] | same (Conv1d k7 p3) |
| `{encoder,decoder}.0.norm.{weight,bias}` | [384] | same (LayerNorm, eps 1e-6) |
| `{encoder,decoder}.0.convnext.{0..11}.dwconv.{weight,bias}` | [384, 1, 7] / [384] | same (depthwise k7 p3) |
| `{encoder,decoder}.0.convnext.{N}.norm.{weight,bias}` | [384] | same |
| `{encoder,decoder}.0.convnext.{N}.pwconv1.{weight,bias}` | [2048, 384] / [2048] | same (Linear + GELU) |
| `{encoder,decoder}.0.convnext.{N}.pwconv2.{weight,bias}` | [384, 2048] / [384] | same |
| `{encoder,decoder}.0.convnext.{N}.gamma` | [384] | same (layer scale, init 1/12) |
| `{encoder,decoder}.0.final_layer_norm.{weight,bias}` | [384] | same |
| `{encoder,decoder}.1.{weight,bias}` | [1024, 384] / [1024] | same (Linear 384→1024) |
| `quantizer.quantizers.0.in_project.weight_{g,v}` + `.bias` | [8,1,1] / [8,1024,1] / [8] | **fused** → `quantizer.quantizers.0.in_project.weight` + `.bias` |
| `quantizer.quantizers.0.out_project.weight_{g,v}` + `.bias` | [1024,1,1] / [1024,8,1] / [1024] | **fused** → `…out_project.weight` + `.bias` |
| `quantizer.quantizers.0.codebook.weight` | [8192, 8] | same |

---

## 4. Emotion conditioning dataflow

Three input modalities collapse to one 1280-d `emovec`, which is **added to the
speaker latent** before the GPT prefix.

### 4.1 The always-on audio path (`merge_emovec`)

```
speaker wav  --librosa.load(sr=22050), cut 15 s--> audio (1, n)      infer_v2_5.py:626
             --Resample 22050->16000-------------> audio_16k          :628
             --SeamlessM4TFeatureExtractor-------> input_features + attention_mask  :630-634
             --Wav2Vec2BertModel, hidden_states[17]--> feat (1,T,1024) :287
             --(feat - mean)/sqrt(var)-----------> spk_cond_emb        :288   [wav2vec2bert_stats.pt]

emotion wav  --librosa.load(sr=16000), cut 15 s--> emo audio          :685
             --same two steps--------------------> emo_cond_emb       :691

emo_vec_syn_ori = emo_conditioning_encoder(x.transpose(1,2), lens)    model_v2.py:591-596
                    # ConformerEncoder 4L/512d/4h, conv2d2 subsampling
                = emo_perceiver_encoder(that, emo_cond_mask_pad(mask)) # 1 latent, dim 1024
emo_vec_syn     = emovec_layer(emo_vec_syn_ori)      # Linear 1024->1280   :829
emo_vec         = emo_layer(emo_vec_syn)             # Linear 1280->1280   :830

base_vec = get_emovec(spk_cond_emb)                                    :835
emo_vec  = get_emovec(emo_cond_emb)                                    :834
emovec   = base_vec + alpha * (emo_vec - base_vec)                     :837
```

Two behaviours worth pinning:

* **`cond_lengths` is bogus upstream.** `infer_v2_5.py:758-764` (the length arg is on :761) passes
  `spk_cond_emb.shape[-1]`, i.e. the **channel** count 1024, as the sequence
  length. `ConformerEncoder` clamps it to the real `T`, so the effect is
  "no masking at all". Reproduce this, do not fix it — a masked C++ port would
  diverge from the oracle.
* **No external emotion reference ⇒ `emo_alpha` is forced to 1.0**
  (`infer_v2_5.py:612-616`), and the emotion prompt becomes the speaker prompt.
  A CLI `--emo-alpha` without `--emo-voice` is therefore a no-op.

### 4.2 The 8-d vector path (`--emo-vector`)

```
emo_vector (8 floats, order: happy angry sad afraid disgusted melancholic surprised calm)
  -> normalize_emo_vec: elementwise * EMO_BIAS, then scale so sum <= 0.8   :488-502
  -> if emo_alpha != 1: v = trunc4(v * clamp(emo_alpha, 0, 1))             :605-611
  -> per emotion e in 0..7:
        row = argmax_i cos( style, spk_matrix[e][i] )   # use_random=False  :673
        proto[e] = emo_matrix[e][row]                                       :674-676
     # spk_matrix/emo_matrix are feat1/feat2 split by emo_num=[3,17,2,8,4,5,10,24]
  -> emovec_mat = sum_e v[e] * proto[e]                                     :677-679
  -> emovec = emovec_mat + (1 - sum(v)) * emovec_audio                      :767
```

Selecting an emotion vector *disables* the external emotion reference audio
(`infer_v2_5.py:582-585`).

### 4.3 Where it lands

```
speech_conditioning_latent = spk_emb_proj(campplus_style)        # (1,1,1280)  model_v2.py:754
conds_latent = cat( speech_conditioning_latent + emovec.unsqueeze(1),
                    zeros(B, 2, 1280) ), dim=1)                  # (1,3,1280)  model_v2.py:768
prefix       = [ pad | conds_latent | text_emb ] , then start_mel_token       model_v2.py:652-712
text_emb     = text_embedding(text_ids) + text_pos(arange) + lang_embedding(lang_id)   :679-681
```

The two zero rows in `conds_latent` are the slots the non-CAMPPlus branch fills
with `speed_emb` — in `campplus` mode they are literally zeros, but they still
occupy prefix positions and shift `mel_pos` indices. `text_ids` here are the
tokenizer ids with `start_text_token`/`stop_text_token` re-added after stripping
(`model_v2.py:675-678`) — note the padding goes on the **left**, so the attention
mask has leading zeros (`model_v2.py:688-697`).

### 4.4 The rest of the pipeline

```
codes  = GPT-2 AR, stop at 8193, truncate                        infer_v2_5.py:770-816
S_infer = semantic_codec.decode(codes)
          = up( interpolate( decoder( quantizer.vq2emb(codes) ), x2 ) )  codec/models.py:205-231
target_lengths = int(S_infer.shape[1] * 1.72 * duration_factor)  infer_v2_5.py:832
cond   = length_regulator(S_infer, ylens=target_lengths, n_quantizers=3)  :834-837
prompt_condition = length_regulator(spk_cond_emb, ylens=ref_mel.T, n_quantizers=3)  :650-655
cat_condition = cat([prompt_condition, cond], dim=1)             :839
vc     = cfm.inference(cat_condition, lens, ref_mel, style, None, 25, cfg=0.7)  :840-844
vc     = vc[:, :, ref_mel.size(-1):]                              :845
wav    = bigvgan(vc)                                              :849
```

---

## 5. Confucius4 / IndexTTS reuse map, at symbol level

### 5.1 Reused essentially unchanged — `src/confucius4_tts.cpp`

Phase 2's s2mel stage should call these with the `indextts2-s2mel-*.gguf`, which
uses the identical key family:

| symbol | what it does | change needed |
|---|---|---|
| `s2a_find` | tensor lookup in the S2A map | none |
| `s2a_read_f32`, `s2a_linear`, `s2a_silu_inplace`, `s2a_mish_inplace`, `s2a_rms_norm`, `s2a_adaln`, `s2a_conv1d_ct`, `s2a_group_norm1_ct` | CPU kernels | none |
| `s2a_sinusoidal_embed`, `s2a_timestep_embed_cpu` | `TimestepEmbedder`, `.time_mlp.{0,2}` | none — the converter emits `time_mlp` |
| `s2a_input_embed_cpu` | `cat(x, prompt_x, mu_proj, spks) → input_embed.proj`; already auto-detects `mel_dim` from `proj.weight` (`confucius4_tts.cpp:636-640`), and 512+2·80+192 = 864 falls out correctly | none |
| `s2a_dit_cache_build`, `s2a_dit_forward`, `s2a_dit_forward_cfg_fused`, `s2a_dit_run` | the 13-block uvit DiT + WaveNet + `final_layer` + `conv2` graph | none — depth/heads/dim come from KV |
| `s2a_flow_matching` | 25-step Euler with CFG 0.7 | **one change**: expose the initial noise so the harness can inject a fixed `z` (see §7 risk 3) |
| WaveNet `weight_g`/`weight_v` folding (`confucius4_tts.cpp:650-730`) | folds 17 pairs at load | none — the converter deliberately leaves those unfused |
| `s2a_build_conditioning` (the `length_regulator` half) | `content_in_proj` → 4× (Conv1d/GroupNorm/Mish) → `model.12` | **check the interpolation**: Confucius4 and IndexTTS-2.5 both use `sampling_ratios=[1,1,1,1]` and `F.interpolate(..., size=ylens.max(), mode='nearest')`, so the same code applies — but IndexTTS-2.5 calls it **twice** with different `ylens` (prompt vs target) whereas Confucius4 calls it once |
| `confucius4_bind_campplus`, `cb_campplus_*` (`src/chatterbox_campplus.h`) | 80-band kaldi fbank + CAMPPlus → 192-d | none; `compute_fbank` must subtract the per-utterance mean (`infer_v2_5.py:647`) |
| `confucius4_tts_set_w2v_path` + the sidon w2v-BERT loader | layer-17 hidden states | **change the normalisation source**: read `indextts2.w2v_bert.{mean,var}` instead of `confucius4.w2v_bert.*` |
| BigVGAN loader (`confucius4_tts_set_vocoder_path` → `indextts_voc.h` 22 kHz path) | mel-80 → 22.05 kHz PCM | none |

### 5.2 Reused with new dims — `src/indextts.cpp`

| symbol | reuse | change |
|---|---|---|
| `build_cond_enc_graph` | the ConformerEncoder graph (`conv2d2` subsampling, rel-pos MHA with `pos_bias_u/v`, macaron FFN, conv module) | rebind to `emo_cond_enc.*`; **4** blocks not 6, **4** heads not 8, FFN **1024** not 2048, input **1024**-d w2v-BERT features not 100-mel — so `embed.conv.0` sees a 1024-wide freq axis and `embed.out.0` is `(512, 261632)` |
| `build_perceiver_graph` | PerceiverResampler (RMSNorm, GEGLU FF, cross-attn with `to_q`/`to_kv`/`to_out`, `cross_attn_include_queries=True`) | rebind to `emo_perc.*`; **1** latent not 32, dim **1024** not 1280, context **512**, still **2** layers; output is `.squeeze(1)` → a single 1024-vector |
| `build_gpt2_kv_graph`, `run_gpt2_kv`, `kv_alloc` | 24L/1280d/20h GPT-2 with KV cache | none structurally; the prefix builder changes |
| `build_prefill_embeds` | assembles `[cond][text]` | rewrite: prefix is `[left-pad][spk_emb_proj(style)+emovec][0][0][text_emb]`, and `text_emb` gains `+ lang_embedding(lang_id)` |
| `build_mel_token_embed` | `mel_embedding(code) + mel_pos(step)` | none |
| `sample_rng`, top-k/top-p/repetition penalty | AR sampling | none |
| `compute_ref_mel` | 22.05 kHz mel | reuse the Confucius4/`core/mel.h` 22 kHz `center=false` path, **not** the 1.5 24 kHz/100-mel one |
| `maybe_external_normalize`, `preprocess_indextts_text` | the `INDEXTTS_HAS_SUBPROCESS` normaliser hook | reuse for ja/es/ar TN |
| `is_cjk_codepoint`, `utf8_decode/encode` | text helpers | none |

### 5.3 `src/core/` helpers phase 2 should call rather than re-write

| header | namespace | use here |
|---|---|---|
| `core/gguf_loader.h` | `core_gguf` | the multi-file tensor map (4 GGUFs) |
| `core/bpe.h` | `core_bpe` | byte-level BPE driver — needs a tiktoken-rank front end, see §7 risk 1 |
| `core/unicode_categ.h` | `core_unicode` | the `\p{L}` / `\p{N}` classes the tiktoken regex needs |
| `core/kaldi_fbank.h` | `core_kaldi` | CAMPPlus's 80-band fbank |
| `core/mel.h` | `core_mel` | the 22.05 kHz, `center=false` prompt mel |
| `core/adaln.h` | `core_adaln` | AdaLN modulation in the DiT |
| `core/conv.h` | `core_convt` | length-regulator + WaveNet + ConvNeXt conv1d |
| `core/attention.h` | `core_attn` | Conformer / Perceiver / DiT attention |
| `core/cpu_ops.h` | `core_cpu` | CPU kernels for the non-graph paths |
| `core/rvq.h` | `core_rvq` | *encode* only — the codec's `decode` is a plain embedding lookup, so this is needed only if `EnhancedCodec.quantize` is ever ported (§5.4 item 4) |
| `core/torch_rng.h` | `crispasr` | seeded noise that reproduces `torch.randn` for the CFM `z` |
| `core/tts_ref_cache.h` | `crispasr_ref_cache` | cache the speaker-derived tensors across calls — exactly what `infer_v2_5.py:611-696` does |
| `core/tts_voice_policy.h` | `core_tts_voice` | `--i-have-rights` consent gate |
| `core/crispasr_watermark.h` | `crispasr_wm` | audio watermark |
| `core/crispasr_c2pa.h` | — | C2PA manifest |
| `src/chatterbox_campplus.h` | `cb_campplus_*` structs | the CAMPPlus encoder itself |

The last three are the shipped TTS provenance chain and are **mandatory** for a
new TTS backend, not optional.

### 5.4 Genuinely new C++ (~700 LOC)

1. **`EnhancedCodec.decode`** (~250 LOC): `codebook.weight[code]` → fused
   `out_project` 1×1 → 12 ConvNeXt blocks (dwconv k7 → LN → pwconv1 → GELU →
   pwconv2 → `gamma` scale → residual) → `final_layer_norm` → Linear(384→1024) →
   nearest-neighbour ×2 upsample → `up` Conv1d. `src/outetts_wavtok.cpp:556-570`
   (`convnext_block`) and `src/f5_tts.cpp:1810` are the closest existing graphs —
   both have an AdaNorm variant that must be dropped here
   (`adanorm_num_embeddings=None`).
2. **tiktoken BPE + the 1 673-special table** (~250 LOC).
3. **`emo_matrix`/`spk_matrix` prototype selection** (~80 LOC): cosine-argmax per
   emotion group, bias + 0.8 cap, the `(1 - Σw)` blend.
4. **`EnhancedCodec.quantize`** — only if the C++ ever needs prompt semantic
   codes. Note `infer_v2_5.py:638` computes `S_ref` and then **never uses it**;
   the length regulator is fed `spk_cond_emb` (raw w2v-BERT features), not codes
   (`infer_v2_5.py:650-655`, with the `S_ref` argument commented out on :651). So the codec **encoder** is dead at inference and
   need not be ported — but it is still converted, because the diff harness wants
   it.

---

## 6. Reference oracle

`tools/reference_backends/indextts2.py`, stage-checkpointed to `<out-dir>/<stage>.npy`.

| stage group | outputs | where it runs | state |
|---|---|---|---|
| `text` | `text_tokens`, `lang_id` | VPS | **run** — `[58838, 7627, 11, 341, 307, 257, 6316, 11193, 13, 1]`, i.e. `<\|en\|>` (58 836+2) … `stop_text_token`=1; `lang_id`=0 |
| `campplus` | `fbank`, `style` | VPS | **run** — `fbank (1098, 80)` mean 0.0, `style (192,)` ‖·‖ = 13.48 |
| `mel` | `ref_mel` | VPS | **run** — `(80, 947)` for the 11 s `samples/jfk.wav` |
| `w2v` | `spk_cond_emb`, `emo_cond_emb` | VPS (≈2.4 GB) | **run**, both `(549, 1024)` |
| `lr_prompt` | `prompt_condition` | VPS | ready; not yet run (the box was at load 13 with swap exhausted) |
| `emovec` | `emovec`, `emovec_mat`, `spk_latent` | **Kaggle** — needs `transformers==4.52.1` | ready, blocked locally |
| `gpt` | `gpt_prefix_embeds`, `semantic_codes` | **Kaggle** — 3.26 GB fp32 + 1 500 AR steps | ready, blocked locally |
| `codec` | `s_infer` | Kaggle (or VPS alone) | ready |
| `lr_cond` | `cond`, `cat_condition` | Kaggle | ready |
| `s2mel` | `cfm_noise`, `s2mel_mel` | Kaggle | ready |
| `bigvgan` | `audio` | Kaggle | ready |

Kaggle kernel: `tools/kaggle/indextts2-refdump/` (id
`chr1s4/crispasr-indextts2-refdump`, GPU, internet, `chr1s4/crispasr-hf-token`).
It clones CrispASR + index-tts `v2.5.0`, pins `transformers==4.52.1`, downloads
the checkpoints minus QwenEmotion plus the three aux models, runs `--stages all`,
bundles the `.npy` set into `indextts2-ref.npz`, does an ASR round-trip on the
synthesised WAV, and uploads to the private `cstr/crispasr-regression-fixtures`.
**Not pushed and not run** — that is the operator's call.

**The `transformers==4.52.1` pin is not optional.** Upstream vendors its own copy
of HF's generation utils; `indextts/gpt/transformers_generation_utils.py:28`
imports `OffloadedCache`, which transformers 5.x removed, so importing
`indextts.gpt.model_v2` — and therefore `indextts.infer_v2_5` — raises
`ImportError` on anything newer. Only the `emovec` and `gpt` stages touch that
module; the dumper's other stages import the leaf modules directly and are
unaffected, which is why `apply_pronunciation_annotations` and
`find_most_similar_cosine` are transcribed verbatim into the dumper instead of
imported from `infer_v2_5`.

Determinism levers: GPT-2 forced to `do_sample=False, num_beams=1,
repetition_penalty=1.0`; the CFM's `z` drawn from a seeded `torch.Generator` and
dumped as `cfm_noise`; `use_random=False` for prototype selection; text
normalisation off by default (so `wetext`/`nemo_text_processing` never load).

Gate, per house rules: cos ≥ 0.999 per stage plus an ASR round-trip on the final
WAV, before any parity claim.

---

## 7. The three riskiest points for phase 2

### Risk 1 — the tiktoken tokenizer (highest, and it fails silently)

`core/bpe.h` implements byte-level BPE keyed on a `"a b"` merge-rank table.
tiktoken is a *different* algorithm: base64-keyed mergeable ranks, a regex
pre-tokenizer (`utils/tokenizer.py:215`) that must be applied per-chunk, and
1 673 special tokens injected at ids ≥ 58 836 that `allowed_special='all'` lets
match **anywhere in the string** — including the `<|zh|>` prefix and the
`<|SPECIAL_TOKEN_1|>` pronunciation wrappers. Three specific traps:

* The regex needs `\p{L}` / `\p{N}` Unicode property classes. `std::regex` has
  neither; a hand-rolled classifier over `core/unicode_categ.h` is required, and
  a wrong class boundary silently changes the merge chunking for CJK.
* The specials list ordering — in particular the *six*-element
  `translate/transcribe/startoflm/startofprev/nospeech/notimestamps` block — must
  match exactly, or every language token is off by one and conditioning shifts to
  the wrong language with no error.
* A tokenizer that is merely *close* still produces fluent-sounding audio, so this
  will not show up in listening tests. Gate it on an exact id-sequence match
  against `text_tokens.npy` for a multilingual corpus (zh + ja + es + ar), not on
  a cosine.

### Risk 2 — w2v-BERT layer-17 parity, because one drift corrupts two consumers

`spk_cond_emb` feeds **both** `merge_emovec` (→ the emotion vector added to the
GPT prefix) **and** `length_regulator` (→ `prompt_condition`, the first half of
the DiT conditioning). A small drift therefore shows up simultaneously as wrong
prosody and wrong timbre, and the two symptoms mask each other. Contributing
factors:

* The feature extractor is `SeamlessM4TFeatureExtractor`, not a plain log-mel.
  `facebook/w2v-bert-2.0/preprocessor_config.json` says `num_mel_bins: 80`,
  `stride: 2` (so `input_features` are **160**-d stacked pairs),
  `return_attention_mask: true`, `padding_side: "right"` and — the trap —
  **`padding_value: 1`, not 0**. It also does per-utterance mean/var
  normalisation. Odd frame counts are dropped by the stride-2 stacking, so the
  tail behaviour differs from a naive reshape.
* Normalisation is `(feat - mean) / sqrt(var)` with `var` (not `std`) on disk —
  the converter bakes `var` to stay diffable against
  `confucius4.w2v_bert.var`, so the C++ must take the square root itself.
* The audio path differs per consumer: the speaker prompt arrives via
  `librosa.load(default sr=22050) → resample 16 k`, the emotion prompt via
  `librosa.load(sr=16000)` directly. Those are **not** the same 16 kHz signal for
  the same file, and the reference cut is 15 s at *each* rate.
  Measured on `samples/jfk.wav` with **no** `--emo-voice` (so both branches read
  the *same* file): `spk_cond_emb` and `emo_cond_emb` are both `(549, 1024)`,
  both z-normalised (mean ≈ −0.0009, std ≈ 1.036), and they share a min of
  −9.689 — but their maxima differ (9.316 vs 9.339). The two resampling routes
  really do produce different features from one file. A C++ port that computes
  the 16 kHz signal once and reuses it will look right on every summary
  statistic and still be wrong.

Diff `spk_cond_emb` first, before anything downstream, and diff the extractor's
`input_features` too, not just the layer-17 output.

### Risk 3 — the CFM noise contract and the CFG batching

`solve_euler` (`flow_matching.py:57-115`) does three things that are easy to get
subtly wrong and that only manifest as "slightly noisy" audio:

* `x[..., :prompt_len] = 0` is re-applied **at the end of every step**
  (`:113`), not once — and `prompt_x` keeps the reference mel in those columns
  the whole time. Dropping the per-step re-zero drifts slowly.
* The dt schedule is off-by-one on purpose: `dt` for step *k* is computed from
  `t_span[k] - t_span[k-1]`, then **overwritten** to `t_span[k+1] - t` at the end
  of the body (`:111-112`) while `t` has already advanced. Transcribe the loop
  literally.
* CFG runs as a **single batched forward over `cat([cond, uncond])`** with the
  style, mu and prompt all zeroed in the second half (`:88-98`).
  `s2a_dit_forward_cfg_fused` already does exactly this, so the reuse is free —
  but the C++ must not "optimise" it into two passes, because the fused batch
  shares the attention mask and any per-batch normalisation.

And for the harness specifically: `torch.randn` inside `BASECFM.inference`
(`:52`) is the only source of nondeterminism left in the acoustic stage.
`confucius4_tts.cpp`'s `s2a_flow_matching` currently draws its own noise; phase 2
must add a "load `z` from `CRISPASR_INDEXTTS2_NOISE`" hook (mirroring
`CRISPASR_CONFUCIUS4_COND_DIR`) or every s2mel diff is meaningless.

---

## 8. Phase-2 shopping list (not started)

* `src/indextts2_tts.{h,cpp}` — a **separate** backend; leave `indextts` (=1.5) alone.
* Registry entry next to `crispasr_model_registry.cpp:1026` (the Confucius4 row),
  primary = `indextts2-gpt-q4_k.gguf`, companion = `indextts2-s2mel-q4_k.gguf`,
  plus `set_codec_path` / `set_w2v_path` / `set_vocoder_path` for the rest.
  The third-and-beyond files use the `ExtraCompanion` sibling-discovery mechanism
  (`crispasr_model_registry.cpp:1436-1443` is exactly this shape for Confucius4:
  BigVGAN + w2v-BERT ride along next to the primary) — and here two of the three
  entries can point at the **existing** `cstr/confucius4-tts-GGUF` URLs, since the
  vocoder and w2v-BERT GGUFs are the same upstream models. Only
  `indextts2-codec-*.gguf` needs a new source, and it has **no `cstr/` URL** until
  the licence question is settled: point it at the local conversion step.
* CLI: `--voice`, `--emo-voice`, `--emo-alpha`, `--emo-vector h,a,s,af,d,m,su,c`,
  `--duration-factor`, `--lang`, all behind `CRISPASR_*` env gates.
* `tools/format.sh --fix` on every changed C/C++ file before commit.
