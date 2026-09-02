# IndexTTS-2.5 — scoping for issue #346

Sources read: `github.com/index-tts/index-tts` **tag `v2.5.0`** (the `indextts-2.5` branch named in
the issue no longer exists — it was merged to `main` and tagged) cloned to
`/mnt/volume1/tmp-overflow/indextts25-src`; HF `IndexTeam/IndexTTS-2.5` `config.yaml` +
blob listing; our `src/indextts.cpp`, `src/indextts_voc.cpp`,
`models/convert-indextts-to-gguf.py`, `src/confucius4_tts.{h,cpp}`.

## 1. Verdict up front

**It is a new architecture, not a new checkpoint.** Of our shipped IndexTTS pipeline only the
GPT-2 backbone shape survives; the tokenizer, the conditioning, the acoustic stage and the
vocoder are all replaced. Only ~25–30% of `indextts.cpp`/`indextts_voc.cpp` transfers.

**But** we already shipped almost the whole *new* stack under a different name: **Confucius4-TTS
(#377)** is the same lineage. `src/confucius4_tts.cpp:600-603` declares `estimator_depth=13`,
`estimator_num_heads=8`, `estimator_hidden_dim=512`, `wavenet_num_layers=8`, mel=80 @ 22050 —
identical, field for field, to IndexTTS-2.5's `s2mel:` block in `config.yaml`
(`DiT.depth: 13`, `hidden_dim: 512`, `num_heads: 8`, `wavenet.num_layers: 8`, `in_channels: 80`,
`preprocess_params.sr: 22050`, `style_encoder.dim: 192`). Confucius4 also already has native
w2v-BERT-2.0 layer-17, CAMPPlus, an InterpolateRegulator and BigVGAN-v2-22 kHz
(`src/confucius4_tts.h:6-23`). That is exactly IndexTTS-2.5's back half.

## 2. Version-diff table

| | **IndexTTS-1.5 (what we implement)** | **IndexTTS-2.5** |
|---|---|---|
| Repo/tag | Apache-2.0 | tag `v2.5.0`, **bilibili Model Use License** (`LICENSE`) |
| Entry point | `indextts/infer.py` | `indextts/infer_v2_5.py` (imports `UnifiedVoice` from `gpt/model_v2.py`, **not** `model_v2_5.py`, which is dead code — nothing references it) |
| AR backbone | GPT-2 24L/1280d/20h/FFN5120 (`indextts.cpp:16-20`) | **same shape** (`config.yaml gpt: layers 24, model_dim 1280, heads 20`) |
| Text vocab | 12 001 (SentencePiece `bpe.model`) | **60 509**, whisper/tiktoken BPE + `gpt2.tiktoken` (`utils/tokenizer.py:180-189, 279`) + `<\|zh\|>/<\|en\|>` lang prefix (`infer_v2_5.py:699`) + `lang_embedding` added to text emb (`model_v2.py:388-390, 680-681`) |
| Mel/semantic vocab | 8194 mel codes | 8194 **semantic** codes (same numbers, different meaning) |
| max_mel_tokens | 800 | 1815 |
| Speaker conditioning | mel-100 → Conformer 6L/512 → Perceiver 32 latents (`indextts.cpp:1072,1237`) — wired and working; the "dummy zeros" header comment at `indextts.cpp:7,12` is stale | **CAMPPlus 192-d → `spk_emb_proj` Linear(192→1280)**; `spk_cond_mode="campplus"` (`infer_v2_5.py:138`, `model_v2.py:740-755`). The conformer/perceiver path is built but bypassed for the speaker |
| Emotion conditioning | none | see §3 |
| Acoustic stage | **none** — GPT hidden states go straight to the vocoder | **3 new stages**: `EnhancedCodec.decode` → `s2mel.length_regulator` → `s2mel.cfm` flow-matching DiT (`infer_v2_5.py:831-845`) |
| Semantic codec | – | `EnhancedCodec` = ResidualVQ(8192×8) + 2× `VocosBackbone`(384d/12L) + ×2 down/upsample (`codec/models.py:29,102-135,205`) |
| Acoustic model | – | CFM, 25 Euler steps, `inference_cfg_rate=0.7` (`infer_v2_5.py:829-830`), `s2mel/modules/flow_matching.py:31-115`; estimator = DiT 13L/512d/8h + WaveNet 8L final layer |
| Vocoder | BigVGAN over **GPT latents** (1280-in), 24 kHz, ups `[4,4,4,4,2,2]` (`indextts_voc.cpp:1-28`) | stock `nvidia/bigvgan_v2_22khz_80band_256x` over **mel-80**, 22.05 kHz, ups `[4,4,2,2,2,2]` |
| Languages | zh/en | zh/en/ja/es/ar + Japanese G2P + NeMo TN (`infer_v2_5.py:25-26, 704-717`) |
| Duration control | none | `duration_factor`, `target_len = S_infer.T * 1.72 * f` (`infer_v2_5.py:832`) |
| Weights | gpt 1.17 GB + bigvgan 536 MB | gpt **3.26 GB**, codec 607 MB, s2mel 415 MB, QwenEmotion 1.19 GB = **5.49 GB**, plus external w2v-BERT-2.0 **2.32 GB**, BigVGAN 449 MB, CAMPPlus 28 MB → **~8.3 GB fp32** |

## 3. The emotion / dubbing conditioning the issue praises

Three input modalities that all collapse to one 1280-d `emovec`, **added to the speaker latent**
before the GPT prefix (`model_v2.py:768`: `cat(spk_latent + emo_vec.unsqueeze(1), zeros(B,2,d))`):

1. **Reference audio** — w2v-BERT-2.0 hidden_states[17], mean/var-normalised by `wav2vec2bert_stats.pt`
   (`infer_v2_5.py:281-289`) → `emo_conditioning_encoder` Conformer **4L/512d/4h** →
   `emo_perceiver_encoder` PerceiverResampler(1024, **1 latent**) (`model_v2.py:375-388`) →
   `emovec_layer` Linear(1024→1280) → `emo_layer` Linear(1280→1280) (`model_v2.py:391-392, 827-831`).
   Blended against the *speaker's own* emo vector by `alpha`:
   `base + alpha*(emo - base)` (`model_v2.py:833-838`).
2. **Explicit 8-d vector** `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`
   (`infer_v2_5.py:492-502`). Prototype rows are looked up in `emo_matrix` (`feat2.pt`, split by
   `emo_num: [3,17,2,8,4,5,10,24]`) picking, per emotion, the row whose `spk_matrix` (`feat1.pt`)
   CAMPPlus style is cosine-nearest to the reference speaker (`infer_v2_5.py:668-679`), then
   `emovec = Σwᵢ·protoᵢ + (1-Σw)·emovec_audio` (`infer_v2_5.py:767`).
3. **Free text** — QwenEmotion (Qwen3-0.6B fine-tune, 1.19 GB) emits the 8-d JSON vector
   (`infer_v2_5.py:596-599, 1007-1059`). Optional; `use_qwen_emo=False` by default.

So: **no dubbing-specific module.** The "dubbing quality" is emotion-timbre disentanglement —
timbre from CAMPPlus + s2mel prompt-mel, prosody/emotion from a separate scalar-mixable vector.

## 4. Reuse estimate

| Component | Source | Status |
|---|---|---|
| GPT-2 AR + KV cache + sampling | `src/indextts.cpp` | reuse, retarget prefix layout |
| Conformer encoder + PerceiverResampler | `src/indextts.cpp:970-1362` | reuse **with new dims** (4L/512/4h, 1 latent, input 1024 not mel-100) |
| CAMPPlus 192-d | `src/chatterbox_campplus.h` (used by confucius4, cosyvoice3) | reuse as-is |
| w2v-BERT-2.0 layer-17 GGUF | confucius4 `set_w2v_path` | reuse as-is |
| InterpolateRegulator | confucius4 S2A | reuse; check `n_quantizers=3`, `is_discrete:false`, `in_channels:1024` |
| CFM DiT 13L/512 + WaveNet, 25-step Euler, CFG 0.7 | confucius4 S2A | reuse; hparams already identical |
| BigVGAN v2 22 kHz mel-80 | confucius4 vocoder GGUF | reuse as-is (**not** our `indextts_voc.cpp`, which is latent-in/24 kHz) |
| Vocos backbone | `src/f5_tts.cpp`, `src/outetts_wavtok.cpp` | adapt for the codec decoder |
| **New:** `EnhancedCodec` RVQ decode (8192×8, ×2 upsample, 2× Vocos 384/12) | – | ~300 LOC |
| **New:** tiktoken/whisper BPE + lang tokens + emo_matrix/spk_matrix selection | – | ~400 LOC |
| **New:** ja/es/ar text normalisation | – | route through the existing external normaliser hook (`INDEXTTS_HAS_SUBPROCESS`) |
| **Skip:** QwenEmotion | – | expose the 8-d vector on the CLI; text→emotion is optional |

Net: ~65–70% of the runtime already exists in-tree, but almost none of it in `indextts.cpp`.

## 5. Converter + runtime delta sketch

- New `models/convert-indextts2-to-gguf.py` — do **not** extend the 1.5 converter; source
  checkpoints, tensor names and tensor count all differ.
  - `indextts2-gpt.gguf` — GPT-2 + `spk_emb_proj` + `lang_embedding` + `emo_conditioning_encoder`
    + `emo_perceiver_encoder` + `emovec_layer`/`emo_layer` + tiktoken vocab + `feat1.pt`/`feat2.pt`
    baked as `emo_matrix`/`spk_matrix` tensors + `emo_num` as a KV array.
  - `indextts2-codec.gguf` — `codec.pth`: RVQ codebook + both Vocos backbones.
  - `indextts2-s2mel.gguf` — `s2mel.pth`: length regulator + DiT + WaveNet + CAMPPlus.
    weight_norm fusion already implemented (`convert-indextts-to-gguf.py:fuse_weight_norm`,
    and the same pairs list exists in `confucius4_tts.cpp:653-663`).
  - Reuse confucius4's `w2v` and `bigvgan-22k` GGUFs verbatim — same upstream repos.
- New `src/indextts2_tts.{h,cpp}` as a **separate backend** (`indextts2`), leaving `indextts`
  (=1.5) untouched. Registry entry alongside `crispasr_model_registry.cpp:1061`.
- CLI: `--voice`, `--emo-voice`, `--emo-alpha`, `--emo-vector h,a,s,af,d,m,su,c`,
  `--duration-factor`, `--lang`. All behind `CRISPASR_*` env gates per house rules.

## 6. Validation plan (diff harness, mandatory)

Five reference dumps from `infer_v2_5.py` on one fixed `(ref.wav, text, seed)`, compared stage
by stage with `crispasr-diff`:
1. w2v-BERT layer-17 post-normalisation features (`get_emb`) — feeds **two** consumers.
2. `emovec` after `merge_emovec`, and the 8-d-vector path with `use_random=False`.
3. GPT semantic codes — greedy (`do_sample=False`, `num_beams=1`) for determinism; note upstream
   defaults are `num_beams=3, repetition_penalty=10.0, top_k=30, top_p=0.8` (`infer_v2_5.py:731-739`).
4. `EnhancedCodec.decode` output, then `length_regulator` output at fixed `target_lengths`.
5. CFM mel with `n_timesteps=25, cfg=0.7` from a **fixed noise tensor**, then BigVGAN PCM.
Gate: cos ≥ 0.999 per stage, plus an ASR round-trip on the final WAV (mandatory per house rules).
Ref dumper must match C++ audio conditioning exactly — 15 s truncation (`_load_and_cut_audio`),
22.05 k + 16 k resample, `center=False` mel.

## 7. Effort + recommendation

**Effort: ~14–20 focused days.** Converter 3, GPT+emotion front half 5–7, codec decode 3,
s2mel wiring on top of confucius4 4, diff harness + parity chase 3–4. Model size is fine for
Q4_K (~2.5 GB total) but **all validation must run on Kaggle** — fp32 refs are 8.3 GB and the
VPS has 8 GB RAM.

**Recommendation: GO on the runtime, DEFER on hosting converted weights.**
Ship `src/indextts2_tts.cpp` + the converter; point users at a local conversion step rather than
publishing `cstr/indextts-2.5-GGUF` until the license question is settled.

**Biggest risk: licensing, not engineering.** `LICENSE` at `v2.5.0` defines "Model" as
"model weights **and final code**" (§1.4), so unlike 1.5 the *inference code* is covered too.
A GGUF is a "Derivative Work" (§1.5(iii) names quantization explicitly), which triggers:
carry the licence + copyright notice in every copy (§3.4(b)), publish the prescribed disclaimer
sentence (§4.1(a)), contractually bind downstream recipients (§3.4(a)), the 100 M-MAU / RMB 1 bn
carve-out (§2.2), the high-risk-deployment ban (§4.2), and §3.4(c) — may not be used to improve
any AI model other than indextts2 itself or non-commercial models. That last clause is the one
that does not compose cleanly with an MIT-licensed toolkit.
Biggest *technical* risk: w2v-BERT layer-17 parity, because one drift there corrupts both the
emotion vector and the s2mel prompt condition simultaneously.

## 8. Suggested 3-sentence reply to the reporter

> To be clear, nothing here was a rejection — the earlier comment was only flagging that our
> current backend is 1.5-specific and that 2.5 ships under bilibili's Model Use License rather
> than Apache-2.0, which changes what we can *redistribute*, not what we can implement.
> Contributions are genuinely welcome, and I've now scoped 2.5 properly: it's a new architecture
> (CAMPPlus + w2v-BERT conditioning, an emotion vector, a semantic codec and a flow-matching
> s2mel stage), but roughly two thirds of it already exists in-tree from our Confucius4-TTS
> backend, whose DiT/WaveNet/BigVGAN hyperparameters are identical to IndexTTS-2.5's.
> If you want to take a run at it, open a draft PR against a `feat/indextts2` branch and I'll
> help with the converter and the stage-by-stage diff harness — the one thing to keep separate
> is publishing pre-converted GGUFs, since those are "Derivative Works" under bilibili's terms
> and need the licence notice, disclaimer and downstream flow-down attached.
