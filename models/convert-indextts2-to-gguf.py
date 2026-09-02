#!/usr/bin/env python3
"""
Convert IndexTTS-2.5 checkpoints to GGUF for the CrispASR ``indextts2`` TTS backend.

IndexTTS-2.5 (``IndexTeam/IndexTTS-2.5``, upstream tag ``v2.5.0``) is a *different*
architecture from IndexTTS-1.5 — do NOT extend ``convert-indextts-to-gguf.py``.
Only the GPT-2 backbone *shape* is shared; the tokenizer (tiktoken, 60 509),
the speaker conditioning (CAMPPlus, not conformer/perceiver), the acoustic stage
(semantic codec + flow-matching s2mel) and the vocoder (stock BigVGAN-v2 22 kHz)
are all new.

Three GGUFs are produced (plus two files that are *reused verbatim* from the
Confucius4-TTS conversion — see "Not converted here" below):

  indextts2-gpt-{f16,f32}.gguf      arch ``indextts2.gpt``
      gpt.pth (3.26 GB fp32) → GPT-2 24L/1280d/20h + ``spk_emb_proj`` (192→1280)
      + ``lang_embedding`` (107×1280) + the emotion Conformer(4L/512d/4h) +
      emotion PerceiverResampler(1 latent, 1024d) + ``emovec_layer``/``emo_layer``
      + the tiktoken BPE vocab (ranks + the ordered special-token list) +
      ``feat1.pt``/``feat2.pt`` baked as ``spk_matrix``/``emo_matrix`` tensors +
      ``emo_num`` as a KV array + the w2v-BERT mean/var normalisation stats.

      Tensor names deliberately mirror ``convert-indextts-to-gguf.py``'s shortened
      family (``cond_enc.``/``perc.``/``sa.``/``ff.``/``conv.``) so that
      ``src/indextts.cpp``'s Conformer + PerceiverResampler binders are reusable
      with only an ``emo_`` prefix and new dims.

  indextts2-codec-{f16,f32}.gguf    arch ``indextts2.codec``
      codec.pth → ``EnhancedCodec``: down/up Conv1d(1024), 2× VocosBackbone
      (384d, 12 ConvNeXt layers, intermediate 2048) and the factorised RVQ
      (1 quantizer, codebook 8192×8, l2-normalised, weight-normed in/out 1×1 convs).
      Names are kept verbatim from the checkpoint.

  indextts2-s2mel-{f16,f32}.gguf    arch ``indextts2.s2mel``
      s2mel.pth → flow-matching CFM DiT (13L/512d/8h) + WaveNet(8L, k=5) +
      InterpolateRegulator, **renamed into the exact GGUF key family that
      ``tools/kaggle/confucius4-tts-convert/confucius4-tts-convert.py`` emits for
      the byte-for-byte identical Confucius4 S2A block** (``decoder.estimator.*``,
      ``length_regulator.*``), so ``src/confucius4_tts.cpp``'s S2A code path can be
      reused mechanically in phase 2.  CAMPPlus (funasr/campplus,
      ``campplus_cn_common.bin``) is baked under ``campplus.*``, same as Confucius4.

Not converted here — reuse the Confucius4 GGUFs verbatim (same upstream repos):

  * BigVGAN v2 22 kHz / 80-band / 256× — ``nvidia/bigvgan_v2_22khz_80band_256x``
    → ``cstr/confucius4-tts-GGUF/confucius4-tts-bigvgan-22k-f16.gguf`` (224.6 MB)
  * w2v-BERT 2.0 encoder (layer-17 features) — ``facebook/w2v-bert-2.0``
    → ``cstr/confucius4-tts-GGUF/confucius4-tts-w2v-f16.gguf`` (823.5 MB)

  Pass ``--check-companions`` to verify both are reachable before converting.

LICENSE
-------
IndexTTS-2.5 ships under bilibili's *Model Use License*, whose §1.4 defines
"Model" to include the **final code**, and whose §1.5(iii) names quantization as a
Derivative Work.  A GGUF produced by this script is therefore a Derivative Work:
carry the licence + copyright notice with every copy (§3.4(b)), publish the
prescribed disclaimer (§4.1(a)), bind downstream recipients (§3.4(a)), and honour
the §3.4(c) no-other-model-improvement clause.  **Do not upload the outputs of
this script to a public model hub** — convert locally.

Usage
-----
    python models/convert-indextts2-to-gguf.py \\
        --model-dir /mnt/storage/gguf-models/indextts25-src \\
        --output-dir /mnt/storage/gguf-models \\
        [--outtype f16|f32] [--only gpt,codec,s2mel] \\
        [--campplus /path/to/campplus_cn_common.bin]

Memory: every checkpoint is opened with ``torch.load(..., mmap=True)`` and written
one tensor at a time, so peak RSS stays well under 1 GB even for the 3.26 GB
gpt.pth.  Never materialise a full state dict here — the reference VPS has 8 GB.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")


# ---------------------------------------------------------------------------
# Constants — every value line-cited against the upstream v2.5.0 tree.
# ---------------------------------------------------------------------------

# config.yaml gpt:
GPT_MODEL_DIM = 1280            # config.yaml:14
GPT_LAYERS = 24                 # config.yaml:20
GPT_HEADS = 20                  # config.yaml:17
GPT_MAX_MEL_TOKENS = 1815       # config.yaml:15
GPT_MAX_TEXT_TOKENS = 600       # config.yaml:16
GPT_NUMBER_TEXT_TOKENS = 60509  # config.yaml:21
GPT_NUMBER_MEL_CODES = 8194     # config.yaml:22
GPT_START_MEL_TOKEN = 8192      # config.yaml:23
GPT_STOP_MEL_TOKEN = 8193       # config.yaml:24
GPT_START_TEXT_TOKEN = 0        # config.yaml:25
GPT_STOP_TEXT_TOKEN = 1         # config.yaml:26
GPT_MEL_LENGTH_COMPRESSION = 1024  # config.yaml:19
# LearnedPositionEmbeddings sizes (gpt/model_v2.py:398-400):
#   mel  = max_mel_tokens + 2 + max_conditioning_inputs = 1815 + 2 + 1 = 1818
#   text = max_text_tokens + 2                          =  600 + 2 =  602
GPT_MEL_POS_SIZE = GPT_MAX_MEL_TOKENS + 2 + 1
GPT_TEXT_POS_SIZE = GPT_MAX_TEXT_TOKENS + 2

# emo_condition_module (config.yaml:36-42) — the *emotion* Conformer/Perceiver.
EMO_COND_OUTPUT_SIZE = 512      # config.yaml:37
EMO_COND_LINEAR_UNITS = 1024    # config.yaml:38
EMO_COND_ATTENTION_HEADS = 4    # config.yaml:39
EMO_COND_NUM_BLOCKS = 4         # config.yaml:40
EMO_COND_INPUT_LAYER = "conv2d2"  # config.yaml:41
EMO_PERCEIVER_MULT = 2          # config.yaml:42
EMO_PERCEIVER_NUM_LATENTS = 1   # gpt/model_v2.py:384
EMO_PERCEIVER_DIM = 1024        # gpt/model_v2.py:381
EMO_PERCEIVER_DEPTH = 2         # gpt/perceiver.py:227 default; model_v2.py:381 does
                                # NOT pass `depth` — confirmed by layers.0 + layers.1
EMO_PERCEIVER_DIM_HEAD = 64     # gpt/perceiver.py:230 default → 4*64 = 256 inner
EMO_INPUT_SIZE = 1024           # gpt/model_v2.py:375 (w2v-BERT hidden dim)

# semantic_codec (config.yaml:43-49) + EnhancedCodec defaults (codec/models.py:38-39)
CODEC_CODEBOOK_SIZE = 8192      # config.yaml:44
CODEC_HIDDEN_SIZE = 1024        # config.yaml:45
CODEC_CODEBOOK_DIM = 8          # config.yaml:46
CODEC_VOCOS_DIM = 384           # config.yaml:47
CODEC_VOCOS_INTERMEDIATE = 2048  # config.yaml:48
CODEC_VOCOS_NUM_LAYERS = 12     # config.yaml:49
CODEC_NUM_QUANTIZERS = 1        # codec/models.py:38 (cfg has no num_quantizers)
CODEC_DOWNSAMPLE_SCALE = 2      # codec/models.py:39 (cfg has no downsample_scale)

# s2mel (config.yaml:50-104)
S2MEL_SR = 22050                # config.yaml:52
S2MEL_N_FFT = 1024              # config.yaml:54
S2MEL_WIN_LENGTH = 1024         # config.yaml:55
S2MEL_HOP_LENGTH = 256          # config.yaml:56
S2MEL_N_MELS = 80               # config.yaml:57
S2MEL_FMIN = 0                  # config.yaml:58
S2MEL_STYLE_DIM = 192           # config.yaml:63
LR_CHANNELS = 512               # config.yaml:65
LR_IN_CHANNELS = 1024           # config.yaml:67
LR_CODEBOOK_SIZE = 2048         # config.yaml:68
LR_SAMPLING_RATIOS = 4          # config.yaml:69 -> [1,1,1,1]
DIT_HIDDEN_DIM = 512            # config.yaml:76
DIT_NUM_HEADS = 8               # config.yaml:77
DIT_DEPTH = 13                  # config.yaml:78
DIT_IN_CHANNELS = 80            # config.yaml:81
DIT_CONTENT_DIM = 512           # config.yaml:85
WAVENET_HIDDEN_DIM = 512        # config.yaml:99
WAVENET_NUM_LAYERS = 8          # config.yaml:100
WAVENET_KERNEL_SIZE = 5         # config.yaml:101
WAVENET_DILATION_RATE = 1       # config.yaml:102

# Inference-time constants (infer_v2_5.py)
DIFFUSION_STEPS = 25            # infer_v2_5.py:829
INFERENCE_CFG_RATE = 0.7        # infer_v2_5.py:830
LENGTH_RATIO = 1.72             # infer_v2_5.py:832  target_len = S.T * 1.72 * duration_factor
LR_N_QUANTIZERS = 3             # infer_v2_5.py:654 (prompt), 836 (target)
MAX_REF_AUDIO_SECONDS = 15      # infer_v2_5.py:626, 642, 685 (_load_and_cut_audio)
SILENT_TOKEN = 52               # infer_v2_5.py:remove_long_silence default
EMO_NUM = [3, 17, 2, 8, 4, 5, 10, 24]   # config.yaml:110 (sums to 73)
# infer_v2_5.py:493 — de-emphasis applied to the 8-d user vector before use.
EMO_BIAS = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
EMO_SUM_CAP = 0.8               # infer_v2_5.py:496-500
EMO_LABELS = ["happy", "angry", "sad", "afraid",
              "disgusted", "melancholic", "surprised", "calm"]  # infer_v2_5.py:492

# tokenizer (utils/tokenizer.py:180-218)
TIKTOKEN_FILE = "multilingual_zh_ja_yue_char_del.tiktoken"
TIKTOKEN_PAT = (r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?"""
                r"""[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")   # utils/tokenizer.py:215
NUM_LANGUAGES = 99              # utils/tokenizer.py:181 default


# ---------------------------------------------------------------------------
# Weight-norm fusion (identical maths to convert-indextts-to-gguf.py)
# ---------------------------------------------------------------------------

def fuse_weight_norm_pair(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """weight = weight_v * (weight_g / ||weight_v||), norm over all dims but dim-0."""
    g = g.to(torch.float32)
    v = v.to(torch.float32)
    flat = v.reshape(v.shape[0], -1)
    norm = flat.norm(p=2, dim=1).reshape(g.shape)
    return v * (g / norm.clamp_min(1e-12))


# ---------------------------------------------------------------------------
# GGUF writing helpers
# ---------------------------------------------------------------------------

def _dtypes(outtype: str):
    if outtype == "f16":
        return np.float16, GGMLQuantizationType.F16
    return np.float32, GGMLQuantizationType.F32


def _keep_f32(name: str, t: torch.Tensor) -> bool:
    """1-D tensors, biases, norms and BN running stats stay F32 (ggml wants F32
    for these, and quantizing them costs accuracy for ~nothing)."""
    return (
        t.ndim <= 1
        or name.endswith(".bias")
        or "norm" in name
        or "gamma" in name
        or "beta" in name
        or "running_mean" in name
        or "running_var" in name
        or name.endswith(".pe")          # conformer positional table
        or name.endswith("latents")
        or name.endswith("mask_token")
    )


class TensorSink:
    """Streams tensors into a GGUFWriter, one at a time, freeing as it goes."""

    def __init__(self, writer: GGUFWriter, outtype: str, verbose: bool = False):
        self.w = writer
        self.np_dtype, self.qt = _dtypes(outtype)
        self.n = 0
        self.verbose = verbose

    def add(self, name: str, t: torch.Tensor) -> None:
        if len(name) >= 128:
            raise SystemExit(f"tensor name too long for GGML_MAX_NAME=128: {name}")
        if _keep_f32(name, t):
            arr = t.to(torch.float32).detach().numpy().astype(np.float32)
            self.w.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            arr = (t.to(torch.float32).clamp_(-65504.0, 65504.0)
                   .detach().numpy().astype(self.np_dtype))
            self.w.add_tensor(name, arr, raw_dtype=self.qt)
        if self.verbose:
            print(f"    {name:78s} {tuple(t.shape)}", file=sys.stderr)
        self.n += 1
        del arr

    def finish(self, out_path: Path) -> None:
        self.w.write_header_to_file()
        self.w.write_kv_data_to_file()
        self.w.write_tensors_to_file()
        self.w.close()
        size = out_path.stat().st_size
        print(f"  wrote {self.n} tensors → {out_path} ({size / 1024**2:.1f} MiB)",
              file=sys.stderr)


def _load_sd(path: Path, key: str | None = None):
    """mmap-backed load; returns the (possibly nested) state dict."""
    sd = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    if key is not None:
        sd = sd[key]
    return sd


# ---------------------------------------------------------------------------
# Tokenizer bake
# ---------------------------------------------------------------------------

def _special_tokens() -> list[str]:
    """Reproduce utils/tokenizer.py:191-206's ``specials`` list exactly.

    The list order *is* the token-id order (ids start at len(ranks)), so it must
    match byte-for-byte or every ``<|zh|>``/``<|SPECIAL_TOKEN_n|>`` id shifts.
    """
    # utils/tokenizer.py:11-118 — order matters (dict insertion order).
    languages = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
        "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
        "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
        "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
        "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
        "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
        "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
        "ba", "jw", "su", "yue", "minnan", "wuyu", "dialect", "zh/en", "en/zh",
        "common",
    ]
    audio_event = ["ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM",
                   "Laughter", "/Laughter", "Applause", "/Applause"]   # :140-152
    emotion = ["HAPPY", "SAD", "ANGRY", "NEUTRAL"]                     # :154-159
    tts_vocal = (["TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H"]
                 + [f"TTS/SP{i:02d}" for i in range(1, 14)])           # :160-170
    specials = ["<|endoftext|>", "<|startoftranscript|>"]
    specials += [f"<|{lang}|>" for lang in languages[:NUM_LANGUAGES]]
    specials += [f"<|{e}|>" for e in audio_event]
    specials += [f"<|{e}|>" for e in emotion]
    specials += ["<|translate|>", "<|transcribe|>", "<|startoflm|>",
                 "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>"]
    specials += [f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 31)]
    specials += [f"<|{t}|>" for t in tts_vocal]
    specials += [f"<|{i * 0.02:.2f}|>" for i in range(1501)]
    return specials, languages


def bake_tokenizer(w: GGUFWriter, tiktoken_path: Path) -> int:
    """Bake the tiktoken mergeable ranks + specials.

    ``tokenizer.ggml.tokens`` holds base64-encoded byte strings for the merge
    ranks (index == rank, exactly as tiktoken keys them) followed by the
    literal special-token strings (index == their id).  Storing base64 keeps the
    array valid UTF-8 for the GGUF string type; the runtime base64-decodes any
    entry below ``indextts2.tokenizer.n_ranks``.
    """
    ranks: list[str] = []
    with open(tiktoken_path) as f:
        for line in f:
            if not line.strip():
                continue
            token_b64, rank = line.split()
            ranks.append((int(rank), token_b64))
    ranks.sort(key=lambda kv: kv[0])
    if [r for r, _ in ranks] != list(range(len(ranks))):
        raise SystemExit("tiktoken ranks are not a dense 0..N-1 range")
    tokens = [b64 for _, b64 in ranks]
    n_ranks = len(tokens)

    specials, languages = _special_tokens()
    tokens += specials

    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_string("tokenizer.ggml.model", "tiktoken")
    w.add_string("indextts2.tokenizer.pattern", TIKTOKEN_PAT)
    w.add_uint32("indextts2.tokenizer.n_ranks", n_ranks)
    w.add_uint32("indextts2.tokenizer.n_specials", len(specials))
    # LANGUAGE_DICT (utils/tokenizer.py:121) = index of the language in LANGUAGES;
    # this is the id fed to `lang_embedding`, and is NOT the `<|xx|>` text token.
    w.add_array("indextts2.tokenizer.languages", languages)
    # Verified against v2.5.0: 58 836 merge ranks + 1 673 specials = 60 509,
    # exactly config.yaml's number_text_tokens.  text_embedding has one extra row
    # (number_text_tokens*types + 1 = 60 510, gpt/model_v2.py:388) that the
    # tokenizer can never emit — upstream's spare slot, not a conversion bug.
    print(f"  baked tokenizer: {n_ranks} ranks + {len(specials)} specials "
          f"= {len(tokens)} reachable ids "
          f"(config number_text_tokens={GPT_NUMBER_TEXT_TOKENS}, "
          f"text_embedding rows={GPT_NUMBER_TEXT_TOKENS + 1})", file=sys.stderr)
    if len(tokens) != GPT_NUMBER_TEXT_TOKENS:
        print(f"  WARNING: expected {GPT_NUMBER_TEXT_TOKENS} reachable ids, got "
              f"{len(tokens)} — the tiktoken file or the specials list changed "
              "upstream; the `<|zh|>` / `<|SPECIAL_TOKEN_n|>` ids will be wrong",
              file=sys.stderr)
    return len(tokens)


# ---------------------------------------------------------------------------
# GPT GGUF
# ---------------------------------------------------------------------------

# GPT-2 Conv1D keeps weights as [in, out]; nn.Linear (and our ggml mul_mat
# convention) wants [out, in].  Same list as convert-indextts-to-gguf.py:108.
_GPT2_CONV1D_SUFFIXES = (
    ".attn.c_attn.weight",
    ".attn.c_proj.weight",
    ".mlp.c_fc.weight",
    ".mlp.c_proj.weight",
)

# Present in gpt.pth but unused at inference.
_GPT_SKIP = {
    "text_head.weight",   # text LM head — inference is mel-autoregressive only
    "text_head.bias",
}


def _shorten_gpt(name: str) -> str:
    """Mirror convert-indextts-to-gguf.py:_shorten_gpt so that src/indextts.cpp's
    Conformer/Perceiver binders transfer with only the `emo_` prefix added.

    Note the ORDER: the emotion prefixes must be rewritten before the generic
    `encoders.`/`self_attn.`/... rules, and `emo_conditioning_encoder.` before
    `conditioning_encoder.` would ever match.
    """
    name = name.replace("emo_conditioning_encoder.", "emo_cond_enc.")
    name = name.replace("emo_perceiver_encoder.", "emo_perc.")
    name = name.replace("conditioning_encoder.", "cond_enc.")
    name = name.replace("perceiver_encoder.", "perc.")
    name = name.replace("encoders.", "enc.")
    name = name.replace("conv_module.", "conv.")
    name = name.replace("depthwise_conv.", "dw.")
    name = name.replace("pointwise_conv1.", "pw1.")
    name = name.replace("pointwise_conv2.", "pw2.")
    name = name.replace("self_attn.", "sa.")
    name = name.replace("feed_forward.", "ff.")
    name = name.replace("text_pos_embedding.emb.", "text_pos.")
    name = name.replace("mel_pos_embedding.emb.", "mel_pos.")
    return name


def convert_gpt(model_dir: Path, out_path: Path, outtype: str, verbose: bool) -> None:
    gpt_pth = model_dir / "gpt.pth"
    print(f"\n=== GPT: {gpt_pth.name} → {out_path.name} ===", file=sys.stderr)
    sd = _load_sd(gpt_pth)
    if isinstance(sd, dict) and "model" in sd and not any(k.startswith("gpt.") for k in sd):
        sd = sd["model"]

    w = GGUFWriter(str(out_path), arch="indextts2.gpt", use_temp_file=True)
    w.add_name("indextts2-gpt")

    def u32(k, v):
        w.add_uint32(k, int(v))

    u32("indextts2.gpt.model_dim", GPT_MODEL_DIM)
    u32("indextts2.gpt.layers", GPT_LAYERS)
    u32("indextts2.gpt.heads", GPT_HEADS)
    u32("indextts2.gpt.head_dim", GPT_MODEL_DIM // GPT_HEADS)
    u32("indextts2.gpt.ff_dim", 4 * GPT_MODEL_DIM)
    u32("indextts2.gpt.number_text_tokens", GPT_NUMBER_TEXT_TOKENS)
    u32("indextts2.gpt.number_mel_codes", GPT_NUMBER_MEL_CODES)
    u32("indextts2.gpt.start_mel_token", GPT_START_MEL_TOKEN)
    u32("indextts2.gpt.stop_mel_token", GPT_STOP_MEL_TOKEN)
    u32("indextts2.gpt.start_text_token", GPT_START_TEXT_TOKEN)
    u32("indextts2.gpt.stop_text_token", GPT_STOP_TEXT_TOKEN)
    u32("indextts2.gpt.max_mel_tokens", GPT_MAX_MEL_TOKENS)
    u32("indextts2.gpt.max_text_tokens", GPT_MAX_TEXT_TOKENS)
    u32("indextts2.gpt.mel_pos_size", GPT_MEL_POS_SIZE)
    u32("indextts2.gpt.text_pos_size", GPT_TEXT_POS_SIZE)
    u32("indextts2.gpt.mel_length_compression", GPT_MEL_LENGTH_COMPRESSION)
    u32("indextts2.gpt.silent_token", SILENT_TOKEN)
    w.add_string("indextts2.gpt.spk_cond_mode", "campplus")   # infer_v2_5.py:138
    u32("indextts2.gpt.spk_embed_dim", S2MEL_STYLE_DIM)       # CAMPPlus 192 → 1280

    # Emotion Conformer + Perceiver
    u32("indextts2.emo_conformer.num_blocks", EMO_COND_NUM_BLOCKS)
    u32("indextts2.emo_conformer.output_size", EMO_COND_OUTPUT_SIZE)
    u32("indextts2.emo_conformer.linear_units", EMO_COND_LINEAR_UNITS)
    u32("indextts2.emo_conformer.attention_heads", EMO_COND_ATTENTION_HEADS)
    u32("indextts2.emo_conformer.input_size", EMO_INPUT_SIZE)
    w.add_string("indextts2.emo_conformer.input_layer", EMO_COND_INPUT_LAYER)
    u32("indextts2.emo_perceiver.num_layers", EMO_PERCEIVER_DEPTH)
    u32("indextts2.emo_perceiver.num_latents", EMO_PERCEIVER_NUM_LATENTS)
    u32("indextts2.emo_perceiver.dim", EMO_PERCEIVER_DIM)
    u32("indextts2.emo_perceiver.dim_context", EMO_COND_OUTPUT_SIZE)
    u32("indextts2.emo_perceiver.dim_head", EMO_PERCEIVER_DIM_HEAD)
    u32("indextts2.emo_perceiver.heads", EMO_COND_ATTENTION_HEADS)
    u32("indextts2.emo_perceiver.ff_mult", EMO_PERCEIVER_MULT)

    # Emotion prototype bookkeeping
    w.add_array("indextts2.emo.num", [int(x) for x in EMO_NUM])
    w.add_array("indextts2.emo.bias", [float(x) for x in EMO_BIAS])
    w.add_float32("indextts2.emo.sum_cap", EMO_SUM_CAP)
    w.add_array("indextts2.emo.labels", EMO_LABELS)

    u32("indextts2.sample_rate", S2MEL_SR)

    bake_tokenizer(w, model_dir / TIKTOKEN_FILE)

    # w2v-BERT layer-17 normalisation stats (infer_v2_5.py:177-179).
    # std = sqrt(var); we bake `var` verbatim (same convention as Confucius4's
    # confucius4.w2v_bert.{mean,var}) so the two GGUFs stay diffable.
    stats = _load_sd(model_dir / "wav2vec2bert_stats.pt")
    w.add_array("indextts2.w2v_bert.mean", stats["mean"].float().numpy().tolist())
    w.add_array("indextts2.w2v_bert.var", stats["var"].float().numpy().tolist())
    u32("indextts2.w2v_bert.layer", 17)          # infer_v2_5.py:287
    u32("indextts2.w2v_bert.dim", EMO_INPUT_SIZE)
    del stats

    sink = TensorSink(w, outtype, verbose)

    for name, tensor in sd.items():
        if name in _GPT_SKIP:
            continue
        short = _shorten_gpt(name)
        if any(name.endswith(s) for s in _GPT2_CONV1D_SUFFIXES):
            tensor = tensor.t().contiguous()
        sink.add(short, tensor)

    # Emotion prototype tables, baked as tensors (feat1.pt / feat2.pt).
    #   feat1.pt = spk_matrix (73, 192)   CAMPPlus style prototypes  (config.yaml:109)
    #   feat2.pt = emo_matrix (73, 1280)  emovec prototypes          (config.yaml:108)
    # Both are split by emo_num=[3,17,2,8,4,5,10,24] at load (infer_v2_5.py:245-253).
    for fname, tname in (("feat1.pt", "spk_matrix"), ("feat2.pt", "emo_matrix")):
        t = _load_sd(model_dir / fname)
        if not isinstance(t, torch.Tensor):
            raise SystemExit(f"{fname}: expected a bare tensor, got {type(t)}")
        if t.shape[0] != sum(EMO_NUM):
            raise SystemExit(f"{fname}: rows {t.shape[0]} != sum(emo_num)={sum(EMO_NUM)}")
        sink.add(tname, t.detach())
        del t

    del sd
    gc.collect()
    sink.finish(out_path)


# ---------------------------------------------------------------------------
# Codec GGUF (EnhancedCodec: down/up + 2× VocosBackbone + factorised RVQ)
# ---------------------------------------------------------------------------

def convert_codec(model_dir: Path, out_path: Path, outtype: str, verbose: bool) -> None:
    codec_pth = model_dir / "codec.pth"
    print(f"\n=== codec: {codec_pth.name} → {out_path.name} ===", file=sys.stderr)
    sd = _load_sd(codec_pth, key="model")

    w = GGUFWriter(str(out_path), arch="indextts2.codec", use_temp_file=True)
    w.add_name("indextts2-codec")

    def u32(k, v):
        w.add_uint32(k, int(v))

    u32("indextts2.codec.codebook_size", CODEC_CODEBOOK_SIZE)
    u32("indextts2.codec.codebook_dim", CODEC_CODEBOOK_DIM)
    u32("indextts2.codec.hidden_size", CODEC_HIDDEN_SIZE)
    u32("indextts2.codec.vocos_dim", CODEC_VOCOS_DIM)
    u32("indextts2.codec.vocos_intermediate_dim", CODEC_VOCOS_INTERMEDIATE)
    u32("indextts2.codec.vocos_num_layers", CODEC_VOCOS_NUM_LAYERS)
    u32("indextts2.codec.num_quantizers", CODEC_NUM_QUANTIZERS)
    u32("indextts2.codec.downsample_scale", CODEC_DOWNSAMPLE_SCALE)
    w.add_bool("indextts2.codec.use_l2_normlize", True)   # codec/models.py:131

    sink = TensorSink(w, outtype, verbose)

    # Fuse the RVQ's weight-normed 1×1 projections (in_project/out_project).
    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in sd.items():
        if name.endswith(".weight_g") or name.endswith(".weight_v"):
            stem, suf = name.rsplit(".weight_", 1)
            pending.setdefault(stem, {})[suf] = tensor
            continue
        sink.add(name, tensor)
    for stem, parts in sorted(pending.items()):
        if "g" not in parts or "v" not in parts:
            raise SystemExit(f"unpaired weight_norm tensor: {stem}")
        sink.add(f"{stem}.weight", fuse_weight_norm_pair(parts["g"], parts["v"]))
    print(f"  weight_norm fused: {len(pending)} pairs", file=sys.stderr)

    del sd, pending
    gc.collect()
    sink.finish(out_path)


# ---------------------------------------------------------------------------
# s2mel GGUF — renamed into the Confucius4 S2A key family
# ---------------------------------------------------------------------------

# Present in s2mel.pth but never used by MyModel's inference path.
_S2MEL_SKIP_EXACT = {
    "estimator.input_pos",                  # buffer, arange(16384)
    "estimator.t_embedder.freqs",           # buffer, recomputed in C++
    "estimator.t_embedder2.freqs",
    "estimator.cond_embedder.weight",       # discrete content path — unused
    "estimator.content_mask_embedder.weight",  # class-dropout only (training)
}
_S2MEL_SKIP_PREFIX = (
    "estimator.x_embedder.",                # built but never called in DiT.forward
)
# InterpolateRegulator with is_discrete=False never touches these.
_LR_SKIP = {"embedding.weight", "mask_token"}


def _map_cfm_name(k: str) -> str | None:
    """s2mel.pth['net']['cfm'] key → Confucius4 S2A GGUF key.

    Confucius4's S2A checkpoint spells the same modules differently; matching its
    names lets src/confucius4_tts.cpp bind this block unchanged.

      estimator.transformer.layers.N.        → decoder.estimator.transformer_blocks.N.
      estimator.transformer.norm.            → decoder.estimator.transformer_norm.
      <adaLN>.project_layer.                 → <adaLN>.modulation.
      estimator.cond_x_merge_linear.         → decoder.estimator.input_embed.proj.
      estimator.cond_projection.             → decoder.estimator.input_embed.mu_projection.
      estimator.t_embedder{,2}.mlp.N.        → decoder.estimator.t_embedder{,2}.time_mlp.N.
      estimator.wavenet.*.conv.conv.         → decoder.estimator.wavenet.*.conv.
    Returns None for tensors that must be skipped.
    """
    if k in _S2MEL_SKIP_EXACT or k.startswith(_S2MEL_SKIP_PREFIX):
        return None

    n = k
    n = n.replace("estimator.transformer.layers.", "estimator.transformer_blocks.")
    n = n.replace("estimator.transformer.norm.", "estimator.transformer_norm.")
    n = n.replace(".project_layer.", ".modulation.")
    n = n.replace("estimator.cond_x_merge_linear.", "estimator.input_embed.proj.")
    n = n.replace("estimator.cond_projection.", "estimator.input_embed.mu_projection.")
    n = n.replace("estimator.t_embedder.mlp.", "estimator.t_embedder.time_mlp.")
    n = n.replace("estimator.t_embedder2.mlp.", "estimator.t_embedder2.time_mlp.")
    # SConv1d → NormConv1d → Conv1d nests one extra `.conv`; Confucius4 has one.
    n = n.replace(".conv.conv.", ".conv.")
    return "decoder." + n


def convert_s2mel(model_dir: Path, out_path: Path, outtype: str, verbose: bool,
                  campplus_path: Path | None) -> None:
    s2mel_pth = model_dir / "s2mel.pth"
    print(f"\n=== s2mel: {s2mel_pth.name} → {out_path.name} ===", file=sys.stderr)
    net = _load_sd(s2mel_pth, key="net")
    if "cfm" not in net or "length_regulator" not in net:
        raise SystemExit(f"s2mel.pth['net'] groups: {list(net)} — expected cfm + length_regulator")

    w = GGUFWriter(str(out_path), arch="indextts2.s2mel", use_temp_file=True)
    w.add_name("indextts2-s2mel")

    def u32(k, v):
        w.add_uint32(k, int(v))

    # Mirrors confucius4.s2a.* (confucius4-tts-convert.py:348-356) one-for-one.
    u32("indextts2.s2mel.input_size", DIT_CONTENT_DIM)
    u32("indextts2.s2mel.output_size", DIT_IN_CHANNELS)
    u32("indextts2.s2mel.spk_embed_dim", S2MEL_STYLE_DIM)
    u32("indextts2.s2mel.semantic_embed_dim", LR_IN_CHANNELS)
    u32("indextts2.s2mel.estimator_depth", DIT_DEPTH)
    u32("indextts2.s2mel.estimator_num_heads", DIT_NUM_HEADS)
    u32("indextts2.s2mel.estimator_hidden_dim", DIT_HIDDEN_DIM)
    u32("indextts2.s2mel.wavenet_num_layers", WAVENET_NUM_LAYERS)
    u32("indextts2.s2mel.wavenet_kernel_size", WAVENET_KERNEL_SIZE)
    u32("indextts2.s2mel.wavenet_dilation_rate", WAVENET_DILATION_RATE)
    u32("indextts2.s2mel.lr_channels", LR_CHANNELS)
    u32("indextts2.s2mel.lr_sampling_ratios", LR_SAMPLING_RATIOS)
    u32("indextts2.s2mel.lr_n_quantizers", LR_N_QUANTIZERS)
    u32("indextts2.s2mel.diffusion_steps", DIFFUSION_STEPS)
    w.add_float32("indextts2.s2mel.cfg_rate", INFERENCE_CFG_RATE)
    w.add_float32("indextts2.s2mel.length_ratio", LENGTH_RATIO)
    w.add_bool("indextts2.s2mel.uvit_skip_connection", True)     # config.yaml:96
    w.add_bool("indextts2.s2mel.long_skip_connection", True)     # config.yaml:92
    w.add_bool("indextts2.s2mel.style_condition", True)          # config.yaml:82
    w.add_string("indextts2.s2mel.final_layer_type", "wavenet")  # config.yaml:83
    # Mel front-end (config.yaml:51-59; s2mel/modules/audio.py, center=False)
    u32("indextts2.mel.n_fft", S2MEL_N_FFT)
    u32("indextts2.mel.win_length", S2MEL_WIN_LENGTH)
    u32("indextts2.mel.hop_length", S2MEL_HOP_LENGTH)
    u32("indextts2.mel.n_mels", S2MEL_N_MELS)
    u32("indextts2.mel.fmin", S2MEL_FMIN)
    w.add_bool("indextts2.mel.center", False)                    # infer_v2_5.py:263
    u32("indextts2.sample_rate", S2MEL_SR)

    sink = TensorSink(w, outtype, verbose)

    # --- CFM / DiT -----------------------------------------------------------
    cfm = net["cfm"]
    pending: dict[str, dict[str, torch.Tensor]] = {}
    n_skipped = 0
    for k, t in cfm.items():
        mapped = _map_cfm_name(k)
        if mapped is None:
            n_skipped += 1
            continue
        # WaveNet weight_norm pairs are kept UNFUSED: confucius4_tts.cpp:650-730
        # folds them at load time and looks up `<base>.weight_g` / `.weight_v`.
        if ".wavenet." in mapped and (mapped.endswith(".weight_g") or mapped.endswith(".weight_v")):
            sink.add(mapped, t)
            continue
        if mapped.endswith(".weight_g") or mapped.endswith(".weight_v"):
            stem, suf = mapped.rsplit(".weight_", 1)
            pending.setdefault(stem, {})[suf] = t
            continue
        sink.add(mapped, t)
    for stem, parts in sorted(pending.items()):
        if "g" not in parts or "v" not in parts:
            raise SystemExit(f"unpaired weight_norm tensor: {stem}")
        sink.add(f"{stem}.weight", fuse_weight_norm_pair(parts["g"], parts["v"]))
    print(f"  cfm: skipped {n_skipped} unused, fused {len(pending)} non-WaveNet "
          f"weight_norm pairs (WaveNet pairs left for load-time folding)",
          file=sys.stderr)

    # --- length regulator ----------------------------------------------------
    lr = net["length_regulator"]
    n_lr = 0
    for k, t in lr.items():
        if k in _LR_SKIP:
            continue
        sink.add("length_regulator." + k, t)
        n_lr += 1
    print(f"  length_regulator: {n_lr} tensors", file=sys.stderr)

    if "gpt_layer" in net:
        print("  note: s2mel.pth carries a `gpt_layer` group; MyModel is built with "
              "use_gpt_latent=False (commons.py:409-420) so it is dead weight — skipped",
              file=sys.stderr)

    # --- CAMPPlus bake (identical to confucius4-tts-convert.py:371-393) ------
    if campplus_path is None:
        try:
            from huggingface_hub import hf_hub_download
            campplus_path = Path(hf_hub_download("funasr/campplus",
                                                 filename="campplus_cn_common.bin"))
        except Exception as e:      # noqa: BLE001
            print(f"  WARNING: CAMPPlus not baked ({e}); pass --campplus to add it",
                  file=sys.stderr)
            campplus_path = None
    if campplus_path is not None:
        spk = torch.load(str(campplus_path), map_location="cpu", weights_only=True)
        if isinstance(spk, dict) and "state_dict" in spk:
            spk = spk["state_dict"]
        n_spk = 0
        for k in sorted(spk):
            if k.endswith("num_batches_tracked"):
                continue
            sink.add("campplus." + k, spk[k])
            n_spk += 1
        print(f"  baked {n_spk} CAMPPlus tensors under campplus.*", file=sys.stderr)
        u32("indextts2.campplus.feat_dim", 80)
        u32("indextts2.campplus.embedding_size", S2MEL_STYLE_DIM)
        del spk

    del net
    gc.collect()
    sink.finish(out_path)


# ---------------------------------------------------------------------------
# Companion check
# ---------------------------------------------------------------------------

_COMPANIONS = [
    ("cstr/confucius4-tts-GGUF", "confucius4-tts-bigvgan-22k-f16.gguf",
     "BigVGAN v2 22 kHz / 80-band / 256x (nvidia/bigvgan_v2_22khz_80band_256x)"),
    ("cstr/confucius4-tts-GGUF", "confucius4-tts-w2v-f16.gguf",
     "w2v-BERT 2.0 encoder, layer-17 features (facebook/w2v-bert-2.0)"),
]


def check_companions() -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    print("\n=== companion GGUFs (reused verbatim from Confucius4) ===", file=sys.stderr)
    for repo, fname, what in _COMPANIONS:
        try:
            info = api.model_info(repo, files_metadata=True)
            sib = next((s for s in info.siblings if s.rfilename == fname), None)
            if sib is None:
                print(f"  MISSING  {repo}/{fname}", file=sys.stderr)
            else:
                print(f"  OK       {repo}/{fname}  "
                      f"({(sib.size or 0) / 1024**2:.1f} MiB)  — {what}", file=sys.stderr)
        except Exception as e:      # noqa: BLE001
            print(f"  ERROR    {repo}/{fname}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert IndexTTS-2.5 checkpoints to GGUF (arch: indextts2)")
    ap.add_argument("--model-dir", required=True,
                    help="dir with gpt.pth, codec.pth, s2mel.pth, feat1.pt, feat2.pt, "
                         "wav2vec2bert_stats.pt, " + TIKTOKEN_FILE)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--outtype", default="f16", choices=["f16", "f32"])
    ap.add_argument("--only", default="gpt,codec,s2mel",
                    help="comma-separated subset of gpt,codec,s2mel")
    ap.add_argument("--campplus", default=None,
                    help="path to campplus_cn_common.bin (else fetched from funasr/campplus)")
    ap.add_argument("--check-companions", action="store_true",
                    help="verify the reused BigVGAN / w2v-BERT GGUFs are reachable")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    suffix = args.outtype

    if args.check_companions:
        check_companions()

    required = {
        "gpt": ["gpt.pth", "feat1.pt", "feat2.pt", "wav2vec2bert_stats.pt", TIKTOKEN_FILE],
        "codec": ["codec.pth"],
        "s2mel": ["s2mel.pth"],
    }
    for stage in sorted(only):
        for f in required.get(stage, []):
            if not (model_dir / f).is_file():
                sys.exit(f"not found: {model_dir / f}")

    if "gpt" in only:
        convert_gpt(model_dir, out_dir / f"indextts2-gpt-{suffix}.gguf",
                    args.outtype, args.verbose)
    if "codec" in only:
        convert_codec(model_dir, out_dir / f"indextts2-codec-{suffix}.gguf",
                      args.outtype, args.verbose)
    if "s2mel" in only:
        convert_s2mel(model_dir, out_dir / f"indextts2-s2mel-{suffix}.gguf",
                      args.outtype, args.verbose,
                      Path(args.campplus) if args.campplus else None)

    print("\nDone. Reminder: bilibili Model Use License — these GGUFs are Derivative "
          "Works; do NOT publish them to a model hub.", file=sys.stderr)


if __name__ == "__main__":
    main()
