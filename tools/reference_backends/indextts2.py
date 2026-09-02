"""IndexTTS-2.5 (``IndexTeam/IndexTTS-2.5``, upstream tag ``v2.5.0``) reference dumper.

Stage-by-stage torch oracle for the CrispASR ``indextts2`` backend.  Every stage
is computed by calling the **upstream modules directly** — the pipeline below is
a line-for-line transcription of ``indextts/infer_v2_5.py::infer_generator``
(v2.5.0), with the source line numbers cited inline.  Nothing is re-derived.

Why not just call ``IndexTTS2.infer()``?  Because it is a single opaque call that
returns only a WAV, and because its constructor eagerly loads all six sub-models
(≈8.3 GB fp32) plus the QwenEmotion LLM.  The reference VPS has 8 GB of RAM, so
the pipeline is split into stages that each load only what they need.

Stages
------
    text_tokens          int32  (L,)      lang_prefix + text through the tiktoken BPE
    lang_id              int32  (1,)      LANGUAGE_DICT[lang] → `lang_embedding` index
    fbank                f32 (T_f, 80)    kaldi fbank, mean-subtracted (CAMPPlus input)
    style                f32 (192,)       CAMPPlus speaker embedding
    ref_mel              f32 (80, T_m)    22.05 kHz mel of the speaker prompt, center=False
    spk_cond_emb         f32 (T_w, 1024)  w2v-BERT layer-17, z-normalised (speaker audio)
    emo_cond_emb         f32 (T_e, 1024)  same, for the emotion reference audio
    prompt_condition     f32 (T_m, 512)   length_regulator(spk_cond_emb, ylens=T_m)
    emovec               f32 (1280,)      merge_emovec(spk, emo, alpha)
    emovec_mat           f32 (1280,)      8-d-vector prototype mix (only with --emo-vector)
    spk_latent           f32 (1, 1280)    spk_emb_proj(style)
    gpt_prefix_embeds    f32 (S, 1280)    [pad][cond+emovec][text] prefix fed to GPT-2
    semantic_codes       int32 (N,)       GPT-2 AR output, greedy
    s_infer              f32 (T_s, 1024)  semantic_codec.decode(codes)
    cond                 f32 (T_t, 512)   length_regulator(s_infer, target_lengths)
    cat_condition        f32 (T_m+T_t,512) [prompt_condition | cond]
    cfm_noise            f32 (80, T_all)  the FIXED noise tensor handed to the ODE
    s2mel_mel            f32 (80, T_t)    CFM output with the prompt frames stripped
    audio                f32 (n,)         BigVGAN PCM at 22 050 Hz, in [-1, 1]

Determinism
-----------
* GPT-2 runs with ``do_sample=False, num_beams=1`` (upstream defaults are
  ``num_beams=3, repetition_penalty=10.0, top_k=30, top_p=0.8`` —
  ``infer_v2_5.py:737-739`` — which are not reproducible).
* The CFM noise ``z`` (``s2mel/modules/flow_matching.py:52``) is drawn from a
  seeded ``torch.Generator`` and dumped, so C++ can be fed the identical tensor.
* ``emo_vector`` paths run with ``use_random=False`` (``infer_v2_5.py:673``), i.e.
  prototype rows are chosen by cosine-nearest CAMPPlus style, not randomly.

Audio conditioning — must match the C++ byte for byte
-----------------------------------------------------
``_load_and_cut_audio`` (``infer_v2_5.py:396-408``) uses ``librosa.load`` with its
**default sr=22050** for the speaker prompt (so "22 kHz" is the native rate, not a
resample of 24 kHz) and ``sr=16000`` for the emotion prompt, and truncates to the
first 15 s.  The speaker branch then resamples 22050→22050 (identity) for the mel
and 22050→16000 for both w2v-BERT and the kaldi fbank
(``infer_v2_5.py:626-634, 642-648``).  The mel uses ``center=False``
(``infer_v2_5.py:263``).  Any deviation here poisons every downstream stage.

Environment
-----------
    python -m venv --system-site-packages ~/venvs/indextts2-ref
    export INDEXTTS2_SRC=/mnt/volume1/tmp-overflow/indextts25-src

Only what the upstream *inference* path needs — no training deps, no
``nemo_text_processing``/``wetext`` (text normalisation is off by default here, so
they are never imported), no QwenEmotion (``use_qwen_emo=False`` upstream default,
``infer_v2_5.py:126-131``), no deepspeed/flash-attn (``use_accel=False``).

**transformers must be pinned to 4.52.1** (upstream ``pyproject.toml:61``) for the
``emovec`` and ``gpt`` stages: ``indextts/gpt/transformers_generation_utils.py:28``
imports ``OffloadedCache``, which transformers 5.x removed, so importing
``indextts.gpt.model_v2`` raises ``ImportError`` on anything newer.  The
``text`` / ``campplus`` / ``mel`` / ``w2v`` / ``lr_prompt`` / ``codec`` / ``lr_cond``
/ ``s2mel`` / ``bigvgan`` stages do **not** touch that module and run fine against a
modern transformers — which is why the two GPT stages are grouped with the heavy
Kaggle set.

    ~/venvs/indextts2-ref/bin/pip install 'transformers==4.52.1' 'tokenizers<0.22'

CLI
---
    python tools/reference_backends/indextts2.py \\
        --model-dir /mnt/storage/gguf-models/indextts25-src \\
        --src       /mnt/volume1/tmp-overflow/indextts25-src \\
        --audio     samples/jfk.wav \\
        --text      "Hello, this is a reference dump." --lang en \\
        --out-dir   /mnt/volume1/tmp-overflow/it25-ref \\
        --stages    text,campplus,mel

Stages are checkpointed as ``<out-dir>/<stage>.npy``; a later invocation with more
stages reuses whatever is already on disk, so the light stages can run on the VPS
and the heavy ones (``gpt``, ``s2mel``, ``bigvgan``) on Kaggle
(``tools/kaggle/indextts2-refdump/``) against the same ``out-dir``.

LICENSE: IndexTTS-2.5 ships under bilibili's Model Use License; this file only
*calls* the upstream code, it does not vendor it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import numpy as np

# --------------------------------------------------------------------------
# Stage plumbing
# --------------------------------------------------------------------------

# Ordered; each entry names the stage *group* that produces it.
STAGE_GROUPS: dict[str, tuple[str, ...]] = {
    "text":     ("text_tokens", "lang_id"),
    "campplus": ("fbank", "style"),
    "mel":      ("ref_mel",),
    "w2v":      ("spk_cond_emb", "emo_cond_emb"),
    "lr_prompt": ("prompt_condition",),
    "emovec":   ("emovec", "emovec_mat", "spk_latent"),
    "gpt":      ("gpt_prefix_embeds", "semantic_codes"),
    "codec":    ("s_infer",),
    "lr_cond":  ("cond", "cat_condition"),
    "s2mel":    ("cfm_noise", "s2mel_mel"),
    "bigvgan":  ("audio",),
}

ALL_STAGES = list(STAGE_GROUPS)

# Stages that fit in 8 GB on CPU with a modern transformers.  `w2v` needs ~2.4 GB
# for the fp32 encoder; `codec` / `lr_*` / `s2mel` need ~1.1 GB together.
# `emovec` and `gpt` both construct `UnifiedVoice` (3.26 GB fp32) *and* need the
# transformers==4.52.1 pin, so both belong to the Kaggle set even though only
# `gpt` does the 1 500-step AR decode.
LOCAL_SAFE_STAGES = ["text", "campplus", "mel", "w2v", "lr_prompt"]
HEAVY_STAGES = ["emovec", "gpt", "codec", "lr_cond", "s2mel", "bigvgan"]

DEFAULT_STAGES = [s for g in ALL_STAGES for s in STAGE_GROUPS[g]]

DEFAULT_TEXT = "Hello, this is a reference dump."
DEFAULT_LANG = "en"
DEFAULT_SEED = 1234

# infer_v2_5.py:492 — the fixed order of the 8-d emotion vector.
EMO_LABELS = ["happy", "angry", "sad", "afraid",
              "disgusted", "melancholic", "surprised", "calm"]


# --------------------------------------------------------------------------
# Two tiny helpers transcribed verbatim from infer_v2_5.py.
#
# They live at module scope in `infer_v2_5`, whose import pulls in
# `indextts.gpt.model_v2` → the vendored `transformers_generation_utils`, which
# hard-requires **transformers==4.52.1** (pyproject.toml:61) — it imports
# `OffloadedCache`, removed in transformers 5.x.  The text/CAMPPlus/mel stages
# need neither torch-GPT nor generation utils, so copying these two keeps the
# light stages runnable against a modern transformers.
# --------------------------------------------------------------------------

_PRONUNCIATION_ANNOTATION_PATTERN = None   # compiled lazily (needs `re`)


def _is_kana(s: str) -> bool:            # infer_v2_5.py:41-49
    import re
    if re.compile(r'^[぀-ゟ]+$').fullmatch(s):
        return True
    if re.compile(r'^[゠-ヿ]+$').fullmatch(s):
        return True
    return False


def apply_pronunciation_annotations(text: str) -> str:   # infer_v2_5.py:52-72
    """<文字|发音> → <|SPECIAL_TOKEN_1|>/<|SPECIAL_TOKEN_2|>-wrapped, upper-cased."""
    import re
    global _PRONUNCIATION_ANNOTATION_PATTERN
    if _PRONUNCIATION_ANNOTATION_PATTERN is None:
        _PRONUNCIATION_ANNOTATION_PATTERN = re.compile(r'<([^|>\n]+)\|([^>\n]+)>')

    def _replace(match):
        word = match.group(1)
        pronunciation = match.group(2).upper()
        has_chinese = bool(re.search(r"[一-鿿]", word))
        if _is_kana(pronunciation):
            return f' {pronunciation} '
        token = 'SPECIAL_TOKEN_2' if has_chinese else 'SPECIAL_TOKEN_1'
        return f'<|{token}|>{pronunciation}<|{token}|>'

    return _PRONUNCIATION_ANNOTATION_PATTERN.sub(_replace, text)


def find_most_similar_cosine(query_vector, matrix):      # infer_v2_5.py:901-907
    import torch.nn.functional as F
    return int(F.cosine_similarity(query_vector.float(), matrix.float(), dim=1).argmax())


def _import_upstream(src: Optional[str]):
    """Put the upstream v2.5.0 checkout on sys.path and return the module set.

    ``INDEXTTS2_SRC`` (or --src) must point at a clone of
    github.com/index-tts/index-tts at tag ``v2.5.0``.
    """
    src = src or os.environ.get("INDEXTTS2_SRC")
    if not src:
        raise SystemExit(
            "set --src or INDEXTTS2_SRC to the index-tts v2.5.0 checkout "
            "(git clone -b v2.5.0 https://github.com/index-tts/index-tts)")
    src = str(Path(src).resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    if not (Path(src) / "indextts" / "infer_v2_5.py").is_file():
        raise SystemExit(f"{src} does not look like an index-tts v2.5.0 checkout")
    return src


def _cfg(model_dir: Path):
    from omegaconf import OmegaConf
    return OmegaConf.load(str(model_dir / "config.yaml"))


def _np(t) -> np.ndarray:
    import torch
    if isinstance(t, torch.Tensor):
        return t.detach().to(torch.float32).cpu().numpy()
    return np.asarray(t)


class _Store:
    """Stage cache backed by ``<out_dir>/<stage>.npy``."""

    def __init__(self, out_dir: Optional[Path]):
        self.out_dir = out_dir
        self.mem: Dict[str, np.ndarray] = {}
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

    def has(self, name: str) -> bool:
        if name in self.mem:
            return True
        return self.out_dir is not None and (self.out_dir / f"{name}.npy").is_file()

    def get(self, name: str) -> np.ndarray:
        if name in self.mem:
            return self.mem[name]
        if self.out_dir is not None:
            p = self.out_dir / f"{name}.npy"
            if p.is_file():
                a = np.load(p)
                self.mem[name] = a
                return a
        raise KeyError(f"stage output '{name}' not available — run its stage first")

    def put(self, name: str, arr) -> np.ndarray:
        a = _np(arr)
        self.mem[name] = a
        if self.out_dir is not None:
            np.save(self.out_dir / f"{name}.npy", a)
        print(f"  [{name}] {a.shape} {a.dtype}", file=sys.stderr)
        return a


# --------------------------------------------------------------------------
# Audio loading — must mirror infer_v2_5.py exactly
# --------------------------------------------------------------------------

def _load_and_cut_audio(audio_path: str, max_seconds: int, sr: Optional[int] = None):
    """Verbatim port of infer_v2_5.py:396-408 (librosa default sr=22050 when sr=None)."""
    import librosa
    import torch
    if not sr:
        audio, sr = librosa.load(audio_path)
    else:
        audio, _ = librosa.load(audio_path, sr=sr)
    audio = torch.tensor(audio).unsqueeze(0)
    max_samples = int(max_seconds * sr)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]
    return audio, sr


MAX_REF_SECONDS = 15   # infer_v2_5.py:626, 642, 685


# --------------------------------------------------------------------------
# Individual stages
# --------------------------------------------------------------------------

def _stage_text(store, model_dir: Path, text: str, lang: str, normalize: bool):
    """infer_v2_5.py:698-725 — lang prefix, char remap, casing, tokenise, pad stop."""
    import re
    import torch
    import torch.nn.functional as F
    from indextts.utils.tokenizer import get_tokenizer, lang_to_token
    from indextts.utils.front import TextNormalizer

    tp = TextNormalizer(enable_glossary=True)
    # infer_v2_5.py:701 — the char_rep_map substitution always runs, even when
    # text_normalization is off.  TextNormalizer.load() pulls in wetext / tn, so
    # only call it when normalisation was actually requested.
    text = tp.clean_pattern.sub(lambda x: tp.char_rep_map[x.group()], text)
    if normalize:
        tp.load()
        low = lang.lower()
        if low in ("zh", "zhen", "en"):
            text = tp.normalize(text)
        elif low in ("ja", "es"):
            from indextts.utils.nemo_tn import normalize_text as nemo_text_normalize
            text = nemo_text_normalize(text, low)

    low = lang.lower()
    if low in ("ja", "zh", "zhen", "en"):
        text = text.lower()                         # infer_v2_5.py:711
    if low == "es":
        text = text.upper()                         # infer_v2_5.py:713
    text = apply_pronunciation_annotations(text)    # infer_v2_5.py:714
    if low == "ja":
        from indextts.utils.ja_g2p import JapaneseG2PProcessor
        text = JapaneseG2PProcessor(g2p_ratio=0).process_ja_text(text)
    text = re.sub(r'<\|([^|]+)\|>', lambda m: f'<|{m.group(1).upper()}|>', text)

    lang_prefix = f'<|{lang.lower()}|> '            # infer_v2_5.py:699
    tok = get_tokenizer(multilingual=True, model_dir=str(model_dir))
    ids = tok.encode(lang_prefix + text, allowed_special='all')   # infer_v2_5.py:723
    t = torch.IntTensor(ids).unsqueeze(0)
    t = F.pad(t, (0, 1), value=1)                   # infer_v2_5.py:725 stop_text_token
    store.put("text_tokens", t[0].to(torch.int32).numpy().astype(np.int32))
    store.put("lang_id", np.asarray([lang_to_token(lang)], dtype=np.int32))
    print(f"  text after front-end: {text!r}", file=sys.stderr)


def _stage_campplus(store, model_dir: Path, audio_path: str):
    """infer_v2_5.py:642-648 — kaldi fbank(80, dither=0) − per-utt mean → CAMPPlus."""
    import torch
    import torchaudio
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

    audio, sr = _load_and_cut_audio(audio_path, MAX_REF_SECONDS)
    audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
    feat = torchaudio.compliance.kaldi.fbank(
        audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    store.put("fbank", feat)

    ckpt = _campplus_ckpt(model_dir)
    m = CAMPPlus(feat_dim=80, embedding_size=192)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    with torch.no_grad():
        style = m(feat.unsqueeze(0))                # (1, 192)
    store.put("style", style[0])
    del m


def _campplus_ckpt(model_dir: Path) -> str:
    p = model_dir / "hf_cache" / "campplus_cn_common.bin"
    if p.is_file():
        return str(p)
    from huggingface_hub import hf_hub_download
    return hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")


def _mel_fn(cfg):
    """infer_v2_5.py:255-266 — note center=False (:263) and the fmax "None"→8000 remap."""
    from indextts.s2mel.modules.audio import mel_spectrogram
    sp = cfg.s2mel["preprocess_params"]["spect_params"]
    args = {
        "n_fft": sp["n_fft"],
        "win_size": sp["win_length"],
        "hop_size": sp["hop_length"],
        "num_mels": sp["n_mels"],
        "sampling_rate": cfg.s2mel["preprocess_params"]["sr"],
        "fmin": sp.get("fmin", 0),
        "fmax": None if sp.get("fmax", "None") == "None" else 8000,
        "center": False,
    }
    return lambda x: mel_spectrogram(x, **args)


def _stage_mel(store, model_dir: Path, audio_path: str):
    import torchaudio
    cfg = _cfg(model_dir)
    audio, sr = _load_and_cut_audio(audio_path, MAX_REF_SECONDS)
    audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)   # infer_v2_5.py:627
    ref_mel = _mel_fn(cfg)(audio_22k.float())                      # (1, 80, T)
    store.put("ref_mel", ref_mel[0])


def _w2v_dir(model_dir: Path) -> str:
    d = model_dir / "hf_cache" / "w2v-bert-2.0"
    if d.is_dir():
        return str(d)
    return "facebook/w2v-bert-2.0"


def _stage_w2v(store, model_dir: Path, spk_audio: str, emo_audio: Optional[str]):
    """infer_v2_5.py:285-289 (get_emb) + 626-634 / 685-691 (the two call sites)."""
    import torch
    import torchaudio
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

    cfg = _cfg(model_dir)
    stats = torch.load(str(model_dir / cfg.w2v_stat), map_location="cpu")
    mean = stats["mean"]
    std = torch.sqrt(stats["var"])

    d = _w2v_dir(model_dir)
    fe = SeamlessM4TFeatureExtractor.from_pretrained(d)
    sm = Wav2Vec2BertModel.from_pretrained(d)
    sm.eval()

    def get_emb(wav_16k):
        inputs = fe(wav_16k, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            out = sm(input_features=inputs["input_features"],
                     attention_mask=inputs["attention_mask"],
                     output_hidden_states=True)
        feat = out.hidden_states[17]                 # infer_v2_5.py:287
        return (feat - mean) / std                   # infer_v2_5.py:288

    audio, sr = _load_and_cut_audio(spk_audio, MAX_REF_SECONDS)
    audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)   # infer_v2_5.py:628
    store.put("spk_cond_emb", get_emb(audio_16k)[0])

    # infer_v2_5.py:685 — the emotion prompt is loaded with sr=16000 directly.
    emo_path = emo_audio or spk_audio
    emo_wav, _ = _load_and_cut_audio(emo_path, MAX_REF_SECONDS, sr=16000)
    store.put("emo_cond_emb", get_emb(emo_wav)[0])
    del sm, fe


def _load_s2mel(model_dir: Path):
    """MyModel(cfg.s2mel) + load_checkpoint2 — commons.py:390-420, 579-623."""
    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
    cfg = _cfg(model_dir)
    m = MyModel(cfg.s2mel)
    m, _, _, _ = load_checkpoint2(m, None, str(model_dir / cfg.s2mel_checkpoint),
                                 load_only_params=True, ignore_modules=[],
                                 is_distributed=False)
    m.eval()
    return m, cfg


def _stage_lr_prompt(store, model_dir: Path):
    """infer_v2_5.py:650-655 — n_quantizers=3, ylens = ref_mel frame count."""
    import torch
    spk = torch.from_numpy(store.get("spk_cond_emb")).unsqueeze(0)
    ref_mel = torch.from_numpy(store.get("ref_mel"))
    ylens = torch.LongTensor([ref_mel.size(-1)])
    m, _ = _load_s2mel(model_dir)
    with torch.no_grad():
        out = m.models["length_regulator"](spk, ylens=ylens, n_quantizers=3, f0=None)[0]
    store.put("prompt_condition", out[0])
    del m


def _load_gpt(model_dir: Path):
    """UnifiedVoice(**cfg.gpt, spk_cond_mode="campplus") — infer_v2_5.py:138-140."""
    import torch
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint
    cfg = _cfg(model_dir)
    gpt = UnifiedVoice(**cfg.gpt, use_accel=False, spk_cond_mode="campplus")
    load_checkpoint(gpt, str(model_dir / cfg.gpt_checkpoint))
    gpt.eval()
    return gpt, cfg


def _emovec_mat(model_dir: Path, style, emo_vector):
    """infer_v2_5.py:669-679 with use_random=False."""
    import torch
    cfg = _cfg(model_dir)
    emo_num = list(cfg.emo_num)
    emo_matrix = torch.split(torch.load(str(model_dir / cfg.emo_matrix)), emo_num)
    spk_matrix = torch.split(torch.load(str(model_dir / cfg.spk_matrix)), emo_num)
    w = torch.tensor(emo_vector, dtype=torch.float32)
    idx = [find_most_similar_cosine(style, tmp) for tmp in spk_matrix]
    picked = torch.cat([t[i].unsqueeze(0) for i, t in zip(idx, emo_matrix)], 0)
    return (w.unsqueeze(1) * picked).sum(0).unsqueeze(0), w, idx


def _stage_emovec(store, model_dir: Path, emo_alpha: float,
                  emo_vector: Optional[list]):
    """infer_v2_5.py:758-767 + model_v2.py:826-837 (merge_emovec / get_emovec)."""
    import torch
    spk = torch.from_numpy(store.get("spk_cond_emb")).unsqueeze(0)
    emo = torch.from_numpy(store.get("emo_cond_emb")).unsqueeze(0)
    style = torch.from_numpy(store.get("style")).unsqueeze(0)
    gpt, _ = _load_gpt(model_dir)
    with torch.no_grad():
        # NOTE: upstream passes `spk_cond_emb.shape[-1]` (= 1024, the channel dim)
        # as the *length*; ConformerEncoder clamps it to the real T, so the effect
        # is "no masking".  Reproduced verbatim — infer_v2_5.py:758-764.
        emovec = gpt.merge_emovec(spk, emo,
                                  torch.tensor([spk.shape[-1]]),
                                  torch.tensor([emo.shape[-1]]),
                                  alpha=emo_alpha)
        spk_latent = gpt.spk_emb_proj(style)      # model_v2.py:754
    store.put("emovec", emovec[0])
    store.put("spk_latent", spk_latent)
    if emo_vector is not None:
        mat, w, idx = _emovec_mat(model_dir, style, emo_vector)
        print(f"  emo prototype rows (cosine-nearest, use_random=False): {idx}",
              file=sys.stderr)
        with torch.no_grad():
            merged = mat + (1 - torch.sum(w)) * emovec   # infer_v2_5.py:767
        store.put("emovec_mat", merged[0])
    del gpt


def _stage_gpt(store, model_dir: Path, max_mel_tokens: int):
    """infer_v2_5.py:770-800 → model_v2.py:719-812, forced greedy for determinism."""
    import torch
    spk = torch.from_numpy(store.get("spk_cond_emb")).unsqueeze(0)
    emo = torch.from_numpy(store.get("emo_cond_emb")).unsqueeze(0)
    style = torch.from_numpy(store.get("style")).unsqueeze(0)
    text_tokens = torch.from_numpy(store.get("text_tokens")).unsqueeze(0).int()
    lang = torch.LongTensor(store.get("lang_id").astype(np.int64))
    emovec = torch.from_numpy(
        store.get("emovec_mat") if store.has("emovec_mat") else store.get("emovec")
    ).unsqueeze(0)

    gpt, cfg = _load_gpt(model_dir)
    gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)

    # Capture the assembled [pad][cond][text] prefix that GPT2InferenceModel is
    # primed with (model_v2.py:701-712) — the single most useful early diff point.
    captured = {}
    orig_store_mel_emb = gpt.inference_model.store_mel_emb

    def _spy(emb):
        captured["prefix"] = emb.detach().clone()
        return orig_store_mel_emb(emb)

    gpt.inference_model.store_mel_emb = _spy

    with torch.no_grad():
        codes, _ = gpt.inference_speech(
            spk, text_tokens, lang, emo,
            cond_lengths=torch.tensor([spk.shape[-1]]),
            emo_cond_lengths=torch.tensor([emo.shape[-1]]),
            emo_vec=emovec,
            campplus_embedding=style,
            wav=None,
            do_sample=False, num_beams=1,           # determinism, not upstream default
            temperature=1.0, top_p=1.0, top_k=0,
            repetition_penalty=1.0, length_penalty=0.0,
            num_return_sequences=1,
            max_generate_length=max_mel_tokens,
        )
    if "prefix" in captured:
        store.put("gpt_prefix_embeds", captured["prefix"][0])

    # infer_v2_5.py:800-816 — truncate at the first stop_mel_token.
    stop = int(cfg.gpt.stop_mel_token)
    c = codes[0]
    hit = (c == stop).nonzero(as_tuple=False)
    n = int(hit[0].item()) if hit.numel() > 0 else int(c.numel())
    store.put("semantic_codes", c[:n].to(torch.int32).cpu().numpy().astype(np.int32))
    del gpt


def _stage_codec(store, model_dir: Path):
    """infer_v2_5.py:831 — EnhancedCodec.decode(codes) (codec/models.py:205-231)."""
    import torch
    from indextts.codec.models import EnhancedCodec
    cfg = _cfg(model_dir)
    codec = EnhancedCodec(**cfg.semantic_codec, cfg=cfg.semantic_codec)
    codec.load_checkpoint(str(model_dir / "codec.pth"))
    codec.eval()
    codes = torch.from_numpy(store.get("semantic_codes").astype(np.int64)).unsqueeze(0)
    with torch.no_grad():
        s_infer = codec.decode(codes)               # (1, T, 1024)
    store.put("s_infer", s_infer[0])
    del codec


def _stage_lr_cond(store, model_dir: Path, duration_factor: float):
    """infer_v2_5.py:832-841 — target_len = S.T * 1.72 * duration_factor."""
    import torch
    s_infer = torch.from_numpy(store.get("s_infer")).unsqueeze(0)
    target_lengths = torch.LongTensor([int(s_infer.shape[1] * 1.72 * duration_factor)])
    m, _ = _load_s2mel(model_dir)
    with torch.no_grad():
        cond = m.models["length_regulator"](s_infer, ylens=target_lengths,
                                            n_quantizers=3, f0=None)[0]
    store.put("cond", cond[0])
    prompt = torch.from_numpy(store.get("prompt_condition")).unsqueeze(0)
    store.put("cat_condition", torch.cat([prompt, cond], dim=1)[0])
    del m


def _stage_s2mel(store, model_dir: Path, seed: int, steps: int, cfg_rate: float):
    """infer_v2_5.py:840-844 with the ODE's initial noise pinned and dumped."""
    import torch
    cat_condition = torch.from_numpy(store.get("cat_condition")).unsqueeze(0)
    ref_mel = torch.from_numpy(store.get("ref_mel")).unsqueeze(0)
    style = torch.from_numpy(store.get("style")).unsqueeze(0)
    m, _ = _load_s2mel(model_dir)
    m.models["cfm"].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    B, T = cat_condition.size(0), cat_condition.size(1)
    in_ch = m.models["cfm"].in_channels
    g = torch.Generator().manual_seed(seed)
    z = torch.randn([B, in_ch, T], generator=g)     # flow_matching.py:52
    store.put("cfm_noise", z[0])

    # Feed the pinned noise: solve_euler is called directly with the same
    # arguments `inference` would pass (flow_matching.py:53-55).
    t_span = torch.linspace(0, 1, steps + 1)
    with torch.inference_mode():
        vc = m.models["cfm"].solve_euler(
            z.clone(), torch.LongTensor([T]), ref_mel, cat_condition, style, None,
            t_span, inference_cfg_rate=cfg_rate)
    vc = vc[:, :, ref_mel.size(-1):]                # infer_v2_5.py:845
    store.put("s2mel_mel", vc[0])
    del m


def _stage_bigvgan(store, model_dir: Path):
    """infer_v2_5.py:849 — stock nvidia/bigvgan_v2_22khz_80band_256x over mel-80."""
    import torch
    from indextts.s2mel.modules.bigvgan import bigvgan as bvg
    d = model_dir / "hf_cache" / "bigvgan"
    model = bvg.BigVGAN.from_pretrained(str(d) if d.is_dir()
                                        else "nvidia/bigvgan_v2_22khz_80band_256x",
                                        use_cuda_kernel=False)
    model.remove_weight_norm()
    model.eval()
    mel = torch.from_numpy(store.get("s2mel_mel")).unsqueeze(0)
    with torch.no_grad():
        wav = model(mel.float()).squeeze()
    store.put("audio", wav)
    del model


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(model_dir: Path, audio_path: str, out_dir: Optional[Path],
        groups: Iterable[str], *, src: Optional[str] = None,
        text: str = DEFAULT_TEXT, lang: str = DEFAULT_LANG,
        emo_audio: Optional[str] = None, emo_alpha: float = 1.0,
        emo_vector: Optional[list] = None, duration_factor: float = 1.0,
        seed: int = DEFAULT_SEED, steps: int = 25, cfg_rate: float = 0.7,
        max_mel_tokens: int = 1500, normalize_text: bool = False,
        ) -> Dict[str, np.ndarray]:
    _import_upstream(src)
    store = _Store(out_dir)
    groups = [g for g in ALL_STAGES if g in set(groups)]
    for g in groups:
        print(f"[{g}]", file=sys.stderr)
        if g == "text":
            _stage_text(store, model_dir, text, lang, normalize_text)
        elif g == "campplus":
            _stage_campplus(store, model_dir, audio_path)
        elif g == "mel":
            _stage_mel(store, model_dir, audio_path)
        elif g == "w2v":
            _stage_w2v(store, model_dir, audio_path, emo_audio)
        elif g == "lr_prompt":
            _stage_lr_prompt(store, model_dir)
        elif g == "emovec":
            _stage_emovec(store, model_dir, emo_alpha, emo_vector)
        elif g == "gpt":
            _stage_gpt(store, model_dir, max_mel_tokens)
        elif g == "codec":
            _stage_codec(store, model_dir)
        elif g == "lr_cond":
            _stage_lr_cond(store, model_dir, duration_factor)
        elif g == "s2mel":
            _stage_s2mel(store, model_dir, seed, steps, cfg_rate)
        elif g == "bigvgan":
            _stage_bigvgan(store, model_dir)
        import gc
        gc.collect()
    return dict(store.mem)


def dump(model_dir, audio, stages: Set[str], **kwargs) -> Dict[str, np.ndarray]:
    """tools/dump_reference.py contract.

    ``audio`` is ignored: this backend needs a *file path* (librosa's resampling
    and the 15 s cut are part of the reference), so pass ``--audio-path`` through
    kwargs, or set INDEXTTS2_AUDIO.
    """
    model_dir = Path(model_dir)
    audio_path = kwargs.pop("audio_path", None) or os.environ.get("INDEXTTS2_AUDIO")
    if not audio_path:
        raise SystemExit("indextts2 reference dump needs --audio-path / INDEXTTS2_AUDIO")
    wanted = set(stages) if stages else set(DEFAULT_STAGES)
    groups = [g for g, outs in STAGE_GROUPS.items() if wanted & set(outs)]
    out = run(model_dir, audio_path, kwargs.pop("out_dir", None), groups, **kwargs)
    return {k: v for k, v in out.items() if k in wanted}


def _parse_emo_vector(s: Optional[str]) -> Optional[list]:
    if not s:
        return None
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 8:
        raise SystemExit(f"--emo-vector needs 8 values ({','.join(EMO_LABELS)})")
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True,
                    help="IndexTTS-2.5 checkpoint dir (gpt.pth, s2mel.pth, codec.pth, ...)")
    ap.add_argument("--src", default=None,
                    help="index-tts v2.5.0 checkout (or set INDEXTTS2_SRC)")
    ap.add_argument("--audio", required=True, help="speaker reference WAV")
    ap.add_argument("--emo-audio", default=None,
                    help="emotion reference WAV (default: the speaker WAV, "
                         "which also forces emo_alpha=1.0 — infer_v2_5.py:616)")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--lang", default=DEFAULT_LANG)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stages", default=",".join(LOCAL_SAFE_STAGES),
                    help=f"comma list of {ALL_STAGES}, or 'all' / 'local' / 'heavy'")
    ap.add_argument("--emo-alpha", type=float, default=1.0)
    ap.add_argument("--emo-vector", default=None,
                    help="8 comma-separated floats: " + ",".join(EMO_LABELS))
    ap.add_argument("--duration-factor", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg-rate", type=float, default=0.7)
    ap.add_argument("--max-mel-tokens", type=int, default=1500)
    ap.add_argument("--normalize-text", action="store_true",
                    help="run the wetext/nemo front-end (extra deps; off by default "
                         "so the dump stays reproducible without them)")
    args = ap.parse_args()

    if args.stages == "all":
        groups = ALL_STAGES
    elif args.stages == "local":
        groups = LOCAL_SAFE_STAGES
    elif args.stages == "heavy":
        groups = HEAVY_STAGES
    else:
        groups = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(groups) - set(ALL_STAGES)
    if unknown:
        raise SystemExit(f"unknown stage(s): {sorted(unknown)}; valid: {ALL_STAGES}")

    run(Path(args.model_dir), args.audio, Path(args.out_dir), groups,
        src=args.src, text=args.text, lang=args.lang,
        emo_audio=args.emo_audio, emo_alpha=args.emo_alpha,
        emo_vector=_parse_emo_vector(args.emo_vector),
        duration_factor=args.duration_factor, seed=args.seed, steps=args.steps,
        cfg_rate=args.cfg_rate, max_mel_tokens=args.max_mel_tokens,
        normalize_text=args.normalize_text)
    print(f"\nwrote stage .npy files to {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
