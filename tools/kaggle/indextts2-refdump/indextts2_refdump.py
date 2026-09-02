#!/usr/bin/env python3
"""Kaggle kernel: IndexTTS-2.5 stage-by-stage reference dump (torch oracle).

Runs `tools/reference_backends/indextts2.py` over the full pipeline on a fixed
(reference wav, text, seed) and saves every stage as .npy plus one bundled .npz,
so `crispasr-diff` can compare the C++ `indextts2` backend stage by stage.

Why a kernel: the fp32 checkpoint set is ~8.3 GB (gpt.pth 3.26 GB + codec 0.61 GB
+ s2mel 0.42 GB + w2v-BERT 2.32 GB + BigVGAN 0.45 GB + CAMPPlus 28 MB) and the
GPT-2 AR decode is 1500 steps.  The reference VPS has 8 GB of RAM, so only the
light stages (text / campplus / mel / w2v / lr_prompt / emovec) run there; the
heavy ones (gpt / codec / lr_cond / s2mel / bigvgan) run here.

Determinism: GPT-2 is forced greedy (do_sample=False, num_beams=1) and the CFM
ODE's initial noise is drawn from a seeded generator and dumped as `cfm_noise`,
so the C++ side can be fed the identical tensor.  Upstream's sampling defaults
(num_beams=3, repetition_penalty=10.0, top_k=30, top_p=0.8 —
infer_v2_5.py:731-739) are NOT reproducible and are deliberately overridden.

LICENSE: IndexTTS-2.5 is under bilibili's Model Use License.  This kernel only
runs the upstream code to produce *activations*; it must not publish weights or
GGUFs.  The .npy/.npz activation dump is uploaded to the private regression
fixtures repo only.

Push (under chr1s4):
  export KAGGLE_API_TOKEN=...
  python -m kaggle kernels push -p tools/kaggle/indextts2-refdump
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
SRC = WORK / "index-tts"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
CKPT = TEMP / "indextts25-src"
OUT = WORK / "indextts2-ref"

BRANCH = os.environ.get("CRISPASR_REF", "main")
UPSTREAM_TAG = "v2.5.0"
SRC_REPO = "IndexTeam/IndexTTS-2.5"
HF_FIXTURES = "cstr/crispasr-regression-fixtures"

# Fixed diff inputs — keep these stable, the C++ side must use the same triple.
REF_TEXT = "Hello, this is a reference dump."
REF_LANG = "en"
REF_SEED = 1234
REF_WAV_REL = "samples/jfk.wav"

# ── Phase 0: clone CrispASR ──────────────────────────────────────────────────
print(f"[0] cloning CrispASR {BRANCH}", flush=True)
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call([
    "git", "clone", "--depth", "1", "--branch", BRANCH,
    "https://github.com/CrispStrobe/CrispASR.git", str(REPO),
])
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
hf_token = kh.resolve_hf_token()
kh.step("cloned", branch=BRANCH, hf_token_ok=bool(hf_token))
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

# ── Phase 1: clone the upstream v2.5.0 tree ──────────────────────────────────
kh.step("clone upstream")
if SRC.exists():
    shutil.rmtree(SRC)
subprocess.check_call([
    "git", "clone", "--depth", "1", "--branch", UPSTREAM_TAG,
    "https://github.com/index-tts/index-tts.git", str(SRC),
])
print(f"  upstream: {SRC} @ {UPSTREAM_TAG}", flush=True)

# ── Phase 2: deps ────────────────────────────────────────────────────────────
# Only what the upstream *inference* path needs.  No training deps, no
# nemo_text_processing / wetext (the dumper runs with text normalisation off),
# no deepspeed, no flash-attn (use_accel=False).
kh.step("install deps")
# transformers is PINNED: index-tts v2.5.0 vendors its own copy of HF's
# generation utils (indextts/gpt/transformers_generation_utils.py:28) which
# imports `OffloadedCache` — removed in transformers 5.x — so anything newer
# than the upstream pin (pyproject.toml:61) fails at import time.
kh.sh_with_progress(
    "pip install -q "
    "'transformers==4.52.1' 'tokenizers<0.22' "
    "torchaudio librosa soundfile tiktoken "
    "omegaconf munch einops openai-whisper sentencepiece "
    "huggingface_hub hf_transfer numpy scipy"
)

# ── Phase 3: download the checkpoint set ─────────────────────────────────────
kh.step("download IndexTTS-2.5")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download, HfApi  # noqa: E402

CKPT.mkdir(parents=True, exist_ok=True)
free = kh.free_gb(str(CKPT))
print(f"  cache: {CKPT} (free: {free:.1f} GiB)" if free else f"  cache: {CKPT}",
      flush=True)
snapshot_download(
    repo_id=SRC_REPO,
    local_dir=str(CKPT),
    # Skip qwen0.6bemo4-merge/: QwenEmotion is only needed for use_emo_text=True,
    # which is off by default (infer_v2_5.py:126-131) and not part of the diff.
    allow_patterns=["*.pth", "*.pt", "*.tiktoken", "config.yaml"],
    max_workers=2,
)
for f in sorted(CKPT.iterdir()):
    if f.is_file():
        print(f"  {f.stat().st_size / 1024**2:9.1f} MiB  {f.name}", flush=True)

# Auxiliary models the pipeline loads by name — pre-stage them into hf_cache/ so
# the dumper's local_files_only lookups hit (infer_v2_5.py:170-234).
kh.step("download aux models")
aux = CKPT / "hf_cache"
aux.mkdir(exist_ok=True)
snapshot_download("facebook/w2v-bert-2.0", local_dir=str(aux / "w2v-bert-2.0"))
snapshot_download("nvidia/bigvgan_v2_22khz_80band_256x", local_dir=str(aux / "bigvgan"))
from huggingface_hub import hf_hub_download  # noqa: E402

cp = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
shutil.copy(cp, aux / "campplus_cn_common.bin")
print("  aux models staged under", aux, flush=True)

# ── Phase 4: run every stage ─────────────────────────────────────────────────
kh.step("run reference dump")
OUT.mkdir(parents=True, exist_ok=True)
env = dict(os.environ)
env["INDEXTTS2_SRC"] = str(SRC)
env["PYTHONPATH"] = f"{SRC}:{REPO / 'tools'}"
subprocess.check_call([
    sys.executable, str(REPO / "tools" / "reference_backends" / "indextts2.py"),
    "--model-dir", str(CKPT),
    "--src", str(SRC),
    "--audio", str(REPO / REF_WAV_REL),
    "--text", REF_TEXT,
    "--lang", REF_LANG,
    "--seed", str(REF_SEED),
    "--out-dir", str(OUT),
    "--stages", "all",
], env=env)

import numpy as np  # noqa: E402

stages = {p.stem: np.load(p) for p in sorted(OUT.glob("*.npy"))}
for k, v in stages.items():
    print(f"  {k:22s} {str(v.shape):18s} {v.dtype}", flush=True)
bundle = WORK / "indextts2-ref.npz"
np.savez_compressed(bundle, **stages)
print(f"  bundle: {bundle} ({bundle.stat().st_size / 1024**2:.1f} MiB)", flush=True)
kh.step("dump_done", n_stages=len(stages),
        size_mib=round(bundle.stat().st_size / 1024**2, 1))

# ── Phase 5: ASR round-trip on the synthesised WAV (house rule) ──────────────
kh.step("asr roundtrip")
try:
    import soundfile as sf
    import whisper

    wav_path = WORK / "indextts2-ref.wav"
    sf.write(wav_path, stages["audio"].astype(np.float32), 22050)
    asr = whisper.load_model("small")
    text = asr.transcribe(str(wav_path), language="en")["text"].strip()
    print(f"  reference text : {REF_TEXT}", flush=True)
    print(f"  ASR round-trip : {text}", flush=True)
    kh.step("asr_done", transcript=text)
except Exception as e:  # noqa: BLE001
    print(f"  ASR round-trip skipped: {e!r}", flush=True)

# ── Phase 6: upload the activation dump ─────────────────────────────────────
if hf_token:
    kh.step("upload fixtures")
    api = HfApi(token=hf_token)
    try:
        api.create_repo(repo_id=HF_FIXTURES, repo_type="model", exist_ok=True,
                        private=True)
    except Exception as e:  # noqa: BLE001
        print(f"  repo create: {e}", flush=True)
    api.upload_file(
        path_or_fileobj=str(bundle),
        path_in_repo="indextts2-ref.npz",
        repo_id=HF_FIXTURES, repo_type="model",
        commit_message="Add IndexTTS-2.5 stage-by-stage reference activations",
    )
    print("  uploaded indextts2-ref.npz", flush=True)
else:
    print("[6] no HF_TOKEN — staged locally in /kaggle/working", flush=True)

print("\n=== Done ===", flush=True)
