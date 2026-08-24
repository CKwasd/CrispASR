#!/usr/bin/env python3
"""Kaggle kernel: verify the Confucius4-TTS S2A fixes (#377, feat/confucius4-cfg).

Runs the full pipeline and then does what the VPS cannot: drives the REAL
PyTorch S2A (confuciustts/flow) on exactly the inputs and initial noise the C++
runtime produced, and compares per stage.  Finishes with a TTS -> ASR roundtrip,
which is the only acceptance gate (HARD RULE #3).

What is under test (all on feat/confucius4-cfg):
  1. s2a_sinusoidal_embed  -- scale=1000, cos-then-sin
  2. InterpolateRegulator  -- the learned conv stack, was skipped entirely
  3. classifier-free guidance in the ODE
  4. T2S sampling temperature (the CLI was forcing greedy)
  5. linear ODE time schedule (the port used cosine)
  6. the prompt string -- LANGUAGE_TOKEN_MAP["en"] is CHINESE, and the old test
     kernel invented an English one, so every previous run fed the T2S a prompt
     it was never trained on

Push (chr1s4 is the default account):
  python -m kaggle kernels push -p tools/kaggle/confucius4-tts-cfg-verify
"""

import os
import re
import subprocess
import sys
from pathlib import Path

BRANCH = "feat/confucius4-cfg"
WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
REF = TEMP / "confucius4-ref"
DUMP = TEMP / "s2a_dump"

# Keep the run tractable on a Kaggle CPU box: short text -> few semantic codes.
TEST_TEXT = "The quick brown fox jumps over the lazy dog."
LANG = "en"
ODE_STEPS = int(os.environ.get("ODE_STEPS", "25"))

# ── Phase 0: clone repo (the branch under test) ─────────────────────────────
print(f"=== Phase 0: clone {BRANCH} ===", flush=True)
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "-b", BRANCH,
        "https://github.com/CrispStrobe/CrispASR", str(REPO),
    ])
subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=str(REPO))
head = subprocess.run(["git", "log", "--oneline", "-3"], cwd=str(REPO),
                      capture_output=True, text=True).stdout
print("  HEAD:\n   " + "\n   ".join(head.strip().split("\n")))

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh
kh.init_progress()

# ── Phase 1: deps + HF token ────────────────────────────────────────────────
kh.step("install deps")
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer tokenizers pyyaml librosa")

kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
from huggingface_hub import hf_hub_download

# ── Phase 2: the Python blueprint ───────────────────────────────────────────
kh.step("clone Python blueprint")
if not REF.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/netease-youdao/Confucius4-TTS", str(REF),
    ])
sys.path.insert(0, str(REF))
from confuciustts.utils.text_utils import LANGUAGE_TOKEN_MAP

# ── Phase 3: models ─────────────────────────────────────────────────────────
kh.step("download GGUFs")
mdir = str(TEMP / "models")


def grab(repo, fname, **kw):
    p = hf_hub_download(repo, fname, local_dir=kw.pop("d", mdir), token=hf_token, **kw)
    print(f"  {fname}: {os.path.getsize(p) / 1024**2:.0f} MB")
    return p


t2s_path = grab("cstr/confucius4-tts-GGUF", "confucius4-tts-t2s-q4_k.gguf")
s2a_path = grab("cstr/confucius4-tts-GGUF", "confucius4-tts-s2a-q4_k.gguf")
voc_path = grab("cstr/confucius4-tts-GGUF", "confucius4-tts-bigvgan-22k-f16.gguf")
s2a_ckpt = grab("netease-youdao/Confucius4-TTS", "s2a_model.pt", d=str(TEMP / "torch"))

# ── Phase 4: tokenize with the CORRECT prompt ───────────────────────────────
kh.step("tokenize (real LANGUAGE_TOKEN_MAP)")
tok_path = hf_hub_download("netease-youdao/Confucius4-TTS", "tokenizer.json",
                           local_dir=str(TEMP / "tokenizer"), token=hf_token)
from tokenizers import Tokenizer

tok = Tokenizer.from_file(tok_path)
lang_token = LANGUAGE_TOKEN_MAP.get(LANG, f"请用{LANG}朗读接下来的文字")
formatted = f"You are a helpful assistant. {lang_token}:{TEST_TEXT}"
ids = tok.encode(formatted).ids
token_ids_str = ",".join(str(x) for x in ids)
print(f"  lang_token : {lang_token!r}   <-- Chinese, per LANGUAGE_TOKEN_MAP")
print(f"  formatted  : {formatted}")
print(f"  n_ids={len(ids)}  ids={token_ids_str}")

# ── Phase 5: build ──────────────────────────────────────────────────────────
kh.step("build CrispASR")
BUILD = TEMP / "build"
BUILD.mkdir(parents=True, exist_ok=True)
flags = kh.cache_and_link_flags()
kh.sh_with_progress(
    f"cmake -G Ninja -B {BUILD} -S {REPO} -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF "
    + " ".join(flags)
)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"cmake --build {BUILD} -j{kh.safe_build_jobs(gpu=False)} --target crispasr-cli"
    )
crispasr_bin = str(BUILD / "bin" / "crispasr")
print(f"  binary: {crispasr_bin} ({os.path.getsize(crispasr_bin) / 1024**2:.0f} MB)")

# ── Phase 6: synthesize, dumping the S2A stages ─────────────────────────────
kh.step(f"TTS ({ODE_STEPS} ODE steps, CFG on)")
DUMP.mkdir(parents=True, exist_ok=True)
tts_wav = TEMP / "confucius4_cfg.wav"

env = os.environ.copy()
env["CRISPASR_CONFUCIUS4_TEXT_IDS"] = token_ids_str
env["CRISPASR_CONFUCIUS4_DUMP_S2A"] = str(DUMP)

res = subprocess.run(
    [crispasr_bin, "--backend", "confucius4-tts", "-m", t2s_path,
     "--codec-model", s2a_path, "--tts", TEST_TEXT,
     "--tts-output", str(tts_wav), "--tts-steps", str(ODE_STEPS), "-v"],
    capture_output=True, text=True, timeout=7200, env=env,
)
print(f"  TTS rc={res.returncode}")
for line in res.stderr.split("\n"):
    if any(k in line for k in ("confucius4:", "output:", "DiT", "ODE", "schedule", "cfg=")):
        print("  " + line.strip())
if res.returncode != 0:
    print("  --- last 30 stderr lines ---")
    for line in [l for l in res.stderr.split("\n") if l.strip()][-30:]:
        print("  " + line.strip())

wav_ok = tts_wav.exists() and os.path.getsize(str(tts_wav)) > 100
print(f"  WAV: {'%d bytes' % os.path.getsize(str(tts_wav)) if wav_ok else 'NOT PRODUCED'}")

# ── Phase 7: per-stage parity against the real PyTorch S2A ──────────────────
kh.step("S2A parity vs PyTorch reference")
parity = REPO / "tools" / "s2a_parity.py"
if not (DUMP / "shapes.txt").exists():
    print("  SKIP: no dump produced (TTS failed before the S2A stage)")
elif not parity.exists():
    print(f"  SKIP: {parity} missing on this branch")
else:
    pres = subprocess.run(
        [sys.executable, str(parity), "--dump-dir", str(DUMP),
         "--ref-repo", str(REF), "--s2a-ckpt", s2a_ckpt,
         "--steps", str(ODE_STEPS), "--cfg", "0.7"],
        capture_output=True, text=True, timeout=7200,
    )
    print(f"  parity rc={pres.returncode}")
    for line in pres.stdout.split("\n"):
        if line.strip():
            print("  " + line.rstrip())
    if pres.returncode != 0:
        for line in [l for l in pres.stderr.split("\n") if l.strip()][-25:]:
            print("  ! " + line.strip())

# ── Phase 8: ASR roundtrip (the acceptance gate) ────────────────────────────
kh.step("ASR roundtrip")
if wav_ok:
    import urllib.request

    whisper_model = str(TEMP / "models" / "ggml-base.en.bin")
    os.makedirs(os.path.dirname(whisper_model), exist_ok=True)
    if not os.path.exists(whisper_model):
        urllib.request.urlretrieve(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
            whisper_model,
        )
    asr = subprocess.run([crispasr_bin, "-m", whisper_model, "-f", str(tts_wav), "--no-prints"],
                         capture_output=True, text=True, timeout=600)
    # Feed the FULL ansi-stripped stdout to a word-OVERLAP check: the CLI
    # interleaves device/model-load noise and appends an AI-disclosure line.
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", asr.stdout)
    print(f"  ASR rc={asr.returncode}")
    print(f"  ASR stdout: {clean.strip()[:600]}")
    orig = set(w.strip(".,!?").lower() for w in TEST_TEXT.split())
    heard = clean.lower()
    hit = {w for w in orig if w in heard}
    print(f"  word overlap: {len(hit)}/{len(orig)} = {100 * len(hit) / max(len(orig), 1):.0f}%  "
          f"missing={sorted(orig - hit)}")
else:
    print("  SKIP: no WAV")

# ── Phase 9: summary ────────────────────────────────────────────────────────
kh.step("summary")
m = re.search(r"generated (\d+) semantic codes", res.stderr)
print(f"  semantic codes : {m.group(1) if m else '?'}")
m = re.search(r"time schedule: (\w+)", res.stderr)
print(f"  time schedule  : {m.group(1) if m else '?'}   (expect linear)")
m = re.search(r"cfg=([0-9.]+)", res.stderr)
print(f"  cfg rate       : {m.group(1) if m else '?'}   (expect 0.70)")
m = re.search(r"regulator OK \(([0-9]+)→([0-9]+)→([0-9]+) dims\)", res.stderr)
print(f"  regulator dims : {'->'.join(m.groups()) if m else '?'}   (expect 2304->1024->512)")
print(f"  TTS rc         : {res.returncode}")
print("\n=== Done ===", flush=True)
