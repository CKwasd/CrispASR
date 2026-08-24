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
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer tokenizers pyyaml librosa scipy")

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
# F16 too: running the S2A at F16 separates quantization error compounding
# through the 25-step ODE from a genuine remaining port bug.
s2a_f16 = grab("cstr/confucius4-tts-GGUF", "confucius4-tts-s2a-f16.gguf")
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
kh.step(f"TTS ({ODE_STEPS} ODE steps, CFG on) -- q4_k and f16")

env_base = os.environ.copy()
env_base["CRISPASR_CONFUCIUS4_TEXT_IDS"] = token_ids_str

runs = {}
for tag, s2a in (("q4_k", s2a_path), ("f16", s2a_f16)):
    dump = TEMP / f"dump_{tag}"
    dump.mkdir(parents=True, exist_ok=True)
    wav = TEMP / f"confucius4_{tag}.wav"
    env = dict(env_base)
    env["CRISPASR_CONFUCIUS4_DUMP_S2A"] = str(dump)
    r = subprocess.run(
        [crispasr_bin, "--backend", "confucius4-tts", "-m", t2s_path,
         "--codec-model", s2a, "--tts", TEST_TEXT,
         "--tts-output", str(wav), "--tts-steps", str(ODE_STEPS), "-v"],
        capture_output=True, text=True, timeout=7200, env=env,
    )
    ok = wav.exists() and os.path.getsize(str(wav)) > 100
    print(f"  [{tag}] rc={r.returncode}  wav={'%d B' % os.path.getsize(str(wav)) if ok else 'NONE'}")
    for line in r.stderr.split("\n"):
        if any(k in line for k in ("flow-matching:", "regulator OK", "time schedule", "BigVGAN:")):
            print(f"  [{tag}] " + line.strip())
    if r.returncode != 0:
        for line in [l for l in r.stderr.split("\n") if l.strip()][-20:]:
            print(f"  [{tag}] ! " + line.strip())
    runs[tag] = {"rc": r.returncode, "stderr": r.stderr, "wav": wav, "ok": ok, "dump": dump}

res = runs["q4_k"]["stderr"]
wav_ok = runs["q4_k"]["ok"]
tts_wav = runs["q4_k"]["wav"]


# ── Phase 7: per-stage parity against the real PyTorch S2A ──────────────────
kh.step("S2A parity vs PyTorch reference")
parity = REPO / "tools" / "s2a_parity.py"
for tag in ("q4_k", "f16"):
    dump = runs[tag]["dump"]
    print(f"  --- {tag} ---")
    if not (dump / "shapes.txt").exists():
        print("  SKIP: no dump produced")
        continue
    cmd = [sys.executable, str(parity), "--dump-dir", str(dump),
           "--ref-repo", str(REF), "--s2a-ckpt", s2a_ckpt,
           "--steps", str(ODE_STEPS), "--cfg", "0.7"]
    if tag == "q4_k":                       # vocode once, from the shipped quant
        cmd += ["--vocode-out", str(TEMP / "vocoded")]
    pres = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    print(f"  parity rc={pres.returncode}")
    for line in pres.stdout.split("\n"):
        if line.strip():
            print("  " + line.rstrip())
    if pres.returncode != 0:
        for line in [l for l in pres.stderr.split("\n") if l.strip()][-25:]:
            print("  ! " + line.strip())


# ── Phase 8: ASR roundtrip (the acceptance gate) ────────────────────────────
kh.step("ASR roundtrip")
import urllib.request

whisper_model = str(TEMP / "models" / "ggml-base.en.bin")
os.makedirs(os.path.dirname(whisper_model), exist_ok=True)
if not os.path.exists(whisper_model):
    urllib.request.urlretrieve(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        whisper_model,
    )

ORIG = set(w.strip(".,!?").lower() for w in TEST_TEXT.split())


def asr_score(tag, path):
    """Transcribe and word-overlap score one wav.  The decisive comparison is
    cpp vs ref: if the REFERENCE audio is also unintelligible then the S2A port
    is not the blocker, the missing speaker conditioning is."""
    if not (path and os.path.exists(path) and os.path.getsize(path) > 100):
        print(f"  [{tag}] SKIP: no wav")
        return
    a = subprocess.run([crispasr_bin, "-m", whisper_model, "-f", str(path), "--no-prints"],
                       capture_output=True, text=True, timeout=600)
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", a.stdout)
    hit = {w for w in ORIG if w in clean.lower()}
    print(f"  [{tag}] rc={a.returncode}  overlap {len(hit)}/{len(ORIG)} = "
          f"{100 * len(hit) / max(len(ORIG), 1):.0f}%")
    print(f"  [{tag}] transcript: {clean.strip()[:400]}")
    return len(hit)


asr_score("cpp-cli-q4_k", str(runs["q4_k"]["wav"]) if runs["q4_k"]["ok"] else None)
asr_score("cpp-cli-f16", str(runs["f16"]["wav"]) if runs["f16"]["ok"] else None)
asr_score("cpp-mel/torch-voc", str(TEMP / "vocoded" / "cpp_mel.wav"))
asr_score("REF-mel/torch-voc", str(TEMP / "vocoded" / "ref_mel.wav"))
print("  NOTE: if REF-mel is also ~0%, the port is not the blocker -- the model is")
print("        zero-shot and always has a speaker prompt, which is still all zeros.")

# ── Phase 9: summary ────────────────────────────────────────────────────────
kh.step("summary")
m = re.search(r"generated (\d+) semantic codes", res)
print(f"  semantic codes : {m.group(1) if m else '?'}")
m = re.search(r"time schedule: (\w+)", res)
print(f"  time schedule  : {m.group(1) if m else '?'}   (expect linear)")
m = re.search(r"cfg=([0-9.]+)", res)
print(f"  cfg rate       : {m.group(1) if m else '?'}   (expect 0.70)")
m = re.search(r"regulator OK \(([0-9]+)→([0-9]+)→([0-9]+) dims\)", res)
print(f"  regulator dims : {'->'.join(m.groups()) if m else '?'}   (expect 2304->1024->512)")
print(f"  TTS rc         : q4_k={runs['q4_k']['rc']} f16={runs['f16']['rc']}")
print("\n=== Done ===", flush=True)
