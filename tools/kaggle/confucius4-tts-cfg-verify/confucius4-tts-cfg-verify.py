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
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer tokenizers pyyaml librosa scipy "
                    "torchaudio safetensors transformers")

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
# The raw tokenizers post_processor only prepends <s>, but the reference loads
# AutoTokenizer -> LlamaTokenizerFast, whose add_eos_token=True (from
# tokenizer_config.json) rebuilds the template to ALSO append </s> (id 2).
# Verified: AutoTokenizer ids end [..., 28723, 2]; raw ids end [..., 28723].
if ids and ids[-1] != 2:
    ids = ids + [2]
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


# ── Phase 6b: speaker conditioning ──────────────────────────────────────────
# The model is zero-shot -- run 3 showed the PyTorch reference itself babbles
# with zero conditioning.  Feed real conditioning derived from a reference wav
# and see whether the roundtrip passes.  Two arms, cheap one first:
#   s2a-only : CAMPPlus style + reference mel  (needs only campplus, ~28 MB)
#   full     : + the T2S condition_emb         (needs w2v-BERT 2.4 GB + T2S 2.6 GB)
# If s2a-only already yields speech, the semantic codes were fine all along and
# the T2S condition_emb is a voice-identity refinement rather than a blocker.
kh.step("speaker conditioning")
dumper = REPO / "tools" / "confucius4_dump_conditioning.py"
prompt_wav = REPO / "samples" / "jfk.wav"
w2v_stats = None

cond_runs = {}
t2s_ckpt = None
for arm in ("s2a-only", "full"):
    cdir = TEMP / f"cond_{arm}"
    cmd = [sys.executable, str(dumper), "--ref-repo", str(REF),
           "--prompt-wav", str(prompt_wav), "--out-dir", str(cdir)]
    if arm == "s2a-only":
        cmd.append("--no-w2v")
    else:
        if w2v_stats is None:
            w2v_stats = grab("netease-youdao/Confucius4-TTS", "wav2vec2bert_stats.pt",
                             d=str(TEMP / "torch"))
        t2s_ckpt = grab("netease-youdao/Confucius4-TTS", "t2s_model.safetensors",
                        d=str(TEMP / "torch"))
        cmd += ["--t2s-ckpt", t2s_ckpt, "--w2v-stats", w2v_stats]

    dres = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    print(f"  [{arm}] dumper rc={dres.returncode}")
    for line in dres.stdout.split("\n"):
        if line.strip():
            print(f"  [{arm}] " + line.strip())
    if dres.returncode != 0:
        for line in [l for l in dres.stderr.split("\n") if l.strip()][-20:]:
            print(f"  [{arm}] ! " + line.strip())
        continue

    wav = TEMP / f"confucius4_cond_{arm}.wav"
    dump = TEMP / f"dump_cond_{arm}"
    dump.mkdir(parents=True, exist_ok=True)
    env = dict(env_base)
    env["CRISPASR_CONFUCIUS4_COND_DIR"] = str(cdir)
    env["CRISPASR_CONFUCIUS4_DUMP_S2A"] = str(dump)
    r = subprocess.run(
        [crispasr_bin, "--backend", "confucius4-tts", "-m", t2s_path,
         "--codec-model", s2a_f16, "--tts", TEST_TEXT,
         "--tts-output", str(wav), "--tts-steps", str(ODE_STEPS), "-v"],
        capture_output=True, text=True, timeout=7200, env=env,
    )
    ok = wav.exists() and os.path.getsize(str(wav)) > 100
    print(f"  [{arm}] TTS rc={r.returncode} wav={'%d B' % os.path.getsize(str(wav)) if ok else 'NONE'}")
    for line in r.stderr.split("\n"):
        if any(k in line for k in ("conditioning set", "flow-matching:", "EOS at", "semantic codes",
                                   "BigVGAN:")):
            print(f"  [{arm}] " + line.strip())
    if r.returncode != 0:
        for line in [l for l in r.stderr.split("\n") if l.strip()][-20:]:
            print(f"  [{arm}] ! " + line.strip())
    cond_runs[arm] = {"wav": wav, "ok": ok}


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


# ── Phase 7b: T2S parity ────────────────────────────────────────────────────
# S2A is exact yet the reference S2A babbles on these codes, so the codes are
# wrong and the bug is upstream.  This is the stage the harness never covered.
kh.step("T2S parity vs PyTorch reference")
t2s_parity = REPO / "tools" / "t2s_parity.py"
_full = TEMP / "dump_cond_full"
_w2v = TEMP / "cond_full" / "w2v_features.bin"
if not (t2s_parity.exists() and (_full / "shapes.txt").exists() and _w2v.exists()):
    print("  SKIP: needs the full-conditioning run (dump + w2v features)")
else:
    tres = subprocess.run(
        [sys.executable, str(t2s_parity), "--dump-dir", str(_full), "--ref-repo", str(REF),
         "--t2s-ckpt", t2s_ckpt, "--w2v-features", str(_w2v)],
        capture_output=True, text=True, timeout=7200,
    )
    print(f"  t2s parity rc={tres.returncode}")
    for line in tres.stdout.split("\n"):
        if line.strip():
            print("  " + line.rstrip())
    if tres.returncode != 0:
        for line in [l for l in tres.stderr.split("\n") if l.strip()][-25:]:
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


asr_score("nocond-q4_k", str(runs["q4_k"]["wav"]) if runs["q4_k"]["ok"] else None)
asr_score("nocond-f16", str(runs["f16"]["wav"]) if runs["f16"]["ok"] else None)
for _arm, _r in cond_runs.items():
    asr_score(f"COND-{_arm}", str(_r["wav"]) if _r["ok"] else None)
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
