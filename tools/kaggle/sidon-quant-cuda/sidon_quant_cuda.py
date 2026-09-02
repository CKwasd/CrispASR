#!/usr/bin/env python3
"""
Issue #416 — Sidon quantized GGUFs decode to a full-length file of SILENCE
while the f16 model restores the same clip correctly. PROOF kernel for the
CUDA arm, on a real CUDA box (Kaggle P100/T4).

The report (Disonantemus): `-m sidon-v0.1-f16.gguf --s2s` works;
`sidon-v0.1-q4_k.gguf` and `sidon-v0.1-q8_0.gguf` both write 1.80 s of pure
silence. rc=0, correct duration, correct sample rate, no error message.

Already eliminated before this kernel, so do not re-litigate them here:
  * x86-64 CPU — all three artifacts decode normally (f16 rms 0.128101,
    q8_0 0.127733, q4_k 0.126898 on the same clip). q8_0 is ~lossless, so the
    silence CANNOT be a precision effect; it is a code path only quantized
    weights take.
  * Vulkan — `distance_embedding` is the only quantized tensor feeding a
    broadcasting matmul (src0 [64,73,1,1] over H=16 heads), the classic
    CPU-tolerant/GPU-strict shape. Probed at sidon's exact shapes on lavapipe:
    F16 cos 0.999999, Q8_0/Q4_0 0.999985 vs CPU. test-backend-ops -o MUL_MAT
    passed 1536/1536. Hypothesis dead.

So this kernel asks exactly one question: does CUDA reproduce it?

Design notes that matter for trusting the answer:
  * EVERY model runs on BOTH backends on the SAME box, back to back. A quant
    that is silent on CUDA but fine under -ng on the same machine is a GPU-path
    fault; silent on both would mean the VPS CPU result was the outlier.
    Without the paired control a "silence" reading proves nothing.
  * rc=0 is NOT the verdict (HARD RULE #8 / #4a). Silence IS rc=0 here — that
    is the whole bug. The verdict comes from measuring the produced PCM:
    rms, peak, and the fraction of non-zero samples, read back out of the WAV.
  * main now warns when a stage handoff is degenerate and names WHICH stage
    (w2v-BERT predictor vs DAC decoder) and whether it is NaN or all-zero.
    Those lines are captured verbatim — if CUDA reproduces, they localize the
    fault to a stage without a second run.

Results -> /kaggle/working/sidon416_results.json
"""
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp/i416")
TMP.mkdir(parents=True, exist_ok=True)
MODELS = Path("/kaggle/temp/models")
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "sidon416_results.json"

HERE = Path(__file__).resolve().parent
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = "main"
CLONE = Path("/kaggle/temp/CrispASR")
if not CLONE.exists():
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                        "--recurse-submodules", CRISPASR_URL, str(CLONE)],
                       check=True, timeout=1800)
    except Exception as e:  # fall through to the bundled harness copy
        print(f"clone failed: {e}", flush=True)
sys.path.insert(0, str(CLONE / "tools" / "kaggle") if CLONE.exists() else str(HERE))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
HF_TOKEN = kh.resolve_hf_token()
kh.step("hf.token", present=bool(HF_TOKEN))
from huggingface_hub import hf_hub_download  # noqa: E402

SIDON_REPO = "cstr/Sidon-GGUF"
# q6_k is included even though the reporter did not try it: if q8_0 and q4_k
# fail it says whether EVERY quant is affected or only some, which separates
# "quantized weights in general" from "one specific quant type's kernel".
SIDON_FILES = ["sidon-v0.1-f16.gguf", "sidon-v0.1-q8_0.gguf",
               "sidon-v0.1-q6_k.gguf", "sidon-v0.1-q4_k.gguf"]

CLIP = TMP / "in3s.wav"


def sh(cmd, timeout=None, env=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout, env=env)


def build_crispasr():
    kh.install_build_toolchain()
    if not (CLONE / "CMakeLists.txt").exists():
        raise RuntimeError("repo clone missing")
    if not (CLONE / "ggml" / "CMakeLists.txt").exists():
        sh(f"cd {CLONE} && git submodule update --init ggml", timeout=1200)
    arch = kh.detect_cuda_arch()
    flags = (["-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_OPUS_FETCH=ON"]
             + kh.cuda_build_flags(arch) + kh.cache_and_link_flags())
    kh.step("build.configure", arch=arch)
    r = sh(f"cd {CLONE} && cmake -G Ninja -B build " + " ".join(flags), timeout=2400)
    if r.returncode != 0:
        kh.step("build.configure_FAILED", stderr=(r.stderr or "")[-2000:])
        raise RuntimeError("cmake configure failed")
    jobs = kh.safe_build_jobs(gpu=True)
    with kh.build_heartbeat("build.ninja", interval_s=30):
        kh.sh_with_progress(f"cmake --build build -j{jobs} --target crispasr-cli", cwd=str(CLONE))
    binp = CLONE / "build" / "bin" / "crispasr"
    if not binp.exists():
        raise RuntimeError("crispasr binary not produced")
    kh.step("build.ready", path=str(binp))
    try:
        kh.export_ccache_tar()
    except Exception:
        pass
    return binp


def make_clip():
    """3 s of samples/jfk.wav at 16 kHz — near the reporter's 1.8 s, and short
    enough that 8 sidon runs fit the kernel budget (the DAC dominates)."""
    src = CLONE / "samples" / "jfk.wav"
    w = wave.open(str(src))
    n = int(3.0 * w.getframerate())
    data = w.readframes(n)
    o = wave.open(str(CLIP), "wb")
    o.setnchannels(w.getnchannels())
    o.setsampwidth(w.getsampwidth())
    o.setframerate(w.getframerate())
    o.writeframes(data)
    o.close()
    kh.step("clip.ready", seconds=3.0, rate=w.getframerate())


def measure(path):
    """Read the produced PCM back. This — not the exit code — is the verdict:
    the reported failure is a well-formed WAV whose samples are all zero."""
    if not Path(path).exists():
        return None
    w = wave.open(str(path))
    n = w.getnframes()
    raw = w.readframes(n)
    w.close()
    if n == 0:
        return dict(n=0, rms=0.0, peak=0.0, nonzero_frac=0.0)
    import array
    a = array.array("h")
    a.frombytes(raw)
    acc = 0.0
    peak = 0
    nz = 0
    for s in a:
        acc += float(s) * s
        av = -s if s < 0 else s
        if av > peak:
            peak = av
        if s:
            nz += 1
    return dict(n=len(a), rms=(acc / len(a)) ** 0.5 / 32768.0,
                peak=peak / 32768.0, nonzero_frac=nz / len(a))


def run_sidon(binp, model, tag, gpu):
    out = TMP / f"out_{tag}.wav"
    if out.exists():
        out.unlink()
    ng = "" if gpu else "-ng "
    cmd = (f"{binp} {ng}-m {MODELS / model} -f {CLIP} --s2s "
           f"--s2s-output {out} -t 4")
    with kh.build_heartbeat(f"run.{tag}", interval_s=30):
        try:
            r = sh(cmd, timeout=2400)
            rc, err = r.returncode, (r.stderr or "")
        except subprocess.TimeoutExpired:
            rc, err = -99, "TIMEOUT"
    m = measure(out)
    # The #416 signature: rc=0, right length, no energy.
    silent = bool(m and m["n"] > 0 and m["peak"] == 0.0)
    warns = [ln.strip() for ln in err.splitlines() if "WARNING" in ln and "sidon" in ln]
    res = dict(tag=tag, model=model, gpu=gpu, rc=rc, measured=m,
               silent=silent, sidon_warnings=warns,
               tail=(err[-700:] if rc != 0 else ""))
    kh.step(f"run.{tag}", rc=rc, silent=silent,
            rms=(round(m["rms"], 6) if m else None),
            peak=(round(m["peak"], 6) if m else None),
            warns=len(warns))
    return res


def main():
    smi = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
    gpu_name = (smi.stdout or "").strip()
    kh.step("gpu", device=gpu_name)

    binp = build_crispasr()
    make_clip()

    for f in SIDON_FILES:
        with kh.build_heartbeat(f"models.{f}", interval_s=30):
            hf_hub_download(repo_id=SIDON_REPO, filename=f,
                            local_dir=str(MODELS), token=HF_TOKEN or None)
    kh.step("models.ready", files=SIDON_FILES)

    results = {"gpu": gpu_name, "branch": BRANCH, "runs": {}}

    # Paired arms: every model on CUDA and on CPU, same box, back to back.
    # The CPU arm is the control that makes a CUDA silence attributable.
    for f in SIDON_FILES:
        short = f.replace("sidon-v0.1-", "").replace(".gguf", "")
        for gpu in (True, False):
            tag = f"{short}_{'cuda' if gpu else 'cpu'}"
            try:
                results["runs"][tag] = run_sidon(binp, f, tag, gpu)
            except Exception as e:
                results["runs"][tag] = dict(tag=tag, error=repr(e))
            RESULTS.write_text(json.dumps(results, indent=2))

    # Verdict. Reproduced == a quant is silent on CUDA while its own CPU arm on
    # this same box is not, with f16 healthy on CUDA (otherwise the fault is not
    # quantization-specific and the framing is wrong).
    def ok(tag):
        r = results["runs"].get(tag) or {}
        m = r.get("measured") or {}
        return bool(r.get("rc") == 0 and m.get("peak", 0) > 0.01)

    quants = [f.replace("sidon-v0.1-", "").replace(".gguf", "")
              for f in SIDON_FILES if "f16" not in f]
    repro = [q for q in quants
             if not ok(f"{q}_cuda") and ok(f"{q}_cpu") and ok("f16_cuda")]
    results["verdict"] = {
        "f16_cuda_ok": ok("f16_cuda"),
        "f16_cpu_ok": ok("f16_cpu"),
        "quants_silent_on_cuda_only": repro,
        "reproduced": bool(repro),
    }
    RESULTS.write_text(json.dumps(results, indent=2))
    kh.step("verdict", **results["verdict"])
    print(json.dumps(results["verdict"], indent=2), flush=True)


if __name__ == "__main__":
    main()
