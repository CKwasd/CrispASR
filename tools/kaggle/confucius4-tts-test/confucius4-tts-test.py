#!/usr/bin/env python3
"""Kaggle kernel: Confucius4-TTS end-to-end T2S decode test.

Builds CrispASR from main, downloads the Q4_K T2S GGUF, tokenizes a test
string with the HF tokenizer, and runs the T2S decode loop to generate
semantic codes. Validates the full pipeline: model load → text projector
MLP → GPT-2 prefill → autoregressive decode → EOS/max.

Push (under chr1str):
  export KAGGLE_API_TOKEN=KGAT_cb3f25c81b9e65d706ebcf655f1daa42
  python -m kaggle kernels push -p tools/kaggle/confucius4-tts-test
"""

import os
import sys
import subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")

# ── Phase 0: Clone repo ─────────────────────────────────────────────────────
print("=== Phase 0: clone repo ===", flush=True)
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "-b", "main",
        "https://github.com/CrispStrobe/CrispASR", str(REPO),
    ])

# Init ALL submodules
subprocess.check_call(
    ["git", "submodule", "update", "--init", "--recursive"],
    cwd=str(REPO),
)

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh
kh.init_progress()

# ── Phase 1: Install deps ───────────────────────────────────────────────────
kh.step("install deps")
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer tokenizers")

# ── Phase 2: Resolve HF token ───────────────────────────────────────────────
kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# ── Phase 3: Download T2S Q4_K GGUF ─────────────────────────────────────────
kh.step("download T2S GGUF")
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    "cstr/confucius4-tts-GGUF",
    "confucius4-tts-t2s-q4_k.gguf",
    local_dir=str(TEMP / "models"),
    token=hf_token,
)
print(f"  model: {model_path} ({os.path.getsize(model_path) / 1024**2:.0f} MB)")

# ── Phase 4: Download tokenizer ──────────────────────────────────────────────
kh.step("download tokenizer")
tok_path = hf_hub_download(
    "netease-youdao/Confucius4-TTS",
    "tokenizer.json",
    local_dir=str(TEMP / "tokenizer"),
    token=hf_token,
)
print(f"  tokenizer: {tok_path}")

# ── Phase 5: Tokenize test string ────────────────────────────────────────────
kh.step("tokenize")
from tokenizers import Tokenizer

tok = Tokenizer.from_file(tok_path)
# Match the Python inference format exactly:
# formatted = "You are a helpful assistant. {lang_token}:{text}"
# lang_token for English = "Please read the following English text"
test_text = "Hello world, this is a test of the Confucius four text to speech system."
formatted = f"You are a helpful assistant. Please read the following English text:{test_text}"
enc = tok.encode(formatted)
token_ids_str = ",".join(str(x) for x in enc.ids)
print(f"  text: {test_text}")
print(f"  formatted: {formatted[:80]}...")
print(f"  token IDs ({len(enc.ids)}): {token_ids_str[:100]}...")
print(f"  tokens: {enc.tokens[:15]}...")

# ── Phase 6: Build CrispASR ─────────────────────────────────────────────────
kh.step("build CrispASR")
BUILD = TEMP / "build"
BUILD.mkdir(parents=True, exist_ok=True)
flags = kh.cache_and_link_flags()
kh.sh_with_progress(
    f"cmake -G Ninja -B {BUILD} -S {REPO} "
    f"-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF " + " ".join(flags),
)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"cmake --build {BUILD} -j{kh.safe_build_jobs(gpu=False)} --target crispasr-cli"
    )
crispasr_bin = BUILD / "bin" / "crispasr"
print(f"  binary: {crispasr_bin} ({os.path.getsize(str(crispasr_bin)) / 1024**2:.0f} MB)")

# ── Phase 7: Verify model loads ──────────────────────────────────────────────
kh.step("verify model load")
result = subprocess.run(
    [str(crispasr_bin), "--backend", "confucius4-tts",
     "-m", model_path, "--list-backends"],
    capture_output=True, text=True, timeout=30,
)
print(f"  rc={result.returncode}")
for line in result.stderr.split("\n"):
    if "confucius4" in line.lower() or "T2S" in line:
        print(f"  {line}")

# ── Phase 7b: Standalone get_rows test ────────────────────────────────────────
kh.step("get_rows smoke test")
# Write a tiny C program that loads the GGUF, does get_rows on the text embedding, prints result
smoke_code = f'''
#include <stdio.h>
#include <stdlib.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "core/gguf_loader.h"
#include "core/ggml_cpu_backend.h"

int main() {{
    ggml_backend_t backend = core_cpu_backend::init();
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights("{model_path}", backend, "smoke", wl)) {{
        fprintf(stderr, "load failed\\n"); return 1;
    }}
    auto it = wl.tensors.find("text_projector.embed.weight");
    if (it == wl.tensors.end()) {{ fprintf(stderr, "tensor not found\\n"); return 1; }}
    ggml_tensor* embed = it->second;
    fprintf(stderr, "embed: ne=[%lld,%lld] type=%d buffer=%p\\n",
            (long long)embed->ne[0], (long long)embed->ne[1], (int)embed->type, (void*)embed->buffer);

    // Build a tiny graph: get_rows(embed, [1])
    size_t ctx_size = ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(16, false);
    ggml_init_params ip = {{ctx_size, NULL, true}};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16, false);

    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(ids, "ids"); ggml_set_input(ids);
    ggml_tensor* out = ggml_get_rows(ctx0, embed, ids);
    ggml_set_name(out, "out"); ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t g = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(g, gf)) {{ fprintf(stderr, "alloc failed\\n"); return 1; }}

    int32_t id_val = 1;
    ggml_backend_tensor_set(ids, &id_val, 0, sizeof(int32_t));
    fprintf(stderr, "ids: ne=[%lld,%lld] buffer=%p\\n",
            (long long)ids->ne[0], (long long)ids->ne[1], (void*)ids->buffer);

    ggml_backend_graph_compute(backend, gf);
    float v;
    ggml_backend_tensor_get(out, &v, 0, sizeof(float));
    fprintf(stderr, "get_rows OK: out[0]=%.6f\\n", v);

    ggml_gallocr_free(g);
    ggml_free(ctx0);
    ggml_backend_free(backend);
    return 0;
}}
'''
smoke_path = TEMP / "smoke_getrows.cpp"
with open(smoke_path, "w") as f:
    f.write(smoke_code)

# Compile and run
BUILD_bin = BUILD / "bin"
kh.sh_with_progress(
    f"cd {REPO} && /usr/bin/c++ -O2 -std=c++17 -I src -I ggml/include -I ggml/src "
    f"-o {TEMP}/smoke_getrows {smoke_path} "
    f"-L {BUILD}/src -L {BUILD}/ggml/src "
    f"-lcrispasr-core -lggml -lggml-base -lggml-cpu "
    f"-Wl,-rpath,{BUILD}/ggml/src "
    f"-lm -lpthread -ldl -fopenmp 2>&1 || echo 'COMPILE FAILED'"
)
if (TEMP / "smoke_getrows").exists():
    r = subprocess.run([str(TEMP / "smoke_getrows")], capture_output=True, text=True, timeout=30,
                       env={**os.environ, "LD_LIBRARY_PATH": f"{BUILD}/ggml/src"})
    print(f"  smoke rc={r.returncode}")
    print(f"  smoke stderr: {r.stderr.strip()}")
else:
    print("  smoke test compile failed, skipping")

# ── Phase 8: Run T2S decode ──────────────────────────────────────────────────
kh.step("T2S decode")
env = os.environ.copy()
env["CRISPASR_CONFUCIUS4_TEXT_IDS"] = token_ids_str
env["CRISPASR_CONFUCIUS4_GALLOCR"] = "1"  # use gallocr instead of sched for GPT-2 step

result = subprocess.run(
    [str(crispasr_bin), "--backend", "confucius4-tts",
     "-m", model_path, "--tts", test_text,
     "-v"],
    capture_output=True, text=True, timeout=120, env=env,
)
print(f"  rc={result.returncode}")
print("--- stderr ---")
print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print("--- stdout ---")
print(result.stdout[-500:] if result.stdout else "(empty)")

# ── Phase 9: Summary ─────────────────────────────────────────────────────────
kh.step("summary")
# Check what happened
if "prefill done" in result.stderr:
    print("  PREFILL: OK")
else:
    print("  PREFILL: not reached")

if "EOS at step" in result.stderr:
    import re
    m = re.search(r"EOS at step (\d+)", result.stderr)
    print(f"  EOS: step {m.group(1)}")
elif "generated" in result.stderr and "semantic codes" in result.stderr:
    m = re.search(r"generated (\d+) semantic codes", result.stderr)
    if m:
        print(f"  CODES: {m.group(1)} semantic codes generated")
else:
    print("  DECODE: did not produce codes")

# Extract any error messages
for line in result.stderr.split("\n"):
    if "error" in line.lower() or "failed" in line.lower() or "FAIL" in line:
        print(f"  ERR: {line.strip()}")

print("\n=== Done ===", flush=True)
