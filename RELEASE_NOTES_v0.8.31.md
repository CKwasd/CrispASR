# CrispASR v0.8.31

This release broadens small, practical TTS and makes several accelerated paths
safer on the machines where they previously failed: Pocket-TTS now speaks all
six upstream languages, Chatterbox Nano gains a proven Finnish checkpoint, and
Windows joins Linux with a native CUDA 13 package. It also fixes GPU-specific
failures in source separation, Chatterbox, Qwen3-TTS, and old-CPU CUDA packages.

The other theme is responsiveness. Streaming partials avoid repeated audio
slicing, CosyVoice3 uses the proven packed Conv1d path by default, Chatterbox's
T3 decoder stops materializing redundant KV-cache copies, and servers expose
live transcription progress.

---

## New and expanded models

### Pocket-TTS speaks German, Spanish, Italian, Portuguese, and French (#411)

The complete current Kyutai language set is now in the registry:

| Language | Backend selected by `-l` | Default GGUF |
|---|---|---|
| English | `pocket-tts` | English F16, 219 MB |
| German | `pocket-tts-de` | distilled 6L Q8_0, 124 MB |
| Spanish | `pocket-tts-es` | distilled 6L Q8_0, 124 MB |
| Italian | `pocket-tts-it` | distilled 6L Q8_0, 124 MB |
| Portuguese | `pocket-tts-pt` | distilled 6L Q8_0, 124 MB |
| French | `pocket-tts-fr` | undistilled 24L preview Q8_0, 365 MB |

`--backend pocket-tts -m auto -l de|es|it|pt|fr` routes to the correct model
before auto-download. The explicit backend names work through both the CLI and
C session ABI. Every model includes the Mimi encoder and speaker projection for
reference-WAV voice cloning; F16 variants are available in the same model
repository.

The converter now downloads one language snapshot at a time instead of every
checkpoint in the gated upstream repository. It also records the French
preview's 24 layers even though Kyutai's weights repository omits the source
YAML—a packaging detail that would otherwise have silently run only six layers.

### Finnish Chatterbox Nano (#382)

`chatterbox-finnish-nano` registers JJarvinen's v0.1.3 T3 checkpoint with the
shared Turbo S3Gen companion. `--backend chatterbox -m auto -l fi` selects it,
and the variant is reachable through the C ABI as well as the CLI. The Finnish
checkpoint is monolingual, so CrispASR does not inject a nonexistent `[fi]`
multilingual control token.

Hosted CPU proof downloaded both public GGUFs and synthesized a valid 4.84 s,
24 kHz WAV after the full registry and backend-wiring suites passed.

## Performance and streaming

### Chatterbox direct KV-cache views (#410)

Chatterbox T3 attention no longer calls `ggml_cont()` on each K/V layer view.
The selected layer is already logically contiguous and differs only by its base
offset, so materializing it copied the growing cache once per layer and decode
step for no semantic benefit.

The merge was audited beyond its original benchmark: a graph-level test covers
nonzero-offset cache layers, and a hosted model A/B produced the same 40-token
trajectory and byte-identical decoded PCM through the direct and old
materialized paths. Set `CRISPASR_CHATTERBOX_KV_CONT=1` to retain the old path
for diagnostics.

### Exact streaming work reductions (#404)

The streaming orchestrator memoizes repeated audio slices. This is enabled by
default after quiet-box A/Bs reduced decoded audio and wall time without
changing any final or partial transcript; set
`CRISPASR_STREAM_SLICE_MEMO=0` to revert.

`--stream-partial-tail-sec N` is a new opt-in for long open utterances. It
commits a stable text prefix and decodes only a bounded recent tail for live
partials, while the default final redecode remains byte-exact. It is opt-in
because partial joins can have cosmetic punctuation/capitalization seams even
though finals are unchanged.

### Packed Conv1d and CosyVoice3 default (#406)

A generic packed CPU Conv1d implementation now serves Chatterbox F0/S3Gen and
CosyVoice3 HiFT, with scalar, NEON, AVX2, and AVX-512F dispatch. CosyVoice3's
path is enabled by default after exact roundtrips and wins on both tested CPUs
(1.07× on the hosted Xeon, 1.34× on Zen 4); use
`CRISPASR_COSYVOICE3_SIMDCONV=0` to revert. Chatterbox remains opt-in with
`CRISPASR_S3GEN_SIMDCONV=1`: it won on Zen 4 but regressed 5.4% on the hosted
Xeon.

## GPU and package reliability

### CUDA packages work on pre-AVX2 hosts (#405)

CUDA packages previously carried one `libggml-cpu.so` compiled for the GitHub
runner's AVX2-class CPU. On an older Xeon ggml rejected that module, registered
CUDA alone, and then aborted when model setup assumed a CPU backend existed.

Linux CUDA release packages now include ggml's CPU variant set, including the
portable x64 baseline, and select the best supported module at runtime. Model
loading, Whisper LID, the CLI, and the server also fail cleanly with a useful
diagnosis if no CPU module can load instead of reaching a ggml assertion.
Native and QEMU-Nehalem tests transcribe successfully; a no-module arm proves
the clean failure; and the CUDA-shaped path passed on a Kaggle P100.

### Native Windows CUDA 13 package (#400)

The release matrix now produces `windows-x64-cuda13` alongside the existing
CUDA package. It targets sm_75+ (CUDA 13 no longer supports
Maxwell/Pascal/Volta), bundles the split CUDA 13 runtime, cuBLAS, NVVM, CRT, and
compiler components, and publishes the same non-CUDA archive, bare DLLs, and
checksums as the other Windows GPU flavor. CI checks the resulting binary
imports `cublas64_13.dll`, not an older runtime by accident.

### HTDemucs GPU broadcast crash (#398)

HTDemucs fed F16 one-dimensional GroupNorm weights into F32 broadcast
multiply/add nodes. ggml's broadcast operators require matching F32 operands:
CUDA aborted on the stride and the CPU graph path rejected the mixed types.
Weights are now cast to F32 in the graph, and future conversions retain all
one-dimensional tensors as F32. P100 proof produced all four stems for F16 and
Q8_0 with CPU-reference cosine 0.9997–1.000000 and unit magnitude ratio.

### Chatterbox Turbo/Nano on Vulkan (#402)

The T3 GPT-2 side defaults to explicit softmax attention on Vulkan, avoiding a
RADV `FLASH_ATTN_EXT` pipeline crash reported on Radeon 780M. Other backends
retain flash attention; Vulkan users can retest newer drivers with
`CRISPASR_CHATTERBOX_FLASH_ATTN=1`.

### Qwen3-TTS on ROCm: safe defaults, native kernels still open (#337)

Two gfx1100 failures are contained by default:

- reference-audio codec encoding routes to CPU instead of using the
  content-dependent divergent HIP path;
- the affected 0.6B-F16 code predictor routes to CPU instead of producing
  all-NaN logits;
- every code-predictor step rejects non-finite logits rather than sampling a
  runaway until the context ceiling.

The normal CPU/CUDA paths remain accelerated. Native HIP experiments are
available through `CRISPASR_QWEN3_TTS_HIP_CODEC_NATIVE=1` and
`CRISPASR_QWEN3_TTS_HIP_CP_NATIVE=1`, but are intentionally not the default.
This fixes the user-visible failure safely; it does **not** claim that the two
native gfx1100 kernels are repaired. Issue #337 remains open pending a real RX
7900-class runner/reporter retest; the repository now includes a complete
public Daphne-reference workflow ready for that machine.

## Recognition, language ID, and correctness

### Silero LID no longer trusts collapsed logits (#409)

The Silero language classifier can return a confident-looking softmax winner
when all raw logits collapse on difficult or codec-damaged audio. CrispASR now
applies a raw-evidence floor (default `-2.0`, configurable with
`CRISPASR_SILERO_LID_MIN_LOGIT`) and falls back to Whisper-tiny LID when Silero
is inconclusive. Confidence returned by Silero is now a probability rather than
an unnormalized logit.

### Smaller correctness and robustness fixes

- Thirteen backend initialization-error paths now release their ggml backends;
  the final FireRed and FastConformer cache leaks are fixed as well.
- `crispasr-diff` and ten unit-test sources no longer use POSIX-only
  `setenv`/`unsetenv` APIs on Windows.
- The MOSS diarization reference dump uses the live environment variable name.
- Companion-model tests assert an isolated cache contract instead of depending
  on what a developer previously downloaded.
- The feature-matrix generator reports its output paths clearly and emits a
  complete capability legend.

## Server, CLI, and bindings

### Live server progress (#408)

`GET /progress` reports active transcription progress. It is auth-gated like
the other operational routes, remains correct with multiple server workers,
does not let a queued request reset the running job, and stays at 100% through
post-processing until the request completes.

### TTS output padding

`--tts-pad-silence-ms` adds a requested silence tail to generated audio. This
works around players such as VLC dropping the final buffered audio when C2PA
metadata is appended immediately after PCM.

### Go diarization parity (#395)

The Go binding now forwards FoxNose speaker turns instead of discarding the
native session's diarization labels.

## Build, diagnostics, and documentation

- A portable legacy x86_64 CPU build is covered in CI.
- Windows illegal-instruction troubleshooting was verified on a real Windows
  runner with a captured dump, including the attach timing needed by ProcDump.
- Release packages remove maintainer-machine paths from shipped code.
- Top-level documentation was audited against current code, and installation
  guidance now explains CPU/GPU package flavors and `mise` defaults.

---

**Upgrading.** Replace CUDA archives rather than copying only the executable:
the new packages rely on their accompanying runtime-selected CPU modules. If
you use Pocket-TTS, `-l de|es|it|pt|fr` is sufficient to choose the new model;
existing English commands are unchanged. ROCm Qwen3-TTS users should leave the
new safe defaults enabled unless collecting an explicit native-kernel A/B for
#337.
