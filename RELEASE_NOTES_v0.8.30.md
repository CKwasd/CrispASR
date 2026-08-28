# CrispASR v0.8.30

235 commits. Two themes, both about things that were wrong before anyone got to
judge the model:

* **Audio was degraded on the way in.** Every input whose sample rate did not
  already match the backend was resampled by linear interpolation. A 10 kHz tone
  that must vanish entirely when decoded to 16 kHz survived at −10.3 dB and
  folded down into the speech band. That is the front door for every backend.
* **Binaries could not start on the machines that downloaded them.** The
  official Windows CUDA build of v0.8.29 died with `SIGILL` on any CPU without
  AVX-512, and the fault is one the Windows console swallows — banner, then
  nothing.

**If you feed CrispASR anything other than 16 kHz audio, or you downloaded a GPU
build of v0.8.29, this release changes results you have been living with.**

Also new: a complete Confucius4-TTS backend, true realtime Nemotron sessions,
and source separation over HTTP.

---

## Audio input

### Every non-16 kHz input was aliased (`ac4aa478`)

`read_audio_data()` asked `ma_decoder` for the target rate, so miniaudio did the
conversion with `ma_resample_algorithm_linear` — linear interpolation behind a
4th-order low-pass. Measured on that path:

| | linear (was) | polyphase (now) | soxr_vhq (reference) |
|---|---|---|---|
| 10 kHz tone → 16 kHz (must vanish) | **−10.3 dB, folds to 6 kHz** | −89 dB | −165 dB |
| 48 → 16 kHz log sweep, cos vs soxr | **0.702** | 0.998 | — |
| 16 → 24 kHz, cos vs soxr | **0.944**, 2-sample group delay | 0.99997, no delay | — |

So every 44.1/48 kHz recording anyone has fed to a 16 kHz backend carried
audible alias *inside the speech band*. Files are now decoded at their own rate
and resampled with `core_audio::resample_polyphase` — the Kaiser-windowed sinc
that was already in the tree and unit-tested, but wired only into chatterbox's
voice-clone path. `CRISPASR_HQ_RESAMPLE=0` restores the old behaviour.

The blast radius is narrower than it looks: resampling only happens when the
file's rate differs from the backend's, and every fixture in the repo is 16 kHz
— which is exactly why this survived a regression corpus for so long. It lands
on real-world input, which is where it was hurting.

This also closed the long-running "our sigma-VAE conditioning diverges from
upstream" gap in #369: feeding both sides identical 24 kHz bytes puts every
encoder stage at cos 0.9999. The divergence was in what we handed the encoder,
not in the encoder.

## Builds that could not run

### The official Windows CUDA build died on any non-AVX-512 CPU (#374)

`build-windows-cuda` set `-DGGML_CUDA=ON` and nothing else, so `GGML_NATIVE`
stayed ON and ggml compiled with `-march=native` — against a GitHub runner that
*has* AVX-512. A reporter's RTX 4060 laptop could not use its GPU at all: the
CUDA build crashed on startup, and the portable AVX2 build they fell back to is
CPU-only.

**Nine of eleven GPU jobs were in that state** — every Linux CUDA/HIP/Vulkan one
too, not only Windows. Fixed, and guarded so it cannot come back.

### `--no-gpu` on a pre-AVX2 CPU: banner, then nothing (#380)

The AVX2+FMA Windows CPU zip on a pre-AVX2 machine printed the banner and then
died at the first ggml compute op with an illegal-instruction fault the Windows
console swallows — no output, no error, no exit message. The CLI now checks the
build's ISA against the host's and fails fast with a message naming the zip you
should have downloaded instead.

### CUDA tarballs fall back to CPU when the NVIDIA libraries are absent (#355)

Previously a CUDA tarball on a machine without the driver libraries simply
failed to load.

### Windows: cached GGUFs larger than 2 GiB were re-downloaded every run (#393)

On MSVC `stat` resolves to `_stat64i32`, whose `st_size` is a 32-bit field, and
the call *fails outright* above 2 GiB. Every GGUF worth caching is bigger than
that, so the cache probe reported "missing" for a model sitting right there and
`-m auto` fetched it again. The same helper validates a finished download, so a
>2 GiB fetch could also be judged failed after it had succeeded.

## Transcription correctness

### VibeVoice-ASR answered in the wrong language (#369)

Several fixes converge here. The input was never normalised to −25 dBFS before
the VAE encoders, as upstream's `audio.cpp` does; the 1.5B model was being sent
the 7B's JSON prompt when its own runtime uses plain text; the prompt told the
model to do something other than transcribe; the sigma-VAE used SiLU where
upstream uses GELU; and the ASR attention needed pinning to `GGML_PREC_F32`.
Two of three Korean fixtures now come back character-exact.

Related: the model's own `[Silence]` marker was being written into transcripts
and SRT files, where it also silently disabled the empty-transcript warning.
That is model output, not transcript.

### Punctuation restoration returned nothing at all (`2af349dd`)

Every `fireredpunc`-architecture model with `tokenizer_type == "sentencepiece"`
— `fullstop-punc`, `punctuate-all` — produced an **empty string** on the default
path. Not degraded: nothing. `tokenize_ex` returns early for SentencePiece and
never fills the word-alignment arrays, but the caller branched on the HF
tokenizer alone and then ran a loop bounded by a zero-length vector.
`CRISPEMBED_FIREREDPUNC_HF_TOK=0` worked, which is why it went unnoticed.
`punctuate-all-f16` goes from 4 empty lines to output byte-identical to the
legacy arm.

### Canary long-form: seam artifacts gone (#375)

Ported canary-1b-v2's real dynamic chunking instead of the parakeet-shaped
machinery it had been given. Chunk-boundary duplication also gets an opt-in
seam dedup (#365).

### Smaller correctness fixes

* `#388` — the pause between utterances is kept (backport of whisper.cpp#2279).
* `#385` — progress is reported on the chunk-encoded single-decode routes, the
  JA sliced path, and the unified long-form dispatch, not just the common one.
* `#367` — quantized KV caches were sized with `ggml_nbytes()` instead of
  through ggml.
* `#372` — f5-tts counted bytes, not UTF-8 characters, in its duration estimate.
* `#366` — kyutai `stt-1b-en_fr` no longer claims to be English-only.
* `#363` — omnivoice's target-token arithmetic and duration weights now match
  upstream exactly, on all 1.1M codepoints.
* `#371` — a per-request voice reference is applied at synth time.
* `#368` — the minimum speaker count is clamped to the distinct pyannote tracks.

## New

### Confucius4-TTS (#377)

A complete port: GPT-2 T2S with a faithful `LlamaTokenizer` SP-BPE, beam-sample
decode matching transformers 4.52.4, native ECAPA and CAMPPlus speaker
conditioning, an S2A DiT flow-matching estimator with WaveNet, and the BigVGAN
vocoder. `--voice` does native voice cloning. Metal gets a fused-CFG path
(−12.7% on q4_k).

### True realtime Nemotron sessions (#383)

`/v1/realtime` previously called the backend every 500 ms with the *entire*
accumulated PCM, so each call recomputed the frontend and reset encoder and
RNN-T state — work grew with the session and CPU transcription fell
progressively behind. The stream session now owns frontend progress, per-layer
attention and convolution caches, and predictor state across appends, with a
proper commit-and-reset. Server VAD runs before ASR with explicit
speech-start/stopped events.

### Source separation over HTTP (#381)

`POST /v1/audio/separation`.

### Diarization: FoxNose speaker turns across the C ABI (#395)

## Bindings

### The C# binding could not coexist with its own native library (#291)

The managed assembly was `CrispASR.dll` and the native library is
`crispasr.dll` — the same file name on a case-insensitive filesystem. On Windows
the two could not be installed in one directory, and bare-name P/Invoke probing
handed the managed assembly to the native loader. **The assembly is now
`CrispASR.Net.dll`**; the namespace and every type name are unchanged, so no
consumer source changes. A resolver loads the library by explicit full path
(`CRISPASR_LIBRARY_PATH`, then `runtimes/<rid>/native/`, `native/`, `bin/`,
`lib/`, then the app base), refuses to hand a managed assembly to the native
loader, and reports every path it probed when it fails.

Two more things that had made the binding unusable in practice:

* **There was no way to read an audio file.** Every ASR entry point takes
  `float[]` PCM and .NET has no decoder in its standard library, so
  "works with `crispasr.exe`, not with `libcrispasr`" was the predictable
  result. `Audio.Load` and `Session.TranscribeFile` decode with the same
  decoder the CLI uses.
* **`Segment` / `Word` / `AlignedWord` reported centiseconds** while `VadSpan`
  and the music types in the same file reported seconds. **Breaking:** they are
  `double` seconds now. A backend with no timing for a unit reports `-1`, and
  that sentinel is passed through unscaled.

The C# suite also reported *193 passed, 0 skipped* on a machine with no native
library at all — every live test began `if (!CanLoadLibrary()) return;`.
`CRISPASR_CS_REQUIRE_LIVE=1` makes that a failure, CI now drives a real model
end to end, and a windows-latest job installs the native DLL beside the managed
assembly.

### Other binding and build fixes

* `#384` — bare voice names resolve against `--voice-dir` in five TTS adapters,
  over HTTP as well as the CLI.
* `#50` — `crisp_audio`'s GPU link assumed CrispASR's own CMake targets.
* `#317` — `--align-only` accepts JSON input, for a JSON→JSON pipeline.

## Documentation (#397)

A beginner on Windows could not get any command to work and ended up copying
commands out of a GPU-numerics bug report, because that was the most runnable
material in the repo. README now leads with Start-here — which file to download
per platform, then two commands to working audio — ahead of the backend
catalogue, and `docs/troubleshooting.md` leads with the "banner, then nothing"
symptom that #380 explains.

---

**Upgrading.** If you use a GPU build, replace it — the v0.8.29 GPU artifacts
are `-march=native` and may not run on your CPU. If you consume the C# binding,
note the assembly rename and the seconds change above. If you have scripted
around the linear resampler by pre-converting your audio to 16 kHz, you no
longer need to.
