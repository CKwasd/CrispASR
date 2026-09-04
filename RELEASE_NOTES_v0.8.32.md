# CrispASR v0.8.32

Five new backends. Raon-OpenTTS and Quds bring Korean-trained F5 speech
synthesis and Persian recognition; MT3 and Basic Pitch turn the `--piano`
surface into general music transcription with Standard MIDI output; and
mel-band-roformer joins HTDemucs as a GPU source separator, contributed by
tilllt with a fused single graph that runs a 359 s track at RTF 0.076.

The second theme is that accelerated paths now *default* to being fast.
HTDemucs picks the fused GPU graph on any host that has a real GPU, FunASR
caches its decode and encoder graphs, piano transcription and the Bananamind
vocoder use every core, and Sidon chooses its relative-position formulation
from the actual sequence length. Every one of those is bit-identical to the
path it replaces, or says plainly where it is not.

The third is language handling. Nemotron's `-l auto` was silently conditioning
on the English prompt no matter what language ID detected; subtitle output was
returning transliterated text as the transcript; three backends accepted
languages they cannot recognise and answered anyway. Those are fixed, and a
malformed audio file can no longer make the decoder allocate 11 GB.

---

## New backends

### Raon-OpenTTS (#387)

`--backend raon` synthesises speech through the F5-family DiT and a HiFi-GAN
vocoder, from a single self-contained GGUF (`cstr/raon-opentts-0.3b-GGUF`)
carrying the DiT, the vocoder, the mel filterbank and window, and the vocab.
The TTS→ASR roundtrip gate passes at 0.90 word overlap; the vocoder was
validated in isolation at cosine 0.9979 against the reference.

The vocoder runs as a ggml graph on the backend scheduler — the same
`core_hifigan` path FastPitch, SpeechT5 and Bananamind use — rather than the
naive CPU loops it started on: cosine 1.000000 against the CPU reference and
19.6x faster, with `CRISPASR_F5_HIFIGAN_CPU=1` retained as the A/B fallback.

Two F5-family fixes landed with it and apply to **every** f5 backend,
including F5-TTS:

- **Flash attention was corrupting the DiT.** `ggml_flash_attn_ext` accumulates
  the KQ product in F16, and on kernels that ignore `GGML_PREC_F32` — P100 /
  sm_60 confirmed — the precision hint is silently dropped. Measured against a
  torch-f16 mirror oracle, C++ drifted ~16x more than pure weight-f16 rounding
  on the 0.3B and ~85x on the 1B, compounding to NaN. **Flash attention is now
  opt-in on the f5 DiT**: the default is a manual F32 SDPA path that no
  precision hint can silently drop, and it collapses the drift onto the oracle
  exactly. `CRISPASR_F5_FLASH=1` restores the fused kernel where the hint is
  honoured (`CRISPASR_F5_NO_FLASH=1` still forces manual, for back-compat).
  Adding the precision hint by itself changed nothing, byte-for-byte, which is
  how the kernel was identified as the cause — and why the default flipped
  rather than a hardware allowlist being added.
- **One-word `--tts` was rushed or truncated** on every f5-family backend.
  Upstream drops to `local_speed = 0.3` for generated text under 10 bytes, and
  that is now ported. It *replaces* rather than multiplies the user's
  `--tts-speed`, matching upstream. `CRISPASR_F5_SHORT_PROMPT_SPEED=0` disables
  it. Upstream's 12 chars/s VAD floor is deliberately **not** ported: applying
  it literally would reintroduce #294, and the release notes for the fix say so
  rather than leaving it to be rediscovered.

### Quds — Persian FastConformer-RNNT (#387)

`--backend quds` (alias `quds-fa`) recognises Persian from a 122 MB Q8_0
default. The author publishes only a NeMo ONNX export, so the release adds a
generic NeMo FastConformer-RNNT ONNX→GGUF converter: anonymous initializers
recovered by tracing consumer-node scopes, ONNX LSTM gate order remapped to
torch, and the mel filterbank and window **copied** from the base `.nemo`
buffers rather than recomputed. Single-LSTM predictors with no CTC head now
decode on both the CPU and ggml RNNT paths.

On five real Common Voice fa clips, Q8_0 and F16 are byte-identical to the
upstream ONNX under the model card's own runner on four; on the fifth CrispASR
keeps an onset word the reference drops. Q4_K shows minor word drift and the
model card says so.

### MT3 — multi-instrument music transcription (#250)

`--piano --backend mt3` (alias `music-transcription`) transcribes polyphonic
multi-instrument audio to note events. Encoder mel cosine 1.000000000, encoder
output 0.999999879, greedy tokens identical for 22/22 steps, and 88/88
positional note match against the reference at its own step budget. The tie
state machine is pinned independently by feeding CrispASR's own token stream
through the reference implementation, and four deliberately doctored GGUFs are
all refused at load so the runtime cannot silently fall into the T5 translate
path.

### Basic Pitch — polyphonic note events (#250)

`--piano --backend basic-pitch` ports Spotify's model: nnAudio CQT2010v2 front
end, harmonic stacking, three convolutional heads, window stitching, and the
upstream note-creation post-processing. All ten pipeline stages are at cosine
≥ 0.9991 against upstream ONNX on resampled audio and 1.000000 at native rate;
end-to-end note events are **exact** at F32 (27/27 synthetic, 11/11 jfk). The
CQT kernels, decimation FIR and rescale vector are copied bit-for-bit from the
ONNX initializers, so there is no reimplementation that can drift.

### MIDI output

`--piano-format midi` writes note events as a Standard MIDI File, through a new
core SMF writer shared by all three note-event backends.

## Source separation

### mel-band-roformer on the GPU (#422, contributed by tilllt)

A cleanroom ggml port: band-split and the RoFormer blocks as graphs, then a
**fused single graph** running band-split, the full 12-block time/frequency
stack and the mask estimator on-device in one pass, removing 13 host↔device
roundtrips per segment. Demucs-style segmentation with 25% overlap handles long
audio. Measured by the contributor on an RTX 3090 Ti: a 30 s clip goes from
28 minutes on CPU to 4:47 on per-layer GPU graphs to **2.57 s** fused; RTF
0.076 on a 359 s track.

Two things were corrected on the way in. The segment length defaulted to a
hardcoded 10 s while the Kim vocals checkpoint declares an 8.0 s trained chunk,
so the time transformer's rotary embedding ran to 1000 positions against the
800 it was trained on, on every segment of long audio — the converter now
carries `chunk_size` into GGUF metadata and the default derives from it. And
that default is applied in **samples**, not round-tripped through integer
seconds, which is exact for Kim but would have silently dropped 0.8 s of
trained context from any checkpoint whose chunk is not a whole number of
seconds.

The overlap-add machinery was verified independently across ~200
length/segment combinations including every boundary case: no coverage holes,
reconstruction to 2.4e-07. A separate within-arm boundary check confirms the
crossfade is healthy on the shipped default, measuring the +3.01 dB that
averaging two independent estimates must produce — a *positive* signature,
where the absence of a difference would have been the suspicious result.

### HTDemucs uses the GPU by default (#413, #414)

`src/htdemucs_gates.h` resolves the three coupled decisions — graph path, fused
graph, GPU placement — once per init, and AUTO now selects graph + fused + GPU
whenever a real GPU backend is present and permitted, BLAS otherwise. CPU hosts
are unchanged. The contributor measured fused GPU at RTF 0.37 against CPU/BLAS
7.4. `CRISPASR_HTDEMUCS_GPU=0` opts out, and it works in both directions now:
the previous precedence made the advertised opt-out dead, because the CLI
always passes `use_gpu = true`. The `--separate` CLI also forwards `use_gpu`,
`n_threads` and `gpu_device` to the separation backends, which it was dropping.

The nine-case decision table is locked by unit tests, and CPU-host AUTO,
forced-fused engagement and BLAS-vs-fused per-stem parity are all covered.

## Recognition and language handling

### Nemotron `-l auto` was always English (#425, contributed by jltjarvinen)

`nemotron_set_language()` was called exactly once, at load, with whatever
`params.language` held — which for `-l auto` is the literal string "auto",
resolving to the English prompt. The dispatcher's language ID then detected the
real language and nothing re-applied it, so **every** automatic-language
Nemotron run was conditioned on the English prompt regardless. On CUDA, `-l
auto` on Russian audio ignored a p = 0.993 detection and used en-US. The
resolved language is now re-applied per entry point, and the explicit `-l xx`
path is byte-identical.

Built on top of the contribution: the language table is read from the **model**
rather than a literal in our source. The GGUF carries the prompt dictionary the
checkpoint was trained with, and it lists 121 names where the hardcoded table
reached 75 — so **51 languages were unreachable by name**, among them Persian,
Afrikaans, Bengali, Gujarati and Hawaiian, each of which previously reported
"unknown language, defaulting to en-US". `auto` now takes its id from the model
instead of a magic number, and the literal table is demoted to a fallback for
older GGUFs and five short aliases the file omits.

### Subtitles no longer show transliterated text (#419)

Any output wanting word timestamps (`-sp`, `-sow`, SRT, VTT) auto-enables the
Canary CTC aligner, and the aligner romanized the whole transcript — needed,
because its vocabulary is Latin — then returned the **romanized** strings as
the aligned words. Russian subtitles came out as `vikingi, otvazhnye voyny`
instead of Cyrillic, and Japanese and Chinese as romaji and pinyin.
Romanization is now the aligner's internal label only: the original words are
tokenised, romanized 1:1 for alignment, and mapped back onto the resulting
timings. The GGUF vocabulary, prompt construction, quantization and GPU path
were each cleared with evidence before the fix, not assumed innocent.

### Canary diagnoses wrong-language conditioning (#419)

A separate transliteration report could not be reproduced on any axis — CPU,
CUDA, Vulkan, single-pass, streamed, at the reporter's exact flags. The one
thing that reproduces it is conditioning on the wrong language, which produced
exactly the reported output *silently*. Canary now prints its effective source
and target language, so a frontend dropping `-l` becomes visible in any log,
and warns when a Cyrillic or Greek target produces Latin-dominated text. The
script census requires at least 20 letters of evidence, so code-switched
fragments never trip it.

### Backends no longer answer in languages they do not know

- **MOSS transcribe is English-only.** Chinese and German audio came back as
  rough English *translations* at a success return code. `-l zh` is now an
  explicit rejection and `-l auto` skips the pointless language-ID download.
- The C session ABI and the server now validate a requested language against a
  monolingual backend before dispatch, closing the last two surfaces after the
  CLI guard. The server answers 400 rather than transcribing anyway.
- That CLI guard had regressed the German fine-tunes — `--backend moonshine-de
  -l de` was rejected as English-only, because the shared adapter hardcodes its
  sole language. Variant-aware construction fixes it, and `*-de` model files
  are resolved to the German variant by filename.
- **Silero language ID refuses quantized weights** instead of misdetecting.
  Quantizing this classifier changes its answer, which reads as a bad model
  rather than a bad file.

### Audio decode can no longer be amplified into an out-of-memory

Found by the seeded fuzzer in CI. A structurally valid 344 KB WAV declaring
`sampleRate = 1` resamples 16000x to the 16 kHz target — 2.816e9 frames,
11.3 GB — and the chunked decode loops grew geometrically with no ceiling. It
reaches every surface that accepts a user-supplied file, server upload
included. Decoded frames are now bounded against **input size**, because the
defect is amplification rather than length: 256 output frames per input byte,
roughly 9.5x more than the most extreme ratio any real encoder produces, plus a
24-hour absolute backstop. `CRISPASR_MAX_DECODED_FRAMES` replaces the ceiling
for anyone who genuinely needs more.

### Live and streaming WebM decoded only the first 0.1 s (#417)

Chrome's `MediaRecorder` writes to a non-seekable stream, so it emits the
Segment and every Cluster with the EBML *unknown size* marker and starts a new
Cluster per timeslice. The demuxer read that marker as a literal length,
bounded the first Cluster there, and jumped to the end — surviving only the
first timeslice, and reporting success, so the ffmpeg fallback never ran. The
marker is length-dependent (the one-byte form `0xFF` decodes to 127), so it is
now reported out-of-band from the decoded value.

### VibeVoice on Vulkan (#418)

The ASR tokenizer encoders route to CPU on Vulkan devices with low workgroup
limits, and BitNet TQ weights route to CPU under Vulkan, rather than failing.

## Performance

All of the following are bit-identical to the path they replace unless stated.

- **FunASR graph caches.** The decode step graph is cached in buckets of 16 KV
  positions with a four-graph FIFO, padded slots masked to −inf so flash
  attention never sees them; the encoder reuses graphs for exact frame counts.
  18/18 arms byte-identical across three clips, two quantizations and three
  binaries. The design that was on file — fixing the KV length at the maximum
  context — measured **+69% decode CPU** and was dropped: every wasted key
  costs more in the attention than the graph build it saves.
- **Piano transcription threads its convolutions, GRU and linear layers**
  (#305) over disjoint output chunks, leaving inner accumulation order
  untouched: 1.7x, sha1-identical JSON. `CRISPASR_PIANO_SERIAL=1` opts out.
- **The Bananamind vocoder** (#305) was the one HiFi-GAN-family vocoder still
  on the legacy transpose path and pinned to CPU. It now has pre-permuted
  transposed-convolution weights, the fast-convolution bake, and all three
  graphs on a GPU-capable scheduler. PCM cosine 1.000000000.
- **Sidon picks its relative-position formulation from the sequence length**
  (#416). The `expand` formulation is 2.7x faster in the predictor at T = 557
  but materialises a table costing an extra 196·T² bytes — 58 MiB there, but
  1.64 GiB at the 3000-frame cap. It is now selected automatically within a
  memory budget, on GPU backends, and re-resolved per graph build.
- **Nemotron GPU fast paths** (#424, contributed by jltjarvinen) — a one-shot
  GPU streaming encoder keeping attention and convolution caches on-device,
  cached chunk graphs, direct convolutions, a batched joint projection and a
  GPU prompt MLP. **Merged with everything default-off.** The CPU arm, the GPU
  one-shot arms and both cross-utterance leak shapes are byte-identical to the
  merge base; the chunked GPU arm is not, producing deterministic single-token
  doublings. RNNT emits per frame with no dedup, so one borderline logit flip
  re-emits a token — a numerics difference, not a cache replaying audio. That
  arm bundles four independent changes and so cannot attribute the cause, so
  each is an opt-in gate, ordered for bisection:
  `CRISPASR_NEMOTRON_GPU_JOINT`, `_DIRECT_CONV`, `_STREAM_CACHE`, `_PROMPT`, or
  `CRISPASR_NEMOTRON_GPU_FASTPATH=1` for all four. Gates are read per call, so
  an A/B cannot silently compare a path against itself.

### Quantized-weight broadcasting matmuls (#416)

A quantized weight as the left operand of a broadcasting matrix multiply is
routed through Vulkan's MMQ path on integer-dot GPUs, where it produces silence
on the reporting user's hardware. Two independent static searches each found
sites the other missed, so this ships a **runtime** detector instead —
`CRISPASR_AUDIT_QUANT_BCAST=1` walks the node list at compute time and prints
the real shapes. Validated in both directions: eight sites with the fix off,
zero with it on. Sidon's own case is fixed, and the beat-this and CosyVoice3
sites are folded by collapsing the batch dimension into the token dimension —
an exact restatement for a per-token-independent projection, and unlike the
Sidon fix it does not dequantize. The CosyVoice3 fold is bit-identical.

Sidon also now reports a degenerate decode instead of returning empty output
at a success code.

## Server, CLI and bindings

- **Kokoro honours `--tts-speed`** (#423, contributed by Nixes). The backend
  never read it, and the server applied its own linear resampler
  unconditionally — so wiring the backend without gating the resampler would
  have compounded them into 4x speed and an octave of pitch shift. Both
  resampler sites are now gated on a new `tts-speed` capability. Measured
  rather than assumed: fundamental frequency stays flat (ratio 1.0000 at speed
  2.0) while duration scales, which is native scaling and also proves the
  resampler is not double-applying.
- `--help` writes to stdout so it can be piped (#420).
- The CLI validates its arguments before dispatching to a backend, and a
  mistyped `-m` no longer silently substitutes the backend's registry default.
- Enrolling a speaker through an open handle updates that handle's in-memory
  roster.
- Pocket-TTS loads the official voice embeddings; Piper resolves exact
  community voices from model directories and can download its G2P data on a
  clean install; the model cache follows Hugging Face redirects on Windows.

## Build, tests and documentation

- The path-selection decision table shared by HTDemucs and mel-band-roformer is
  hoisted into `core/backend_path_gates.h` — the two copies were byte-identical.
- `cppcheck` moved to a non-cancelling deep-lint workflow: it never completed
  on main, because a newer push cancelled it every time.
- An orphaned-label audit found `ci/run.sh` was testing nothing, and the CLI
  regression tests were not labelled `unit`, so CI never ran them.
- The fuzz regression corpus now actually replays. `tests/.gitignore` excludes
  `*.ogg`, so the one documented seed had never been committed and the copy
  step's glob matched nothing on every run since it was written — an empty
  replay set being indistinguishable from a passing one. The directory now
  re-includes audio extensions, and the CI step counts what it copied and fails
  at zero.
- Arm64 HiFT SIMD convolution identity tests pin `-ffp-contract=off` (#421).
- Feasibility studies for Breeze-TTS-2 (#412) and IndexTTS-2.5 (#346) are on
  file with their converters and reference oracles.
- A beginner tutorial is in `docs/getting-started.md`, every command in it
  execution-verified.

---

**Upgrading.** No configuration changes are required. HTDemucs and
mel-band-roformer now use the GPU by default on hosts that have one; set
`CRISPASR_HTDEMUCS_GPU=0` or `CRISPASR_MELBAND_GPU=0` to keep the previous
behaviour. Nemotron users on `-l auto` will see it select a language other than
English for the first time — that is the fix, and `-l xx` is unchanged. If you
transcribe audio that legitimately decodes to more than 256 frames per input
byte, raise `CRISPASR_MAX_DECODED_FRAMES`; the rejection message names it.
