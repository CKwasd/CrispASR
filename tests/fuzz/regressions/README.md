# Fuzz regression seeds

Inputs that once crashed a parser, kept as permanent corpus entries. The
smoke-fuzz job copies this directory into its corpus, so every run replays them
— a fuzzing campaign rediscovers a crash only by luck, and a 45 s seeded run is
stochastic (the stb_vorbis crash below failed CI on a *documentation-only*
commit and passed on the four before it).

Each file is the minimal input that reproduces, crafted rather than harvested
where that was practical, so it is deterministic and small enough to read.

**These files are source, and `tests/.gitignore` excludes `*.wav` / `*.ogg`.**
`ogg-huge-comment-count.ogg` below was written, documented here, and named in
`ci.yml` — and never committed, because `git add` obeyed that pattern without
saying so. The glob matched nothing on every run from then on, and an empty
replay set is indistinguishable from a passing one. `.gitignore` in this
directory now re-includes the audio extensions, and the CI step counts what it
copied and fails at zero. The ogg seed itself is LOST and has to be re-crafted
from the description below before that row is a gate again.

| File | Bug | Fixed in |
|---|---|---|
| `wav-1hz-resample-oom.wav` | 344 KB, harvested from the run that found it (CI run 33840954769). A structurally valid RIFF/WAVE declaring `sampleRate = 1`. miniaudio resamples it to the 16 kHz target — 16000x — so 176 000 stored samples become 2.816e9 output frames, 11.3 GB, and the chunked decode loops doubled their buffer with no ceiling. Reachable from any surface that accepts a user file, including server upload. | `src/crispasr_audio.cpp`, `crispasr_max_decoded_frames()` bounds decoded frames against input size at all three loops |
| `ogg-huge-comment-count.ogg` **(MISSING — see above)** | 102 bytes. Ogg/Vorbis comment header declaring `comment_list_length = 0x3FFFFFFF`. The allocation of `sizeof(char*) * length` fails, and stb_vorbis returned from the error path with the length still set and `comment_list` NULL — `vorbis_deinit` then indexed the null array. ASAN: `SEGV in vorbis_deinit`, reached from `crispasr_audio_load`. | `examples/stb_vorbis.c`, guard in `vorbis_deinit` + reset the length on the error path |

## Adding one

Craft or minimise the input, drop it here, and add a row. Keep them small: they
run on every CI push. To check a seed still reproduces against an unpatched
build, revert the fix and run the single input directly — `libcrispasr` is a
shared library, so rebuild it, not just the harness, or you will test the new
code with an old-looking binary:

```
./build-fuzz/bin/crispasr-fuzz-audio tests/fuzz/regressions/<file>
```
