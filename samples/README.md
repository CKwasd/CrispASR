# Audio samples

This folder contains various audio files used for testing.
If you want to quickly get some more samples, simply run `make samples`. This will download several public audio files and convert them to appropriate 16-bit WAV format using `ffmpeg`

https://github.com/ggerganov/crispasr/blob/a09ce6e8899198015729ffc49ae10f67370906b1/Makefile#L104-L123

## `jfk-live.webm` — streaming/live WebM regression fixture

`jfk.webm` is a normal, seekable WebM: every element carries an explicit size.
`jfk-live.webm` holds the same Opus packets re-muxed the way a **live** muxer
writes them — the way Chrome's `MediaRecorder` (libwebm `mkvmuxer` against a
non-seekable writer) does:

- the **Segment** and every **Cluster** carry the EBML *unknown size* marker,
  because a streaming muxer cannot seek back to patch the length once the
  recording ends;
- one Cluster per 100 ms `timeslice`, so there are 110 of them;
- no `Duration`, `Cues` or `SeekHead` (`ffprobe` reports `Duration: N/A`).

Regenerate with `tools/make-live-webm.py samples/jfk.webm out.webm 100`.

It guards issue #417: the EBML demuxer used to stop at the end of the first
Cluster, returning ~0.1 s of the 11 s recording.
