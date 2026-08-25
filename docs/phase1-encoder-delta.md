# Phase 1: Cohere Encoder Streaming Delta Encoding

> Corresponds to lag_optimization_proposals.md section 2.3

## 1. Motivation

In JSON+VAD streaming, the rolling window re-encodes ~90% overlapping audio.
Cohere encoder is ~87% of total inference time.

Benefit: ~90% encoder reduction per step.

## 2. Architecture

Step N: full encode, cache cross-KV on CPU.
Step N+1: only encode delta samples, splice cached + delta cross-KV.

## 3. Key Mechanisms

- cohere_encode_and_extract_cross_kv() helper function
- stream_cached_k/v per-layer CPU F32 cache
- CAP_STREAM_DELTA capability flag
- set_stream_delta() backend interface

## 4. Files Changed

src/cohere.h, src/cohere.cpp, examples/cli/crispasr_backend.h,
examples/cli/crispasr_backend_cohere.cpp, examples/cli/crispasr_run.cpp

## 5. Test Results

Build: PASSED
Inference (JFK): correct output
Streaming integration: set_stream_delta() called correctly
