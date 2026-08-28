// crispasr_stream_postprocess.h — streaming post-processing placement
// policy for PCS and truecaser models.
//
// Follows the same three-mode design as crispasr_stream_punc.h (PR #112).
// Controls whether the PCS model and the three truecaser variants
// (statistical, CRF, BiLSTM) run on partial decodes or only on finals.
//
// Three modes selectable via `--stream-postprocess-mode`:
//   "off"      — PCS + truecaser run on neither partials nor finals.
//   "final"    — PCS + truecaser run on finals only. Default; recommended
//                for realtime use because it keeps the high-frequency
//                partial path cheap while still restoring proper casing
//                and punctuation on finalized utterances.
//   "partial"  — PCS + truecaser run on partials AND finals. Equivalent
//                to the historical pre-PR behaviour.

#pragma once

#include <string>

inline bool crispasr_stream_postprocess_partials_enabled(const std::string& mode) {
    return mode == "partial";
}

inline bool crispasr_stream_postprocess_finals_enabled(const std::string& mode) {
    return mode == "final" || mode == "partial";
}

/// Returns true when the mode string is one of the three accepted
/// values. The CLI argument parser uses an equivalent check and
/// exits with code 2 on a bad value; this is the shared
/// truth-source.
inline bool crispasr_stream_postprocess_mode_valid(const std::string& mode) {
    return mode == "off" || mode == "final" || mode == "partial";
}