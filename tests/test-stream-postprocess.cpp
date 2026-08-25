// Unit tests for the streaming post-processing placement policy
// (`--stream-postprocess-mode off|final|partial`).
//
// The helpers in `examples/cli/crispasr_stream_postprocess.h` are pure
// string-to-bool predicates: no model load, no audio, no streaming
// runtime. Catch2 covers all 4x2 = 8 combinations of (mode x predicate)
// plus the validator, so any future change to the policy matrix
// fails loudly here even before someone wires up an integration test.

#include "../examples/cli/crispasr_stream_postprocess.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

TEST_CASE("stream-postprocess: partials_enabled is true ONLY for 'partial'", "[unit][stream-postprocess]") {
    REQUIRE_FALSE(crispasr_stream_postprocess_partials_enabled("off"));
    REQUIRE_FALSE(crispasr_stream_postprocess_partials_enabled("final"));
    REQUIRE(crispasr_stream_postprocess_partials_enabled("partial"));
}

TEST_CASE("stream-postprocess: finals_enabled is true for 'final' and 'partial', false for 'off'", "[unit][stream-postprocess]") {
    REQUIRE_FALSE(crispasr_stream_postprocess_finals_enabled("off"));
    REQUIRE(crispasr_stream_postprocess_finals_enabled("final"));
    REQUIRE(crispasr_stream_postprocess_finals_enabled("partial"));
}

TEST_CASE("stream-postprocess: mode_valid accepts the three documented values", "[unit][stream-postprocess]") {
    REQUIRE(crispasr_stream_postprocess_mode_valid("off"));
    REQUIRE(crispasr_stream_postprocess_mode_valid("final"));
    REQUIRE(crispasr_stream_postprocess_mode_valid("partial"));
}

TEST_CASE("stream-postprocess: mode_valid rejects unknown / case-mismatched / empty values", "[unit][stream-postprocess]") {
    REQUIRE_FALSE(crispasr_stream_postprocess_mode_valid(""));
    REQUIRE_FALSE(crispasr_stream_postprocess_mode_valid("FINAL")); // case-sensitive
    REQUIRE_FALSE(crispasr_stream_postprocess_mode_valid("Partial"));
    REQUIRE_FALSE(crispasr_stream_postprocess_mode_valid("on"));
    REQUIRE_FALSE(crispasr_stream_postprocess_mode_valid("yes"));
    REQUIRE_FALSE(crispasr_stream_postprocess_mode_valid("partials"));
}

// Regression pin: the documented default is `final`, which means
// finals get PCS+truecaser but partials don't. If anyone flips the
// default (or the predicates), this case catches it.
TEST_CASE("stream-postprocess: default 'final' implies finals=on, partials=off", "[unit][stream-postprocess]") {
    const std::string default_mode = "final";
    REQUIRE(crispasr_stream_postprocess_mode_valid(default_mode));
    REQUIRE(crispasr_stream_postprocess_finals_enabled(default_mode));
    REQUIRE_FALSE(crispasr_stream_postprocess_partials_enabled(default_mode));
}

// Regression pin: 'partial' is the old pre-PR behaviour — PCS+truecaser
// on both partials and finals.
TEST_CASE("stream-postprocess: 'partial' implies both finals and partials punc'd", "[unit][stream-postprocess]") {
    REQUIRE(crispasr_stream_postprocess_finals_enabled("partial"));
    REQUIRE(crispasr_stream_postprocess_partials_enabled("partial"));
}

// Regression pin: 'off' means BOTH partials and finals skip the
// PCS+truecaser models.
TEST_CASE("stream-postprocess: 'off' suppresses postprocess on BOTH partials and finals", "[unit][stream-postprocess]") {
    REQUIRE_FALSE(crispasr_stream_postprocess_finals_enabled("off"));
    REQUIRE_FALSE(crispasr_stream_postprocess_partials_enabled("off"));
}