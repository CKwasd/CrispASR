#include "core/realtime_turn_buffer.h"

#include <catch2/catch_test_macros.hpp>
#include <vector>

TEST_CASE("realtime turns buffer without requesting prefix reprocessing", "[unit][realtime]") {
    core_realtime::TurnBuffer turn(10);
    const std::vector<float> first(4, 0.25f);
    const std::vector<float> second(5, -0.25f);
    const std::vector<float> last(1, 0.5f);

    REQUIRE_FALSE(turn.append(first.data(), first.size()));
    REQUIRE_FALSE(turn.append(second.data(), second.size()));
    REQUIRE(turn.size() == 9);
    REQUIRE(turn.append(last.data(), last.size()));
    REQUIRE(turn.size() == 10);
}

TEST_CASE("commit reset prevents audio leaking into the next turn", "[unit][realtime]") {
    core_realtime::TurnBuffer turn(10);
    const std::vector<float> audio(6, 0.25f);
    REQUIRE_FALSE(turn.append(audio.data(), audio.size()));
    turn.clear();
    REQUIRE(turn.empty());
    REQUIRE_FALSE(turn.append(audio.data(), 2));
    REQUIRE(turn.size() == 2);
}

TEST_CASE("a large append cannot exceed the turn safety cap", "[unit][realtime]") {
    core_realtime::TurnBuffer turn(10);
    const std::vector<float> audio(25, 0.25f);
    REQUIRE(turn.append(audio.data(), audio.size()));
    REQUIRE(turn.size() == 10);
}
