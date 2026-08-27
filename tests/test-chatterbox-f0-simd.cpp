#include "core/chatterbox_f0_conv.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

void fill_exact(std::vector<float>& values, std::uint32_t seed) {
    std::size_t index = 0;
    for (auto& value : values) {
        const auto integer = static_cast<int>((index++ * 37u + seed * 17u) % 127u) - 63;
        value = static_cast<float>(integer) / 4096.0f;
    }
}

void require_matches_scalar(int T, int C_in, int n_threads, core_chatterbox_f0::Isa isa) {
    constexpr int C_out = 512;
    std::vector<float> x(static_cast<std::size_t>(T) * C_in);
    std::vector<float> w(static_cast<std::size_t>(C_out) * C_in * 3);
    std::vector<float> bias(C_out);
    fill_exact(x, 11);
    fill_exact(w, 23);
    fill_exact(bias, 47);

    std::vector<float> reference(static_cast<std::size_t>(T) * C_out);
    std::vector<float> actual(reference.size());
    core_chatterbox_f0::Conv1dEluK3{x.data(), w.data(), bias.data(), reference.data(), T, C_in, C_out}.run_scalar();
    core_chatterbox_f0::Conv1dEluK3{x.data(), w.data(), bias.data(), actual.data(), T, C_in, C_out}.run(n_threads, isa);

    INFO("isa=" << static_cast<int>(isa) << " T=" << T << " C_in=" << C_in << " threads=" << n_threads);
    REQUIRE(std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(float)) == 0);
}

} // namespace

TEST_CASE("Chatterbox F0 SIMD Conv1d is bit-identical to scalar", "[unit][tts][chatterbox][simd]") {
    const std::array<core_chatterbox_f0::Isa, 3> candidates{{
        core_chatterbox_f0::Isa::scalar,
        core_chatterbox_f0::Isa::avx2,
        core_chatterbox_f0::Isa::avx512f,
    }};

    for (const auto isa : candidates) {
        if (!core_chatterbox_f0::isa_available(isa))
            continue;
        require_matches_scalar(7, 80, 1, isa);
        require_matches_scalar(7, 512, 1, isa);
        require_matches_scalar(17, 80, 4, isa);
        require_matches_scalar(17, 512, 4, isa);
    }
}

TEST_CASE("Chatterbox F0 auto dispatch preserves scalar output", "[unit][tts][chatterbox][simd]") {
    require_matches_scalar(17, 512, 4, core_chatterbox_f0::best_isa());
}
