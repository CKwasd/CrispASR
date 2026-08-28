#pragma once

#include "core/parallel_for.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define CRISPASR_CHATTERBOX_F0_X86_TARGET_DISPATCH 1
#else
#define CRISPASR_CHATTERBOX_F0_X86_TARGET_DISPATCH 0
#endif

namespace core_chatterbox_f0 {

enum class Isa {
    scalar,
    avx2,
    avx512f,
};

inline bool isa_available(Isa isa) {
    if (isa == Isa::scalar)
        return true;
#if CRISPASR_CHATTERBOX_F0_X86_TARGET_DISPATCH
    if (isa == Isa::avx512f)
        return __builtin_cpu_supports("avx512f");
    if (isa == Isa::avx2)
        return __builtin_cpu_supports("avx2");
#endif
    return false;
}

inline Isa best_isa() {
    static const Isa isa = [] {
        if (isa_available(Isa::avx512f))
            return Isa::avx512f;
        if (isa_available(Isa::avx2))
            return Isa::avx2;
        return Isa::scalar;
    }();
    return isa;
}

// Non-owning view of one k=3, p=1 Conv1d + ELU operation.
struct Conv1dEluK3 {
    const float* input;
    const float* weights;
    const float* bias;
    float* output;
    int frames;
    int input_channels;
    int output_channels;

    void run(int n_threads, Isa isa = best_isa()) const;
    void run_scalar(int n_threads = 1) const;
};

namespace detail {

inline float elu(float value) {
    return value < 0.0f ? std::exp(value) - 1.0f : value;
}

inline void run_scalar_rows(const Conv1dEluK3& conv, int begin, int end) {
    for (int frame = begin; frame < end; ++frame) {
        for (int output_channel = 0; output_channel < conv.output_channels; ++output_channel) {
            float sum = conv.bias[output_channel];
            for (int kernel = 0; kernel < 3; ++kernel) {
                const int input_frame = frame + kernel - 1;
                if (input_frame < 0 || input_frame >= conv.frames)
                    continue;
                const float* input_row = conv.input + static_cast<std::size_t>(input_frame) * conv.input_channels;
                const float* kernel_weights =
                    conv.weights + static_cast<std::size_t>(output_channel) * conv.input_channels * 3 + kernel;
                for (int input_channel = 0; input_channel < conv.input_channels; ++input_channel)
                    sum += kernel_weights[input_channel * 3] * input_row[input_channel];
            }
            conv.output[static_cast<std::size_t>(frame) * conv.output_channels + output_channel] = elu(sum);
        }
    }
}

// PRECONDITION: width divides conv.output_channels. `packed` is sized for
// exactly output_channels entries, but the block layout below addresses
// ceil(output_channels / width) WHOLE blocks — so a partial trailing block
// would both read weights past output_channels and write past `packed`. The
// only caller, Conv1dEluK3::run, enforces the divisibility before choosing a
// width; this asserts it here too, because the guard and the code that
// depends on it are 100 lines apart.
inline std::vector<float> pack_weights(const Conv1dEluK3& conv, int width) {
    assert(width > 0 && conv.output_channels % width == 0);
    std::vector<float> packed(static_cast<std::size_t>(conv.output_channels) * conv.input_channels * 3);
    for (int output = 0; output < conv.output_channels; output += width) {
        const int block = output / width;
        for (int kernel = 0; kernel < 3; ++kernel) {
            for (int input = 0; input < conv.input_channels; ++input) {
                float* destination =
                    packed.data() +
                    ((static_cast<std::size_t>(block) * 3 + kernel) * conv.input_channels + input) * width;
                for (int lane = 0; lane < width; ++lane) {
                    destination[lane] = conv.weights[static_cast<std::size_t>(output + lane) * conv.input_channels * 3 +
                                                     input * 3 + kernel];
                }
            }
        }
    }
    return packed;
}

#if CRISPASR_CHATTERBOX_F0_X86_TARGET_DISPATCH

#if defined(__clang__)
#define CRISPASR_CHATTERBOX_F0_TARGET_AVX2 __attribute__((target("avx2")))
#define CRISPASR_CHATTERBOX_F0_TARGET_AVX512 __attribute__((target("avx512f")))
#define CRISPASR_CHATTERBOX_F0_NO_CONTRACT
#define CRISPASR_CHATTERBOX_F0_CLANG_NO_CONTRACT _Pragma("clang fp contract(off)")
#else
#define CRISPASR_CHATTERBOX_F0_TARGET_AVX2 __attribute__((target("avx2")))
#define CRISPASR_CHATTERBOX_F0_TARGET_AVX512 __attribute__((target("avx512f")))
#define CRISPASR_CHATTERBOX_F0_NO_CONTRACT __attribute__((optimize("fp-contract=off")))
#define CRISPASR_CHATTERBOX_F0_CLANG_NO_CONTRACT
#endif

constexpr int avx2_width = 8;
constexpr int avx512_width = 16;

CRISPASR_CHATTERBOX_F0_TARGET_AVX2
CRISPASR_CHATTERBOX_F0_NO_CONTRACT
inline void run_avx2_rows(const Conv1dEluK3& conv, const float* packed, int begin, int end) {
    CRISPASR_CHATTERBOX_F0_CLANG_NO_CONTRACT
    for (int frame = begin; frame < end; ++frame) {
        for (int block = 0; block < conv.output_channels / avx2_width; ++block) {
            __m256 sum = _mm256_loadu_ps(conv.bias + block * avx2_width);
            const float* block_weights =
                packed + static_cast<std::size_t>(block) * 3 * conv.input_channels * avx2_width;
            for (int kernel = 0; kernel < 3; ++kernel) {
                const int input_frame = frame + kernel - 1;
                if (input_frame < 0 || input_frame >= conv.frames)
                    continue;
                const float* input = conv.input + static_cast<std::size_t>(input_frame) * conv.input_channels;
                const float* weights =
                    block_weights + static_cast<std::size_t>(kernel) * conv.input_channels * avx2_width;
                for (int channel = 0; channel < conv.input_channels; ++channel) {
                    const __m256 weight = _mm256_loadu_ps(weights + channel * avx2_width);
                    const __m256 value = _mm256_set1_ps(input[channel]);
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(weight, value));
                }
            }
            std::array<float, avx2_width> lanes;
            _mm256_storeu_ps(lanes.data(), sum);
            float* output = conv.output + static_cast<std::size_t>(frame) * conv.output_channels + block * avx2_width;
            for (const auto value : lanes)
                *output++ = elu(value);
        }
    }
}

CRISPASR_CHATTERBOX_F0_TARGET_AVX512
CRISPASR_CHATTERBOX_F0_NO_CONTRACT
inline void run_avx512_rows(const Conv1dEluK3& conv, const float* packed, int begin, int end) {
    CRISPASR_CHATTERBOX_F0_CLANG_NO_CONTRACT
    for (int frame = begin; frame < end; ++frame) {
        for (int block = 0; block < conv.output_channels / avx512_width; ++block) {
            __m512 sum = _mm512_loadu_ps(conv.bias + block * avx512_width);
            const float* block_weights =
                packed + static_cast<std::size_t>(block) * 3 * conv.input_channels * avx512_width;
            for (int kernel = 0; kernel < 3; ++kernel) {
                const int input_frame = frame + kernel - 1;
                if (input_frame < 0 || input_frame >= conv.frames)
                    continue;
                const float* input = conv.input + static_cast<std::size_t>(input_frame) * conv.input_channels;
                const float* weights =
                    block_weights + static_cast<std::size_t>(kernel) * conv.input_channels * avx512_width;
                for (int channel = 0; channel < conv.input_channels; ++channel) {
                    const __m512 weight = _mm512_loadu_ps(weights + channel * avx512_width);
                    const __m512 value = _mm512_set1_ps(input[channel]);
                    sum = _mm512_add_ps(sum, _mm512_mul_ps(weight, value));
                }
            }
            std::array<float, avx512_width> lanes;
            _mm512_storeu_ps(lanes.data(), sum);
            float* output = conv.output + static_cast<std::size_t>(frame) * conv.output_channels + block * avx512_width;
            for (const auto value : lanes)
                *output++ = elu(value);
        }
    }
}

using SimdRows = void (*)(const Conv1dEluK3&, const float*, int, int);

inline void run_simd(const Conv1dEluK3& conv, int width, int n_threads, SimdRows run_rows) {
    const auto packed = pack_weights(conv, width);
    core_parallel::for_each_chunk(conv.frames, n_threads,
                                  [&](int begin, int end) { run_rows(conv, packed.data(), begin, end); });
}

#undef CRISPASR_CHATTERBOX_F0_CLANG_NO_CONTRACT
#undef CRISPASR_CHATTERBOX_F0_TARGET_AVX2
#undef CRISPASR_CHATTERBOX_F0_TARGET_AVX512
#undef CRISPASR_CHATTERBOX_F0_NO_CONTRACT

#endif // CRISPASR_CHATTERBOX_F0_X86_TARGET_DISPATCH

} // namespace detail

inline void Conv1dEluK3::run_scalar(int n_threads) const {
    core_parallel::for_each_chunk(frames, n_threads,
                                  [this](int begin, int end) { detail::run_scalar_rows(*this, begin, end); });
}

// `isa` is a CEILING, not an exact request: each kernel needs
// output_channels to be a whole number of its vector width, so a width that
// does not divide falls through to the next narrower one. Without the
// cascade, asking for avx512f with output_channels % 16 != 0 skipped AVX2
// entirely and landed on scalar — e.g. 520 channels (a multiple of 8, not of
// 16) ran ~8x slower than it had to. The F0 predictor only ever passes 512,
// so this is about reuse of this header, not about today's caller.
inline void Conv1dEluK3::run(int n_threads, Isa isa) const {
#if CRISPASR_CHATTERBOX_F0_X86_TARGET_DISPATCH
    if (isa == Isa::avx512f && isa_available(Isa::avx512f) && output_channels % detail::avx512_width == 0) {
        detail::run_simd(*this, detail::avx512_width, n_threads, detail::run_avx512_rows);
        return;
    }
    if ((isa == Isa::avx2 || isa == Isa::avx512f) && isa_available(Isa::avx2) &&
        output_channels % detail::avx2_width == 0) {
        detail::run_simd(*this, detail::avx2_width, n_threads, detail::run_avx2_rows);
        return;
    }
#endif
    run_scalar(n_threads);
}

} // namespace core_chatterbox_f0

#undef CRISPASR_CHATTERBOX_F0_X86_TARGET_DISPATCH
