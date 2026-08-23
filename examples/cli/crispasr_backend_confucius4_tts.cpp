// crispasr_backend_confucius4_tts.cpp — CLI adapter for Confucius4-TTS (§377).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"
#include "confucius4_tts.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

class Confucius4TtsBackend : public CrispasrBackend {
public:
    const char* name() const override { return "confucius4-tts"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD; }

    bool init(const whisper_params& p) override {
        auto cp = confucius4_tts_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.temperature = p.temperature;
        cp.seed = p.seed;
        if (p.tts_steps > 0)
            cp.ode_steps = p.tts_steps;

        ctx_ = confucius4_tts_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[confucius4-tts]: failed to load T2S model '%s'\n", p.model.c_str());
            return false;
        }

        // Load the S2A companion model (--codec-model or auto-resolved sibling)
        if (!p.tts_codec_model.empty()) {
            if (confucius4_tts_set_s2a_path(ctx_, p.tts_codec_model.c_str()) != 0) {
                fprintf(stderr, "crispasr[confucius4-tts]: failed to load S2A model '%s'\n", p.tts_codec_model.c_str());
                // Non-fatal: T2S works without S2A (generates codes but no audio)
            }
        }

        return true;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_)
            return {};

        const std::string lang = params.language.empty() ? "en" : params.language;
        int n_samples = 0;
        float* pcm = confucius4_tts_synthesize(ctx_, text.c_str(), lang.c_str(), &n_samples);
        if (!pcm || n_samples <= 0)
            return {};

        std::vector<float> out(pcm, pcm + n_samples);
        confucius4_tts_pcm_free(pcm);
        return out;
    }

    int tts_sample_rate() const override { return confucius4_tts_sample_rate(ctx_); }

    void shutdown() override {
        if (ctx_) {
            confucius4_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~Confucius4TtsBackend() override { Confucius4TtsBackend::shutdown(); }

private:
    confucius4_tts_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_confucius4_tts_backend() {
    return std::make_unique<Confucius4TtsBackend>();
}
