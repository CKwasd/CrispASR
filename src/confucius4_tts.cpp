// confucius4_tts.cpp — Confucius4-TTS backend (§377).
//
// T2S: GPT-2 (24L/1280d/20h) causal LM with custom embedding concatenation:
//   [condition_emb(1,1,1280) | text_emb(1,T,1280) | semantic_emb(1,T',1280)]
// Generates semantic codes (vocab 8194) autoregressively.
//
// S2A: Flow-matching DiT(13L) + WaveNet(8L) → 80-band mel.
// Vocoder: BigVGAN → 22050 Hz PCM (external, not yet ported).

#include "confucius4_tts.h"

#include "core/ggml_cpu_backend.h"
#include "core/gpu_backend_pref.h"
#include "core/gguf_loader.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Model hparams
// ---------------------------------------------------------------------------

struct confucius4_t2s_hparams {
    int num_layers = 24;
    int model_dim = 1280;
    int num_heads = 20;
    int max_text_seq_lens = 520;
    int max_semantic_seq_lens = 1520;
    int vocab_size = 32000;
    int semantic_vocab_size = 8194;
    int text_embedding_dim = 4096;
    int speaker_embedding_dim = 1024;
    int start_semantic_token = 8192;
    int stop_semantic_token = 8193;
    int sample_rate = 22050;
    int head_dim() const { return model_dim / num_heads; }
    int ffn_dim() const { return model_dim * 4; } // GPT-2 default: 4x
};

// ---------------------------------------------------------------------------
// T2S model weights
// ---------------------------------------------------------------------------

struct confucius4_t2s_layer {
    // Pre-attention LayerNorm
    struct ggml_tensor* ln_1_w = nullptr;
    struct ggml_tensor* ln_1_b = nullptr;
    // QKV fused + output projection (GPT-2 Conv1D style)
    struct ggml_tensor* attn_qkv_w = nullptr;  // [3*d, d]
    struct ggml_tensor* attn_qkv_b = nullptr;  // [3*d]
    struct ggml_tensor* attn_proj_w = nullptr; // [d, d]
    struct ggml_tensor* attn_proj_b = nullptr;
    // Pre-FFN LayerNorm
    struct ggml_tensor* ln_2_w = nullptr;
    struct ggml_tensor* ln_2_b = nullptr;
    // FFN (GPT-2 MLP: c_fc + c_proj)
    struct ggml_tensor* ffn_fc_w = nullptr; // [4d, d]
    struct ggml_tensor* ffn_fc_b = nullptr;
    struct ggml_tensor* ffn_proj_w = nullptr; // [d, 4d]
    struct ggml_tensor* ffn_proj_b = nullptr;
};

struct confucius4_t2s_model {
    confucius4_t2s_hparams hp;

    // Text projector: Embedding(vocab,4096) → Linear(4096,4096) → SiLU → Linear(4096,d)
    struct ggml_tensor* text_embed_w = nullptr;    // [4096, vocab]
    struct ggml_tensor* text_proj_fc1_w = nullptr; // [4096, 4096]
    struct ggml_tensor* text_proj_fc1_b = nullptr;
    struct ggml_tensor* text_proj_fc2_w = nullptr; // [4096, 1280]  (note: ggml col-major)
    struct ggml_tensor* text_proj_fc2_b = nullptr;

    // Semantic embedding + position embeddings
    struct ggml_tensor* semantic_embed_w = nullptr; // [d, semantic_vocab]
    struct ggml_tensor* text_pos_embed_w = nullptr; // [d, max_text]
    struct ggml_tensor* sem_pos_embed_w = nullptr;  // [d, max_semantic]

    // GPT-2 transformer layers
    std::vector<confucius4_t2s_layer> layers;

    // Final norm + semantic head
    struct ggml_tensor* final_norm_w = nullptr;
    struct ggml_tensor* final_norm_b = nullptr;
    struct ggml_tensor* semantic_head_w = nullptr; // [d, semantic_vocab]
    struct ggml_tensor* semantic_head_b = nullptr;

    // Speaker encoder (ECAPA-TDNN) — loaded but not yet wired for GPU compute
    // (the speaker encoder runs on Wav2Vec2-BERT output, which is external)
    // For now, conditioning comes pre-computed via confucius4_tts_set_speaker().
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct confucius4_tts_context {
    confucius4_tts_params params;
    confucius4_t2s_model t2s;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    struct ggml_context* ctx_w = nullptr;

    // Pre-computed speaker conditioning (from external Wav2Vec2-BERT + CAMPPlus)
    std::vector<float> speaker_semantic_features; // (n_frames, 1024)
    int speaker_n_frames = 0;
    std::vector<float> speaker_style_embedding; // (192,)
    bool has_speaker = false;

    // Tokenizer (SentencePiece model bytes, baked in GGUF)
    std::vector<uint8_t> tokenizer_model;

    // w2v-bert normalisation stats
    std::vector<float> w2v_mean; // (1024,)
    std::vector<float> w2v_var;  // (1024,)
};

// ---------------------------------------------------------------------------
// Default params
// ---------------------------------------------------------------------------

confucius4_tts_params confucius4_tts_default_params(void) {
    confucius4_tts_params p{};
    p.n_threads = 0;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature = 0.8f;
    p.top_p = 0.8f;
    p.top_k = 30;
    p.repetition_penalty = 10.0f;
    p.max_semantic_tokens = 0; // 0 = use hparams default (1520)
    p.ode_steps = 0;           // 0 = default (25)
    p.cfg_rate = 0.0f;         // 0 = default (0.7)
    p.seed = 0;
    return p;
}

// ---------------------------------------------------------------------------
// Load T2S model from GGUF
// ---------------------------------------------------------------------------

static bool load_t2s(confucius4_tts_context* ctx, const char* path) {
    auto& m = ctx->t2s;
    auto& hp = m.hp;

    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta) {
        fprintf(stderr, "confucius4: cannot open T2S GGUF '%s'\n", path);
        return false;
    }

    // Read hparams from GGUF KV
    hp.num_layers = core_gguf::kv_u32(meta, "confucius4.t2s.num_layers", 24);
    hp.model_dim = core_gguf::kv_u32(meta, "confucius4.t2s.model_dim", 1280);
    hp.num_heads = core_gguf::kv_u32(meta, "confucius4.t2s.num_heads", 20);
    hp.max_text_seq_lens = core_gguf::kv_u32(meta, "confucius4.t2s.max_text_seq_lens", 520);
    hp.max_semantic_seq_lens = core_gguf::kv_u32(meta, "confucius4.t2s.max_semantic_seq_lens", 1520);
    hp.vocab_size = core_gguf::kv_u32(meta, "confucius4.t2s.vocab_size", 32000);
    hp.semantic_vocab_size = core_gguf::kv_u32(meta, "confucius4.t2s.semantic_vocab_size", 8194);
    hp.text_embedding_dim = core_gguf::kv_u32(meta, "confucius4.t2s.text_embedding_dim", 4096);
    hp.speaker_embedding_dim = core_gguf::kv_u32(meta, "confucius4.t2s.speaker_embedding_dim", 1024);
    hp.start_semantic_token = core_gguf::kv_u32(meta, "confucius4.t2s.start_semantic_token", 8192);
    hp.stop_semantic_token = core_gguf::kv_u32(meta, "confucius4.t2s.stop_semantic_token", 8193);
    hp.sample_rate = core_gguf::kv_u32(meta, "confucius4.sample_rate", 22050);

    // Read w2v-bert normalisation stats
    ctx->w2v_mean = core_gguf::kv_f32_array(meta, "confucius4.w2v_bert.mean");
    ctx->w2v_var = core_gguf::kv_f32_array(meta, "confucius4.w2v_bert.var");
    // Tokenizer: loaded separately from companion tokenizer.json or .model file.
    // The GGUF carries the raw bytes in "tokenizer.model" but loading them
    // requires a SentencePiece protobuf parser — deferred to the synthesis path.

    core_gguf::free_metadata(meta);

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "confucius4: T2S hparams: %dL/%dd/%dh, text_vocab=%d, semantic_vocab=%d\n", hp.num_layers,
                hp.model_dim, hp.num_heads, hp.vocab_size, hp.semantic_vocab_size);
    }

    // Two-pass GGUF load via core_gguf::load_weights
    m.layers.resize(hp.num_layers);

    ctx->backend_cpu = core_cpu_backend::init();
    core_cpu_backend::set_n_threads(ctx->backend_cpu, ctx->params.n_threads);
    if (ctx->params.use_gpu) {
        ctx->backend = crispasr_init_gpu_backend();
        if (!ctx->backend)
            ctx->backend = ctx->backend_cpu;
    } else {
        ctx->backend = ctx->backend_cpu;
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "confucius4-t2s", wl)) {
        fprintf(stderr, "confucius4: failed to load T2S weights\n");
        return false;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;

    auto find = [&](const char* name) -> ggml_tensor* {
        auto it = wl.tensors.find(name);
        return it != wl.tensors.end() ? it->second : nullptr;
    };

    // Text projector
    m.text_embed_w = find("text_projector.embed.weight");
    m.text_proj_fc1_w = find("text_projector.text_projection_fc1.weight");
    m.text_proj_fc1_b = find("text_projector.text_projection_fc1.bias");
    m.text_proj_fc2_w = find("text_projector.text_projection_fc2.weight");
    m.text_proj_fc2_b = find("text_projector.text_projection_fc2.bias");

    // Semantic + positional embeddings
    m.semantic_embed_w = find("semantic_embedding.weight");
    m.text_pos_embed_w = find("text_position_embedding.embedding.weight");
    m.sem_pos_embed_w = find("semantic_position_embedding.embedding.weight");

    // GPT-2 transformer layers
    for (int i = 0; i < hp.num_layers; i++) {
        auto& L = m.layers[i];
        char buf[128];
        auto tn = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "transformer.h.%d.%s", i, suffix);
            return find(buf);
        };
        L.ln_1_w = tn("ln_1.weight");
        L.ln_1_b = tn("ln_1.bias");
        L.attn_qkv_w = tn("attn.c_attn.weight");
        L.attn_qkv_b = tn("attn.c_attn.bias");
        L.attn_proj_w = tn("attn.c_proj.weight");
        L.attn_proj_b = tn("attn.c_proj.bias");
        L.ln_2_w = tn("ln_2.weight");
        L.ln_2_b = tn("ln_2.bias");
        L.ffn_fc_w = tn("mlp.c_fc.weight");
        L.ffn_fc_b = tn("mlp.c_fc.bias");
        L.ffn_proj_w = tn("mlp.c_proj.weight");
        L.ffn_proj_b = tn("mlp.c_proj.bias");
    }

    // Final norm + head
    m.final_norm_w = find("final_norm.weight");
    m.final_norm_b = find("final_norm.bias");
    m.semantic_head_w = find("semantic_head.weight");
    m.semantic_head_b = find("semantic_head.bias");

    // Verify critical tensors
    if (!m.text_embed_w || !m.semantic_embed_w || !m.semantic_head_w) {
        fprintf(stderr, "confucius4: missing critical T2S tensors\n");
        return false;
    }
    if (!m.layers[0].attn_qkv_w || !m.layers[hp.num_layers - 1].attn_qkv_w) {
        fprintf(stderr, "confucius4: missing transformer layer tensors\n");
        return false;
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "confucius4: T2S loaded %zu tensors OK\n", wl.tensors.size());

    return true;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

confucius4_tts_context* confucius4_tts_init_from_file(const char* path_t2s, confucius4_tts_params params) {
    auto* ctx = new confucius4_tts_context();
    ctx->params = params;

    if (!load_t2s(ctx, path_t2s)) {
        delete ctx;
        return nullptr;
    }

    return ctx;
}

int confucius4_tts_set_s2a_path(confucius4_tts_context* /*ctx*/, const char* /*path_s2a*/) {
    // TODO: load S2A model
    fprintf(stderr, "confucius4: S2A loading not yet implemented\n");
    return -1;
}

int confucius4_tts_set_speaker(confucius4_tts_context* ctx, const float* semantic_features, int n_frames,
                               const float* style_embedding) {
    if (!ctx || !semantic_features || !style_embedding || n_frames <= 0)
        return -1;

    ctx->speaker_semantic_features.assign(semantic_features, semantic_features + n_frames * 1024);
    ctx->speaker_n_frames = n_frames;
    ctx->speaker_style_embedding.assign(style_embedding, style_embedding + 192);
    ctx->has_speaker = true;
    return 0;
}

float* confucius4_tts_synthesize(confucius4_tts_context* ctx, const char* text, const char* lang, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;

    // TODO: implement the full pipeline:
    // 1. Tokenize text with SentencePiece
    // 2. Run speaker encoder on conditioning features → condition_emb
    // 3. Build prefix: [condition_emb | text_emb | BOS]
    // 4. Autoregressive GPT-2 decode → semantic codes
    // 5. S2A flow-matching → mel
    // 6. BigVGAN vocoder → PCM

    (void)lang;
    fprintf(stderr, "confucius4: synthesis not yet implemented\n");
    *out_n_samples = 0;
    return nullptr;
}

void confucius4_tts_pcm_free(float* pcm) {
    free(pcm);
}

void confucius4_tts_free(confucius4_tts_context* ctx) {
    if (!ctx)
        return;

    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);

    delete ctx;
}

int confucius4_tts_sample_rate(const confucius4_tts_context* ctx) {
    return ctx ? ctx->t2s.hp.sample_rate : 22050;
}
