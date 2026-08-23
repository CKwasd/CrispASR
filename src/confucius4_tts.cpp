// confucius4_tts.cpp — Confucius4-TTS backend (§377).
//
// T2S: GPT-2 (24L/1280d/20h) causal LM with custom embedding concatenation:
//   [condition_emb(1,1,1280) | text_emb(1,T,1280) | semantic_emb(1,T',1280)]
// Generates semantic codes (vocab 8194) autoregressively.
//
// S2A: Flow-matching DiT(13L) + WaveNet(8L) → 80-band mel.
// Vocoder: BigVGAN → 22050 Hz PCM (external, not yet ported).

#include "confucius4_tts.h"

#include "core/attention.h"
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

// KV cache for the GPT-2 T2S model.
struct confucius4_kv_cache {
    ggml_tensor* k = nullptr; // (hd, max_seq, n_heads, n_layers) F16
    ggml_tensor* v = nullptr;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    int max_seq_len = 0;
};

struct confucius4_tts_context {
    confucius4_tts_params params;
    confucius4_t2s_model t2s;
    confucius4_kv_cache kv;

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
// GPT-2 transformer forward: one step (prefill or single-token decode)
// ---------------------------------------------------------------------------

// Build a single GPT-2 forward pass graph for T input tokens.
// Returns the logits tensor (T, semantic_vocab_size).
// `kv_k` / `kv_v` are the KV cache tensors per layer (pre-allocated).
// `n_past` is the KV cache position (0 for prefill).
static ggml_tensor* gpt2_forward(confucius4_tts_context* ctx, ggml_context* ctx0, ggml_cgraph* gf,
                                 ggml_tensor* input_emb, // (model_dim, T)
                                 ggml_tensor* kv_k, ggml_tensor* kv_v, int n_past) {
    const auto& m = ctx->t2s;
    const auto& hp = m.hp;

    core_attn::KvSelfAttnParams ap;
    ap.n_heads = hp.num_heads;
    ap.n_kv_heads = hp.num_heads; // GPT-2: no GQA
    ap.n_kv_grp = 1;
    ap.head_dim = hp.head_dim();
    ap.rope_theta = 0.0f; // GPT-2 uses learned positional embeddings, not RoPE

    ggml_tensor* x = input_emb;

    for (int il = 0; il < hp.num_layers; il++) {
        const auto& L = m.layers[il];

        // Pre-attention LayerNorm
        ggml_tensor* ln1 = ggml_norm(ctx0, x, 1e-5f);
        ln1 = ggml_add(ctx0, ggml_mul(ctx0, ln1, L.ln_1_w), L.ln_1_b);

        // Self-attention with fused QKV (GPT-2 style)
        ggml_tensor* attn_out =
            core_attn::kv_self_attn(ctx0, gf, ln1,
                                    /*q_w=*/nullptr, /*k_w=*/nullptr, /*v_w=*/nullptr, L.attn_proj_w,
                                    /*q_norm_w=*/nullptr, /*k_norm_w=*/nullptr,
                                    /*positions=*/nullptr, /*causal_mask=*/nullptr, kv_k, kv_v, il, n_past, ap,
                                    /*qkv_w=*/L.attn_qkv_w, /*fixed_kv_len=*/0,
                                    /*kv_indices=*/nullptr,
                                    /*q_b=*/nullptr, /*k_b=*/nullptr, /*v_b=*/nullptr,
                                    /*o_b=*/L.attn_proj_b, /*qkv_b=*/L.attn_qkv_b);

        // Residual
        x = ggml_add(ctx0, x, attn_out);

        // Pre-FFN LayerNorm
        ggml_tensor* ln2 = ggml_norm(ctx0, x, 1e-5f);
        ln2 = ggml_add(ctx0, ggml_mul(ctx0, ln2, L.ln_2_w), L.ln_2_b);

        // FFN: Linear → GELU → Linear (GPT-2 MLP)
        ggml_tensor* ff = ggml_mul_mat(ctx0, L.ffn_fc_w, ln2);
        ff = ggml_add(ctx0, ff, L.ffn_fc_b);
        ff = ggml_gelu(ctx0, ff);
        ff = ggml_mul_mat(ctx0, L.ffn_proj_w, ff);
        ff = ggml_add(ctx0, ff, L.ffn_proj_b);

        // Residual
        x = ggml_add(ctx0, x, ff);
    }

    // Final LayerNorm
    x = ggml_norm(ctx0, x, 1e-5f);
    x = ggml_add(ctx0, ggml_mul(ctx0, x, m.final_norm_w), m.final_norm_b);

    // Semantic head: Linear(model_dim, semantic_vocab_size)
    ggml_tensor* logits = ggml_mul_mat(ctx0, m.semantic_head_w, x);
    logits = ggml_add(ctx0, logits, m.semantic_head_b);

    return logits;
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

// ---------------------------------------------------------------------------
// KV cache helpers
// ---------------------------------------------------------------------------

static bool kv_init(confucius4_kv_cache& kv, const confucius4_t2s_hparams& hp, int max_seq, ggml_backend_t backend) {
    kv.max_seq_len = max_seq;
    size_t ctx_size = 2 * ggml_tensor_overhead() + 64;
    ggml_init_params ip = {ctx_size, nullptr, true};
    kv.ctx = ggml_init(ip);
    if (!kv.ctx)
        return false;
    kv.k = ggml_new_tensor_4d(kv.ctx, GGML_TYPE_F16, hp.head_dim(), max_seq, hp.num_heads, hp.num_layers);
    kv.v = ggml_new_tensor_4d(kv.ctx, GGML_TYPE_F16, hp.head_dim(), max_seq, hp.num_heads, hp.num_layers);
    ggml_set_name(kv.k, "kv_k");
    ggml_set_name(kv.v, "kv_v");
    kv.buf = ggml_backend_alloc_ctx_tensors(kv.ctx, backend);
    if (!kv.buf)
        return false;
    ggml_backend_tensor_memset(kv.k, 0, 0, ggml_nbytes(kv.k));
    ggml_backend_tensor_memset(kv.v, 0, 0, ggml_nbytes(kv.v));
    return true;
}

static void kv_free(confucius4_kv_cache& kv) {
    if (kv.buf)
        ggml_backend_buffer_free(kv.buf);
    if (kv.ctx)
        ggml_free(kv.ctx);
    kv = {};
}

// ---------------------------------------------------------------------------
// Top-p sampling
// ---------------------------------------------------------------------------

static int sample_top_p(const float* logits, int n_vocab, float temperature, float top_p, int top_k,
                        std::mt19937& rng) {
    std::vector<std::pair<float, int>> candidates(n_vocab);
    for (int i = 0; i < n_vocab; i++)
        candidates[i] = {logits[i] / temperature, i};

    // Top-k filter
    if (top_k > 0 && top_k < n_vocab) {
        std::partial_sort(candidates.begin(), candidates.begin() + top_k, candidates.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        candidates.resize(top_k);
    } else {
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    // Softmax
    float max_val = candidates[0].first;
    float sum = 0.0f;
    for (auto& c : candidates) {
        c.first = expf(c.first - max_val);
        sum += c.first;
    }
    for (auto& c : candidates)
        c.first /= sum;

    // Top-p nucleus filter
    float cumsum = 0.0f;
    int last = (int)candidates.size();
    for (int i = 0; i < (int)candidates.size(); i++) {
        cumsum += candidates[i].first;
        if (cumsum >= top_p) {
            last = i + 1;
            break;
        }
    }
    candidates.resize(last);

    // Re-normalise and sample
    sum = 0.0f;
    for (auto& c : candidates)
        sum += c.first;
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng);
    cumsum = 0.0f;
    for (auto& c : candidates) {
        cumsum += c.first;
        if (cumsum >= r)
            return c.second;
    }
    return candidates.back().second;
}

// ---------------------------------------------------------------------------
// T2S decode: generate semantic codes from text token IDs
// ---------------------------------------------------------------------------

// Build and run the prefix embedding graph:
//   text_ids → Embedding(32k,4096) → Linear(4096,4096) → SiLU → Linear(4096,1280) + pos_emb
//   condition: zero vector (1, 1280) — speaker encoder not yet wired
//   BOS: semantic_embedding[start_semantic_token]
// Returns the concatenated prefix (prefix_len, model_dim) as float32.
static std::vector<float> build_prefix_embedding(confucius4_tts_context* ctx, const std::vector<int32_t>& text_ids) {
    const auto& m = ctx->t2s;
    const auto& hp = m.hp;
    const int D = hp.model_dim;
    const int T_text = (int)text_ids.size();
    const int prefix_len = 1 + T_text + 1; // condition(1) + text(T) + BOS(1)

    // Build the text projector graph: embed → fc1 → silu → fc2
    const int n_tensors = 64;
    size_t ctx_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(256, false);
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0)
        return {};

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 256, false);

    // Input: text token IDs
    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_text);
    ggml_set_name(ids, "text_ids");
    ggml_set_input(ids);

    // Embedding lookup: (4096, T_text)
    ggml_tensor* emb = ggml_get_rows(ctx0, m.text_embed_w, ids);

    // MLP: fc1(4096→4096) → SiLU → fc2(4096→1280)
    ggml_tensor* h = ggml_mul_mat(ctx0, m.text_proj_fc1_w, emb);
    h = ggml_add(ctx0, h, m.text_proj_fc1_b);
    h = ggml_silu(ctx0, h);
    h = ggml_mul_mat(ctx0, m.text_proj_fc2_w, h);
    h = ggml_add(ctx0, h, m.text_proj_fc2_b); // (D, T_text)

    // Add text positional embedding: pos_emb[0..T_text-1]
    ggml_tensor* pos_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_text);
    ggml_set_name(pos_ids, "text_pos_ids");
    ggml_set_input(pos_ids);

    ggml_tensor* pos_emb = ggml_get_rows(ctx0, m.text_pos_embed_w, pos_ids);
    ggml_tensor* text_emb = ggml_add(ctx0, h, pos_emb); // (D, T_text)
    ggml_set_name(text_emb, "text_emb_out");
    ggml_set_output(text_emb);

    // BOS semantic embedding: semantic_embed[start_token]
    ggml_tensor* bos_id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(bos_id, "bos_id");
    ggml_set_input(bos_id);

    ggml_tensor* bos_emb = ggml_get_rows(ctx0, m.semantic_embed_w, bos_id);
    // Add semantic position embedding at position 0
    ggml_tensor* sem_pos0_id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(sem_pos0_id, "sem_pos0_id");
    ggml_set_input(sem_pos0_id);
    ggml_tensor* sem_pos0 = ggml_get_rows(ctx0, m.sem_pos_embed_w, sem_pos0_id);
    ggml_tensor* bos_out = ggml_add(ctx0, bos_emb, sem_pos0);
    ggml_set_name(bos_out, "bos_emb_out");
    ggml_set_output(bos_out);

    ggml_build_forward_expand(gf, text_emb);
    ggml_build_forward_expand(gf, bos_out);

    // Allocate compute buffers with gallocr. Weight tensors already have valid
    // buffers from load_weights — gallocr skips them and only allocates the new
    // compute intermediates + inputs.
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "confucius4: prefix graph alloc failed\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return {};
    }

    // Debug: print shapes for diagnosis
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "confucius4: prefix graph: text_embed_w ne=[%lld,%lld] type=%d\n",
                (long long)m.text_embed_w->ne[0], (long long)m.text_embed_w->ne[1], (int)m.text_embed_w->type);
        fprintf(stderr, "confucius4: prefix graph: T_text=%d, max_id=%d\n", T_text,
                *std::max_element(text_ids.begin(), text_ids.end()));
        fprintf(stderr, "confucius4: prefix graph: semantic_embed ne=[%lld,%lld], BOS=%d\n",
                (long long)m.semantic_embed_w->ne[0], (long long)m.semantic_embed_w->ne[1], hp.start_semantic_token);
    }

    // Set inputs
    ggml_backend_tensor_set(ids, text_ids.data(), 0, T_text * sizeof(int32_t));

    std::vector<int32_t> pos_data(T_text);
    for (int i = 0; i < T_text; i++)
        pos_data[i] = i;
    ggml_backend_tensor_set(pos_ids, pos_data.data(), 0, T_text * sizeof(int32_t));

    int32_t bos_val = hp.start_semantic_token;
    ggml_backend_tensor_set(bos_id, &bos_val, 0, sizeof(int32_t));

    int32_t sem_pos0_val = 0;
    ggml_backend_tensor_set(sem_pos0_id, &sem_pos0_val, 0, sizeof(int32_t));

    // Debug: verify index tensor contents before compute
    if (ctx->params.verbosity >= 1) {
        int32_t id0 = -1;
        ggml_backend_tensor_get(ids, &id0, 0, sizeof(int32_t));
        fprintf(stderr, "confucius4: prefix graph: ids[0]=%d (expect %d), ids tensor ne=[%lld,%lld]\n", id0,
                text_ids[0], (long long)ids->ne[0], (long long)ids->ne[1]);
        fprintf(stderr, "confucius4: prefix graph: graph nodes=%d leafs=%d\n", ggml_graph_n_nodes(gf),
                ggml_graph_n_nodes(gf)); // leafs count not exposed, use nodes as proxy
    }

    ggml_backend_graph_compute(ctx->backend, gf);

    // Read results
    std::vector<float> result(prefix_len * D, 0.0f);

    // Slot 0: condition_emb — zero for now (no speaker encoder)
    // (already zeroed)

    // Slot 1..T_text: text embeddings
    ggml_backend_tensor_get(text_emb, result.data() + D, 0, (size_t)T_text * D * sizeof(float));

    // Slot T_text+1: BOS embedding
    ggml_backend_tensor_get(bos_out, result.data() + (1 + T_text) * D, 0, (size_t)D * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return result;
}

// Build and run a single-token semantic embedding graph for decode step.
// Returns (D,) float32.
static std::vector<float> embed_semantic_token(confucius4_tts_context* ctx, int32_t token_id, int sem_pos) {
    const auto& m = ctx->t2s;
    const int D = m.hp.model_dim;

    size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(64, false);
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0)
        return {};

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);

    ggml_tensor* tok = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(tok, "sem_tok");
    ggml_set_input(tok);

    ggml_tensor* emb = ggml_get_rows(ctx0, m.semantic_embed_w, tok);

    ggml_tensor* pos_id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(pos_id, "sem_pos");
    ggml_set_input(pos_id);

    ggml_tensor* pos = ggml_get_rows(ctx0, m.sem_pos_embed_w, pos_id);
    ggml_tensor* out = ggml_add(ctx0, emb, pos);
    ggml_set_name(out, "sem_emb_out");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return {};
    }

    ggml_backend_tensor_set(tok, &token_id, 0, sizeof(int32_t));
    int32_t pos_val = sem_pos;
    ggml_backend_tensor_set(pos_id, &pos_val, 0, sizeof(int32_t));

    ggml_backend_graph_compute(ctx->backend, gf);

    std::vector<float> result(D);
    ggml_backend_tensor_get(out, result.data(), 0, D * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return result;
}

// Run a single GPT-2 forward step via ggml_backend_sched.
// Input: (D, T) float embeddings. Output: last-token logits (semantic_vocab_size,).
static std::vector<float> run_gpt2_step(confucius4_tts_context* ctx, const float* input_emb, int T, int n_past) {
    const auto& hp = ctx->t2s.hp;
    const int D = hp.model_dim;

    // kv_self_attn creates ~40 intermediates per layer (QKV split, views, permutes,
    // cache writes, softmax, output permute, reshape) + FFN adds ~10.
    const int n_tensors = hp.num_layers * 50 + 64;
    size_t ctx_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(8192, false);
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0)
        return {};

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // Input embedding tensor
    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T);
    ggml_set_name(x, "gpt2_input");
    ggml_set_input(x);

    // Forward pass
    ggml_tensor* logits = gpt2_forward(ctx, ctx0, gf, x, ctx->kv.k, ctx->kv.v, n_past);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    // Allocate with sched (weights are already loaded)
    ggml_backend_sched_t sched = ggml_backend_sched_new(&ctx->backend, nullptr, 1, n_tensors, false, false);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "confucius4: GPT-2 graph alloc failed\n");
        ggml_backend_sched_free(sched);
        ggml_free(ctx0);
        return {};
    }

    ggml_backend_tensor_set(x, input_emb, 0, (size_t)D * T * sizeof(float));
    ggml_backend_sched_graph_compute(sched, gf);

    // Read last-token logits
    const int V = hp.semantic_vocab_size;
    std::vector<float> out_logits(V);
    size_t offset = (T > 1) ? (size_t)(T - 1) * V * sizeof(float) : 0;
    ggml_backend_tensor_get(logits, out_logits.data(), offset, V * sizeof(float));

    ggml_backend_sched_free(sched);
    ggml_free(ctx0);
    return out_logits;
}

static std::vector<int32_t> t2s_decode(confucius4_tts_context* ctx, const std::vector<int32_t>& text_token_ids) {
    const auto& hp = ctx->t2s.hp;
    const int T_text = (int)text_token_ids.size();
    const int prefix_len = 1 + T_text + 1; // condition(1) + text(T) + BOS(1)
    const int max_new =
        ctx->params.max_semantic_tokens > 0 ? ctx->params.max_semantic_tokens : hp.max_semantic_seq_lens;
    const int max_seq = prefix_len + max_new;
    const int vb = ctx->params.verbosity;

    if (vb >= 1)
        fprintf(stderr, "confucius4: T2S decode: text_len=%d, prefix_len=%d, max_new=%d\n", T_text, prefix_len,
                max_new);

    // Allocate KV cache
    kv_free(ctx->kv);
    if (!kv_init(ctx->kv, hp, max_seq, ctx->backend)) {
        fprintf(stderr, "confucius4: KV cache allocation failed\n");
        return {};
    }

    // Seed RNG
    std::mt19937 rng(ctx->params.seed ? ctx->params.seed : 42);

    // ── Step 1: Build prefix embedding via ggml graph ──
    std::vector<float> prefix_emb = build_prefix_embedding(ctx, text_token_ids);
    if (prefix_emb.empty()) {
        fprintf(stderr, "confucius4: prefix embedding failed\n");
        kv_free(ctx->kv);
        return {};
    }

    // ── Step 2: Prefill — run GPT-2 on the full prefix ──
    std::vector<float> logits = run_gpt2_step(ctx, prefix_emb.data(), prefix_len, 0);
    if (logits.empty()) {
        fprintf(stderr, "confucius4: prefill failed\n");
        kv_free(ctx->kv);
        return {};
    }

    if (vb >= 2)
        fprintf(stderr, "confucius4: prefill done, logits[0..3] = %.3f %.3f %.3f %.3f\n", logits[0], logits[1],
                logits[2], logits[3]);

    // ── Step 3: Autoregressive decode ──
    std::vector<int32_t> semantic_codes;
    int n_past = prefix_len;

    for (int step = 0; step < max_new; step++) {
        // Sample from logits
        int token = sample_top_p(logits.data(), hp.semantic_vocab_size, ctx->params.temperature, ctx->params.top_p,
                                 ctx->params.top_k, rng);

        // Check for EOS
        if (token == hp.stop_semantic_token) {
            if (vb >= 1)
                fprintf(stderr, "confucius4: EOS at step %d\n", step);
            break;
        }

        semantic_codes.push_back(token);

        // Embed the new token (semantic_embed + position)
        std::vector<float> tok_emb = embed_semantic_token(ctx, token, step + 1);
        if (tok_emb.empty()) {
            fprintf(stderr, "confucius4: token embedding failed at step %d\n", step);
            break;
        }

        // Run one GPT-2 step
        logits = run_gpt2_step(ctx, tok_emb.data(), 1, n_past);
        if (logits.empty()) {
            fprintf(stderr, "confucius4: decode step %d failed\n", step);
            break;
        }
        n_past++;

        if (vb >= 2 && step < 5)
            fprintf(stderr, "confucius4: step %d: token=%d\n", step, token);
    }

    if (vb >= 1)
        fprintf(stderr, "confucius4: generated %zu semantic codes\n", semantic_codes.size());

    kv_free(ctx->kv);
    return semantic_codes;
}

float* confucius4_tts_synthesize(confucius4_tts_context* ctx, const char* text, const char* lang, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;

    (void)lang;
    const int vb = ctx->params.verbosity;

    // Step 1: Tokenize text. For now, use a hardcoded test sequence.
    // TODO: wire SentencePiece tokenizer from companion file.
    std::vector<int32_t> text_ids;
    const char* env_ids = std::getenv("CRISPASR_CONFUCIUS4_TEXT_IDS");
    if (env_ids) {
        // Parse comma-separated token IDs from env var (testing path)
        std::string s(env_ids);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            if (comma == std::string::npos)
                comma = s.size();
            text_ids.push_back(std::stoi(s.substr(pos, comma - pos)));
            pos = comma + 1;
        }
        if (vb >= 1)
            fprintf(stderr, "confucius4: using %zu text IDs from CRISPASR_CONFUCIUS4_TEXT_IDS\n", text_ids.size());
    } else {
        fprintf(stderr, "confucius4: no tokenizer available yet. Set CRISPASR_CONFUCIUS4_TEXT_IDS=id1,id2,... "
                        "to test with pre-tokenized input.\n");
        *out_n_samples = 0;
        return nullptr;
    }

    // Step 2: T2S decode → semantic codes
    std::vector<int32_t> semantic_codes = t2s_decode(ctx, text_ids);
    if (semantic_codes.empty()) {
        if (vb >= 1)
            fprintf(stderr, "confucius4: T2S produced no semantic codes\n");
        *out_n_samples = 0;
        return nullptr;
    }

    // Steps 3-4: S2A flow-matching → mel → BigVGAN → PCM
    // TODO: implement S2A and vocoder
    fprintf(stderr, "confucius4: generated %zu semantic codes (S2A + vocoder not yet implemented)\n",
            semantic_codes.size());
    *out_n_samples = 0;
    return nullptr;
}

void confucius4_tts_pcm_free(float* pcm) {
    free(pcm);
}

void confucius4_tts_free(confucius4_tts_context* ctx) {
    if (!ctx)
        return;

    kv_free(ctx->kv);
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
