#!/usr/bin/env python
"""Per-stage parity for the Confucius4 S2A stage.

Drives the real PyTorch S2A (confuciustts/flow) on exactly the inputs the C++
runtime dumped (CRISPASR_CONFUCIUS4_DUMP_S2A=<dir>), including the identical
initial noise, and compares the length-regulator output and the final mel.

Usage:
  python s2a_parity.py --dump-dir <dir> --s2a-ckpt <s2a_model.pt> [--cfg 0.7] [--steps 25]
"""
import argparse, os, sys
import numpy as np
import torch

# The Python blueprint (github.com/netease-youdao/Confucius4-TTS) is not vendored;
# point --ref-repo at a clone of it.
def _import_ref(ref_repo):
    sys.path.insert(0, ref_repo)
    from confuciustts.flow.flow import MaskedDiffWithXvec, MaskedDiffWithXvecConfig
    return MaskedDiffWithXvec, MaskedDiffWithXvecConfig


def load_shapes(d):
    shp = {}
    with open(os.path.join(d, "shapes.txt")) as f:
        for line in f:
            if not line.strip():
                continue
            name, dims = line.rstrip("\n").split("\t")
            shp[name] = tuple(int(x) for x in dims.split(","))
    return shp


def load(d, name, shp, dtype=np.float32):
    a = np.fromfile(os.path.join(d, name + ".bin"), dtype=dtype)
    return a.reshape(shp[name])


def cmp(tag, mine, ref):
    mine = np.asarray(mine, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    if mine.shape != ref.shape:
        print(f"{tag:22s} SHAPE MISMATCH mine={mine.shape} ref={ref.shape}")
        return
    nm, nr = np.linalg.norm(mine), np.linalg.norm(ref)
    cos = float(mine @ ref / (nm * nr + 1e-12))
    # magnitudes printed next to cosine: cosine is scale-blind
    print(f"{tag:22s} cos={cos:.6f}  |mine|={nm:12.4f}  |ref|={nr:12.4f}  "
          f"ratio={nm/(nr+1e-12):7.4f}  max_abs_diff={np.abs(mine-ref).max():.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--s2a-ckpt", required=True)
    ap.add_argument("--ref-repo", required=True,
                    help="clone of github.com/netease-youdao/Confucius4-TTS")
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--steps", type=int, default=25)
    a = ap.parse_args()

    MaskedDiffWithXvec, MaskedDiffWithXvecConfig = _import_ref(a.ref_repo)

    d = a.dump_dir
    shp = load_shapes(d)
    print("dumped stages:", {k: v for k, v in shp.items()})

    codes = load(d, "semantic_codes_i32", shp, np.int32)
    lm_latent = load(d, "lm_latent", shp)
    z_init = load(d, "z_init", shp)          # (T_mel, mel_dim) row-major
    cond_cpp = load(d, "cond", shp)          # (T_mel, 512)
    mel_cpp = load(d, "mel", shp)            # (T_mel, mel_dim)

    T_sem = codes.shape[0]
    T_mel, mel_dim = z_init.shape
    print(f"T_sem={T_sem} T_mel={T_mel} mel_dim={mel_dim}")

    # Build from the SHIPPED config, not the dataclass defaults: the checkpoint
    # has estimator_mlp_ratio=3.0 (ff 1536, not 2048), and cfm_t_scheduler stays
    # at its dataclass default "linear" -- ConditionalCFM's own signature says
    # "cosine", but flow.py passes the config value.
    import yaml
    cfg_path = os.path.join(a.ref_repo, "config", "inference_config.yaml")
    s2a_cfg = yaml.safe_load(open(cfg_path))["s2a_model"]
    cfg = MaskedDiffWithXvecConfig(**s2a_cfg)
    print(f"config: ff_intermediate={int(cfg.estimator_hidden_dim * cfg.estimator_mlp_ratio)} "
          f"t_scheduler={cfg.cfm_t_scheduler!r} cfg_rate={cfg.cfm_inference_cfg_rate}")
    model = MaskedDiffWithXvec(cfg)
    state = torch.load(a.s2a_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=True)
    model.eval()

    sem = torch.from_numpy(codes.astype(np.int64)).unsqueeze(0)
    lat = torch.from_numpy(lm_latent).unsqueeze(0)

    with torch.no_grad():
        # --- stage 1: conditioning (encoder_proj + InterpolateRegulator) ---
        semantic_emb = model.input_embedding(sem).transpose(1, 2)
        text_cond = model.encoder_proj(torch.cat([lat, semantic_emb], dim=-1))
        cond_ref, _ = model.length_regulator(text_cond, torch.tensor([T_mel]))
        cmp("cond (regulator)", cond_cpp, cond_ref[0].numpy())

        # --- stage 2: the Euler ODE, driven on the C++ noise ---
        # mirrors ConditionalCFM.solve_euler with prompt_len == 0
        dec = model.decoder
        x = torch.from_numpy(z_init.T.copy()).unsqueeze(0)   # (1, mel_dim, T_mel)
        mu = cond_ref
        spks = torch.zeros(1, model.spk_embed_dim)
        mask = torch.ones(1, T_mel, dtype=torch.bool)
        prompt_x = torch.zeros_like(x)

        t_span = torch.linspace(0, 1, a.steps + 1)
        if cfg.cfm_t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        t, dt = t_span[0], t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            if a.cfg > 0:
                x_in = torch.cat([x, x], 0)
                p_in = torch.cat([prompt_x, torch.zeros_like(prompt_x)], 0)
                mu_in = torch.cat([mu, torch.zeros_like(mu)], 0)
                t_in = torch.full((2,), float(t))
                s_in = torch.cat([spks, torch.zeros_like(spks)], 0)
                m_in = torch.cat([mask, mask], 0)
                v = dec.estimator(x_in, m_in, mu_in, t_in, s_in, p_in)
                v_c, v_u = torch.split(v, [1, 1], dim=0)
                v = (1.0 + a.cfg) * v_c - a.cfg * v_u
            else:
                v = dec.estimator(x, mask, mu, torch.tensor([float(t)]), spks, prompt_x)
            x = x + dt * v
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            print(f"  step {step:2d}/{a.steps} t={float(t):.4f} |v|max={v.abs().max():.4f}", flush=True)

        mel_ref = x[0].numpy().T   # back to (T_mel, mel_dim) row-major
        cmp("mel (final)", mel_cpp, mel_ref)
        np.save(os.path.join(d, "mel_ref.npy"), mel_ref)
        print("wrote", os.path.join(d, "mel_ref.npy"))


if __name__ == "__main__":
    main()
