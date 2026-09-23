"""A fake Qwen3.5-shaped model, small enough to run on a login node.

Phase 3's modelling core (encoder / hypernet / diagnostics) has to be provable
without the 19 GB checkpoint: every `--selftest` in this package builds one of
these instead. The tree is byte-for-byte the one `targets.enumerate_sites`
predicts -- `model.language_model.layers.{i}.{self_attn|linear_attn|mlp}.{proj}`
-- so `inject.inject()` resolves every site by name and a shape mismatch here is
a real bug in `targets.py`, not a test artefact.

The forward pass is deliberately *not* a faithful Qwen3.5: no gated delta rule,
no RoPE, no QK-norm. It only has to (a) touch every projection so gradients
reach every injected site, and (b) produce a scalar loss. Anything more would be
re-implementing the model to test the harness.

Shape parity with the real 9B (verified 2026-08-06, config.json of
Qwen/Qwen3.5-9B, and see targets.py:89-128 for the derivations):

    real 9B                       tiny default
    hidden            4096        32
    q_proj  out       8192        64    (attn_output_gate doubles it)
    k/v_proj out      1024        16
    in_proj_qkv out   8192        64    (2*key_dim + value_dim)
    in_proj_z   out   4096        32    (value_dim)
    in_proj_b/a out     32         4    (linear_num_value_heads)
    out_proj  in      4096        32    (value_dim)
    intermediate     12288        64
    layers              32         4    (full_attention at 3,7,... / at 1,3)

Requires torch. Import from the `llamafactory` env python (paths.PY_TRAIN).
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import targets


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def tiny_text_config(n_layers=4, hidden=32, intermediate=64,
                     full_attention_interval=2, n_heads=4, head_dim=8,
                     n_kv_heads=2, attn_output_gate=True,
                     linear_num_key_heads=2, linear_key_head_dim=8,
                     linear_num_value_heads=4, linear_value_head_dim=8,
                     vocab_size=64):
    """A `text_config`-shaped dict `targets.enumerate_sites` accepts.

    `layer_types` is left implicit so `targets.layer_types` exercises its
    `full_attention_interval` fallback (targets.py:208-216), which is the branch
    a checkpoint without an explicit list would take.
    """
    return {
        "_model_type": "qwen3_5_toy",
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": n_layers,
        "full_attention_interval": full_attention_interval,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "attn_output_gate": bool(attn_output_gate),
        "linear_num_key_heads": linear_num_key_heads,
        "linear_key_head_dim": linear_key_head_dim,
        "linear_num_value_heads": linear_num_value_heads,
        "linear_value_head_dim": linear_value_head_dim,
        "vocab_size": vocab_size,
    }


class ToyConfig(object):
    """Just enough of a `PretrainedConfig` for `inject._text_config_from_model`."""

    def __init__(self, text_config):
        self.text_config = dict(text_config)
        self.model_type = text_config.get("_model_type", "qwen3_5_toy")
        self.vocab_size = int(text_config.get("vocab_size", 64))


# --------------------------------------------------------------------------
# Modules
# --------------------------------------------------------------------------

def _shape(cfg, module):
    """(d_in, d_out) for one module type, straight from targets' own table."""
    return targets._SHAPES[module]["shape_fn"](cfg)


class ToyAttention(nn.Module):
    """`self_attn`: q/k/v/o with the real widths, a cartoon of the math."""

    def __init__(self, cfg):
        super(ToyAttention, self).__init__()
        for m in ("q_proj", "k_proj", "v_proj", "o_proj"):
            d_in, d_out = _shape(cfg, m)
            setattr(self, m, nn.Linear(d_in, d_out, bias=False))
        self.n_rep = self.o_proj.in_features // self.v_proj.out_features

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        gate = torch.sigmoid(q.mean(-1, keepdim=True) + k.mean(-1, keepdim=True))
        v = v.repeat_interleave(self.n_rep, dim=-1)
        return self.o_proj(v * gate)


class ToyLinearAttention(nn.Module):
    """`linear_attn`: the five DeltaNet projections, state-free."""

    def __init__(self, cfg):
        super(ToyLinearAttention, self).__init__()
        for m in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
            d_in, d_out = _shape(cfg, m)
            setattr(self, m, nn.Linear(d_in, d_out, bias=False))
        self.key_dim = (self.in_proj_qkv.out_features - self.in_proj_z.out_features) // 2
        self.value_dim = self.in_proj_z.out_features

    def forward(self, x):
        qkv = self.in_proj_qkv(x)
        q = qkv[..., :self.key_dim]
        k = qkv[..., self.key_dim:2 * self.key_dim]
        v = qkv[..., 2 * self.key_dim:]
        z = self.in_proj_z(x)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        drift = (q.mean(-1, keepdim=True) + k.mean(-1, keepdim=True)
                 + b.mean(-1, keepdim=True) + a.mean(-1, keepdim=True))
        return self.out_proj(v * torch.sigmoid(z) + drift)


class ToyMLP(nn.Module):
    def __init__(self, cfg):
        super(ToyMLP, self).__init__()
        for m in ("gate_proj", "up_proj", "down_proj"):
            d_in, d_out = _shape(cfg, m)
            setattr(self, m, nn.Linear(d_in, d_out, bias=False))

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ToyDecoderLayer(nn.Module):
    def __init__(self, cfg, layer_type):
        super(ToyDecoderLayer, self).__init__()
        self.layer_type = layer_type
        if layer_type == targets.FULL_ATTENTION:
            self.self_attn = ToyAttention(cfg)
        else:
            self.linear_attn = ToyLinearAttention(cfg)
        self.mlp = ToyMLP(cfg)
        h = int(cfg["hidden_size"])
        self.norm1 = nn.LayerNorm(h)
        self.norm2 = nn.LayerNorm(h)

    def forward(self, x):
        mixer = self.self_attn if hasattr(self, "self_attn") else self.linear_attn
        x = x + mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ToyTextModel(nn.Module):
    def __init__(self, cfg):
        super(ToyTextModel, self).__init__()
        h = int(cfg["hidden_size"])
        self.embed_tokens = nn.Embedding(int(cfg["vocab_size"]), h)
        lts = targets.layer_types(cfg)
        self.layers = nn.ModuleList([ToyDecoderLayer(cfg, t) for t in lts])
        self.norm = nn.LayerNorm(h)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class ToyInner(nn.Module):
    """Stands in for `Qwen3_5Model`: `.visual` + `.language_model`."""

    def __init__(self, cfg):
        super(ToyInner, self).__init__()
        self.language_model = ToyTextModel(cfg)


class ToyOutput(object):
    __slots__ = ("loss", "logits")

    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


class TinyQwen35(nn.Module):
    """`Qwen3_5ForConditionalGeneration`-shaped root: `.model.language_model`."""

    def __init__(self, cfg):
        super(TinyQwen35, self).__init__()
        self.config = ToyConfig(cfg)
        self.text_config = dict(cfg)
        self.model = ToyInner(cfg)
        self.lm_head = nn.Linear(int(cfg["hidden_size"]),
                                 int(cfg["vocab_size"]), bias=False)

    def forward(self, input_ids, labels=None, **_ignored):
        """`**_ignored` swallows attention_mask/use_cache: callers written for
        the real model pass them, and a toy that rejected them would force
        every caller to branch on which model it is holding.

        The labels are SHIFTED by one, exactly as
        `transformers`' `*ForCausalLM` does. Without the shift this toy would
        compute a different loss from the model it stands in for, and
        `train_hypernet.weighted_token_loss` -- which reimplements the loss so
        E9 can weight it -- could not be checked against it.
        """
        h = self.model.language_model(input_ids)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1))
        return ToyOutput(loss, logits)


def tiny_model(cfg=None, seed=0):
    """A deterministic TinyQwen35 plus the config it was built from."""
    cfg = cfg or tiny_text_config()
    torch.manual_seed(seed)
    return TinyQwen35(cfg), cfg


def tiny_batch(cfg, batch=2, seq=5, seed=0):
    """(input_ids, labels) for the toy model."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, int(cfg["vocab_size"]), (batch, seq), generator=g)
    return ids, ids


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    model, cfg = tiny_model()
    sites = targets.enumerate_sites(cfg, "all")
    named = dict(model.named_modules())
    bad = []
    for s in sites:
        m = named.get(s.name)
        if m is None:
            bad.append((s.name, "missing"))
        elif (m.in_features, m.out_features) != (s.d_in, s.d_out):
            bad.append((s.name, "%dx%d != %dx%d" % (m.out_features, m.in_features,
                                                    s.d_out, s.d_in)))
    ids, labels = tiny_batch(cfg)
    out = model(ids, labels=labels)
    print("toy model: %d params, %d sites, loss=%.4f"
          % (sum(p.numel() for p in model.parameters()), len(sites),
             float(out.loss)))
    for name, why in bad:
        print("  MISMATCH %s: %s" % (name, why))
    if bad:
        raise SystemExit("toy tree does not match targets.enumerate_sites")
    print("OK")
