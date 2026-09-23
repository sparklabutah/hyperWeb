"""Injection-site catalogue for Qwen3.5 (adapterCL.md 4.5).

Qwen3.5 is a 3:1 Gated-DeltaNet / Gated-Attention hybrid, so "where do you put
the LoRA" is a real experimental axis rather than a default. This module derives
the full site list *analytically from config.json* -- no model weights, no GPU --
so target sets can be planned, costed and validated on a login node.

Verified against transformers 5.6.0 `models/qwen3_5/modeling_qwen3_5.py`
(llamafactory env) on 2026-08-06:

  Qwen3_5GatedDeltaNet  L356  in_proj_qkv, in_proj_z, in_proj_b, in_proj_a,
                              out_proj  (+ conv1d, which is nn.Conv1d)
  Qwen3_5Attention      L620  q_proj, k_proj, v_proj, o_proj
  Qwen3_5MLP            L695  gate_proj, up_proj, down_proj

Three corrections to adapterCL.md 4.5's table, all load-bearing:

* HF emits `in_proj_b` and `in_proj_a` as **separate** Linears -- they are not
  fused. vLLM fuses them as `in_proj_ba` and declares the mapping
  `in_proj_ba: [in_proj_b, in_proj_a]`, so PEFT-named adapters do resolve. The
  same holds for `in_proj_qkvz: [in_proj_qkv, in_proj_z]`.
* `conv1d` is an `nn.Conv1d`. vLLM's `get_supported_lora_modules` only collects
  `LinearBase` subclasses, so a LoRA on conv1d is silently dropped at serve
  time. It is excluded from every target set here.
* `q_proj` is **twice** the usual width (`num_attention_heads * head_dim * 2`)
  because `attn_output_gate: true` -- the projection emits gate and query
  concatenated, split downstream. q_norm/k_norm apply per-head after the
  reshape, so a LoRA on q_proj/k_proj is shape-safe; the CUBLAS failure mode
  4.5 warns about does not apply to this architecture.

Pure stdlib -- importable from the system python (3.6+).
"""

from __future__ import print_function

import collections
import json
import os

from . import paths

# --------------------------------------------------------------------------
# Module-type registry
# --------------------------------------------------------------------------

LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"
MLP = "mlp"

#: parent submodule name per layer type, as it appears in named_modules()
SUBMODULE = {
    LINEAR_ATTENTION: "linear_attn",
    FULL_ATTENTION: "self_attn",
    MLP: "mlp",
}

#: module-type -> (layer_type, submodule, shape_fn(cfg) -> (d_in, d_out))
#: cfg is the *text_config* dict from config.json.
_SHAPES = collections.OrderedDict()


def _reg(name, layer_type, shape_fn, vllm_packed=None, note=""):
    _SHAPES[name] = {
        "name": name,
        "layer_type": layer_type,
        "submodule": SUBMODULE[layer_type],
        "shape_fn": shape_fn,
        "vllm_packed": vllm_packed,   # the fused name vLLM expects, if any
        "note": note,
    }


def _h(cfg):
    return int(cfg["hidden_size"])


def _head_dim(cfg):
    return int(cfg.get("head_dim") or (_h(cfg) // int(cfg["num_attention_heads"])))


def _key_dim(cfg):
    return int(cfg["linear_num_key_heads"]) * int(cfg["linear_key_head_dim"])


def _value_dim(cfg):
    return int(cfg["linear_num_value_heads"]) * int(cfg["linear_value_head_dim"])


# -- full attention ---------------------------------------------------------
# q_proj is doubled by attn_output_gate (gate || query), see modeling L632-634.
_reg("q_proj", FULL_ATTENTION,
     lambda c: (_h(c), int(c["num_attention_heads"]) * _head_dim(c)
                * (2 if c.get("attn_output_gate") else 1)),
     vllm_packed="qkv_proj", note="output-gated: width x2")
_reg("k_proj", FULL_ATTENTION,
     lambda c: (_h(c), int(c["num_key_value_heads"]) * _head_dim(c)),
     vllm_packed="qkv_proj")
_reg("v_proj", FULL_ATTENTION,
     lambda c: (_h(c), int(c["num_key_value_heads"]) * _head_dim(c)),
     vllm_packed="qkv_proj")
_reg("o_proj", FULL_ATTENTION,
     lambda c: (int(c["num_attention_heads"]) * _head_dim(c), _h(c)),
     vllm_packed="o_proj")

# -- linear attention (Gated DeltaNet) --------------------------------------
_reg("in_proj_qkv", LINEAR_ATTENTION,
     lambda c: (_h(c), _key_dim(c) * 2 + _value_dim(c)),
     vllm_packed="in_proj_qkvz")
_reg("in_proj_z", LINEAR_ATTENTION,
     lambda c: (_h(c), _value_dim(c)),
     vllm_packed="in_proj_qkvz")
_reg("in_proj_b", LINEAR_ATTENTION,
     lambda c: (_h(c), int(c["linear_num_value_heads"])),
     vllm_packed="in_proj_ba", note="rank-32 output; LoRA here is nearly free")
_reg("in_proj_a", LINEAR_ATTENTION,
     lambda c: (_h(c), int(c["linear_num_value_heads"])),
     vllm_packed="in_proj_ba", note="rank-32 output; LoRA here is nearly free")
_reg("out_proj", LINEAR_ATTENTION,
     lambda c: (_value_dim(c), _h(c)),
     vllm_packed="out_proj")

# -- MLP --------------------------------------------------------------------
_reg("gate_proj", MLP, lambda c: (_h(c), int(c["intermediate_size"])),
     vllm_packed="gate_up_proj")
_reg("up_proj", MLP, lambda c: (_h(c), int(c["intermediate_size"])),
     vllm_packed="gate_up_proj")
_reg("down_proj", MLP, lambda c: (int(c["intermediate_size"]), _h(c)),
     vllm_packed="down_proj")

MODULE_TYPES = tuple(_SHAPES.keys())

#: nn.Conv1d, not a Linear. Never LoRA-able through PEFT and silently dropped by
#: vLLM's LinearBase-only support scan (vllm/lora/utils.py:219-240).
EXCLUDED_MODULES = ("conv1d",)

#: Vision-tower linears -- excluded from every target set (4.5: "filter on the
#: language_model.* prefix"). Listed so the survey can assert they were skipped.
VISION_MODULES = ("qkv", "proj", "linear_fc1", "linear_fc2")

#: Module types the *stable* index uses. Fixed order == stable module_ids for
#: the hypernetwork's module embedding, so a checkpoint stays loadable when a
#: target set changes.
MODULE_ID = dict((m, i) for i, m in enumerate(MODULE_TYPES))


# --------------------------------------------------------------------------
# Target sets (adapterCL.md 4.5 "Recommended target sets for the Phase 1 sweep")
# --------------------------------------------------------------------------

TARGET_SETS = collections.OrderedDict([
    ("attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
    ("attn_vo", ("v_proj", "o_proj")),
    ("deltanet", ("in_proj_qkv", "out_proj")),
    ("deltanet_full", ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                       "out_proj")),
    ("mlp", ("gate_proj", "up_proj", "down_proj")),
    ("attn_mlp", ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")),
    ("deltanet_mlp", ("in_proj_qkv", "out_proj",
                      "gate_proj", "up_proj", "down_proj")),
    ("all_mixers", ("q_proj", "k_proj", "v_proj", "o_proj",
                    "in_proj_qkv", "in_proj_z", "out_proj")),
    ("all", MODULE_TYPES),
])

#: Description shown in the sweep report; keep in sync with TARGET_SETS.
TARGET_SET_NOTE = {
    "attn": "attention only -- the S0-Tuning-comparable configuration (4.5.1)",
    "attn_vo": "v/o only -- the fallback if QK-norm ever breaks q/k (4.5 gotcha 2)",
    "deltanet": "the recurrent pathway (4.5.2)",
    "deltanet_full": "recurrent pathway incl. the cheap gate projections",
    "mlp": "the most numerous sites (4.5.3)",
    "attn_mlp": "attention + MLP",
    "deltanet_mlp": "recurrent + MLP",
    "all_mixers": "both token mixers, no MLP -- isolates 'mixing' from 'features'",
    "all": "every LoRA-able linear in the language model",
}

DEFAULT_TARGET_SET = "attn_mlp"


def target_set(name):
    if name not in TARGET_SETS:
        raise KeyError("unknown target set %r (have %r)"
                       % (name, list(TARGET_SETS)))
    return TARGET_SETS[name]


# --------------------------------------------------------------------------
# Config reading + site enumeration
# --------------------------------------------------------------------------

def load_text_config(model=None):
    """Return the text_config dict of a Qwen3.5 checkpoint (local or hub id)."""
    model = model or paths.BASE_MODEL
    d = paths.hf_snapshot(model) if "/" in model and not os.path.isdir(model) else model
    if d is None:
        raise IOError("model %r is not cached locally; pass a directory" % (model,))
    cfg_path = os.path.join(d, "config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    text = cfg.get("text_config", cfg)
    text.setdefault("_model_type", cfg.get("model_type"))
    text.setdefault("_vision_config", cfg.get("vision_config", {}))
    return text


def layer_types(cfg):
    """The explicit per-layer pattern. Falls back to full_attention_interval.

    A UNIFORM-ATTENTION backbone (Llama-3.1, and any other plain transformer)
    carries NEITHER key: it has no hybrid pattern to describe. The old
    `interval=4` default then labelled three of every four layers
    `linear_attention`, and `enumerate_sites` drops a mixer site whose layer
    type does not match -- so a Llama handle came out with a QUARTER of its
    q/k/v/o sites and `load_into_handle(strict=False)` silently skipped the
    rest. That is a wrong NLL, not an error. Only assume the Qwen3.5 hybrid
    pattern when the config actually names an interval.
    """
    lt = cfg.get("layer_types")
    if lt:
        return list(lt)
    n = int(cfg["num_hidden_layers"])
    if "full_attention_interval" not in cfg:
        return [FULL_ATTENTION] * n
    interval = int(cfg["full_attention_interval"])
    return [FULL_ATTENTION if (i + 1) % interval == 0 else LINEAR_ATTENTION
            for i in range(n)]


class Site(object):
    """One (layer, module) injection point.

    `layer_type` is the *layer's* type (what kind of token mixer this decoder
    layer has); `submodule` is the parent attribute the projection actually
    hangs off. They differ for MLP modules, which exist on every layer:
    layer 0 is a linear_attention layer but its gate_proj lives under `.mlp`.
    """

    __slots__ = ("layer", "layer_type", "module", "d_in", "d_out", "prefix",
                 "submodule")

    def __init__(self, layer, layer_type, module, d_in, d_out, prefix):
        self.layer = layer
        self.layer_type = layer_type
        self.module = module
        self.d_in = d_in
        self.d_out = d_out
        self.prefix = prefix
        self.submodule = SUBMODULE[_SHAPES[module]["layer_type"]]

    @property
    def name(self):
        """Full dotted path as it appears in named_modules()."""
        return "%s.layers.%d.%s.%s" % (self.prefix, self.layer,
                                       self.submodule, self.module)

    @property
    def rel_name(self):
        """Path relative to the language model root (prefix-independent key)."""
        return "layers.%d.%s.%s" % (self.layer, self.submodule, self.module)

    @property
    def module_id(self):
        return MODULE_ID[self.module]

    def n_params(self, rank):
        return rank * (self.d_in + self.d_out)

    def as_dict(self):
        return {"name": self.name, "rel_name": self.rel_name,
                "layer": self.layer, "layer_type": self.layer_type,
                "module": self.module, "submodule": self.submodule,
                "d_in": self.d_in, "d_out": self.d_out,
                "module_id": self.module_id}

    def __repr__(self):
        return "Site(%s, %dx%d)" % (self.name, self.d_out, self.d_in)


#: Module path prefix for the language model inside Qwen3_5ForConditionalGeneration.
LM_PREFIX = "model.language_model"


def enumerate_sites(cfg, modules=None, layers=None, prefix=LM_PREFIX):
    """All (layer, module) injection points for a target set.

    `modules` is a target-set name or an explicit sequence of module types.
    `layers` optionally restricts to a set of layer indices.
    """
    if modules is None:
        modules = TARGET_SETS[DEFAULT_TARGET_SET]
    elif isinstance(modules, str):
        modules = target_set(modules)
    modules = tuple(modules)
    for m in modules:
        if m in EXCLUDED_MODULES:
            raise ValueError(
                "%r cannot carry a LoRA: it is an nn.Conv1d and vLLM's "
                "LinearBase-only support scan drops it silently" % (m,))
        if m not in _SHAPES:
            raise KeyError("unknown module type %r" % (m,))

    lts = layer_types(cfg)
    sites = []
    for i, lt in enumerate(lts):
        if layers is not None and i not in layers:
            continue
        for m in modules:
            spec = _SHAPES[m]
            # An MLP site exists on every layer; a mixer site only on layers of
            # the matching type.
            if spec["layer_type"] != MLP and spec["layer_type"] != lt:
                continue
            d_in, d_out = spec["shape_fn"](cfg)
            sites.append(Site(i, lt, m, d_in, d_out, prefix))
    return sites


def site_budget(sites, rank):
    """Trainable-parameter count for a plain LoRA over these sites."""
    return sum(s.n_params(rank) for s in sites)


def hypernet_output_dim(sites, rank):
    """Free-generation output dimensionality -- the number 6.1 warns about."""
    return site_budget(sites, rank)


def peft_target_modules(modules):
    """`lora_target` value for LLaMA-Factory / PEFT.

    PEFT matches on the module *suffix*, so the bare names are correct and are
    exactly what vLLM's packed_modules_mapping expects to unpack.
    """
    if isinstance(modules, str):
        modules = target_set(modules)
    return list(modules)


def vllm_expected_names(modules):
    """The fused names vLLM will resolve each PEFT module name into."""
    if isinstance(modules, str):
        modules = target_set(modules)
    return dict((m, _SHAPES[m]["vllm_packed"]) for m in modules)


def summarize(cfg=None, rank=16, fh=None):
    """Print the architecture survey + a target-set cost table."""
    import sys
    fh = fh or sys.stdout
    cfg = cfg or load_text_config()
    lts = layer_types(cfg)
    counts = collections.Counter(lts)
    n_layers = len(lts)
    print("model_type=%s  layers=%d  hidden=%d  intermediate=%d"
          % (cfg.get("_model_type"), n_layers, _h(cfg), cfg["intermediate_size"]),
          file=fh)
    print("layer_types: %s"
          % ", ".join("%s=%d" % (k, v) for k, v in sorted(counts.items())), file=fh)
    full_idx = [i for i, t in enumerate(lts) if t == FULL_ATTENTION]
    print("full_attention layers: %s" % (full_idx,), file=fh)
    print("", file=fh)
    print("module shapes (d_out x d_in):", file=fh)
    for m, spec in _SHAPES.items():
        d_in, d_out = spec["shape_fn"](cfg)
        n = counts[spec["layer_type"]] if spec["layer_type"] != MLP else n_layers
        print("  %-12s %-18s x%-3d  %6d x %-6d  vllm:%-14s %s"
              % (m, spec["layer_type"], n, d_out, d_in,
                 spec["vllm_packed"], spec["note"]), file=fh)
    print("  %-12s %-18s      (excluded: nn.Conv1d, dropped by vLLM)"
          % ("conv1d", LINEAR_ATTENTION), file=fh)
    print("", file=fh)
    print("target sets at rank=%d:" % rank, file=fh)
    print("  %-14s %6s %14s   %s" % ("name", "sites", "params", "note"), file=fh)
    for name in TARGET_SETS:
        sites = enumerate_sites(cfg, name)
        print("  %-14s %6d %14s   %s"
              % (name, len(sites), "{:,}".format(site_budget(sites, rank)),
                 TARGET_SET_NOTE[name]), file=fh)
    return cfg


if __name__ == "__main__":
    summarize()
