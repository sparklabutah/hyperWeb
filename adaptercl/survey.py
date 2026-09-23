"""Architecture survey of Qwen3.5-9B (adapterCL.md 8.2, 7 Phase 0, 11.2).

8.2 says to do this first: "Read `config.layer_types` and count `full_attention`
layers; enumerate all `nn.Linear` modules". 11.2 asks the question this module
exists to answer in one line -- *how many full-attention layers does Qwen3.5-9B
have?* -- because if the answer is small, attention-only injection has too few
sites to carry UI-specific behaviour (4.5, risk register row 4).

Three entry points, in increasing cost:

* `survey_from_config()`  -- pure arithmetic off config.json via `targets`.
  No torch, no weights, no GPU. Runs on a login node in milliseconds.
* `survey_from_model()`   -- instantiates the real model on the **meta device**
  (`accelerate.init_empty_weights`), so every module exists but no storage is
  allocated: ~5 s and ~0 bytes of model RAM for the 9B. Walks `named_modules()`,
  collects every `nn.Linear`/`nn.Conv1d`, and **cross-checks the result against
  `targets.enumerate_sites`**. That cross-check is the guard that keeps
  targets.py honest against the checkpoint: any name predicted-but-absent,
  present-but-unpredicted, or with a mismatched shape is reported.
* `vllm_supportability()` -- which module types survive the trip through vLLM's
  LoRA loader, and which are dropped silently. 4.5 calls this a
  "silent-failure class"; 9 grades it "High, and insidious".

The vLLM finding, verified against the vllm_q35 env rather than taken from the
plan's table: `get_supported_lora_modules` (vllm/lora/utils.py:219-240) collects
`name.split(".")[-1]` for every module that `isinstance(module, (LinearBase,))`
(plus MoERunner and embedding modules). `conv1d` on the DeltaNet layers is an
`nn.Conv1d`, never a `LinearBase`, so **it can never appear in that set** --
regardless of the fact that adapterCL.md 4.5 lists `conv1d` in the reported
supported-target list. Do not target it.

Needs torch + transformers + accelerate: run under `paths.PY_TRAIN`
(llamafactory env, python 3.11, transformers 5.6.0, torch 2.12.1, accelerate
1.11.0). `--config-only` drops the torch dependency entirely.

Run:  cd adapter_project && $PY_TRAIN -m adaptercl.survey
      cd adapter_project && python3 -m adaptercl.survey --config-only
"""

from __future__ import print_function

import argparse
import collections
import json
import os
import re
import sys

from . import cells, paths, targets

#: Ranks costed in the config survey. 4.5's reference point is a Qwen3.5-4B
#: paper hitting 4.7M params at rank 24 over 8 attention layers.
RANKS = (4, 8, 16, 32, 64)

#: The supported-target list adapterCL.md 4.5 reports for Qwen3.5. Recorded so
#: the survey can flag where it disagrees with what vLLM actually does.
PLAN_REPORTED_VLLM_TARGETS = (
    "conv1d", "down_proj", "gate_up_proj", "in_proj_ba", "in_proj_qkv",
    "in_proj_z", "linear_fc1", "linear_fc2", "o_proj", "out_proj", "proj",
    "qkv", "qkv_proj",
)

_VLLM_LINEARBASE_CITE = (
    "nn.Conv1d -- vLLM's get_supported_lora_modules only collects LinearBase, "
    "vllm/lora/utils.py:219-240"
)
_VISION_EXCLUDED_CITE = "vision tower excluded per 8.2"


def model_slug(model=None):
    """Filesystem-safe name for a model id or directory."""
    model = model or paths.BASE_MODEL
    if os.path.isdir(model):
        # .../models--Qwen--Qwen3.5-9B/snapshots/<sha> -> Qwen3.5-9B
        m = re.search(r"models--([^/]+)", model)
        model = m.group(1).replace("--", "/") if m else os.path.basename(model)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")


# --------------------------------------------------------------------------
# 1. Pure-config survey (8.2)
# --------------------------------------------------------------------------

def survey_from_config(model=None, ranks=RANKS):
    """Layer types, module shapes, site counts and LoRA budgets -- no weights.

    Everything here is arithmetic on config.json, so it is the cheapest way to
    answer 11.2 and to cost a target set before committing GPU time to it.
    """
    model = model or paths.BASE_MODEL
    cfg = targets.load_text_config(model)
    lts = targets.layer_types(cfg)
    counts = collections.Counter(lts)
    full_idx = [i for i, t in enumerate(lts) if t == targets.FULL_ATTENTION]
    lin_idx = [i for i, t in enumerate(lts) if t == targets.LINEAR_ATTENTION]

    module_shapes = collections.OrderedDict()
    for m in targets.MODULE_TYPES:
        spec = targets._SHAPES[m]
        d_in, d_out = spec["shape_fn"](cfg)
        n_layers = (len(lts) if spec["layer_type"] == targets.MLP
                    else counts[spec["layer_type"]])
        module_shapes[m] = {
            "layer_type": spec["layer_type"],
            "submodule": spec["submodule"],
            "d_in": d_in, "d_out": d_out,
            "n_layers": n_layers,
            "n_weights": d_in * d_out * n_layers,
            "vllm_packed": spec["vllm_packed"],
            "note": spec["note"],
            "module_id": targets.MODULE_ID[m],
        }

    ts = collections.OrderedDict()
    for name in targets.TARGET_SETS:
        sites = targets.enumerate_sites(cfg, name)
        by_layer_type = collections.Counter(s.layer_type for s in sites)
        ts[name] = {
            "modules": list(targets.TARGET_SETS[name]),
            "note": targets.TARGET_SET_NOTE[name],
            "n_sites": len(sites),
            "sites_by_layer_type": dict(by_layer_type),
            "params": dict((str(r), targets.site_budget(sites, r))
                           for r in ranks),
            "serving": target_set_serving_risks(name),
        }

    vis = cfg.get("_vision_config") or {}
    return {
        "model": model,
        "model_dir": paths.model_dir_or_id(model),
        "model_type": cfg.get("_model_type"),
        "n_layers": len(lts),
        "hidden_size": int(cfg["hidden_size"]),
        "intermediate_size": int(cfg["intermediate_size"]),
        "head_dim": targets._head_dim(cfg),
        "num_attention_heads": int(cfg["num_attention_heads"]),
        "num_key_value_heads": int(cfg["num_key_value_heads"]),
        "attn_output_gate": bool(cfg.get("attn_output_gate")),
        "full_attention_interval": cfg.get("full_attention_interval"),
        "layer_types": lts,
        "layer_type_counts": dict(counts),
        "n_full_attention": len(full_idx),
        "full_attention_layers": full_idx,
        "n_linear_attention": len(lin_idx),
        "linear_attention_layers": lin_idx,
        "module_shapes": module_shapes,
        "target_sets": ts,
        "ranks": list(ranks),
        "vision": {
            "depth": vis.get("depth"),
            "hidden_size": vis.get("hidden_size"),
            "out_hidden_size": vis.get("out_hidden_size"),
            "patch_size": vis.get("patch_size"),
            "spatial_merge_size": vis.get("spatial_merge_size"),
            "excluded_reason": _VISION_EXCLUDED_CITE,
        },
    }


# --------------------------------------------------------------------------
# 2. Real-module survey on the meta device (8.2's "enumerate all nn.Linear")
# --------------------------------------------------------------------------

def _require(mod_name, hint):
    try:
        return __import__(mod_name)
    except ImportError as e:
        raise RuntimeError(
            "survey_from_model needs %s (%s).\n"
            "  Run this module under %s, or pass --config-only to stay on the "
            "pure-config path.\n  underlying error: %s"
            % (mod_name, hint, paths.PY_TRAIN, e))


def _detect_lm_prefix(names):
    """Derive the language-model prefix from observed names rather than trusting
    `targets.LM_PREFIX`; the wrapper class differs between AutoModel classes and
    a wrong prefix would make the whole cross-check look like a total mismatch."""
    for n in names:
        m = re.match(r"^(.*)\.layers\.\d+\.(?:self_attn|linear_attn|mlp)\.", n)
        if m:
            return m.group(1)
    return targets.LM_PREFIX


def _instantiate_meta(cfg):
    """Empty (meta-device) model + the AutoModel class that produced it."""
    import transformers
    from accelerate import init_empty_weights
    errors = []
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM",
                     "AutoModel"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            with init_empty_weights():
                return cls.from_config(cfg), cls_name
        except Exception as e:            # noqa: BLE001 - report, then try next
            errors.append("%s: %s: %s" % (cls_name, type(e).__name__,
                                          str(e)[:200]))
    raise RuntimeError(
        "could not instantiate %s on the meta device with any AutoModel "
        "class.\n  %s" % (cfg.__class__.__name__, "\n  ".join(errors)))


def survey_from_model(model=None, device="meta", ranks=RANKS):
    """Walk the real module tree and cross-check it against `targets`.

    `device="meta"` is the only mode that is free; anything else allocates real
    storage for a 9B model and is refused unless you ask for it explicitly.
    """
    model = model or paths.BASE_MODEL
    if device != "meta":
        raise ValueError(
            "device=%r would allocate real storage for the whole model. This "
            "survey only needs the module tree; use device='meta'." % (device,))
    _require("torch", "the base tensor library")
    _require("transformers", "model definitions")
    _require("accelerate", "init_empty_weights, so no weights are allocated")

    import torch.nn as nn
    from transformers import AutoConfig

    model_dir = paths.model_dir_or_id(model)
    hf_cfg = AutoConfig.from_pretrained(model_dir)
    net, auto_class = _instantiate_meta(hf_cfg)

    observed = collections.OrderedDict()
    for name, mod in net.named_modules():
        if isinstance(mod, nn.Linear):
            w = tuple(mod.weight.shape)          # (d_out, d_in)
            observed[name] = {"cls": "Linear", "shape": list(w),
                              "d_in": w[1], "d_out": w[0],
                              "bias": mod.bias is not None}
        elif isinstance(mod, nn.Conv1d):
            w = tuple(mod.weight.shape)          # (out_ch, in_ch/groups, k)
            observed[name] = {"cls": "Conv1d", "shape": list(w),
                              "d_in": w[1], "d_out": w[0],
                              "kernel_size": mod.kernel_size[0],
                              "groups": mod.groups,
                              "bias": mod.bias is not None}

    prefix = _detect_lm_prefix(list(observed))
    lm, vision, other = (collections.OrderedDict(), collections.OrderedDict(),
                         collections.OrderedDict())
    for name, rec in observed.items():
        if name.startswith(prefix + "."):
            lm[name] = rec
        elif name == "lm_head" or name.endswith(".lm_head"):
            other[name] = rec
        else:
            vision[name] = rec

    text_cfg = targets.load_text_config(model)
    check = cross_check(text_cfg, lm, prefix=prefix)

    lm_leaf = collections.Counter(n.split(".")[-1] for n in lm)
    vis_leaf = collections.Counter(n.split(".")[-1] for n in vision)

    return {
        "model": model,
        "model_dir": model_dir,
        "device": device,
        "auto_class": auto_class,
        "torch_version": __import__("torch").__version__,
        "transformers_version": __import__("transformers").__version__,
        "lm_prefix": prefix,
        "n_modules_total": len(observed),
        "n_lm": len(lm),
        "n_vision": len(vision),
        "n_other": len(other),
        "lm_leaf_counts": dict(lm_leaf),
        "vision_leaf_counts": dict(vis_leaf),
        "other_modules": collections.OrderedDict(
            (k, v) for k, v in other.items()),
        "lm_modules": lm,
        "vision_modules": vision,
        "cross_check": check,
    }


def cross_check(cfg, observed_lm, prefix=None):
    """Compare `targets.enumerate_sites(cfg, "all")` against the real tree.

    This is the guard 8.2 implies ("Exact names vary with transformers version
    and fusion choices -- read, don't assume"). Three failure modes are reported
    separately because they mean different things:

      predicted_missing   -- targets.py invents a site that does not exist;
                             every LoRA on it would be a no-op.
      unpredicted         -- a real LoRA-able Linear that no target set can
                             reach; a blind spot in the injection-site sweep.
      shape_mismatch      -- the site exists but the hypernetwork would emit
                             factors of the wrong size (this is what the
                             `attn_output_gate` doubling of q_proj would break).
    """
    prefix = prefix or targets.LM_PREFIX
    sites = targets.enumerate_sites(cfg, "all", prefix=prefix)
    predicted = collections.OrderedDict((s.name, s) for s in sites)

    missing, mismatch = [], []
    for name, site in predicted.items():
        rec = observed_lm.get(name)
        if rec is None:
            missing.append(name)
            continue
        if rec["cls"] != "Linear":
            mismatch.append({"name": name, "problem": "not an nn.Linear",
                             "observed_cls": rec["cls"]})
            continue
        if (rec["d_in"], rec["d_out"]) != (site.d_in, site.d_out):
            mismatch.append({
                "name": name, "problem": "shape",
                "predicted": [site.d_in, site.d_out],
                "observed": [rec["d_in"], rec["d_out"]]})

    unpredicted = []
    for name, rec in observed_lm.items():
        if name in predicted:
            continue
        leaf = name.split(".")[-1]
        unpredicted.append({
            "name": name, "cls": rec["cls"], "leaf": leaf,
            "shape": rec["shape"],
            "expected_exclusion": (
                _VLLM_LINEARBASE_CITE if leaf in targets.EXCLUDED_MODULES
                else None),
        })

    unexplained = [u for u in unpredicted if u["expected_exclusion"] is None]
    return {
        "prefix": prefix,
        "n_predicted": len(predicted),
        "n_observed_lm": len(observed_lm),
        "predicted_missing": missing,
        "unpredicted": unpredicted,
        "unpredicted_unexplained": unexplained,
        "shape_mismatch": mismatch,
        "ok": not missing and not mismatch and not unexplained,
    }


# --------------------------------------------------------------------------
# 3. vLLM serve-time supportability (4.5 gotcha 1, risk register row 2)
# --------------------------------------------------------------------------

def _packed_groups():
    """fused vLLM name -> the PEFT module names it unpacks into."""
    groups = collections.OrderedDict()
    for m in targets.MODULE_TYPES:
        fused = targets._SHAPES[m]["vllm_packed"]
        groups.setdefault(fused, []).append(m)
    return groups


PACKED_GROUPS = _packed_groups()


def vllm_supportability(modules=None):
    """Per-module-type serve-time verdict: fused name + LoRA-able or not.

    `modules` defaults to everything a target set could ever mention plus the
    two families that must be reported UNSUPPORTED: `conv1d` and the vision
    tower's linears.
    """
    if modules is None:
        modules = (tuple(targets.MODULE_TYPES) + tuple(targets.EXCLUDED_MODULES)
                   + tuple(targets.VISION_MODULES))
    elif isinstance(modules, str):
        modules = targets.target_set(modules)

    out = collections.OrderedDict()
    for m in modules:
        if m in targets.EXCLUDED_MODULES:
            out[m] = {
                "family": "linear_attention",
                "vllm_fused_name": None,
                "supported": False,
                "reason": _VLLM_LINEARBASE_CITE,
                "in_plan_reported_list": m in PLAN_REPORTED_VLLM_TARGETS,
                "group_members": [],
            }
            continue
        if m in targets.VISION_MODULES and m not in targets._SHAPES:
            out[m] = {
                "family": "vision",
                "vllm_fused_name": m,
                "supported": False,
                "reason": _VISION_EXCLUDED_CITE,
                "in_plan_reported_list": m in PLAN_REPORTED_VLLM_TARGETS,
                "group_members": [],
            }
            continue
        spec = targets._SHAPES[m]
        fused = spec["vllm_packed"]
        members = PACKED_GROUPS[fused]
        out[m] = {
            "family": spec["layer_type"],
            "submodule": spec["submodule"],
            "vllm_fused_name": fused,
            "supported": True,
            "reason": ("PEFT name %r resolves through vLLM's "
                       "packed_modules_mapping to %r" % (m, fused)),
            "in_plan_reported_list": fused in PLAN_REPORTED_VLLM_TARGETS,
            "group_members": list(members),
            "group_is_packed": len(members) > 1,
        }
    return out


def target_set_serving_risks(modules):
    """Flag target sets that target *part* of a packed vLLM group.

    4.5 reports "crashes for partial LoRA on the DeltaNet group (`in_proj_qkv`
    without `in_proj_z`) in `expand_packed_lora`". The same shape of problem
    applies to any fused group -- q/k/v collapse into a single `qkv_proj`
    weight at serve time, so a LoRA on `v_proj` alone has to be expanded into a
    slice of a tensor whose other slices have no adapter.
    """
    if isinstance(modules, str):
        modules = targets.target_set(modules)
    targeted = set(modules)
    risks = []
    for fused, members in PACKED_GROUPS.items():
        if len(members) < 2:
            continue
        present = [m for m in members if m in targeted]
        if present and len(present) < len(members):
            risks.append({
                "fused": fused,
                "targeted": present,
                "missing": [m for m in members if m not in targeted],
                "risk": "partial_packed_group",
                "reason": ("vLLM fuses %s into %r; a LoRA on only %s must be "
                           "expanded into a slice of that fused weight "
                           "(expand_packed_lora). 4.5 reports crashes for "
                           "exactly this on the DeltaNet group."
                           % (members, fused, present)),
            })
    return risks


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_config_report(s, fh=None, rank=16):
    fh = fh or sys.stdout
    p = lambda x="": print(x, file=fh)
    p("=" * 78)
    p("adapterCL architecture survey (8.2) -- %s" % s["model"])
    p("=" * 78)
    p("model_type=%s  layers=%d  hidden=%d  intermediate=%d  head_dim=%d"
      % (s["model_type"], s["n_layers"], s["hidden_size"],
         s["intermediate_size"], s["head_dim"]))
    p("heads: %d attention / %d kv   attn_output_gate=%s   "
      "full_attention_interval=%s"
      % (s["num_attention_heads"], s["num_key_value_heads"],
         s["attn_output_gate"], s["full_attention_interval"]))
    p("")
    # --- the one line 11.2 asks for --------------------------------------
    p("ANSWER TO 11.2: %s has %d full_attention layers out of %d "
      "(%.0f%%), at indices %s."
      % (s["model"], s["n_full_attention"], s["n_layers"],
         100.0 * s["n_full_attention"] / s["n_layers"],
         s["full_attention_layers"]))
    p("  The other %d layers are linear_attention (Gated DeltaNet)."
      % s["n_linear_attention"])
    attn = s["target_sets"]["attn"]
    p("  Consequence for 4.5: attention-only injection gives %d sites "
      "(%s params at rank %d)."
      % (attn["n_sites"], "{:,}".format(attn["params"][str(rank)]), rank))
    p("")

    p("module shapes (per layer, d_out x d_in)")
    p("  %-12s %-17s %5s %16s %14s  %s"
      % ("module", "layer type", "count", "shape", "vllm fused", "note"))
    for m, r in s["module_shapes"].items():
        p("  %-12s %-17s %5d %16s %14s  %s"
          % (m, r["layer_type"], r["n_layers"],
             "%d x %d" % (r["d_out"], r["d_in"]), r["vllm_packed"], r["note"]))
    p("  %-12s %-17s %5d %16s %14s  %s"
      % ("conv1d", "linear_attention", s["n_linear_attention"], "-", "-",
         "EXCLUDED: " + _VLLM_LINEARBASE_CITE))
    p("")

    p("target sets: site counts and plain-LoRA parameter budgets")
    hdr = "  %-14s %6s %-18s" % ("name", "sites", "by layer type")
    hdr += "".join("%12s" % ("r=%d" % r) for r in s["ranks"]) + "  serving"
    p(hdr)
    for name, r in s["target_sets"].items():
        blt = ",".join("%s=%d" % (k[:4], v)
                       for k, v in sorted(r["sites_by_layer_type"].items()))
        row = "  %-14s %6d %-18s" % (name, r["n_sites"], blt)
        row += "".join("%12s" % "{:,}".format(r["params"][str(k)])
                       for k in s["ranks"])
        row += "  " + ("RISK" if r["serving"] else "ok")
        p(row)
    for name, r in s["target_sets"].items():
        for risk in r["serving"]:
            p("  ! %-12s partial packed group %-14s targets %s, missing %s"
              % (name, risk["fused"], risk["targeted"], risk["missing"]))
    p("")
    for name, r in s["target_sets"].items():
        p("  %-14s %s" % (name, r["note"]))
    p("")
    v = s["vision"]
    p("vision tower (excluded from every target set -- %s)" % v["excluded_reason"])
    p("  depth=%s hidden=%s out_hidden=%s patch=%s spatial_merge=%s"
      % (v["depth"], v["hidden_size"], v["out_hidden_size"], v["patch_size"],
         v["spatial_merge_size"]))
    p("  4.2 uses this tower frozen as the conditioning encoder, so it must not")
    p("  also be adapted -- otherwise the conditioning signal moves with the")
    p("  policy and the 'appearance determines parameters' claim is circular.")
    return s


def print_vllm_report(sup, fh=None):
    fh = fh or sys.stdout
    p = lambda x="": print(x, file=fh)
    p("")
    p("vLLM serve-time LoRA supportability (4.5 gotcha 1; risk register row 2)")
    p("  %-12s %-17s %-15s %-11s %s"
      % ("module", "family", "vllm fused", "supported", "note"))
    for m, r in sup.items():
        note = r["reason"] if not r["supported"] else (
            "packed with %s" % ",".join(x for x in r["group_members"] if x != m)
            if r.get("group_is_packed") else "1:1")
        p("  %-12s %-17s %-15s %-11s %s"
          % (m, r["family"], r["vllm_fused_name"] or "-",
             "yes" if r["supported"] else "NO", note))
    p("")
    p("  NOTE adapterCL.md 4.5 lists `conv1d` in the reported vLLM supported-")
    p("       target list. That list cannot be right for a LoRA: %s."
      % _VLLM_LINEARBASE_CITE)
    p("       Verified against the vllm_q35 env, not taken from the plan.")
    return sup


def print_model_report(s, fh=None):
    fh = fh or sys.stdout
    p = lambda x="": print(x, file=fh)
    c = s["cross_check"]
    p("")
    p("-" * 78)
    p("meta-device module walk (%s, torch %s / transformers %s)"
      % (s["auto_class"], s["torch_version"], s["transformers_version"]))
    p("  language model prefix : %s" % s["lm_prefix"])
    p("  Linear/Conv1d modules : %d total = %d language model + %d vision "
      "tower + %d other"
      % (s["n_modules_total"], s["n_lm"], s["n_vision"], s["n_other"]))
    p("  language-model leaves : %s"
      % ", ".join("%s x%d" % (k, v)
                  for k, v in sorted(s["lm_leaf_counts"].items())))
    p("  vision-tower leaves   : %s"
      % ", ".join("%s x%d" % (k, v)
                  for k, v in sorted(s["vision_leaf_counts"].items())))
    p("  other                 : %s"
      % ", ".join("%s %s" % (k, v["shape"])
                  for k, v in s["other_modules"].items()))
    p("")
    p("cross-check targets.enumerate_sites(cfg, 'all') vs the real module tree")
    p("  predicted %d sites, observed %d language-model Linear/Conv1d modules"
      % (c["n_predicted"], c["n_observed_lm"]))
    if c["predicted_missing"]:
        p("  PREDICTED BUT ABSENT (%d) -- a LoRA on these would be a silent "
          "no-op:" % len(c["predicted_missing"]))
        for n in c["predicted_missing"]:
            p("    %s" % n)
    else:
        p("  predicted-but-absent : none")
    if c["shape_mismatch"]:
        p("  SHAPE MISMATCH (%d):" % len(c["shape_mismatch"]))
        for m in c["shape_mismatch"]:
            p("    %s  %s" % (m["name"], m))
    else:
        p("  shape mismatches     : none")
    expl = [u for u in c["unpredicted"] if u["expected_exclusion"]]
    if expl:
        by_leaf = collections.Counter(u["leaf"] for u in expl)
        p("  present-but-unpredicted, EXPECTED (%d): %s"
          % (len(expl), ", ".join("%s x%d" % kv for kv in sorted(by_leaf.items()))))
        p("    reason: %s" % expl[0]["expected_exclusion"])
    if c["unpredicted_unexplained"]:
        p("  PRESENT BUT UNPREDICTED (%d) -- blind spots in the injection-site "
          "sweep:" % len(c["unpredicted_unexplained"]))
        for u in c["unpredicted_unexplained"][:20]:
            p("    %-60s %s %s" % (u["name"], u["cls"], u["shape"]))
        if len(c["unpredicted_unexplained"]) > 20:
            p("    ... %d more" % (len(c["unpredicted_unexplained"]) - 20))
    else:
        p("  unexplained extras   : none")
    p("")
    p("  CROSS-CHECK: %s" % ("PASS -- targets.py matches the checkpoint"
                             if c["ok"] else
                             "FAIL -- targets.py disagrees with the checkpoint"))
    return s


def summarize(model=None, config_only=False, rank=16, ranks=RANKS, fh=None):
    """Full survey: config + (optionally) meta-device walk + vLLM verdict."""
    fh = fh or sys.stdout
    report = collections.OrderedDict()
    report["model"] = model or paths.BASE_MODEL
    report["model_slug"] = model_slug(model)
    report["config"] = survey_from_config(model, ranks=ranks)
    print_config_report(report["config"], fh=fh, rank=rank)

    report["vllm"] = vllm_supportability()
    print_vllm_report(report["vllm"], fh=fh)

    if config_only:
        report["model_walk"] = {"skipped": "--config-only"}
        print("", file=fh)
        print("(meta-device module walk skipped: --config-only)", file=fh)
    else:
        report["model_walk"] = survey_from_model(model)
        print_model_report(report["model_walk"], fh=fh)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Qwen3.5 architecture / injection-site survey "
                    "(adapterCL.md 8.2, Phase 0)")
    ap.add_argument("--model", default=None,
                    help="hub id or local dir (default paths.BASE_MODEL)")
    ap.add_argument("--config-only", action="store_true",
                    help="skip the meta-device walk; no torch needed")
    ap.add_argument("--rank", type=int, default=16,
                    help="rank highlighted in the 11.2 answer line")
    ap.add_argument("--ranks", default=",".join(str(r) for r in RANKS))
    ap.add_argument("--out-dir", default=None, help="default paths.OUT_SURVEY")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    ranks = tuple(int(x) for x in args.ranks.split(",") if x.strip())
    rep = summarize(model=args.model, config_only=args.config_only,
                    rank=args.rank, ranks=ranks)

    if not args.no_write:
        out_dir = args.out_dir or paths.OUT_SURVEY
        paths.ensure_out_dirs()
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        path = os.path.join(out_dir,
                            "survey_%s.json" % model_slug(args.model))
        fh = open(path, "w")
        try:
            json.dump(rep, fh, indent=2, sort_keys=True, default=str)
        finally:
            fh.close()
        print("")
        print("wrote %s" % path)

    walk = rep.get("model_walk") or {}
    if "cross_check" in walk and not walk["cross_check"]["ok"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
