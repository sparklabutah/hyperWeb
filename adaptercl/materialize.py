"""Turn generated LoRA factors into something the serving stack can load.

The rollout path in this repo is vLLM, and 4.5 flags adapter application as a
*silent* failure class: a mis-named module produces a warning and an eval that
reads exactly like the base model. So every adapter this project produces goes
through one writer, in canonical PEFT layout, and `verify_adapter.py` proves
end-to-end that the serving stack actually applies it.

Layout written (standard PEFT):

    <dir>/adapter_config.json
    <dir>/adapter_model.safetensors      (or .bin if safetensors is missing)
    <dir>/adaptercl_meta.json            (our provenance: cell, split, generator)

Key format:

    base_model.model.<hf module path>.lora_A.weight    (r, d_in)
    base_model.model.<hf module path>.lora_B.weight    (d_out, r)

vLLM strips `base_model.model.` (lora/utils.py:174) and then remaps HF names
through the model's `hf_to_vllm_mapper.get_unstacked_mapper()`
(lora/worker_manager.py:135-151), which for Qwen3.5 declares
`.in_proj_qkv -> .in_proj_qkvz[0:3]`, `.in_proj_b -> .in_proj_ba[0]` and so on
(model_executor/models/qwen3_5.py:203-210). So HF/PEFT names are the correct
thing to write -- do not pre-fuse them.

Why per-episode adapters are still servable: 4.3 freezes the adapter for the
whole episode, and a TimeWarp episode never leaves its (environment, era) cell.
So a generated adapter is constant for the length of a rollout and can be
materialised once and served like any static LoRA. `granularity` records which
level was used.
"""

from __future__ import print_function

import json
import os
import shutil
import sys

from . import paths, targets

PEFT_PREFIX = "base_model.model."


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def _save_tensors(tensors, out_dir, verbose=True):
    """Write a state dict, preferring safetensors. Returns the filename used.

    The `.bin` fallback is not a degradation -- vLLM looks for both
    (`vllm/lora/lora_model.py:205-206`) and PEFT reads either -- but which one
    you get depends on the interpreter, since the system python has torch and
    no safetensors while the `llamafactory` env has both. Say which was used so
    a `.bin` in an adapter dir is never a surprise.
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        import torch
        fn = os.path.join(out_dir, "adapter_model.bin")
        torch.save(tensors, fn)
        if verbose:
            print("  note: safetensors unavailable under %s; wrote "
                  "adapter_model.bin (vLLM and PEFT both accept it)"
                  % sys.executable)
        return os.path.basename(fn)
    # safetensors refuses shared storage; clone to be safe.
    tensors = dict((k, v.contiguous().clone()) for k, v in tensors.items())
    fn = os.path.join(out_dir, "adapter_model.safetensors")
    save_file(tensors, fn, metadata={"format": "pt"})
    return os.path.basename(fn)


def adapter_config(modules, rank, alpha, base_model=None, layers=None):
    """The `adapter_config.json` dict PEFT and vLLM both read."""
    if isinstance(modules, str):
        modules = targets.target_set(modules)
    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "auto_mapping": None,
        "base_model_name_or_path": base_model or paths.BASE_MODEL,
        "revision": None,
        "r": int(rank),
        "lora_alpha": float(alpha),
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "modules_to_save": None,
        "target_modules": sorted(targets.peft_target_modules(modules)),
        "rank_pattern": {},
        "alpha_pattern": {},
        "use_rslora": False,
        "use_dora": False,
        "layers_to_transform": (sorted(layers) if layers else None),
        "layers_pattern": None,
    }
    return cfg


def write_adapter(out_dir, factors, sites, rank, alpha, modules,
                  base_model=None, meta=None, layers=None, overwrite=True,
                  dtype=None):
    """Write one PEFT adapter directory.

    Parameters
    ----------
    factors : {rel_name: (A, B)} or a list of (A, B) parallel to `sites`
    sites   : list of targets.Site (gives the full HF module path per factor)
    meta    : provenance dict merged into adaptercl_meta.json
    dtype   : torch dtype to cast to (default: keep float32; bf16 halves size)
    """
    import torch

    if os.path.isdir(out_dir):
        if not overwrite:
            raise IOError("%s exists" % out_dir)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    if not isinstance(factors, dict):
        factors = dict((s.rel_name, ab) for s, ab in zip(sites, factors))

    tensors, written = {}, []
    for s in sites:
        ab = factors.get(s.rel_name)
        if ab is None:
            continue
        A, B = ab
        A = torch.as_tensor(A).detach().cpu()
        B = torch.as_tensor(B).detach().cpu()
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError("site %s: factors must be 2-D, got %r / %r"
                             % (s.rel_name, tuple(A.shape), tuple(B.shape)))
        if tuple(A.shape) != (rank, s.d_in):
            raise ValueError("site %s: A is %r, expected (%d, %d)"
                             % (s.rel_name, tuple(A.shape), rank, s.d_in))
        if tuple(B.shape) != (s.d_out, rank):
            raise ValueError("site %s: B is %r, expected (%d, %d)"
                             % (s.rel_name, tuple(B.shape), s.d_out, rank))
        if dtype is not None:
            A, B = A.to(dtype), B.to(dtype)
        tensors[PEFT_PREFIX + s.name + ".lora_A.weight"] = A
        tensors[PEFT_PREFIX + s.name + ".lora_B.weight"] = B
        written.append(s.rel_name)

    if not tensors:
        raise ValueError("no factors matched any site -- nothing to write")

    weight_file = _save_tensors(tensors, out_dir)

    cfg = adapter_config(modules, rank, alpha, base_model=base_model, layers=layers)
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)

    prov = {
        "n_sites": len(written),
        "n_tensors": len(tensors),
        "rank": int(rank),
        "alpha": float(alpha),
        "target_modules": cfg["target_modules"],
        "weight_file": weight_file,
        "module_prefix": sites[0].prefix if sites else None,
        "sites": written,
    }
    if meta:
        prov.update(meta)
    with open(os.path.join(out_dir, "adaptercl_meta.json"), "w") as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)
    return out_dir


def write_from_handle(out_dir, handle, base_model=None, meta=None, **kw):
    """Materialise the factors currently attached to an InjectionHandle."""
    return write_adapter(out_dir, handle.state(), handle.sites, handle.rank,
                         handle.alpha,
                         sorted(set(s.module for s in handle.sites)),
                         base_model=base_model, meta=meta, **kw)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def _load_tensors(adapter_dir):
    st = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.exists(st):
        from safetensors.torch import load_file
        return load_file(st)
    bn = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.exists(bn):
        import torch
        return torch.load(bn, map_location="cpu")
    raise IOError("no adapter weights in %s" % adapter_dir)


def read_adapter(adapter_dir):
    """Return (factors_by_rel_name, config_dict, meta_dict).

    rel_name is `layers.<n>.<submodule>.<module>` -- prefix-independent, so an
    adapter written against `Qwen3_5ForConditionalGeneration` still loads into a
    handle built on `Qwen3_5ForCausalLM`.
    """
    tensors = _load_tensors(adapter_dir)
    with open(os.path.join(adapter_dir, "adapter_config.json")) as fh:
        cfg = json.load(fh)
    meta = {}
    mp = os.path.join(adapter_dir, "adaptercl_meta.json")
    if os.path.exists(mp):
        with open(mp) as fh:
            meta = json.load(fh)

    pairs = {}
    for key, val in tensors.items():
        name = key
        if name.startswith(PEFT_PREFIX):
            name = name[len(PEFT_PREFIX):]
        parts = name.split(".")
        if len(parts) < 3 or parts[-1] != "weight" or parts[-2] not in ("lora_A", "lora_B"):
            continue
        which = parts[-2]
        mod_path = ".".join(parts[:-2])
        # rel_name == everything from 'layers.' onward
        if ".layers." in mod_path:
            rel = mod_path[mod_path.index("layers."):]
        else:
            rel = mod_path
        pairs.setdefault(rel, {})[which] = val

    factors = {}
    for rel, d in pairs.items():
        if "lora_A" in d and "lora_B" in d:
            factors[rel] = (d["lora_A"], d["lora_B"])
    return factors, cfg, meta


def load_into_handle(handle, adapter_dir, strict=True):
    """Load a saved adapter into an existing InjectionHandle.

    This is how 6.2's cross-application works: build one handle on the frozen
    base, then swap cell (i)'s factors in and evaluate on cell (j) -- with no
    model reload between pairs.
    """
    factors, cfg, meta = read_adapter(adapter_dir)
    if int(cfg.get("r", handle.rank)) != handle.rank:
        raise ValueError("rank mismatch: adapter r=%s, handle r=%s"
                         % (cfg.get("r"), handle.rank))
    missing = handle.load_state(factors, strict=strict)
    return {"loaded": len(handle.sites) - len(missing), "missing": missing,
            "config": cfg, "meta": meta}


def merge_into_model(model, adapter_dir, scaling=None):
    """Fold an adapter into a live model's weights (merge-then-serve).

    Returns the number of modules merged. Mirrors merge_and_upload.py's role but
    works from our own writer's key layout and does not require peft.
    """
    import torch
    factors, cfg, _ = read_adapter(adapter_dir)
    if scaling is None:
        scaling = float(cfg["lora_alpha"]) / float(cfg["r"])
    named = dict(model.named_modules())
    merged = 0
    for name, mod in named.items():
        if not hasattr(mod, "weight") or not hasattr(mod, "in_features"):
            continue
        rel = name[name.index("layers."):] if ".layers." in name else None
        if rel is None or rel not in factors:
            continue
        A, B = factors[rel]
        dw = scaling * (B.to(torch.float32) @ A.to(torch.float32))
        with torch.no_grad():
            mod.weight.add_(dw.to(mod.weight.device, mod.weight.dtype))
        merged += 1
    return merged


# --------------------------------------------------------------------------
# Naming / layout for a run
# --------------------------------------------------------------------------

def cell_adapter_dir(cell, tag="percell", root=None):
    """Where a per-cell adapter for Phase 2 lives."""
    root = root or paths.OUT_CELLS
    return os.path.join(root, tag, cell.key)


def generated_adapter_dir(run, key, root=None):
    """Where a hypernetwork-generated adapter for one conditioning lives."""
    root = root or paths.OUT_HYPERNET
    return os.path.join(root, run, "generated", key)


def describe(adapter_dir):
    """One-line summary of an adapter on disk, for run logs."""
    factors, cfg, meta = read_adapter(adapter_dir)
    n = len(factors)
    params = 0
    for A, B in factors.values():
        params += A.numel() + B.numel()
    return ("%s: %d sites, r=%s alpha=%s, %s params, targets=%s"
            % (os.path.basename(adapter_dir.rstrip("/")), n, cfg.get("r"),
               cfg.get("lora_alpha"), "{:,}".format(params),
               ",".join(cfg.get("target_modules", []))))
