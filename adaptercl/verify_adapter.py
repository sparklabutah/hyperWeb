"""Phase-0 guard against the silent-adapter failure class (adapterCL.md 4.5 / 9).

The risk register calls this one "High, and insidious": vLLM warns about LoRA
modules it cannot wrap and then *serves the base model anyway*. Nothing raises,
the eval completes, the numbers look like a bad adapter rather than an absent
one, and a whole phase of results is noise. 7's Phase 0 therefore requires
"verify adapter application end-to-end through your eval path" **before** any
real training, which is what this module does.

    green HF check  !=  green vLLM check

They exercise different code. `verify_hf` proves the *factors* are non-trivial
and that `inject.InjectedLoRALinear` wires them into the right sites -- it is a
check on this repo's own code. `verify_vllm` proves the *serving stack* named in
paths.VLM_LORA_LAUNCHER actually applies them, which is a check on vLLM's
name-mapping (qwen3_5.py:203-210 -> lora/worker_manager.py:135-151) and its
LinearBase-only support scan (lora/utils.py:219-240). Only the vLLM check gates
Phase 3 evals, because Phase 3 rollouts go through vLLM and nothing else.

The probe adapter is deliberately *large-effect random* -- B is emphatically not
zero. A zero-init adapter is indistinguishable from an ignored adapter, so it is
useless as a probe; see `make_probe_adapter` for the scale derivation.

Failure strings scraped from the server log, all present in this env's
vLLM 0.23.1rc1:

    lora/model_manager.py:406,420   "... will be ignored."
    lora/model_manager.py:219       "does not support it. This will be ignored."
    lora/model_manager.py:505       "could not be wrapped by any LoRA layer
                                     implementation. It will be ignored."
    lora/utils.py:207               "<name> is unsupported LoRA weight"
    lora/lora_model.py:236-241      "expected target modules in {...} but
                                     received [...]"

Stdlib-only at import time (torch is imported lazily inside `verify_hf` /
`make_probe_adapter`), so `verify_vllm`, `scrape_server_log` and
`expected_module_report` run from the system python.
"""

from __future__ import print_function

import json
import os
import re
import sys

from . import materialize, paths, targets

# --------------------------------------------------------------------------
# Probe adapter
# --------------------------------------------------------------------------

#: Default per-site relative perturbation of the probe (see derivation below).
PROBE_SCALE = 0.5
PROBE_SEED = 20260806
PROBE_RANK = 8

#: vLLM only accepts these values for --max-lora-rank (config/lora.py:26).
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def probe_factors(sites, rank, scale=PROBE_SCALE, seed=PROBE_SEED, dtype=None):
    """Random (A, B) per site, scaled so the adapter is impossible to miss.

    Scale derivation. A site computes ``y = Wx + (alpha/r) * B A x``. With
    ``A ~ N(0, 1/d_in)`` and ``B ~ N(0, scale^2 / r)``, for an input x whose
    per-component RMS is ``rms_x``:

        RMS[(A x)_k]        = rms_x                        (sum of d_in terms
                                                            each var rms_x^2/d_in)
        RMS[(B A x)_i]      = sqrt(r) * (scale/sqrt(r)) * rms_x = scale * rms_x

    so with ``alpha == r`` (scaling 1) every injected site's output is perturbed
    by ``scale`` times its own scale. ``scale=0.5`` is a ~50% relative
    perturbation at every site -- three to four orders of magnitude larger than a
    trained LoRA, which is the point: the probe is a *detector*, not a model.
    Greedy continuations change on essentially every prompt and last-position
    logits move by tens of nats, so "adapter applied" and "adapter ignored" can
    never be confused for one another.

    RMSNorm at each block boundary keeps the residual stream from blowing up, so
    0.5 does not overflow bf16 across a 48-layer stack. If a future checkpoint
    does produce NaN logits, halve `scale`: anything above ~0.05 is still
    unmistakable.
    """
    import torch

    g = torch.Generator().manual_seed(int(seed))
    dtype = dtype or torch.float32
    out = {}
    for s in sites:
        A = torch.randn(rank, s.d_in, generator=g, dtype=torch.float32)
        B = torch.randn(s.d_out, rank, generator=g, dtype=torch.float32)
        A = A * (1.0 / (float(s.d_in) ** 0.5))
        B = B * (float(scale) / (float(rank) ** 0.5))
        out[s.rel_name] = (A.to(dtype), B.to(dtype))
    return out


def make_probe_adapter(out_dir, model_dir=None, target_set="attn_mlp",
                       rank=PROBE_RANK, scale=PROBE_SCALE, seed=PROBE_SEED,
                       alpha=None, layers=None):
    """Write a deliberately large-effect random LoRA to `out_dir`.

    `alpha` defaults to `rank`, i.e. scaling == 1, so `scale` *is* the relative
    perturbation (see `probe_factors`). The adapter is fully deterministic in
    `seed`, so the HF and vLLM checks compare the same weights.

    Returns the adapter directory.
    """
    cfg = targets.load_text_config(model_dir or paths.BASE_MODEL)
    sites = targets.enumerate_sites(cfg, target_set, layers=layers)
    if not sites:
        raise ValueError("target set %r produced no sites for this config"
                         % (target_set,))
    if rank not in VLLM_LORA_RANKS:
        print("  note: probe rank %d is not one of vLLM's --max-lora-rank values "
              "%r; the launcher will round the *server's* max rank up, which is "
              "fine, but keep the probe rank at 8 unless you mean otherwise."
              % (rank, VLLM_LORA_RANKS))
    alpha = float(rank if alpha is None else alpha)
    factors = probe_factors(sites, rank, scale=scale, seed=seed)
    modules = (targets.target_set(target_set)
               if isinstance(target_set, str) else tuple(target_set))
    materialize.write_adapter(
        out_dir, factors, sites, rank, alpha, modules,
        base_model=(model_dir or paths.BASE_MODEL),
        meta={
            "probe": True,
            "purpose": "adapterCL.md 7 Phase 0 / 9 -- detect a silently ignored "
                       "adapter in the serving stack",
            "probe_scale": float(scale),
            "probe_seed": int(seed),
            "target_set": target_set if isinstance(target_set, str) else list(target_set),
            "scaling": alpha / float(rank),
            "warning": "NOT a trained adapter. Random, ~%g relative perturbation "
                       "per site. Never report numbers produced with it."
                       % (scale,),
        })
    return out_dir


# --------------------------------------------------------------------------
# HF-side check
# --------------------------------------------------------------------------

DEFAULT_PROMPTS = (
    "The capital city of France is",
    "def add(a, b):\n    return",
    "To click the search button on this page I should issue the action",
)


def _load_base(model_dir, device="auto", dtype=None):
    """Load the base model with whichever auto class this checkpoint answers to."""
    import torch
    from transformers import AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    import transformers
    tried = []
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                     "AutoModelForVision2Seq", "AutoModel"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_dir, dtype=dtype,
                                        trust_remote_code=True)
        except TypeError:
            # transformers <5 spells it torch_dtype
            try:
                model = cls.from_pretrained(model_dir, torch_dtype=dtype,
                                            trust_remote_code=True)
            except Exception as exc:            # noqa: BLE001
                tried.append("%s: %s" % (cls_name, exc))
                continue
        except Exception as exc:                # noqa: BLE001
            tried.append("%s: %s" % (cls_name, exc))
            continue
        model = model.to(device).eval()
        return model, tok, device, dtype
    raise RuntimeError("could not load %s with any auto class:\n  %s"
                       % (model_dir, "\n  ".join(tried)))


def _last_logits_and_greedy(model, tok, prompt, device, max_new_tokens):
    import torch

    enc = tok(prompt, return_tensors="pt")
    enc = dict((k, v.to(device)) for k, v in enc.items())
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits[0, -1, :].float().cpu()
        gen = model.generate(max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=(tok.pad_token_id or tok.eos_token_id),
                             **enc)
    new = gen[0][enc["input_ids"].shape[1]:]
    return logits, tok.decode(new, skip_special_tokens=True)


def _kl(p_logits, q_logits):
    """KL(softmax(p) || softmax(q)) in nats."""
    import torch

    p = torch.log_softmax(p_logits.double(), dim=-1)
    q = torch.log_softmax(q_logits.double(), dim=-1)
    return float((p.exp() * (p - q)).sum())


def verify_hf(model_dir=None, adapter_dir=None, prompts=None, device="auto",
              max_new_tokens=16, atol=1e-3, dtype=None):
    """Load the base with transformers, then apply the adapter and compare.

    Proves that *this repo's* injection path (inject.inject +
    materialize.load_into_handle) changes the model's outputs. It says nothing
    about vLLM -- see the module docstring.

    Returns a report dict with `ok`, per-prompt `max_logit_delta`, `kl` and
    `greedy_changed`.
    """
    from . import inject as inject_mod

    model_dir = model_dir or paths.model_dir_or_id()
    if adapter_dir is None:
        raise ValueError("verify_hf needs an adapter_dir "
                         "(make one with `python -m adaptercl.verify_adapter probe`)")
    prompts = list(prompts or DEFAULT_PROMPTS)

    _factors, acfg, ameta = materialize.read_adapter(adapter_dir)
    rank = int(acfg["r"])
    alpha = float(acfg["lora_alpha"])
    modules = tuple(acfg["target_modules"])

    model, tok, device, dtype = _load_base(model_dir, device=device, dtype=dtype)

    base = []
    for p in prompts:
        base.append(_last_logits_and_greedy(model, tok, p, device, max_new_tokens))

    handle = inject_mod.inject(model, modules=modules, rank=rank, alpha=alpha)
    load_report = materialize.load_into_handle(handle, adapter_dir, strict=True)

    after = []
    for p in prompts:
        after.append(_last_logits_and_greedy(model, tok, p, device, max_new_tokens))

    rows = []
    for i, p in enumerate(prompts):
        b_logits, b_text = base[i]
        a_logits, a_text = after[i]
        delta = float((a_logits - b_logits).abs().max())
        finite = bool(a_logits.isfinite().all())
        rows.append({
            "prompt": p,
            "max_logit_delta": delta,
            "kl_nats": (_kl(b_logits, a_logits) if finite else None),
            "greedy_changed": (b_text != a_text),
            "base_greedy": b_text,
            "adapted_greedy": a_text,
            "finite": finite,
        })

    handle.remove()

    moved = [r for r in rows if r["max_logit_delta"] > atol]
    changed = [r for r in rows if r["greedy_changed"]]
    nonfinite = [r for r in rows if not r["finite"]]
    blocking = []
    if len(moved) != len(rows):
        blocking.append(
            "HF: %d/%d prompt(s) produced an identical last-position logit "
            "vector with the adapter loaded (|delta| <= %g). The injection path "
            "is not applying the factors."
            % (len(rows) - len(moved), len(rows), atol))
    if not changed:
        blocking.append(
            "HF: no greedy continuation changed. With a probe this large that "
            "means the adapter is a no-op; check that adapter B is not zero "
            "(materialize.describe %s)." % adapter_dir)
    if nonfinite:
        blocking.append(
            "HF: %d prompt(s) produced non-finite logits. The probe scale is too "
            "large for this checkpoint -- rerun `probe --scale 0.05`."
            % len(nonfinite))

    return {
        "check": "hf",
        "ok": not blocking,
        "blocking": blocking,
        "model_dir": model_dir,
        "adapter_dir": adapter_dir,
        "device": device,
        "rank": rank,
        "alpha": alpha,
        "target_modules": list(modules),
        "n_sites_loaded": load_report["loaded"],
        "n_sites_missing": len(load_report["missing"]),
        "adapter_meta": ameta,
        "n_prompts": len(rows),
        "n_logits_moved": len(moved),
        "n_greedy_changed": len(changed),
        "prompts": rows,
    }


# --------------------------------------------------------------------------
# vLLM-side check -- the one that gates Phase 3
# --------------------------------------------------------------------------

#: Substrings that mean "vLLM decided not to apply part or all of your adapter".
#: Any hit is a HARD failure: the eval will silently read as the base model.
IGNORED_PATTERNS = (
    "will be ignored",
    "It will be ignored",
    "could not be wrapped by any LoRA layer implementation",
    "is unsupported LoRA weight",
    "expected target modules in",
    "unsupported LoRA",
    "no matching PunicaWrapper",
    "not supported",
)


#: Module-path prefixes whose "will be ignored" warnings are EXPECTED and benign.
#: vLLM emits one per vision-tower linear at startup for every Qwen3.5 server,
#: adapter or not -- 111 of them on the 9B (27 blocks x 4, plus patch_embed and
#: the merger). 8.2 says to exclude the vision tower from every target set, so a
#: warning naming one is confirmation that the exclusion worked, not a failure.
#: Measured on grn075, 2026-08-06: 111 warnings, all `visual.*`, zero naming a
#: language-model module, while all 3 probe prompts changed (max dlogprob 4.06).
BENIGN_IGNORED_PREFIXES = ("visual.", "vision_tower.", "model.visual.")


#: Tokens that mark a line as being about adapter loading at all. "will be
#: ignored" is generic English and vLLM uses it for unrelated things -- the
#: Inductor-compilation notice matches it on every single startup. A line with
#: neither a module name nor one of these is not evidence about the adapter.
LORA_CONTEXT_TOKENS = ("lora", "punica", "adapter", "peft", "target modules",
                       "target_modules")


def _ignored_module_name(line):
    """The module a vLLM ignored-module warning is about, or None.

    Matches the shape `...; <module path> will be ignored.` used by
    vllm/lora/models.py's PunicaWrapper warning.
    """
    m = re.search(r"[;:]\s*([A-Za-z_][\w.]*)\s+will be ignored", line)
    return m.group(1) if m else None


def _is_lora_context(line):
    low = line.lower()
    return any(tok in low for tok in LORA_CONTEXT_TOKENS)


def scrape_server_log(log_path, patterns=IGNORED_PATTERNS,
                      benign_prefixes=BENIGN_IGNORED_PREFIXES,
                      target_modules=None):
    """Grep a vLLM server log for the ignored-module warnings (4.5).

    Returns {"ok", "hits", "benign", "path", "read", "n_lines"}. A missing log
    is reported as `read=False` and treated as *not ok*: the whole point is that
    a silent adapter looks exactly like a healthy run, so "I could not check"
    must never read as "fine".

    **Only warnings about modules that could plausibly be ours count.** A hit is
    benign when the module it names is in the vision tower (which 8.2 excludes
    by construction), and -- when `target_modules` is given -- when its leaf name
    is not one we actually asked for. Counting the benign ones as failures makes
    this gate impossible to pass on Qwen3.5, which is worse than not having it:
    a check that always fails gets disabled, and then the real silent-adapter
    failure walks straight through.
    """
    out = {"path": log_path, "read": False, "hits": [], "benign": [],
           "ok": False, "n_lines": 0}
    if not log_path or not os.path.exists(log_path):
        out["error"] = ("no server log at %r -- pass log_path=, or point it at "
                        "$OUTPUT_ROOT/logs/vlm_qwen35_<port>.log "
                        "(run_v6_eval_q35_sonnet.sh:163)" % (log_path,))
        return out
    targets_set = set(target_modules or ())
    hits, benign, n = [], [], 0
    with open(log_path, "rb") as fh:
        for i, raw in enumerate(fh, 1):
            n = i
            line = raw.decode("utf-8", "replace").rstrip("\n")
            for pat in patterns:
                if pat not in line:
                    continue
                rec = {"line_no": i, "pattern": pat, "line": line.strip()}
                mod = _ignored_module_name(line)
                rec["module"] = mod
                if mod is None and not _is_lora_context(line):
                    # e.g. "Inductor compilation was disabled ... will be
                    # ignored" -- generic English, fires on every startup.
                    rec["why_benign"] = "not an adapter-loading message"
                    benign.append(rec)
                    break
                if mod is not None:
                    if any(mod.startswith(p) or ("." + p) in mod
                           for p in benign_prefixes):
                        rec["why_benign"] = "vision tower -- excluded by 8.2"
                        benign.append(rec)
                        break
                    if targets_set and mod.rsplit(".", 1)[-1] not in targets_set:
                        rec["why_benign"] = ("leaf %r is not in this adapter's "
                                             "target set" % mod.rsplit(".", 1)[-1])
                        benign.append(rec)
                        break
                hits.append(rec)
                break
    out["read"] = True
    out["n_lines"] = n
    out["hits"] = hits
    out["benign"] = benign
    out["n_benign"] = len(benign)
    out["ok"] = not hits
    if benign and not hits:
        out["note"] = ("%d ignored-module warning(s), all about modules this "
                       "adapter never targets (%s). Expected."
                       % (len(benign),
                          ", ".join(sorted(set(b.get("why_benign", "?")
                                               for b in benign)))))
    return out


def _post_json(url, payload, timeout=120):
    try:
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError
    except ImportError:                       # pragma: no cover - py2
        raise RuntimeError("python 3 required")
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data,
                  headers={"Content-Type": "application/json",
                           "Authorization": "Bearer EMPTY"})
    try:
        resp = urlopen(req, timeout=timeout)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError("POST %s -> HTTP %s\n%s" % (url, exc.code, body[:2000]))
    except URLError as exc:
        raise RuntimeError("POST %s failed: %s (is the server up? "
                           "curl -sf %s/health)" % (url, exc, url.rsplit("/v1", 1)[0]))
    return json.loads(resp.read().decode("utf-8"))


def _get_json(url, timeout=30):
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError
    req = Request(url, headers={"Authorization": "Bearer EMPTY"})
    try:
        resp = urlopen(req, timeout=timeout)
    except (HTTPError, URLError) as exc:
        raise RuntimeError("GET %s failed: %s" % (url, exc))
    return json.loads(resp.read().decode("utf-8"))


def served_models(port, host="localhost"):
    """The `model` strings this server will accept, from /v1/models."""
    url = "http://%s:%d/v1/models" % (host, port)
    data = _get_json(url)
    return [m["id"] for m in data.get("data", [])]


def _complete(port, model, prompt, host="localhost", max_tokens=32, chat=False,
              n_logprobs=1):
    kind = "chat/completions" if chat else "completions"
    url = "http://%s:%d/v1/%s" % (host, port, kind)
    if chat:
        payload = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
                   "logprobs": True, "top_logprobs": n_logprobs,
                   "messages": [{"role": "user", "content": prompt}]}
    else:
        payload = {"model": model, "prompt": prompt, "temperature": 0.0,
                   "max_tokens": max_tokens, "logprobs": n_logprobs}
    data = _post_json(url, payload)
    try:
        choice = data["choices"][0]
        if chat:
            text = (choice["message"].get("content") or "")
            lp = choice.get("logprobs") or {}
            vals = [float(t["logprob"]) for t in (lp.get("content") or [])]
        else:
            text = choice.get("text") or ""
            lp = choice.get("logprobs") or {}
            vals = [float(v) for v in (lp.get("token_logprobs") or [])
                    if v is not None]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "malformed /v1/%s response from %s for model=%r (%s: %s). Is this "
            "really a vLLM OpenAI server? Response head: %s"
            % (kind, url, model, type(exc).__name__, exc,
               json.dumps(data)[:400]))
    return text, vals


def verify_vllm(port, base_name=None, lora_name="probe", prompts=None,
                host="localhost", log_path=None, chat=False, max_tokens=32,
                model_dir=None, adapter_dir=None, atol=1e-4):
    """A/B the base served-name against the LoRA served-name on a live server.

    Assumes a vLLM server is **already running** with `--enable-lora` and
    `--lora-modules <lora_name>=<adapter_dir>` -- start it with
    scripts/startVLM_lora.sh; this function starts nothing.

    Three independent failure detectors, because each catches a different way
    the adapter can go missing:

    1. `/v1/models` must list `lora_name`. If it does not, every request the
       benchmark sends with that model string 404s or silently falls back.
    2. The completion text / token logprobs must differ from the base's. This is
       the only check that survives a vLLM version bump.
    3. The server log must contain none of IGNORED_PATTERNS. A *partially*
       ignored adapter (say, `in_proj_b` dropped while the MLP sites load) still
       changes outputs, so check 2 passes and the run is still invalid.

    Returns a report dict with `ok` and `blocking`.
    """
    prompts = list(prompts or DEFAULT_PROMPTS)
    report = {"check": "vllm", "port": port, "host": host,
              "lora_name": lora_name, "base_name": base_name,
              "adapter_dir": adapter_dir, "chat": chat, "prompts": []}
    blocking = []

    try:
        models = served_models(port, host=host)
    except RuntimeError as exc:
        return {"check": "vllm", "ok": False, "port": port,
                "blocking": ["vLLM: %s" % exc], "served_models": []}
    report["served_models"] = models

    if base_name is None:
        others = [m for m in models if m != lora_name]
        if not others:
            blocking.append("vLLM: /v1/models lists only %r -- the base is not "
                            "served, so there is nothing to A/B against. The "
                            "launcher must pass --served-model-name." % (models,))
            base_name = lora_name
        else:
            base_name = others[0]
        report["base_name"] = base_name

    if lora_name not in models:
        blocking.append(
            "vLLM: LoRA served-name %r is not in /v1/models (%r). The server was "
            "started without --lora-modules %s=<dir>, or the adapter failed to "
            "load. THIS IS THE 4.5 SILENT FAILURE: benchmarkGeneral.py would "
            "send model=%r, vLLM would reject or fall back, and the eval would "
            "read as the base model."
            % (lora_name, models, lora_name, lora_name))

    if not blocking:
        for p in prompts:
            try:
                b_text, b_lp = _complete(port, base_name, p, host=host,
                                         max_tokens=max_tokens, chat=chat)
                a_text, a_lp = _complete(port, lora_name, p, host=host,
                                         max_tokens=max_tokens, chat=chat)
            except RuntimeError as exc:
                blocking.append("vLLM: %s" % exc)
                break
            k = min(len(b_lp), len(a_lp))
            dlp = max([abs(b_lp[i] - a_lp[i]) for i in range(k)]) if k else 0.0
            report["prompts"].append({
                "prompt": p,
                "text_changed": b_text != a_text,
                "max_logprob_delta": dlp,
                "n_compared_tokens": k,
                "base_text": b_text,
                "lora_text": a_text,
            })
        n_changed = sum(1 for r in report["prompts"] if r["text_changed"])
        n_moved = sum(1 for r in report["prompts"]
                      if r["max_logprob_delta"] > atol)
        report["n_text_changed"] = n_changed
        report["n_logprob_moved"] = n_moved
        # Only conclude "ignored" if every prompt actually completed; a request
        # that errored out has no evidence either way.
        if report["prompts"] and n_changed == 0 and n_moved == 0:
            blocking.append(
                "vLLM: model=%r and model=%r produced byte-identical text AND "
                "identical token logprobs on all %d probe prompts. The adapter "
                "is being ignored. Check the server log for "
                "'will be ignored' / 'expected target modules in'; then compare "
                "`expected_module_report()` against "
                "vllm/lora/utils.py:219-240's LinearBase-only scan."
                % (base_name, lora_name, len(prompts)))

    # Scope the scrape to the modules this adapter actually targets, so the
    # vision tower's expected warnings cannot fail the gate (see
    # BENIGN_IGNORED_PREFIXES).
    target_modules = None
    if adapter_dir:
        try:
            _f, _cfg, _m = materialize.read_adapter(adapter_dir)
            target_modules = _cfg.get("target_modules")
        except Exception:                        # noqa: BLE001 - best effort
            target_modules = None
    log = (scrape_server_log(log_path, target_modules=target_modules)
           if log_path else None)
    report["log_scrape"] = log
    if log is not None:
        if not log["read"]:
            blocking.append("vLLM: %s" % log.get("error"))
        elif log["hits"]:
            blocking.append(
                "vLLM: server log has %d ignored-module warning(s) naming a "
                "module this adapter targets, e.g. %r. Even if outputs changed, "
                "part of the adapter was dropped and the run is invalid (4.5)."
                % (len(log["hits"]), log["hits"][0]["line"][:220]))
    else:
        report["log_note"] = (
            "no log_path given -- the text/logprob A/B cannot detect a "
            "PARTIALLY ignored adapter. Pass "
            "log_path=$OUTPUT_ROOT/logs/vlm_qwen35_<port>.log before trusting "
            "this as the Phase-3 gate.")

    report["ok"] = not blocking
    report["blocking"] = blocking
    return report


# --------------------------------------------------------------------------
# Name-resolution report
# --------------------------------------------------------------------------

def expected_module_report(target_set="attn_mlp", fh=None):
    """Per PEFT module name, the fused vLLM name it must resolve into.

    Reads targets.vllm_expected_names, which encodes qwen3_5.py's
    `packed_modules_mapping` / `hf_to_vllm_mapper`. Print this next to the
    server log when a module goes missing: a name that does not appear on the
    right of this table is one vLLM has no wrapper for.
    """
    fh = fh or sys.stdout
    modules = (targets.target_set(target_set)
               if isinstance(target_set, str) else tuple(target_set))
    mapping = targets.vllm_expected_names(modules)
    print("target set %r -> vLLM fused names" % (target_set,), file=fh)
    print("  %-14s %-16s %s" % ("peft name", "vllm name", "note"), file=fh)
    rows = []
    for m in modules:
        fused = mapping[m]
        note = "" if fused != m else "(not fused -- 1:1)"
        if m in targets.EXCLUDED_MODULES:
            note = "EXCLUDED: nn.Conv1d, dropped by LinearBase-only scan"
        print("  %-14s %-16s %s" % (m, fused, note), file=fh)
        rows.append({"peft": m, "vllm": fused, "note": note})
    print("", file=fh)
    print("  excluded everywhere: %s" % (", ".join(targets.EXCLUDED_MODULES),),
          file=fh)
    print("  a PEFT name absent from vLLM's runtime support list produces "
          "'will be ignored' and an eval that reads as the base model (4.5).",
          file=fh)
    return rows


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

VERIFICATION_JSON = "adapter_verification.json"


def launch_commands(adapter_dir, model_dir=None, port=8011, lora_name="probe",
                    cuda_devices="0", log_path=None, rank=PROBE_RANK):
    """The exact shell commands the operator must run. This module starts nothing."""
    model_dir = model_dir or paths.model_dir_or_id()
    log_path = log_path or os.path.join(paths.OUT_LOGS,
                                        "vlm_lora_%d.log" % port)
    max_rank = min([r for r in VLLM_LORA_RANKS if r >= rank] or [512])
    serve = (
        "CUDA_VISIBLE_DEVICES=%s LORA_MODULES='%s=%s' MAX_LORA_RANK=%d "
        "bash %s --model %s --port %d > %s 2>&1 &"
        % (cuda_devices, lora_name, adapter_dir, max_rank,
           paths.VLM_LORA_LAUNCHER, model_dir, port, log_path))
    wait = "until curl -sf http://localhost:%d/health; do sleep 5; done" % port
    check = (
        "%s -m adaptercl.verify_adapter vllm --port %d --lora-name %s "
        "--adapter-dir %s --log %s"
        % (sys.executable, port, lora_name, adapter_dir, log_path))
    return {"serve": serve, "wait": wait, "verify": check, "log_path": log_path,
            "max_lora_rank": max_rank}


def verify_end_to_end(adapter_dir=None, model_dir=None, target_set="attn_mlp",
                      rank=PROBE_RANK, scale=PROBE_SCALE, seed=PROBE_SEED,
                      do_hf=False, do_vllm=False, port=8011, lora_name="probe",
                      host="localhost", log_path=None, prompts=None,
                      cuda_devices="0", device="auto", chat=False,
                      out_json=None, fh=None):
    """Build the probe, print the commands to run, run whatever is enabled.

    Never starts a server. `do_hf` / `do_vllm` default to False so importing or
    invoking this is always safe on a login node; the caller opts in.

    Writes `paths.OUT_EVAL/adapter_verification.json` with a top-level `ok` and
    a `blocking` list. `ok` is False whenever the vLLM check did not run, since
    the vLLM check is the Phase-3 gate and "not checked" must never read as
    "passed".
    """
    fh = fh or sys.stdout
    paths.ensure_out_dirs()
    model_dir = model_dir or paths.model_dir_or_id()
    if adapter_dir is None:
        adapter_dir = os.path.join(paths.OUT_EVAL, "probe_adapter")
    if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        print("building probe adapter in %s" % adapter_dir, file=fh)
        make_probe_adapter(adapter_dir, model_dir=model_dir,
                           target_set=target_set, rank=rank, scale=scale,
                           seed=seed)
    print("  %s" % materialize.describe(adapter_dir), file=fh)
    print("", file=fh)

    expected_module_report(target_set, fh=fh)
    print("", file=fh)

    cmds = launch_commands(adapter_dir, model_dir=model_dir, port=port,
                           lora_name=lora_name, cuda_devices=cuda_devices,
                           log_path=log_path, rank=rank)
    log_path = log_path or cmds["log_path"]
    print("run these, in order, on a node with a GPU:", file=fh)
    print("  1) %s" % cmds["serve"], file=fh)
    print("  2) %s" % cmds["wait"], file=fh)
    print("  3) %s" % cmds["verify"], file=fh)
    print("", file=fh)

    checks, blocking = [], []
    if do_hf:
        print("running HF check ...", file=fh)
        hf = verify_hf(model_dir=model_dir, adapter_dir=adapter_dir,
                       prompts=prompts, device=device)
        checks.append(hf)
        blocking.extend(hf["blocking"])
        print("  hf: %s (%d/%d prompts moved, %d greedy changed)"
              % ("OK" if hf["ok"] else "FAIL", hf["n_logits_moved"],
                 hf["n_prompts"], hf["n_greedy_changed"]), file=fh)
    if do_vllm:
        print("running vLLM check ...", file=fh)
        vl = verify_vllm(port=port, lora_name=lora_name, prompts=prompts,
                         host=host, log_path=log_path, chat=chat,
                         adapter_dir=adapter_dir, model_dir=model_dir)
        checks.append(vl)
        blocking.extend(vl["blocking"])
        print("  vllm: %s (%s)" % ("OK" if vl["ok"] else "FAIL",
                                   ", ".join(vl.get("served_models", []))),
              file=fh)
    else:
        blocking.append(
            "vLLM check NOT RUN. adapterCL.md 7 Phase 0 requires end-to-end "
            "verification in the serving stack used for rollouts; a green HF "
            "check does not imply a green vLLM check. Rerun with --vllm once "
            "the server from step 1 is healthy.")

    report = {
        "ok": not blocking,
        "blocking": blocking,
        "adapter_dir": adapter_dir,
        "model_dir": model_dir,
        "target_set": target_set,
        "rank": rank,
        "probe_scale": scale,
        "probe_seed": seed,
        "commands": cmds,
        "checks": checks,
        "gate": ("vllm check is the Phase-3 gate; hf check only validates "
                 "adaptercl.inject"),
    }
    out_json = out_json or os.path.join(paths.OUT_EVAL, VERIFICATION_JSON)

    # NEVER downgrade a passing verdict from a run that checked nothing.
    # `run_phase0.sh`'s skipped branch calls this to print the commands, and on
    # 2026-08-06 23:26 that planning call overwrote a genuine ok=True from two
    # minutes earlier with ok=False/checks=[] -- silently un-verifying a
    # verified system and re-closing the Phase-3 gate. A report that ran no
    # checks is not evidence of anything and must not overwrite one that did.
    if not checks and os.path.exists(out_json):
        try:
            with open(out_json) as f:
                prev = json.load(f)
        except (ValueError, IOError):
            prev = None
        if prev and prev.get("ok") and not (prev.get("blocking") or []):
            print("", file=fh)
            print("NOT overwriting %s: it records a PASSING verification (%d "
                  "check(s)) and this invocation ran none. Re-run with --vllm "
                  "to replace it." % (out_json, len(prev.get("checks") or [])),
                  file=fh)
            return prev

    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
    print("", file=fh)
    print("wrote %s  ->  ok=%s" % (out_json, report["ok"]), file=fh)
    for b in blocking:
        print("  BLOCKING: %s" % b, file=fh)
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print_json(obj):
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Phase-0 adapter-application verifier (adapterCL.md 4.5/9). "
                    "Starts no servers.")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("probe", help="write a large-effect random probe adapter")
    p.add_argument("--out", default=os.path.join(paths.OUT_EVAL, "probe_adapter"))
    p.add_argument("--model", default=None)
    p.add_argument("--target-set", default="attn_mlp")
    p.add_argument("--rank", type=int, default=PROBE_RANK)
    p.add_argument("--scale", type=float, default=PROBE_SCALE)
    p.add_argument("--seed", type=int, default=PROBE_SEED)

    h = sub.add_parser("hf", help="transformers-side check (needs torch + weights)")
    h.add_argument("--model", default=None)
    h.add_argument("--adapter-dir", required=True)
    h.add_argument("--device", default="auto")
    h.add_argument("--max-new-tokens", type=int, default=16)

    v = sub.add_parser("vllm", help="A/B a live vLLM server (must already be up)")
    v.add_argument("--port", type=int, required=True)
    v.add_argument("--host", default="localhost")
    v.add_argument("--lora-name", default="probe")
    v.add_argument("--base-name", default=None)
    v.add_argument("--adapter-dir", default=None)
    v.add_argument("--log", dest="log_path", default=None,
                   help="vLLM server log to scrape for ignored-module warnings")
    v.add_argument("--chat", action="store_true")
    v.add_argument("--max-tokens", type=int, default=32)
    v.add_argument("--out-json", default=None,
                   help="where to persist the verdict (default: "
                        "out/eval/%s, which is the Phase-3 gate). Pass an "
                        "empty string to print only." % VERIFICATION_JSON)

    m = sub.add_parser("modules", help="print the PEFT -> vLLM name table")
    m.add_argument("--target-set", default="attn_mlp")

    s = sub.add_parser("scrape", help="grep a vLLM log for ignored-module warnings")
    s.add_argument("log_path")

    e = sub.add_parser("e2e", help="probe + print commands + run enabled checks")
    e.add_argument("--adapter-dir", default=None)
    e.add_argument("--model", default=None)
    e.add_argument("--target-set", default="attn_mlp")
    e.add_argument("--rank", type=int, default=PROBE_RANK)
    e.add_argument("--scale", type=float, default=PROBE_SCALE)
    e.add_argument("--seed", type=int, default=PROBE_SEED)
    e.add_argument("--port", type=int, default=8011)
    e.add_argument("--lora-name", default="probe")
    e.add_argument("--host", default="localhost")
    e.add_argument("--log", dest="log_path", default=None)
    e.add_argument("--cuda-devices", default="0")
    e.add_argument("--device", default="auto")
    e.add_argument("--hf", action="store_true", help="run the transformers check")
    e.add_argument("--vllm", action="store_true", help="run the live-server check")
    e.add_argument("--chat", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd is None:
        ap.print_help()
        return 2

    if args.cmd == "probe":
        d = make_probe_adapter(args.out, model_dir=args.model,
                               target_set=args.target_set, rank=args.rank,
                               scale=args.scale, seed=args.seed)
        print(materialize.describe(d))
        print("wrote %s" % d)
        print("NOT a trained adapter -- random, ~%g relative perturbation per "
              "site. Detection only." % args.scale)
        return 0

    if args.cmd == "hf":
        rep = verify_hf(model_dir=args.model, adapter_dir=args.adapter_dir,
                        device=args.device, max_new_tokens=args.max_new_tokens)
        _print_json(rep)
        return 0 if rep["ok"] else 1

    if args.cmd == "vllm":
        rep = verify_vllm(port=args.port, host=args.host,
                          lora_name=args.lora_name, base_name=args.base_name,
                          adapter_dir=args.adapter_dir, log_path=args.log_path,
                          chat=args.chat, max_tokens=args.max_tokens)
        _print_json(rep)
        # Persist it. The vLLM leg IS the Phase-3 gate, and a passing check that
        # is only printed is a passing check that gets re-run tomorrow -- or
        # worse, remembered as passing without evidence. Written in the same
        # schema `cli.adapter_verification_state()` reads, so a standalone
        # `verify vllm` opens the gate exactly as `verify e2e --vllm` does.
        if args.out_json != "":
            out_json = args.out_json or os.path.join(paths.OUT_EVAL,
                                                     VERIFICATION_JSON)
            d = os.path.dirname(out_json)
            if d and not os.path.isdir(d):
                os.makedirs(d)
            report = {
                "ok": bool(rep.get("ok")),
                "blocking": list(rep.get("blocking") or []),
                "checks": [dict(rep, name="vllm")],
                "adapter_dir": args.adapter_dir,
                "source": "verify_adapter vllm",
                "gate": "vllm check is the Phase-3 gate",
            }
            with open(out_json, "w") as f:
                json.dump(report, f, indent=2, sort_keys=True, default=str)
            print("wrote %s  ->  ok=%s" % (out_json, report["ok"]))
        return 0 if rep["ok"] else 1

    if args.cmd == "modules":
        expected_module_report(args.target_set)
        return 0

    if args.cmd == "scrape":
        rep = scrape_server_log(args.log_path)
        _print_json(rep)
        return 0 if rep["ok"] else 1

    if args.cmd == "e2e":
        rep = verify_end_to_end(
            adapter_dir=args.adapter_dir, model_dir=args.model,
            target_set=args.target_set, rank=args.rank, scale=args.scale,
            seed=args.seed, do_hf=args.hf, do_vllm=args.vllm, port=args.port,
            lora_name=args.lora_name, host=args.host, log_path=args.log_path,
            cuda_devices=args.cuda_devices, device=args.device, chat=args.chat)
        return 0 if rep["ok"] else 1

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
