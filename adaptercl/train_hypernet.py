"""End-to-end training of the adapter generator (adapterCL.md 4.4 / 8.4).

This is 8.4's `train_step` made real. What it trains, and what it deliberately
does not:

* **Task loss only.** Conditioning -> generator -> LoRA factors -> frozen base
  model -> next-token cross-entropy on the teacher's completion. The gradient
  reaches the generator through `InjectedLoRALinear`, whose factors are plain
  tensors rather than `nn.Parameter`s (inject.py:41).
* **No reconstruction objective, ever.** 4.4: LoRA weight space is not clustered
  -- for any invertible R, `BA = (BR)(R^-1 A)` -- so regressing a hypernetwork
  onto independently trained per-cell adapters fits an arbitrary member of a
  symmetry orbit. It is not implemented here and should not be added. The
  per-cell adapters from `percell.py` are for the 6.2 *functional* transfer
  matrix, not as regression targets.

Five things 8.4's sketch gets wrong or leaves out, all fixed here:

1. **`h = encoder(batch["first_observation"])` is one vector for a whole batch.**
   A BC batch mixes cells, so the adapter must differ per example. We generate
   `(Bsz, r, d_in)` / `(Bsz, d_out, r)` factors and let `InjectedLoRALinear`'s
   batched path apply them (inject.py:163-172). With a shared `h` the loss is
   silently wrong -- every example gets the first example's interface adapter.
2. **`site.set_factors(A[i], B[i])` indexes the wrong axis** once factors are
   batched; the site loop indexes *sites*, the batch axis is inside each tensor.
   `handle.set_factors(A_list, B_list)` keeps the two straight.
3. **`base_model(**batch["inputs"]).loss` has no loss mask.** BC on a full
   ShareGPT sample would train on the 16k-character AXTree prompt as if the
   model had produced it. Labels are masked to the assistant turn only.
4. **No `finally`.** If the forward raises, the factors stay attached and the
   next (unrelated) forward silently uses a stale adapter. `handle.clear()` runs
   in a `finally` here, and again after the whole loop.
5. **No freeze assertion.** `inject.check_base_frozen` is asserted after
   injection and again at the end, and the optimizer is built from the
   generator's parameters only -- 9B of base weights leaking into the optimizer
   would be a silent OOM at best and a confound at worst.

Chat formatting reproduces the LLaMA-Factory template selected for the
backbone by `percell.template_for`: `qwen3_5` for Qwen3.5 and `llama3` for
Llama-3.1. It does not use the tokenizer's jinja `chat_template`. The two Qwen
templates differ in a way that matters for this corpus: the jinja template
injects `<think>\\n` after the assistant header, and our teacher completions
already start with a literal `<think>` block, so the HF template produces
doubled thinking tags. LF's `ReasoningTemplate` instead injects an empty
thinking block only when the completion has no thought tags at all. Llama3 has
no corresponding thinking rule. Reproducing LF exactly also means the
hypernetwork's task loss is the same objective the per-cell LoRAs minimise.

Requires torch + transformers: run under `paths.PY_TRAIN`
(a Python environment with torch and transformers). `--smoke` runs the entire loop on a
~200k-parameter randomly initialised Qwen3.5 on CPU and asserts the loss falls
and the base stays frozen.
"""

from __future__ import print_function

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from . import cells, inject, materialize, paths, percell, targets

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclasses.dataclass
class TrainConfig:
    """Everything that defines a hypernetwork run (8.4 + 6.5's ablation axes)."""

    # -- what is generated ------------------------------------------------
    generator: str = "basis"          # "free" | "mixture" | "basis" (6.5)
    generator_kwargs: dict = dataclasses.field(default_factory=dict)
    target_set: str = "attn_mlp"      # targets.TARGET_SETS (4.5)
    rank: int = 8
    alpha: float = None               # defaults to rank (PEFT scaling = alpha/r)
    dropout: float = 0.0
    layers: tuple = None              # optional layer subset

    # -- what it is conditioned on ---------------------------------------
    conditioning: str = "vision"      # encoder.ConditioningSpec modality, or "hash"
    conditioning_kwargs: dict = dataclasses.field(default_factory=dict)
    embeddings: str = None            # precomputed encoder.save_embeddings .npz
    d_cond: int = None                # inferred from the conditioner when None

    # -- data --------------------------------------------------------------
    corpus: str = None                # bcdata.build_cell_corpora manifest.json
    split: str = "forward"            # cells.SPLITS (6.1)
    # The adapter unit. "version" (6 units) is the default and matches a
    # corpus built by `bcdata cell-corpora versions`; "cell" (18) is only
    # for 6.2's crossing. Resolving a split to Cell keys against a
    # version-keyed corpus loads zero samples and fails confusingly.
    granularity: str = "version"
    bank_tag: str = "ver"             # out/cells/<bank_tag>/<unit> for mixture
    train_cells: tuple = None         # override the split
    eval_cells: tuple = None
    max_samples_per_cell: int = None
    max_len: int = 8192               # tokens; the full recipe uses 65536
    min_target_tokens: int = 8

    # -- optimisation ------------------------------------------------------
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 2               # examples per micro-batch
    grad_accum: int = 8
    max_steps: int = 500              # optimizer steps
    epochs: float = None              # derives max_steps from corpus size
    warmup_steps: int = 20
    warmup_frac: float = None         # derives warmup_steps after max_steps
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    gen_l2: float = 0.0               # L2 on factors generated for this step
    seed: int = 0
    diversity_weight: float = 0.0     # 6.7 diversity-collapse mitigation
    diversity_kind: str = "cosine"    # hypernet.diversity_penalty kind
    # T2L_PLAN E9: up-weight the loss on `send_msg_to_user` steps. 1 and 12
    # both say answer RATE, not action quality, is what predicts success on
    # every fold, so this is the one loss knob aimed at the binding constraint.
    # Reported as a loss-reweighting ablation: if it lifts every arm equally it
    # is measuring the benchmark floor, not the generator.
    term_weight: float = 1.0
    term_action: str = "send_msg_to_user"

    # -- systems -----------------------------------------------------------
    base_model: str = None            # defaults to paths.BASE_MODEL
    device: str = "cuda"
    bf16: bool = True
    gradient_checkpointing: bool = True
    chat_template: str = "llamafactory"   # "llamafactory" | "hf"
    vocab_clamp: int = None           # smoke only: fold token ids into a tiny vocab

    # -- warm start / partial training (T2L_PLAN E2, E4b) ------------------
    #: a checkpoint written by save_checkpoint or scripts/train_t2l.py. E2 asks
    #: whether the 30k-step reconstruction solution is a useful warm start for
    #: BC or a harmful one (it encodes the shrunken conditional mean, A.3).
    init_checkpoint: str = None
    init_strict: bool = True
    #: substring filter on generator parameter names; everything else is frozen.
    #: E4b is exactly `train_only="log_gain"` on a loaded recon checkpoint --
    #: the DIRECTION stays whatever reconstruction produced and only the
    #: magnitude is learned through the policy.
    train_only: str = None

    # -- bookkeeping -------------------------------------------------------
    out_dir: str = None
    run_name: str = "hypernet"
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 100
    eval_batches: int = 8
    allow_fallback_generator: bool = False    # smoke only

    # ----------------------------------------------------------------------
    def __post_init__(self):
        self.base_model = self.base_model or paths.BASE_MODEL
        self.alpha = float(self.rank if self.alpha is None else self.alpha)
        self.out_dir = self.out_dir or os.path.join(paths.OUT_HYPERNET,
                                                    self.run_name)
        if self.epochs is not None and float(self.epochs) <= 0:
            raise ValueError("epochs must be positive, got %r" % self.epochs)
        if self.warmup_frac is not None \
                and not 0.0 <= float(self.warmup_frac) <= 1.0:
            raise ValueError("warmup_frac must be in [0, 1], got %r"
                             % self.warmup_frac)
        if not 0.0 <= float(self.label_smoothing) <= 1.0:
            raise ValueError("label_smoothing must be in [0, 1], got %r"
                             % self.label_smoothing)
        if float(self.gen_l2) < 0:
            raise ValueError("gen_l2 must be non-negative, got %r" % self.gen_l2)

    def resolved_cells(self):
        """(train_units, eval_units) as Cell OR Version objects.

        `granularity` decides which. The default adapter unit is the VERSION
        (2), and a corpus built by `bcdata cell-corpora versions` is keyed
        v1..v6; resolving a split to Cell keys against it loads zero samples and
        dies with a confusing "no samples for cells [wiki_e1, ...]".
        `parse_unit` accepts either key form.
        """
        if self.train_cells is not None:
            tr = [cells.parse_unit(c) if isinstance(c, str) else c
                  for c in self.train_cells]
            ev = [cells.parse_unit(c) if isinstance(c, str) else c
                  for c in (self.eval_cells or ())]
            return tr, ev
        sp = cells.split_by_name(self.split)
        if getattr(self, "granularity", "version") == "version":
            return (list(cells.versions_for(sp.train_eras)),
                    list(cells.versions_for(sp.test_eras)))
        return list(sp.train_cells()), list(sp.test_cells())

    def to_dict(self):
        d = dataclasses.asdict(self)
        d["train_cells"] = [c.key if hasattr(c, "key") else c
                            for c in (self.train_cells or ())] or None
        d["eval_cells"] = [c.key if hasattr(c, "key") else c
                           for c in (self.eval_cells or ())] or None
        return d

    def save(self, path=None):
        path = path or os.path.join(self.out_dir, "config.json")
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True, default=str)
        return path


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


class BCSample(object):
    """One ShareGPT sample plus the cell it was rolled out on.

    `action` is the bare function name inside the `<action>` block
    (`click`, `fill`, `send_msg_to_user`, ...). It is parsed once at load time
    because two things need it: E9's termination re-weighting during training,
    and `scripts/score_adapter_nll.py`'s split of held-out NLL into a
    termination proxy and an accuracy proxy.
    """

    __slots__ = ("cell_key", "system", "user", "assistant", "images", "action",
                 "struct")

    def __init__(self, cell_key, system, user, assistant, images=None,
                 action=None, struct=None):
        self.cell_key = cell_key
        self.action = action if action is not None else parse_action(assistant)
        self.struct = struct
        self.system = system
        self.user = user
        self.assistant = assistant
        self.images = images or []

    def __repr__(self):
        return "BCSample(%s, %s, |u|=%d, |a|=%d)" % (
            self.cell_key, self.action, len(self.user), len(self.assistant))


#: `<action>\n click('42') \n</action>` -> "click". Returns None when the
#: completion has no parseable action block, which is itself informative: those
#: samples are neither termination nor a clean action step.
_ACTION_RE = re.compile(r"<action>\s*([A-Za-z_][A-Za-z_0-9]*)\s*\(", re.S)


def parse_action(assistant):
    m = _ACTION_RE.search(assistant or "")
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# AXTree role histograms (T2L_PLAN E5: escaping N = 5)
# --------------------------------------------------------------------------

#: The observation block in a BC prompt is BrowserGym's AXTree, not
#: playwright's `aria_snapshot()`, so the role token spelling differs from
#: `capture.ROLE_VOCAB`. These three are the only systematic renames on this
#: benchmark's pages; everything else already matches (link, button, textbox,
#: heading, paragraph, table, cell, row, banner, main, contentinfo, ...).
#: Mapping them explicitly rather than letting them fall into `_other` keeps the
#: AXTree histogram in the SAME coordinate system as the capture histogram, so
#: the two conditioning routes are comparable.
AXTREE_ROLE_ALIASES = {
    "statictext": "text",
    "rootwebarea": "document",
    "section": "region",
    "image": "img",
    # BrowserGym reports presentational tables as Layout*; folding them into
    # the semantic table roles is the difference between v1/v2 (table-driven
    # layouts, ~14k such nodes across the corpus) reading as structurally
    # distinctive and reading as `_other`.
    "layouttable": "table",
    "layouttablerow": "row",
    "layouttablecell": "cell",
    "descriptionlist": "list",
    "listmarker": "_other",
    "labeltext": "_other",
}

#: `\t[42] textbox 'Search', clickable, visible` -> role "textbox";
#: `\t\tStaticText 'W'` -> role "statictext". A leading `[bid]` is optional.
_AX_LINE = re.compile(r"^\s*(?:\[[A-Za-z0-9_-]+\]\s*)?([A-Za-z][A-Za-z0-9]*)\b")

_AX_START = "## AXTree:"
#: The AXTree block ends at the next top-level section of the prompt.
_AX_END = ("## Focused element:", "# History of interaction",
           "# Action space:", "## Currently open tabs:")

#: Step-1 prompts say "You just executed step -1 of the previously proposed
#: plan". The converter emits samples in episode order, so this marker is what
#: partitions a flat corpus back into episodes.
_FIRST_STEP = re.compile(r"executed step\s+-1\b")


def axtree_block(user):
    """The AXTree section of a BC prompt, or "" if it has none.

    Starts at `RootWebArea`, not at the `## AXTree:` header: two prose "Note:"
    paragraphs sit between them, and counting their words as roles put a
    spurious `note` and `present` into every histogram on this benchmark.
    """
    i = user.find(_AX_START)
    if i < 0:
        return ""
    root = user.find("RootWebArea", i)
    if root < 0:
        return ""
    j = len(user)
    for marker in _AX_END:
        k = user.find(marker, root)
        if 0 <= k < j:
            j = k
    return user[root:j]


def axtree_role_counts(user):
    """{role: count} over the AXTree in a BC prompt, in ROLE_VOCAB spelling."""
    counts = {}
    for line in axtree_block(user).splitlines():
        m = _AX_LINE.match(line)
        if not m:
            continue
        role = m.group(1).lower()
        role = AXTREE_ROLE_ALIASES.get(role, role)
        counts[role] = counts.get(role, 0) + 1
    return counts


def axtree_vector(user):
    """Fixed-order, log1p-ed, L2-normalised role vector for one observation.

    Same transform `encoder.StructureConditioner` applies to a capture-derived
    histogram, reimplemented here rather than imported so this path stays
    usable on the analysis interpreter (encoder.py pulls in torchvision).
    """
    from . import capture
    v = np.asarray(capture.role_vector(axtree_role_counts(user)),
                   dtype=np.float32)
    v = np.log1p(np.clip(v, 0, None))
    n = float(np.linalg.norm(v))
    return (v / n) if n > 0 else v


def attach_axtree_conditioning(samples, level="episode"):
    """Fill `sample.struct` in place. Returns (n_distinct, n_episodes).

    `level="step"` gives each sample its own step's histogram -- thousands of
    conditioning points, but the generator is then trained on a distribution
    (mid-episode pages) it will never be served on (a landing page).
    `level="episode"` gives every sample of an episode its episode's FIRST
    observation, which is exactly the object available at serving time, at the
    cost of ~125 distinct points per version instead of thousands.
    """
    if level not in ("episode", "step"):
        raise ValueError("axtree level must be 'episode' or 'step'")
    vecs = [axtree_vector(s.user) for s in samples]
    n_ep = 0
    if level == "step":
        for s, v in zip(samples, vecs):
            s.struct = v
    else:
        current = None
        for i, s in enumerate(samples):
            if _FIRST_STEP.search(s.user) or current is None:
                current = vecs[i]
                n_ep += 1
            s.struct = current
    distinct = len(set(tuple(np.round(s.struct, 6)) for s in samples))
    return distinct, n_ep


def load_corpus(corpus, cell_list=None, max_per_cell=None):
    """Load per-cell ShareGPT corpora into flat `BCSample`s.

    `corpus` is either the manifest.json written by
    `bcdata.build_cell_corpora` or a directory containing `<cell_key>.json`.
    A *pooled* corpus cannot be used: the converter concatenates cells without
    labels, and per-example conditioning needs the label.
    """
    files = {}
    if os.path.isdir(corpus):
        man = os.path.join(corpus, "manifest.json")
        corpus = man if os.path.exists(man) else corpus
    if os.path.isfile(corpus):
        with open(corpus) as fh:
            man = json.load(fh)
        if "cells" not in man:
            raise ValueError(
                "%s is not a build_cell_corpora manifest. train_hypernet needs "
                "per-cell corpora so each sample keeps its conditioning cell."
                % corpus)
        for key, rec in man["cells"].items():
            files[key] = rec["json"]
    else:
        for name in sorted(os.listdir(corpus)):
            if name.endswith(".json") and not name.endswith(".stats.json"):
                files[name[:-5]] = os.path.join(corpus, name)

    want = None
    if cell_list is not None:
        want = set(c.key if hasattr(c, "key") else c for c in cell_list)
    out, per_cell = [], {}
    for key in sorted(files):
        if want is not None and key not in want:
            continue
        with open(files[key]) as fh:
            data = json.load(fh)
        n = 0
        for rec in data:
            user = assistant = None
            for turn in rec.get("conversations", []):
                if turn.get("from") == "human" and user is None:
                    user = turn.get("value", "")
                elif turn.get("from") == "gpt" and assistant is None:
                    assistant = turn.get("value", "")
            if not user or not assistant:
                continue
            out.append(BCSample(key, rec.get("system", ""), user, assistant,
                                rec.get("images")))
            n += 1
            if max_per_cell and n >= max_per_cell:
                break
        per_cell[key] = n
    if not out:
        raise ValueError("no samples loaded from %s for cells %s"
                         % (corpus, sorted(want) if want else "all"))
    missing = sorted(want - set(per_cell)) if want else []
    if missing:
        print("WARNING: no corpus for %d requested cell(s): %s. They will get "
              "no gradient (era 1 is the usual culprit -- see "
              "`python -m adaptercl.bcdata census`)." % (len(missing), missing))
    return out, per_cell


# -- LLaMA-Factory backbone templates, reproduced --------------------------

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
#: template.py:554 -- the default thought words, and therefore the empty-CoT
#: string LF prepends to the *response* when the completion has no thinking.
THOUGHT_OPEN, THOUGHT_CLOSE = "<think>\n", "\n</think>\n\n"
EMPTY_THOUGHT = THOUGHT_OPEN + THOUGHT_CLOSE
LLAMA3_START = "<|start_header_id|>"
LLAMA3_END = "<|end_header_id|>"
LLAMA3_EOT = "<|eot_id|>"


def format_sample(sample, enable_thinking=True, template="qwen3_5"):
    """(prompt_text, response_text) as the named LLaMA-Factory template renders.

    The llama3 BOS is a token id rather than text and is prepended by
    `encode_sample`. Keeping that distinction matches Template._encode even for
    tokenizers whose BOS spelling is not `<|begin_of_text|>`.
    """
    if template == "qwen3_5":
        prompt = ""
        if sample.system:
            prompt += "%ssystem\n%s%s\n" % (IM_START, sample.system, IM_END)
        prompt += "%suser\n%s%s\n%sassistant\n" % (
            IM_START, sample.user, IM_END, IM_START)
        response = sample.assistant + IM_END + "\n"
        if ("<think>" not in sample.assistant
                and "</think>" not in sample.assistant and enable_thinking):
            response = EMPTY_THOUGHT + response
        return prompt, response
    if template == "llama3":
        prompt = ""
        if sample.system:
            prompt += "%ssystem%s\n\n%s%s" % (
                LLAMA3_START, LLAMA3_END, sample.system, LLAMA3_EOT)
        prompt += "%suser%s\n\n%s%s%sassistant%s\n\n" % (
            LLAMA3_START, LLAMA3_END, sample.user, LLAMA3_EOT,
            LLAMA3_START, LLAMA3_END)
        return prompt, sample.assistant + LLAMA3_EOT
    raise ValueError("unsupported LLaMA-Factory template %r" % template)


def encode_sample(tokenizer, sample, max_len, enable_thinking=True,
                  vocab_clamp=None, template="qwen3_5"):
    """Token ids + labels, with loss on the assistant turn only.

    Truncation policy: keep the *whole* response and drop tokens off the
    **front** of the prompt. That differs from LLaMA-Factory, which shrinks
    source and target proportionally; keeping the response whole matters here
    because the response is the entire training signal and is only ~100 tokens,
    while the AXTree prompt is thousands. Documented rather than silent.
    """
    prompt, response = format_sample(sample, enable_thinking=enable_thinking,
                                     template=template)
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if (template == "llama3"
            and getattr(tokenizer, "bos_token_id", None) is not None):
        p_ids = [tokenizer.bos_token_id] + list(p_ids)
    r_ids = tokenizer.encode(response, add_special_tokens=False)
    if len(r_ids) > max_len // 2:
        r_ids = r_ids[: max_len // 2]
    room = max_len - len(r_ids)
    if len(p_ids) > room:
        p_ids = p_ids[len(p_ids) - room:]
    ids = list(p_ids) + list(r_ids)
    labels = [-100] * len(p_ids) + list(r_ids)
    if vocab_clamp:
        ids = [i % vocab_clamp for i in ids]
        labels = [(l % vocab_clamp) if l != -100 else -100 for l in labels]
    return ids, labels


def collate(batch_ids, batch_labels, pad_id, device):
    n = max(len(x) for x in batch_ids)
    input_ids, labels, mask = [], [], []
    for ids, lab in zip(batch_ids, batch_labels):
        pad = n - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        mask.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(mask, dtype=torch.long, device=device),
    }


class BatchStream(object):
    """Infinite shuffled stream of encoded micro-batches.

    Batches deliberately mix cells: that is what forces per-example conditioning
    (correction 1 above) and what the diversity penalty measures across.
    """

    def __init__(self, samples, tokenizer, cfg, seed=None, shuffle=True):
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.rng = random.Random(cfg.seed if seed is None else seed)
        self.shuffle = shuffle
        # Resolve from the actual backbone for every stream. Unknown families
        # must fail here: silently training Llama data in Qwen delimiters gives
        # an ordinary-looking loss curve and a useless adapter.
        self.template = percell.template_for(cfg.base_model)
        self.pad_id = (getattr(tokenizer, "pad_token_id", None)
                       or getattr(tokenizer, "eos_token_id", None) or 0)
        if cfg.vocab_clamp:
            # smoke only: the pad id has to live in the tiny vocabulary too.
            self.pad_id %= cfg.vocab_clamp
        self._order = []
        self.epochs = 0

    def _next_index(self):
        if not self._order:
            self._order = list(range(len(self.samples)))
            if self.shuffle:
                self.rng.shuffle(self._order)
            self.epochs += 1
        return self._order.pop()

    def next_batch(self, device, batch_size=None):
        bs = batch_size or self.cfg.batch_size
        ids, labels, keys, acts, structs = [], [], [], [], []
        while len(ids) < bs:
            s = self.samples[self._next_index()]
            i, l = encode_sample(self.tokenizer, s, self.cfg.max_len,
                                 vocab_clamp=self.cfg.vocab_clamp,
                                 template=self.template)
            if sum(1 for x in l if x != -100) < self.cfg.min_target_tokens:
                continue
            ids.append(i)
            labels.append(l)
            keys.append(s.cell_key)
            acts.append(s.action)
            structs.append(s.struct)
        batch = collate(ids, labels, self.pad_id, device)
        batch["cell_keys"] = keys
        batch["actions"] = acts
        if all(x is not None for x in structs):
            batch["struct"] = torch.as_tensor(
                np.stack(structs), dtype=torch.float32, device=device)
        return batch

    def epoch_batches(self, device, limit=None, batch_size=None,
                      group_by_cell=True):
        """Deterministic single pass, for evaluation.

        `group_by_cell` does two things 6.6 needs. Each batch holds samples from
        exactly one cell, so the per-cell loss attribution is exact rather than
        "the mixed batch's mean, credited to every cell in it"; and the batches
        are emitted round-robin across cells, so a `limit`-truncated eval still
        touches every held-out cell instead of just the alphabetically first.
        """
        bs = batch_size or self.cfg.batch_size
        groups = {}
        for s in self.samples:
            key = s.cell_key if group_by_cell else "_all"
            groups.setdefault(key, []).append(s)
        chunks = {}
        for key, items in groups.items():
            chunks[key] = [items[i:i + bs] for i in range(0, len(items), bs)]
        order, keys, i = [], sorted(chunks), 0
        while any(chunks[k] for k in keys):
            k = keys[i % len(keys)]
            i += 1
            if chunks[k]:
                order.append(chunks[k].pop(0))

        out = []
        for chunk in order:
            ids, labels, ckeys, acts, structs = [], [], [], [], []
            for s in chunk:
                a, l = encode_sample(self.tokenizer, s, self.cfg.max_len,
                                     vocab_clamp=self.cfg.vocab_clamp,
                                     template=self.template)
                if sum(1 for x in l if x != -100) < self.cfg.min_target_tokens:
                    continue
                ids.append(a)
                labels.append(l)
                ckeys.append(s.cell_key)
                acts.append(s.action)
                structs.append(s.struct)
            if not ids:
                continue
            b = collate(ids, labels, self.pad_id, device)
            b["cell_keys"] = ckeys
            b["actions"] = acts
            if all(x is not None for x in structs):
                b["struct"] = torch.as_tensor(
                    np.stack(structs), dtype=torch.float32, device=device)
            out.append(b)
            if limit and len(out) >= limit:
                break
        return out


# --------------------------------------------------------------------------
# Conditioning
# --------------------------------------------------------------------------

def hash_conditioning(cell_keys, d_cond=64, seed=0):
    """Deterministic per-cell vector with no perceptual content.

    This is *not* a conditioner -- it is 6.7's version-identity control. If the
    hypernetwork does as well with this as with the vision encoder, the encoder
    is contributing nothing and the model is a 18-entry lookup table.
    """
    out = {}
    for key in cell_keys:
        h = hashlib.sha1(("%s|%d" % (key, seed)).encode("utf-8")).digest()
        g = torch.Generator().manual_seed(int.from_bytes(h[:8], "big")
                                          % (2 ** 31 - 1))
        v = torch.randn(d_cond, generator=g)
        out[key] = v / v.norm()
    return out


def axtree_conditioning(samples, cell_keys, level="episode"):
    """{cell_key: mean role vector} + per-sample `struct`, from the corpus.

    The per-cell mean is what gets SERVED: a generated adapter is constant for
    a whole episode (4.3), so materialisation needs one vector per version. The
    per-sample vectors are what the generator is TRAINED on. Both come from the
    same parser on the same channel, so the train/serve gap is a distribution
    shift within one modality rather than a change of modality.
    """
    attach_axtree_conditioning(samples, level=level)
    by_cell = {}
    for s in samples:
        by_cell.setdefault(s.cell_key, []).append(s.struct)
    out = {}
    for k, vs in by_cell.items():
        m = np.mean(np.stack(vs), axis=0)
        n = float(np.linalg.norm(m))
        out[k] = torch.as_tensor(m / n if n > 0 else m, dtype=torch.float32)
    missing = [k for k in cell_keys if k not in out]
    if missing:
        raise KeyError(
            "no corpus samples for %s, so no AXTree conditioning vector. The "
            "held-out version needs its OWN corpus loaded for the landing-page "
            "histogram -- that is an observation, not a demonstration, but it "
            "must be present." % missing)
    return out


def build_conditioning(cfg, cell_keys, device=None):
    """{cell_key: (d_cond,) tensor}. Frozen -- conditioning never trains (4.2).

    Three sources, in the order you should prefer them:

    1. `cfg.embeddings` -- a `.npz` written by `encoder.save_embeddings`. This is
       the normal route: the vision tower is a GPU-sized frozen encoder and its
       output is constant per cell (4.3), so it is computed once by
       `adaptercl.encoder`/`capture` and reused by every run.
    2. `cfg.conditioning in {"vision","structure","both"}` -- call
       `encoder.embed_cells` live. Vision needs one reference capture per cell;
       we resolve those through `bcdata.default_image_for_cell` (paths.OUT_CAPTURE)
       and fail loudly rather than substituting anything.
    3. `"hash"` -- the 6.7 version-identity control. Not a conditioner.
    4. `"none"` -- one CONSTANT vector for every cell (T2L_PLAN E5). Not a
       control on the encoder but on generation itself: with identical
       conditioning the generator can only emit one adapter, so this arm is "a
       single generated adapter for all versions". If it matches the
       conditioned arms, nothing is being conditioned on.
    """
    cell_keys = list(cell_keys)
    if cfg.conditioning == "axtree":
        raise ValueError(
            "conditioning 'axtree' is computed from the corpus, not from a "
            "capture; train() calls axtree_conditioning() with the loaded "
            "samples instead of this function.")
    if cfg.conditioning == "none":
        d = int(cfg.d_cond or 64)
        v = torch.ones(d, dtype=torch.float32) / math.sqrt(d)
        return dict((k, v.clone().to(device) if device is not None else v.clone())
                    for k in cell_keys)
    if cfg.embeddings:
        from . import encoder
        raw, spec = encoder.load_embeddings(cfg.embeddings)
        missing = [k for k in cell_keys if k not in raw]
        if missing:
            raise KeyError("%s has no embedding for %s (spec=%s)"
                           % (cfg.embeddings, missing, spec))
        emb = dict((k, raw[k]) for k in cell_keys)
    elif cfg.conditioning == "hash":
        emb = hash_conditioning(cell_keys, d_cond=cfg.d_cond or 64,
                                seed=cfg.seed)
    else:
        try:
            from . import bcdata, encoder
        except ImportError as exc:
            raise ImportError(
                "conditioning %r needs adaptercl/encoder.py, which is not "
                "importable (%s). Use --conditioning hash for the "
                "version-identity control." % (cfg.conditioning, exc))
        kw = dict(cfg.conditioning_kwargs)
        image_root = kw.pop("image_root", None)
        structure_vectors = kw.pop("structure_vectors", None)
        if isinstance(structure_vectors, str):
            with open(structure_vectors) as fh:
                structure_vectors = json.load(fh)
        spec = encoder.ConditioningSpec(cfg.conditioning, **kw)
        cell_objs = [cells.Cell.parse(k) for k in cell_keys]
        image_paths = None
        if spec.uses_vision:
            image_paths = {}
            for c in cell_objs:
                p = bcdata.default_image_for_cell(c, root=image_root)
                if p is None:
                    raise IOError(
                        "no reference capture for cell %s under %s. The 4.2 "
                        "vision conditioning needs one screenshot per cell; the "
                        "TimeTraj trajectories contain none (see bcdata's "
                        "docstring). Run the capture pass first, or train with "
                        "--conditioning hash to sanity-check the plumbing."
                        % (c.key, image_root or paths.OUT_CAPTURE))
                image_paths[c.key] = p
        emb = encoder.embed_cells(cell_objs, image_paths=image_paths, spec=spec,
                                  structure_vectors=structure_vectors,
                                  seed=cfg.seed)
    out = {}
    for k, v in emb.items():
        t = torch.as_tensor(v, dtype=torch.float32)
        out[k] = t.to(device) if device is not None else t
    missing = [k for k in cell_keys if k not in out]
    if missing:
        raise KeyError("no conditioning vector for %s" % missing)
    return out


def stack_conditioning(cond, cell_keys):
    return torch.stack([cond[k] for k in cell_keys], dim=0)


def conditioning_for_batch(cond, batch, gen=None):
    """(Bsz, d_cond) for one batch.

    Normally one vector per cell (4.3: the conditioning is constant for a whole
    episode, which is what makes a generated adapter servable). When the batch
    carries a per-sample `struct` vector -- `--conditioning axtree`, T2L_PLAN
    E5's escape from N=5 -- that is used instead, so the generator sees
    thousands of distinct conditioning points during training while serving is
    unchanged.
    """
    st = batch.get("struct")
    if st is not None:
        return st
    return stack_conditioning(cond, batch["cell_keys"])


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

class FallbackGenerator(nn.Module):
    """A minimal shared-trunk generator (8.3), used only when
    `adaptercl/hypernet.py` is unavailable and `allow_fallback_generator` is set.

    Real runs must use hypernet.build_generator -- 6.1 argues the learned-basis
    model, not free generation, is the honest headline method, and that lives
    there. This exists so `--smoke` can exercise the loop end to end.

    Heads are shared per (d_in, d_out) group: a hybrid stack has several distinct
    projection shapes, and one head per shape is the padding-free version of
    8.3's "you'll need per-module-type heads".
    """

    def __init__(self, sites, rank, d_cond, d_hidden=64, d_emb=16):
        super(FallbackGenerator, self).__init__()
        self.rank = int(rank)
        self.sites = list(sites)
        n_layers = max(s.layer for s in sites) + 1
        n_modules = len(targets.MODULE_TYPES)
        self.layer_emb = nn.Embedding(n_layers, d_emb)
        self.module_emb = nn.Embedding(n_modules, d_emb)
        self.trunk = nn.Sequential(
            nn.Linear(d_cond + 2 * d_emb, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU())
        self.shapes = sorted(set((s.d_in, s.d_out) for s in sites))
        self.head_A = nn.ModuleDict()
        self.head_B = nn.ModuleDict()
        for d_in, d_out in self.shapes:
            k = "%dx%d" % (d_in, d_out)
            self.head_A[k] = nn.Linear(d_hidden, self.rank * d_in)
            b = nn.Linear(d_hidden, d_out * self.rank)
            nn.init.zeros_(b.weight)          # 8.3: start as a no-op adapter
            nn.init.zeros_(b.bias)
            self.head_B[k] = b

    def forward(self, h, layer_ids, module_ids):
        if h.dim() == 1:
            h = h.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        bsz = h.shape[0]
        A_list, B_list = [], []
        for q, site in enumerate(self.sites):
            e = torch.cat([h,
                           self.layer_emb(layer_ids[q]).expand(bsz, -1),
                           self.module_emb(module_ids[q]).expand(bsz, -1)],
                          dim=-1)
            z = self.trunk(e)
            k = "%dx%d" % (site.d_in, site.d_out)
            A = self.head_A[k](z).view(bsz, self.rank, site.d_in)
            B = self.head_B[k](z).view(bsz, site.d_out, self.rank)
            if squeeze:
                A, B = A[0], B[0]
            A_list.append(A)
            B_list.append(B)
        return A_list, B_list


def _train_bank(cfg):
    """(adapter dirs, keys) for the TRAIN units only -- the leak guard.

    Including an eval unit would put the held-out version's OWN LoRA in the
    bank, and a leave-one-out run could score by simply selecting it: the
    zero-shot claim (6.1) would be measuring a lookup, not synthesis. With v1
    held out the bank must be v2..v6 and the generator must COMPOSE an adapter
    for an era it has never seen. Used by the mixture bank and by
    `fixed_A="bank_mean"`.
    """
    tr, ev = cfg.resolved_cells()
    leaked = [u.key for u in ev if u in tr]
    if leaked:
        raise ValueError(
            "eval units %s are also train units; a bank built from them leaks "
            "the held-out adapter" % (leaked,))
    bank, missing = [], []
    for u in list(tr):
        d = os.path.join(paths.OUT_CELLS, cfg.bank_tag, u.key)
        (bank.append(d) if os.path.isfile(
            os.path.join(d, "adapter_config.json")) else missing.append(u.key))
    if not bank:
        raise ValueError(
            "no Phase-2 LoRA bank under %s/%s/<unit>. Train them first "
            "(scripts/run_version_train.sh)."
            % (paths.OUT_CELLS, cfg.bank_tag))
    if missing:
        print("  bank: %d adapter(s), MISSING %s"
              % (len(bank), ", ".join(missing)))
    return bank, [os.path.basename(d) for d in bank]


def build_generator(cfg, sites, d_cond, device=None, n_layers=None):
    """hypernet.build_generator(kind, sites, rank, d_cond, **kw), lazily.

    `n_layers` is forwarded as the *model's* layer count, not the target set's:
    hypernet's layer embedding then has an entry for every layer, so a
    checkpoint survives an injection-site change (hypernet.py:296-305).
    """
    gen = None
    try:
        from . import hypernet
    except ImportError as exc:
        hypernet = None
        why = str(exc)
    if hypernet is not None:
        kw = dict(cfg.generator_kwargs)
        if n_layers and "n_layers" not in kw:
            kw["n_layers"] = int(n_layers)
        # fixed_A='bank_mean' needs the same TRAIN-ONLY bank the mixture uses,
        # for the same reason: the held-out version's own A must not reach the
        # generator, even through a mean (T2L_PLAN E3).
        if kw.get("fixed_A") == "bank_mean" and "bank_adapters" not in kw:
            bank, keys = _train_bank(cfg)
            kw["bank_adapters"] = bank
            kw["bank_keys"] = keys
            print("  fixed_A = mean of the TRAIN bank: %s" % ", ".join(keys))
        # The mixture generator predicts coefficients over the Phase-2 LoRA
        # bank, so it needs those adapter dirs. Default to the per-version
        # adapters trained by run_version_train.sh, in a FIXED unit order so the
        # coefficient vector is interpretable (index k == unit k) -- that
        # interpretability is the whole reason to run mixture first (6.5).
        if cfg.generator == "mixture" and "adapters" not in kw:
            bank, keys = _train_bank(cfg)
            kw["adapters"] = bank
            kw["keys"] = keys
            print("  mixture bank (TRAIN units only): %s" % ", ".join(keys))
        gen = hypernet.build_generator(cfg.generator, sites, cfg.rank, d_cond,
                                       **kw)
    elif cfg.allow_fallback_generator:
        print("WARNING: adaptercl/hypernet.py is unavailable (%s); using "
              "train_hypernet.FallbackGenerator. Never report a number from "
              "this generator." % why)
        gen = FallbackGenerator(sites, cfg.rank, d_cond,
                                **cfg.generator_kwargs)
    else:
        raise ImportError(
            "adaptercl/hypernet.py is required for generator %r but is not "
            "importable (%s). Pass allow_fallback_generator=True (or --smoke) "
            "only for plumbing tests." % (cfg.generator, why))
    return gen.to(device) if device is not None else gen


def generator_factors(gen, h, layer_ids, module_ids):
    """Call a generator and sanity-check what came back."""
    A_list, B_list = gen(h, layer_ids, module_ids)
    if len(A_list) != len(B_list) or len(A_list) != layer_ids.shape[0]:
        raise ValueError(
            "generator returned %d A / %d B for %d query points -- the site "
            "order contract in inject.InjectionHandle is violated"
            % (len(A_list), len(B_list), layer_ids.shape[0]))
    return A_list, B_list


# --------------------------------------------------------------------------
# Diversity penalty (6.7 "diversity collapse")
# --------------------------------------------------------------------------

def diversity_penalty(A_list, B_list, cell_keys, max_sites=16, kind="cosine"):
    """Diversity of the generated adapters across the examples of a batch.

    Prefers `hypernet.diversity_penalty` (sketch-based, and the version the
    Phase-3 diagnostics report) and falls back to the exact Frobenius-cosine
    below when hypernet is unavailable. Returns None when the batch has fewer
    than two distinct cells, because "these two adapters are identical" is
    trivially true when the conditioning is identical.
    """
    keys = list(cell_keys)
    if len(keys) < 2 or len(set(keys)) < 2 or A_list[0].dim() != 3:
        return None
    try:
        from . import hypernet
    except ImportError:
        hypernet = None
    if hypernet is not None:
        try:
            return hypernet.diversity_penalty(A_list, B_list, kind=kind)
        except ValueError:
            return None
    return _exact_diversity_penalty(A_list, B_list, keys, max_sites=max_sites)


def _exact_diversity_penalty(A_list, B_list, cell_keys, max_sites=16):
    """Mean pairwise cosine similarity of the generated `dW`, across cells.

    Weight-space comparison is valid *here* and only here (6.2): these adapters
    come from one generator and share a parametrization. The Frobenius inner
    product of two rank-r deltas is computed in factored form --
    <B_i A_i, B_j A_j>_F = sum((B_i^T B_j) * (A_i A_j^T)) -- so it costs O(r^2 d)
    rather than materialising d_out x d_in.

    Returns 0 when the batch has fewer than two distinct cells.
    """
    keys = list(cell_keys)
    if len(keys) < 2 or len(set(keys)) < 2:
        return None
    n = len(keys)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if keys[i] != keys[j]]
    if not pairs:
        return None
    stride = max(1, len(A_list) // max_sites)
    total, count = None, 0
    for q in range(0, len(A_list), stride):
        A, B = A_list[q], B_list[q]
        if A.dim() != 3:
            continue
        # pairwise via explicit index gather (n is small: micro-batch size)
        Ai = A[[i for i, _ in pairs]]
        Aj = A[[j for _, j in pairs]]
        Bi = B[[i for i, _ in pairs]]
        Bj = B[[j for _, j in pairs]]
        cross = (torch.bmm(Bi.transpose(1, 2), Bj)
                 * torch.bmm(Ai, Aj.transpose(1, 2))).sum(dim=(1, 2))
        nii = (torch.bmm(Bi.transpose(1, 2), Bi)
               * torch.bmm(Ai, Ai.transpose(1, 2))).sum(dim=(1, 2))
        njj = (torch.bmm(Bj.transpose(1, 2), Bj)
               * torch.bmm(Aj, Aj.transpose(1, 2))).sum(dim=(1, 2))
        cos = cross / (nii.clamp_min(1e-12).sqrt() * njj.clamp_min(1e-12).sqrt()
                       + 1e-12)
        total = cos.mean() if total is None else total + cos.mean()
        count += 1
    if not count:
        return None
    return total / count


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------

def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    percell.template_for(cfg.base_model)
    return AutoTokenizer.from_pretrained(paths.model_dir_or_id(cfg.base_model),
                                         trust_remote_code=True)


def load_base_model(cfg):
    """Load the frozen policy. Returns (model, text_config_dict)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    template = percell.template_for(cfg.base_model)
    path = paths.model_dir_or_id(cfg.base_model)
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    cfg_obj = AutoConfig.from_pretrained(path, trust_remote_code=True)
    if template == "llama3":
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, trust_remote_code=True)
        text = cfg_obj
    elif template == "qwen3_5":
        # Qwen3.5 is a VL composite; AutoModelForCausalLM resolves to
        # Qwen3_5ForCausalLM, which expects the *text* config. If that path
        # fails, load the conditional-generation class (vision tower included --
        # it is frozen and also serves as the 4.2 conditioning encoder).
        try:
            model = AutoModelForCausalLM.from_pretrained(
                path, dtype=dtype, trust_remote_code=True)
        except Exception as exc:                   # noqa: BLE001
            print("AutoModelForCausalLM failed (%s); falling back to "
                  "AutoModelForImageTextToText." % exc)
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                path, dtype=dtype, trust_remote_code=True)
        text = getattr(cfg_obj, "text_config", None) or cfg_obj
    else:                                           # template_for is exhaustive
        raise ValueError("unsupported template %r for %s"
                         % (template, cfg.base_model))
    return model, (text.to_dict() if hasattr(text, "to_dict") else dict(text))


def build_tiny_model(vocab_size=1024, hidden=64, n_layers=4, dtype=None):
    """A ~200k-parameter Qwen3.5 for CPU smoke tests.

    Two constraints discovered by running it:

    * **All layers must be `full_attention`.** The Gated-DeltaNet path calls
      `fla`'s triton kernels unconditionally and dies with "Pointer argument
      cannot be accessed from Triton (cpu tensor?)" on CPU. A tiny model
      therefore exercises the attention + MLP sites only, which is exactly what
      `--target-set attn_mlp` injects.
    * **`use_cache=False` is mandatory.** `Qwen3_5TextModel.forward` builds a
      `DynamicCache` from the config and then calls `has_previous_state()` on it,
      which raises for a stack with no linear-attention layers
      (transformers/models/qwen3_5/modeling_qwen3_5.py:1267,1300). Training
      passes `use_cache=False` anyway.
    """
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    path = paths.model_dir_or_id()
    cfg_obj = AutoConfig.from_pretrained(path, trust_remote_code=True)
    tc = copy.deepcopy(getattr(cfg_obj, "text_config", cfg_obj))
    overrides = dict(
        hidden_size=hidden, num_hidden_layers=n_layers, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=2 * hidden,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        vocab_size=vocab_size, layer_types=["full_attention"] * n_layers,
        full_attention_interval=1, max_position_embeddings=4096,
        mtp_num_hidden_layers=0, tie_word_embeddings=False)
    for k, v in overrides.items():
        setattr(tc, k, v)
    model = Qwen3_5ForCausalLM._from_config(tc, dtype=dtype or torch.float32)
    return model, tc.to_dict()


class ByteTokenizer(object):
    """Last-resort tokenizer for --smoke when the HF snapshot is unavailable."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + (b % 250) for b in text.encode("utf-8", "replace")]


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def _lr_at(cfg, step):
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return cfg.lr * float(step + 1) / cfg.warmup_steps
    if cfg.max_steps <= cfg.warmup_steps:
        return cfg.lr
    t = float(step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return 0.5 * cfg.lr * (1.0 + math.cos(math.pi * min(1.0, t)))


def sample_weights(batch, cfg, device):
    """Per-example loss weights for E9's termination re-weighting.

    `term_weight=1` returns None, which keeps the ordinary
    `model(labels=...)` path -- that path uses the fused loss kernel and is
    both faster and lower-memory than materialising logits ourselves, so the
    manual path is entered only when a run actually asks for re-weighting.
    """
    if float(cfg.term_weight) == 1.0:
        return None
    acts = batch.get("actions")
    if acts is None:
        raise ValueError("term_weight != 1 needs per-sample actions; rebuild "
                         "the batch with BatchStream (it attaches them)")
    w = [float(cfg.term_weight) if a == cfg.term_action else 1.0 for a in acts]
    return torch.tensor(w, dtype=torch.float32, device=device)


def weighted_token_loss(logits, labels, weights=None, label_smoothing=0.0):
    """Mean NLL over supervised tokens, each token weighted by its example.

    Two things this does that `model(labels=...)` does not, both load-bearing:

    1. **It gathers the supervised positions BEFORE upcasting.** HF's
       `ForCausalLMLoss` calls `logits.float()` on the whole `(B, T, V)`
       tensor, which at T=8192 and V=152k is 5 GB of transient float32 on top
       of the 2.5 GB of bf16 logits -- enough to OOM an 80 GB A100 mid-run even
       though step 0 succeeded (it dies on the first long sample, not the
       first sample). An assistant turn is ~100-300 tokens, so upcasting only
       those is three orders of magnitude smaller and the same arithmetic.
    2. **It can weight tokens** (T2L_PLAN E9). Per TOKEN rather than per
       example on purpose: a `send_msg_to_user` completion is long (it carries
       the answer) and an ordinary action completion is short, so an
       example-level mean would already give termination steps a smaller
       per-token share. `term_weight` then means exactly "each termination
       token counts w times", which is the quantity the ablation is about.

    `weights=None` reproduces the model's own fused loss (checked to 1e-4 in
    tests/test_t2l_plan.py), so the term-weight ablation is not confounded with
    a change of loss implementation.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    n_tok = mask.sum(dim=1)
    per_ex = []
    for i in range(labels.shape[0]):
        idx = mask[i].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            per_ex.append(shift_logits.new_zeros(()))
            continue
        lg = shift_logits[i].index_select(0, idx).float()
        tg = shift_labels[i].index_select(0, idx)
        if float(label_smoothing) == 0.0:
            # Keep the old call byte-for-byte on the default path. Besides
            # preserving the result, this avoids depending on version-specific
            # smoothing details when the feature is disabled.
            val = torch.nn.functional.cross_entropy(lg, tg, reduction="sum")
        else:
            val = torch.nn.functional.cross_entropy(
                lg, tg, reduction="sum", label_smoothing=label_smoothing)
        per_ex.append(val)
    per_ex = torch.stack(per_ex)
    if weights is None:
        return per_ex.sum() / n_tok.sum().clamp(min=1)
    w = weights.to(per_ex.dtype)
    return (w * per_ex).sum() / (w * n_tok).sum().clamp(min=1)


def generated_factor_l2(A_list, B_list):
    """Mean square over every scalar generated for the sites in this step."""
    factors = list(A_list) + list(B_list)
    if not factors:
        raise ValueError("cannot compute generated-factor L2 over no factors")
    total = factors[0].square().sum(dtype=torch.float32)
    n = factors[0].numel()
    for x in factors[1:]:
        total = total + x.square().sum(dtype=torch.float32)
        n += x.numel()
    return total / float(n)


def forward_loss(model, handle, gen, cond, batch, cfg, layer_ids, module_ids,
                 retain_for_backward=False):
    """One conditioned forward, retaining factors through checkpoint backward.

    Non-reentrant gradient checkpointing recomputes decoder layers in backward.
    Clearing immediately after forward makes that recomputation base-only and
    PyTorch correctly rejects the changed graph. The training loop clears after
    each micro-batch backward; evaluation and non-checkpointed forwards still
    clear here, as does every exception path.
    """
    h = conditioning_for_batch(cond, batch, gen)
    A_list, B_list = generator_factors(gen, h, layer_ids, module_ids)
    weights = sample_weights(batch, cfg, batch["input_ids"].device)
    keep_for_backward = False
    try:
        handle.set_factors(A_list, B_list)
        # Never pass `labels=`: see weighted_token_loss. The fused path costs
        # 5 GB of transient float32 at 8192 tokens and OOMs an 80 GB card
        # partway through a run, which reads as a flaky node rather than as a
        # memory bug.
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"], use_cache=False)
        task = weighted_token_loss(out.logits, batch["labels"], weights,
                                   label_smoothing=cfg.label_smoothing)
        loss = task
        div = None
        if cfg.diversity_weight:
            div = diversity_penalty(A_list, B_list, batch["cell_keys"],
                                    kind=cfg.diversity_kind)
            if div is not None:
                loss = loss + cfg.diversity_weight * div
        gen_l2 = None
        if cfg.gen_l2:
            gen_l2 = float(cfg.gen_l2) * generated_factor_l2(A_list, B_list)
            loss = loss + gen_l2
        keep_for_backward = bool(retain_for_backward
                                 and torch.is_grad_enabled() and gen.training)
        return loss, (float(task.detach()),
                      float(div.detach()) if div is not None else None,
                      float(gen_l2.detach()) if gen_l2 is not None else None)
    finally:
        if not keep_for_backward:
            handle.clear()


def evaluate_loss(model, handle, gen, cond, batches, cfg, layer_ids,
                  module_ids):
    """Mean task loss over held-out cells (no diversity term, no grad)."""
    was_training = gen.training
    gen.eval()
    model.eval()
    per_cell, total, n = {}, 0.0, 0
    # term_weight=1 at eval on purpose: a re-weighted eval loss is not
    # comparable across arms that used different weights, and E9 has to be
    # judged on the same held-out quantity as every other run.
    eval_cfg = dataclasses.replace(cfg, diversity_weight=0.0, term_weight=1.0,
                                   label_smoothing=0.0, gen_l2=0.0)
    with torch.no_grad():
        for batch in batches:
            _loss, (task, _div, _l2) = forward_loss(
                model, handle, gen, cond, batch, eval_cfg, layer_ids, module_ids)
            total += task
            n += 1
            for k in set(batch["cell_keys"]):
                rec = per_cell.setdefault(k, [0.0, 0])
                rec[0] += task
                rec[1] += 1
    if was_training:
        gen.train()
    model.train()
    return {"loss": (total / n) if n else float("nan"), "n_batches": n,
            "per_cell": dict((k, v[0] / v[1]) for k, v in per_cell.items())}


def generate_and_materialize(cfg, handle, gen, cond, cell_list, out_root=None,
                             meta=None, dtype=None):
    """Run the generator on each cell's conditioning and write PEFT adapters.

    This is the bridge into the eval harness: a generated adapter is constant for
    a whole episode (4.3), so it can be served exactly like a trained LoRA.

    Delegates to `hypernet.generate_for_cells` + `hypernet.materialize_generated`
    when hypernet is importable, because the latter rescales `lora_alpha` when
    the generator's `output_rank` differs from the handle's rank (a mixture in
    `mode="delta"` emits rank K*r). Writing those factors with the handle's alpha
    would serve an adapter K times weaker than the one that was trained --
    silently. The fallback below is only for the no-hypernet smoke path, where
    output_rank == rank by construction.
    """
    out_root = out_root or os.path.join(cfg.out_dir, "generated")
    device = next(gen.parameters()).device
    keys = [c.key if hasattr(c, "key") else c for c in cell_list]
    keys = [k for k in keys if k in cond]
    try:
        from . import hypernet
    except ImportError:
        hypernet = None
    if hypernet is not None and hasattr(hypernet, "materialize_generated"):
        emb = dict((k, cond[k]) for k in keys)
        factors = hypernet.generate_for_cells(gen, emb, handle=handle,
                                              device=device, keys=keys,
                                              dtype=dtype)
        info = {"conditioning": cfg.conditioning, "split": cfg.split,
                "run_name": cfg.run_name, "generated_by":
                "adaptercl.train_hypernet", "trained_rank": int(cfg.rank),
                "trained_alpha": float(cfg.alpha),
                "trained_alpha_over_r": float(cfg.alpha) / float(cfg.rank)}
        info.update(meta or {})
        written = hypernet.materialize_generated(
            factors, handle, out_root, run="", generator=gen,
            base_model=cfg.base_model, meta=info, dtype=dtype,
            # ONE alpha convention (T2L_PLAN Part D). The handle here is the
            # training handle, so this is trivially satisfied -- which is the
            # point: it fails loudly if anyone ever materialises through a
            # differently-scaled handle, as materialize_generated_any.py did.
            trained_scaling=float(cfg.alpha) / float(cfg.rank))
        return dict(written)

    layer_ids, module_ids = handle.query_ids(device=device)
    written = {}
    gen.eval()
    try:
        with torch.no_grad():
            for c in cell_list:
                key = c.key if hasattr(c, "key") else c
                if key not in cond:
                    print("  skip %s: no conditioning vector" % key)
                    continue
                h = cond[key]                       # unbatched -> 2-D factors
                A_list, B_list = generator_factors(gen, h, layer_ids, module_ids)
                A_list = [a[0] if a.dim() == 3 else a for a in A_list]
                B_list = [b[0] if b.dim() == 3 else b for b in B_list]
                handle.set_factors(A_list, B_list)
                d = os.path.join(out_root, key)
                m = {"generated_by": "adaptercl.train_hypernet",
                     "generator": cfg.generator, "cell": key,
                     "conditioning": cfg.conditioning, "split": cfg.split,
                     "run_name": cfg.run_name, "granularity": "per_episode"}
                m.update({"trained_rank": int(cfg.rank),
                          "trained_alpha": float(cfg.alpha),
                          "trained_alpha_over_r": (float(cfg.alpha)
                                                   / float(cfg.rank)),
                          "served_alpha_over_r": (float(handle.alpha)
                                                  / float(handle.rank))})
                m.update(meta or {})
                materialize.write_from_handle(d, handle,
                                              base_model=cfg.base_model,
                                              meta=m, dtype=dtype)
                handle.clear()
                written[key] = d
    finally:
        handle.clear()
        gen.train()
    return written


def train(cfg, model=None, tokenizer=None, samples=None, eval_samples=None,
          text_cfg=None):
    """The 8.4 loop, corrected. Returns a history dict."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    if not os.path.isdir(cfg.out_dir):
        os.makedirs(cfg.out_dir)

    train_cells, eval_cells = cfg.resolved_cells()
    train_keys = set(c.key for c in train_cells)
    eval_keys = set(c.key for c in eval_cells)
    leaked = sorted(train_keys & eval_keys)
    if leaked:
        raise ValueError("eval units are also train units: %s" % leaked)
    device = torch.device(cfg.device if (cfg.device != "cuda"
                                         or torch.cuda.is_available())
                          else "cpu")

    if tokenizer is None:
        tokenizer = load_tokenizer(cfg)
    if model is None:
        model, text_cfg = load_base_model(cfg)
        model.to(device)
    if samples is None:
        if not cfg.corpus:
            raise ValueError("cfg.corpus is required (a build_cell_corpora "
                             "manifest.json)")
        samples, per_cell = load_corpus(cfg.corpus, train_cells,
                                        cfg.max_samples_per_cell)
        print("train corpus: %d samples over %d cells %s"
              % (len(samples), len(per_cell), per_cell))
    if eval_samples is None and cfg.corpus and eval_cells:
        try:
            eval_samples, ev_per_cell = load_corpus(cfg.corpus, eval_cells,
                                                    cfg.max_samples_per_cell)
            print("eval corpus : %d samples over %d cells %s"
                  % (len(eval_samples), len(ev_per_cell), ev_per_cell))
        except ValueError as exc:
            print("no eval corpus (%s)" % exc)
            eval_samples = None

    sample_keys = set(s.cell_key for s in samples)
    leaked_samples = sorted(sample_keys - train_keys)
    if leaked_samples:
        raise ValueError("training samples contain non-train units: %s"
                         % leaked_samples)
    if eval_samples:
        eval_sample_keys = set(s.cell_key for s in eval_samples)
        leaked_eval = sorted(eval_sample_keys - eval_keys)
        if leaked_eval:
            raise ValueError("eval samples contain non-eval units: %s"
                             % leaked_eval)

    # Epochs count optimiser examples, including gradient accumulation. Resolve
    # this only after corpus filtering/max-per-cell so the requested dose and
    # the recorded max_steps cannot disagree.
    if cfg.epochs is not None:
        denom = int(cfg.batch_size) * int(cfg.grad_accum)
        cfg.max_steps = int(math.ceil(float(cfg.epochs) * len(samples) / denom))
    if cfg.max_steps <= 0:
        raise ValueError("max_steps must be positive, got %r" % cfg.max_steps)
    if cfg.warmup_frac is not None:
        cfg.warmup_steps = int(math.ceil(float(cfg.warmup_frac)
                                         * cfg.max_steps))
    cfg.save()

    # -- inject ------------------------------------------------------------
    handle = inject.inject(model, cfg.target_set, rank=cfg.rank,
                           alpha=cfg.alpha, dropout=cfg.dropout,
                           layers=cfg.layers, cfg=text_cfg)
    frozen = inject.check_base_frozen(model)
    if not frozen["ok"]:
        raise RuntimeError("base model is not frozen: %d trainable tensors, "
                           "e.g. %s" % (len(frozen["leaked"]),
                                        frozen["leaked"][:3]))
    print("injected %d sites (%s, r=%d, alpha=%g); base frozen"
          % (len(handle), cfg.target_set, cfg.rank, cfg.alpha))
    if cfg.gradient_checkpointing and hasattr(model,
                                              "gradient_checkpointing_enable"):
        # use_reentrant=False: with every base parameter frozen, the reentrant
        # implementation sees no input requiring grad and skips the backward.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    # -- conditioning + generator -----------------------------------------
    all_keys = sorted(set([c.key for c in train_cells]
                          + [c.key for c in (eval_cells or [])]))
    if cfg.conditioning == "axtree":
        level = (cfg.conditioning_kwargs or {}).get("level", "episode")
        pool = list(samples) + list(eval_samples or [])
        cond = axtree_conditioning(pool, all_keys, level=level)
        cond = dict((k, v.to(device) if device is not None else v)
                    for k, v in cond.items())
        n_distinct = len(set(tuple(np.round(s.struct, 6)) for s in samples))
        print("axtree conditioning (level=%s): %d distinct vectors over %d "
              "training samples, d=%d" % (level, n_distinct, len(samples),
                                          int(cond[all_keys[0]].shape[-1])))
    else:
        cond = build_conditioning(cfg, all_keys, device=device)
    d_cond = int(next(iter(cond.values())).shape[-1])
    if cfg.d_cond and cfg.d_cond != d_cond:
        print("NOTE: cfg.d_cond=%s but the conditioner returned %d; using %d"
              % (cfg.d_cond, d_cond, d_cond))
    cfg.d_cond = d_cond
    n_layers = int((text_cfg or {}).get("num_hidden_layers")
                   or (max(s.layer for s in handle.sites) + 1))
    gen = build_generator(cfg, handle.sites, d_cond, device=device,
                          n_layers=n_layers)
    if cfg.init_checkpoint:
        load_generator_checkpoint(gen, cfg.init_checkpoint,
                                  strict=cfg.init_strict)
    if cfg.train_only:
        freeze_except(gen, cfg.train_only)
    n_gen = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print("generator %s: %s trainable parameters (d_cond=%d, %d sites, "
          "%s adapter params)" % (cfg.generator, "{:,}".format(n_gen), d_cond,
                                  len(handle),
                                  "{:,}".format(targets.site_budget(
                                      handle.sites, cfg.rank))))

    opt = torch.optim.AdamW([p for p in gen.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Guard against 9B of base weights sneaking into the optimizer.
    gen_ids = set(id(p) for p in gen.parameters())
    for group in opt.param_groups:
        for p in group["params"]:
            if id(p) not in gen_ids:
                raise RuntimeError("non-generator parameter in the optimizer")

    layer_ids, module_ids = handle.query_ids(device=device)
    stream = BatchStream(samples, tokenizer, cfg)
    eval_batches = []
    if eval_samples:
        eval_stream = BatchStream(eval_samples, tokenizer, cfg, shuffle=False)
        eval_batches = eval_stream.epoch_batches(device, limit=cfg.eval_batches)

    history = {"steps": [], "eval": [], "config": cfg.to_dict()}
    model.train()
    gen.train()
    t0 = time.time()
    try:
        for step in range(cfg.max_steps):
            for g in opt.param_groups:
                g["lr"] = _lr_at(cfg, step)
            opt.zero_grad(set_to_none=True)
            task_sum, div_sum, l2_sum, n_micro = 0.0, 0.0, 0.0, 0
            for _ in range(cfg.grad_accum):
                batch = stream.next_batch(device)
                try:
                    loss, (task, div, gen_l2) = forward_loss(
                        model, handle, gen, cond, batch, cfg, layer_ids,
                        module_ids,
                        retain_for_backward=cfg.gradient_checkpointing)
                    (loss / cfg.grad_accum).backward()
                finally:
                    # Required after checkpoint recomputation, and harmless on
                    # the ordinary path where forward_loss already cleared.
                    handle.clear()
                task_sum += task
                div_sum += 0.0 if div is None else div
                l2_sum += 0.0 if gen_l2 is None else gen_l2
                n_micro += 1
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in gen.parameters() if p.requires_grad], cfg.grad_clip)
            opt.step()
            rec = {"step": step, "loss": task_sum / n_micro,
                   "div": div_sum / n_micro, "lr": _lr_at(cfg, step),
                   "gen_l2": l2_sum / n_micro,
                   "grad_norm": float(gnorm), "epochs": stream.epochs,
                   "sec": time.time() - t0}
            history["steps"].append(rec)
            if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
                print("step %5d  loss %.4f  div %.4f  gen_l2 %.4g  lr %.2e  "
                      "|g| %.3f  %.1fs"
                      % (step, rec["loss"], rec["div"], rec["gen_l2"],
                         rec["lr"], rec["grad_norm"], rec["sec"]))
                # Flush explicitly. Redirected to a file, stdout is block
                # buffered, so a healthy multi-hour run writes NOTHING for its
                # first few thousand steps and is indistinguishable from a hung
                # one -- three live runs at 100 % GPU were nearly killed for
                # looking dead.
                sys.stdout.flush()
            if eval_batches and cfg.eval_every and \
                    (step + 1) % cfg.eval_every == 0:
                ev = evaluate_loss(model, handle, gen, cond, eval_batches, cfg,
                                   layer_ids, module_ids)
                ev["step"] = step
                history["eval"].append(ev)
                print("  eval loss %.4f over %d batches %s"
                      % (ev["loss"], ev["n_batches"], ev["per_cell"]))
                sys.stdout.flush()
            if cfg.save_every and (step + 1) % cfg.save_every == 0:
                save_checkpoint(cfg, gen, handle, step)
    finally:
        handle.clear()

    frozen = inject.check_base_frozen(model)
    if not frozen["ok"]:
        raise RuntimeError("base model became trainable during training: %s"
                           % frozen["leaked"][:3])
    history["base_frozen"] = True
    if eval_batches:
        ev = evaluate_loss(model, handle, gen, cond, eval_batches, cfg,
                           layer_ids, module_ids)
        ev["step"] = cfg.max_steps
        history["eval"].append(ev)
        print("final eval loss %.4f  %s" % (ev["loss"], ev["per_cell"]))
    save_checkpoint(cfg, gen, handle, cfg.max_steps, name="final")
    with open(os.path.join(cfg.out_dir, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2, sort_keys=True, default=str)
    history["handle"] = handle
    history["generator"] = gen
    history["conditioning"] = cond
    history["model"] = model
    return history


def load_generator_checkpoint(gen, path, strict=True):
    """Warm-start a generator from a saved checkpoint (T2L_PLAN E2).

    Accepts either this module's checkpoints or `scripts/train_t2l.py`'s --
    both store the state under a `generator` key alongside the geometry that
    produced it. The geometry is CHECKED, not assumed: loading a rank-16
    checkpoint into a rank-8 generator would succeed on some tensors and
    silently mismatch on the heads.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("generator", ck)
    want_rank, have_rank = ck.get("rank"), getattr(gen, "rank", None)
    if want_rank is not None and have_rank is not None \
            and int(want_rank) != int(have_rank):
        raise ValueError("checkpoint %s has rank %s but this generator is "
                         "rank %s" % (path, want_rank, have_rank))
    want_ts = ck.get("target_set")
    if want_ts and ck.get("sites") and len(ck["sites"]) != len(gen.sites):
        raise ValueError("checkpoint %s was built on %d sites (%s) but this "
                         "generator has %d" % (path, len(ck["sites"]), want_ts,
                                               len(gen.sites)))
    missing, unexpected = gen.load_state_dict(sd, strict=strict)
    print("warm start from %s (step %s): %d missing, %d unexpected tensor(s)"
          % (path, ck.get("step"), len(missing), len(unexpected)))
    if missing and strict is False:
        print("  missing e.g. %s" % (list(missing)[:3],))
    return {"missing": list(missing), "unexpected": list(unexpected),
            "step": ck.get("step")}


def freeze_except(gen, substring):
    """Freeze every generator parameter whose name lacks `substring`.

    Returns the names left trainable. Raises if that set is empty -- a run
    with nothing to optimise trains for an hour and produces the checkpoint it
    started with, which is an expensive way to learn about a typo.
    """
    kept = []
    for name, p in gen.named_parameters():
        train = substring in name
        p.requires_grad_(train)
        if train:
            kept.append(name)
    if not kept:
        raise ValueError(
            "train_only=%r matches no parameter in this %s. It has: %s"
            % (substring, type(gen).__name__,
               sorted(set(n.split(".")[0] for n, _ in gen.named_parameters()))))
    print("train_only=%r: %d trainable tensor(s) %s"
          % (substring, len(kept), kept[:4]))
    return kept


def save_checkpoint(cfg, gen, handle, step, name=None):
    """Generator state + everything needed to rebuild the injection."""
    d = os.path.join(cfg.out_dir, "checkpoints")
    if not os.path.isdir(d):
        os.makedirs(d)
    path = os.path.join(d, "%s.pt" % (name or ("step%06d" % step)))
    torch.save({"generator": gen.state_dict(), "config": cfg.to_dict(),
                "step": step, "rank": handle.rank, "alpha": handle.alpha,
                "target_set": cfg.target_set,
                "sites": [s.as_dict() for s in handle.sites]}, path)
    return path


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------

_SMOKE_USER = (
    "# Instructions\nReview the current state of the page.\n\n## Goal:\n%s\n\n"
    "## AXTree:\nRootWebArea 'TimeWarp %s'\n  [31] link 'Home'\n"
    "  [42] textbox 'Search'\n  [59] button 'Go'\n")
_SMOKE_ASSISTANT = (
    "<think>\nThe search box is [42]; type the query there.\n</think>\n\n"
    "<action>\nfill('42', '%s')\n</action>")


def smoke(out_dir=None, steps=40, seed=0):
    """Run the whole loop on a tiny CPU model and assert it behaves.

    Asserts: (a) the base model is frozen before and after, (b) the mean loss of
    the last quarter of training is below the first quarter, (c) generated
    adapters materialise and read back with the right shapes.

    Writes to a temporary directory unless `out_dir` is given. A test must not
    leave artefacts in `out/`: `adaptercl status` reads that tree to decide what
    has been run, and a smoke run masquerading as a real one there is exactly
    the kind of bookkeeping lie this project cannot afford.
    """
    import shutil
    import tempfile

    tmp = None
    if out_dir is None:
        tmp = tempfile.mkdtemp(prefix="adaptercl_smoke_")
        out_dir = os.path.join(tmp, "run")
    try:
        return _smoke_inner(out_dir, steps, seed)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def _smoke_inner(out_dir, steps, seed):
    cfg = TrainConfig(
        generator="free", target_set="attn_mlp", rank=4, alpha=8.0,
        conditioning="hash", d_cond=32, split="forward",
        train_cells=("wiki_e2", "news_e3", "shop_e4"),
        eval_cells=("wiki_e5", "news_e5"),
        lr=3e-3, batch_size=2, grad_accum=2, max_steps=steps, warmup_steps=5,
        max_len=192, min_target_tokens=1, seed=seed, diversity_weight=0.01,
        device="cpu", bf16=False, gradient_checkpointing=False,
        out_dir=out_dir,
        run_name="smoke", log_every=5, eval_every=20, save_every=1000,
        eval_batches=2, allow_fallback_generator=True,
        generator_kwargs={"d_hidden": 64, "d_emb": 8},
        vocab_clamp=1024)

    print("=" * 72)
    print("adapterCL train_hypernet SMOKE (tiny CPU Qwen3.5, random init)")
    print("=" * 72)
    try:
        tokenizer = load_tokenizer(cfg)
        print("tokenizer: %s (vocab %d), ids clamped to %d"
              % (type(tokenizer).__name__, len(tokenizer), cfg.vocab_clamp))
    except Exception as exc:                        # noqa: BLE001
        print("real tokenizer unavailable (%s); using ByteTokenizer" % exc)
        tokenizer = ByteTokenizer()
    model, text_cfg = build_tiny_model(vocab_size=cfg.vocab_clamp)
    print("tiny model: %s params, %d layers, all full_attention"
          % ("{:,}".format(sum(p.numel() for p in model.parameters())),
             text_cfg["num_hidden_layers"]))

    rng = random.Random(seed)
    goals = ["find the article about biophysics", "add the blue mug to the cart",
             "read the top headline", "search for the 2004 election results"]
    def mk(cell_key, n):
        out = []
        for i in range(n):
            g = rng.choice(goals)
            out.append(BCSample(cell_key, "You are a web agent.",
                                _SMOKE_USER % (g, cell_key),
                                _SMOKE_ASSISTANT % g.split()[-1]))
        return out
    samples = []
    for key in cfg.train_cells:
        samples.extend(mk(key, 8))
    eval_samples = []
    for key in cfg.eval_cells:
        eval_samples.extend(mk(key, 4))
    print("fake corpus: %d train / %d eval samples over %d + %d cells"
          % (len(samples), len(eval_samples), len(cfg.train_cells),
             len(cfg.eval_cells)))

    pre = inject.check_base_frozen(model)
    print("base trainable BEFORE inject: %d tensors" % len(pre["leaked"]))

    hist = train(cfg, model=model, tokenizer=tokenizer, samples=samples,
                 eval_samples=eval_samples, text_cfg=text_cfg)

    losses = [r["loss"] for r in hist["steps"]]
    k = max(1, len(losses) // 4)
    first, last = sum(losses[:k]) / k, sum(losses[-k:]) / k
    print("")
    print("loss: first %d steps %.4f -> last %d steps %.4f (delta %+.4f)"
          % (k, first, k, last, last - first))

    handle, gen, cond = hist["handle"], hist["generator"], hist["conditioning"]
    written = generate_and_materialize(
        cfg, handle, gen, cond,
        [cells.Cell.parse(c) for c in cfg.eval_cells],
        meta={"smoke": True})
    print("materialised %d adapter(s): %s"
          % (len(written), sorted(written)))
    for key, d in sorted(written.items()):
        factors, acfg, meta = materialize.read_adapter(d)
        A, B = factors[handle.sites[0].rel_name]
        print("  %-10s %d sites, r=%s, first site A%s B%s"
              % (key, len(factors), acfg["r"], tuple(A.shape), tuple(B.shape)))
        assert tuple(A.shape) == (cfg.rank, handle.sites[0].d_in)
        assert tuple(B.shape) == (handle.sites[0].d_out, cfg.rank)

    post = inject.check_base_frozen(hist["model"])
    print("base trainable AFTER training: %d tensors" % len(post["leaked"]))

    ok = True
    if not post["ok"]:
        print("FAIL: base parameters are trainable: %s" % post["leaked"][:3])
        ok = False
    if not (last < first):
        print("FAIL: loss did not decrease (%.4f -> %.4f)" % (first, last))
        ok = False
    if len(written) != len(cfg.eval_cells):
        print("FAIL: expected %d generated adapters, got %d"
              % (len(cfg.eval_cells), len(written)))
        ok = False
    if not hist["eval"]:
        print("FAIL: no eval pass ran")
        ok = False
    print("")
    print("SMOKE %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Train the adapter generator end-to-end (adapterCL.md 8.4)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU end-to-end test; asserts loss decreases")
    ap.add_argument("--smoke-steps", type=int, default=40)
    ap.add_argument("--corpus", help="bcdata.build_cell_corpora manifest.json")
    ap.add_argument("--split", default="forward", choices=list(cells.SPLITS))
    ap.add_argument("--generator", default="basis",
                    choices=("free", "mixture", "basis", "t2l"))
    ap.add_argument("--generator-kwargs", default=None,
                    help="JSON dict forwarded verbatim to "
                         "hypernet.build_generator, e.g. "
                         "'{\"head_rank\":32,\"cross_layer\":true}' for the free "
                         "generator or '{\"fixed_A\":\"bank_mean\"}' for E3's "
                         "B-only arm. Every generator dataclass field is "
                         "reachable this way; there is deliberately no flag per "
                         "knob, because the knobs differ per generator.")
    ap.add_argument("--target-set", default="attn_mlp",
                    choices=list(targets.TARGET_SETS))
    ap.add_argument("--conditioning", default="vision",
                    help="encoder conditioner kind ('vision' | 'structure' | "
                         "'both'), 'hash' for the 6.7 version-identity "
                         "control, 'none' for one constant vector (T2L_PLAN "
                         "E5: is generation doing anything at all?), or "
                         "'axtree' for per-sample role histograms parsed from "
                         "each step's own AXTree (E5's escape from N=5).")
    ap.add_argument("--axtree-level", default="episode",
                    choices=("episode", "step"),
                    help="with --conditioning axtree: condition each sample on "
                         "its EPISODE's first observation (~125 points per "
                         "version, the serving-time object) or on its own STEP "
                         "(thousands of points, but an eval-time distribution "
                         "shift that has to be measured).")
    ap.add_argument("--embeddings", default=None,
                    help="precomputed encoder.save_embeddings .npz keyed by unit "
                         "(v1..v6). Strongly preferred over encoding on the fly: "
                         "the tower is frozen, so re-encoding the same 6 "
                         "screenshots every run wastes GPU and risks a different "
                         "capture protocol silently changing the conditioning.")
    ap.add_argument("--train-cells", default=None,
                    help="comma-separated units to TRAIN on, e.g. v2,v3,v4,v5,v6 "
                         "(leave-one-out). Overrides --split.")
    ap.add_argument("--eval-cells", default=None,
                    help="comma-separated held-out units, e.g. v1")
    ap.add_argument("--bank-tag", default=None,
                    help="out/cells/<bank-tag>/<unit> supplies the mixture bank. "
                         "Defaults to 'ver' (the 9B adapters); a 4B run MUST pass "
                         "ver4b or it silently builds a bank from the wrong backbone.")
    ap.add_argument("--granularity", default="version",
                    choices=["version", "cell"],
                    help="adapter unit; must match how the corpus was built")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--epochs", type=float, default=None,
                    help="derive max_steps as ceil(epochs * n_train / "
                         "(batch_size * grad_accum))")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diversity-weight", type=float, default=0.0)
    ap.add_argument("--term-weight", type=float, default=1.0,
                    help="loss weight on `send_msg_to_user` tokens (T2L_PLAN "
                         "E9). 1.0 keeps the fused-loss path; anything else "
                         "computes the weighted token loss by hand.")
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--warmup-frac", type=float, default=None,
                    help="fraction of resolved max_steps used for warmup; "
                         "overrides --warmup-steps")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--gen-l2", type=float, default=0.0,
                    help="coefficient on mean squared generated A/B factors")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--max-samples-per-cell", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--run-name", default="hypernet")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--init-checkpoint", default=None,
                    help="warm-start the generator from a saved checkpoint "
                         "(T2L_PLAN E2: is the reconstruction solution a "
                         "useful start for BC, or a harmful one?)")
    ap.add_argument("--init-non-strict", action="store_true",
                    help="allow the warm start to leave tensors unset (e.g. a "
                         "learn_scale gain a recon checkpoint does not have)")
    ap.add_argument("--train-only", default=None,
                    help="freeze every generator parameter whose name lacks "
                         "this substring. `--train-only log_gain` on a loaded "
                         "recon checkpoint is E4b: learned scale on a frozen "
                         "direction.")
    ap.add_argument("--materialize", action="store_true",
                    help="after training, write a PEFT adapter per eval cell")
    args = ap.parse_args(argv)

    if args.smoke:
        return smoke(out_dir=args.out_dir, steps=args.smoke_steps,
                     seed=args.seed)

    if not args.corpus:
        ap.error("--corpus is required (build it with "
                 "`python -m adaptercl.bcdata cell-corpora`)")
    if args.epochs is not None and args.max_steps != 500:
        ap.error("--epochs cannot be combined with a non-default --max-steps")
    gkw = {}
    if args.generator_kwargs:
        try:
            gkw = json.loads(args.generator_kwargs)
        except ValueError as exc:
            ap.error("--generator-kwargs is not valid JSON (%s): %s"
                     % (exc, args.generator_kwargs))
        if not isinstance(gkw, dict):
            ap.error("--generator-kwargs must be a JSON OBJECT, got %s"
                     % type(gkw).__name__)
    cfg = TrainConfig(
        generator=args.generator, generator_kwargs=gkw,
        target_set=args.target_set, rank=args.rank,
        alpha=args.alpha, conditioning=args.conditioning, corpus=args.corpus,
        conditioning_kwargs={"level": args.axtree_level},
        embeddings=args.embeddings, granularity=args.granularity,
        bank_tag=(args.bank_tag or "ver"),
        train_cells=(tuple(args.train_cells.split(',')) if args.train_cells else None),
        eval_cells=(tuple(args.eval_cells.split(',')) if args.eval_cells else None),
        split=args.split, lr=args.lr, batch_size=args.batch_size,
        grad_accum=args.grad_accum, max_steps=args.max_steps,
        epochs=args.epochs, warmup_steps=args.warmup_steps,
        warmup_frac=args.warmup_frac, weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing, gen_l2=args.gen_l2,
        eval_every=args.eval_every,
        save_every=args.save_every,
        max_samples_per_cell=args.max_samples_per_cell,
        max_len=args.max_len, seed=args.seed,
        diversity_weight=args.diversity_weight, term_weight=args.term_weight,
        init_checkpoint=args.init_checkpoint,
        init_strict=not args.init_non_strict, train_only=args.train_only,
        device=args.device,
        bf16=not args.fp32, gradient_checkpointing=not args.no_grad_ckpt,
        run_name=args.run_name, out_dir=args.out_dir)
    hist = train(cfg)
    if args.materialize:
        _tr, ev = cfg.resolved_cells()
        written = generate_and_materialize(cfg, hist["handle"],
                                           hist["generator"],
                                           hist["conditioning"], ev)
        print("generated adapters: %s" % json.dumps(written, indent=2,
                                                    sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
