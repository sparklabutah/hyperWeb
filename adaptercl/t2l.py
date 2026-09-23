"""Text-to-LoRA (T2L) style adapter generation, ported to the version setting.

Reference: Charakorn et al., *Text-to-LoRA: Instant Transformer Adaption*
(Sakana AI, arXiv:2506.06105; https://github.com/SakanaAI/text-to-lora).

Why this exists alongside `hypernet.MixtureGenerator`. The mixture generator
emits K coefficients over K **pre-trained** adapters, so everything it can
produce lies in the convex hull of the bank -- it routes, it does not
synthesise. T2L is the opposite commitment: a head regresses the LoRA factors
for each (layer, module) **directly**, with no bank, conditioned on a natural
language description of the target. It is therefore the arm that can actually
support a claim of weight *generation*, and it is the honest baseline to put
next to a router.

Two further reasons it is the right contrast here:

* **Modality.** Our policy reads accessibility trees, never pixels, so a
  vision-conditioned generator is conditioned on a channel the policy cannot
  see. A text description is at least the same kind of object the agent
  consumes.
* **Gauge invariance.** T2L's reconstruction loss is taken on the *product*
  dW = BA, not on (A, B) separately. Since (RA, BR^-1) induces the same dW for
  any invertible R, a loss on the factors would be optimising an arbitrary
  gauge; a loss on the product is well posed. This matters more here than in
  the original setting because our oracle adapters were trained independently
  and share no gauge.

Architecture follows the paper's **M** variant: one output layer per weight
*shape*, shared between A and B and selected by a learnable A/B embedding.
For our 128 `attn_mlp` sites that is ~60M parameters (the paper's M is 34M on
64 q/v sites of a 4096-wide model; ours are wider and more numerous).
"""
from __future__ import print_function

import json
import math
import os

import torch
import torch.nn as nn

from . import targets
from .hypernet import AdapterGenerator

#: The reference implementation uses `Alibaba-NLP/gte-large-en-v1.5`. That
#: checkpoint ships a custom architecture whose bundled `modeling.py` is not
#: compatible with the transformers version in this environment: its rotary
#: embedding path indexes `rope_cos[position_ids]` against a 130-entry table and
#: raises `IndexError: index 1163281 is out of bounds` (on GPU the same fault
#: appears only as a CUDA device-side assert). Rather than pin an older
#: transformers for one frozen encoder, we substitute a standard-architecture
#: model of the **same output width (1024)** and comparable size (335M vs 434M).
#: The encoder is frozen in either case, so it contributes no trainable
#: parameters and the substitution does not change what is being learned --
#: but it is a deviation from the reference and is reported as such.
DEFAULT_TEXT_ENCODER = "BAAI/bge-large-en-v1.5"
REFERENCE_TEXT_ENCODER = "Alibaba-NLP/gte-large-en-v1.5"  # T2L's own choice

#: Natural-language descriptions of each interface version.
#:
#: These are the *conditioning input* -- the analogue of T2L's task
#: descriptions. Three properties are deliberate and load-bearing:
#:
#: 1. Each is written from that version's own theme sources and nominal years
#:    (cells.THEME_NAME / cells.NOMINAL_YEAR) and mentions **no other version**.
#:    A description that said "more modern than v3" would leak the ordering the
#:    generator is supposed to infer.
#: 2. None mentions task success, difficulty, or anything about the agent. They
#:    describe the interface, not how well anything performs on it.
#: 3. Era 6 is described as style-neutral rather than as "newest". It is not a
#:    point on the time axis (cells.TEMPORAL_ERAS excludes it), and calling it
#:    modern would inject a false ordering.
VERSION_DESCRIPTIONS = {
    "v1": (
        "A web interface in the visual idiom of roughly 2000 to 2001. Pages are "
        "laid out with nested HTML tables rather than CSS boxes. Navigation is a "
        "dense column of plain blue underlined hyperlinks. Typography is a small "
        "default serif or Times face at a fixed pixel size, text is packed tightly "
        "with little whitespace, and the palette is limited to web-safe colours on "
        "a plain or lightly tiled background. Widgets are unstyled browser "
        "defaults: square grey buttons, bare text inputs, and horizontal rule "
        "separators. There is no rounded geometry, no shadowing, and no responsive "
        "behaviour."
    ),
    "v2": (
        "A web interface in the visual idiom of roughly 2002 to 2005. Layout is "
        "still largely table-driven but beginning to use CSS for colour and "
        "spacing. Navigation appears as a horizontal bar or a boxed sidebar with "
        "background fills and visible borders. Typography moves to small "
        "sans-serif faces at fixed sizes. The palette introduces gradients, "
        "beveled edges and coloured header bands. Form controls are still close to "
        "browser defaults but are placed inside bordered panels, and pages are "
        "built to a fixed width rather than filling the window."
    ),
    "v3": (
        "A web interface in the visual idiom of roughly 2003 to 2010. Layout is "
        "CSS-driven with floated columns and a fixed-width centred content well. "
        "Navigation uses tabbed or pill-shaped elements with hover states. "
        "Typography is sans-serif with a clearer heading hierarchy and more "
        "generous line spacing. The palette features glossy gradients, rounded "
        "corners, drop shadows and reflective button treatments. Content is "
        "grouped into visually distinct boxed modules with headers, and pages "
        "carry decorative chrome around the main content."
    ),
    "v4": (
        "A web interface in the visual idiom of roughly 2013 to 2016. Layout uses "
        "a responsive grid with generous padding and clearly separated content "
        "cards. Navigation is a flat horizontal bar, sometimes fixed to the top of "
        "the viewport, with icon-and-label items. Typography is a large-x-height "
        "sans-serif with strong size contrast between headings and body text. The "
        "palette is flat and saturated with no gradients or bevels, using solid "
        "fills, thin dividers and ample whitespace. Controls are rectangular with "
        "slight corner rounding and flat colour states."
    ),
    "v5": (
        "A web interface in the visual idiom of roughly 2024 to 2025. Layout is "
        "built on CSS flexbox and grid with large responsive spacing and full-width "
        "sections. Navigation is minimal, often reduced to a compact bar with an "
        "icon menu. Typography is a modern variable sans-serif with large headings "
        "and high contrast against a restrained neutral palette. Surfaces use soft "
        "shadows, large corner radii, and subtle borders to separate cards. "
        "Controls are pill-shaped or softly rounded with clear focus states, and "
        "the page adapts fluidly to the viewport width."
    ),
    "v6": (
        "A deliberately style-neutral web interface with no period styling of any "
        "kind. It applies close to the browser's default rendering: a single "
        "column of unstyled semantic HTML, default typography at default sizes, "
        "and no decorative colour, imagery, or layout chrome. Navigation is a plain "
        "list of links. Controls are unstyled native form elements. The page "
        "carries no visual theme, no branding, and no era-specific design "
        "conventions; it is a minimal structural baseline rather than a point on "
        "any timeline."
    ),
}


def load_descriptions(path=None):
    """Descriptions from JSON if given, else the built-in table."""
    if path:
        with open(path) as fh:
            return json.load(fh)
    return dict(VERSION_DESCRIPTIONS)


class TextConditioner(nn.Module):
    """Description string -> a single frozen embedding vector.

    Frozen, like the vision tower it replaces: the encoder contributes no
    trainable parameters, so the only thing that learns is the hypernetwork
    itself and the comparison to the mixture arm stays like-for-like.
    """

    def __init__(self, model_id=DEFAULT_TEXT_ENCODER, device="cpu",
                 dtype=torch.float32, pool="mean", max_len=512):
        super(TextConditioner, self).__init__()
        from transformers import AutoModel, AutoTokenizer
        # trust_remote_code is needed on BOTH: gte-large-en-v1.5 ships a custom
        # architecture, and loading the tokenizer without it silently falls back
        # to a mismatched one whose ids overflow the model's embedding tables.
        # That surfaces only as a CUDA device-side assert inside
        # token_type_embeddings, which is an unhelpful way to learn about it.
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval().to(device=device, dtype=dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device, self.dtype, self.pool, self.max_len = device, dtype, pool, max_len

    @property
    def d_out(self):
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(self, texts):
        """[str] -> (N, d) float32 on CPU, L2-normalised."""
        if isinstance(texts, str):
            texts = [texts]
        b = self.tok(list(texts), padding=True, truncation=True,
                     max_length=self.max_len, return_tensors="pt").to(self.device)
        out = self.model(**b).last_hidden_state              # (N, T, d)
        if self.pool == "cls":
            v = out[:, 0]
        else:
            m = b["attention_mask"].unsqueeze(-1).to(out.dtype)
            v = (out * m).sum(1) / m.sum(1).clamp(min=1)
        v = v.float().cpu()
        return torch.nn.functional.normalize(v, dim=-1)


class _ResBlock(nn.Module):
    def __init__(self, d):
        super(_ResBlock, self).__init__()
        self.f = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.n = nn.LayerNorm(d)

    def forward(self, x):
        return self.n(x + self.f(x))


class T2LHypernet(AdapterGenerator):
    """T2L-M: shared-per-shape output head, A/B selected by a learnable embedding.

    forward(h, layer_ids, module_ids) -> ([A_q], [B_q]) in the order of the
    QUERY, matching `hypernet.AdapterGenerator` so `generate_for_cells`,
    `materialize_generated` and `train_hypernet` all work unchanged. `h` may be
    `(d_cond,)` or `(Bsz, d_cond)`; the batched form is what end-to-end BC
    training needs, because a BC batch mixes versions (T2L_PLAN E2).

    Subclassing `AdapterGenerator` adds no parameters and renames nothing, so
    checkpoints written by `scripts/train_t2l.py` before this change load
    unmodified -- which is the point: E2 fine-tunes them.

    `variant`
        `"M"` (default, the paper's and the trained runs') shares one output
        head per distinct `(d_in, d_out)`; on this backbone `gate_proj`/
        `up_proj` and `k_proj`/`v_proj` therefore share a head. `"L"` gives one
        head per module type. Only "M" is checkpoint-compatible with the
        existing runs.
    `fixed_A`
        As `hypernet.FreeHypernet` -- hold A at a frozen per-site constant and
        regress only B (T2L_PLAN E3a). Accepts the same spellings.
    `learn_scale`
        Add a per-module-type log-gain on B, trainable (T2L_PLAN E4b). The
        direction is then whatever the trunk produces and the magnitude is
        learned separately, which is the one knob 10 showed dominates.
    `sft_init`
        Start from the reference SFT recipe's condition-blind PEFT LoRA: every
        head is zero, while a shared-per-head A bias is Kaiming-uniform and its
        B bias is zero. Thus dW is exactly zero at construction, but B receives
        a useful first-step gradient through the nonzero A factor.
    """

    kind = "t2l"
    starts_as_noop = False

    def __init__(self, sites, rank, d_cond, d_emb=32, d_hidden=128, depth=2,
                 n_layers=None, out_scale=1.0, init_dw=1e-6, variant="M",
                 fixed_A=None, bank_adapters=None, bank_keys=None,
                 learn_scale=False, sft_init=False):
        super(T2LHypernet, self).__init__(sites, rank, d_cond,
                                          n_layers=n_layers)
        self.out_scale = float(out_scale)
        if variant not in ("M", "L"):
            raise ValueError("variant must be 'M' (per-shape heads) or 'L' "
                             "(per-module heads), got %r" % (variant,))
        self.variant = variant
        self.fixed_A = fixed_A is not None
        self.sft_init = bool(sft_init)
        # Keep the class attribute meaningful for callers that inspect the
        # generator TYPE, while reporting the actual construction per instance.
        self.starts_as_noop = self.sft_init

        self.layer_emb = nn.Embedding(self.n_layers, d_emb)
        self.module_emb = nn.Embedding(len(targets.MODULE_TYPES), d_emb)
        self.ab_emb = nn.Embedding(2, d_emb)          # 0 = A, 1 = B

        trunk = [nn.Linear(d_cond + 3 * d_emb, d_hidden), nn.SiLU()]
        for _ in range(max(0, depth)):
            trunk.append(_ResBlock(d_hidden))
        self.trunk = nn.Sequential(*trunk)
        self.d_hidden = d_hidden

        # One head per distinct (d_in, d_out) for "M". Shared between A and B --
        # the paper's M variant -- so the head emits max(r*d_in, d_out*r) and
        # the caller slices. Sharing across layers is what keeps this ~60M
        # rather than ~60M x n_layers. "L" keys the same table by module type.
        if variant == "M":
            self.shapes = sorted({(s.d_in, s.d_out) for s in sites})
            self.shape_index = {sh: i for i, sh in enumerate(self.shapes)}
            self._head_of = {s.rel_name: self.shape_index[(s.d_in, s.d_out)]
                             for s in sites}
            widths = [max(self.rank * di, do * self.rank)
                      for (di, do) in self.shapes]
        else:
            mods = sorted({s.module_id for s in sites})
            self.shapes = [self.shape_by_module_id[m] for m in mods]
            self.shape_index = dict((m, i) for i, m in enumerate(mods))
            self._head_of = {s.rel_name: self.shape_index[s.module_id]
                             for s in sites}
            widths = [max(self.rank * di, do * self.rank)
                      for (di, do) in self.shapes]
        self.heads = nn.ModuleList([nn.Linear(d_hidden, w) for w in widths])
        # Init so the *product* starts near zero rather than the factors. With
        # A, B ~ N(0, s^2), an element of B@A sums r products, so its scale is
        # about sqrt(r) * s^2 -- quadratic in s, which is why a std that looks
        # tiny can still yield a huge dW. The trained oracles sit at
        # |dW| ~ 5e-5, so starting anywhere near O(1) would swamp them.
        if self.sft_init:
            for h in self.heads:
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)

            # These are separate from Linear.bias because the M architecture
            # uses one Linear for both A and B. The slot-specific parameters
            # reproduce PEFT's default LoRA init without giving up that sharing.
            if not self.fixed_A:
                self.bias_A = nn.ParameterList([
                    nn.Parameter(torch.empty(self.rank, di))
                    for (di, _do) in self.shapes])
                for a in self.bias_A:
                    nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.bias_B = nn.ParameterList([
                nn.Parameter(torch.zeros(do, self.rank))
                for (_di, do) in self.shapes])
        else:
            s = (float(init_dw) / (self.rank ** 0.5)) ** 0.5
            for h in self.heads:
                nn.init.normal_(h.weight, std=s / (d_hidden ** 0.5))
                nn.init.zeros_(h.bias)

        if self.fixed_A:
            from . import hypernet as _hn
            for i, A in enumerate(_hn.resolve_fixed_A(
                    fixed_A, self.sites, self.rank, adapters=bank_adapters,
                    keys=bank_keys)):
                self.register_buffer("A_fixed_%d" % i, A, persistent=True)
        self.log_gain = None
        if learn_scale:
            # one gain per module TYPE, indexed globally so the tensor's shape
            # does not depend on the target set (a checkpoint stays loadable).
            self.log_gain = nn.Parameter(torch.zeros(len(targets.MODULE_TYPES)))

    # -- sizes -------------------------------------------------------------
    @property
    def output_rank(self):
        return self.rank

    def n_output_dims(self):
        if not self.fixed_A:
            return targets.site_budget(self.sites, self.rank)
        return sum(self.rank * s.d_out for s in self.sites)

    def __len__(self):
        return len(self.sites)

    # -- trunk -------------------------------------------------------------
    def _z(self, h, layer_id, module_id, ab):
        """(Bsz, d_hidden) trunk features for one site and one A/B slot."""
        dev = next(self.parameters()).device
        h2, _batched = self._prep_h(h)
        h2 = h2.to(dev)
        bsz = h2.shape[0]
        idx = torch.tensor([layer_id], device=dev)
        mdx = torch.tensor([module_id], device=dev)
        adx = torch.tensor([ab], device=dev)
        parts = [h2,
                 self.layer_emb(idx).expand(bsz, -1),
                 self.module_emb(mdx).expand(bsz, -1),
                 self.ab_emb(adx).expand(bsz, -1)]
        return self.trunk(torch.cat(parts, dim=-1))

    def _gain(self, site):
        if self.log_gain is None:
            return 1.0
        return torch.exp(self.log_gain[site.module_id])

    def factors_for_site(self, h, site):
        """-> (A, B) for one site.

        Unbatched `h` gives A (r, d_in), B (d_out, r) -- the shape
        `scripts/train_t2l.py` has always consumed. Batched `h` gives a leading
        Bsz on both.
        """
        _h2, batched = self._prep_h(h)
        head_i = self._head_of[site.rel_name]
        head = self.heads[head_i]
        b = head(self._z(h, site.layer, site.module_id, 1))
        b = b[:, : site.d_out * self.rank]
        if self.sft_init:
            b = b + self.bias_B[head_i].reshape(-1)
        B = b.reshape(-1, site.d_out, self.rank)
        # out_scale multiplies the *delta*, so it is applied once (to B) and
        # not to both factors -- scaling both would scale B@A quadratically.
        B = B * self.out_scale * self._gain(site)
        if self.fixed_A:
            i = self.site_index[(site.layer, site.module_id)]
            Af = getattr(self, "A_fixed_%d" % i).to(B.dtype)
            A = Af.unsqueeze(0).expand(B.shape[0], *Af.shape)
        else:
            a = head(self._z(h, site.layer, site.module_id, 0))
            a = a[:, : self.rank * site.d_in]
            if self.sft_init:
                a = a + self.bias_A[head_i].reshape(-1)
            A = a.reshape(-1, self.rank, site.d_in)
        if not batched:
            return A[0], B[0]
        return A, B

    def forward(self, h, layer_ids=None, module_ids=None):
        """Site order follows the QUERY, not this generator's own site list."""
        _h2, batched = self._prep_h(h)
        if layer_ids is None or module_ids is None:
            layer_ids, module_ids = self.query_ids()
        self._check_query(layer_ids, module_ids)
        idx = self._site_indices(layer_ids, module_ids)
        A_list, B_list = [], []
        for i in idx:
            A, B = self.factors_for_site(h, self.sites[i])
            A_list.append(A)
            B_list.append(B)
        # Unbatched `h` returns 2-D factors, like every other AdapterGenerator.
        # The pre-port version returned a leading 1 here; nothing consumed it
        # (train_t2l.py goes through factors_for_site), and keeping it would
        # have made `hypernet.generate_for_cells` emit batch-1 factors that
        # `InjectedLoRALinear` then rejects against a batch-2 input.
        return A_list, B_list


def oracle_factors(adapter_dir, sites, alpha_over_r=1.0):
    """{rel_name: (A, B, scale)} -- the oracle kept in FACTORED form.

    Prefer this to `oracle_deltas` for training. Materialising every dW costs
    sum(d_in * d_out) floats per version, which for a 9B backbone over 128 sites
    is 5.3e9 floats = 21.2 GB in fp32; five training versions is ~106 GB of host
    RAM and gets the job OOM-killed inside any constrained cgroup (it took out
    one run at step 5000). The factors are 29M params -- about 116 MB -- and the
    product is cheap to form per site on the GPU when it is actually needed.
    """
    from . import materialize
    factors, cfg, _ = materialize.read_adapter(adapter_dir)
    scale = alpha_over_r if alpha_over_r is not None else \
        float(cfg.get("lora_alpha", cfg.get("r", 1))) / float(cfg.get("r", 1))
    out = {}
    for s in sites:
        if s.rel_name not in factors:
            continue
        A, B = factors[s.rel_name]
        out[s.rel_name] = (torch.as_tensor(A, dtype=torch.float32),
                           torch.as_tensor(B, dtype=torch.float32),
                           float(scale))
    return out


def oracle_delta_at(fac, rel_name, device=None):
    """Form one oracle dW on demand from stored factors."""
    A, B, scale = fac[rel_name]
    if device is not None:
        A, B = A.to(device), B.to(device)
    return (B @ A) * scale


def oracle_deltas(adapter_dir, sites, alpha_over_r=1.0, device="cpu"):
    """{rel_name: dW} for one trained adapter -- the reconstruction target.

    Returned as the *product* B@A, never the factors: (RA, BR^-1) realises the
    same dW, so a target expressed in factors would be gauge-dependent and the
    regression would be fitting an arbitrary basis.
    """
    from . import materialize
    factors, cfg, _ = materialize.read_adapter(adapter_dir)
    scale = alpha_over_r if alpha_over_r is not None else \
        float(cfg.get("lora_alpha", cfg.get("r", 1))) / float(cfg.get("r", 1))
    out = {}
    for s in sites:
        if s.rel_name not in factors:
            continue
        A, B = factors[s.rel_name]
        A = torch.as_tensor(A, dtype=torch.float32, device=device)
        B = torch.as_tensor(B, dtype=torch.float32, device=device)
        out[s.rel_name] = (B @ A) * scale
    return out
