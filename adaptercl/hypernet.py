"""Adapter generators: free / mixture / learned-basis (4.1, 6.5, 8.3).

6.5 says the mixture must be run *before* the hypernetwork is built and 6.1 says
"plan for the learned-basis model to be the primary method". Both are only true
if swapping generator is a config change, so all three implement one interface:

    A_list, B_list = generator(h, layer_ids, module_ids)

with `h` either `(d_cond,)` or `(Bsz, d_cond)`, `layer_ids`/`module_ids` from
`InjectionHandle.query_ids()`, and the returned per-site tensors shaped exactly
as `InjectedLoRALinear` expects -- `(r, d_in)`/`(d_out, r)` unbatched,
`(Bsz, r, d_in)`/`(Bsz, d_out, r)` batched (inject.py:150-173).

Four places where 8.3's sketch is wrong or underspecified, all fixed here:

1. **One `d_in`/`d_out` per model is a fiction.** In this hybrid stack `q_proj`
   is 8192x4096 (output-gated, so double width), `k_proj` 1024x4096, `down_proj`
   4096x12288. 8.3 offers "per-module-type heads or padding"; padding is the
   wrong choice -- padding every head to `max(d_out)=12288` would make the
   `k_proj` head emit 12x the numbers it needs and force the trunk to learn to
   zero 92 % of its own output. `FreeHypernet` therefore keeps one head per
   module type.
2. **Those heads are the parameter budget, not the trunk.** A dense
   `Linear(512 -> rank*d_in)` head is `d_hidden * rank * d_in` parameters; on
   `attn_mlp` at rank 16 the heads total ~650 M parameters to generate a 29 M
   adapter. 8.3 never mentions this. `head_rank` factorises them
   (`d_hidden -> k -> rank*d_in`) and `parameter_report` prints both.
3. **A mixture of LoRAs is not a mixture in factor space.** With
   `A = sum_k c_k A_k` and `B = sum_k c_k B_k` the induced weight delta is
   `sum_{j,k} c_j c_k B_j A_k` -- quadratic in `c`, with cross terms between
   adapter j's B and adapter k's A that no trained adapter ever contained. The
   honest object is `dW = sum_k c_k B_k A_k`, which is exact if you *stack*:
   `A = concat_k(c_k A_k)` (rank K*r) and `B = concat_k(B_k)`. That is
   `mode="delta"`, the default for `MixtureGenerator`. `mode="factor"` keeps
   rank r and the cross terms, and is what most "LoRA mixing" code silently
   does; it stays available because 6.5's baseline should be reported as the
   stronger of the two, not as whichever one the implementation happened to do.
4. **Zero-init has two forms and only one of them keeps a basis usable.**
   Zeroing the B *head* (8.3) is right for `FreeHypernet`. Zeroing a learned
   *basis* `{B_k}` is not: with every `B_k = 0`, `dL/dB_k = c_k * G A^T`, so the
   first update makes all K directions exactly collinear. `BasisGenerator`
   instead keeps a diverse random basis behind a scalar `out_scale`
   initialised to 0 -- generated B is still exactly zero (the invariant 8.3
   cares about, tested by `diagnostics.noop_at_init`) without the collapse.
5. **The no-op start makes the whole trunk dead for exactly one step, and 8.3
   does not say so.** With `B == 0` the gradient into A is `0` (it is
   proportional to B), and the gradient into the trunk through the B head is
   `dL/dB * W_B = 0` because `W_B` is the tensor that was zeroed. So on step 1
   *only the B head's output layer* receives gradient -- not the trunk, not the
   layer/module embeddings, and not the conditioning. Everything unfreezes on
   step 2. This is the same dynamics as ordinary LoRA (B=0 means A does not move
   first) and is not a bug, but it does mean: a one-step gradient check on a
   freshly built generator will report "conditioning has no gradient" and be
   right; check after a step. `--selftest` tests both phases explicitly, and any
   LR-warmup schedule should be read with this in mind.

Requires torch. Import from the `llamafactory` env python (paths.PY_TRAIN).
"""

from __future__ import print_function

import collections
import math

import torch
import torch.nn as nn

from . import inject as inject_mod
from . import materialize, targets

# --------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------


class AdapterGenerator(nn.Module):
    """Common interface for every generator in 6.5's comparison.

    Two different "sizes" matter and are easy to conflate:

    `n_adapter_dims()`
        How many scalars end up in the materialised adapter -- identical for all
        three generators, equal to `targets.site_budget(sites, rank)`.
    `n_output_dims()`
        How many *degrees of freedom the conditioning actually chooses*. For
        free generation that is the same number (4-43 M against 12-18 training
        interfaces -- 6.1's memorisation risk in one ratio). For a mixture or a
        basis it is K (or K x Q for per-site coefficients). This is the number
        `parameter_report` puts next to the training-cell count.
    """

    kind = "abstract"
    #: does forward() return B == 0 exactly before any training? (8.3)
    starts_as_noop = True

    def __init__(self, sites, rank, d_cond, n_layers=None):
        super(AdapterGenerator, self).__init__()
        sites = list(sites)
        if not sites:
            raise ValueError("no sites -- call targets.enumerate_sites first")
        self.sites = sites
        self.rank = int(rank)
        self.d_cond = int(d_cond)
        self.n_layers = int(n_layers or (max(s.layer for s in sites) + 1))
        shapes, names = {}, {}
        for s in sites:
            prev = shapes.get(s.module_id)
            if prev is not None and prev != (s.d_in, s.d_out):
                raise ValueError(
                    "module type %r has two shapes in this site list: %r and %r"
                    % (s.module, prev, (s.d_in, s.d_out)))
            shapes[s.module_id] = (s.d_in, s.d_out)
            names[s.module_id] = s.module
        self.shape_by_module_id = shapes
        self.name_by_module_id = names
        self.site_index = collections.OrderedDict(
            ((s.layer, s.module_id), i) for i, s in enumerate(sites))
        if len(self.site_index) != len(sites):
            raise ValueError("duplicate (layer, module) in the site list")

    # -- query bookkeeping -------------------------------------------------
    def query_ids(self, device=None):
        """(layer_ids, module_ids) in this generator's own site order."""
        layer_ids = torch.tensor([s.layer for s in self.sites], dtype=torch.long)
        module_ids = torch.tensor([s.module_id for s in self.sites], dtype=torch.long)
        if device is not None:
            layer_ids, module_ids = layer_ids.to(device), module_ids.to(device)
        return layer_ids, module_ids

    def shapes(self):
        return [(s.d_in, s.d_out) for s in self.sites]

    def _check_query(self, layer_ids, module_ids):
        if layer_ids.shape != module_ids.shape or layer_ids.dim() != 1:
            raise ValueError("layer_ids/module_ids must be 1-D and the same "
                             "length, got %r / %r"
                             % (tuple(layer_ids.shape), tuple(module_ids.shape)))
        unknown = sorted(set(int(m) for m in module_ids.tolist())
                         - set(self.shape_by_module_id))
        if unknown:
            have = sorted(self.name_by_module_id.values())
            raise KeyError(
                "module_id(s) %r were never in this generator's site list "
                "(it knows %r). Rebuild the generator for this target set."
                % ([targets.MODULE_TYPES[u] for u in unknown], have))
        if int(layer_ids.max()) >= self.n_layers:
            raise IndexError(
                "layer id %d >= n_layers %d; pass n_layers=config."
                "num_hidden_layers so the embedding covers unseen layers"
                % (int(layer_ids.max()), self.n_layers))

    def _site_indices(self, layer_ids, module_ids):
        """Positions in `self.sites` for a query, for per-site parameters."""
        out = []
        for l, m in zip(layer_ids.tolist(), module_ids.tolist()):
            i = self.site_index.get((int(l), int(m)))
            if i is None:
                raise KeyError(
                    "query point (layer=%d, module=%s) has no per-site "
                    "parameters in this %s. Per-site generators cannot answer "
                    "for sites they were not built on."
                    % (l, targets.MODULE_TYPES[int(m)], type(self).__name__))
            out.append(i)
        return out

    @staticmethod
    def _prep_h(h):
        """-> ((Bsz, d_cond) tensor, was_batched)."""
        if not torch.is_tensor(h):
            h = torch.as_tensor(h)
        h = h.float() if h.dtype not in (torch.float32, torch.float64) else h
        if h.dim() == 1:
            return h.unsqueeze(0), False
        if h.dim() == 2:
            return h, True
        raise ValueError("h must be (d_cond,) or (Bsz, d_cond), got %r"
                         % (tuple(h.shape),))

    @staticmethod
    def _unbatch(A_list, B_list, batched):
        if batched:
            return A_list, B_list
        return [a[0] for a in A_list], [b[0] for b in B_list]

    # -- sizes -------------------------------------------------------------
    @property
    def output_rank(self):
        """Rank of the tensors this generator emits (see mode="delta")."""
        return self.rank

    def n_adapter_dims(self):
        return targets.site_budget(self.sites, self.rank)

    def n_output_dims(self):
        return targets.site_budget(self.sites, self.rank)

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self):
        return collections.OrderedDict([
            ("kind", self.kind), ("n_sites", len(self.sites)),
            ("rank", self.rank), ("output_rank", self.output_rank),
            ("d_cond", self.d_cond), ("n_layers", self.n_layers),
            ("trainable", self.n_trainable()),
            ("adapter_dims", self.n_adapter_dims()),
            ("output_dims", self.n_output_dims()),
            ("starts_as_noop", self.starts_as_noop),
        ])

    def forward(self, h, layer_ids, module_ids):
        raise NotImplementedError


# --------------------------------------------------------------------------
# 1. Free generation (8.3, corrected)
# --------------------------------------------------------------------------


class _FactorHead(nn.Module):
    """Per-module-type A/B heads. B is zero-initialised (8.3's no-op start).

    `head_rank` factorises `d_hidden -> rank*d_in` through a bottleneck. Without
    it a single `gate_proj` head at rank 16 is `512 * 16 * 12288 = 100 M`
    parameters; with `head_rank=32` it is `32 * (512 + 196608) = 6.3 M`.
    """

    def __init__(self, d_hidden, rank, d_in, d_out, head_rank=None,
                 emit_a=True):
        super(_FactorHead, self).__init__()
        self.rank, self.d_in, self.d_out = int(rank), int(d_in), int(d_out)
        self.emit_a = bool(emit_a)
        out_a, out_b = rank * d_in, d_out * rank
        if head_rank:
            k = int(head_rank)
            if self.emit_a:
                self.a = nn.Sequential(nn.Linear(d_hidden, k, bias=False),
                                       nn.Linear(k, out_a))
                _init_std(self.a[0].weight, 1.0 / math.sqrt(d_hidden))
                _init_std(self.a[1].weight, 1.0 / math.sqrt(3.0 * d_in * k))
                nn.init.zeros_(self.a[1].bias)
            self.b = nn.Sequential(nn.Linear(d_hidden, k, bias=False),
                                   nn.Linear(k, out_b))
            _init_std(self.b[0].weight, 1.0 / math.sqrt(d_hidden))
        else:
            if self.emit_a:
                self.a = nn.Linear(d_hidden, out_a)
                _init_std(self.a.weight, 1.0 / math.sqrt(3.0 * d_in * d_hidden))
                nn.init.zeros_(self.a.bias)
            self.b = nn.Linear(d_hidden, out_b)
        self.zero_b()

    def zero_b(self):
        """Exactly zero, so the generated adapter is a genuine no-op (8.3)."""
        last = self.b[-1] if isinstance(self.b, nn.Sequential) else self.b
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, z):
        """z: (..., d_hidden) -> A (..., r, d_in), B (..., d_out, r).

        With `emit_a=False` (the `fixed_A` generator) A is None and the caller
        substitutes the frozen factor -- the head then costs half as much and
        generates only the tensor that actually varies across versions.
        """
        lead = z.shape[:-1]
        A = (self.a(z).reshape(*(lead + (self.rank, self.d_in)))
             if self.emit_a else None)
        B = self.b(z).reshape(*(lead + (self.d_out, self.rank)))
        return A, B


def _init_std(w, std):
    with torch.no_grad():
        w.normal_(0.0, std)


class _QueryMixer(nn.Module):
    """4.1's "batch all (layer, module) query points and let the trunk attend
    over them jointly" -- one pre-LN self-attention block over the Q axis.

    T2L generates each layer independently, which has been criticised for
    missing cross-layer structure; this is the cheap mitigation 4.1 asks for and
    is a named ablation, so it is a flag, not a rewrite.
    """

    def __init__(self, d, n_heads=4, dropout=0.0):
        super(_QueryMixer, self).__init__()
        if d % n_heads:
            n_heads = 1
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(),
                                nn.Linear(2 * d, d))

    def forward(self, z):
        y = self.ln1(z)
        a, _ = self.attn(y, y, y, need_weights=False)
        z = z + a
        return z + self.ff(self.ln2(z))


class FreeHypernet(AdapterGenerator):
    """Shared trunk + per-module-type heads (4.1 / 8.3).

    `n_layers` should be the model's `num_hidden_layers`, not the number of
    layers in the current target set: the layer embedding then has an entry for
    every layer and a checkpoint stays loadable when the injection-site sweep
    (4.5) changes the target set. Module ids are already global
    (`targets.MODULE_ID`).

    `fixed_A` (T2L_PLAN E3) holds A at a constant, frozen value per site and
    generates only B. The motivation is measured, not aesthetic: across the six
    per-version oracles the A factors have pairwise cosine 0.90-0.95 (they all
    start from the same seed-42 init and barely move) while the B factors have
    cosine 0.11-0.15. So A is already very nearly a shared constant, and
    generating it costs half the head budget to reproduce a constant -- while
    also leaving the LoRA gauge free, since (RA, BR^-1) induces the same dW.
    Pinning A fixes the gauge across every generated adapter *and* across the
    oracle bank, which is what makes weight-space diagnostics comparable at all.

    Accepted values:

    ``None``        generate both factors (the original behaviour).
    ``"init"``      one PEFT-default Kaiming-uniform A per site, frozen.
    ``"bank_mean"`` requires `bank_adapters=`; the mean of those adapters' A.
    ``dict``        {rel_name: A} -- e.g. from `bank_mean_A`.
    ``sequence``    one (r, d_in) tensor per site, in site order.
    """

    kind = "free"

    def __init__(self, sites, rank, d_cond, d_hidden=512, d_emb=64, depth=2,
                 n_layers=None, cross_layer=False, n_heads=4, head_rank=None,
                 dropout=0.0, fixed_A=None, bank_adapters=None, bank_keys=None):
        super(FreeHypernet, self).__init__(sites, rank, d_cond, n_layers=n_layers)
        self.d_hidden = int(d_hidden)
        self.head_rank = head_rank
        self.cross_layer = bool(cross_layer)
        self.fixed_A = fixed_A is not None
        self.layer_emb = nn.Embedding(self.n_layers, d_emb)
        self.module_emb = nn.Embedding(len(targets.MODULE_TYPES), d_emb)
        layers = [nn.Linear(self.d_cond + 2 * d_emb, d_hidden), nn.SiLU()]
        for _ in range(max(0, int(depth) - 1)):
            layers += [nn.Linear(d_hidden, d_hidden), nn.SiLU()]
            if dropout:
                layers += [nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.mixer = _QueryMixer(d_hidden, n_heads, dropout) if cross_layer else None
        self.heads = nn.ModuleDict()
        for mid, (d_in, d_out) in sorted(self.shape_by_module_id.items()):
            self.heads[self.name_by_module_id[mid]] = _FactorHead(
                d_hidden, rank, d_in, d_out, head_rank=head_rank,
                emit_a=not self.fixed_A)
        if self.fixed_A:
            # persistent=True on purpose: A is part of this generator's
            # function, so a checkpoint that did not carry it would silently
            # reload a DIFFERENT adapter family. 128 sites x 16 x 4096 is 33 MB.
            for i, A in enumerate(resolve_fixed_A(fixed_A, self.sites, rank,
                                                  adapters=bank_adapters,
                                                  keys=bank_keys)):
                self.register_buffer("A_fixed_%d" % i, A, persistent=True)

    def zero_b(self):
        for head in self.heads.values():
            head.zero_b()
        return self

    def trunk_features(self, h, layer_ids, module_ids):
        """(Bsz, Q, d_hidden) -- exposed so probes can look at z, not just A/B."""
        h2, _ = self._prep_h(h)
        bsz, q = h2.shape[0], layer_ids.shape[0]
        le = self.layer_emb(layer_ids).unsqueeze(0).expand(bsz, q, -1)
        me = self.module_emb(module_ids).unsqueeze(0).expand(bsz, q, -1)
        hh = h2.unsqueeze(1).expand(bsz, q, h2.shape[-1])
        z = self.trunk(torch.cat([hh, le, me], dim=-1))
        if self.mixer is not None:
            z = self.mixer(z)
        return z

    def n_output_dims(self):
        """With `fixed_A`, only B is chosen by the conditioning."""
        if not self.fixed_A:
            return targets.site_budget(self.sites, self.rank)
        return sum(self.rank * s.d_out for s in self.sites)

    def forward(self, h, layer_ids, module_ids):
        self._check_query(layer_ids, module_ids)
        h2, batched = self._prep_h(h)
        z = self.trunk_features(h2, layer_ids, module_ids)      # (B, Q, dh)
        q = layer_ids.shape[0]
        bsz = z.shape[0]
        site_idx = self._site_indices(layer_ids, module_ids) if self.fixed_A \
            else None
        A_list, B_list = [None] * q, [None] * q
        mids = module_ids.tolist()
        by_type = collections.OrderedDict()
        for i, m in enumerate(mids):
            by_type.setdefault(int(m), []).append(i)
        for mid, idx in by_type.items():
            head = self.heads[self.name_by_module_id[mid]]
            sel = torch.as_tensor(idx, dtype=torch.long, device=z.device)
            A, B = head(z.index_select(1, sel))                  # (B, n, r, d_in)
            for j, i in enumerate(idx):
                if self.fixed_A:
                    Af = getattr(self, "A_fixed_%d" % site_idx[i])
                    A_list[i] = Af.to(z.dtype).unsqueeze(0).expand(
                        bsz, *Af.shape)
                else:
                    A_list[i] = A[:, j]
                B_list[i] = B[:, j]
        return self._unbatch(A_list, B_list, batched)


# --------------------------------------------------------------------------
# 2/3. Coefficient generators over a bank of adapter directions
# --------------------------------------------------------------------------


class _CoefficientHead(nn.Module):
    """h (+ layer/module embeddings) -> K coefficients.

    `per_site=False` gives one coefficient vector per conditioning, shared
    across every injection site -- the more constrained and, at 12-18 training
    cells, the more honest model: the conditioning picks a point in a
    K-dimensional space and nothing else. `per_site=True` lets the coefficients
    depend on the query point too, which multiplies the generated degrees of
    freedom by Q (128 sites on `attn_mlp`) and is much closer to free generation
    in capacity. Report which one produced a number.
    """

    def __init__(self, d_cond, n_basis, per_site=False, n_layers=32, d_emb=32,
                 d_hidden=256, simplex=True, temperature=1.0,
                 learn_temperature=False, uniform_init=True, depth=2):
        super(_CoefficientHead, self).__init__()
        self.per_site = bool(per_site)
        self.n_basis = int(n_basis)
        self.simplex = bool(simplex)
        d_in = d_cond + (2 * d_emb if per_site else 0)
        if per_site:
            self.layer_emb = nn.Embedding(n_layers, d_emb)
            self.module_emb = nn.Embedding(len(targets.MODULE_TYPES), d_emb)
        layers = [nn.Linear(d_in, d_hidden), nn.SiLU()]
        for _ in range(max(0, int(depth) - 2)):
            layers += [nn.Linear(d_hidden, d_hidden), nn.SiLU()]
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(d_hidden, self.n_basis)
        if uniform_init:
            # zero logits -> exactly uniform coefficients at init. For a mixture
            # that means "start at the mean trained adapter", which is a sane
            # and inspectable starting point.
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        else:
            _init_std(self.out.weight, 1.0 / math.sqrt(d_hidden))
            with torch.no_grad():
                self.out.bias.fill_(1.0 / self.n_basis)
        if learn_temperature:
            self.log_temp = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer("log_temp",
                                 torch.tensor(math.log(temperature)),
                                 persistent=False)

    def forward(self, h2, layer_ids=None, module_ids=None):
        """-> (B, K) if per_site is False else (B, Q, K)."""
        if not self.per_site:
            logits = self.out(self.body(h2))
        else:
            if layer_ids is None:
                raise ValueError("per-site coefficients need layer/module ids")
            bsz, q = h2.shape[0], layer_ids.shape[0]
            le = self.layer_emb(layer_ids).unsqueeze(0).expand(bsz, q, -1)
            me = self.module_emb(module_ids).unsqueeze(0).expand(bsz, q, -1)
            hh = h2.unsqueeze(1).expand(bsz, q, h2.shape[-1])
            logits = self.out(self.body(torch.cat([hh, le, me], dim=-1)))
        if self.simplex:
            return torch.softmax(logits / torch.exp(self.log_temp), dim=-1)
        return logits


def _combine(A_bank, B_bank, c, mode, out_scale=None):
    """(K, r, d_in), (K, d_out, r), coefficients (B, K) -> (A, B) per example.

    mode="factor": A = sum_k c_k A_k, B = sum_k c_k B_k. Rank stays r; the
        induced dW = sum_{j,k} c_j c_k B_j A_k is quadratic in c (see module
        docstring).
    mode="delta": A = concat_k(c_k A_k) (rank K*r), B = concat_k(B_k), so
        B @ A = sum_k c_k B_k A_k exactly, with no cross terms.
    """
    K, r, d_in = A_bank.shape
    d_out = B_bank.shape[1]
    bsz = c.shape[0]
    if mode == "factor":
        A = torch.einsum("bk,kri->bri", c, A_bank)
        B = torch.einsum("bk,kor->bor", c, B_bank)
    elif mode == "delta":
        A = (c.reshape(bsz, K, 1, 1) * A_bank.unsqueeze(0)).reshape(bsz, K * r, d_in)
        B = B_bank.permute(1, 0, 2).reshape(d_out, K * r)
        B = B.unsqueeze(0).expand(bsz, d_out, K * r)
    else:
        raise ValueError("mode must be 'factor' or 'delta', got %r" % (mode,))
    if out_scale is not None:
        B = B * out_scale
    return A, B


def _concat_adapters(A1, B1, A2, B2):
    """Adapter addition done exactly: stack on the rank axis."""
    return torch.cat([A1, A2], dim=-2), torch.cat([B1, B2], dim=-1)


class _BankGenerator(AdapterGenerator):
    """Shared machinery for "coefficients over K adapter directions"."""

    def __init__(self, sites, rank, d_cond, n_basis, per_site, simplex,
                 temperature, d_hidden, d_emb, mode, n_layers=None,
                 learn_temperature=False, uniform_init=True, coeff_depth=2):
        super(_BankGenerator, self).__init__(sites, rank, d_cond, n_layers=n_layers)
        self.n_basis = int(n_basis)
        self.mode = mode
        self.per_site = bool(per_site)
        self.coeff = _CoefficientHead(
            d_cond, n_basis, per_site=per_site, n_layers=self.n_layers,
            d_emb=d_emb, d_hidden=d_hidden, simplex=simplex,
            temperature=temperature, learn_temperature=learn_temperature,
            uniform_init=uniform_init, depth=coeff_depth)
        self.last_coefficients = None

    # -- subclasses provide the bank --------------------------------------
    def bank(self, site_idx):
        raise NotImplementedError

    def _out_scale(self):
        return None

    @property
    def output_rank(self):
        return self.rank * self.n_basis if self.mode == "delta" else self.rank

    def n_output_dims(self):
        return self.n_basis * (len(self.sites) if self.per_site else 1)

    def coefficients(self, h, layer_ids=None, module_ids=None, detach=True):
        """The interpretable object (6.5). (B, K) or (B, Q, K)."""
        h2, batched = self._prep_h(h)
        c = self.coeff(h2, layer_ids, module_ids)
        c = c.detach() if detach else c
        return c if batched else c[0]

    def forward(self, h, layer_ids, module_ids):
        self._check_query(layer_ids, module_ids)
        idx = self._site_indices(layer_ids, module_ids)
        h2, batched = self._prep_h(h)
        c = self.coeff(h2, layer_ids, module_ids)
        self.last_coefficients = c.detach()
        scale = self._out_scale()
        A_list, B_list = [], []
        for j, i in enumerate(idx):
            A_bank, B_bank = self.bank(i)
            ci = c[:, j] if c.dim() == 3 else c
            A, B = _combine(A_bank, B_bank, ci, self.mode, out_scale=scale)
            A_list.append(A)
            B_list.append(B)
        A_list, B_list = self._residual(A_list, B_list, h2, layer_ids, module_ids, idx)
        return self._unbatch(A_list, B_list, batched)

    def _residual(self, A_list, B_list, h2, layer_ids, module_ids, idx):
        return A_list, B_list


class MixtureGenerator(_BankGenerator):
    """6.5's deciding ablation: K coefficients over K *pre-trained* adapters.

    The bank is read-only: adapters trained per cell in Phase 2, loaded through
    `materialize.read_adapter` and held as non-persistent buffers (so a
    checkpoint stays small; `adapter_sources` records where to reload them).

    **This generator does not start as a no-op** and should not: with uniform
    coefficients it starts at the mean of the trained adapters, which is the
    natural baseline for 6.5. `starts_as_noop = False` tells
    `diagnostics.noop_at_init` to report rather than fail.
    """

    kind = "mixture"
    starts_as_noop = False

    def __init__(self, sites, rank, d_cond, adapters, keys=None, per_site=False,
                 simplex=True, temperature=1.0, d_hidden=256, d_emb=32,
                 mode="delta", n_layers=None, residual_basis=0,
                 learn_temperature=False, residual_kw=None):
        bank_keys, banks, sources = _load_adapter_bank(sites, rank, adapters, keys)
        super(MixtureGenerator, self).__init__(
            sites, rank, d_cond, len(bank_keys), per_site, simplex, temperature,
            d_hidden, d_emb, mode, n_layers=n_layers,
            learn_temperature=learn_temperature, uniform_init=True)
        self.bank_keys = bank_keys
        self.adapter_sources = sources
        for i, (A, B) in enumerate(banks):
            # persistent=False: the bank is pre-trained data on disk, not state
            # this generator learned. Saving 12 x 29 M floats per checkpoint is
            # not a reproducibility win, `adapter_sources` is.
            self.register_buffer("A_bank_%d" % i, A, persistent=False)
            self.register_buffer("B_bank_%d" % i, B, persistent=False)
        self.residual = None
        if residual_basis:
            kw = dict(residual_kw or {})
            kw.setdefault("mode", "factor")
            self.residual = BasisGenerator(sites, rank, d_cond,
                                           n_basis=int(residual_basis),
                                           n_layers=n_layers, **kw)

    def bank(self, site_idx):
        return (getattr(self, "A_bank_%d" % site_idx),
                getattr(self, "B_bank_%d" % site_idx))

    @property
    def output_rank(self):
        r = self.rank * self.n_basis if self.mode == "delta" else self.rank
        return r + (self.residual.output_rank if self.residual is not None else 0)

    def _residual(self, A_list, B_list, h2, layer_ids, module_ids, idx):
        if self.residual is None:
            return A_list, B_list
        rA, rB = self.residual(h2, layer_ids, module_ids)
        out_a, out_b = [], []
        for a, b, ra, rb in zip(A_list, B_list, rA, rB):
            a2, b2 = _concat_adapters(a, b, ra, rb)
            out_a.append(a2)
            out_b.append(b2)
        return out_a, out_b

    def coefficient_table(self, embeddings, layer_ids=None, module_ids=None):
        """{cell key: coefficients} -- goes straight into the paper (6.5)."""
        out = collections.OrderedDict()
        for k in sorted(embeddings):
            h = torch.as_tensor(embeddings[k]).float()
            out[k] = self.coefficients(h, layer_ids, module_ids).cpu().numpy()
        return out


class BasisGenerator(_BankGenerator):
    """6.1's predicted winner: coefficients over a *learned* basis.

    `A = sum_k c_k A_k`, `B = sum_k c_k B_k` with `{A_k, B_k}` trainable and
    `c = g(h, layer, module)`. Two coefficient regimes, and the distinction is
    the whole point at this scale:

    * `per_site=False` -- one coefficient vector per conditioning, shared across
      all Q sites. The conditioning chooses K numbers, full stop. With 12
      training cells and K ~ 8 this is a model you can defend.
    * `per_site=True` (default, matching 4.1's per-query-point factorisation) --
      coefficients depend on the query point, so the conditioning chooses K x Q
      numbers. More expressive, closer to free generation, and correspondingly
      easier to memorise with.

    Basis parameters cost `K x site_budget`. That is a large *trainable* count
    but not a large *generated* count, and 6.7's collapse risk is driven by the
    latter -- `parameter_report` prints them separately for exactly this reason.
    """

    kind = "basis"

    def __init__(self, sites, rank, d_cond, n_basis=8, per_site=True,
                 simplex=False, temperature=1.0, d_hidden=256, d_emb=32,
                 mode="factor", n_layers=None, zero_init="gate",
                 learn_temperature=False, basis_std=None):
        super(BasisGenerator, self).__init__(
            sites, rank, d_cond, n_basis, per_site, simplex, temperature,
            d_hidden, d_emb, mode, n_layers=n_layers,
            learn_temperature=learn_temperature,
            uniform_init=bool(simplex))
        if zero_init not in ("gate", "b_basis", "none"):
            raise ValueError("zero_init must be 'gate' | 'b_basis' | 'none'")
        self.zero_init = zero_init
        K = self.n_basis
        A_params, B_params = [], []
        for s in sites:
            A = torch.empty(K, rank, s.d_in)
            for k in range(K):
                nn.init.kaiming_uniform_(A[k], a=math.sqrt(5))
            B = torch.zeros(K, s.d_out, rank)
            if zero_init != "b_basis":
                std = basis_std if basis_std else 1.0 / math.sqrt(rank * K)
                B.normal_(0.0, std)
            A_params.append(nn.Parameter(A))
            B_params.append(nn.Parameter(B))
        self.A_basis = nn.ParameterList(A_params)
        self.B_basis = nn.ParameterList(B_params)
        if zero_init == "gate":
            self.out_scale = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("out_scale", None)
        self.starts_as_noop = zero_init in ("gate", "b_basis")

    def bank(self, site_idx):
        return self.A_basis[site_idx], self.B_basis[site_idx]

    def _out_scale(self):
        return self.out_scale if self.out_scale is not None else None


# --------------------------------------------------------------------------
# Bank loading
# --------------------------------------------------------------------------

def _load_adapter_bank(sites, rank, adapters, keys=None):
    """-> (keys, [(A (K,r,d_in), B (K,d_out,r))] per site, sources).

    `adapters` may be a list of PEFT dirs, a dict key -> dir, or a dict
    key -> {rel_name: (A, B)} (already in memory). Every adapter must cover
    every site at the handle's rank -- a partial bank would silently make some
    sites weaker than others, which is the kind of thing that shows up as an
    inexplicable per-module result three weeks later.
    """
    if isinstance(adapters, dict):
        keys = list(keys or sorted(adapters.keys()))
        items = [adapters[k] for k in keys]
    else:
        items = list(adapters)
        keys = list(keys or ["adapter_%02d" % i for i in range(len(items))])
    if not items:
        raise ValueError("empty adapter bank -- MixtureGenerator needs the "
                         "Phase-2 per-cell adapters (6.5)")
    if len(keys) != len(items):
        raise ValueError("%d keys for %d adapters" % (len(keys), len(items)))

    loaded, sources = [], []
    for k, it in zip(keys, items):
        if isinstance(it, dict):
            loaded.append(it)
            sources.append(None)
        else:
            factors, cfg, _ = materialize.read_adapter(it)
            if int(cfg.get("r", rank)) != int(rank):
                raise ValueError("adapter %s has r=%s but the bank needs r=%d"
                                 % (it, cfg.get("r"), rank))
            loaded.append(factors)
            sources.append(str(it))

    banks = []
    for s in sites:
        As, Bs = [], []
        for k, f in zip(keys, loaded):
            ab = f.get(s.rel_name)
            if ab is None:
                raise KeyError(
                    "adapter %r has no factors for %s. Every adapter in the "
                    "bank must cover every site; retrain it on this target set."
                    % (k, s.rel_name))
            A = torch.as_tensor(ab[0]).float()
            B = torch.as_tensor(ab[1]).float()
            if tuple(A.shape) != (rank, s.d_in) or tuple(B.shape) != (s.d_out, rank):
                raise ValueError(
                    "adapter %r at %s: got A%r B%r, expected A(%d,%d) B(%d,%d)"
                    % (k, s.rel_name, tuple(A.shape), tuple(B.shape),
                       rank, s.d_in, s.d_out, rank))
            As.append(A)
            Bs.append(B)
        banks.append((torch.stack(As, 0), torch.stack(Bs, 0)))
    return keys, banks, sources


# --------------------------------------------------------------------------
# Fixed-A support (T2L_PLAN E3)
# --------------------------------------------------------------------------

def bank_mean_A(sites, rank, adapters, keys=None):
    """{rel_name: mean A over a bank of adapters}, in the bank's own gauge.

    Only defensible because the bank shares a gauge: every per-version oracle
    was trained from the same `seed: 42` init, so the A factors end at pairwise
    cosine 0.90-0.95 and their mean is close to any member (T2L_PLAN A.3).
    Averaging A factors from adapters trained with *different* seeds would be
    averaging unrelated bases and is not what this is for.
    """
    keys, banks, _sources = _load_adapter_bank(sites, rank, adapters, keys)
    out = collections.OrderedDict()
    for s, (A_bank, _B_bank) in zip(sites, banks):
        out[s.rel_name] = A_bank.mean(0)
    return out


def _peft_init_A(sites, rank, generator=None):
    """PEFT's own LoRA-A init (Kaiming-uniform, a=sqrt(5)), one per site."""
    out = collections.OrderedDict()
    for s in sites:
        A = torch.empty(rank, s.d_in)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        out[s.rel_name] = A
    return out


def resolve_fixed_A(fixed_A, sites, rank, adapters=None, keys=None):
    """-> [A (rank, d_in)] in site order, for every `fixed_A` spelling."""
    if fixed_A is None:
        raise ValueError("resolve_fixed_A called with None")
    if isinstance(fixed_A, str):
        if fixed_A == "init":
            fixed_A = _peft_init_A(sites, rank)
        elif fixed_A == "bank_mean":
            if adapters is None:
                raise ValueError(
                    "fixed_A='bank_mean' needs bank_adapters=<adapter dirs>; "
                    "the mean is taken over that bank's A factors")
            fixed_A = bank_mean_A(sites, rank, adapters, keys)
        else:
            raise ValueError("fixed_A must be None, 'init', 'bank_mean', a "
                             "{rel_name: A} dict or a per-site sequence, got %r"
                             % (fixed_A,))
    out = []
    if isinstance(fixed_A, dict):
        for s in sites:
            if s.rel_name not in fixed_A:
                raise KeyError("fixed_A has no A for site %s" % s.rel_name)
            out.append(torch.as_tensor(fixed_A[s.rel_name]).float())
    else:
        seq = list(fixed_A)
        if len(seq) != len(sites):
            raise ValueError("fixed_A has %d entries for %d sites"
                             % (len(seq), len(sites)))
        out = [torch.as_tensor(a).float() for a in seq]
    for s, A in zip(sites, out):
        if tuple(A.shape) != (rank, s.d_in):
            raise ValueError("fixed A at %s is %r, expected (%d, %d)"
                             % (s.rel_name, tuple(A.shape), rank, s.d_in))
    return out


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

GENERATOR_KINDS = ("free", "mixture", "basis", "t2l")


def build_generator(kind, sites, rank, d_cond, **kw):
    """`kind in {"free", "mixture", "basis", "t2l"}` -- 6.5's arms plus T2L.

    `"t2l"` is `t2l.T2LHypernet`, the reconstruction-trained architecture, made
    available here so the *same* architecture can be trained end-to-end through
    the policy (T2L_PLAN E2) instead of only against oracle deltas. It is
    imported lazily: t2l.py imports this module for the base class, so a
    top-level import would be circular.
    """
    if kind == "free":
        return FreeHypernet(sites, rank, d_cond, **kw)
    if kind == "mixture":
        if "adapters" not in kw:
            raise ValueError(
                "kind='mixture' needs adapters=<per-cell adapter dirs>; they "
                "come from Phase 2 (train one LoRA per cell first, 6.5)")
        return MixtureGenerator(sites, rank, d_cond, **kw)
    if kind == "basis":
        return BasisGenerator(sites, rank, d_cond, **kw)
    if kind == "t2l":
        from . import t2l as t2l_mod
        return t2l_mod.T2LHypernet(sites, rank, d_cond, **kw)
    raise ValueError("unknown generator kind %r (have %r)" % (kind, GENERATOR_KINDS))


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def estimate_trainable(kind, sites, rank, d_cond, d_hidden=None, d_emb=None,
                       depth=2, head_rank=None, cross_layer=False, n_layers=None,
                       n_basis=8, per_site=None, learn_temperature=False,
                       zero_init="gate", mode=None, residual_basis=0,
                       coeff_depth=2, fixed_A=False):
    """Parameter counts WITHOUT allocating anything.

    A dense-headed `FreeHypernet` on the real `all` target set is a multi-GB
    allocation; costing the sweep should not require the RAM to run it. Every
    term here mirrors a constructor above and `--selftest` asserts exact
    agreement with the built modules, so this cannot silently drift.

    Returns (components OrderedDict, buffers int).
    """
    sites = list(sites)
    n_layers = int(n_layers or (max(s.layer for s in sites) + 1))
    n_mtypes = len(targets.MODULE_TYPES)
    budget = targets.site_budget(sites, rank)
    c = collections.OrderedDict()

    if kind == "free":
        d_hidden = int(d_hidden or 512)
        d_emb = int(d_emb or 64)
        c["layer_emb"] = n_layers * d_emb
        c["module_emb"] = n_mtypes * d_emb
        t = (d_cond + 2 * d_emb) * d_hidden + d_hidden
        t += max(0, depth - 1) * (d_hidden * d_hidden + d_hidden)
        c["trunk"] = t
        if cross_layer:
            d = d_hidden
            c["mixer"] = (2 * d) + (3 * d * d + 3 * d) + (d * d + d) + (2 * d) \
                + (d * 2 * d + 2 * d) + (2 * d * d + d)
        heads = 0
        shapes = {}
        for s in sites:
            shapes[s.module] = (s.d_in, s.d_out)
        for d_in, d_out in shapes.values():
            out_a, out_b = rank * d_in, d_out * rank
            if head_rank:
                k = int(head_rank)
                if not fixed_A:
                    heads += d_hidden * k + k * out_a + out_a
                heads += d_hidden * k + k * out_b + out_b
            else:
                if not fixed_A:
                    heads += d_hidden * out_a + out_a
                heads += d_hidden * out_b + out_b
        c["heads"] = heads
        # the frozen A factors are buffers, not parameters
        return c, (sum(rank * s.d_in for s in sites) if fixed_A else 0)

    # mixture / basis share the coefficient head
    d_hidden = int(d_hidden or 256)
    d_emb = int(d_emb or 32)
    if per_site is None:
        per_site = (kind == "basis")
    d_in_c = d_cond + (2 * d_emb if per_site else 0)
    coeff = d_in_c * d_hidden + d_hidden
    coeff += max(0, coeff_depth - 2) * (d_hidden * d_hidden + d_hidden)
    coeff += d_hidden * n_basis + n_basis
    if per_site:
        coeff += n_layers * d_emb + n_mtypes * d_emb
    if learn_temperature:
        coeff += 1
    c["coeff"] = coeff
    # log_temp is a 1-element non-persistent buffer unless it is learned
    buffers = 0 if learn_temperature else 1
    if kind == "mixture":
        buffers += n_basis * budget
        if residual_basis:
            sub, sub_buf = estimate_trainable("basis", sites, rank, d_cond,
                                              n_basis=residual_basis,
                                              n_layers=n_layers)
            c["residual"] = sum(sub.values())
            buffers += sub_buf
        return c, buffers
    if kind == "basis":
        c["A_basis"] = n_basis * sum(rank * s.d_in for s in sites)
        c["B_basis"] = n_basis * sum(rank * s.d_out for s in sites)
        if zero_init == "gate":
            c["out_scale"] = 1
        return c, buffers
    raise ValueError("unknown generator kind %r" % (kind,))


def parameter_report(generator, sites=None, rank=None, n_train_cells=12,
                     fh=None, show_components=True):
    """Generator size vs adapter size vs generated degrees of freedom (6.7).

    The last column is the one 6.1/6.7 actually warn about: how many numbers the
    conditioning has to choose, against how many distinct interfaces exist to
    learn them from.
    """
    import sys
    fh = fh or sys.stdout
    sites = sites or generator.sites
    rank = rank or generator.rank
    budget = targets.site_budget(sites, rank)
    out_dims = generator.n_output_dims()
    comps = collections.OrderedDict()
    for name, p in generator.named_parameters():
        top = name.split(".")[0]
        comps[top] = comps.get(top, 0) + p.numel()
    buf = sum(b.numel() for b in generator.buffers())
    total = sum(comps.values())

    print("generator=%s  sites=%d  rank=%d (emits rank %d)  d_cond=%d"
          % (generator.kind, len(sites), rank, generator.output_rank,
             generator.d_cond), file=fh)
    if show_components:
        for k, v in sorted(comps.items(), key=lambda kv: -kv[1]):
            print("  %-22s %16s  (%5.1f%%)"
                  % (k, "{:,}".format(v), 100.0 * v / max(1, total)), file=fh)
    print("  %-22s %16s" % ("TRAINABLE TOTAL", "{:,}".format(total)), file=fh)
    if buf:
        print("  %-22s %16s  (frozen adapter bank)" % ("buffers", "{:,}".format(buf)),
              file=fh)
    print("  %-22s %16s  (targets.site_budget: a plain LoRA on these sites)"
          % ("adapter params", "{:,}".format(budget)), file=fh)
    print("  %-22s %16s  (degrees of freedom the conditioning chooses)"
          % ("generated dims", "{:,}".format(out_dims)), file=fh)
    print("  %-22s %16s  x = generated dims per training interface"
          % ("dims / train cell", "{:,.1f}".format(out_dims / float(n_train_cells))),
          file=fh)
    print("  %-22s %16s" % ("generator / adapter",
                            "%.2fx" % (total / float(budget))), file=fh)
    return {"kind": generator.kind, "components": comps, "trainable": total,
            "buffers": buf, "adapter_params": budget, "generated_dims": out_dims,
            "n_sites": len(sites), "rank": rank, "output_rank": generator.output_rank,
            "dims_per_cell": out_dims / float(n_train_cells)}


# --------------------------------------------------------------------------
# Diversity (9's risk register: "contrastive term over generated adapters")
# --------------------------------------------------------------------------

def adapter_signature(A_list, B_list, n_probe=8, seed=0, normalize_sites=False):
    """A cheap, functional fingerprint of a generated adapter.

    Comparing adapters by their raw factors is meaningless (4.4: `BA = (BR)(R^-1 A)`),
    and materialising `dW` per site is 12288x4096 floats. So sketch the operator
    instead: with fixed random `U (d_in, p)` and `V (d_out, q)`,
    `V^T (B A) U` is a `p*q` linear read-out of the actual weight delta.
    Deterministic in `seed`, differentiable, and invariant to nothing that
    matters.

    Returns (Bsz, n_sites*p*q) for batched factors, or (n_sites*p*q,).
    """
    batched = A_list[0].dim() == 3
    g = torch.Generator().manual_seed(int(seed))
    parts = []
    for A, B in zip(A_list, B_list):
        if not batched:
            A, B = A.unsqueeze(0), B.unsqueeze(0)
        d_in, d_out = A.shape[-1], B.shape[-2]
        p = min(n_probe, d_in)
        q = min(n_probe, d_out)
        U = (torch.randn(d_in, p, generator=g) / math.sqrt(d_in)).to(A.device, A.dtype)
        V = (torch.randn(d_out, q, generator=g) / math.sqrt(d_out)).to(B.device, B.dtype)
        t = torch.matmul(A, U)                      # (B, r, p)
        t = torch.matmul(B, t)                      # (B, d_out, p)
        t = torch.matmul(V.transpose(0, 1), t)      # (B, q, p)
        t = t.reshape(t.shape[0], -1)
        if normalize_sites:
            t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        parts.append(t)
    sig = torch.cat(parts, dim=-1)
    return sig if batched else sig[0]


def diversity_penalty(A_list, B_list, kind="cosine", groups=None, n_probe=4,
                      seed=0, temperature=0.2, eps=1e-8, center=False):
    """Penalise "mean adapter plus noise" (9, 6.7). Lower is more diverse.

    kind="cosine"   mean pairwise cosine similarity of the signatures. NOT
                    mean-centred by default, and that matters: under collapse
                    every adapter is `mean + iid noise`, so centring throws away
                    the mean -- the very thing that is shared -- and the residual
                    noise directions are near-orthogonal, giving ~0. The
                    uncentred similarity is ~1 under collapse and drops as the
                    adapters genuinely differ, which is the signal you want.
                    `center=True` measures alignment of the residuals instead.
    kind="variance" negative mean variance of unit-normalised signatures.
    kind="infonce"  needs `groups` (e.g. two augmented views of the same
                    interface share a group id): pulls views of one interface
                    together and pushes different interfaces apart. This is the
                    form 6.7's augmentation probe wants, because "diverse" must
                    mean "varies with the interface", not "varies at all".

    Requires batched factors -- a penalty over a batch of one is meaningless.
    """
    if A_list[0].dim() != 3:
        raise ValueError("diversity_penalty needs per-example factors "
                         "(A of shape (Bsz, r, d_in)); got %r"
                         % (tuple(A_list[0].shape),))
    X = adapter_signature(A_list, B_list, n_probe=n_probe, seed=seed)
    bsz = X.shape[0]
    if bsz < 2:
        raise ValueError("diversity_penalty needs a batch of >= 2 conditionings")
    if kind == "variance":
        Xn = X / X.norm(dim=-1, keepdim=True).clamp_min(eps)
        return -Xn.var(dim=0, unbiased=False).mean()
    Xc = X - X.mean(0, keepdim=True) if (center and kind == "cosine") else X
    Xn = Xc / Xc.norm(dim=-1, keepdim=True).clamp_min(eps)
    S = Xn @ Xn.transpose(0, 1)
    if kind == "cosine":
        off = ~torch.eye(bsz, dtype=torch.bool, device=S.device)
        return S[off].mean()
    if kind == "infonce":
        if groups is None:
            raise ValueError("kind='infonce' needs groups=(Bsz,) ids")
        groups = torch.as_tensor(groups, device=S.device).reshape(-1)
        logits = S / float(temperature)
        eye = torch.eye(bsz, dtype=torch.bool, device=S.device)
        logits = logits.masked_fill(eye, -1e9)
        pos = (groups.unsqueeze(0) == groups.unsqueeze(1)) & ~eye
        if not bool(pos.any()):
            raise ValueError("no positive pairs in `groups` -- InfoNCE needs at "
                             "least two views of one interface in the batch")
        log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        n_pos = pos.sum(-1).clamp_min(1)
        return -((log_prob * pos).sum(-1) / n_pos)[pos.any(-1)].mean()
    raise ValueError("unknown kind %r" % (kind,))


# --------------------------------------------------------------------------
# Materialising generated adapters
# --------------------------------------------------------------------------

def generate_for_cells(generator, embeddings, handle=None, device=None,
                       keys=None, dtype=None):
    """{cell key: {rel_name: (A, B)}} -- ready for `materialize.write_adapter`.

    Use the handle's query ids when you have one, so the emitted order is the
    order the injected sites are in (inject.py:243-249). With a handle you can
    also go straight through `materialize.write_from_handle`:

        factors = generate_for_cells(gen, emb, handle)
        handle.load_state(factors["wiki_e5"])
        materialize.write_from_handle(out_dir, handle)

    ...but read `materialize_generated` first if the generator emits a rank
    other than the handle's (mixture in delta mode does).
    """
    src = handle if handle is not None else generator
    layer_ids, module_ids = src.query_ids(device)
    sites = src.sites
    out = collections.OrderedDict()
    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        for k in (keys or sorted(embeddings)):
            h = torch.as_tensor(embeddings[k]).float()
            if device is not None:
                h = h.to(device)
            A_list, B_list = generator(h, layer_ids, module_ids)
            d = collections.OrderedDict()
            for s, A, B in zip(sites, A_list, B_list):
                A = A.detach().cpu()
                B = B.detach().cpu()
                if dtype is not None:
                    A, B = A.to(dtype), B.to(dtype)
                d[s.rel_name] = (A, B)
            out[k] = d
    if was_training:
        generator.train()
    return out


def materialize_generated(factors_by_key, handle, out_root, run="run",
                          generator=None, base_model=None, meta=None,
                          dtype=None, trained_scaling=None,
                          allow_scale_mismatch=False):
    """Write one PEFT adapter per conditioning, with the scaling preserved.

    The trap this exists to avoid: `InjectedLoRALinear` applies
    `scaling = alpha / r` fixed at *injection* time, but PEFT and vLLM recompute
    it from the adapter's own `r`. A mixture in `mode="delta"` emits rank `K*r`,
    so writing it with the handle's alpha would serve an adapter `K` times
    weaker than the one that was trained -- a silent numerical mismatch on top
    of the silent name mismatch 4.5 already warns about. So alpha is rescaled by
    `output_rank / handle.rank`.

    **One alpha convention, asserted here** (T2L_PLAN Part D). The rescale above
    keeps `served alpha/r == handle.alpha / handle.rank`, so this function is
    self-consistent -- but the *handle* is built by the caller, and
    `materialize_generated_any.py` deliberately built it at alpha 32 for a
    generator trained at alpha 16, putting a silent 2x on every published 9B
    mixture/LOO number (FINDINGS 17). Pass `trained_scaling=alpha/r as trained`
    and a mismatch raises instead of being discovered by a magnitude sweep three
    weeks later. Legacy numbers are reproduced with
    `allow_scale_mismatch=True`, which records the factor in the adapter's meta
    rather than hiding it. Both scalings always land in `adaptercl_meta.json`.
    """
    out_rank = generator.output_rank if generator is not None else handle.rank
    alpha = handle.alpha * (float(out_rank) / float(handle.rank))
    served_scaling = alpha / float(out_rank)
    ratio = None
    if trained_scaling is not None:
        trained_scaling = float(trained_scaling)
        if trained_scaling <= 0:
            raise ValueError("trained_scaling must be positive, got %r"
                             % (trained_scaling,))
        ratio = served_scaling / trained_scaling
        if abs(ratio - 1.0) > 1e-6 and not allow_scale_mismatch:
            raise ValueError(
                "alpha convention mismatch: this adapter would be SERVED at "
                "alpha/r = %.6g but the generator was TRAINED at %.6g -- every "
                "emitted dW would be %.4gx the one that was learned. Build the "
                "handle with alpha = %.6g * rank, or pass "
                "allow_scale_mismatch=True to reproduce a legacy run on "
                "purpose." % (served_scaling, trained_scaling, ratio,
                              trained_scaling))
    modules = sorted(set(s.module for s in handle.sites))
    written = collections.OrderedDict()
    for key, factors in factors_by_key.items():
        d = materialize.generated_adapter_dir(run, key, root=out_root)
        info = {"generator": getattr(generator, "kind", "unknown"),
                "conditioning_key": key,
                "handle_rank": handle.rank, "emitted_rank": out_rank,
                "alpha_rescaled_from": handle.alpha,
                "served_alpha_over_r": served_scaling,
                "trained_alpha_over_r": trained_scaling,
                "served_over_trained": ratio,
                "granularity": "per_episode"}
        if meta:
            info.update(meta)
        materialize.write_adapter(d, factors, handle.sites, out_rank, alpha,
                                  modules, base_model=base_model, meta=info,
                                  dtype=dtype)
        written[key] = d
    return written


# --------------------------------------------------------------------------
# Selftest
# --------------------------------------------------------------------------

def small_free_trainable(sites, d_cond, rank=2, d_hidden=32, d_emb=8):
    """Trainable count of the ordinary (A-and-B) free generator, for the
    fixed_A comparison in the selftest."""
    comps, _buf = estimate_trainable("free", sites, rank, d_cond,
                                     d_hidden=d_hidden, d_emb=d_emb)
    return sum(comps.values())


def _selftest():
    from . import toy

    ok = []

    def check(name, cond, extra=""):
        ok.append(bool(cond))
        print("  %-52s %s %s" % (name, "PASS" if cond else "FAIL", extra))

    torch.manual_seed(0)
    model, cfg = toy.tiny_model()
    sites = targets.enumerate_sites(cfg, "attn_mlp")
    handle = inject_mod.inject(model, "attn_mlp", rank=2, cfg=cfg)
    layer_ids, module_ids = handle.query_ids()
    d_cond, Q = 12, len(handle)
    print("toy model: %d sites, rank=2, d_cond=%d, layers=%s"
          % (Q, d_cond, cfg["num_hidden_layers"]))

    def shapes_ok(gen, A_list, B_list, bsz=None):
        r = gen.output_rank
        for (d_in, d_out), A, B in zip(handle.shapes(), A_list, B_list):
            want_a = (r, d_in) if bsz is None else (bsz, r, d_in)
            want_b = (d_out, r) if bsz is None else (bsz, d_out, r)
            if tuple(A.shape) != want_a or tuple(B.shape) != want_b:
                return False, "%r/%r != %r/%r" % (tuple(A.shape), tuple(B.shape),
                                                  want_a, want_b)
        return True, ""

    def _fwd_fn(gen, bsz=2):
        ids, labels = toy.tiny_batch(cfg, batch=bsz)
        h = torch.randn(bsz, d_cond)

        def fwd():
            A_list, B_list = gen(h, layer_ids, module_ids)
            handle.set_factors(A_list, B_list)
            out = model(ids, labels=labels)
            handle.clear()
            return out.loss
        return fwd

    def grad_check(gen, bsz=2):
        """Loss through InjectedLoRALinear -> generator params (inject.py:390)."""
        return inject_mod.check_gradient_flow(handle, gen, _fwd_fn(gen, bsz))

    def warm_grad_check(gen, bsz=2, lr=1.0, steps=2):
        """Same, after a few steps -- see module docstring point 5."""
        fwd = _fwd_fn(gen, bsz)
        opt = torch.optim.SGD([p for p in gen.parameters() if p.requires_grad], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            fwd().backward()
            opt.step()
        return inject_mod.check_gradient_flow(handle, gen, fwd)

    print("[1] FreeHypernet")
    free = FreeHypernet(sites, 2, d_cond, d_hidden=32, d_emb=8)
    A, B = free(torch.randn(d_cond), layer_ids, module_ids)
    good, why = shapes_ok(free, A, B)
    check("unbatched shapes match handle.shapes()", good, why)
    A, B = free(torch.randn(3, d_cond), layer_ids, module_ids)
    good, why = shapes_ok(free, A, B, bsz=3)
    check("batched shapes match handle.shapes()", good, why)
    check("B is exactly zero at init (8.3 no-op)",
          all(float(b.abs().max()) == 0.0 for b in B))
    check("A is not zero at init", max(float(a.abs().max()) for a in A) > 0)
    check("per-module-type heads, not padding",
          len(free.heads) == len(set(s.module for s in sites)) == 7,
          "%d heads" % len(free.heads))
    g = grad_check(free)
    check("gradients reach the generator", g["ok"] and g["n_with_grad"] > 5,
          "%d/%d tensors, loss=%.4f" % (g["n_with_grad"], g["n_params"], g["loss"]))
    b_head_grads = [v for k, v in g["grad_norms"].items() if ".b." in k and v > 0]
    check("... including the zero-initialised B heads", len(b_head_grads) >= 7,
          "%d B-head tensors with grad" % len(b_head_grads))
    trunk_g0 = [v for k, v in g["grad_norms"].items()
                if k.startswith("trunk") and v > 0]
    check("at init the trunk is dead (docstring pt 5)", len(trunk_g0) == 0,
          "%d/%d trunk tensors with grad" % (len(trunk_g0), 4))
    free_w = FreeHypernet(sites, 2, d_cond, d_hidden=32, d_emb=8)
    gw = warm_grad_check(free_w)
    trunk_g1 = [v for k, v in gw["grad_norms"].items()
                if k.startswith(("trunk", "layer_emb", "module_emb")) and v > 0]
    check("after 2 steps gradients reach trunk + embeddings", len(trunk_g1) >= 5,
          "%d tensors" % len(trunk_g1))
    freex = FreeHypernet(sites, 2, d_cond, d_hidden=32, d_emb=8, cross_layer=True)
    A2, B2 = freex(torch.randn(2, d_cond), layer_ids, module_ids)
    good, why = shapes_ok(freex, A2, B2, bsz=2)
    check("cross_layer=True keeps the contract", good, why)
    check("cross_layer adds a query mixer", freex.mixer is not None
          and sum(p.numel() for p in freex.mixer.parameters()) > 0)
    check("cross_layer still starts as a no-op",
          all(float(b.abs().max()) == 0.0 for b in B2))
    g = warm_grad_check(freex)
    mixer_grads = [v for k, v in g["grad_norms"].items() if k.startswith("mixer") and v > 0]
    check("... and (after 2 steps) gradients reach the mixer",
          len(mixer_grads) > 0, "%d tensors" % len(mixer_grads))
    small = FreeHypernet(sites, 2, d_cond, d_hidden=32, d_emb=8, head_rank=4)
    fixa = FreeHypernet(sites, 2, d_cond, d_hidden=32, d_emb=8, fixed_A="init")
    fixed_As = [getattr(fixa, "A_fixed_%d" % i) for i in range(len(sites))]
    Af, Bf = fixa(torch.randn(3, d_cond), layer_ids, module_ids)
    check("fixed_A: A is the frozen constant for every example",
          all(torch.equal(a[j], fixed_As[q].to(a.dtype))
              for q, a in enumerate(Af) for j in range(3)))
    check("fixed_A: no A head at all (half the head budget)",
          all(not h.emit_a for h in fixa.heads.values())
          and fixa.n_trainable() < small_free_trainable(sites, d_cond),
          "%d trainable" % fixa.n_trainable())
    check("fixed_A: generated dims are B only",
          fixa.n_output_dims() == sum(2 * s.d_out for s in sites))
    check("fixed_A: A is a buffer, never a parameter",
          not any("A_fixed" in n for n, _ in fixa.named_parameters())
          and any("A_fixed" in n for n, _ in fixa.named_buffers()))
    check("head_rank shrinks the heads",
          small.n_trainable() < free.n_trainable(),
          "%s vs %s" % ("{:,}".format(small.n_trainable()),
                        "{:,}".format(free.n_trainable())))
    try:
        free(torch.randn(d_cond), layer_ids, torch.full_like(module_ids, 4))
        check("unknown module_id rejected", False)
    except KeyError:
        check("unknown module_id rejected", True)

    print("[2] BasisGenerator")
    basis = BasisGenerator(sites, 2, d_cond, n_basis=4, d_hidden=16, d_emb=8)
    A, B = basis(torch.randn(d_cond), layer_ids, module_ids)
    good, why = shapes_ok(basis, A, B)
    check("unbatched shapes", good, why)
    A, B = basis(torch.randn(3, d_cond), layer_ids, module_ids)
    good, why = shapes_ok(basis, A, B, bsz=3)
    check("batched shapes", good, why)
    check("B exactly zero at init (out_scale gate)",
          all(float(b.abs().max()) == 0.0 for b in B))
    check("basis directions are NOT collinear at init",
          float(torch.linalg.matrix_rank(
              basis.B_basis[0].detach().reshape(4, -1))) == 4)
    g = grad_check(basis)
    check("at init only the out_scale gate moves (docstring pt 5)",
          g["ok"] and g["grad_norms"].get("out_scale", 0) > 0
          and g["n_with_grad"] == 1,
          "%d/%d tensors" % (g["n_with_grad"], g["n_params"]))
    basis_w = BasisGenerator(sites, 2, d_cond, n_basis=4, d_hidden=16, d_emb=8)
    gw = warm_grad_check(basis_w)
    n_basis_grad = len([v for k, v in gw["grad_norms"].items()
                        if k.startswith(("A_basis", "B_basis")) and v > 0])
    n_coeff_grad = len([v for k, v in gw["grad_norms"].items()
                        if k.startswith("coeff") and v > 0])
    check("after 2 steps gradients reach the basis and the coefficients",
          n_basis_grad >= 2 * Q and n_coeff_grad >= 2,
          "%d basis / %d coeff tensors" % (n_basis_grad, n_coeff_grad))
    per_ad = BasisGenerator(sites, 2, d_cond, n_basis=4, per_site=False,
                            d_hidden=16, d_emb=8)
    check("per-adapter coefficients: K dims, not K*Q",
          per_ad.n_output_dims() == 4 and basis.n_output_dims() == 4 * Q,
          "%d vs %d" % (per_ad.n_output_dims(), basis.n_output_dims()))
    A, B = per_ad(torch.randn(2, d_cond), layer_ids, module_ids)
    good, why = shapes_ok(per_ad, A, B, bsz=2)
    check("per-adapter shapes", good, why)
    simp = BasisGenerator(sites, 2, d_cond, n_basis=5, simplex=True, d_hidden=16,
                          d_emb=8)
    c = simp.coefficients(torch.randn(2, d_cond), layer_ids, module_ids)
    check("simplex coefficients sum to 1",
          torch.allclose(c.sum(-1), torch.ones_like(c.sum(-1)), atol=1e-6)
          and bool((c >= 0).all()), "shape %r" % (tuple(c.shape),))

    print("[3] MixtureGenerator")
    bank = {}
    for i, key in enumerate(["wiki_e1", "wiki_e2", "news_e1"]):
        torch.manual_seed(10 + i)
        bank[key] = dict((s.rel_name, (torch.randn(2, s.d_in) * 0.05,
                                       torch.randn(s.d_out, 2) * 0.05))
                         for s in sites)
    mix = MixtureGenerator(sites, 2, d_cond, adapters=bank, d_hidden=16, d_emb=8)
    A, B = mix(torch.randn(d_cond), layer_ids, module_ids)
    good, why = shapes_ok(mix, A, B)
    check("delta mode emits rank K*r", good and mix.output_rank == 6, why)
    c = mix.coefficients(torch.randn(2, d_cond))
    check("coefficients sum to 1 (simplex)",
          torch.allclose(c.sum(-1), torch.ones(2), atol=1e-6))
    check("uniform at init (starts at the mean adapter)",
          torch.allclose(c, torch.full_like(c, 1.0 / 3), atol=1e-6))
    check("mixture is NOT a no-op at init (documented)",
          mix.starts_as_noop is False
          and max(float(b.abs().max()) for b in B) > 0)
    # exactness of the delta-mode mixture
    with torch.no_grad():
        h1 = torch.randn(d_cond)
        A1, B1 = mix(h1, layer_ids, module_ids)
        cc = mix.coefficients(h1)
        want = sum(float(cc[k]) * (bank[key][sites[0].rel_name][1]
                                   @ bank[key][sites[0].rel_name][0])
                   for k, key in enumerate(mix.bank_keys))
        got = B1[0] @ A1[0]
    check("delta mode: BA == sum_k c_k B_k A_k exactly",
          torch.allclose(got, want, atol=1e-5),
          "max |d| = %.2e" % float((got - want).abs().max()))
    mixf = MixtureGenerator(sites, 2, d_cond, adapters=bank, mode="factor",
                            d_hidden=16, d_emb=8)
    A2, B2 = mixf(h1, layer_ids, module_ids)
    check("factor mode keeps rank r", mixf.output_rank == 2
          and tuple(A2[0].shape) == (2, sites[0].d_in))
    with torch.no_grad():
        got_f = B2[0] @ A2[0]
    check("factor mode differs from the exact mixture (cross terms)",
          not torch.allclose(got_f, want, atol=1e-4),
          "max |d| = %.2e" % float((got_f - want).abs().max()))
    g = warm_grad_check(MixtureGenerator(sites, 2, d_cond, adapters=bank,
                                         d_hidden=16, d_emb=8))
    check("gradients reach the coefficient head only (bank is frozen)",
          g["ok"] and g["n_with_grad"] == 4
          and all(not k.startswith(("A_bank", "B_bank")) for k in g["grad_norms"]),
          "%d/%d tensors" % (g["n_with_grad"], g["n_params"]))
    mixr = MixtureGenerator(sites, 2, d_cond, adapters=bank, residual_basis=2,
                            d_hidden=16, d_emb=8)
    A3, B3 = mixr(torch.randn(d_cond), layer_ids, module_ids)
    good, why = shapes_ok(mixr, A3, B3)
    check("learned residual concatenates on the rank axis",
          good and mixr.output_rank == 8, why)
    try:
        MixtureGenerator(sites, 2, d_cond,
                         adapters={"a": {sites[0].rel_name: bank["wiki_e1"][sites[0].rel_name]}},
                         d_hidden=16, d_emb=8)
        check("incomplete bank rejected", False)
    except KeyError:
        check("incomplete bank rejected", True)

    print("[4] factory + sizes")
    for kind in GENERATOR_KINDS:
        kw = {"adapters": bank} if kind == "mixture" else {}
        gen = build_generator(kind, sites, 2, d_cond, **kw)
        check("build_generator(%r)" % kind, gen.kind == kind)
    check("n_adapter_dims == targets.site_budget",
          free.n_adapter_dims() == targets.site_budget(sites, 2))
    check("free: generated dims == adapter dims",
          free.n_output_dims() == free.n_adapter_dims())
    check("mixture: generated dims == K", mix.n_output_dims() == 3)

    print("[5] signatures, diversity, materialisation")
    A, B = basis(torch.randn(4, d_cond), layer_ids, module_ids)
    sig = adapter_signature(A, B, n_probe=3)
    check("signature shape", tuple(sig.shape) == (4, Q * 9), str(tuple(sig.shape)))
    check("signature is zero for a no-op adapter", float(sig.abs().max()) == 0.0)
    torch.manual_seed(3)
    with torch.no_grad():
        basis.out_scale.fill_(1.0)
    A, B = basis(torch.randn(4, d_cond), layer_ids, module_ids)
    pen = diversity_penalty(A, B, kind="cosine")
    check("cosine penalty is a finite scalar with grad",
          pen.dim() == 0 and pen.requires_grad and abs(float(pen)) <= 1.001,
          "%.4f" % float(pen))
    penv = diversity_penalty(A, B, kind="variance")
    check("variance penalty is negative (more variance = lower loss)",
          float(penv) < 0, "%.4e" % float(penv))
    peni = diversity_penalty(A, B, kind="infonce", groups=[0, 0, 1, 1])
    check("infonce penalty finite", torch.isfinite(peni), "%.4f" % float(peni))
    ident = diversity_penalty([a[:1].repeat(2, 1, 1) for a in A],
                              [b[:1].repeat(2, 1, 1) for b in B], kind="cosine")
    check("identical adapters -> cosine penalty ~ +1 (collapse)",
          float(ident) > 0.99, "%.4f" % float(ident))
    try:
        diversity_penalty([a[0] for a in A], [b[0] for b in B])
        check("unbatched factors rejected", False)
    except ValueError:
        check("unbatched factors rejected", True)

    emb = {"wiki_e1": torch.randn(d_cond).numpy(),
           "wiki_e2": torch.randn(d_cond).numpy()}
    fac = generate_for_cells(basis, emb, handle)
    check("generate_for_cells keys + rel_names",
          sorted(fac) == ["wiki_e1", "wiki_e2"]
          and sorted(fac["wiki_e1"]) == sorted(s.rel_name for s in handle.sites))
    check("factors are 2-D and detached",
          all(a.dim() == 2 and b.dim() == 2 and not a.requires_grad
              for a, b in fac["wiki_e1"].values()))
    missing = handle.load_state(fac["wiki_e1"])
    check("round-trips through handle.load_state", not missing)
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="adaptercl_selftest_")
    try:
        dirs = materialize_generated(fac, handle, tmp, run="selftest",
                                     generator=basis)
        f2, cfg2, meta2 = materialize.read_adapter(dirs["wiki_e1"])
        check("materialize_generated writes a readable PEFT dir",
              len(f2) == Q and cfg2["r"] == basis.output_rank)
        check("... with alpha rescaled to preserve alpha/r",
              abs(cfg2["lora_alpha"] / cfg2["r"] - handle.alpha / handle.rank) < 1e-9,
              "alpha=%s r=%s" % (cfg2["lora_alpha"], cfg2["r"]))
        dirs = materialize_generated(generate_for_cells(mix, emb, handle), handle,
                                     tmp, run="selftest_mix", generator=mix)
        f3, cfg3, _ = materialize.read_adapter(dirs["wiki_e2"])
        check("delta-mode mixture materialises at rank K*r",
              cfg3["r"] == 6 and abs(cfg3["lora_alpha"] / 6.0 - 1.0) < 1e-9,
              "r=%s alpha=%s" % (cfg3["r"], cfg3["lora_alpha"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("[6] parameter_report + analytic estimator")
    rep = parameter_report(mix, n_train_cells=12)
    check("report returns the ratio 6.7 cares about",
          rep["generated_dims"] == 3 and rep["adapter_params"] > 0)
    cases = [
        ("free", free, {"d_hidden": 32, "d_emb": 8}),
        ("free", freex, {"d_hidden": 32, "d_emb": 8, "cross_layer": True}),
        ("free", small, {"d_hidden": 32, "d_emb": 8, "head_rank": 4}),
        ("free", fixa, {"d_hidden": 32, "d_emb": 8, "fixed_A": True}),
        ("basis", basis, {"d_hidden": 16, "d_emb": 8, "n_basis": 4,
                          "per_site": True}),
        ("basis", per_ad, {"d_hidden": 16, "d_emb": 8, "n_basis": 4,
                           "per_site": False}),
        ("mixture", mix, {"d_hidden": 16, "d_emb": 8, "n_basis": 3,
                          "per_site": False}),
        ("mixture", mixr, {"d_hidden": 16, "d_emb": 8, "n_basis": 3,
                           "per_site": False, "residual_basis": 2}),
    ]
    for kind, gen, kw in cases:
        comps, buf = estimate_trainable(kind, sites, 2, d_cond, **kw)
        got = sum(comps.values())
        real_buf = sum(b.numel() for b in gen.buffers())
        check("estimate_trainable matches %s(%s)"
              % (kind, ",".join("%s=%s" % kv for kv in sorted(kw.items())
                                if kv[0] not in ("d_hidden", "d_emb"))),
              got == gen.n_trainable() and buf == real_buf,
              "%d vs %d params, %d vs %d buffers"
              % (got, gen.n_trainable(), buf, real_buf))

    n_fail = len([x for x in ok if not x])
    print("\n%d/%d checks passed" % (len(ok) - n_fail, len(ok)))
    return n_fail


#: The six configurations the Phase-3 sweep costs out. (label, kind, kwargs)
TABLE_CONFIGS = (
    ("free (dense heads, 8.3)", "free", {}),
    ("free (head_rank=32)", "free", {"head_rank": 32}),
    ("free (head_rank=32, cross_layer)", "free",
     {"head_rank": 32, "cross_layer": True}),
    ("mixture K=12 (delta)", "mixture", {"n_basis": 12, "per_site": False}),
    ("mixture K=12 (per-site c)", "mixture", {"n_basis": 12, "per_site": True}),
    ("basis K=8 (per-site c)", "basis", {"n_basis": 8, "per_site": True}),
    ("basis K=8 (per-adapter c)", "basis", {"n_basis": 8, "per_site": False}),
)


def _real_table(rank=16, target_sets=None, n_train_cells=12, model=None):
    """Parameter table for the real Qwen3.5-9B -- config only, no allocation.

    Uses `estimate_trainable` (validated against the built modules in
    `--selftest`): a dense-headed free hypernet on `all` is several GB and this
    has to run on a login node.
    """
    cfg = targets.load_text_config(model)
    n_layers = int(cfg["num_hidden_layers"])
    d_cond = int(cfg["hidden_size"])
    rows = []
    for name in (target_sets or ["attn", "deltanet", "mlp", "attn_mlp", "all"]):
        sites = targets.enumerate_sites(cfg, name)
        budget = targets.site_budget(sites, rank)
        for label, kind, kw in TABLE_CONFIGS:
            comps, buf = estimate_trainable(kind, sites, rank, d_cond,
                                            n_layers=n_layers, **kw)
            n_basis = kw.get("n_basis", 0)
            per_site = kw.get("per_site", False)
            if kind == "free":
                gen_dims = budget
                out_rank = rank
            else:
                gen_dims = n_basis * (len(sites) if per_site else 1)
                out_rank = rank * n_basis if kind == "mixture" else rank
            rows.append((name, len(sites), budget, label, sum(comps.values()),
                         buf, gen_dims, out_rank))
    hdr = ("%-10s %5s %14s | %-33s %15s %15s %14s %5s"
           % ("targets", "sites", "adapter params", "generator", "trainable",
              "frozen bank", "generated dims", "rank"))
    print(hdr)
    print("-" * len(hdr))
    last = None
    for r in rows:
        if last is not None and r[0] != last:
            print("")
        last = r[0]
        print("%-10s %5d %14s | %-33s %15s %15s %14s %5d"
              % (r[0], r[1], "{:,}".format(r[2]), r[3], "{:,}".format(r[4]),
                 "{:,}".format(r[5]) if r[5] else "-", "{:,}".format(r[6]), r[7]))
    print("\nd_cond=%d (= vision out_hidden_size = text hidden_size), rank=%d, "
          "n_layers=%d, training interfaces=%d"
          % (d_cond, rank, n_layers, n_train_cells))
    print("'generated dims' = degrees of freedom the conditioning chooses; "
          "divide by %d for dims per training interface." % n_train_cells)
    return rows


def _fake_bank(sites, rank, k):
    """Zero adapters, only to size a MixtureGenerator without Phase-2 output."""
    return dict(("cell_%02d" % i,
                 dict((s.rel_name, (torch.zeros(rank, s.d_in),
                                    torch.zeros(s.d_out, rank))) for s in sites))
                for i in range(k))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="adapter generators (adapterCL 4.1/6.5)")
    ap.add_argument("--selftest", action="store_true", help="CPU-only checks")
    ap.add_argument("--table", action="store_true",
                    help="parameter table for the real Qwen3.5-9B site sets")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--target-sets", nargs="*", default=None)
    args = ap.parse_args()
    if args.table:
        _real_table(rank=args.rank, target_sets=args.target_sets)
        raise SystemExit(0)
    if args.selftest:
        raise SystemExit(1 if _selftest() else 0)
    ap.print_help()
