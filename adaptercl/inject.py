"""LoRA injection into a frozen base model (adapterCL.md 4.5 / 8.1).

`InjectedLoRALinear` wraps a frozen `nn.Linear`. Its A/B factors are *plain
tensors supplied from outside*, not `nn.Parameter`s, so autograd routes the
gradient straight back to whatever produced them -- a hypernetwork, a mixture
head, or a learned basis. That is the whole trick that makes 4.4's "end-to-end
through the task loss, never reconstruction" implementable.

Three things this adds over the sketch in 8.1:

* **Per-example factors.** A BC batch mixes cells, so every example may need a
  different adapter. Pass A of shape `(Bsz, r, d_in)` and the forward switches
  to a batched path. `(r, d_in)` still means "shared across the batch".
* **Static mode.** For Phase 1/2 (ordinary per-cell LoRA) the same wrapper holds
  `nn.Parameter` factors, so one code path covers plain LoRA, cross-application
  of a trained adapter, and hypernetwork generation. That matters because the
  transfer matrix in 6.2 must evaluate trained adapters through the *same*
  numerical path the hypernetwork uses, or the comparison is confounded.
* **Exact PEFT semantics.** `scaling = alpha / r`, A ~ Kaiming-uniform, B zero,
  optional input dropout -- so `test_peft_parity` in tests/ can assert bitwise
  agreement with `peft` on a toy model.

Requires torch. Import from the `llamafactory` env python (paths.PY_TRAIN).
"""

from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import targets

# --------------------------------------------------------------------------
# The wrapper
# --------------------------------------------------------------------------


class InjectedLoRALinear(nn.Module):
    """A frozen linear plus a LoRA whose factors may be supplied externally.

    Modes
    -----
    injected (default)
        `A`/`B` are plain tensors set by `set_factors`. Gradients flow through
        them to their producer. `clear()` restores base-only behaviour.
    static
        `A`/`B` are `nn.Parameter`s owned by this module -- ordinary LoRA.
        Created by `make_static()`.

    Shapes
    ------
    A: `(r, d_in)` or `(Bsz, r, d_in)`
    B: `(d_out, r)` or `(Bsz, d_out, r)`
    """

    def __init__(self, base_linear, rank=16, alpha=None, dropout=0.0):
        super(InjectedLoRALinear, self).__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("InjectedLoRALinear wraps nn.Linear, got %s"
                            % type(base_linear).__name__)
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout_p = float(dropout)
        self.d_in = base_linear.in_features
        self.d_out = base_linear.out_features
        self.static = False
        self.A = None                # (r, d_in) | (Bsz, r, d_in)
        self.B = None                # (d_out, r) | (Bsz, d_out, r)

    # -- factor plumbing --------------------------------------------------
    def set_factors(self, A, B):
        """Attach externally-produced factors. Tensors, not Parameters."""
        if self.static:
            raise RuntimeError("set_factors() on a static site; call "
                               "make_injected() first")
        if A is not None:
            if A.shape[-1] != self.d_in:
                raise ValueError("A last dim %d != d_in %d" % (A.shape[-1], self.d_in))
            if B.shape[-2] != self.d_out:
                raise ValueError("B dim -2 %d != d_out %d" % (B.shape[-2], self.d_out))
            if A.shape[-2] != B.shape[-1]:
                raise ValueError("rank mismatch: A %r vs B %r"
                                 % (tuple(A.shape), tuple(B.shape)))
        self.A, self.B = A, B

    def clear(self):
        if not self.static:
            self.A = self.B = None

    # -- static (ordinary LoRA) -------------------------------------------
    def make_static(self, init=True, device=None, dtype=None):
        """Convert to an ordinary LoRA with owned parameters."""
        device = device or self.base.weight.device
        dtype = dtype or torch.float32
        A = torch.empty(self.rank, self.d_in, device=device, dtype=dtype)
        B = torch.zeros(self.d_out, self.rank, device=device, dtype=dtype)
        if init:
            # PEFT's default: Kaiming-uniform on A, zeros on B (a=sqrt(5)).
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.static = True
        return self

    def make_injected(self):
        """Drop owned parameters and go back to externally-supplied factors."""
        self.A = self.B = None
        self.static = False
        return self

    def load_factors(self, A, B):
        """Copy trained factors in (used when cross-applying a saved adapter)."""
        A = torch.as_tensor(A)
        B = torch.as_tensor(B)
        if self.static:
            with torch.no_grad():
                self.A.copy_(A.to(self.A.device, self.A.dtype))
                self.B.copy_(B.to(self.B.device, self.B.dtype))
        else:
            dev = self.base.weight.device
            self.set_factors(A.to(dev), B.to(dev))
        return self

    # -- math --------------------------------------------------------------
    def delta_weight(self):
        """`scaling * B @ A` -- the effective weight delta. Batched A/B unsupported."""
        if self.A is None:
            return None
        if self.A.dim() != 2:
            raise ValueError("delta_weight() needs unbatched factors")
        return self.scaling * (self.B @ self.A)

    def merge_into_base(self):
        """Fold the adapter into the base weight (for merge-then-serve evals)."""
        dw = self.delta_weight()
        if dw is None:
            return self.base
        with torch.no_grad():
            self.base.weight.add_(dw.to(self.base.weight.dtype))
        self.clear()
        return self.base

    def forward(self, x):
        out = self.base(x)
        A, B = self.A, self.B
        if A is None or B is None:
            return out
        h = x
        if self.dropout_p > 0.0 and self.training:
            h = F.dropout(h, p=self.dropout_p)
        if A.dim() == 2:
            # shared across the batch
            h = F.linear(F.linear(h, A.to(h.dtype)), B.to(h.dtype))
        else:
            # per-example: x (Bsz, ..., d_in) -> (Bsz, N, d_in)
            bsz = A.shape[0]
            if h.shape[0] != bsz:
                raise ValueError(
                    "per-example factors have batch %d but input batch is %d"
                    % (bsz, h.shape[0]))
            lead = h.shape[:-1]
            flat = h.reshape(bsz, -1, self.d_in)
            z = torch.bmm(flat, A.to(h.dtype).transpose(1, 2))      # (B, N, r)
            z = torch.bmm(z, B.to(h.dtype).transpose(1, 2))          # (B, N, d_out)
            h = z.reshape(*lead, self.d_out)
        return out + self.scaling * h

    def extra_repr(self):
        return "d_in=%d, d_out=%d, r=%d, alpha=%g, mode=%s" % (
            self.d_in, self.d_out, self.rank, self.alpha,
            "static" if self.static else "injected")


# --------------------------------------------------------------------------
# Injecting into a whole model
# --------------------------------------------------------------------------

def _get_parent(model, dotted):
    parts = dotted.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
    return obj, parts[-1]


def detect_lm_prefix(model):
    """Find the dotted path of the module holding `.layers` (the text stack)."""
    for cand in ("model.language_model", "model", "language_model",
                 "base_model.model.model.language_model"):
        obj = model
        ok = True
        for p in cand.split("."):
            if not hasattr(obj, p):
                ok = False
                break
            obj = getattr(obj, p)
        if ok and hasattr(obj, "layers"):
            return cand
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, nn.ModuleList):
            return name.rsplit(".", 1)[0]
    raise RuntimeError("could not locate the decoder stack in %s"
                       % type(model).__name__)


class InjectionHandle(object):
    """Bookkeeping for a set of injected sites, in a stable order.

    The order of `sites` is the order of the leading dim of any (A, B) stack the
    hypernetwork produces, and is what `layer_ids`/`module_ids` index. It is
    derived from `targets.enumerate_sites`, i.e. sorted by (layer, module), so
    it is reproducible across processes and checkpoints.
    """

    def __init__(self, model, sites, modules, rank, alpha, prefix):
        self.model = model
        self.sites = list(sites)              # targets.Site
        self.modules = list(modules)          # InjectedLoRALinear, parallel
        self.rank = rank
        self.alpha = alpha
        self.prefix = prefix
        self._originals = {}

    # -- identity ---------------------------------------------------------
    def __len__(self):
        return len(self.sites)

    @property
    def names(self):
        return [s.name for s in self.sites]

    @property
    def rel_names(self):
        return [s.rel_name for s in self.sites]

    def query_ids(self, device=None):
        """(layer_ids, module_ids) tensors for the hypernetwork, in site order."""
        layer_ids = torch.tensor([s.layer for s in self.sites], dtype=torch.long)
        module_ids = torch.tensor([s.module_id for s in self.sites], dtype=torch.long)
        if device is not None:
            layer_ids, module_ids = layer_ids.to(device), module_ids.to(device)
        return layer_ids, module_ids

    def shapes(self):
        return [(s.d_in, s.d_out) for s in self.sites]

    # -- factors ----------------------------------------------------------
    def set_factors(self, A_list, B_list):
        """Attach one (A, B) per site. Lists, or any indexable of tensors."""
        if len(A_list) != len(self.modules):
            raise ValueError("expected %d A tensors, got %d"
                             % (len(self.modules), len(A_list)))
        for m, a, b in zip(self.modules, A_list, B_list):
            m.set_factors(a, b)

    def clear(self):
        for m in self.modules:
            m.clear()

    def make_static(self, **kw):
        for m in self.modules:
            m.make_static(**kw)
        return self

    def make_injected(self):
        for m in self.modules:
            m.make_injected()
        return self

    def trainable_parameters(self):
        out = []
        for m in self.modules:
            if m.static:
                out.extend([m.A, m.B])
        return out

    def state(self):
        """{rel_name: (A, B)} for the current factors (detached, on CPU)."""
        out = {}
        for s, m in zip(self.sites, self.modules):
            if m.A is None:
                continue
            out[s.rel_name] = (m.A.detach().float().cpu(),
                               m.B.detach().float().cpu())
        return out

    def load_state(self, state, strict=True):
        """Inverse of `state()`; `state` keyed by rel_name."""
        missing = []
        for s, m in zip(self.sites, self.modules):
            if s.rel_name not in state:
                missing.append(s.rel_name)
                continue
            a, b = state[s.rel_name]
            m.load_factors(a, b)
        if strict and missing:
            raise KeyError("%d site(s) missing from state, e.g. %s"
                           % (len(missing), missing[:3]))
        return missing

    # -- teardown ---------------------------------------------------------
    def remove(self):
        """Put the original nn.Linear modules back."""
        for name, orig in self._originals.items():
            parent, attr = _get_parent(self.model, name)
            setattr(parent, attr, orig)
        self._originals = {}
        self.modules = []
        return self.model


def inject(model, modules=None, rank=16, alpha=None, dropout=0.0,
           layers=None, cfg=None, prefix=None, static=False):
    """Wrap every site in a target set with `InjectedLoRALinear`.

    Parameters
    ----------
    modules : target-set name (see targets.TARGET_SETS) or explicit sequence
    rank, alpha, dropout : LoRA hyperparameters (alpha defaults to rank)
    layers : optional set of layer indices to restrict to
    cfg : text_config dict; read from the live model if omitted
    static : create ordinary LoRA parameters instead of injected factors

    Returns an `InjectionHandle`. Also freezes every base parameter -- the whole
    point of 4.1 is that only the generator trains.
    """
    if cfg is None:
        cfg = _text_config_from_model(model)
    prefix = prefix or detect_lm_prefix(model)
    modules = targets.DEFAULT_TARGET_SET if modules is None else modules
    sites = targets.enumerate_sites(cfg, modules, layers=layers, prefix=prefix)

    for p in model.parameters():
        p.requires_grad_(False)

    named = dict(model.named_modules())
    wrapped, kept_sites, originals = [], [], {}
    for s in sites:
        base = named.get(s.name)
        if base is None:
            raise KeyError(
                "site %s not found in %s. Either the target set does not match "
                "this architecture or the module prefix is wrong (detected %r)."
                % (s.name, type(model).__name__, prefix))
        if isinstance(base, InjectedLoRALinear):
            raise RuntimeError("site %s is already injected" % s.name)
        if (base.in_features, base.out_features) != (s.d_in, s.d_out):
            raise ValueError(
                "shape mismatch at %s: config predicts %dx%d, model has %dx%d. "
                "targets.py needs updating for this checkpoint."
                % (s.name, s.d_out, s.d_in, base.out_features, base.in_features))
        w = InjectedLoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)
        parent, attr = _get_parent(model, s.name)
        originals[s.name] = base
        setattr(parent, attr, w)
        wrapped.append(w)
        kept_sites.append(s)

    handle = InjectionHandle(model, kept_sites, wrapped, rank,
                             float(rank if alpha is None else alpha), prefix)
    handle._originals = originals
    if static:
        handle.make_static()
    return handle


def _text_config_from_model(model):
    """Pull a text_config-shaped dict out of a live model's config."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("model has no .config; pass cfg= explicitly")
    text = getattr(cfg, "text_config", None) or cfg
    d = text.to_dict() if hasattr(text, "to_dict") else dict(text)
    d.setdefault("_model_type", getattr(cfg, "model_type", None))
    return d


# --------------------------------------------------------------------------
# Diagnostics used by Phase 0/1 (adapterCL.md 7, "verify gradients reach
# externally-supplied factors")
# --------------------------------------------------------------------------

def check_gradient_flow(handle, producer, forward_fn, atol=0.0):
    """Assert that a loss through the injected sites reaches `producer`.

    `producer` is any nn.Module whose parameters generated the factors;
    `forward_fn()` must run the base model and return a scalar loss.

    Returns a dict with per-parameter grad norms and a boolean `ok`.
    """
    for p in producer.parameters():
        p.grad = None
    loss = forward_fn()
    loss.backward()
    norms = {}
    for name, p in producer.named_parameters():
        norms[name] = 0.0 if p.grad is None else float(p.grad.norm())
    reached = sum(1 for v in norms.values() if v > atol)
    return {"ok": reached > 0, "n_with_grad": reached,
            "n_params": len(norms), "grad_norms": norms,
            "loss": float(loss.detach())}


def check_base_frozen(model):
    """Every base parameter must have requires_grad=False (4.1)."""
    leaked = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            leaked.append(name)
    return {"ok": not leaked, "leaked": leaked}


def adapter_is_noop(handle):
    """True if no site currently carries factors (the zero-init start state)."""
    for m in handle.modules:
        if m.A is None or m.B is None:
            continue
        if float(m.B.abs().sum()) > 0:
            return False
    return True


def random_factors(handle, scale=1.0, device=None, dtype=torch.float32,
                   generator=None, batch=None):
    """Random (A, B) per site, matched in norm to a Kaiming/zero init's scale.

    Used for the 6.7 "generated adapter vs random adapter of matched norm"
    control -- with `scale` set from a real adapter's factor norms.
    """
    device = device or next(handle.model.parameters()).device
    A_list, B_list = [], []
    for s in handle.sites:
        shape_a = (s.d_in,)
        a = torch.randn(*( (batch,) if batch else () ), handle.rank, *shape_a,
                        device=device, dtype=dtype, generator=generator)
        b = torch.randn(*( (batch,) if batch else () ), s.d_out, handle.rank,
                        device=device, dtype=dtype, generator=generator)
        a = a * (scale / math.sqrt(s.d_in))
        b = b * (scale / math.sqrt(handle.rank))
        A_list.append(a)
        B_list.append(b)
    return A_list, B_list
