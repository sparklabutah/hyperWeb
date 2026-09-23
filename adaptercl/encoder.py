"""The conditioning encoder -- what the adapter is generated *from* (4.2).

4.2 makes the screenshot primary and a compact a11y-role signature secondary,
and 6.5 makes "conditioning modality" a required ablation. So all three
modalities are first-class here: `vision`, `structure`, `both`. Everything
downstream (hypernet.py) sees a single `d_cond` vector and does not care which
produced it.

Design decisions worth stating, because they are all load-bearing:

* **Pooling is a plain mean over merged patch tokens, and there is no CLS to
  read instead.** `Qwen3_5VisionModel` is patch_embed -> 27 blocks ->
  `Qwen3_5VisionPatchMerger` (transformers 5.6.0
  models/qwen3_5/modeling_qwen3_5.py:995-1185); there is no class token and no
  learned pooler. Its `pooler_output` is *not* a pooled vector -- it is the
  post-merger token sequence at `out_hidden_size`, which is what the LM
  actually consumes (`get_image_features`, modeling_qwen3_5.py:1500-1521). Mean
  over those tokens is the permutation-invariant summary that stays in the
  policy's own embedding space (4.2: "no extra parameters, ... a space the
  policy already understands"), and for a full-page screenshot it is dominated
  by palette/typography/layout statistics -- exactly the era signal.
  `vision_level="patch"` pools the pre-merger 1152-d stream instead, as an
  ablation.
* **The tower can be loaded without the 19 GB LM.** transformers 5.6.0 has no
  public "load only the vision tower" entry point -- `Qwen3_5VisionModel` is a
  sub-model of `Qwen3_5Model` and its checkpoint keys are prefixed
  `model.visual.`, so `from_pretrained` on it does not resolve them. We do the
  shard-filtered load ourselves (`load_vision_tower`): all 333 `model.visual.*`
  tensors of Qwen3.5-9B live in one shard (`model.safetensors-00004-of-00004`),
  so this touches ~3 GB of file, not 19 GB, and materialises ~0.4 B params.
  `from_full_model(model)` reuses an already-loaded model's tower when you do
  have the LM in memory.
* **Structure is a role histogram, never raw HTML.** 4.2: "every truncation
  knob becomes a confound" -- a DOM string forces choices (max length, which
  attributes, pretty-printed or not) that silently change the conditioning and
  are impossible to report honestly. A fixed-order count vector over
  `capture.ROLE_VOCAB` has no such knob.
* **Cached embeddings hold only the frozen part.** Vision (frozen tower) and
  structure (deterministic) are cached; the *trainable* projection lives in
  `FusedConditioner`, which is part of the generator and trains end-to-end
  (4.4). For `modality="both"` the cached vector is the concatenation
  `[vision ; structure]` and `spec.split_dims()` says where to cut.

Requires torch + PIL + numpy. Import from the `llamafactory` env python
(paths.PY_TRAIN).
"""

from __future__ import print_function

import hashlib
import io
import json
import os

import numpy as np
import torch
import torch.nn as nn

from . import paths

# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------

MODALITIES = ("vision", "structure", "both")
POOLINGS = ("mean", "max", "mean_max", "first")
VISION_LEVELS = ("merged", "patch")
STRUCT_NORMS = ("l2", "l1", "none")

#: Patch grid granularity: patch_size * spatial_merge_size for Qwen3.5 = 32 px
#: per merged token. Pre-fitting images to a multiple of this makes the image
#: processor's smart_resize a no-op, so the capture protocol (4.2) -- not a
#: processor version -- decides the token count.
MERGED_PIXEL = 32

#: Defaults chosen so a 1280x800 screenshot survives at ~576 merged tokens.
#: `min` matches the checkpoint's own preprocessor_config.json shortest_edge.
DEFAULT_MAX_PIXELS = 768 * 768
DEFAULT_MIN_PIXELS = 256 * 256


class ConditioningSpec(object):
    """Everything that changes the value of a conditioning embedding.

    `fingerprint()` covers exactly those fields, so a cache entry can never be
    reused across a semantic change (a different pooling, a different level, a
    different image budget). Fields that only affect *training* (the fused
    projection width) are excluded on purpose.
    """

    #: fields that change the embedding value -> part of the cache key
    _VALUE_FIELDS = ("modality", "pool", "vision_level", "normalize",
                     "max_pixels", "min_pixels", "model",
                     "structure_log1p", "structure_norm", "structure_dim")

    def __init__(self, modality="vision", pool="mean", vision_level="merged",
                 normalize=True, augment=False, aug_strength=1.0,
                 max_pixels=DEFAULT_MAX_PIXELS, min_pixels=DEFAULT_MIN_PIXELS,
                 model=None, d_cond=None, fuse_hidden=None,
                 structure_dim=None, structure_log1p=True, structure_norm="l2",
                 cache_dir=None):
        self.modality = modality
        self.pool = pool
        self.vision_level = vision_level
        self.normalize = bool(normalize)
        self.augment = bool(augment)
        self.aug_strength = float(aug_strength)
        self.max_pixels = int(max_pixels) if max_pixels else None
        self.min_pixels = int(min_pixels) if min_pixels else None
        self.model = model or paths.BASE_MODEL
        self.d_cond = d_cond                 # None == "whatever the parts give"
        self.fuse_hidden = fuse_hidden       # None == single linear projection
        self.structure_dim = structure_dim   # len(capture.ROLE_VOCAB)
        self.structure_log1p = bool(structure_log1p)
        self.structure_norm = structure_norm
        self.cache_dir = cache_dir
        self.validate()

    def validate(self):
        if self.modality not in MODALITIES:
            raise ValueError("modality %r not in %r" % (self.modality, MODALITIES))
        if self.pool not in POOLINGS:
            raise ValueError("pool %r not in %r" % (self.pool, POOLINGS))
        if self.vision_level not in VISION_LEVELS:
            raise ValueError("vision_level %r not in %r"
                             % (self.vision_level, VISION_LEVELS))
        if self.structure_norm not in STRUCT_NORMS:
            raise ValueError("structure_norm %r not in %r"
                             % (self.structure_norm, STRUCT_NORMS))
        if self.modality in ("structure", "both") and not self.structure_dim:
            # not fatal at construction -- capture.py may fill it in later --
            # but every consumer needs it, so say so early.
            pass
        return self

    @property
    def uses_vision(self):
        return self.modality in ("vision", "both")

    @property
    def uses_structure(self):
        return self.modality in ("structure", "both")

    def pool_factor(self):
        return 2 if self.pool == "mean_max" else 1

    def vision_dim(self, tower_dims):
        """tower_dims: (patch_hidden, merged_hidden) e.g. (1152, 4096)."""
        base = tower_dims[1] if self.vision_level == "merged" else tower_dims[0]
        return base * self.pool_factor()

    def split_dims(self, tower_dims=None):
        """(d_vision, d_structure) of a stored embedding under this spec."""
        d_v = self.vision_dim(tower_dims) if (self.uses_vision and tower_dims) else 0
        d_s = int(self.structure_dim or 0) if self.uses_structure else 0
        return d_v, d_s

    def as_dict(self):
        return dict((k, getattr(self, k)) for k in (
            "modality", "pool", "vision_level", "normalize", "augment",
            "aug_strength", "max_pixels", "min_pixels", "model", "d_cond",
            "fuse_hidden", "structure_dim", "structure_log1p", "structure_norm",
            "cache_dir"))

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d.pop("_fingerprint", None)
        return ConditioningSpec(**d)

    def fingerprint(self):
        payload = json.dumps(dict((k, getattr(self, k)) for k in self._VALUE_FIELDS),
                             sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def __repr__(self):
        return ("ConditioningSpec(%s, pool=%s, level=%s, norm=%s, aug=%s, fp=%s)"
                % (self.modality, self.pool, self.vision_level, self.normalize,
                   self.augment, self.fingerprint()))


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------

def _pil():
    from PIL import Image
    return Image


def load_image(img):
    """PIL image from a path, bytes, or an already-open image. Always RGB."""
    Image = _pil()
    if hasattr(img, "convert") and hasattr(img, "size"):
        return img.convert("RGB")
    if isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(img)).convert("RGB")
    if not os.path.exists(img):
        raise IOError("no such image: %s" % (img,))
    with open(img, "rb") as fh:
        return Image.open(io.BytesIO(fh.read())).convert("RGB")


def content_hash(img):
    """Stable hash of what will actually be encoded.

    A path hashes its bytes (cheap); a PIL image hashes its pixels. Both are
    content hashes, so an *augmented* view never collides with its original --
    which is what keeps the 6.7 augmentation probe honest under caching.
    """
    h = hashlib.sha1()
    if not hasattr(img, "size") and not isinstance(img, (bytes, bytearray)):
        with open(img, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    if isinstance(img, (bytes, bytearray)):
        h.update(img)
        return h.hexdigest()
    h.update(("%s|%dx%d|" % (img.mode, img.size[0], img.size[1])).encode("utf-8"))
    h.update(img.tobytes())
    return h.hexdigest()


def fit_image(img, max_pixels=DEFAULT_MAX_PIXELS, min_pixels=DEFAULT_MIN_PIXELS,
              multiple=MERGED_PIXEL):
    """Resize into [min_pixels, max_pixels] on a `multiple` grid, aspect kept.

    Done here rather than through the processor's `size=` argument on purpose:
    the arg name/semantics moved between transformers versions, and 4.2 wants
    the capture protocol -- not a library default -- to fix the token count.
    """
    Image = _pil()
    w, h = img.size
    area = float(w * h)
    scale = 1.0
    if max_pixels and area > max_pixels:
        scale = (max_pixels / area) ** 0.5
    elif min_pixels and area < min_pixels:
        scale = (min_pixels / area) ** 0.5
    nw = max(multiple, int(round(w * scale / multiple)) * multiple)
    nh = max(multiple, int(round(h * scale / multiple)) * multiple)
    if (nw, nh) == (w, h):
        return img
    return img.resize((nw, nh), Image.BICUBIC)


def augment_image(img, rng, strength=1.0):
    """The pixel-space augmentation of 6.7's memorisation probe.

    Deterministic given `rng` (an int seed, `np.random.Generator`, or
    `np.random.RandomState`). Same size and mode out as in.

    What it does: a small area-preserving-ish crop (>=90 % of area) resized
    back, +-8 % brightness / contrast / saturation, a +-2/255 hue rotation, a
    JPEG round-trip at quality 60-95, and mild pixel noise.

    **What it deliberately does not do, and why.** The augmentation must destroy
    nuisance invariances (exact pixel values, compression artefacts, a few
    pixels of scroll offset) without changing what the interface *is*. So: no
    horizontal or vertical flip (mirrored layout is a different reading order
    and a different interface), no rotation, no grayscale or aggressive colour
    shift (palette is one of the strongest era cues -- destroying it destroys
    the conditioning signal 4.2 depends on, not just the shortcut), no aspect
    distortion (layout proportions are era-identifying), and no crop deep enough
    to remove the nav chrome. If a re-probe after augmentation still recovers
    version ID near-perfectly, that is a real result about the manifold; if
    augmentation had to break the style to beat the probe, the probe was
    measuring the style, not a lookup table.
    """
    from PIL import Image, ImageEnhance
    rng = _as_rng(rng)
    s = float(strength)
    img = img.convert("RGB")
    w, h = img.size

    # 1. crop >= 90% of the area, keep aspect, resize back
    keep = 1.0 - 0.10 * s * float(rng.random())
    keep = min(1.0, max(0.5, keep))
    cw, ch = max(1, int(w * keep ** 0.5)), max(1, int(h * keep ** 0.5))
    x0 = int(rng.integers(0, w - cw + 1)) if w > cw else 0
    y0 = int(rng.integers(0, h - ch + 1)) if h > ch else 0
    img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BICUBIC)

    # 2. small photometric jitter
    for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast,
                ImageEnhance.Color):
        f = 1.0 + 0.08 * s * float(rng.uniform(-1.0, 1.0))
        img = enh(img).enhance(f)

    # 3. tiny hue rotation (HSV round-trip; +-2/255 is invisible to a human)
    shift = int(round(2 * s * rng.uniform(-1.0, 1.0)))
    if shift:
        hue, sat, val = img.convert("HSV").split()
        hue = hue.point(lambda p, k=shift: (p + k) % 256)
        img = Image.merge("HSV", (hue, sat, val)).convert("RGB")

    # 4. JPEG round-trip
    q = int(95 - 35 * s * float(rng.random()))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=max(30, min(100, q)))
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    # 5. mild pixel noise
    if s > 0:
        arr = np.asarray(img, dtype=np.float32)
        arr = arr + rng.normal(0.0, 2.0 * s, size=arr.shape).astype(np.float32)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def _as_rng(rng):
    if rng is None:
        return np.random.default_rng(0)
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):
        # adapt the legacy API to the Generator names used above
        return _LegacyRng(rng)
    raise TypeError("rng must be int | np.random.Generator | np.random.RandomState")


class _LegacyRng(object):
    def __init__(self, rs):
        self._rs = rs

    def random(self, *a, **k):
        return self._rs.random_sample(*a, **k)

    def integers(self, low, high=None, **k):
        return self._rs.randint(low, high, **k)

    def uniform(self, *a, **k):
        return self._rs.uniform(*a, **k)

    def normal(self, *a, **k):
        return self._rs.normal(*a, **k)


# --------------------------------------------------------------------------
# Embedding cache
# --------------------------------------------------------------------------

class EmbeddingCache(object):
    """Content-addressed on-disk cache of frozen-tower embeddings.

    A sweep re-runs the same conditioning for every fold, seed and generator
    kind (4.3 freezes it per episode), so recomputation is pure waste. Keys are
    `<content hash>-<spec fingerprint>`; nothing about the *model* under
    training enters the key, because nothing about it enters the embedding.
    """

    def __init__(self, root=None, enabled=True):
        self.root = root
        self.enabled = bool(enabled and root)
        self.mem = {}
        self.hits = 0
        self.misses = 0
        if self.enabled and not os.path.isdir(self.root):
            os.makedirs(self.root)

    def key(self, img, spec):
        return "%s-%s" % (content_hash(img), spec.fingerprint())

    def _path(self, key):
        return os.path.join(self.root, key + ".npy")

    def get(self, key):
        if key in self.mem:
            self.hits += 1
            return self.mem[key]
        if self.enabled:
            p = self._path(key)
            if os.path.exists(p):
                v = np.load(p)
                self.mem[key] = v
                self.hits += 1
                return v
        self.misses += 1
        return None

    def put(self, key, value):
        value = np.asarray(value, dtype=np.float32)
        self.mem[key] = value
        if self.enabled:
            # np.save appends '.npy' unless the name already ends in it, so the
            # temp name has to end in '.npy' for the rename to be atomic.
            tmp = os.path.join(self.root, ".tmp%d-%s.npy" % (os.getpid(), key))
            np.save(tmp, value)
            os.rename(tmp, self._path(key))
        return value

    def stats(self):
        return {"hits": self.hits, "misses": self.misses,
                "n": len(self.mem), "root": self.root}


# --------------------------------------------------------------------------
# Vision
# --------------------------------------------------------------------------

VISION_KEY_PREFIX = "model.visual."


def load_vision_tower(model_dir=None, dtype=torch.float32, device="cpu",
                      verbose=False):
    """Instantiate `Qwen3_5VisionModel` and load ONLY its weights.

    Why this exists: `Qwen3_5VisionModel.from_pretrained(<9B dir>)` does not
    work -- the checkpoint stores the tower under `model.visual.*` (it is a
    submodule of `Qwen3_5Model`, modeling_qwen3_5.py:1316) and there is no
    public API in transformers 5.6.0 to load a nested sub-model in isolation.
    So we read `model.safetensors.index.json`, keep the `model.visual.*` keys,
    open only the shards they live in, and `load_state_dict` into a fresh tower.
    For Qwen3.5-9B that is 333 tensors in a single 3.1 GB shard.

    Returns (tower, vision_config).
    """
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    model_dir = paths.model_dir_or_id(model_dir)
    if not os.path.isdir(model_dir):
        raise IOError(
            "vision tower needs a local snapshot dir, got %r. Cache the model "
            "first (paths.hf_snapshot) or pass from_full_model()." % (model_dir,))
    cfg = AutoConfig.from_pretrained(model_dir)
    vcfg = getattr(cfg, "vision_config", None)
    if vcfg is None:
        raise ValueError("%s has no vision_config -- not a VL checkpoint" % model_dir)

    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as fh:
            weight_map = json.load(fh)["weight_map"]
        wanted = dict((k, v) for k, v in weight_map.items()
                      if k.startswith(VISION_KEY_PREFIX))
    else:
        single = None
        for cand in ("model.safetensors",):
            if os.path.exists(os.path.join(model_dir, cand)):
                single = cand
        if single is None:
            raise IOError("no safetensors index and no model.safetensors in %s"
                          % model_dir)
        with safe_open(os.path.join(model_dir, single), framework="pt") as f:
            wanted = dict((k, single) for k in f.keys()
                          if k.startswith(VISION_KEY_PREFIX))
    if not wanted:
        raise IOError("no %s* tensors in %s" % (VISION_KEY_PREFIX, model_dir))

    state = {}
    by_shard = {}
    for k, shard in wanted.items():
        by_shard.setdefault(shard, []).append(k)
    for shard, keys in sorted(by_shard.items()):
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            for k in keys:
                state[k[len(VISION_KEY_PREFIX):]] = f.get_tensor(k).to(dtype)
    if verbose:
        print("[encoder] %d vision tensors from %d shard(s)"
              % (len(state), len(by_shard)))

    try:
        from transformers.modeling_utils import no_init_weights
        with no_init_weights():
            tower = Qwen3_5VisionModel._from_config(vcfg)
    except Exception:
        tower = Qwen3_5VisionModel._from_config(vcfg)
    missing, unexpected = tower.load_state_dict(state, strict=False)
    missing = [m for m in missing if not m.endswith("inv_freq")]
    if missing or unexpected:
        raise RuntimeError(
            "vision tower state mismatch: %d missing (e.g. %s), %d unexpected "
            "(e.g. %s). The checkpoint layout changed; fix VISION_KEY_PREFIX."
            % (len(missing), missing[:3], len(unexpected), unexpected[:3]))
    tower = tower.to(device=device, dtype=dtype).eval()
    for p in tower.parameters():
        p.requires_grad_(False)
    return tower, vcfg


def load_image_processor(model_dir=None):
    """The checkpoint's own image processor (patchifier + normaliser)."""
    from transformers import AutoImageProcessor
    model_dir = paths.model_dir_or_id(model_dir)
    try:
        return AutoImageProcessor.from_pretrained(model_dir)
    except Exception:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(model_dir).image_processor


class VisionConditioner(nn.Module):
    """Frozen Qwen3.5 vision tower -> one vector per screenshot (4.2).

    Frozen by construction: `requires_grad_(False)` on every parameter and every
    forward under `torch.no_grad()`. It costs no trainable parameters, which is
    the whole reason 4.2 prefers the model's own tower to a separate ViT.
    """

    def __init__(self, tower, image_processor=None, spec=None, device="cpu",
                 dtype=torch.float32, cache=None):
        super(VisionConditioner, self).__init__()
        self.spec = spec or ConditioningSpec()
        self.tower = tower.eval()
        for p in self.tower.parameters():
            p.requires_grad_(False)
        self.processor = image_processor
        self.device = torch.device(device)
        self.dtype = dtype
        self.cache = cache or EmbeddingCache(self.spec.cache_dir)
        vcfg = tower.config
        self.merge = int(getattr(vcfg, "spatial_merge_size", 2))
        self.patch = int(getattr(vcfg, "patch_size", 16))
        self.temporal_patch = int(getattr(vcfg, "temporal_patch_size", 2))
        self.in_channels = int(getattr(vcfg, "in_channels", 3))
        self.tower_dims = (int(vcfg.hidden_size), int(vcfg.out_hidden_size))

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_dir=None, spec=None, device="cpu",
                        dtype=torch.float32, cache=None, verbose=False):
        """Tower-only load -- ~0.4 B params instead of the full 9 B."""
        spec = spec or ConditioningSpec()
        tower, _ = load_vision_tower(model_dir or spec.model, dtype=dtype,
                                     device=device, verbose=verbose)
        proc = load_image_processor(model_dir or spec.model)
        return cls(tower, proc, spec=spec, device=device, dtype=dtype, cache=cache)

    @classmethod
    def from_full_model(cls, model, spec=None, image_processor=None, cache=None):
        """Reuse an already-loaded `Qwen3_5ForConditionalGeneration`'s tower.

        Use this inside the training loop: the LM is resident anyway, and this
        guarantees the conditioning comes from exactly the tower the policy
        uses. `model.model.visual` is the documented path
        (modeling_qwen3_5.py:1316).
        """
        tower = None
        for attr in ("model.visual", "visual", "model.model.visual"):
            obj = model
            ok = True
            for part in attr.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok:
                tower = obj
                break
        if tower is None:
            raise AttributeError(
                "no vision tower on %s (looked for .model.visual). This model is "
                "text-only; use modality='structure'." % type(model).__name__)
        spec = spec or ConditioningSpec()
        proc = image_processor or load_image_processor(spec.model)
        dev = next(tower.parameters()).device
        dt = next(tower.parameters()).dtype
        return cls(tower, proc, spec=spec, device=dev, dtype=dt, cache=cache)

    # -- dims --------------------------------------------------------------
    @property
    def d_out(self):
        return self.spec.vision_dim(self.tower_dims)

    # -- encoding ----------------------------------------------------------
    def _preprocess(self, images):
        """PIL images -> (pixel_values, grid_thw) exactly as the LM would."""
        if self.processor is None:
            raise RuntimeError(
                "VisionConditioner has no image processor; pass one or use "
                "encode_pixels() with pre-patchified input")
        imgs = [fit_image(load_image(im), self.spec.max_pixels,
                          self.spec.min_pixels, self.patch * self.merge)
                for im in images]
        out = self.processor(images=imgs, return_tensors="pt")
        return out["pixel_values"], out["image_grid_thw"]

    def _pool(self, tokens):
        """(n_tokens, d) -> (d,) or (2d,) for mean_max."""
        p = self.spec.pool
        if p == "mean":
            return tokens.mean(0)
        if p == "max":
            return tokens.max(0).values
        if p == "mean_max":
            return torch.cat([tokens.mean(0), tokens.max(0).values], dim=-1)
        if p == "first":
            # documented as a trap: this tower has no CLS, so token 0 is just
            # the top-left patch. Kept only so the ablation can show it is bad.
            return tokens[0]
        raise ValueError("unknown pooling %r" % (p,))

    @torch.no_grad()
    def encode_pixels(self, pixel_values, grid_thw):
        """The tower forward + pooling, for pre-patchified input.

        Splitting per image mirrors `get_image_features`
        (modeling_qwen3_5.py:1517): merged tokens per image are
        `prod(grid) / merge**2`, patch tokens are `prod(grid)`.
        """
        pixel_values = pixel_values.to(self.device, self.dtype)
        grid_thw = grid_thw.to(self.device)
        out = self.tower(pixel_values, grid_thw=grid_thw)
        if self.spec.vision_level == "merged":
            stream = out.pooler_output
            sizes = (grid_thw.prod(-1) // (self.merge ** 2)).tolist()
        else:
            stream = out.last_hidden_state
            sizes = grid_thw.prod(-1).tolist()
        chunks = torch.split(stream, sizes)
        vecs = [self._pool(c.float()) for c in chunks]
        v = torch.stack(vecs, dim=0)
        if self.spec.normalize:
            v = torch.nn.functional.normalize(v, dim=-1)
        return v.cpu()

    def encode(self, images, augment=None, rng=None, use_cache=True):
        """PIL images / paths -> (N, d_out) float32 on CPU."""
        if not isinstance(images, (list, tuple)):
            images = [images]
        augment = self.spec.augment if augment is None else augment
        prepared, keys = [], []
        for i, im in enumerate(images):
            if augment:
                sub = _as_rng(rng if rng is not None else i)
                im = augment_image(load_image(im), sub, self.spec.aug_strength)
            prepared.append(im)
            keys.append(self.cache.key(im, self.spec) if use_cache else None)

        out = [None] * len(prepared)
        todo = []
        for i, k in enumerate(keys):
            hit = self.cache.get(k) if k else None
            if hit is not None:
                out[i] = torch.from_numpy(np.asarray(hit, dtype=np.float32))
            else:
                todo.append(i)
        if todo:
            px, grid = self._preprocess([prepared[i] for i in todo])
            vecs = self.encode_pixels(px, grid)
            for j, i in enumerate(todo):
                out[i] = vecs[j]
                if keys[i]:
                    self.cache.put(keys[i], vecs[j].numpy())
        return torch.stack(out, dim=0)

    def encode_one(self, image, **kw):
        return self.encode([image], **kw)[0]

    def forward(self, images, **kw):
        return self.encode(images, **kw)


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------

def role_vocab(strict=False):
    """`capture.ROLE_VOCAB` if capture.py is present, else None.

    Imported lazily and defensively: capture.py is written by a different agent
    and this module must remain usable (and testable) without it. The primary
    input to `StructureConditioner` is the raw count vector, so the vocabulary
    is only needed to accept dict input.
    """
    try:
        from . import capture
    except Exception:
        if strict:
            raise ImportError("adaptercl.capture is not importable; pass a raw "
                              "role-count vector instead of a dict")
        return None
    vocab = getattr(capture, "ROLE_VOCAB", None)
    if vocab is None and strict:
        raise AttributeError("adaptercl.capture has no ROLE_VOCAB")
    return vocab


class StructureConditioner(nn.Module):
    """4.2's "compact deterministic signature ... histogram over a11y roles".

    Input is the fixed-order count vector `capture.py` produces (a few hundred
    dims). Transform: `log1p` then L2 (or L1) normalise.

    * `log1p` because role counts are heavy-tailed -- a page with 400 `link`s
      and one `search` is not 400x more "link-like" than a page with one of
      each, and a raw count vector would let the single largest role dominate
      the whole embedding.
    * normalisation because page length is a nuisance variable: the same
      interface with more content must not move in conditioning space.

    Explicitly NOT raw HTML/DOM text (4.2): every truncation knob -- max length,
    which attributes survive, whether it is pretty-printed -- becomes a confound
    you cannot report, and the resulting manifold is dominated by content rather
    than interface style.

    `d_cond` adds a trainable projection. That projection is part of the
    generator (it trains end-to-end through the task loss, 4.4), not part of the
    frozen encoder -- so it is *not* covered by the embedding cache.
    """

    def __init__(self, d_in=None, d_cond=None, spec=None, log1p=None,
                 normalize=None, vocab=None):
        super(StructureConditioner, self).__init__()
        self.spec = spec or ConditioningSpec(modality="structure",
                                             structure_dim=d_in)
        self.vocab = vocab if vocab is not None else role_vocab()
        d_in = d_in or self.spec.structure_dim or (len(self.vocab) if self.vocab else None)
        if not d_in:
            raise ValueError(
                "StructureConditioner needs d_in (len of the role vector). "
                "Pass d_in=, set spec.structure_dim, or make adaptercl.capture "
                "importable so ROLE_VOCAB can supply it.")
        self.d_in = int(d_in)
        self.log1p = self.spec.structure_log1p if log1p is None else bool(log1p)
        self.norm = self.spec.structure_norm if normalize is None else normalize
        self.proj = None
        if d_cond and int(d_cond) != self.d_in:
            self.proj = nn.Linear(self.d_in, int(d_cond))
        self._d_out = int(d_cond) if d_cond else self.d_in

    @property
    def d_out(self):
        return self._d_out

    def vector_from_counts(self, counts):
        """{role: count} -> fixed-order vector. Needs the vocabulary."""
        vocab = self.vocab or role_vocab(strict=True)
        v = np.zeros(len(vocab), dtype=np.float32)
        unknown = [k for k in counts if k not in vocab]
        if unknown:
            raise KeyError("roles outside ROLE_VOCAB: %r. Extend the vocabulary "
                           "in capture.py -- silently dropping roles changes the "
                           "conditioning signal." % unknown[:5])
        idx = dict((r, i) for i, r in enumerate(vocab))
        for k, n in counts.items():
            v[idx[k]] = float(n)
        return v

    def normalize_raw(self, roles):
        """The deterministic, parameter-free part. Accepts vectors or dicts."""
        if isinstance(roles, dict):
            roles = self.vector_from_counts(roles)
        x = torch.as_tensor(np.asarray(roles, dtype=np.float32)
                            if not torch.is_tensor(roles) else roles).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.d_in:
            raise ValueError("role vector has %d dims, expected %d"
                             % (x.shape[-1], self.d_in))
        if self.log1p:
            x = torch.log1p(x.clamp_min(0))
        if self.norm == "l2":
            x = torch.nn.functional.normalize(x, dim=-1)
        elif self.norm == "l1":
            x = x / x.abs().sum(-1, keepdim=True).clamp_min(1e-8)
        return x

    def forward(self, roles):
        x = self.normalize_raw(roles)
        if self.proj is not None:
            x = self.proj(x)
        return x

    def encode(self, roles):
        """Frozen part only -- what `embed_cells` caches."""
        with torch.no_grad():
            return self.normalize_raw(roles)


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------

class FusedConditioner(nn.Module):
    """vision (+) structure -> d_cond, with a small TRAINABLE projection.

    Unlike `VisionConditioner` this one is *not* frozen: it belongs to the
    generator and is optimised through the task loss (4.4). Keeping it separate
    from the cache is what makes `modality="both"` a one-flag change in 6.5's
    ablation rather than a re-extraction of every embedding.
    """

    def __init__(self, d_vision=0, d_structure=0, d_cond=None, hidden=None,
                 normalize=True, dropout=0.0):
        super(FusedConditioner, self).__init__()
        self.d_vision = int(d_vision)
        self.d_structure = int(d_structure)
        d_in = self.d_vision + self.d_structure
        if d_in <= 0:
            raise ValueError("FusedConditioner needs d_vision + d_structure > 0")
        self.d_in = d_in
        self._d_out = int(d_cond or d_in)
        self.normalize = bool(normalize)
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(d_in, int(hidden)), nn.SiLU(),
                nn.Dropout(dropout) if dropout else nn.Identity(),
                nn.Linear(int(hidden), self._d_out))
        else:
            self.net = nn.Linear(d_in, self._d_out)

    @property
    def d_out(self):
        return self._d_out

    def forward(self, vision=None, structure=None):
        parts = []
        if self.d_vision:
            if vision is None:
                raise ValueError("FusedConditioner expects a vision part")
            parts.append(_at_least_2d(vision))
        if self.d_structure:
            if structure is None:
                raise ValueError("FusedConditioner expects a structure part")
            parts.append(_at_least_2d(structure))
        x = torch.cat(parts, dim=-1)
        return self.forward_concat(x)

    def forward_concat(self, x):
        """For cached `[vision ; structure]` vectors from `embed_cells`."""
        x = _at_least_2d(x)
        if x.shape[-1] != self.d_in:
            raise ValueError("fused input has %d dims, expected %d (=%d vision + "
                             "%d structure)" % (x.shape[-1], self.d_in,
                                                self.d_vision, self.d_structure))
        y = self.net(x)
        if self.normalize:
            y = torch.nn.functional.normalize(y, dim=-1)
        return y


def _at_least_2d(x):
    x = torch.as_tensor(x).float() if not torch.is_tensor(x) else x.float()
    return x.unsqueeze(0) if x.dim() == 1 else x


# --------------------------------------------------------------------------
# Cell-level embedding bank
# --------------------------------------------------------------------------

def embed_cells(cells, image_paths=None, spec=None, conditioner=None,
                structure_vectors=None, cache=None, augment=None, seed=0,
                structure=None):
    """{cell.key: embedding} for a set of cells (4.3: one conditioning per cell).

    `image_paths` maps cell key (or Cell) -> a path, an image, or a list of
    either; a list is averaged, which is how you use several captures of the
    same interface without giving the generator per-episode noise to memorise.
    `structure_vectors` maps the same keys -> role count vectors.

    For `modality="both"` the returned vector is `[vision ; structure]`; the
    trainable fusion happens later in `FusedConditioner` (see module docstring).
    """
    spec = spec or (conditioner.spec if conditioner is not None
                    else ConditioningSpec())
    keys = [c.key if hasattr(c, "key") else str(c) for c in cells]

    def _lookup(table, key, cell):
        if table is None:
            return None
        if key in table:
            return table[key]
        if cell in table:
            return table[cell]
        raise KeyError("no entry for cell %r" % (key,))

    if spec.uses_vision and conditioner is None and image_paths is not None:
        conditioner = VisionConditioner.from_pretrained(spec.model, spec=spec,
                                                        cache=cache)
    if spec.uses_structure and structure is None:
        structure = StructureConditioner(d_in=spec.structure_dim, spec=spec)

    out = {}
    for key, cell in zip(keys, cells):
        parts = []
        if spec.uses_vision:
            imgs = _lookup(image_paths, key, cell)
            if imgs is None:
                raise KeyError("modality %r needs an image for cell %s"
                               % (spec.modality, key))
            if not isinstance(imgs, (list, tuple)):
                imgs = [imgs]
            rng = None if not (augment if augment is not None else spec.augment) \
                else _as_rng(hash((seed, key)) % (2 ** 31))
            v = conditioner.encode(imgs, augment=augment, rng=rng)
            v = v.mean(0)
            if spec.normalize:
                v = torch.nn.functional.normalize(v, dim=-1)
            parts.append(v.numpy())
        if spec.uses_structure:
            r = _lookup(structure_vectors, key, cell)
            if r is None:
                raise KeyError("modality %r needs a role vector for cell %s"
                               % (spec.modality, key))
            parts.append(structure.encode(r)[0].numpy())
        out[key] = np.concatenate(parts).astype(np.float32)
    return out


def save_embeddings(path, embeddings, spec=None):
    """Write {key: vector} (+ the spec) to a single .npz."""
    d = dict((k, np.asarray(v, dtype=np.float32)) for k, v in embeddings.items())
    meta = json.dumps(spec.as_dict() if spec is not None else {})
    d["__spec__"] = np.array([meta])
    d["__keys__"] = np.array(sorted(embeddings.keys()))
    if not path.endswith(".npz"):
        path = path + ".npz"
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    np.savez(path, **d)
    return path


def load_embeddings(path):
    """(embeddings, spec_dict). `spec_dict` is {} if none was stored."""
    if not path.endswith(".npz"):
        path = path + ".npz"
    z = np.load(path, allow_pickle=False)
    spec = {}
    if "__spec__" in z.files:
        spec = json.loads(str(z["__spec__"][0]))
    emb = dict((k, z[k]) for k in z.files if not k.startswith("__"))
    return emb, spec


def embedding_matrix(embeddings, keys=None):
    """(keys, (N, d) float32 array) in a stable order."""
    keys = list(keys or sorted(embeddings.keys()))
    X = np.stack([np.asarray(embeddings[k], dtype=np.float32) for k in keys], 0)
    return keys, X


# --------------------------------------------------------------------------
# Selftest
# --------------------------------------------------------------------------

def _fake_screenshot(h=240, w=320, seed=0):
    """A crude web page: header band, nav column, text lines on a light body.

    White noise is the wrong test image for augmentation -- every crop changes
    every pixel, so "the perturbation is small" is unmeasurable on it. Real
    screenshots are mostly flat regions, which is what these checks assume.
    """
    from PIL import Image
    rng = np.random.default_rng(seed)
    a = np.full((h, w, 3), 245, dtype=np.uint8)
    a[: h // 8, :, :] = np.array([40, 60, 140], dtype=np.uint8)      # header
    a[h // 8:, : w // 5, :] = np.array([225, 228, 235], dtype=np.uint8)  # nav
    for i in range(h // 8 + 6, h - 6, 12):                            # text lines
        wid = int(w * (0.45 + 0.3 * rng.random()))
        a[i:i + 4, w // 5 + 8: w // 5 + 8 + wid, :] = 70
    return Image.fromarray(a)


def _selftest(verbose=True):
    from PIL import Image
    ok = []

    def check(name, cond, extra=""):
        ok.append(bool(cond))
        print("  %-46s %s %s" % (name, "PASS" if cond else "FAIL", extra))

    print("[1] ConditioningSpec")
    s1 = ConditioningSpec(modality="vision", pool="mean")
    s2 = ConditioningSpec(modality="vision", pool="mean_max")
    check("fingerprint stable", s1.fingerprint() == ConditioningSpec(**s1.as_dict()).fingerprint())
    check("fingerprint separates pooling", s1.fingerprint() != s2.fingerprint())
    check("round-trip as_dict/from_dict",
          ConditioningSpec.from_dict(s1.as_dict()).as_dict() == s1.as_dict())
    check("vision_dim(mean_max) doubles", s2.vision_dim((1152, 4096)) == 8192)
    try:
        ConditioningSpec(modality="dom")
        check("bad modality rejected", False)
    except ValueError:
        check("bad modality rejected", True)

    print("[2] augment_image")
    img = _fake_screenshot(240, 320)
    a1 = augment_image(img, 7)
    a2 = augment_image(img, 7)
    a3 = augment_image(img, 8)
    check("deterministic given seed", np.array_equal(np.asarray(a1), np.asarray(a2)))
    check("different seed differs", not np.array_equal(np.asarray(a1), np.asarray(a3)))
    check("size/mode preserved", a1.size == img.size and a1.mode == img.mode)
    # Identity-relevant structure must survive (see augment_image's docstring).
    # Pixel-exact difference is the wrong measure -- a 5 % crop moves every text
    # row -- so check the coarse layout and the palette, which are what carries
    # era identity.
    def _coarse(im):
        return np.asarray(im.resize((20, 15), Image.BOX), np.float32).ravel()
    cos = float(_coarse(a1) @ _coarse(img)
                / (np.linalg.norm(_coarse(a1)) * np.linalg.norm(_coarse(img))))
    worst = min(float(_coarse(augment_image(img, s)) @ _coarse(img)
                      / (np.linalg.norm(_coarse(augment_image(img, s)))
                         * np.linalg.norm(_coarse(img)))) for s in range(6))
    check("coarse layout preserved (cos > 0.98)", worst > 0.98, "worst %.4f" % worst)
    pal0 = np.asarray(img, np.float32).reshape(-1, 3).mean(0)
    pal1 = np.asarray(a1, np.float32).reshape(-1, 3).mean(0)
    check("palette (era cue) preserved to 10%",
          float(np.abs(pal1 - pal0).max()) < 25.5,
          "max |dmean|=%.1f" % float(np.abs(pal1 - pal0).max()))
    check("pixels actually changed (probe is not a no-op)",
          np.abs(np.asarray(a1, np.float32) - np.asarray(img, np.float32)).mean() > 1.0)
    check("content_hash separates original/augmented",
          content_hash(img) != content_hash(a1))

    print("[3] fit_image")
    big = Image.new("RGB", (1920, 1080))
    f = fit_image(big, 768 * 768, 256 * 256)
    check("fits max_pixels", f.size[0] * f.size[1] <= 768 * 768, str(f.size))
    check("multiple of 32", f.size[0] % 32 == 0 and f.size[1] % 32 == 0)
    check("aspect preserved to 2%",
          abs((f.size[0] / float(f.size[1])) / (1920 / 1080.0) - 1) < 0.02)

    print("[4] StructureConditioner")
    sc = StructureConditioner(d_in=8)
    v = sc.encode(np.array([0, 1, 400, 3, 0, 0, 2, 1], dtype=np.float32))
    check("shape (1, d_in)", tuple(v.shape) == (1, 8), str(tuple(v.shape)))
    check("l2 normalised", abs(float(v.norm()) - 1.0) < 1e-5)
    check("log1p tames the tail", float(v[0, 2]) < 0.95,
          "%.3f (raw counts would give ~0.9999)" % float(v[0, 2]))
    raw = torch.nn.functional.normalize(
        torch.tensor([[0., 1., 400., 3., 0., 0., 2., 1.]]), dim=-1)
    check("... vs raw-count baseline", float(raw[0, 2]) > 0.999, "%.4f" % float(raw[0, 2]))
    sc2 = StructureConditioner(d_in=8, d_cond=16)
    y = sc2(np.zeros(8, dtype=np.float32))
    y.sum().backward()
    g = sum(float(p.grad.abs().sum()) for p in sc2.parameters() if p.grad is not None)
    check("projection is trainable", tuple(y.shape) == (1, 16) and g > 0)
    try:
        sc.encode(np.zeros(9, dtype=np.float32))
        check("wrong-width vector rejected", False)
    except ValueError:
        check("wrong-width vector rejected", True)

    print("[5] FusedConditioner")
    fc = FusedConditioner(d_vision=6, d_structure=4, d_cond=5, hidden=8)
    out = fc(torch.randn(3, 6), torch.randn(3, 4))
    check("shape (B, d_cond)", tuple(out.shape) == (3, 5), str(tuple(out.shape)))
    check("L2 normalised rows", abs(float(out.detach().norm(dim=-1).mean()) - 1.0) < 1e-5)
    out.sum().backward()
    n_grad = sum(1 for p in fc.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    check("gradients reach the projection", n_grad >= 2, "%d tensors" % n_grad)
    cat = torch.cat([torch.randn(2, 6), torch.randn(2, 4)], -1)
    check("forward_concat matches split call",
          torch.allclose(fc.forward_concat(cat), fc(cat[:, :6], cat[:, 6:])))

    print("[6] VisionConditioner on a tiny random tower")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    vc = Qwen3_5VisionConfig(depth=2, hidden_size=32, num_heads=2,
                             intermediate_size=64, out_hidden_size=16,
                             num_position_embeddings=64, patch_size=16,
                             spatial_merge_size=2, temporal_patch_size=2,
                             in_channels=3)
    torch.manual_seed(0)
    tower = Qwen3_5VisionModel._from_config(vc).eval()
    spec = ConditioningSpec(pool="mean", vision_level="merged")
    vcond = VisionConditioner(tower, None, spec=spec)
    check("d_out == out_hidden_size", vcond.d_out == 16, str(vcond.d_out))
    grid = torch.tensor([[1, 4, 6], [1, 2, 2]])
    px = torch.randn(int((grid[:, 1] * grid[:, 2]).sum()), 3 * 2 * 16 * 16)
    emb = vcond.encode_pixels(px, grid)
    check("encode_pixels -> (n_img, d)", tuple(emb.shape) == (2, 16), str(tuple(emb.shape)))
    check("rows L2-normalised", torch.allclose(emb.norm(dim=-1), torch.ones(2), atol=1e-5))
    check("tower is frozen",
          all(not p.requires_grad for p in vcond.tower.parameters()))
    vcond_p = VisionConditioner(tower, None,
                                spec=ConditioningSpec(vision_level="patch",
                                                      pool="mean_max"))
    check("patch level + mean_max -> 2*1152-equivalent",
          vcond_p.d_out == 64 and tuple(vcond_p.encode_pixels(px, grid).shape) == (2, 64))

    print("[7] real image processor + tiny tower (skipped if not cached)")
    snap = paths.hf_snapshot(paths.BASE_MODEL)
    if snap is None:
        print("  %-46s SKIP (no local snapshot)" % "processor round-trip")
    else:
        proc = load_image_processor(snap)
        tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".selftest_cache")
        cache = EmbeddingCache(tmp)
        vcond2 = VisionConditioner(tower, proc, spec=spec, cache=cache)
        img2 = _fake_screenshot(240, 320, seed=1)
        e1 = vcond2.encode([img2])
        e2 = vcond2.encode([img2])
        check("processor -> tower -> (1, d)", tuple(e1.shape) == (1, 16), str(tuple(e1.shape)))
        check("cache hit on second call", cache.stats()["hits"] == 1, str(cache.stats()))
        check("cached value identical", torch.allclose(e1, e2))
        cells = ["wiki_e1", "wiki_e2"]
        emb = embed_cells(cells, {"wiki_e1": img2, "wiki_e2": [img2, img2]},
                          spec=spec, conditioner=vcond2)
        check("embed_cells keys", sorted(emb.keys()) == cells)
        check("embed_cells dims", emb["wiki_e1"].shape == (16,))
        p = save_embeddings(os.path.join(tmp, "emb"), emb, spec)
        back, sp = load_embeddings(p)
        check("npz round-trip", np.allclose(back["wiki_e1"], emb["wiki_e1"])
              and sp["pool"] == "mean")
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("[8] structure-only embed_cells")
    sv = {"news_e1": np.array([1., 0., 3., 0., 0., 0., 9., 2.]),
          "news_e2": np.array([0., 5., 1., 0., 1., 0., 0., 0.])}
    spec_s = ConditioningSpec(modality="structure", structure_dim=8)
    es = embed_cells(["news_e1", "news_e2"], structure_vectors=sv, spec=spec_s)
    check("structure-only shapes", es["news_e1"].shape == (8,))
    keys, X = embedding_matrix(es)
    check("embedding_matrix", X.shape == (2, 8) and keys == ["news_e1", "news_e2"])

    n_fail = len([x for x in ok if not x])
    print("\n%d/%d checks passed" % (len(ok) - n_fail, len(ok)))
    return n_fail


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="conditioning encoder (adapterCL 4.2)")
    ap.add_argument("--selftest", action="store_true", help="CPU-only checks")
    ap.add_argument("--tower-info", action="store_true",
                    help="load the real vision tower and report its dims")
    args = ap.parse_args()
    if args.tower_info:
        tower, vcfg = load_vision_tower(verbose=True)
        n = sum(p.numel() for p in tower.parameters())
        print("vision tower: depth=%d hidden=%d out_hidden=%d params=%s"
              % (vcfg.depth, vcfg.hidden_size, vcfg.out_hidden_size, "{:,}".format(n)))
        raise SystemExit(0)
    if args.selftest:
        raise SystemExit(1 if _selftest() else 0)
    ap.print_help()
