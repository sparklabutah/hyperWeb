#!/usr/bin/env python3
"""Compute the Phase-3 conditioning embeddings (4.2) for the six versions.

The hypernetwork conditions on what the interface LOOKS LIKE. `capture.py`
already produced a deterministic screenshot per (site, era) cell; this encodes
each through the frozen Qwen3.5 vision tower and writes one vector per VERSION.

Version, not cell, because the adapter unit is the version (2): a version's
conditioning is the mean of its three sites' embeddings, which is the visual
centroid of "what era 3 looks like" across wiki / news / shop.

That averaging is a real modelling choice and worth stating plainly: it assumes
the three sites at one era share a visual style. `era_style.py` measured that
and found it holds only after the environment main effect is removed
(env-centred contrast +0.46, p=0.002; raw contrast is negative). So the mean is
defensible but not free -- `--per-cell` writes the 18 un-averaged vectors
instead, which is the honest fallback if the averaged conditioning turns out to
carry less signal.

Run under the llamafactory python on a GPU:
  srun ... python scripts/build_conditioning.py --out out/hypernet/cond_ver.npz
"""
from __future__ import print_function

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import numpy as np  # noqa: E402

from adaptercl import cells, encoder, paths  # noqa: E402


def cell_image(cell_key):
    """The landing screenshot capture.py wrote for a cell, or None."""
    p = os.path.join(paths.OUT_CAPTURE, cell_key, "landing", "screenshot.png")
    return p if os.path.exists(p) else None


def cell_roles(cell_key, label="landing"):
    """The a11y-role COUNT vector capture.py wrote for a cell, or None.

    `roles.json` stores both the aria-snapshot counts and a DOM fallback; we
    take `vector`, the aria-snapshot one, because that is the channel
    `capture.py` treats as authoritative and the one `role_vector_sha256` pins.
    """
    p = os.path.join(paths.OUT_CAPTURE, cell_key, label, "roles.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        d = json.load(fh)
    return d.get("vector")


def text_embeddings(keys, device="cpu", descriptions=None):
    """{version key: frozen bge embedding of its description} (T2L's channel).

    Same encoder and the same `t2l.VERSION_DESCRIPTIONS` the reconstruction
    runs used, so a BC-trained generator on text conditioning is comparable to
    the T2L arm rather than to a new conditioner.

    `descriptions` overrides the built-in table with a JSON file in
    `t2l.load_descriptions` format -- which is how a SCOUT-written manual gets
    into the same channel as the hand-written descriptions, with the encoder,
    the pooling and the normalisation all unchanged (scout_plan.md Part E #5).
    """
    from adaptercl import t2l
    desc = t2l.load_descriptions(descriptions)
    missing = [k for k in keys if k not in desc]
    if missing:
        raise SystemExit("no description for %s" % missing)
    tc = t2l.TextConditioner(device=device)
    emb = tc.encode([desc[k] for k in keys])
    return dict((k, emb[i].numpy()) for i, k in enumerate(keys))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/hypernet/cond_ver.npz")
    ap.add_argument("--per-cell", action="store_true",
                    help="write the 18 cell vectors instead of 6 version means")
    ap.add_argument("--modality", default="vision",
                    choices=["vision", "structure", "both", "text"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--descriptions", default=None, metavar="PATH",
                    help="--modality text only: JSON {version key: description} "
                         "overriding t2l.VERSION_DESCRIPTIONS (e.g. "
                         "out/scout/crawl/descriptions_full.json).")
    ap.add_argument("--report-json", default=None, metavar="PATH",
                    help="also write the separability report as JSON (G4).")
    args = ap.parse_args()
    if args.descriptions and args.modality != "text":
        raise SystemExit("--descriptions only means anything for "
                         "--modality text (got %r)" % args.modality)

    # `text` is not a cell-level modality: T2L's descriptions are written per
    # VERSION, so there is nothing to average and --per-cell is meaningless.
    if args.modality == "text":
        keys = [v.key for v in cells.ALL_VERSIONS]
        out = text_embeddings(keys, device=args.device,
                              descriptions=args.descriptions)
        if args.descriptions:
            print("descriptions: %s" % args.descriptions)
        report_separability(out, keys, json_path=args.report_json)
        encoder.save_embeddings(args.out, out, spec=None)
        print("\nwrote %s  (%d vectors, dim %d)"
              % (args.out, len(out), len(next(iter(out.values())))))
        return

    spec = encoder.ConditioningSpec(modality=args.modality)

    # -- which captures do we have? ----------------------------------------
    # 11: the policy reads the accessibility tree and never a pixel, so the
    # structure modality is the only one conditioned on a channel the agent
    # itself consumes. It also needs no GPU -- the transform is log1p + L2 on
    # a role histogram capture.py already wrote.
    paths_by_cell, roles_by_cell, missing = {}, {}, []
    for c in cells.ALL_CELLS:
        need_img = spec.uses_vision
        need_roles = spec.uses_structure
        img = cell_image(c.key) if need_img else None
        rol = cell_roles(c.key) if need_roles else None
        if (need_img and img is None) or (need_roles and rol is None):
            missing.append(c.key)
            continue
        if img is not None:
            paths_by_cell[c.key] = img
        if rol is not None:
            roles_by_cell[c.key] = rol
    if missing:
        raise SystemExit(
            "no capture for %d cell(s): %s\nRun: CAPTURE=1 bash scripts/run_phase0.sh"
            % (len(missing), ", ".join(missing)))
    print("captures found for all %d cells" % len(cells.ALL_CELLS))
    if spec.uses_structure:
        spec.structure_dim = len(next(iter(roles_by_cell.values())))
        print("  role vector dim = %d (capture.ROLE_VOCAB)" % spec.structure_dim)

    cond = None
    if spec.uses_vision:
        print("encoding through the frozen Qwen3.5 vision tower (%s)..."
              % args.device)
        # Tower-only load: ~0.4B params, not the full 9B. The tower is frozen,
        # so the conditioning costs no trainable parameters (4.2).
        import torch
        cond = encoder.VisionConditioner.from_pretrained(
            paths.model_dir_or_id(), spec=spec, device=args.device,
            dtype=torch.bfloat16 if args.device != "cpu" else torch.float32,
            verbose=True)
    cell_emb = encoder.embed_cells(list(cells.ALL_CELLS),
                                   image_paths=paths_by_cell or None,
                                   structure_vectors=roles_by_cell or None,
                                   spec=spec, conditioner=cond)
    dim = len(next(iter(cell_emb.values())))
    print("  %d cell embeddings, dim=%d" % (len(cell_emb), dim))

    if args.per_cell:
        out = cell_emb
    else:
        # version embedding = mean of its three sites, then L2-normalised so
        # the generator sees vectors of comparable scale regardless of how many
        # sites contributed.
        out = {}
        for v in cells.ALL_VERSIONS:
            vecs = [np.asarray(cell_emb[c.key], dtype=np.float64)
                    for c in v.cells if c.key in cell_emb]
            m = np.mean(vecs, axis=0)
            n = np.linalg.norm(m)
            out[v.key] = (m / n if n > 0 else m).astype(np.float32)
        print("  averaged to %d version embeddings" % len(out))

        report_separability(out, [v.key for v in cells.ALL_VERSIONS])

    encoder.save_embeddings(args.out, out, spec=spec)
    print("\nwrote %s  (%d vectors, dim %d)" % (args.out, len(out), dim))


def report_separability(out, keys, json_path=None):
    """How distinguishable are the six versions?

    If this is ~1.0 everywhere the conditioning cannot separate them and the
    hypernetwork is dead on arrival -- better to know now than after a training
    run. Vision sits at 0.983, text at 0.861 (11).

    `json_path` also writes the numbers machine-readably; scout_plan.md's G4
    wants them recorded next to the existing text and vision figures rather than
    left in a terminal that has scrolled away.
    """
    keys = [k for k in keys if k in out]
    X = np.stack([np.asarray(out[k], dtype=np.float64) for k in keys])
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    C = X @ X.T
    off = C[~np.eye(len(keys), dtype=bool)]
    print("\n  pairwise cosine between version embeddings:")
    print("        " + "".join("%8s" % k for k in keys))
    for i, k in enumerate(keys):
        print("  %-6s" % k + "".join("%8.3f" % C[i, j] for j in range(len(keys))))
    print("\n  off-diagonal mean %.4f  min %.4f  max %.4f  (range %.3f)"
          % (off.mean(), off.min(), off.max(), off.max() - off.min()))
    if off.mean() > 0.98:
        print("  WARNING: the six conditionings are nearly identical -- a "
              "generator cannot condition on a signal that is not there.")
    rep = {"keys": list(keys), "off_diagonal_mean": float(off.mean()),
           "off_diagonal_min": float(off.min()),
           "off_diagonal_max": float(off.max()),
           "reference_vision": 0.983, "reference_text": 0.861,
           "cosine": [[float(C[i, j]) for j in range(len(keys))]
                      for i in range(len(keys))]}
    if json_path:
        d = os.path.dirname(os.path.abspath(json_path))
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(json_path, "w") as fh:
            json.dump(rep, fh, indent=2, sort_keys=True)
        print("  wrote %s" % json_path)
    return rep


if __name__ == "__main__":
    main()
