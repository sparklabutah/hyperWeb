#!/usr/bin/env python3
"""Cross-module contract tests for adaptercl -- the analysis layer (no GPU, no weights).

Every module in adaptercl/ has its own `--selftest`; those are unit-level and they
pass. This file tests the *seams*: the places where one module's output becomes
another module's input, which is where this codebase will actually break.

Run: /usr/bin/python3 adapter_project/tests/test_contracts.py

Contracts covered here:

  1. targets.enumerate_sites order is stable ACROSS PROCESSES. The site order is
     the leading dim of every (A, B) stack and the index for layer_ids/module_ids
     (inject.py:213-220, hypernet.py:116-119). A reorder between the process that
     trained and the process that serves is silently-wrong adapters, not a crash.
  2. Site.rel_name round-trips write_adapter -> read_adapter for every target set
     on the REAL Qwen3.5-9B config, and the on-disk PEFT key layout is exactly
     `base_model.model.<name>.lora_{A,B}.weight` with A (r, d_in), B (d_out, r).
  3. read_adapter's rel_name is prefix-independent (materialize.py:190-196).
  4. cells: the 3x6 grid matches the themes on disk, keys round-trip, splits
     partition, tasks partition by environment.
  5. transfer <-> cells: NaN-preserving save/load, and dissociation_scores groups
     on ERA, not on Cell.theme (theme strings differ across sites for one era, so
     grouping on them would silently empty the within-theme bucket).
  6. evalbridge.read_results reproduces the harness's own aggregate on an
     AgentLab-shaped tree, including the multi-trial retry dedupe.
  7. evalbridge.run and percell.run refuse to launch without an explicit flag.
  8. verify_adapter.make_probe_adapter writes a NON-ZERO B (a zero-init probe
     would make the Phase-0 serving verification vacuously pass).

Nothing here starts a server, an eval or a training job, and nothing is written
outside a tempdir.
"""
from __future__ import print_function

import csv
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

import numpy as np

from adaptercl import (cells, evalbridge, materialize, paths, percell, targets,
                       transfer, verify_adapter)

try:
    import torch
    HAVE_TORCH = True
except ImportError:                                   # pragma: no cover
    torch = None
    HAVE_TORCH = False

PASS = [0]
FAIL = [0]
SKIP = [0]


def check(name, cond):
    if cond:
        PASS[0] += 1
        print("  ok   %s" % name)
    else:
        FAIL[0] += 1
        print("  FAIL %s" % name)


def skip(name, why):
    SKIP[0] += 1
    print("  skip %s (%s)" % (name, why))


class _Quiet(object):
    """Swallow a module's stdout without hiding failures."""

    def write(self, s):
        pass

    def flush(self):
        pass


def _silent(fn, *a, **kw):
    """Call `fn` with stdout swallowed -- the writers print informational notes
    (e.g. materialize's safetensors-vs-.bin note) and one line per adapter would
    bury the checks."""
    old = sys.stdout
    sys.stdout = _Quiet()
    try:
        return fn(*a, **kw)
    finally:
        sys.stdout = old


# --------------------------------------------------------------------------
# 1. Site ordering is reproducible across processes
# --------------------------------------------------------------------------

#: Frozen module_id table. These are embedding row indices in every saved
#: hypernetwork checkpoint (hypernet.py:316), so renumbering them silently
#: repoints a trained module embedding at a different projection.
EXPECTED_MODULE_ID = {
    "q_proj": 0, "k_proj": 1, "v_proj": 2, "o_proj": 3,
    "in_proj_qkv": 4, "in_proj_z": 5, "in_proj_b": 6, "in_proj_a": 7,
    "out_proj": 8, "gate_proj": 9, "up_proj": 10, "down_proj": 11,
}

#: sha1 of "<rel_name> <module_id> <d_in> <d_out>" per site, in site order, over
#: the real Qwen/Qwen3.5-9B config. Frozen on 2026-08-06. The cross-process check
#: below catches NONDETERMINISM; this catches a code change that silently
#: renumbers or reorders the sites, which is the same disaster one commit later.
#: If you change a target set on purpose, recompute these and say so in the diff.
EXPECTED_SITE_DIGEST = {
    "attn_mlp": ("4579d43590d5dfa7", 128),
    "all": ("7cb21d87cb4caa00", 248),
}

_ORDER_SNIPPET = """
import json, sys
sys.path.insert(0, %r)
from adaptercl import targets
cfg = targets.load_text_config()
out = {"module_types": list(targets.MODULE_TYPES),
       "module_id": dict(targets.MODULE_ID),
       "sets": {}}
for name in targets.TARGET_SETS:
    out["sets"][name] = [[s.rel_name, s.layer, s.module, s.module_id,
                          s.d_in, s.d_out]
                         for s in targets.enumerate_sites(cfg, name)]
json.dump(out, sys.stdout)
""" % (PROJECT,)


def _enumerate_in_subprocess(hashseed):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hashseed)
    env["PYTHONPATH"] = PROJECT + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.check_output([sys.executable, "-c", _ORDER_SNIPPET],
                                  cwd=PROJECT, env=env)
    return json.loads(out.decode("utf-8"))


def test_site_order_is_stable_across_processes():
    if paths.hf_snapshot(paths.BASE_MODEL) is None:
        skip("targets: site order across processes", "%s not cached" % paths.BASE_MODEL)
        return
    # Two different hash seeds: any accidental dependence on set/dict iteration
    # order shows up as a diff here and nowhere else.
    a = _enumerate_in_subprocess(0)
    b = _enumerate_in_subprocess(524287)
    check("targets: site order identical across two processes", a["sets"] == b["sets"])
    check("targets: MODULE_TYPES order identical across processes",
          a["module_types"] == b["module_types"])
    check("targets: MODULE_ID is the frozen table",
          a["module_id"] == EXPECTED_MODULE_ID and b["module_id"] == EXPECTED_MODULE_ID)
    check("targets: MODULE_ID agrees with MODULE_TYPES position",
          all(a["module_id"][m] == i for i, m in enumerate(a["module_types"])))

    cfg = targets.load_text_config()
    in_proc = dict(
        (name, [[s.rel_name, s.layer, s.module, s.module_id, s.d_in, s.d_out]
                for s in targets.enumerate_sites(cfg, name)])
        for name in targets.TARGET_SETS)
    check("targets: in-process order matches the subprocess order",
          in_proc == a["sets"])

    # The order the hypernetwork indexes into: layer-major, then target-set order.
    ok_sorted, ok_unique = True, True
    for name, rows in a["sets"].items():
        layers = [r[1] for r in rows]
        if layers != sorted(layers):
            ok_sorted = False
        rels = [r[0] for r in rows]
        if len(set(rels)) != len(rels):
            ok_unique = False
    check("targets: sites are layer-major in every target set", ok_sorted)
    check("targets: rel_name is unique within a target set", ok_unique)
    check("targets: 'all' covers every module type once per eligible layer",
          len(a["sets"]["all"]) == 24 * 5 + 8 * 4 + 32 * 3)

    for name, (want_digest, want_n) in sorted(EXPECTED_SITE_DIGEST.items()):
        rows = a["sets"][name]
        blob = "\n".join("%s %d %d %d" % (r[0], r[3], r[4], r[5]) for r in rows)
        got = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
        check("targets: %s site order+shapes match the frozen digest (%d sites)"
              % (name, want_n),
              got == want_digest and len(rows) == want_n)


# --------------------------------------------------------------------------
# 2/3. materialize round trip + PEFT key layout + prefix independence
# --------------------------------------------------------------------------

def _synthetic_factors(sites, rank, seed=7):
    """Deterministic (A, B) per site, distinct per site so a mix-up is visible."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for i, s in enumerate(sites):
        A = torch.randn(rank, s.d_in, generator=g) * 0.01 + float(i)
        B = torch.randn(s.d_out, rank, generator=g) * 0.01 - float(i)
        out[s.rel_name] = (A, B)
    return out


def _raw_key_shapes(adapter_dir):
    """{key: shape} straight off disk, WITHOUT going through read_adapter.

    read_adapter strips the PEFT prefix, so asking it what the keys are would be
    circular -- the point is that the bytes on disk are what vLLM/PEFT expect.
    """
    st = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.exists(st):
        with open(st, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n).decode("utf-8"))
        return dict((k, tuple(v["shape"])) for k, v in header.items()
                    if k != "__metadata__")
    bn = os.path.join(adapter_dir, "adapter_model.bin")
    return dict((k, tuple(v.shape)) for k, v in torch.load(bn, map_location="cpu").items())


def _roundtrip_one(tmp, cfg, ts_name, rank, layers, tag):
    sites = targets.enumerate_sites(cfg, ts_name, layers=layers)
    factors = _synthetic_factors(sites, rank)
    out_dir = os.path.join(tmp, "adapter_%s" % tag)
    _silent(materialize.write_adapter, out_dir, factors, sites, rank,
            2.0 * rank, ts_name, base_model="test/base", layers=layers,
            meta={"tag": tag})

    want = {}
    for s in sites:
        want[materialize.PEFT_PREFIX + s.name + ".lora_A.weight"] = (rank, s.d_in)
        want[materialize.PEFT_PREFIX + s.name + ".lora_B.weight"] = (s.d_out, rank)
    got = _raw_key_shapes(out_dir)

    read, cfg_json, meta = materialize.read_adapter(out_dir)
    values_ok = True
    for s in sites:
        A0, B0 = factors[s.rel_name]
        ab = read.get(s.rel_name)
        if ab is None or not torch.equal(A0, ab[0]) or not torch.equal(B0, ab[1]):
            values_ok = False
            break
    return {
        "sites": sites, "keys_ok": got == want, "n_sites": len(sites),
        "rel_ok": set(read) == set(s.rel_name for s in sites),
        "values_ok": values_ok, "cfg": cfg_json, "meta": meta,
        "factors": factors, "dir": out_dir,
    }


def test_materialize_roundtrip_every_target_set(tmp):
    if not HAVE_TORCH:
        skip("materialize: round trip", "no torch in this interpreter")
        return
    if paths.hf_snapshot(paths.BASE_MODEL) is None:
        skip("materialize: round trip", "%s not cached" % paths.BASE_MODEL)
        return
    cfg = targets.load_text_config()
    rank = 2
    # Layers 0-3 of the real 9B: three linear_attention layers plus the first
    # full_attention layer, so every module type in every target set is present
    # at real widths while the files stay small.
    layers = (0, 1, 2, 3)

    keys_ok = rel_ok = vals_ok = cfg_ok = True
    n_sets = 0
    for ts_name in targets.TARGET_SETS:
        r = _roundtrip_one(tmp, cfg, ts_name, rank, layers, ts_name)
        n_sets += 1
        keys_ok = keys_ok and r["keys_ok"]
        rel_ok = rel_ok and r["rel_ok"]
        vals_ok = vals_ok and r["values_ok"]
        want_modules = sorted(targets.target_set(ts_name))
        cfg_ok = cfg_ok and (r["cfg"]["target_modules"] == want_modules
                             and int(r["cfg"]["r"]) == rank
                             and float(r["cfg"]["lora_alpha"]) == 2.0 * rank
                             and r["cfg"]["peft_type"] == "LORA"
                             and r["meta"]["n_sites"] == r["n_sites"])
    check("materialize: PEFT key layout exact for all %d target sets" % n_sets, keys_ok)
    check("materialize: read_adapter keys are exactly the site rel_names", rel_ok)
    check("materialize: factor values survive the round trip bit-for-bit", vals_ok)
    check("materialize: adapter_config.json matches the target set", cfg_ok)

    # The default target set at full depth -- 128 sites, all 32 layers.
    full = _roundtrip_one(tmp, cfg, targets.DEFAULT_TARGET_SET, rank, None, "full")
    check("materialize: full-depth %s round trip (%d sites)"
          % (targets.DEFAULT_TARGET_SET, full["n_sites"]),
          full["keys_ok"] and full["rel_ok"] and full["values_ok"])
    check("materialize: write_adapter rejects a wrong-shaped factor",
          _rejects_bad_shape(tmp, full["sites"][:2], rank))


def _rejects_bad_shape(tmp, sites, rank):
    """A transposed B must not be accepted -- that is the silent-adapter class."""
    bad = {}
    for s in sites:
        bad[s.rel_name] = (torch.zeros(rank, s.d_in), torch.zeros(rank, s.d_out))
    try:
        _silent(materialize.write_adapter, os.path.join(tmp, "bad"), bad, sites,
                rank, rank, ["q_proj"])
    except ValueError:
        return True
    return False


def test_rel_name_is_prefix_independent(tmp):
    if not HAVE_TORCH:
        skip("materialize: prefix independence", "no torch in this interpreter")
        return
    if paths.hf_snapshot(paths.BASE_MODEL) is None:
        skip("materialize: prefix independence", "%s not cached" % paths.BASE_MODEL)
        return
    cfg = targets.load_text_config()
    rank, ts, layers = 2, "attn", (3,)

    # Same sites, three different module prefixes: the HF root, a PEFT-wrapped
    # root, and a bare causal-LM root.
    prefixes = ("model.language_model",
                "base_model.model.model.language_model",
                "model")
    reads = []
    for i, prefix in enumerate(prefixes):
        sites = targets.enumerate_sites(cfg, ts, layers=layers, prefix=prefix)
        factors = _synthetic_factors(sites, rank, seed=11)
        d = os.path.join(tmp, "prefix_%d" % i)
        _silent(materialize.write_adapter, d, factors, sites, rank, rank, ts)
        reads.append(materialize.read_adapter(d)[0])
        # the on-disk keys DO carry the prefix -- that is what vLLM remaps
        raw = _raw_key_shapes(d)
        if i == 0:
            check("materialize: on-disk keys carry the full module path",
                  all(k.startswith(materialize.PEFT_PREFIX + prefix + ".layers.")
                      for k in raw))

    base_keys = set(reads[0])
    check("materialize: rel_name identical under 3 module prefixes",
          all(set(r) == base_keys for r in reads[1:]))
    same = True
    for r in reads[1:]:
        for k in base_keys:
            if not torch.equal(reads[0][k][0], r[k][0]) or \
               not torch.equal(reads[0][k][1], r[k][1]):
                same = False
    check("materialize: factors identical under 3 module prefixes", same)
    check("materialize: rel_name is what a differently-prefixed site list asks for",
          base_keys == set(s.rel_name for s in targets.enumerate_sites(
              cfg, ts, layers=layers, prefix="something.else.entirely")))


# --------------------------------------------------------------------------
# 4. cells: the grid against the themes on disk
# --------------------------------------------------------------------------

def test_cells_grid():
    check("cells: 3 x 6 = 18 cells", len(cells.ALL_CELLS) == 18)

    missing = []
    for c in cells.ALL_CELLS:
        d = os.path.join(cells.THEME_ROOT[c.env], c.theme)
        if not os.path.isdir(d):
            missing.append(d)
    check("cells: every cell's theme dir exists on disk (%d)" % len(cells.ALL_CELLS),
          not missing)
    if missing:
        print("       missing: %s" % missing[:3])

    check("cells: Cell.parse(key) round-trips for all 18",
          all(cells.Cell.parse(c.key) == c for c in cells.ALL_CELLS))
    check("cells: keys are unique",
          len(set(c.key for c in cells.ALL_CELLS)) == 18)

    # era 6 is the style-neutral baseline, NOT the newest era
    check("cells: TEMPORAL_ERAS excludes the neutral era",
          cells.NEUTRAL_ERA not in cells.TEMPORAL_ERAS
          and set(cells.TEMPORAL_ERAS) | set([cells.NEUTRAL_ERA]) == set(cells.ERAS))
    check("cells: nominal_year is None exactly on the neutral era",
          all((cells.nominal_year(e, v) is None) == (v == cells.NEUTRAL_ERA)
              for e in cells.ENVIRONMENTS for v in cells.ERAS))
    check("cells: the temporal axis is monotone per environment",
          all([cells.nominal_year(e, v) for v in cells.TEMPORAL_ERAS]
              == sorted(cells.nominal_year(e, v) for v in cells.TEMPORAL_ERAS)
              for e in cells.ENVIRONMENTS))
    check("cells: TEMPORAL_CELLS is exactly the non-neutral cells",
          set(c.key for c in cells.TEMPORAL_CELLS)
          == set(c.key for c in cells.ALL_CELLS if c.era != cells.NEUTRAL_ERA))

    # splits partition eras
    bad_overlap, bad_union, bad_cells = [], [], []
    for name, sp in cells.SPLITS.items():
        tr, te = set(sp.train_eras), set(sp.test_eras)
        if tr & te:
            bad_overlap.append(name)
        if not (tr | te) <= set(cells.ERAS):
            bad_union.append(name)
        trc = set(c.key for c in sp.train_cells())
        tec = set(c.key for c in sp.test_cells())
        if trc & tec or len(trc) != 3 * len(tr) or len(tec) != 3 * len(te):
            bad_cells.append(name)
    check("cells: no split has an era in both train and test", not bad_overlap)
    check("cells: every split's eras are real eras", not bad_union)
    check("cells: split cell sets are disjoint and complete", not bad_cells)
    check("cells: the rotation folds partition the temporal eras",
          all(set(cells.SPLITS[n].train_eras) | set(cells.SPLITS[n].test_eras)
              == set(cells.TEMPORAL_ERAS) for n in cells.ROTATION_FOLDS))
    check("cells: neutral_holdout holds out exactly the neutral era",
          cells.SPLITS["neutral_holdout"].test_eras == (cells.NEUTRAL_ERA,))
    check("cells: the headline split exists and is an extrapolation",
          cells.SPLITS[cells.HEADLINE_SPLIT].kind == "extrapolation")
    check("cells: SEQUENTIAL_STREAM ends on the neutral era",
          cells.SEQUENTIAL_STREAM[-1] == cells.NEUTRAL_ERA
          and set(cells.SEQUENTIAL_STREAM) == set(cells.ERAS))


def test_tasks_partition_by_environment():
    try:
        all_ids = set(t["task_id"] for t in cells.load_tasks())
    except (IOError, ValueError) as exc:
        skip("cells: task partition", "task JSON unreadable: %s" % exc)
        return
    per_env = dict((e, set(cells.tasks_for_env(e))) for e in cells.ENVIRONMENTS)
    check("cells: every environment has single-site tasks",
          all(len(v) > 0 for v in per_env.values()))
    overlaps = [(a, b) for a in cells.ENVIRONMENTS for b in cells.ENVIRONMENTS
                if a < b and (per_env[a] & per_env[b])]
    check("cells: single-site task sets are pairwise disjoint", not overlaps)
    check("cells: single-site tasks are a subset of the task file",
          set().union(*per_env.values()) <= all_ids)
    check("cells: multi-site tasks are excluded by single_site=True",
          len(set().union(*per_env.values())) < len(all_ids))
    check("cells: cell_tasks depends only on the environment, not the era",
          all(cells.cell_tasks(cells.Cell(e, 1)) == cells.cell_tasks(cells.Cell(e, 5))
              for e in cells.ENVIRONMENTS))
    tmap = cells.task_cell_map(3)
    check("cells: task_cell_map covers exactly the single-site tasks",
          set(tmap) == set().union(*per_env.values())
          and all(c.era == 3 for c in tmap.values()))


# --------------------------------------------------------------------------
# 5. transfer <-> cells
# --------------------------------------------------------------------------

def _synth_transfer(theme_val=0.60, env_val=0.20, neither_val=0.10, diag=0.80,
                    holes=()):
    """A matrix with the 6.2 dissociation built in, by ERA not by theme string."""
    cs = list(cells.ALL_CELLS)
    tm = transfer.TransferMatrix(cs, meta={"synthetic": True})
    for i, a in enumerate(cs):
        for j, b in enumerate(cs):
            if (i, j) in holes:
                continue
            if i == j:
                v = diag
            elif a.era == b.era:
                v = theme_val
            elif a.env == b.env:
                v = env_val
            else:
                v = neither_val
            tm.set(i, j, v, n_episodes=7, output_root="/dev/null/%d_%d" % (i, j))
    return tm


def test_transfer_matrix_roundtrip(tmp):
    holes = ((0, 1), (5, 5), (17, 3))
    tm = _synth_transfer(holes=holes)
    path = os.path.join(tmp, "transfer_synth.json")
    tm.save(path)
    back = transfer.TransferMatrix.load(path)

    check("transfer: cell keys survive save/load", back.keys == tm.keys)
    check("transfer: keys are cells.ALL_CELLS in order",
          back.keys == [c.key for c in cells.ALL_CELLS])
    check("transfer: NaNs survive the JSON round trip (written as null)",
          np.array_equal(np.isnan(back.M), np.isnan(tm.M)))
    check("transfer: %d holes stay holes" % len(holes),
          all(math.isnan(back.get(i, j)) for i, j in holes))
    finite = np.isfinite(tm.M)
    check("transfer: finite values are unchanged",
          np.allclose(back.M[finite], tm.M[finite]))
    check("transfer: per-entry provenance survives",
          back.provenance("wiki_e1", "wiki_e2") == tm.provenance("wiki_e1", "wiki_e2"))
    check("transfer: coverage counts the holes",
          back.coverage()["filled"] == 18 * 18 - len(holes)
          and not back.is_complete())
    with open(path) as fh:
        raw = fh.read()
    check("transfer: no bare NaN token in the JSON (json.load would still take "
          "it, but nothing else would)", "NaN" not in raw)


def test_dissociation_groups_on_era_not_theme():
    tm = _synth_transfer()
    d = transfer.dissociation_scores(tm)

    # The trap this test exists for: within one era the three sites have three
    # DIFFERENT theme directory names, so grouping on Cell.theme empties the
    # within-theme bucket without raising anything.
    theme_strings = set(cells.Cell(e, 3).theme for e in cells.ENVIRONMENTS)
    check("transfer: the three sites' era-3 theme strings really do differ",
          len(theme_strings) == 3)
    n_if_grouped_on_theme = sum(
        1 for a in cells.ALL_CELLS for b in cells.ALL_CELLS
        if a != b and a.theme == b.theme and a.env != b.env)
    check("transfer: grouping on the theme string would give an EMPTY bucket",
          n_if_grouped_on_theme == 0)

    check("transfer: within-theme bucket is non-empty",
          d["within_theme_across_env"]["n"] > 0)
    check("transfer: within-theme bucket has all 36 ordered pairs",
          d["within_theme_across_env"]["n"] == 36
          and d["coverage"]["theme_pairs_total"] == 36)
    check("transfer: within-env bucket has all 90 ordered pairs",
          d["within_env_across_theme"]["n"] == 90
          and d["coverage"]["env_pairs_total"] == 90)
    check("transfer: 'neither' bucket takes the remaining 180 pairs",
          d["neither_shared"]["n"] == 18 * 17 - 36 - 90)
    check("transfer: the two means are the planted ones",
          abs(d["within_theme_across_env"]["mean"] - 0.60) < 1e-12
          and abs(d["within_env_across_theme"]["mean"] - 0.20) < 1e-12)
    check("transfer: difference is theme minus env",
          abs(d["difference"] - 0.40) < 1e-12)
    check("transfer: diagonal is reported as the oracle",
          abs(d["diagonal_oracle"]["mean"] - 0.80) < 1e-12
          and d["diagonal_oracle"]["n"] == 18)
    check("transfer: per-source rows exist for all 18 cells",
          len(d["per_source_cell"]) == 18 and len(d["per_target_cell"]) == 18)
    check("transfer: every source cell sees 2 theme partners and 5 era partners",
          all(r["theme_n"] == 2 and r["env_n"] == 5 for r in d["per_source_cell"]))
    check("transfer: the module says so out loud (theme == era)",
          d["theme_is_era"] is True and "never group on it" in d["note"])

    # A half-finished matrix must still be analysable: the sweep is 324 evals and
    # will not finish in one sitting. wiki_e1 <-> wiki_e2 are a within-env pair,
    # so exactly two entries drop out of the env bucket and none out of theme.
    partial = _synth_transfer(holes=((0, 1), (1, 0)))
    dp = transfer.dissociation_scores(partial)
    check("transfer: dissociation tolerates missing pairs",
          dp["within_theme_across_env"]["n"] == 36
          and dp["within_env_across_theme"]["n"] == 88
          and dp["coverage"]["env_pairs_total"] == 90
          and abs(dp["within_env_across_theme"]["mean"] - 0.20) < 1e-12)


# --------------------------------------------------------------------------
# 6. evalbridge.read_results on an AgentLab-shaped tree
# --------------------------------------------------------------------------

#: The columns read_results actually reads, in the order the real frames carry
#: them (verified against self-correct/p0_q35_gemma4/v6/result_df_trial_1_of_3.csv).
_RESULT_COLS = ["exp_dir", "env.task_name", "env.task_seed", "cum_reward",
                "cum_raw_reward", "n_steps", "err_msg", "exp_name", "terminated"]

_AGENT = "GenericAgentWithTraining-toy"


def _exp_dir(study_dir, ts, task, seed, repeat=None):
    name = "%s_%s_on_timewarp.%d_%d" % (ts, _AGENT, task, seed)
    if repeat is not None:
        name += "_%d" % repeat
    return os.path.join(study_dir, name)


def _write_frame(path, rows):
    with open(path, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=_RESULT_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _row(exp_dir, task, seed, reward, steps, err=""):
    return {"exp_dir": exp_dir, "env.task_name": "timewarp.%d" % task,
            "env.task_seed": str(seed), "cum_reward": "%.1f" % reward,
            "cum_raw_reward": "%.1f" % reward, "n_steps": str(steps),
            "err_msg": err, "exp_name": os.path.basename(exp_dir),
            "terminated": "True"}


def _build_agentlab_tree(root):
    """A study with a retry round, a (task, seed) collision, and a hard error.

    Trial 2 is AgentLab's full re-dump after the retry round: the two retried
    episodes come back under NEW timestamps, everything else is byte-identical.
    """
    study = os.path.join(root, "results", "toy-model", "v6")
    os.makedirs(study)
    t1, t2 = "2026-08-06_10-00-00", "2026-08-06_11-30-00"

    # (task, seed, repeat, reward_trial1, reward_trial2, steps, err, retried)
    plan = [
        (1, 10, None, 1.0, 1.0, 5, "", False),
        (2, 11, None, 0.0, 1.0, 30, "", True),     # retried, then solved
        (3, 12, None, 0.0, 0.0, 12, "", True),     # retried, still failed
        (3, 12, 0, 1.0, 1.0, 9, "", False),        # same (task, seed), `_0` suffix
        (4, 13, None, 0.0, 0.0, 2, "boom: TimeoutError", False),
    ]
    rows1, rows2, keys = [], [], set()
    for task, seed, rep, r1, r2, steps, err, retried in plan:
        d1 = _exp_dir(study, t1, task, seed, rep)
        d2 = _exp_dir(study, t2 if retried else t1, task, seed, rep)
        rows1.append(_row(d1, task, seed, r1, steps, err))
        rows2.append(_row(d2, task, seed, r2, steps, err))
        keys.add(evalbridge.episode_key(d1))
        for d in (d1, d2):
            if not os.path.isdir(d):
                os.makedirs(d)
    _write_frame(os.path.join(study, "result_df_trial_1_of_2.csv"), rows1)
    _write_frame(os.path.join(study, "result_df_trial_2_of_2.csv"), rows2)

    # The harness's own summary of the authoritative (last) frame.
    rewards = [float(r["cum_reward"]) for r in rows2]
    steps = [int(r["n_steps"]) for r in rows2]
    n = len(rewards)
    mean = sum(rewards) / n
    sem = math.sqrt(sum((x - mean) ** 2 for x in rewards) / (n - 1) / n)
    summary = {"agent.agent_name": _AGENT, "env.benchmark": "timewarp",
               "avg_reward": "%.3f" % mean, "std_err": "%.3f" % sem,
               "avg_steps": "%.3f" % (sum(steps) / float(n)),
               "n_completed": "%d/%d" % (n, n),
               "n_err": str(sum(1 for r in rows2 if r["err_msg"])),
               "cum_cost": "0.0"}
    with open(os.path.join(study, "summary_df_trial_2_of_2.csv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary))
        w.writeheader()
        w.writerow(summary)

    return {"study": study, "rows1": rows1, "rows2": rows2, "summary": summary,
            "n_unique": len(keys),
            "n_exp_dirs": len(set(r["exp_dir"] for r in rows1 + rows2)),
            "n_task_seed": len(set((r["env.task_name"], r["env.task_seed"])
                                   for r in rows1 + rows2))}


def test_read_results_matches_the_harness(tmp):
    root = os.path.join(tmp, "eval_run")
    fx = _build_agentlab_tree(root)
    res = evalbridge.read_results(root)
    agg = res["aggregate"]
    s = fx["summary"]

    check("evalbridge: status ok", res["status"] == "ok")
    check("evalbridge: both trial frames were read",
          res["n_rows_read"] == len(fx["rows1"]) + len(fx["rows2"]))
    check("evalbridge: one record per episode (%d), not per row" % fx["n_unique"],
          agg["n"] == fx["n_unique"] == 5)
    # the two identities that are WRONG here, stated as numbers
    check("evalbridge: naive exp_dir dedupe would have inflated to %d"
          % fx["n_exp_dirs"], fx["n_exp_dirs"] == 7)
    check("evalbridge: naive (task, seed) dedupe would have collapsed to %d"
          % fx["n_task_seed"], fx["n_task_seed"] == 4)
    check("evalbridge: the `_k` repeat suffix keeps the colliding pair apart",
          len([r for r in res["records"] if r["task_id"] == 3]) == 2)
    check("evalbridge: the last trial's row wins",
          all(r["reward"] == 1.0 for r in res["records"]
              if (r["task_id"], r["seed"], r["repeat"]) == (2, 11, 0)))
    check("evalbridge: retried episodes are counted and attributed",
          res["n_retried_episodes"] == 2
          and sorted(r["n_attempts"] for r in res["records"]) == [1, 1, 1, 2, 2])

    check("evalbridge: avg_reward reproduces the harness's summary_df",
          abs(agg["avg_reward"] - float(s["avg_reward"])) < 5e-4)
    check("evalbridge: std_err reproduces the harness's summary_df",
          abs(agg["std_err"] - float(s["std_err"])) < 5e-4)
    check("evalbridge: avg_steps reproduces the harness's summary_df",
          abs(agg["avg_steps"] - float(s["avg_steps"])) < 5e-4)
    check("evalbridge: n matches n_completed",
          agg["n"] == int(s["n_completed"].split("/")[0]))
    check("evalbridge: n_err matches", agg["n_err"] == int(s["n_err"]))
    check("evalbridge: summary_df was located and attached",
          res["summary_df"] is not None
          and res["summary_df"]["avg_reward"] == s["avg_reward"])
    check("evalbridge: success_rate == solved / n",
          abs(evalbridge.success_rate(res["records"])
              - agg["solved"] / float(agg["n"])) < 1e-12
          and agg["solved"] == 3)
    check("evalbridge: records are sorted by (task, seed, repeat)",
          [(r["task_id"], r["seed"], r["repeat"]) for r in res["records"]]
          == sorted((r["task_id"], r["seed"], r["repeat"]) for r in res["records"]))
    check("evalbridge: one study dir found", len(res["study_dirs"]) == 1)

    # a crashed or unstarted eval must be reportable, never fatal
    check("evalbridge: missing root -> status 'missing', no raise",
          evalbridge.read_results(os.path.join(tmp, "nope"))["status"] == "missing")
    empty = os.path.join(tmp, "empty_run")
    os.makedirs(empty)
    check("evalbridge: empty root -> status 'empty', no raise",
          evalbridge.read_results(empty)["status"] == "empty")

    # drift_gap consumes exactly these records (6.6's headline)
    gap = evalbridge.drift_gap(res["records"][:3], res["records"][3:])
    check("evalbridge: drift_gap consumes read_results records",
          gap["n_train"] == 3 and gap["n_test"] == 2 and gap["gap"] is not None)


# --------------------------------------------------------------------------
# 6b. transfer consumes evalbridge: the pair-dir convention end to end
# --------------------------------------------------------------------------

def test_transfer_reads_evalbridge_output(tmp):
    root = os.path.join(tmp, "pairs")
    a, b = cells.Cell("wiki", 3), cells.Cell("shop", 3)
    dirs = {}
    for train, ev in transfer.build_pairs([a, b]):
        d = os.path.join(root, transfer.pair_dir_name(train, ev))
        os.makedirs(d)
        _build_agentlab_tree(d)
        dirs[(train.key, ev.key)] = d

    tm = transfer.from_eval_dirs(root, cells=[a, b])
    check("transfer<-evalbridge: all 4 pair dirs were found",
          tm.is_complete() and tm.meta["from_eval_dirs"]["found"] == 4)
    want = evalbridge.success_rate(
        evalbridge.read_results(dirs[(a.key, b.key)])["records"])
    check("transfer<-evalbridge: the matrix entry IS evalbridge's success rate",
          abs(tm.get(a, b) - want) < 1e-12 and abs(want - 0.6) < 1e-12)
    check("transfer<-evalbridge: provenance records the episode count",
          tm.provenance(a, b)["n_episodes"] == 5
          and tm.provenance(a, b)["source"] == "from_eval_dirs")
    check("transfer<-evalbridge: a missing pair dir stays NaN, not zero",
          math.isnan(transfer.from_eval_dirs(
              os.path.join(tmp, "no_pairs"), cells=[a, b]).get(a, b)))

    bad = [n for train, ev in transfer.build_pairs(cells.ALL_CELLS)
           for n in [transfer.pair_dir_name(train, ev)]
           if transfer.parse_pair_dir(n) != (train, ev)]
    check("transfer: pair_dir_name/parse_pair_dir round-trip over all 324 pairs",
          not bad and len(transfer.build_pairs(cells.ALL_CELLS)) == 324)


# --------------------------------------------------------------------------
# 6c. percell and evalbridge consume what materialize writes
# --------------------------------------------------------------------------

def test_adapter_dir_is_consumable(tmp):
    if not HAVE_TORCH:
        skip("percell/evalbridge: adapter dir contract", "no torch")
        return
    if paths.hf_snapshot(paths.BASE_MODEL) is None:
        skip("percell/evalbridge: adapter dir contract", "base model not cached")
        return
    cfg = targets.load_text_config()
    cell = cells.Cell("wiki", 3)
    root, tag, rank = os.path.join(tmp, "cells"), "unit", 16
    sites = targets.enumerate_sites(cfg, "attn_mlp", layers=(3,))
    d = percell.cell_output_dir(cell, tag, root=root)
    _silent(materialize.write_adapter, d, _synthetic_factors(sites, rank), sites,
            rank, 2 * rank, "attn_mlp")

    got = _silent(percell.collect_adapters, tag=tag, cell_list=[cell], root=root,
                  quiet=True)
    rec = got["found"].get(cell.key)
    check("percell: collect_adapters accepts what write_adapter wrote",
          rec is not None and not got["invalid"] and got["missing"] == [])
    check("percell: ...and reads back r / n_sites / n_params",
          rec and rec["r"] == rank and rec["n_sites"] == len(sites)
          and rec["n_params"] == targets.site_budget(sites, rank))

    # LLaMA-Factory kills happen between epochs: the checkpoint fallback matters
    ck_cell = cells.Cell("news", 3)
    ck = os.path.join(percell.cell_output_dir(ck_cell, tag, root=root),
                      "checkpoint-120")
    _silent(materialize.write_adapter, ck, _synthetic_factors(sites, rank), sites,
            rank, 2 * rank, "attn_mlp")
    got2 = _silent(percell.collect_adapters, tag=tag, cell_list=[ck_cell],
                   root=root, quiet=True)
    check("percell: falls back to the highest checkpoint-N",
          got2["found"].get(ck_cell.key, {}).get("dir") == ck)

    # evalbridge derives the server's LoRA config from that same directory
    spec = evalbridge.EvalSpec(model="toy/model", adapter_dir=d, cell=cell,
                               output_root=os.path.join(tmp, "eo"))
    env = evalbridge.build_env(spec)
    check("evalbridge: max_lora_rank comes from the adapter's own config",
          spec.max_lora_rank == rank and env["MAX_LORA_RANK"] == str(rank))
    check("evalbridge: LORA_MODULES is '<served name>=<adapter dir>'",
          env["LORA_MODULES"] == "%s=%s" % (cell.key, d)
          and env["TW_SERVED_NAME"] == cell.key)
    check("evalbridge: the agent is pointed at the LoRA, not the base name",
          spec.served_name != spec.model and not evalbridge.preflight(spec))
    check("evalbridge: VERSION follows the cell's era",
          env["VERSION"] == str(cell.era)
          and env["TW_TASK_IDS"].split(",")[0].isdigit())

    # a hypernetwork adapter has r = K*rank, which is NOT a legal vLLM server
    # rank; scripts/startVLM_lora.sh rounds the SERVER's max rank up. Keep the
    # python constant and the shell's list in step.
    launcher = open(paths.VLM_LORA_LAUNCHER).read()
    check("evalbridge: the launcher's legal-rank list == VLLM_LORA_RANKS",
          " ".join(str(r) for r in verify_adapter.VLLM_LORA_RANKS) in launcher)
    check("evalbridge: a generated r=K*rank rounds the SERVER up to a legal rank "
          "(6->8, 48->64)",
          min(r for r in verify_adapter.VLLM_LORA_RANKS if r >= 6) == 8
          and min(r for r in verify_adapter.VLLM_LORA_RANKS if r >= 48) == 64)


# --------------------------------------------------------------------------
# 7. Nothing launches without an explicit flag
# --------------------------------------------------------------------------

class _NoLaunch(object):
    """A `subprocess` stand-in that records and refuses every spawn."""

    PIPE = subprocess.PIPE
    STDOUT = subprocess.STDOUT
    CalledProcessError = subprocess.CalledProcessError

    def __init__(self):
        self.calls = []

    def Popen(self, *a, **kw):
        self.calls.append(("Popen",) + a)
        raise AssertionError("Popen was called: %r" % (a,))

    def call(self, *a, **kw):
        self.calls.append(("call",) + a)
        raise AssertionError("subprocess.call was called: %r" % (a,))

    def check_call(self, *a, **kw):
        self.calls.append(("check_call",) + a)
        raise AssertionError("check_call was called: %r" % (a,))

    def check_output(self, *a, **kw):
        # what `nvidia-smi -L` does on a login node
        self.calls.append(("check_output",) + a)
        raise OSError("nvidia-smi is not available (test stub)")


def test_nothing_launches_by_default(tmp):
    guard = _NoLaunch()
    real_eval_sp, real_percell_sp = evalbridge.subprocess, percell.subprocess
    real_stdout = sys.stdout
    evalbridge.subprocess = guard
    percell.subprocess = guard
    try:
        # -- evalbridge -----------------------------------------------------
        spec = evalbridge.EvalSpec(model="toy/model", cell=cells.Cell("wiki", 5),
                                   output_root=os.path.join(tmp, "eval_out"),
                                   deterministic_judge=True)
        out = evalbridge.run(spec, fh=_Quiet())
        check("evalbridge: run() defaults to a dry run", out["dry_run"] is True)
        check("evalbridge: run() with defaults spawned nothing", not guard.calls)
        check("evalbridge: the dry run still produced the exact command",
              out["argv"][0] == "bash" and out["argv"][1] == paths.EVAL_DRIVER
              and "MODEL_PATH=" in out["command"])
        check("evalbridge: the dry run did not create OUTPUT_ROOT",
              not os.path.exists(os.path.join(tmp, "eval_out")))

        # a spec that cannot possibly run must refuse before spawning anything
        bad = evalbridge.EvalSpec(model="toy/model", cell=cells.Cell("wiki", 5),
                                  adapter_dir=None,
                                  output_root=os.path.join(tmp, "eval_out2"))
        bad.adapter_dir = os.path.join(tmp, "no_such_adapter")   # never written
        raised = False
        try:
            evalbridge.run(bad, dry_run=False, fh=_Quiet())
        except RuntimeError:
            raised = True
        except AssertionError:
            raised = False
        check("evalbridge: run(dry_run=False) refuses on preflight problems, "
              "before spawning", raised and not guard.calls)

        # -- percell --------------------------------------------------------
        job = percell.CellJob(cells.Cell("news", 2), "adaptercl_test_news_e2",
                              "attn_mlp", 16, 32.0, "test",
                              os.path.join(tmp, "cell.yaml"),
                              os.path.join(tmp, "cell_out"), n_samples=100)
        sys.stdout = _Quiet()
        res = percell.run([job], log_dir=os.path.join(tmp, "logs"))
        sys.stdout = real_stdout
        check("percell: run() defaults to a dry run", res["executed"] is False)
        check("percell: run() with defaults spawned nothing", not guard.calls)
        check("percell: the dry run still produced a runnable command",
              len(res["commands"]) == 1 and paths.LMF in res["commands"][0]
              and job.yaml_path in res["commands"][0])
        check("percell: the dry run did not create the log dir",
              not os.path.exists(os.path.join(tmp, "logs")))

        # execute=True on a machine with no GPU must raise, not train
        raised = False
        try:
            sys.stdout = _Quiet()
            percell.run([job], execute=True, log_dir=os.path.join(tmp, "logs"))
        except RuntimeError:
            raised = True
        except AssertionError:
            raised = False
        finally:
            sys.stdout = real_stdout
        check("percell: execute=True refuses when nvidia-smi fails", raised)
        check("percell: ...and it checked for a GPU before spawning anything",
              [c[0] for c in guard.calls] == ["check_output"])
    finally:
        sys.stdout = real_stdout
        evalbridge.subprocess = real_eval_sp
        percell.subprocess = real_percell_sp


# --------------------------------------------------------------------------
# 8. The probe adapter must not be a no-op
# --------------------------------------------------------------------------

def test_probe_adapter_is_not_a_noop(tmp):
    if not HAVE_TORCH:
        skip("verify_adapter: probe adapter", "no torch in this interpreter")
        return
    if paths.hf_snapshot(paths.BASE_MODEL) is None:
        skip("verify_adapter: probe adapter", "%s not cached" % paths.BASE_MODEL)
        return
    d = os.path.join(tmp, "probe")
    _silent(verify_adapter.make_probe_adapter, d, target_set="attn_mlp",
            layers=(3,))
    factors, cfg, meta = materialize.read_adapter(d)

    check("verify_adapter: probe covers every site of the target set",
          len(factors) == len(targets.enumerate_sites(
              targets.load_text_config(), "attn_mlp", layers=(3,))) == 7)
    b_norms = [float(B.abs().sum()) for _A, B in factors.values()]
    a_norms = [float(A.abs().sum()) for A, _B in factors.values()]
    check("verify_adapter: EVERY probe B is non-zero (a zero B would make the "
          "Phase-0 verification vacuously pass)", all(v > 0 for v in b_norms))
    check("verify_adapter: every probe A is non-zero", all(v > 0 for v in a_norms))
    check("verify_adapter: scaling is exactly 1 so `scale` is the perturbation",
          abs(float(cfg["lora_alpha"]) / float(cfg["r"]) - 1.0) < 1e-12)
    check("verify_adapter: the probe is labelled as not-a-model",
          meta.get("probe") is True and "Never report numbers" in meta["warning"])

    # the perturbation really is ~PROBE_SCALE of each site's own output scale
    A, B = factors["layers.3.self_attn.v_proj"]
    d_in = A.shape[1]
    x = torch.randn(4096, d_in) / math.sqrt(d_in)      # unit-RMS-ish input
    rel = float((x @ A.t() @ B.t()).pow(2).mean().sqrt()) / float(x.pow(2).mean().sqrt())
    check("verify_adapter: relative perturbation is within 2x of PROBE_SCALE "
          "(%.3f vs %.3f)" % (rel, verify_adapter.PROBE_SCALE),
          0.5 * verify_adapter.PROBE_SCALE < rel < 2.0 * verify_adapter.PROBE_SCALE)

    d2 = os.path.join(tmp, "probe2")
    _silent(verify_adapter.make_probe_adapter, d2, target_set="attn_mlp",
            layers=(3,))
    f2, _, _ = materialize.read_adapter(d2)
    check("verify_adapter: the probe is deterministic in its seed",
          all(torch.equal(factors[k][0], f2[k][0])
              and torch.equal(factors[k][1], f2[k][1]) for k in factors))


# --------------------------------------------------------------------------

def test_split_csv_is_parsed_with_a_real_csv_reader():
    """All 231 tasks must land in the split map, multi-site ones included.

    The `sites` column is a QUOTED field containing commas ("news,wiki"), so
    `line.split(",")` misaligns every column after it and silently dropped all
    50 multi-site tasks -- 23 of 103 test, 27 of 128 train. It went unnoticed
    because single-site rows parse fine, so anything asking for
    `single_site=True` got the right answer; only a run needing the FULL test
    set saw the truncation, and then only as a smaller-than-expected task list.
    """
    sp = cells.load_split()
    test = sorted(k for k, v in sp.items() if v == "test")
    train = sorted(k for k, v in sp.items() if v == "train")
    check("cells: test split is exactly ids 1-103 (%d)" % len(test),
          test == list(range(1, 104)))
    check("cells: train split is exactly ids 104-231 (%d)" % len(train),
          train == list(range(104, 232)))

    # the tasks that used to vanish are precisely the multi-site ones
    tasks = dict((t["task_id"], t.get("sites") or []) for t in cells.load_tasks())
    multi = [t for t, s in tasks.items() if len(s) > 1]
    check("cells: every multi-site task has a split (%d of them)" % len(multi),
          multi and all(sp.get(t) in ("test", "train") for t in multi))

    # and the derived task lists still partition
    per_env = dict((e, set(cells.tasks_for_env(e, split="test")))
                   for e in cells.ENVIRONMENTS)
    union = set().union(*per_env.values())
    check("cells: single-site test tasks are disjoint across environments",
          sum(len(v) for v in per_env.values()) == len(union))
    check("cells: single-site test tasks are a strict subset of the 103",
          union < set(test))


def test_dissociation_refuses_a_version_matrix():
    """6.2's contrast needs BOTH factors; a Version matrix has only one.

    A 6x6 matrix over Versions has no environment factor, so every within-theme
    and within-environment bucket is empty. Before this guard the function
    returned a well-formed dict of Nones -- a dissociation "result" computed
    from zero observations, which is precisely the silent invalidity 9 lists in
    its risk register.
    """
    vs = list(cells.ALL_VERSIONS)
    Mv = transfer.TransferMatrix(vs)
    for i in range(len(vs)):
        for j in range(len(vs)):
            Mv.set(i, j, 0.4 if i == j else 0.2)
    refused = False
    try:
        transfer.dissociation_scores(Mv.M, vs)
    except ValueError as exc:
        refused = "environment" in str(exc)
    check("transfer: dissociation_scores REFUSES a version matrix", refused)

    # ...and distance_decay, the right test at this granularity, still works.
    dd = transfer.distance_decay(Mv.M, vs)
    check("transfer: distance_decay works on versions and reports a verdict",
          "verdict" in dd and "monotone_decay" in dd)

    cs = list(cells.ALL_CELLS)
    Mc = transfer.TransferMatrix(cs)
    for i in range(len(cs)):
        for j in range(len(cs)):
            Mc.set(i, j, 0.4 if i == j else 0.2)
    d = transfer.dissociation_scores(Mc.M, cs)
    check("transfer: the 18-cell crossing still computes both buckets",
          d["within_theme_across_env"] is not None
          and d["within_env_across_theme"] is not None)


def test_every_absolute_path_exists():
    """Nothing in paths.py may point at something that is not there.

    `SPACK_GCC_BIN` / `SPACK_CUDA_HOME` were hand-written during the survey and
    did not exist on disk (wrong spack tree, wrong hashes). Nothing noticed
    until nvcc was invoked deep inside DeepSpeed's JIT on a GPU node -- three
    launch rounds later. Constants that name a filesystem location are checkable
    in milliseconds; there is no excuse for finding out on hardware.
    """
    missing = [(n, v) for n, v, ok in paths.describe() if not ok]
    if missing:
        print("      MISSING: %s" % ", ".join("%s=%s" % (n, v) for n, v in missing))
    check("paths: every absolute constant exists (%d checked)"
          % len(paths.describe()), not missing)

    # The two that actually gate training, checked for the binaries we invoke.
    check("paths: SPACK_GCC_BIN has a C++17 g++",
          os.path.exists(os.path.join(paths.SPACK_GCC_BIN, "g++")))
    check("paths: SPACK_CUDA_HOME has nvcc (DeepSpeed CPUAdam JIT needs it)",
          os.path.exists(os.path.join(paths.SPACK_CUDA_HOME, "bin", "nvcc")))
    check("paths: the llamafactory env ships torchrun (FORCE_TORCHRUN resolves "
          "it via PATH, not via the absolute lmf path)",
          os.path.exists(os.path.join(os.path.dirname(paths.LMF), "torchrun")))


def test_status_gates_read_contents_not_existence(tmp):
    """The Phase-0 gate must not open just because a file is present.

    `verify_adapter e2e` writes adapter_verification.json even when it only
    PLANS the vLLM check, with an honest `blocking` entry. A gate that tests for
    existence would clear the one guard standing between an ignored adapter
    (4.5) and a table of numbers that silently describe the base model.

    Likewise the artefact counters: the dry-run modes emit `_gen_*.yaml` plans
    into out/cells, and counting directory entries would report a sweep as
    trained when nothing has run.
    """
    from adaptercl import cli

    # Own subtree: `tmp` is shared across every test in this file and earlier
    # tests already write adapters under tmp/cells, which would be counted here.
    root = os.path.join(tmp, "status_gates")
    real_out_eval = paths.OUT_EVAL
    fake_eval = os.path.join(root, "eval")
    os.makedirs(fake_eval)
    vpath = os.path.join(fake_eval, "adapter_verification.json")
    try:
        paths.OUT_EVAL = fake_eval

        with open(vpath, "w") as fh:
            json.dump({"ok": False, "checks": [], "blocking": [
                "vLLM check NOT RUN. Rerun with --vllm."]}, fh)
        passed, detail = cli.adapter_verification_state()
        check("status gate: a planned-but-unrun verification does NOT pass",
              passed is False)

        with open(vpath, "w") as fh:
            json.dump({"ok": True, "checks": [{"name": "vllm"}],
                       "blocking": ["something still wrong"]}, fh)
        passed, _ = cli.adapter_verification_state()
        check("status gate: ok=True with a non-empty blocking list does NOT pass",
              passed is False)

        with open(vpath, "w") as fh:
            json.dump({"ok": True, "checks": [{"name": "vllm"}], "blocking": []}, fh)
        passed, detail = cli.adapter_verification_state()
        check("status gate: ok=True and nothing blocking DOES pass",
              passed is True)

        with open(vpath, "w") as fh:
            fh.write("{not json")
        passed, detail = cli.adapter_verification_state()
        check("status gate: an unreadable artefact does NOT pass",
              passed is False)

        os.remove(vpath)
        passed, _ = cli.adapter_verification_state()
        check("status gate: an absent artefact does NOT pass", passed is False)

        # -- artefact counters -------------------------------------------
        fake_cells = os.path.join(root, "cells")
        os.makedirs(os.path.join(fake_cells, "percell", "yaml"))
        for cell in ("wiki_e2", "news_e3"):
            with open(os.path.join(fake_cells, "percell", "yaml",
                                   "_gen_%s.yaml" % cell), "w") as fh:
                fh.write("model_name_or_path: x\n")
        check("status counter: plan YAMLs are not counted as adapters",
              cli._count_adapters(fake_cells) == 0)

        trained = os.path.join(fake_cells, "percell", "wiki_e2")
        os.makedirs(trained)
        with open(os.path.join(trained, "adapter_config.json"), "w") as fh:
            json.dump({"r": 16}, fh)
        check("status counter: a real adapter dir IS counted",
              cli._count_adapters(fake_cells) == 1)

        fake_tr = os.path.join(root, "transfer")
        os.makedirs(fake_tr)
        with open(os.path.join(fake_tr, "pairs.todo"), "w") as fh:
            fh.write("wiki_e2__on__news_e3\n")
        with open(os.path.join(fake_tr, "notes.json"), "w") as fh:
            json.dump({"unrelated": True}, fh)
        check("status counter: todo files and stray JSON are not matrices",
              cli._count_matrices(fake_tr) == 0)
        with open(os.path.join(fake_tr, "m.json"), "w") as fh:
            json.dump({"cells": ["wiki_e2"], "matrix": [[1.0]]}, fh)
        check("status counter: a saved matrix IS counted",
              cli._count_matrices(fake_tr) == 1)
    finally:
        paths.OUT_EVAL = real_out_eval


def test_log_scrape_distinguishes_benign_from_real(tmp):
    """The 4.5 log gate must fail on OUR ignored modules and only those.

    Measured on grn075 2026-08-06: a healthy Qwen3.5-9B vLLM server with a
    working LoRA emits 111 `will be ignored` warnings, every one naming
    `visual.*` -- the vision tower that 8.2 excludes by construction -- plus the
    Inductor-compilation notice, which merely contains the words "will be
    ignored". A scraper that counts those fails every run on this architecture,
    and a gate that can never pass gets switched off, which is strictly worse
    than no gate: the real silent-adapter failure then walks straight through.

    So this pins both directions. The regression that matters is someone
    "simplifying" the classifier back to a substring match.
    """
    targets_ = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
    root = os.path.join(tmp, "logscrape")
    os.makedirs(root)

    def scrape(body, name):
        p = os.path.join(root, name + ".log")
        with open(p, "w") as fh:
            fh.write("INFO server up\n" + body)
        return verify_adapter.scrape_server_log(p, target_modules=targets_)

    VISION = ("WARNING no matching PunicaWrapper is found; "
              "visual.blocks.26.mlp.linear_fc1 will be ignored.\n")
    INDUCTOR = ("WARNING [vllm.py:1144] Inductor compilation was disabled by "
                "user settings, optimizations settings that are only active "
                "during inductor compilation will be ignored.\n")
    LM = ("WARNING no matching PunicaWrapper is found; "
          "model.language_model.layers.3.mlp.gate_proj will be ignored.\n")
    MISMATCH = ("WARNING While loading lora, expected target modules in "
                "['qkv_proj'] but got ['in_proj_a']; will be ignored\n")

    r = scrape(VISION * 27 + INDUCTOR, "benign")
    check("log scrape: vision-tower + Inductor warnings do NOT fail the gate",
          r["ok"] and not r["hits"] and r["n_benign"] == 28)
    check("log scrape: the benign ones are still reported, not swallowed",
          "28 ignored-module warning" in (r.get("note") or ""))

    r = scrape(VISION * 3 + LM, "lm")
    check("log scrape: a warning naming a TARGETED module DOES fail the gate",
          (not r["ok"]) and len(r["hits"]) == 1
          and r["hits"][0]["module"].endswith("mlp.gate_proj"))

    r = scrape(MISMATCH, "mismatch")
    check("log scrape: a PEFT-name mismatch still fails (4.5's headline case)",
          (not r["ok"]) and len(r["hits"]) == 1)

    # a module we never targeted is not our problem
    r = scrape("WARNING no matching PunicaWrapper is found; "
               "model.language_model.layers.3.linear_attn.in_proj_a "
               "will be ignored.\n", "untargeted")
    check("log scrape: an untargeted LM module is benign for THIS adapter",
          r["ok"] and r["n_benign"] == 1)

    check("log scrape: a missing log is NOT ok ('could not check' != 'fine')",
          not verify_adapter.scrape_server_log(
              os.path.join(root, "nope.log"), target_modules=targets_)["ok"])

    # Without a target list the scraper must stay conservative: any LM module
    # counts, because it cannot know which ones were ours.
    p = os.path.join(root, "notargets.log")
    with open(p, "w") as fh:
        fh.write("INFO up\n" + VISION + LM)
    r = verify_adapter.scrape_server_log(p)
    check("log scrape: with no target list, an LM module still fails",
          (not r["ok"]) and len(r["hits"]) == 1)

    # -- a planning run must never un-verify a verified system ---------------
    # On 2026-08-06 23:26 a CAPTURE=1/VERIFY=0 run of run_phase0.sh called
    # `verify e2e` to print commands, and that call overwrote a genuine ok=True
    # from two minutes earlier with ok=False/checks=[]. The gate re-closed with
    # no error and no evidence that anything had changed.
    vdir = os.path.join(root, "noclobber")
    os.makedirs(vdir)
    vjson = os.path.join(vdir, "adapter_verification.json")
    with open(vjson, "w") as fh:
        json.dump({"ok": True, "blocking": [],
                   "checks": [{"name": "vllm", "ok": True}]}, fh)
    _silent(verify_adapter.verify_end_to_end,
            adapter_dir=os.path.join(vdir, "probe1"),
            do_hf=False, do_vllm=False, out_json=vjson)
    with open(vjson) as fh:
        after = json.load(fh)
    check("verify: a run that checked NOTHING does not overwrite a passing verdict",
          after.get("ok") is True and len(after["checks"]) == 1)

    with open(vjson, "w") as fh:
        json.dump({"ok": False, "blocking": ["stale"], "checks": []}, fh)
    _silent(verify_adapter.verify_end_to_end,
            adapter_dir=os.path.join(vdir, "probe2"),
            do_hf=False, do_vllm=False, out_json=vjson)
    with open(vjson) as fh:
        after2 = json.load(fh)
    check("verify: a FAILING verdict is still replaceable (no write-once lock)",
          after2.get("ok") is False)


# --------------------------------------------------------------------------

def main():
    tmp = tempfile.mkdtemp(prefix="adaptercl_contracts_")
    try:
        test_site_order_is_stable_across_processes()
        test_materialize_roundtrip_every_target_set(tmp)
        test_rel_name_is_prefix_independent(tmp)
        test_cells_grid()
        test_tasks_partition_by_environment()
        test_transfer_matrix_roundtrip(tmp)
        test_dissociation_groups_on_era_not_theme()
        test_read_results_matches_the_harness(tmp)
        test_transfer_reads_evalbridge_output(tmp)
        test_adapter_dir_is_consumable(tmp)
        test_nothing_launches_by_default(tmp)
        test_probe_adapter_is_not_a_noop(tmp)
        test_split_csv_is_parsed_with_a_real_csv_reader()
        test_dissociation_refuses_a_version_matrix()
        test_every_absolute_path_exists()
        test_status_gates_read_contents_not_existence(tmp)
        test_log_scrape_distinguishes_benign_from_real(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n%d passed, %d failed, %d skipped" % (PASS[0], FAIL[0], SKIP[0]))
    sys.exit(1 if FAIL[0] else 0)


if __name__ == "__main__":
    main()
