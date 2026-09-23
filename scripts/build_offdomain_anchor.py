#!/usr/bin/env python3
"""Build one OFF-DOMAIN KL ANCHOR training corpus (ablation K, scout_task_plan.md 10.2/10.3).

For fold `h` (the held-out site) and source site `s` (one of the two remaining sites), this writes
one mixed ShareGPT corpus made of:

  * OWN rows -- CE rows -- EXACTLY the samples of the existing, already-built
    ``out/corpus/site6_4b/<s>_v6_d<d>.json`` (site `s`, era 6, draw `d`). Read verbatim off disk, not
    rebuilt: `site6_4b` already produced that draw with `adaptercl.bcdata.load_cell`'s seeded,
    whole-episode selection (`gen_yaml_site.py`), and re-deriving it here would risk a second,
    subtly different implementation of the same draw. Reading the registered file both reuses that
    function and IS the proof that these rows are exactly what site6_4b trained `s`'s own adapter
    on.
  * ANCHOR rows -- KL rows -- a SEEDED SUBSAMPLE of the OTHER source site's own matched-dose corpus
    (``out/corpus/site6_4b/<other>_v6_d<d>.json``), sized ``round(--anchor-ratio * n_own)``. `other`
    is fixed by the fold: BASE_SITES minus {h, s}.

Alongside the mixed corpus this writes the gamma side-car (TW_KL_GAMMA_WEIGHTS schema: gamma=0,
i.e. plain CE, exactly 0 on every own row; gamma=1, i.e. anchor-only, on every anchor row -- see
``data/processor/tw_token_weights.py``'s ``target_kl_gamma``/``GAMMA_DEFAULT_FLOOR`` for why 0 has
to be reachable, and ``require_supervision=False`` for why an all-0 gamma row does not trip the
"empty sample" guard the CE weighting table needs) and an ``.anchorstats.json`` provenance side-car.

SAFETY. Every own-site and other-site task id is checked against `h` via
``adaptercl.cells.task_site_group_map()`` and against its OWN expected site group, and the corpus
build refuses (does not silently drop rows) if either check fails: a leaked held-out sample would
not look like a crash, it would look like a slightly-better number.

COLLISION TRAP. Gamma side-car entries are keyed by the sha1 of the assistant turn TEXT. A turn
that is textually identical in an own row and an anchor row cannot carry both gamma values (0 and
1). This is detected, the colliding rows are dropped from the ANCHOR side (never the own side --
own rows are the fixed, exact reference draw), the count is reported, and >5% dropped is a hard
error (scout_task_plan.md 10.3).

    python scripts/build_offdomain_anchor.py --fold shop --site wiki --draw 1 --btag b0p1

Writes (under --out-root, default the project's real `out/`; NEVER pass the live root outside of a
real, GO=1'd K1 training run -- tests must pass --out-root and --site-bank-root pointing at a temp
dir):

    <out-root>/corpus_anc6_4b/<site>_v6_anc-<other>_<btag>_d<draw>.json           (mixed corpus)
    <out-root>/corpus_anc6_4b/<site>_v6_anc-<other>_<btag>_d<draw>.klgamma.json   (gamma side-car)
    <out-root>/corpus_anc6_4b/<site>_v6_anc-<other>_<btag>_d<draw>.anchorstats.json
    <out-root>/corpus_anc6_4b/dataset_info.json   (LLaMA-Factory dataset registration -- ITS OWN
        registry, deliberately NOT <out-root>/corpus/dataset_info.json: that file is shared with
        live trainings of another session, imported every few minutes, and K1's anchored corpora
        must never be registered into it (DOMSCOUT_KL_RUN_PLAN.md sec.2 K1 rules). gen_yaml_anchor.py
        points the generated yaml's `dataset_dir` at this same private directory so LLaMA-Factory
        resolves the dataset from here, not from the shared registry.)
"""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from adaptercl import bcdata, cells, paths  # noqa: E402

#: The three single-site source environments (never "multi"/"all" -- K1 only ever draws from a
#: single site's own single-site episodes on either side of the anchor).
BASE_SITES = tuple(cells.ENVIRONMENTS)  # ("wiki", "news", "shop")

#: beta tags -> TW_KL_BETA. "Never the dot-deleted 003 convention" (scout_task_plan.md 10.3) --
#: b0p1/b1p0 spell the decimal point as `p`, so a bare grep for "beta003" (which meant 0.03
#: elsewhere in this project and cost a real naming-trap postmortem) can never match a K1 tag.
# 2026-09-21: the sweep was 2-point (0.1, 1.0) and BOTH points said "higher is better"
# (matched-task: beta=0.1 is -12.1 held / -9.0 seen vs beta=1.0), so the optimum is not bracketed.
# b3p0 tests whether it keeps improving or turns over. NAMING: <b><int>p<frac> = int.frac --
# b0p3 is 0.3 and b3p0 is 3.0. NEVER a dot-deleted form ("003" for 0.03 cost a whole campaign
# its interpretation once: memory timewarp-kl-anchor-ablations).
BTAG_BETA = {"b0p1": 0.1, "b0p3": 0.3, "b1p0": 1.0, "b3p0": 3.0}

#: Fraction of ANCHOR rows a sha1 collision with an own row may drop before the build refuses
#: (scout_task_plan.md 10.3).
MAX_COLLISION_FRACTION = 0.05


class AnchorBuildError(Exception):
    """Raised for every condition scout_task_plan.md 10.3 says must be a hard error, never a
    silently-smaller or silently-null corpus."""


def _site_corpus_paths(bank_root, site, draw, era=6):
    base = os.path.join(bank_root, "%s_v%d_d%d" % (site, era, draw))
    return base + ".json", base + ".stats.json"


def _load_site_corpus(bank_root, site, draw, era=6):
    corpus_path, stats_path = _site_corpus_paths(bank_root, site, draw, era)
    if not os.path.isfile(corpus_path) or not os.path.isfile(stats_path):
        raise AnchorBuildError(
            "missing reference corpus for site=%r draw=%d under %s (want %s and %s) -- build it "
            "first, e.g. python scripts/gen_yaml_site.py --backbone 4b --tag site6_4b --eras 6 "
            "--units %s --draws %d" % (site, draw, bank_root, corpus_path, stats_path, site, draw)
        )
    with open(corpus_path) as fh:
        records = json.load(fh)
    with open(stats_path) as fh:
        stats = json.load(fh)
    if not isinstance(records, list) or not records:
        raise AnchorBuildError("reference corpus %s is empty or not a list" % corpus_path)
    if int(stats.get("n_samples", -1)) != len(records):
        raise AnchorBuildError(
            "stale side-car: %s says n_samples=%r but %s has %d records"
            % (stats_path, stats.get("n_samples"), corpus_path, len(records))
        )
    return records, stats, corpus_path, stats_path


def _assistant_content(record, source_path):
    """The single assistant ("gpt") turn's text. K1's corpora are one-turn-pair-per-record
    (adaptercl.bcdata's per-step ShareGPT convention); more than one, or none, means this script's
    one-content-per-record assumption does not hold for that file and it must not guess."""
    turns = record.get("conversations") or []
    gpt_turns = [t.get("value", "") for t in turns if t.get("from") in ("gpt", "assistant")]
    if len(gpt_turns) != 1:
        raise AnchorBuildError(
            "record in %s has %d assistant turns (want exactly 1) -- this corpus is not the "
            "one-step-per-record ShareGPT shape build_offdomain_anchor.py assumes"
            % (source_path, len(gpt_turns))
        )
    return gpt_turns[0]


def content_key(content):
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def _check_no_heldout(stats, site, held_out, task_data, label):
    """Every task id `site`'s corpus drew from must map to `site`'s OWN group, and never to
    `held_out`. Both are checked -- a task id assigned to the WRONG source site would be just as
    silent a corruption as a held-out leak, and both would read as a slightly-different number
    rather than a crash."""
    group_map = cells.task_site_group_map(task_data=task_data)
    bad_heldout, bad_group, unknown = [], [], []
    for tid in stats.get("task_ids", []):
        group = group_map.get(tid)
        if group is None:
            unknown.append(tid)
        elif group == held_out:
            bad_heldout.append(tid)
        elif group != site:
            bad_group.append((tid, group))
    if unknown:
        raise AnchorBuildError(
            "%s: %d task id(s) in %s's corpus are not in the task data at all: %r"
            % (label, len(unknown), site, unknown[:10])
        )
    if bad_heldout:
        raise AnchorBuildError(
            "%s: HELD-OUT SITE LEAK -- %d task id(s) from held-out site %r are in %r's corpus: %r"
            % (label, len(bad_heldout), held_out, site, bad_heldout[:10])
        )
    if bad_group:
        raise AnchorBuildError(
            "%s: %d task id(s) in %r's corpus belong to a DIFFERENT site group than expected: %r"
            % (label, len(bad_group), site, bad_group[:10])
        )


def build(fold, site, draw, btag, anchor_ratio=1.0, anchor_seed=None,
          bank_root=None, out_root=None, task_data=None, register=True):
    """Build one K1 anchored corpus. Returns the stats dict that was written to disk.

    Raises AnchorBuildError (never returns a partial/None result) for every condition
    scout_task_plan.md 10.3 calls out as a hard error.
    """
    if fold not in BASE_SITES:
        raise AnchorBuildError("--fold %r must be one of %r" % (fold, BASE_SITES))
    if site not in BASE_SITES:
        raise AnchorBuildError("--site %r must be one of %r" % (site, BASE_SITES))
    if site == fold:
        raise AnchorBuildError(
            "--site %r == --fold %r: the source site being trained can never be the held-out site"
            % (site, fold)
        )
    if btag not in BTAG_BETA:
        raise AnchorBuildError(
            "--btag %r is not a recognised beta tag (want one of %r); NEVER a dot-deleted form "
            "like '003' -- scout_task_plan.md 10.3 forbids that convention for this ablation"
            % (btag, sorted(BTAG_BETA))
        )
    draw = int(draw)
    others = [s for s in BASE_SITES if s not in (fold, site)]
    if len(others) != 1:
        raise AnchorBuildError(
            "expected exactly one OTHER source site for fold=%r site=%r, got %r"
            % (fold, site, others)
        )
    other = others[0]
    anchor_ratio = float(anchor_ratio)
    if anchor_ratio <= 0:
        raise AnchorBuildError("--anchor-ratio must be > 0, got %r" % anchor_ratio)
    anchor_seed = draw if anchor_seed is None else int(anchor_seed)

    bank_root = bank_root or os.path.join(paths.OUT_CORPUS, "site6_4b")
    out_root = out_root or paths.OUT

    own_records, own_stats, own_path, _ = _load_site_corpus(bank_root, site, draw)
    other_records, other_stats, other_path, _ = _load_site_corpus(bank_root, other, draw)

    _check_no_heldout(own_stats, site, fold, task_data, "own corpus (%s)" % own_path)
    _check_no_heldout(other_stats, other, fold, task_data, "other-source corpus (%s)" % other_path)

    n_own = len(own_records)
    n_other_full = len(other_records)
    n_anchor_target = int(round(anchor_ratio * n_own))
    if n_anchor_target <= 0:
        raise AnchorBuildError(
            "anchor-ratio %g x n_own %d rounds to %d anchor rows -- refusing to build an anchor "
            "corpus with no anchor rows" % (anchor_ratio, n_own, n_anchor_target)
        )
    # THE ERA-6 CORPORA ARE NOT THE SAME SIZE (measured 2026-09-19: news 128, wiki 257, shop 257),
    # so a 257-row own corpus cannot draw 257 DISTINCT anchor rows from news. The anchor:own ratio
    # is what fixes the optimizer-step structure (how often a KL micro-batch occurs, and how far the
    # CE gradient is diluted per step), so it is held at `anchor_ratio` for EVERY unit and the pool
    # is CYCLED instead: a seeded permutation of the whole pool, repeated until the target is met.
    # Anchor rows carry no CE, so a repeated anchor row is one more pull toward the base on the same
    # off-site input, never a second supervised pass. `n_anchor_unique` records the distinct count.
    rng = random.Random(anchor_seed)
    if n_anchor_target <= n_other_full:
        idx = sorted(rng.sample(range(n_other_full), n_anchor_target))
    else:
        idx = []
        while len(idx) < n_anchor_target:
            perm = list(range(n_other_full)); rng.shuffle(perm)
            idx.extend(perm[: n_anchor_target - len(idx)])
    anchor_candidates = [other_records[i] for i in idx]
    n_anchor_unique_candidates = len(set(idx))

    own_keys = set()
    for rec in own_records:
        own_keys.add(content_key(_assistant_content(rec, own_path)))

    kept, dropped_keys = [], []
    for rec in anchor_candidates:
        key = content_key(_assistant_content(rec, other_path))
        if key in own_keys:
            dropped_keys.append(key)
        else:
            kept.append((key, rec))

    if anchor_candidates:
        collision_rate = len(dropped_keys) / float(len(anchor_candidates))
    else:
        collision_rate = 0.0
    if collision_rate > MAX_COLLISION_FRACTION:
        raise AnchorBuildError(
            "%d/%d (%.1f%%) candidate anchor rows collide (sha1-identical assistant turn) with an "
            "own row -- exceeds the %.0f%% hard limit (scout_task_plan.md 10.3). Sample colliding "
            "sha1s: %r" % (len(dropped_keys), len(anchor_candidates), 100.0 * collision_rate,
                            100.0 * MAX_COLLISION_FRACTION, dropped_keys[:5])
        )

    n_anchor = len(kept)
    if n_anchor <= 0:
        raise AnchorBuildError(
            "every candidate anchor row collided with an own row (%d/%d) -- the anchor corpus "
            "would be empty" % (len(dropped_keys), len(anchor_candidates))
        )

    mixed = list(own_records) + [rec for _key, rec in kept]

    gamma_entries = {}
    for rec in own_records:
        content = _assistant_content(rec, own_path)
        key = content_key(content)
        gamma_entries[key] = {"len": len(content), "default_weight": 0.0, "envelope_weight": 0.0}
    for key, rec in kept:
        content = _assistant_content(rec, other_path)
        gamma_entries[key] = {"len": len(content), "default_weight": 1.0, "envelope_weight": 1.0}

    unit_key = "%s_v6_anc-%s_%s_d%d" % (site, other, btag, draw)
    dataset_name = "adaptercl_anc6_4b_%s" % unit_key
    # DELIBERATELY NOT out_root/corpus/anc6_4b: out_root/corpus/ is the SHARED registry another
    # session's live trainings import every few minutes. K1's own corpora + gamma side-cars +
    # dataset_info.json live in their OWN top-level directory, out_root/corpus_anc6_4b/, so this
    # build (and its dataset registration below) never touches the shared file.
    corpus_dir = os.path.join(out_root, "corpus_anc6_4b")
    if not os.path.isdir(corpus_dir):
        os.makedirs(corpus_dir)
    corpus_path = os.path.join(corpus_dir, unit_key + ".json")
    gamma_path = os.path.join(corpus_dir, unit_key + ".klgamma.json")
    stats_out_path = os.path.join(corpus_dir, unit_key + ".anchorstats.json")

    with open(corpus_path, "w") as fh:
        json.dump(mixed, fh, indent=2, sort_keys=True)
    with open(gamma_path, "w") as fh:
        json.dump(
            # "floor": 0.0 is pinned HERE so an own row's gamma of exactly 0 never depends on the
            # trainer's environment (the reader's default floor is 0.1 unless TW_KL_DISJOINT=1).
            {"version": 2, "dataset": unit_key, "kl_role_mask": "offdomain_anchor",
             "floor": 0.0, "entries": gamma_entries},
            fh, indent=2, sort_keys=True,
        )

    stats_out = {
        "unit": unit_key, "dataset_name": dataset_name,
        "fold": fold, "site": site, "other": other, "draw": draw, "btag": btag,
        "beta": BTAG_BETA[btag], "anchor_ratio": anchor_ratio, "anchor_seed": anchor_seed,
        "n_own": n_own, "n_anchor": n_anchor, "n_anchor_candidates": len(anchor_candidates),
        "n_anchor_unique_candidates": n_anchor_unique_candidates,
        "anchor_pool_size": n_other_full, "anchor_pool_cycled": bool(n_anchor_target > n_other_full),
        "n_collisions_dropped": len(dropped_keys), "collision_rate": collision_rate,
        "collision_sha1s": dropped_keys, "n_samples": len(mixed),
        "own_corpus": own_path, "other_corpus": other_path,
        "own_task_ids": own_stats.get("task_ids", []), "other_task_ids": other_stats.get("task_ids", []),
        "corpus_json": corpus_path, "gamma_json": gamma_path,
    }
    with open(stats_out_path, "w") as fh:
        json.dump(stats_out, fh, indent=2, sort_keys=True)

    if register:
        # OWN registry (see the module docstring / REGISTRY ISOLATION comment above): never
        # out_root/corpus/dataset_info.json, which is shared with another session's live
        # trainings. Assert the path we are about to open for writing is not that shared file,
        # as a second, structural guard beside the string literal itself.
        dataset_info_path = os.path.join(out_root, "corpus_anc6_4b", "dataset_info.json")
        shared_dataset_info = os.path.join(out_root, "corpus", "dataset_info.json")
        assert os.path.abspath(dataset_info_path) != os.path.abspath(shared_dataset_info), (
            "REGISTRY ISOLATION VIOLATION: about to write the shared dataset_info.json (%s)"
            % shared_dataset_info
        )
        bcdata.register_dataset(dataset_name, corpus_path, dataset_info=dataset_info_path)
    return stats_out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", required=True, choices=BASE_SITES, help="the held-out site")
    ap.add_argument("--site", required=True, choices=BASE_SITES,
                    help="the source site this component is trained for")
    ap.add_argument("--draw", required=True, type=int, help="episode draw / trainer seed offset")
    ap.add_argument("--btag", required=True, choices=sorted(BTAG_BETA),
                    help="beta tag: %s" % ", ".join("%s=%g" % kv for kv in sorted(BTAG_BETA.items())))
    ap.add_argument("--anchor-ratio", type=float, default=1.0,
                    help="anchor rows = round(ratio * n_own); default 1.0 (equal number)")
    ap.add_argument("--anchor-seed", type=int, default=None,
                    help="seed for the anchor subsample; default = --draw")
    ap.add_argument("--site-bank-root", default=None,
                    help="where site6_4b's per-site corpora live "
                         "(default out/corpus/site6_4b under --out-root's project)")
    ap.add_argument("--out-root", default=None,
                    help="project 'out' directory to write under (default: the real out/ -- "
                         "tests MUST override this)")
    ap.add_argument("--task-data", default=None,
                    help="override the task list used for the held-out-site safety check "
                         "(default: the real TimeWarp task data)")
    ap.add_argument("--no-register", action="store_true",
                    help="skip writing the LLaMA-Factory dataset_info.json registration")
    args = ap.parse_args(argv)

    out_root = args.out_root or paths.OUT
    bank_root = args.site_bank_root or os.path.join(out_root, "corpus", "site6_4b")

    try:
        stats = build(
            fold=args.fold, site=args.site, draw=args.draw, btag=args.btag,
            anchor_ratio=args.anchor_ratio, anchor_seed=args.anchor_seed,
            bank_root=bank_root, out_root=out_root, task_data=args.task_data,
            register=not args.no_register,
        )
    except AnchorBuildError as exc:
        print("BLOCKED: %s" % exc, file=sys.stderr)
        return 1

    print("=== %s ===" % stats["unit"])
    print("fold=%s site=%s other=%s draw=%d btag=%s (beta=%g)"
          % (stats["fold"], stats["site"], stats["other"], stats["draw"], stats["btag"],
             stats["beta"]))
    print("n_own=%d n_anchor=%d (candidates=%d, dropped %d collisions = %.2f%%)"
          % (stats["n_own"], stats["n_anchor"], stats["n_anchor_candidates"],
             stats["n_collisions_dropped"], 100.0 * stats["collision_rate"]))
    print("corpus : %s" % stats["corpus_json"])
    print("gamma  : %s" % stats["gamma_json"])
    print("dataset: %s" % stats.get("dataset_name"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
