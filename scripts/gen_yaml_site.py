#!/usr/bin/env python3
"""Build the SITE corpora and per-site LoRA YAMLs -- the domain axis.

`domain_transfer_timewarp_plan.md` asks the question orthogonal to FINDINGS §2.
Era adapters transfer flat to every other era, so they learn an output protocol
rather than interface knowledge. Wiki, News and Shop differ in *function*
(search-and-read, browse-a-feed, search-cart-checkout), so if site adapters also
transfer flat then adapters on this benchmark carry no domain knowledge on
either axis. This script produces the arms that decide it.

Everything except the training UNIT is copied from the published per-version
`pv` recipe -- LoRA r=16 / alpha=32 on `attn_mlp`, lr 1e-4, 6 epochs, cutoff
65536, per-device batch 1 x grad-accum 4 (effective batch 4) -- so a difference
between the site arms and the era arms is attributable to the corpus.

Three design points that are NOT incidental:

* **Dose is matched on SAMPLES, not episodes.** Steps per episode differ by 1.75x
  across the units (multi ~9.3, news ~5.3, wiki ~6.3), so equal episode counts
  hand `multi` 1.6x the dose and the matrix would measure corpus size, not
  domain (the plan's first trap, and FINDINGS §4/§10/§13's repeated lesson).
  `--max-samples` cuts on samples; the default 770 is news's whole corpus, the
  smallest of the three single-site units.
* **A seed is a full replicate.** `--draws 1,2,3` moves the episode DRAW *and*
  the trainer seed together, because a "seed" that re-draws nothing is a
  replication of one corpus wearing three names -- the failure that invalidated
  an earlier TimeWarp comparison. The draw lands in the unit key (`wiki_d1`), so
  the adapter directory says which draw trained it.
* **The dose lives in the TAG, not in the unit key.** Matched and full-corpus
  arms are the same units at different doses, so they go to different tags
  (`site4b` vs `sitefull4b`) and the unit keys stay comparable across them.

Leak guard, run on what was WRITTEN and not on what was intended (the rule from
gen_yaml_holdout.py):

  1. no episode in any corpus belongs to a TEST-split task;
  2. every YAML's `dataset:` line names only its own unit's corpus;
  3. `dataset_dir` is adapterCL's own registry, never LLaMA-Factory's.

Teacher episodes only exist for train tasks, so (1) should be vacuous -- which
is the point of asserting it: if it ever fires, the trajectory corpus changed.

    # the 5 matched arms x 3 draws  (Phase A)
    python scripts/gen_yaml_site.py --backbone 4b --tag site4b \\
        --units wiki,news,shop,multi,all --draws 1,2,3 --max-samples 770

    # the full-dose controls (Phase B): same units, no sample cap
    python scripts/gen_yaml_site.py --backbone 4b --tag sitefull4b \\
        --units wiki,shop,multi --draws 1,2,3 --max-samples 0

    # cost/plan table only, touching nothing
    python scripts/gen_yaml_site.py --backbone 4b --plan-only

    # LEAVE-ONE-SITE-OUT on one era (LOSO_HANDOFF.md): composite units, era 6
    # only, dose matched to the smallest arm of THIS set (auto -> 385 samples)
    python scripts/gen_yaml_site.py --backbone 4b --tag loso6_4b --eras 6 \\
        --units wiki+news,news+shop,wiki+shop,wiki+news+shop,all \\
        --draws 1,2,3 --max-samples auto

Then:  TAG=site4b GO=1 SLOTS="<jobid>:<gpu> ..." bash scripts/run_site_train.sh

Composite units (`wiki+news`) are the union of those sites' SINGLE-site
episodes; `--eras 6` restricts every unit to era 6 and puts `_v6` in the unit
key (`wiki+news_v6_d1`), so a one-era arm can never be mistaken for a pooled
one on disk. `--max-samples auto` reads the matched dose off the census of the
requested units rather than assuming news's 770: on era 6 alone the smallest
LOSO corpus is news+shop at 385 samples.
"""
from __future__ import print_function

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from adaptercl import bcdata, cells, paths, percell, targets  # noqa: E402

#: The `pv` recipe, verbatim. Named here so a drift is a one-line diff.
PV = {"rank": 16, "alpha": 32.0, "target_set": "attn_mlp", "epochs": 6.0,
      "lr": "1.0e-4", "cutoff": 65536, "grad_accum": 4}

BACKBONE = {"4b": "Qwen/Qwen3.5-4B", "9b": "Qwen/Qwen3.5-9B",
            "l8": "meta-llama/Llama-3.1-8B-Instruct"}

#: INSTRUCT, not base, for Llama: the base weights cannot follow the chat +
#: structured-output format browser-use requires, so that row would measure
#: formatting rather than the web. Same reason the zero-shot baseline used it.
#: NOTE Qwen3.5-9B and Llama-3.1-8B-Instruct BOTH have hidden_size 4096 and 32
#: layers, so a `lora_A` shape of (16, 4096) does NOT tell them apart -- check
#: `base_model_name_or_path` in adapter_config.json instead.

#: Measured throughput per backbone, samples/s on one card (plan §5). Used only
#: for the cost table, so a wrong value misprices a plan -- it never changes a
#: run. Re-measure from a real training log and update here, in one place.
SAMPLES_PER_SEC = {"4b": 0.41, "9b": 0.22, "l8": 0.24}

#: news's whole verified corpus -- the smallest single-site unit, hence the
#: matched dose. Recomputed by --plan-only so a stale value is visible.
DEFAULT_MAX_SAMPLES = 770


def optim_steps(n_samples, epochs=PV["epochs"], grad_accum=PV["grad_accum"],
                gpus=1):
    """Optimizer steps for one arm.

    run_site_train.sh puts ONE adapter on ONE card (NPROC_PER_NODE=1), so the
    effective batch is per_device_train_batch_size(1) x grad_accum, and the
    plan's "effective batch 4" holds only at gpus=1. If an arm is ever trained
    on N cards, grad_accum must become 4/N or the arm is not step-matched to the
    `pv` reference row.
    """
    return int(math.ceil(n_samples * float(epochs) / max(1, grad_accum * gpus)))


def gpu_hours(n_samples, backbone, epochs=PV["epochs"]):
    rate = SAMPLES_PER_SEC.get(backbone, SAMPLES_PER_SEC["4b"])
    return n_samples * float(epochs) / rate / 3600.0


def unit_keys(units, draws, eras=None):
    for d in draws:
        for u in units:
            yield cells.Site(u, draw=d, eras=eras)


def census_for(units, eras=None):
    """The census restricted to exactly the requested units and eras.

    Keyed by the units' BARE keys (no draw), so `av["wiki+news_v6"]` for an
    era-restricted composite and `av["wiki"]` for the classic pooled arm. The
    membership rule is `Site.accepts`, the same one the corpus builder uses.
    """
    return cells.site_availability(
        units=[cells.Site(u, eras=eras) for u in units])


def resolve_max_samples(arg, units, eras=None):
    """'auto' -> the smallest verified corpus among the requested units;
    '0' -> None (full corpus); anything else -> int."""
    if str(arg).strip().lower() == "auto":
        av = census_for(units, eras)
        return int(av["_matched_samples"]) or None
    n = int(arg)
    return n or None


def plan_table(units, draws, max_samples, backbone, eras=None, fh=None,
               epochs_of=None):
    """Print what each arm would be, from the census -- no corpus is built."""
    fh = fh or sys.stdout
    # A step-matched ladder gives each unit its OWN epoch ceiling, so a cost
    # table printed at a single --epochs is wrong by up to 6x on this campaign.
    epochs_of = epochs_of or (lambda _u: PV["epochs"])
    av = census_for(units, eras)
    era_txt = ("eras %s" % ",".join(str(e) for e in av["_eras"])
               if eras else "eras 1-6")
    print("site census (verified episodes / verified BC samples, pooled over "
          "%s):" % era_txt, file=fh)
    print("  %-20s %9s %9s %8s   %s" % ("unit", "episodes", "samples", "st/ep",
                                        "at the matched dose"), file=fh)
    total_h = 0.0
    bare = [cells.Site(u, eras=eras).key for u in units]
    for u, k in zip(units, bare):
        s = av[k]
        n_full = s["verified_steps"]
        n = n_full if not max_samples else min(n_full, int(max_samples))
        note = ("full corpus" if not max_samples
                else "%d samples (%.0f%% of full)" % (n, 100.0 * n / max(1, n_full)))
        ep = epochs_of(k)
        print("  %-20s %9d %9d %8.2f   %-24s %4.0f ep  %5d steps"
              % (k, s["verified"], n_full, s["steps_per_episode"], note,
                 ep, optim_steps(n, ep)), file=fh)
        total_h += len(draws) * gpu_hours(n, backbone, ep)
    print("", file=fh)
    smallest = min(bare, key=lambda k: av[k]["verified_steps"])
    print("  matched dose available from the census: %d samples "
          "(%s, the smallest requested corpus)" % (av["_matched_samples"],
                                                    smallest), file=fh)
    if max_samples and int(max_samples) > av["_matched_samples"]:
        print("  WARNING: --max-samples %d exceeds the smallest corpus (%d); "
              "%s cannot reach it and the arms are NOT dose-matched."
              % (int(max_samples), av["_matched_samples"], smallest), file=fh)
    n_ref = (int(max_samples) if max_samples
             else max(av[k]["verified_steps"] for k in bare))
    print("  reference arm: ~%d optimizer steps (%d samples x %g epochs / "
          "effective batch %d), ~%.1f GPU-h at %.2f samples/s"
          % (optim_steps(n_ref), n_ref, PV["epochs"], PV["grad_accum"],
             gpu_hours(n_ref, backbone), SAMPLES_PER_SEC[backbone]), file=fh)
    print("  %d unit(s) x %d draw(s) = %d adapters, ~%.0f GPU-h total"
          % (len(units), len(draws), len(units) * len(draws), total_h), file=fh)
    return total_h


def leak_check_tasks(stats_by_unit, fh=None):
    """Refuse if any corpus contains a TEST-split task.

    Teacher episodes exist only for train tasks, so this is a tripwire on the
    trajectory corpus rather than on this script: `gen_yaml_holdout.py`'s lesson
    is that the check has to read what was WRITTEN, and the written thing here
    is the `.stats.json` side-car's task_ids.
    """
    fh = fh or sys.stdout
    split = cells.load_split()
    bad = []
    for key, st in stats_by_unit.items():
        for tid in st.get("task_ids", []):
            if split.get(tid) == "test":
                bad.append((key, tid))
    if bad:
        for key, tid in bad:
            print("LEAK: corpus %s contains TEST task %d" % (key, tid), file=fh)
        raise SystemExit("refusing: %d test task(s) in the training corpora"
                         % len(bad))
    n = sum(len(st.get("task_ids", [])) for st in stats_by_unit.values())
    print("leak check 1/3: OK -- %d corpus task id(s) across %d unit(s), none "
          "in the test split" % (n, len(stats_by_unit)), file=fh)


def leak_check_yaml(made, fh=None):
    """Refuse if a YAML's dataset line names anything but its own corpus."""
    fh = fh or sys.stdout
    bad = []
    for key, path in made:
        want = None
        lines = open(path).read().splitlines()
        for ln in lines:
            if ln.startswith("dataset:"):
                want = ln.split(":", 1)[1].strip()
            if ln.startswith("dataset_dir:"):
                d = ln.split(":", 1)[1].strip().split("#")[0].strip()
                if os.path.abspath(d) != os.path.abspath(
                        os.path.dirname(bcdata.local_dataset_info())):
                    bad.append((key, "dataset_dir is %s, not adapterCL's own "
                                     "registry" % d))
        if want is None:
            bad.append((key, "no dataset: line"))
        elif "," in want:
            bad.append((key, "dataset line pools %r -- a site arm trains on "
                             "exactly one corpus" % want))
        elif not want.endswith("_" + key):
            bad.append((key, "dataset %r does not belong to unit %s"
                        % (want, key)))
    if bad:
        for key, why in bad:
            print("LEAK: %s -- %s" % (key, why), file=fh)
        raise SystemExit("refusing: %d bad dataset line(s)" % len(bad))
    print("leak check 2/3: OK -- each of %d YAML(s) names exactly its own "
          "corpus" % len(made), file=fh)
    print("leak check 3/3: OK -- every dataset_dir is adapterCL's own registry",
          file=fh)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="4b", choices=sorted(BACKBONE))
    ap.add_argument("--model", default=None,
                    help="override the backbone id (default: from --backbone)")
    ap.add_argument("--tag", default=None,
                    help="default site<backbone> for a matched run, "
                         "sitefull<backbone> when --max-samples is 0")
    ap.add_argument("--units", default=",".join(cells.SITE_UNITS),
                    help="comma-separated site units: %s, or a composite of "
                         "single sites joined by '+' (e.g. wiki+news = the "
                         "leave-shop-out corpus)"
                         % "|".join(cells.SITE_UNITS))
    ap.add_argument("--eras", default=None,
                    help="comma-separated eras to restrict EVERY unit to "
                         "(e.g. 6). Default: pooled over all six. Puts "
                         "_v<eras> in the unit key.")
    ap.add_argument("--draws", default="1,2,3",
                    help="comma-separated episode draws; each is one seed "
                         "(the draw AND the trainer seed move together)")
    ap.add_argument("--max-samples", default=str(DEFAULT_MAX_SAMPLES),
                    help="dose cap in BC samples; 0 = the whole corpus "
                         "(the Phase-B full-dose controls); auto = the "
                         "smallest requested corpus from the census "
                         "(default %d, news's pooled corpus)"
                         % DEFAULT_MAX_SAMPLES)
    ap.add_argument("--seed-base", type=int, default=42,
                    help="trainer seed for draw d is seed-base + d")
    ap.add_argument("--rank", type=int, default=PV["rank"])
    ap.add_argument("--alpha", type=float, default=PV["alpha"])
    ap.add_argument("--target-set", default=PV["target_set"])
    ap.add_argument("--epochs", type=float, default=PV["epochs"])
    ap.add_argument("--grad-accum", type=int, default=PV["grad_accum"],
                    help="per-device batch is 1, so this IS the effective "
                         "batch at one GPU per adapter. On N cards pass 4/N or "
                         "the arm is not step-matched to the pv row.")
    ap.add_argument("--epochs-by-unit", default=None,
                    help="per-unit epoch ceiling, e.g. "
                         "'news=36,shop=18,wiki=18'. Keys match the unit's "
                         "SITE (no era/draw suffix). Units not named keep "
                         "--epochs. This exists because a step-matched ladder "
                         "needs a different epoch count per corpus size -- "
                         "spelling it in one call keeps ONE manifest for the "
                         "tag instead of each call overwriting the last.")
    ap.add_argument("--template", default=None,
                    help="LLaMA-Factory chat template; default is resolved "
                         "from the model id by percell.template_for(), which "
                         "REFUSES an unknown family rather than guessing")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the census and cost table; build nothing")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild corpora that already exist on disk")
    args = ap.parse_args(argv)

    units = []
    for u in args.units.split(","):
        u = u.strip()
        if not u:
            continue
        try:
            units.append(cells.Site(u).site)      # canonicalises composites
        except ValueError as exc:
            raise SystemExit("unknown site unit %r: %s" % (u, exc))
    eras = None
    if args.eras:
        eras = tuple(sorted(set(int(e) for e in
                                args.eras.replace(" ", ",").split(",") if e)))
        bad = [e for e in eras if e not in cells.ERAS]
        if bad:
            raise SystemExit("bad --eras %r (want a subset of %r)"
                             % (args.eras, cells.ERAS))
    draws = [int(d) for d in args.draws.split(",") if d.strip()]
    epochs_by_unit = {}
    for kv in (args.epochs_by_unit or "").replace(" ", "").split(","):
        if not kv:
            continue
        k, _, v = kv.partition("=")
        if not _:
            raise SystemExit("bad --epochs-by-unit %r (want site=epochs)" % kv)
        try:
            epochs_by_unit[cells.Site(k).site] = float(v)
        except ValueError as exc:
            raise SystemExit("bad --epochs-by-unit key %r: %s" % (k, exc))
    unknown = [k for k in epochs_by_unit if k not in units]
    if unknown:
        raise SystemExit("--epochs-by-unit names %r, which is not in --units %r"
                         % (unknown, units))

    def epochs_of(unit):
        """Epoch ceiling for one unit key, by its SITE (draw/era stripped)."""
        return epochs_by_unit.get(cells.Site.parse(unit).site, args.epochs)
    max_samples = resolve_max_samples(args.max_samples, units, eras)
    model = args.model or BACKBONE[args.backbone]
    era_suffix = ("_v" + "".join(str(e) for e in eras)) if eras else ""
    tag = args.tag or ((("site%s" % args.backbone) if max_samples
                        else ("sitefull%s" % args.backbone)) + era_suffix)

    print("=== site arms: %s ===" % tag)
    print("backbone   : %s" % model)
    print("units      : %s" % ", ".join(units))
    print("eras       : %s" % (",".join(str(e) for e in eras) if eras
                               else "1-6 (pooled)"))
    print("draws      : %s (trainer seed = %d + draw)"
          % (", ".join(str(d) for d in draws), args.seed_base))
    print("dose       : %s"
          % (("%d samples (matched%s)"
              % (max_samples, ", from the census"
                 if str(args.max_samples).lower() == "auto" else ""))
             if max_samples
             else "full corpus (NOT dose-matched -- a Phase-B control)"))
    print("recipe     : r=%d alpha=%g %s lr=%s %g epochs cutoff=%d "
          "effective batch %d"
          % (args.rank, args.alpha, args.target_set, PV["lr"], args.epochs,
             PV["cutoff"], args.grad_accum))
    print("")
    plan_table(units, draws, max_samples, args.backbone, eras=eras,
               epochs_of=epochs_of)
    print("")
    if args.plan_only:
        print("--plan-only: nothing built. Drop the flag to write the corpora "
              "and YAMLs.")
        return 0

    cfg = targets.load_text_config(model)
    sites = targets.enumerate_sites(cfg, args.target_set)
    print("adapter    : %d layers, hidden=%d, %d sites, %s params"
          % (cfg["num_hidden_layers"], cfg["hidden_size"], len(sites),
             "{:,}".format(targets.site_budget(sites, args.rank))))
    print("")

    corpus_dir = os.path.join(paths.OUT_CORPUS, tag)
    if not os.path.isdir(corpus_dir):
        os.makedirs(corpus_dir)

    stats_by_unit, made = {}, []
    for u in unit_keys(units, draws, eras):
        out_json = os.path.join(corpus_dir, "%s.json" % u.key)
        dataset = "adaptercl_%s_%s" % (tag, u.key)
        sc = bcdata._stats_path(out_json)
        if os.path.exists(out_json) and os.path.exists(sc) and not args.rebuild:
            st = json.load(open(sc))
            print("  reuse %-12s %5d samples / %3d episodes"
                  % (u.key, st["n_samples"], st["n_episodes"]))
        else:
            st = bcdata.build_sharegpt(
                [u], out_json, dataset_name=dataset,
                max_samples=max_samples, draw_seed=u.draw)
            print("  built %-12s %5d samples / %3d episodes"
                  % (u.key, st["n_samples"], st["n_episodes"]))
        stats_by_unit[u.key] = st

        y = percell.write_cell_yaml(
            u, st.get("dataset_name") or dataset, target_set=args.target_set,
            rank=args.rank, alpha=args.alpha, tag=tag, model=model,
            epochs=epochs_of(u.key), grad_accum=args.grad_accum, lr=PV["lr"],
            cutoff=PV["cutoff"], n_samples=st["n_samples"],
            template=args.template,
            seed=args.seed_base + (u.draw or 0))
        made.append((u.key, y))

    print("")
    leak_check_tasks(stats_by_unit)
    leak_check_yaml(made)

    print("")
    print("%-12s %8s %9s %7s %6s   %s"
          % ("unit", "samples", "episodes", "steps", "seed", "adapter"))
    manifest = {"tag": tag, "model": model, "units": units, "draws": draws,
                "eras": list(eras) if eras else list(cells.ERAS),
                "max_samples": max_samples, "recipe": dict(PV, rank=args.rank,
                                                           alpha=args.alpha,
                                                           epochs=args.epochs),
                "arms": {}}
    for key, y in made:
        st = stats_by_unit[key]
        out_dir = percell.cell_output_dir(cells.Site.parse(key), tag)
        steps = optim_steps(st["n_samples"], epochs_of(key), args.grad_accum)
        print("%-12s %8d %9d %7d %6d   %s"
              % (key, st["n_samples"], st["n_episodes"], steps,
                 args.seed_base + (cells.Site.parse(key).draw or 0),
                 os.path.relpath(out_dir, PROJECT)))
        manifest["arms"][key] = {
            "yaml": y, "adapter": out_dir, "dataset": st.get("dataset_name"),
            "corpus": st["out_json"], "n_samples": st["n_samples"],
            "n_episodes": st["n_episodes"], "optim_steps": steps,
            "epochs": epochs_of(key),
            "seed": args.seed_base + (cells.Site.parse(key).draw or 0)}

    # Dose spread across the arms. Whole episodes are never split, so a matched
    # arm can undershoot by up to one episode; anything worse than a couple of
    # percent must be reported with the results, not silently averaged.
    ns = [stats_by_unit[k]["n_samples"] for k, _ in made]
    if ns and max_samples:
        print("")
        print("dose spread: %d-%d samples (%.1f%% of the %d requested at "
              "worst) -- whole episodes only, never split"
              % (min(ns), max(ns), 100.0 * min(ns) / max_samples, max_samples))
        # A unit whose full corpus is at (or below) the matched dose has nothing
        # to draw FROM: its three "seeds" are the same episodes with a different
        # trainer seed. That is a legitimate replicate but a narrower one, and
        # reading it as a corpus-draw replicate would overstate the design.
        av = census_for(units, eras)
        narrow = [k for k in (cells.Site(u, eras=eras).key for u in units)
                  if av[k]["verified_steps"] <= 1.02 * max_samples]
        if narrow:
            print("NOTE: %s ha%s no headroom above the matched dose (full "
                  "corpus <= 1.02x %d), so its draws differ by at most one "
                  "episode: those seeds vary the TRAINER seed, not the corpus. "
                  "Say so wherever the seed SE is reported."
                  % (", ".join(narrow), "ve" if len(narrow) > 1 else "s",
                     max_samples))

    mpath = os.path.join(PROJECT, "out", "cells", tag, "manifest.json")
    if not os.path.isdir(os.path.dirname(mpath)):
        os.makedirs(os.path.dirname(mpath))
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print("")
    print("manifest: %s" % os.path.relpath(mpath, PROJECT))
    print("next: TAG=%s UNITS=\"%s\" GO=1 SLOTS=\"<jobid>:<gpu> ...\" "
          "bash scripts/run_site_train.sh"
          % (tag, " ".join(k for k, _ in made)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
