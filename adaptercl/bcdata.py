"""Per-cell BC corpus layer over TimeTraj-Trajectories (adapterCL.md 4.3/4.4/6.1).

Everything the project trains on -- per-cell LoRAs (7 Phase 2) and the
hypernetwork (7 Phase 3) -- comes from the same GPT-5 teacher rollouts sitting in
`TimeWarp/TimeTraj-Trajectories/version_{1..6}/`. This module is the single
reader for that corpus: it enumerates episodes per (environment, era) cell,
converts a selected set of them to LLaMA-Factory ShareGPT, and records what went
into each corpus.

Three facts here are load-bearing and none of them are in the plan:

1. **Era 1 cannot support the 18-cell design.** `wiki_e1` has *zero* episodes and
   `shop_e1` has *two*; every other cell has 25-39. `census()` returns a
   `blocking` list and `print_census()` shouts about it. Phase 2's "18 cells" is
   16 cells until era-1 rollouts are collected (see the report printed by
   `python -m adaptercl.bcdata census`).
2. **There are no screenshots on disk.** The TimeTraj dumps contain
   `system_prompt/user_prompt/agent_output` JSON+TXT only -- no `*.png` anywhere
   under `version_*/`. So `convert2sgptArgs.py --img/--som` cannot be used on this
   corpus, and the 4.2 *vision* conditioning signal has to come from a separate
   capture pass (paths.OUT_CAPTURE), not from the trajectories.
   `attach_conditioning()` reports which episodes actually resolved to an image
   and never silently substitutes one.
3. **Filtered BC is a no-op on the teacher data.** 606/660 -> actually every
   scanned episode with a reward has `cum_reward == 1.0`; the teacher rollouts
   were already filtered upstream. `filtered_bc_corpus()` is therefore only
   interesting on *self-generated* rollouts (7 Phase 4) and it prints the verify
   rate so open question 11.5 gets a number rather than an opinion.

Corpora are produced by shelling out to the repo's own converter
(`dataCreationScripts/convert2sgptArgs.py`, paths.CONVERT_SGPT) over a staging
directory of symlinks, so an adapterCL corpus is byte-identical to what every
other TimeWarp experiment produces. We never re-implement the ShareGPT emitter.

Step-level parsing is delegated to `self-correct/sc_traj.py` (loaded by path --
it is not an importable package), whose tag grammar already mirrors
convert2sgptArgs.parse_content_sections.

Pure stdlib -- importable from the system python (3.6+). numpy is optional and
only used for percentiles in the census.
"""

from __future__ import print_function

import argparse
import collections
import datetime
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys

from . import cells, paths

# --------------------------------------------------------------------------
# sc_traj: the repo's env-free trajectory reader
# --------------------------------------------------------------------------

#: self-correct/ is a script directory, not a package, so we load the module by
#: path instead of copying its parser. Everything below relies on its tag
#: grammar (sc_traj.py:34-40) matching convert2sgptArgs.parse_content_sections
#: (dataCreationScripts/convert2sgptArgs.py:52-80) -- that equivalence is stated
#: in sc_traj.py's own docstring and is why our step counts agree with the
#: converter's sample counts.
SC_TRAJ_PATH = os.path.join(paths.REPO, "self-correct", "sc_traj.py")

_SC_TRAJ = None


def sc_traj():
    """Import self-correct/sc_traj.py by absolute path (cached)."""
    global _SC_TRAJ
    if _SC_TRAJ is None:
        if not os.path.exists(SC_TRAJ_PATH):
            raise IOError(
                "sc_traj.py not found at %s. It is the repo's env-free "
                "trajectory reader and bcdata.py deliberately does not "
                "duplicate its tag parser." % SC_TRAJ_PATH)
        import importlib.util
        spec = importlib.util.spec_from_file_location("adaptercl._sc_traj",
                                                      SC_TRAJ_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SC_TRAJ = mod
    return _SC_TRAJ


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

TAGS = ("think", "plan", "step", "memory", "action")


class Step(object):
    """One BC training sample: a rendered prompt and the teacher's completion.

    `sections` holds the parsed <think>/<plan>/<step>/<memory>/<action> blocks;
    `raw` is the completion string exactly as dumped (what convert2sgptArgs
    re-parses). One Step == one ShareGPT sample, because the converter emits one
    sample per step (convert2sgptArgs.py:199-307).
    """

    __slots__ = ("index", "user_text", "axtree", "goal", "sections", "raw")

    def __init__(self, index, user_text, axtree, goal, sections, raw=""):
        self.index = index
        self.user_text = user_text
        self.axtree = axtree
        self.goal = goal
        self.sections = sections
        self.raw = raw

    @property
    def action(self):
        return self.sections.get("action", "")

    @property
    def think(self):
        return self.sections.get("think", "")

    @property
    def plan(self):
        return self.sections.get("plan", "")

    @property
    def memory(self):
        return self.sections.get("memory", "")

    @property
    def step_text(self):
        """The <step> tag (the current plan step), not the step index."""
        return self.sections.get("step", "")

    def as_dict(self):
        return {"index": self.index, "goal": self.goal,
                "action": self.action, "n_chars_prompt": len(self.user_text),
                "n_chars_axtree": len(self.axtree),
                "has": dict((t, bool(self.sections.get(t))) for t in TAGS)}

    def __repr__(self):
        return "Step(%d, action=%r)" % (self.index, self.action[:40])


class Episode(object):
    """One teacher rollout directory.

    `cell` is the (environment, era) it belongs to, or None for the multi-site
    tasks that cells.task_cell_map deliberately drops (a wiki+news episode is
    not attributable to one interface, so cross-applying a wiki adapter to it
    would confound 6.2).
    """

    __slots__ = ("dir", "task_id", "seed", "era", "cell", "reward", "n_steps",
                 "err_msg", "system", "steps", "summary")

    def __init__(self, dir, task_id, seed, era, cell, reward, n_steps,
                 err_msg, system, steps, summary):
        self.dir = dir
        self.task_id = task_id
        self.seed = seed
        self.era = era
        self.cell = cell
        self.reward = reward
        self.n_steps = n_steps
        self.err_msg = err_msg
        self.system = system
        self.steps = steps
        self.summary = summary

    # -- identity ---------------------------------------------------------
    @property
    def name(self):
        return os.path.basename(os.path.normpath(self.dir))

    @property
    def key(self):
        """Stable, era-qualified id. Episode dir names collide across eras --
        every version_* dir was written in the same batch with the same
        timestamp prefix -- so the era must be part of the key."""
        return "v%s__%s" % (self.era, self.name)

    @property
    def verified(self):
        return self.reward is not None and float(self.reward) >= 1.0

    @property
    def n_bc_steps(self):
        """Samples this episode contributes to a ShareGPT corpus."""
        return len(self.steps)

    @property
    def goal(self):
        return self.steps[0].goal if self.steps else ""

    def final_action(self):
        for st in reversed(self.steps):
            if st.action:
                return st.action
        return ""

    def answered(self):
        """Mirrors filter_training_data.py:47-66 (`--final-answer`)."""
        return sc_traj().is_send_msg(self.final_action())

    def errored(self):
        """Mirrors filter_training_data.py's `--no-error`."""
        return bool(self.err_msg)

    def as_dict(self):
        return {"dir": self.dir, "key": self.key, "task_id": self.task_id,
                "seed": self.seed, "era": self.era,
                "cell": self.cell.key if self.cell else None,
                "reward": self.reward, "n_steps": self.n_steps,
                "n_bc_steps": self.n_bc_steps, "err_msg": self.err_msg,
                "answered": self.answered(), "goal": self.goal[:200]}

    def __repr__(self):
        return "Episode(%s, task=%s, reward=%s, steps=%d)" % (
            self.cell.key if self.cell else "?", self.task_id, self.reward,
            self.n_bc_steps)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_episode(episode_dir, era=None, cell=None):
    """Read one episode dir into an `Episode`, or None if it has no dumps."""
    raw = sc_traj().load_episode(episode_dir)
    if raw is None:
        return None
    steps = []
    for st in raw["steps"]:
        content = ""
        sec = st["sections"]
        # Re-render the completion in the converter's canonical tag form so
        # `raw` is what convert2sgptArgs would emit for a --think --plan
        # --memory corpus (sc_traj.py:68-81).
        content = sc_traj().sections_to_content(sec)
        steps.append(Step(st["step"], st["user_text"], st["axtree"],
                          st["goal"], sec, content))
    seed = None
    m = re.search(r"timewarp\.\d+_(\d+)$", os.path.basename(
        os.path.normpath(episode_dir)))
    if m:
        seed = int(m.group(1))
    if era is None:
        m = re.search(r"version_(\d+)", episode_dir)
        era = int(m.group(1)) if m else None
    if cell is None and era is not None and raw["task_id"] is not None:
        cell = cells.task_cell_map(era).get(raw["task_id"])
    return Episode(dir=episode_dir, task_id=raw["task_id"], seed=seed, era=era,
                   cell=cell, reward=raw["reward"], n_steps=raw["n_steps"],
                   err_msg=raw["err_msg"], system=raw["system"], steps=steps,
                   summary=raw["summary"])


def load_cell(unit, verified_only=True, min_reward=1.0, require_answer=False,
              max_episodes=None, max_samples=None, draw_seed=None, root=None,
              quiet=True):
    """Every teacher episode for one training unit.

    `unit` is one of three things, dispatched on `unit.granularity`:

    * `cells.Version` -- one era pooled over all three sites (the default
      adapter unit, 2);
    * `cells.Cell` -- one environment x era, needed only for 6.2's crossed
      transfer matrix;
    * `cells.Site` -- one site pooled over all six eras (the DOMAIN axis,
      domain_transfer_timewarp_plan.md). Membership is decided by
      `Site.accepts(rec)` and nothing else, so the corpus and the census cannot
      disagree: single-site units take `rec["cell"].env`, `multi` takes exactly
      the records whose `cell` is None, and `all` takes the union.

    Parameters
    ----------
    verified_only : keep only `cum_reward >= min_reward` (4.4's verifier signal)
    require_answer : additionally require the last action to be send_msg_to_user
                     (filter_training_data.py's `--final-answer`)
    max_episodes : truncate to this many episodes -- used to equalise cell sizes
                   so per-cell LoRAs are not confounded by corpus size.
    max_samples : truncate to at most this many BC *samples* (steps). This is
                  the dose-matching knob the site design needs and
                  `max_episodes` cannot be: steps-per-episode differs by a
                  factor of 1.75 across the units (multi ~9.3, news ~5.3), so
                  equal episode counts are NOT equal doses
                  (domain_transfer_timewarp_plan.md 7, first trap). Episodes are
                  never split: we add whole episodes in order and stop at the
                  first one that would overflow, which undershoots by less than
                  one episode's worth of steps and keeps the draw unbiased.
    draw_seed : shuffle the episode order with this seed before truncating. The
                default (None) keeps the historical deterministic order (sorted
                by task_id, seed) so existing corpora are reproducible
                byte-for-byte. A dose-matched arm needs a *draw*, not a prefix
                -- a prefix of a task_id-sorted list is a biased subset of the
                task distribution, and three arms sharing one prefix are not
                three seeds.
    """
    gran = getattr(unit, "granularity", None)
    if gran is None:
        # Pre-`granularity` duck-typing, kept so an old caller passing an
        # object with just .env/.era still works: env is None <=> Version.
        gran = "version" if getattr(unit, "env", "") is None else "cell"

    if gran == "site":
        eras = list(unit.eras)
        accept = unit.accepts
        # The draw is part of the Site's KEY, so it is also the default draw
        # seed. Without this an arm named `wiki_d1` could be trained on the
        # deterministic prefix -- three "seeds" that are the same corpus, which
        # is exactly the unwitting seed-replication that invalidated an earlier
        # TimeWarp result (memory: timewarp-scale-rewards-dead-knob).
        if draw_seed is None:
            draw_seed = unit.draw
    else:
        eras = [unit.era]
        # A Version pools every site for its era, so it accepts episodes whose
        # task spans several sites -- those are unattributable to a Cell but are
        # exactly what version-level training (2) should learn from. A Cell
        # takes only its own single-site episodes.
        if gran == "version":
            accept = lambda rec: True
        else:
            accept = lambda rec: rec["cell"] == unit

    out = []
    for era in eras:
        for rec in cells.scan_episodes(era, root=root):
            if not accept(rec):
                continue
            ep = load_episode(rec["dir"], era=rec.get("era", era),
                              cell=rec["cell"] or (unit if gran != "site" else None))
            if ep is None:
                if not quiet:
                    print("  skip (no training_data): %s" % rec["dir"])
                continue
            if verified_only and (ep.reward is None
                                  or float(ep.reward) < min_reward):
                continue
            if require_answer and not ep.answered():
                continue
            out.append(ep)

    # Sort FIRST, always, then shuffle. Sorting makes the pre-shuffle order
    # independent of the filesystem's listing order, so `draw_seed=k` selects
    # the same subset on any machine and after any purge -- a shuffle applied to
    # an os.listdir order would not.
    out.sort(key=lambda e: (e.era if e.era is not None else -1,
                            e.task_id if e.task_id is not None else -1,
                            e.seed if e.seed is not None else -1))
    if draw_seed is not None:
        random.Random(int(draw_seed)).shuffle(out)
    if max_episodes is not None:
        out = out[:max_episodes]
    if max_samples is not None:
        kept, total = [], 0
        for ep in out:
            n = ep.n_bc_steps
            if total + n > int(max_samples):
                break
            kept.append(ep)
            total += n
        if total < MIN_DOSE_FRACTION * float(max_samples) and not quiet:
            print("NOTE: %s selected %d of %d requested samples (%d episodes); "
                  "the overflowing episode was %d steps."
                  % (unit.key, total, max_samples, len(kept),
                     out[len(kept)].n_bc_steps if len(kept) < len(out) else 0))
        out = kept
    return out


def load_cells(cell_list, **kw):
    """{cell_key: [Episode]} for a list of cells."""
    out = collections.OrderedDict()
    for c in cell_list:
        out[c.key] = load_cell(c, **kw)
    return out


def load_split(split_name, envs=None, **kw):
    """The train/test cell partitions of a named split (6.1).

    Returns {"split": name, "train": {cell_key: [Episode]},
             "test": {cell_key: [Episode]}}. Note this splits *cells*, not
    tasks: the task ids are identical on both sides by construction (a task is
    the same on every era), which is exactly what makes the drift gap in 6.6 a
    paired comparison.
    """
    sp = cells.split_by_name(split_name)
    return {"split": split_name,
            "train_eras": list(sp.train_eras), "test_eras": list(sp.test_eras),
            "train": load_cells(sp.train_cells(envs), **kw),
            "test": load_cells(sp.test_cells(envs), **kw)}


# --------------------------------------------------------------------------
# Census (7 Phase 2 blocker)
# --------------------------------------------------------------------------

#: Phase 2 wants "one LoRA per cell". Below this many verified episodes a cell's
#: LoRA is not a fair member of the 18-cell grid.
MIN_EPISODES_PHASE2 = 20

#: `max_samples` adds whole episodes and stops at the first overflow, so a
#: dose-matched corpus can undershoot by up to one episode's worth of steps
#: (~9 for `multi`). Below this fraction of the request, say so out loud: the
#: arms are then not dose-matched to each other and the matrix must be read at
#: the ACTUAL sample counts (domain_transfer_timewarp_plan.md 2.1/7).
MIN_DOSE_FRACTION = 0.97

#: Measured 2026-08-06 by this module (`python -m adaptercl.bcdata census`).
#: Kept here so a future purge/regeneration is detectable by comparison.
KNOWN_ERA_TOTALS = {1: (30, 178), 2: (126, 848), 3: (127, 891), 4: (128, 847),
                    5: (124, 845), 6: (125, 839)}


_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)


def last_action_is_answer(episode_dir, n_steps):
    """Cheap `--final-answer` test without parsing the whole episode.

    Copied in spirit from filter_training_data.py:47-66: read only
    `agent_output_step_<n_steps-1>.txt` and test the action prefix.
    """
    if not n_steps:
        return False
    p = os.path.join(episode_dir, "training_data",
                     "agent_output_step_%d.txt" % (int(n_steps) - 1))
    if not os.path.exists(p):
        return False
    try:
        with open(p) as fh:
            text = fh.read()
    except IOError:
        return False
    m = _ACTION_RE.search(text)
    if not m:
        return False
    body = m.group(1).strip()
    return (body.startswith("send_msg_to_user(")
            or body.startswith("send_message_to_user("))


def census(eras=None, min_episodes=MIN_EPISODES_PHASE2, root=None):
    """Per-cell BC-data census with an explicit `blocking` list.

    Unlike cells.availability() this also counts *steps* (= ShareGPT samples),
    unattributed multi-site episodes, and per-era totals, and it flags cells
    below `min_episodes` as blocking rather than merely "short".
    """
    eras = tuple(eras or cells.ERAS)
    per_cell = collections.OrderedDict()
    for c in cells.cells_for(None, eras):
        per_cell[c.key] = {"env": c.env, "era": c.era, "episodes": 0,
                           "verified": 0, "answered": 0, "steps": 0,
                           "verified_steps": 0}
    per_era = collections.OrderedDict()
    unattributed = collections.OrderedDict()
    for era in eras:
        recs = cells.scan_episodes(era, root=root)
        per_era[era] = {"episodes": 0, "steps": 0, "verified": 0,
                        "unattributed": 0}
        for rec in recs:
            per_era[era]["episodes"] += 1
            per_era[era]["steps"] += rec["n_dumps"]
            if rec["reward"] is not None and float(rec["reward"]) >= 1.0:
                per_era[era]["verified"] += 1
            if rec["cell"] is None:
                per_era[era]["unattributed"] += 1
                unattributed.setdefault(era, []).append(rec["task_id"])
                continue
            s = per_cell[rec["cell"].key]
            s["episodes"] += 1
            s["steps"] += rec["n_dumps"]
            if last_action_is_answer(rec["dir"], rec["n_steps"] or rec["n_dumps"]):
                s["answered"] += 1
            if float(rec["reward"] or 0.0) >= 1.0:
                s["verified"] += 1
                s["verified_steps"] += rec["n_dumps"]

    blocking = []
    for key, s in per_cell.items():
        s["ok"] = s["verified"] >= min_episodes
        if not s["ok"]:
            blocking.append(key)
    n_ep = sum(s["episodes"] for s in per_cell.values())
    n_ver = sum(s["verified"] for s in per_cell.values())
    # The domain axis's census. Pooled over eras and -- unlike per_cell --
    # counting the multi-site episodes, which are a unit of their own here
    # rather than "unattributable" (domain_transfer_timewarp_plan.md 1.2).
    per_site = cells.site_availability(min_episodes=min_episodes, root=root) \
        if tuple(eras) == tuple(cells.ERAS) else None
    return {
        "eras": list(eras),
        "min_episodes": min_episodes,
        "per_cell": per_cell,
        "per_site": per_site,
        "per_era": per_era,
        "blocking": blocking,
        "unattributed": dict((str(k), sorted(set(v)))
                             for k, v in unattributed.items()),
        "totals": {"episodes": n_ep, "verified": n_ver,
                   "steps": sum(s["steps"] for s in per_cell.values()),
                   "verify_rate": (float(n_ver) / n_ep) if n_ep else 0.0,
                   "cells": len(per_cell), "blocking": len(blocking)},
        "source": paths.TIMETRAJ,
    }


def print_census(cen=None, fh=None):
    """Readable census report. Prints the era-1 shortfall loudly (7 Phase 2)."""
    fh = fh or sys.stdout
    cen = cen or census()
    print("BC corpus census -- %s" % cen["source"], file=fh)
    print("", file=fh)
    hdr = "  %-10s %9s %9s %9s %8s %8s" % ("cell", "episodes", "verified",
                                           "answered", "steps", "ok")
    print(hdr, file=fh)
    print("  " + "-" * (len(hdr) - 2), file=fh)
    for key, s in cen["per_cell"].items():
        print("  %-10s %9d %9d %9d %8d %8s%s"
              % (key, s["episodes"], s["verified"], s["answered"], s["steps"],
                 "yes" if s["ok"] else "NO",
                 "" if s["ok"] else "   <-- BLOCKING"), file=fh)
    print("", file=fh)
    if cen.get("per_site"):
        print("  by SITE -- the domain-axis unit (pooled over eras; `multi` is "
              "the unattributable column above):", file=fh)
        print("    %-8s %9s %9s %8s %9s %7s"
              % ("unit", "episodes", "verified", "steps", "ver.steps", "st/ep"),
              file=fh)
        for key, s in cen["per_site"].items():
            if key.startswith("_"):
                continue
            print("    %-8s %9d %9d %8d %9d %7.2f"
                  % (key, s["episodes"], s["verified"], s["steps"],
                     s["verified_steps"], s["steps_per_episode"]), file=fh)
        print("    dose-matched corpus = %d verified samples; matching on "
              "EPISODES instead would give `multi` %.2fx the dose"
              % (cen["per_site"]["_matched_samples"],
                 cen["per_site"]["multi"]["steps_per_episode"]
                 / max(1e-9, cen["per_site"]["news"]["steps_per_episode"])),
              file=fh)
        print("", file=fh)
    print("  per-era totals (all episodes, incl. multi-site):", file=fh)
    for era, s in cen["per_era"].items():
        print("    era %d: %3d episodes, %4d steps, %3d verified, "
              "%d multi-site (unattributable)"
              % (era, s["episodes"], s["steps"], s["verified"],
                 s["unattributed"]), file=fh)
    t = cen["totals"]
    print("", file=fh)
    print("  attributed: %d episodes / %d steps over %d cells; verify rate %.3f"
          % (t["episodes"], t["steps"], t["cells"], t["verify_rate"]), file=fh)

    if cen["blocking"]:
        print("", file=fh)
        print("  " + "!" * 72, file=fh)
        print("  PHASE-2 BLOCKER: %d of %d cells have < %d verified episodes."
              % (len(cen["blocking"]), t["cells"], cen["min_episodes"]),
              file=fh)
        for key in cen["blocking"]:
            s = cen["per_cell"][key]
            print("    %-10s %d episodes / %d verified / %d BC steps"
                  % (key, s["episodes"], s["verified"], s["steps"]), file=fh)
        print("  adapterCL.md 7 Phase 2 says 'train one LoRA per (environment,"
              " era) cell -- 18 cells'.", file=fh)
        print("  That design is NOT satisfiable from the data on disk: era 1 was"
              " never rolled out for", file=fh)
        print("  wiki at all, and shop_e1 has 2 episodes. Options, in order of"
              " honesty:", file=fh)
        print("    (a) collect era-1 teacher rollouts for wiki+shop (the "
              "TimeTraj pipeline still exists);", file=fh)
        print("    (b) run the 6.2 transfer matrix on the 16 complete cells and"
              " say so in the paper;", file=fh)
        print("    (c) drop era 1 entirely -- but note the 'forward' split "
              "trains on eras 1-3, so", file=fh)
        print("        dropping era 1 changes the headline split's training set"
              " (cells.SPLITS).", file=fh)
        print("  " + "!" * 72, file=fh)
    return cen


# --------------------------------------------------------------------------
# ShareGPT corpus construction
# --------------------------------------------------------------------------

def _converter_python():
    """Interpreter used to run convert2sgptArgs.py.

    paths.PY_BENCH is the canonical choice (it is the env the rest of the repo
    runs the data scripts under). The converter is pure stdlib, so if that env
    has been purged we fall back to the running interpreter rather than dying --
    and say so, because the fallback is a deviation from repo convention.
    """
    if os.path.exists(paths.PY_BENCH):
        return paths.PY_BENCH
    print("WARNING: %s is missing (scratch purge?); running the converter with "
          "%s instead. convert2sgptArgs.py is stdlib-only so output is "
          "unaffected." % (paths.PY_BENCH, sys.executable))
    return sys.executable


def stage_episodes(episodes, stage_dir, clean=True):
    """Symlink episode dirs into a staging dir the converter can walk.

    convert2sgptArgs discovers work by iterating the input dir and testing
    `(item / "training_data").exists()` (convert2sgptArgs.py:351-360), and a
    symlink to a directory satisfies that. Staging by symlink means the corpus
    is produced from the *original* dumps -- no copies to drift out of sync --
    while still letting us pick an arbitrary subset of episodes.
    """
    if clean and os.path.isdir(stage_dir):
        shutil.rmtree(stage_dir)
    if not os.path.isdir(stage_dir):
        os.makedirs(stage_dir)
    staged = []
    for ep in episodes:
        link = os.path.join(stage_dir, ep.key)
        if os.path.islink(link) or os.path.exists(link):
            raise IOError("staging collision at %s -- Episode.key is supposed "
                          "to be unique" % link)
        os.symlink(os.path.abspath(ep.dir), link)
        staged.append(link)
    if not staged:
        raise ValueError("nothing staged into %s: the episode selection is "
                         "empty. Refusing to build an empty corpus." % stage_dir)
    return staged


def run_converter(stage_dir, out_json, think=True, plan=True, memory=True,
                  som=False, img=False, sampling=1, p=1.0, python=None):
    """Invoke the repo's ShareGPT converter. Returns (cmd, stdout)."""
    if not os.path.exists(paths.CONVERT_SGPT):
        raise IOError("converter missing: %s" % paths.CONVERT_SGPT)
    cmd = [python or _converter_python(), paths.CONVERT_SGPT, stage_dir,
           "--output", out_json, "--sampling", str(int(sampling)),
           "--p", str(float(p))]
    if think:
        cmd.append("--think")
    if plan:
        cmd.append("--plan")
    if memory:
        cmd.append("--memory")
    if som:
        cmd.append("--som")
    if img:
        cmd.append("--img")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    out, _ = proc.communicate()
    out = out.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise RuntimeError("convert2sgptArgs.py failed (rc=%d)\n%s\n%s"
                           % (proc.returncode, " ".join(cmd), out))
    return cmd, out


def _stats_path(out_json):
    base = out_json[:-5] if out_json.endswith(".json") else out_json
    return base + ".stats.json"


def build_sharegpt(cell_list, out_json, think=True, plan=True, memory=True,
                   som=False, img=False, verified_only=True, min_reward=1.0,
                   require_answer=False, max_episodes=None, max_samples=None,
                   draw_seed=None, sampling=1, p=1.0,
                   stage_dir=None, register=False, dataset_name=None,
                   keep_stage=False, python=None):
    """Build one ShareGPT corpus from a set of cells.

    The corpus itself is produced by `dataCreationScripts/convert2sgptArgs.py`
    (one sample per step, keys system/conversations[/images]) so it is
    byte-identical to every other TimeWarp corpus. We add a `.stats.json`
    provenance side-car: which cells, which filters, how many episodes and
    samples, and the exact converter command line.

    `som`/`img` are accepted but will fail loudly on the TimeTraj corpus: there
    are no PNGs under version_*/ (see the module docstring), and the converter
    *skips steps* whose image is missing (convert2sgptArgs.py:214-223), which
    would silently shrink the corpus.
    """
    paths.ensure_out_dirs()
    if isinstance(cell_list, (str, cells.Cell)):
        cell_list = [cell_list]
    # `parse_unit` accepts both 'v3' (a Version, the default adapter unit) and
    # 'wiki_e3' (a Cell, for 6.2's crossing).
    cell_list = [cells.parse_unit(c) if isinstance(c, str) else c
                 for c in cell_list]
    episodes = []
    per_cell = collections.OrderedDict()
    for c in cell_list:
        eps = load_cell(c, verified_only=verified_only, min_reward=min_reward,
                        require_answer=require_answer,
                        max_episodes=max_episodes, max_samples=max_samples,
                        draw_seed=draw_seed)
        per_cell[c.key] = {"episodes": len(eps),
                           "steps": sum(e.n_bc_steps for e in eps)}
        episodes.extend(eps)
    if not episodes:
        raise ValueError(
            "no episodes selected for cells %s with verified_only=%s "
            "min_reward=%s require_answer=%s. Run "
            "`python -m adaptercl.bcdata census` -- era 1 is largely empty."
            % ([c.key for c in cell_list], verified_only, min_reward,
               require_answer))

    if (som or img) and not _has_images(episodes):
        raise IOError(
            "--som/--img requested but none of the %d selected episodes "
            "contains a .png. The TimeTraj dumps are text-only; the converter "
            "would silently drop every step (convert2sgptArgs.py:214-223). "
            "Capture screenshots into %s first (adapterCL.md 4.2 capture "
            "protocol)." % (len(episodes), paths.OUT_CAPTURE))

    return build_sharegpt_from_episodes(
        episodes, out_json, cell_keys=[c.key for c in cell_list],
        per_cell=per_cell, think=think, plan=plan, memory=memory, som=som,
        img=img, sampling=sampling, p=p, stage_dir=stage_dir,
        register=register, dataset_name=dataset_name, keep_stage=keep_stage,
        python=python,
        filters={"verified_only": verified_only, "min_reward": min_reward,
                 "require_answer": require_answer,
                 "max_episodes": max_episodes,
                 # The dose. Read this side-car before comparing two arms:
                 # 7's first trap is reading a site gap off arms whose sample
                 # counts differ.
                 "max_samples": max_samples,
                 "draw_seed": draw_seed})


def build_sharegpt_from_episodes(episodes, out_json, cell_keys=None,
                                 per_cell=None, filters=None, think=True,
                                 plan=True, memory=True, som=False, img=False,
                                 sampling=1, p=1.0, stage_dir=None,
                                 register=False, dataset_name=None,
                                 keep_stage=False, python=None):
    """One ShareGPT corpus from an ARBITRARY list of `Episode`s.

    `build_sharegpt` selects episodes from cells and then calls this; the split
    exists because `training_conditional_plan.md` needs corpora whose membership
    is an explicit episode list -- a chunk's 85 % train half and its 15 %
    holdout are not a cell, and the holdout has to be reproducible from the ids
    written in the manifest rather than re-derived from a filter.

    The `.stats.json` side-car is identical in both paths (same keys, same
    order), so provenance does not depend on which entry point built a corpus.
    `episodes` is the authoritative record of what went in.
    """
    if not episodes:
        raise ValueError("refusing to build an empty corpus: no episodes given")
    out_json = os.path.abspath(out_json)
    parent = os.path.dirname(out_json)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    stage_dir = stage_dir or os.path.join(
        paths.OUT_CORPUS, "_stage",
        os.path.basename(out_json)[:-5] if out_json.endswith(".json")
        else os.path.basename(out_json))
    stage_episodes(episodes, stage_dir)
    try:
        cmd, log = run_converter(stage_dir, out_json, think=think, plan=plan,
                                 memory=memory, som=som, img=img,
                                 sampling=sampling, p=p, python=python)
    finally:
        if not keep_stage and os.path.isdir(stage_dir):
            shutil.rmtree(stage_dir)

    with open(out_json) as fh:
        corpus = json.load(fh)
    stats = {
        "out_json": out_json,
        "created": datetime.datetime.now().isoformat(),
        "cells": list(cell_keys or []),
        "per_cell": per_cell if per_cell is not None else {},
        "n_episodes": len(episodes),
        "n_expected_samples": sum(e.n_bc_steps for e in episodes) * int(sampling),
        "n_samples": len(corpus),
        "filters": dict(filters or {}),
        "task_ids": sorted(set(e.task_id for e in episodes
                               if e.task_id is not None)),
        "converter": {"script": paths.CONVERT_SGPT, "cmd": cmd,
                      "think": think, "plan": plan, "memory": memory,
                      "som": som, "img": img, "sampling": sampling, "p": p},
        "episodes": [e.key for e in episodes],
        "source": paths.TIMETRAJ,
    }
    if stats["n_samples"] != stats["n_expected_samples"] and p >= 1.0:
        stats["WARNING"] = (
            "converter emitted %d samples but the dumps have %d steps; the "
            "converter drops steps with no <action> or no user prompt"
            % (stats["n_samples"], stats["n_expected_samples"]))
        print("WARNING: " + stats["WARNING"])
    with open(_stats_path(out_json), "w") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)

    name = dataset_name or ("adaptercl_" + _tag_from_path(out_json))
    stats["dataset_name"] = name
    # Our own registry always gets the entry; it is inside adapter_project, so
    # writing it never touches the parent repo.
    register_dataset(name, out_json, dataset_info=local_dataset_info(),
                     images=bool(som or img))
    if register:
        register_dataset(name, out_json, dataset_info=paths.DATASET_INFO,
                         images=bool(som or img))
    return stats


def _has_images(episodes):
    for ep in episodes:
        for root, _dirs, files in os.walk(ep.dir):
            for f in files:
                if f.endswith(".png"):
                    return True
    return False


def _tag_from_path(out_json):
    base = os.path.basename(out_json)
    if base.endswith(".json"):
        base = base[:-5]
    return re.sub(r"[^A-Za-z0-9_]", "_", base)


def build_cell_corpora(cell_list, out_dir=None, tag="percell", **kw):
    """One ShareGPT corpus per cell + a manifest.

    Two consumers need this instead of a pooled corpus:

    * Phase 2 trains one LoRA per cell, and LLaMA-Factory takes one dataset name;
    * `train_hypernet.py` needs to know which cell each *sample* came from, and
      a pooled corpus loses that -- the converter concatenates without labels.
    """
    out_dir = out_dir or os.path.join(paths.OUT_CORPUS, tag)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # `parse_unit` accepts both 'v3' (a Version, the default adapter unit) and
    # 'wiki_e3' (a Cell, for 6.2's crossing).
    cell_list = [cells.parse_unit(c) if isinstance(c, str) else c
                 for c in cell_list]
    manifest = {"tag": tag, "created": datetime.datetime.now().isoformat(),
                "out_dir": out_dir, "cells": collections.OrderedDict(),
                "skipped": collections.OrderedDict(), "options": dict(kw)}
    for c in cell_list:
        out_json = os.path.join(out_dir, "%s.json" % c.key)
        try:
            stats = build_sharegpt([c], out_json,
                                   dataset_name="adaptercl_%s_%s" % (tag, c.key),
                                   **kw)
        except ValueError as exc:
            # Empty cell (era 1). Record it -- never drop it silently.
            manifest["skipped"][c.key] = str(exc)
            print("SKIP %s: %s" % (c.key, exc))
            continue
        manifest["cells"][c.key] = {
            "json": out_json, "dataset": stats["dataset_name"],
            "n_samples": stats["n_samples"], "n_episodes": stats["n_episodes"]}
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    manifest["manifest"] = mpath
    return manifest


# --------------------------------------------------------------------------
# LLaMA-Factory dataset registration
# --------------------------------------------------------------------------

def local_dataset_info():
    """Our own dataset_info.json, inside adapter_project/out/corpus.

    Pointing LLaMA-Factory's `dataset_dir` at this file keeps every adapterCL
    dataset out of the parent repo's registry (which is shared with half a dozen
    other experiments and is a merge hazard).
    """
    paths.ensure_out_dirs()
    return os.path.join(paths.OUT_CORPUS, "dataset_info.json")


def register_dataset(name, json_path, dataset_info=None, images=False,
                     force=False, backup=True):
    """Add one ShareGPT dataset to a LLaMA-Factory dataset_info.json.

    Read-modify-write: existing keys are never clobbered (unless `force`), and
    the file is backed up first. `file_name` is written absolute -- LF joins it
    with `dataset_dir` and os.path.join with an absolute right-hand side yields
    the absolute path, so an out-of-tree corpus loads unchanged.
    """
    dataset_info = dataset_info or paths.DATASET_INFO
    json_path = os.path.abspath(json_path)
    if not os.path.exists(json_path):
        raise IOError("corpus %s does not exist" % json_path)
    info = {}
    if os.path.exists(dataset_info):
        with open(dataset_info) as fh:
            info = json.load(fh)
        if backup:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            shutil.copyfile(dataset_info, "%s.bak.%s" % (dataset_info, stamp))
    columns = {"messages": "conversations", "system": "system"}
    if images:
        columns["images"] = "images"
    entry = {"file_name": json_path, "formatting": "sharegpt",
             "columns": columns}
    if name in info and info[name] != entry and not force:
        raise KeyError(
            "dataset %r already registered in %s with a different definition "
            "(%r). Pass force=True only if you mean to change what every other "
            "experiment using that name trains on."
            % (name, dataset_info, info[name]))
    info[name] = entry
    d = os.path.dirname(dataset_info)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(dataset_info, "w") as fh:
        json.dump(info, fh, indent=2, sort_keys=True)
    return {"dataset_info": dataset_info, "name": name, "entry": entry}


# --------------------------------------------------------------------------
# Site-manual injection (scout_plan.md arm A)
# --------------------------------------------------------------------------

#: The heading AgentLab renders the agent's `extra_instructions` under
#: (AgentLab/src/agentlab/agents/dynamic_prompting.py:499-511). Every sample in
#: the version corpora carries it, and the eval prompt is built by the same
#: element, which is what makes the injection point identical on both sides.
EXTRA_INSTRUCTIONS_HEADING = "## Extra instructions:"

#: What follows the extra-instructions block in a rendered prompt. The corpus
#: samples were written by the same prompt element, so this boundary is exact:
#: body newline + the element's trailing newline + the next element's leading
#: newline.
_OBS_BOUNDARY = "\n\n\n# Observation of current step:"

#: Duplicated (once) in `adaptercl.scout.manual_block` and in
#: `scripts/benchmark_adapter.py`; `tests/test_scout.py` asserts all three
#: agree. If they drift, arm A trains on a prompt shape it is never evaluated
#: on and the comparison against arm C is no longer one-variable.
MANUAL_BLOCK_HEADER = "# Site manuals"


def manual_block(text):
    """The exact block appended to `extra_instructions` at eval time."""
    return "\n%s\n%s\n" % (MANUAL_BLOCK_HEADER, text.strip())


def inject_manual_into_prompt(prompt, manual_text):
    """Put the manual in the same slot the eval puts it in.

    Eval appends `manual_block(text)` to `extra_instructions`; the prompt
    element then renders `## Extra instructions:\n\n{extra}\n` and the
    observation element follows with its own leading newline. So injecting into
    a rendered corpus prompt means replacing the extra-instructions/observation
    boundary with the manual block plus that same boundary -- which reproduces
    the eval string byte for byte apart from the URLs.

    Raises rather than silently returning the input: a sample that quietly kept
    no manual would train arm A on arm C's prompt.
    """
    if MANUAL_BLOCK_HEADER in prompt:
        raise ValueError("prompt already carries a site manual")
    i = prompt.find(EXTRA_INSTRUCTIONS_HEADING)
    if i < 0:
        raise ValueError("no %r section in this sample"
                         % EXTRA_INSTRUCTIONS_HEADING)
    j = prompt.find(_OBS_BOUNDARY, i)
    if j < 0:
        raise ValueError("no observation boundary after the extra-instructions "
                         "section -- the corpus prompt shape has changed")
    # prompt[:j] ends at the last character of the extra-instructions body, so
    # the body's own trailing newline is the first of the boundary's three.
    # Re-emit it, then the block, then the two the boundary still owes.
    return prompt[:j] + "\n" + manual_block(manual_text) + "\n" + prompt[j + 2:]


def inject_manual(json_in, json_out, manual_text, first_turn_only=True):
    """Write a copy of a ShareGPT corpus with the manual in every sample.

    The corpus samples are `{"system", "conversations"}` (out/corpus/ver/v*.json,
    828 samples for v2); only the FIRST human turn carries the instruction
    block, which is where the manual goes -- the same place, once per sample,
    as the eval prompt.
    """
    with open(json_in) as fh:
        data = json.load(fh)
    n_ok = 0
    for sample in data:
        convs = sample.get("conversations") or []
        for turn in convs:
            if turn.get("from") != "human":
                continue
            turn["value"] = inject_manual_into_prompt(turn["value"], manual_text)
            n_ok += 1
            if first_turn_only:
                break
    d = os.path.dirname(os.path.abspath(json_out))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(json_out, "w") as fh:
        json.dump(data, fh, ensure_ascii=False)
    return {"in": json_in, "out": json_out, "n_samples": len(data),
            "n_injected": n_ok, "manual_chars": len(manual_text.strip()),
            "manual_sha1": _sha1(manual_text.strip())}


# --------------------------------------------------------------------------
# Conditioning attachment (4.3: condition once, freeze for the episode)
# --------------------------------------------------------------------------

#: Where a Phase-0 capture pass is expected to leave one reference screenshot
#: per cell. Nothing in this module creates these -- capture needs a live env.
def default_image_for_cell(cell, root=None):
    """First .png under `<OUT_CAPTURE>/<cell.key>/`, or None."""
    root = root or paths.OUT_CAPTURE
    d = os.path.join(root, cell.key)
    if not os.path.isdir(d):
        return None
    for name in sorted(os.listdir(d)):
        if name.lower().endswith(".png"):
            return os.path.join(d, name)
    return None


def episode_screenshot(episode, step=0):
    """The episode's own step-`step` screenshot, if the rollout saved one.

    TimeTraj did not (there are no PNGs under version_*/), but rollouts produced
    by this project's own eval bridge may, and per-episode conditioning is what
    4.3 actually specifies.
    """
    for name in ("screenshot_step_%d.png" % step, "step_%d.png" % step,
                 "som_step_%d.png" % step):
        p = os.path.join(episode.dir, name)
        if os.path.exists(p):
            return p
    som = os.path.join(episode.dir, "SoM", "som_step_%d.png" % step)
    return som if os.path.exists(som) else None


def _sha1(text):
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def attach_conditioning(episodes, image_for_cell=None, per_episode=True,
                        out_json=None, quiet=False):
    """Record which conditioning signal each episode should be adapted with.

    4.3 fixes the granularity: **condition once on the initial observation and
    freeze the adapter for the episode**. This function makes that concrete and
    auditable -- for each episode it resolves a conditioning *key* (the identity
    the generator is conditioned on) and, if one exists, the image behind it.

    Resolution order, most specific first:

      1. `episode_screenshot(ep, 0)`         -> source "episode_screenshot"
      2. `image_for_cell(ep.cell)`           -> source "cell_capture"
      3. no image at all                     -> source "axtree_text"

    (3) is not a silent fallback: the manifest records the source per episode and
    the report prints the breakdown. On the TimeTraj corpus *every* episode lands
    in (3) unless a capture pass has populated paths.OUT_CAPTURE, which is the
    single most important thing to know before believing any "vision
    conditioning" result. The step-0 AXTree digest is carried either way -- it is
    the deterministic structural side-channel of 4.2 and the only conditioning
    signal recoverable from the trajectories alone.

    `per_episode=False` collapses the key to the cell, which is the
    approximation the Phase-2 per-cell adapters implicitly make.
    """
    image_for_cell = image_for_cell or default_image_for_cell
    rows = []
    counts = collections.Counter()
    for ep in episodes:
        cell_key = ep.cell.key if ep.cell else None
        img = episode_screenshot(ep, 0)
        source = "episode_screenshot"
        if img is None and ep.cell is not None:
            img = image_for_cell(ep.cell)
            source = "cell_capture" if img else "axtree_text"
        elif img is None:
            source = "axtree_text"
        first = ep.steps[0] if ep.steps else None
        key = ep.key if per_episode else (cell_key or "unattributed")
        rows.append({
            "episode": ep.key,
            "dir": ep.dir,
            "cell": cell_key,
            "era": ep.era,
            "task_id": ep.task_id,
            "conditioning_key": key,
            "conditioning_image": img,
            "conditioning_source": source,
            "axtree_sha1": _sha1(first.axtree) if first else None,
            "axtree_chars": len(first.axtree) if first else 0,
            "goal": (first.goal[:300] if first else ""),
            "frozen_for_episode": True,       # 4.3
        })
        counts[source] += 1
    manifest = {"created": datetime.datetime.now().isoformat(),
                "granularity": "per_episode" if per_episode else "per_cell",
                "n_episodes": len(rows), "by_source": dict(counts),
                "episodes": rows}
    if out_json:
        d = os.path.dirname(os.path.abspath(out_json))
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(out_json, "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        manifest["out_json"] = out_json
    if not quiet:
        print("conditioning manifest: %d episodes  %s"
              % (len(rows), dict(counts)))
        if counts["episode_screenshot"] == 0 and counts["cell_capture"] == 0:
            print("  NOTE: no image resolved for any episode. The vision "
                  "conditioning of 4.2 is unavailable from the trajectories; "
                  "run a capture pass into %s or condition on the AXTree "
                  "signature instead." % paths.OUT_CAPTURE)
    return manifest


# --------------------------------------------------------------------------
# Verifier-filtered BC (4.4 "cheap upgrade that isn't RL")
# --------------------------------------------------------------------------

def scan_rollout_dir(root, era=None, recursive=True):
    """Enumerate episodes under an arbitrary rollout root (not TimeTraj).

    Phase 4 rolls the *current policy* out and keeps what verifies. Those
    directories are AgentLab study dirs with the same per-episode layout, so the
    same reader works; `era` is needed only to attribute tasks to cells.
    """
    out = []
    if not os.path.isdir(root):
        raise IOError("rollout root %s does not exist" % root)
    stack = [root]
    seen = set()
    while stack:
        d = stack.pop()
        if d in seen:
            continue
        seen.add(d)
        if os.path.isdir(os.path.join(d, "training_data")):
            era_i = era
            if era_i is None:
                m = re.search(r"version_(\d+)", d)
                era_i = int(m.group(1)) if m else None
            ep = load_episode(d, era=era_i)
            if ep is not None:
                out.append(ep)
            continue
        if not recursive:
            continue
        try:
            for name in sorted(os.listdir(d)):
                p = os.path.join(d, name)
                if os.path.isdir(p):
                    stack.append(p)
        except OSError:
            pass
    out.sort(key=lambda e: e.dir)
    return out


def filtered_bc_corpus(rollout_root, out_json, min_reward=1.0, era=None,
                       require_answer=True, require_no_error=True,
                       think=True, plan=True, memory=True, cells_filter=None,
                       stage_dir=None, keep_stage=False, register=False,
                       dataset_name=None, python=None):
    """Rejection-sampling BC: keep only rollouts the verifier accepts (4.4).

    This is *not* RL -- there is no policy gradient and no importance weight; it
    is ordinary supervised training on a verifier-selected subset. Its value is
    that it turns the deterministic terminal reward into training signal on eras
    where the teacher's plans transfer badly.

    On the TimeTraj teacher corpus this is nearly a no-op: every episode with a
    reward has `cum_reward == 1.0` (they were filtered upstream), so the filter
    removes nothing and the corpus equals `build_sharegpt`. Its real use is a
    *model's own* rollout directory in Phase 4.

    The returned stats carry `verify_rate` -- the number adapterCL.md 11.5 asks
    for ("is the verifier's terminal reward dense enough for filtered BC?").
    Below roughly 0.1 on a held-out era, filtered BC cannot bootstrap and the
    honest answer to 11.5 is "no".
    """
    paths.ensure_out_dirs()
    episodes = scan_rollout_dir(rollout_root, era=era)
    if not episodes:
        raise ValueError("no episodes with training_data/ under %s"
                         % rollout_root)
    if cells_filter:
        want = set(c if isinstance(c, str) else c.key for c in cells_filter)
        episodes = [e for e in episodes if e.cell and e.cell.key in want]

    by_cell = collections.OrderedDict()
    kept, rejected = [], collections.Counter()
    for ep in episodes:
        ck = ep.cell.key if ep.cell else "unattributed"
        st = by_cell.setdefault(ck, {"seen": 0, "verified": 0, "kept": 0,
                                     "steps_kept": 0, "reward_sum": 0.0,
                                     "reward_n": 0})
        st["seen"] += 1
        if ep.reward is not None:
            st["reward_sum"] += float(ep.reward)
            st["reward_n"] += 1
        ok_reward = ep.reward is not None and float(ep.reward) >= min_reward
        if ok_reward:
            st["verified"] += 1
        if not ok_reward:
            rejected["reward<%.3g" % min_reward] += 1
            continue
        if require_no_error and ep.errored():
            rejected["err_msg"] += 1
            continue
        if require_answer and not ep.answered():
            rejected["no_final_answer"] += 1
            continue
        kept.append(ep)
        st["kept"] += 1
        st["steps_kept"] += ep.n_bc_steps

    n_seen = len(episodes)
    n_ver = sum(s["verified"] for s in by_cell.values())
    verify_rate = float(n_ver) / n_seen if n_seen else 0.0
    for st in by_cell.values():
        st["verify_rate"] = (float(st["verified"]) / st["seen"]) if st["seen"] else 0.0
        st["mean_reward"] = (st["reward_sum"] / st["reward_n"]) if st["reward_n"] else None

    print("filtered BC over %s" % rollout_root)
    print("  episodes seen      : %d" % n_seen)
    print("  verified (r>=%.3g) : %d  (verify rate %.3f)  <-- 11.5"
          % (min_reward, n_ver, verify_rate))
    print("  kept after filters : %d  (%d BC steps)"
          % (len(kept), sum(e.n_bc_steps for e in kept)))
    if rejected:
        print("  rejected           : %s" % dict(rejected))
    if n_seen and len(kept) == n_ver == n_seen:
        print("  NOTE: the filter removed nothing -- every episode verifies. On "
              "the TimeTraj teacher corpus this is expected (4.4); filtered BC "
              "only does work on self-generated rollouts.")
    if verify_rate < 0.1:
        print("  WARNING: verify rate %.3f < 0.10. adapterCL.md 11.5's failure "
              "case: filtered BC cannot bootstrap at this density." % verify_rate)

    if not kept:
        raise ValueError("filtered BC kept 0 episodes (verify rate %.3f). "
                         "Nothing to train on." % verify_rate)

    out_json = os.path.abspath(out_json)
    stage_dir = stage_dir or os.path.join(paths.OUT_CORPUS, "_stage",
                                          "filtered_" + _tag_from_path(out_json))
    stage_episodes(kept, stage_dir)
    try:
        cmd, _log = run_converter(stage_dir, out_json, think=think, plan=plan,
                                  memory=memory, python=python)
    finally:
        if not keep_stage and os.path.isdir(stage_dir):
            shutil.rmtree(stage_dir)
    with open(out_json) as fh:
        corpus = json.load(fh)

    stats = {
        "out_json": out_json,
        "created": datetime.datetime.now().isoformat(),
        "kind": "filtered_bc",
        "rollout_root": os.path.abspath(rollout_root),
        "min_reward": min_reward,
        "require_answer": require_answer,
        "require_no_error": require_no_error,
        "n_seen": n_seen,
        "n_verified": n_ver,
        "verify_rate": verify_rate,
        "n_kept": len(kept),
        "n_samples": len(corpus),
        "rejected": dict(rejected),
        "per_cell": by_cell,
        "converter": {"cmd": cmd, "think": think, "plan": plan,
                      "memory": memory},
        "episodes": [e.key for e in kept],
    }
    with open(_stats_path(out_json), "w") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)
    name = dataset_name or ("adaptercl_" + _tag_from_path(out_json))
    stats["dataset_name"] = name
    register_dataset(name, out_json, dataset_info=local_dataset_info())
    if register:
        register_dataset(name, out_json, dataset_info=paths.DATASET_INFO)
    return stats


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _cells_arg(spec):
    """'v3' | 'versions' | 'wiki_e2,news_e3' | 'cells' | 'era:2' | 'env:wiki'.

    'versions' (6 Versions) is the default adapter unit (2); 'cells' is the
    18-way crossing 6.2 needs. Bare 'all' means versions, because that is the
    primary experiment -- ask for 'cells' explicitly.
    """
    if not spec or spec in ("all", "versions"):
        return list(cells.ALL_VERSIONS)
    if spec == "cells":
        return list(cells.ALL_CELLS)
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("era:"):
            out.extend(cells.cells_for(None, [int(part[4:])]))
        elif re.match(r"^v[1-6]$", part):
            out.append(cells.Version.parse(part))
        elif part.startswith("env:"):
            out.extend(cells.cells_for([part[4:]], None))
        else:
            out.append(cells.Cell.parse(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Per-cell BC corpus layer (adapterCL.md 4.3/4.4)")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("census", help="per-cell BC data census (reads only)")
    p.add_argument("--min-episodes", type=int, default=MIN_EPISODES_PHASE2)
    p.add_argument("--json", help="also write the census as JSON here")

    p = sub.add_parser("episode", help="dump one parsed episode")
    p.add_argument("dir")

    p = sub.add_parser("cell", help="list a cell's episodes")
    p.add_argument("cell")
    p.add_argument("--all", action="store_true", help="include unverified")

    p = sub.add_parser("sharegpt", help="build one pooled ShareGPT corpus")
    p.add_argument("cells")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--no-plan", action="store_true")
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--all", action="store_true", help="include unverified")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--register", action="store_true",
                   help="also register in the PARENT LLaMA-Factory "
                        "dataset_info.json (writes to the shared repo)")

    p = sub.add_parser("cell-corpora", help="one corpus per cell + manifest")
    p.add_argument("cells", nargs="?", default="all")
    p.add_argument("--tag", default="percell")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--no-plan", action="store_true")
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--max-episodes", type=int, default=None)

    p = sub.add_parser("conditioning", help="conditioning manifest for cells")
    p.add_argument("cells", nargs="?", default="all")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--per-cell", action="store_true",
                   help="collapse the conditioning key to the cell")

    p = sub.add_parser("inject-manual",
                       help="scout_plan.md arm A: copy a corpus with the "
                            "version's site manual in every sample's prompt")
    p.add_argument("json_in")
    p.add_argument("--manual", required=True, help="manual_full.md for THIS "
                                                   "corpus's version")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--register", default=None, metavar="NAME",
                   help="also register under this dataset_info name "
                        "(e.g. adaptercl_scout_ver_v2)")

    p = sub.add_parser("filtered", help="verifier-filtered BC corpus (4.4)")
    p.add_argument("rollout_root")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--min-reward", type=float, default=1.0)
    p.add_argument("--era", type=int, default=None)
    p.add_argument("--no-require-answer", action="store_true")
    p.add_argument("--allow-error", action="store_true")

    args = ap.parse_args(argv)
    cmd = args.cmd or "census"

    if cmd == "census":
        cen = census(min_episodes=getattr(args, "min_episodes",
                                          MIN_EPISODES_PHASE2))
        print_census(cen)
        if getattr(args, "json", None):
            with open(args.json, "w") as fh:
                json.dump(cen, fh, indent=2, sort_keys=True, default=str)
            print("\nwrote %s" % args.json)
        return 0

    if cmd == "inject-manual":
        with open(args.manual) as fh:
            manual = fh.read()
        info = inject_manual(args.json_in, args.out, manual)
        print("wrote %s" % info["out"])
        print("  %d sample(s), %d injected, manual %d chars (sha1 %s)"
              % (info["n_samples"], info["n_injected"], info["manual_chars"],
                 info["manual_sha1"][:12]))
        if args.register:
            r = register_dataset(args.register, info["out"])
            print("  registered %s in %s" % (r["name"], r["dataset_info"]))
        return 0

    if cmd == "episode":
        ep = load_episode(args.dir)
        if ep is None:
            print("could not load %s" % args.dir)
            return 1
        print(json.dumps(ep.as_dict(), indent=2, sort_keys=True))
        for st in ep.steps:
            print("  step %-2d  axtree=%6dch  tags=%s  action=%s"
                  % (st.index, len(st.axtree),
                     "".join(t[0] for t in TAGS if st.sections.get(t)),
                     st.action.replace("\n", " ")[:60]))
        return 0

    if cmd == "cell":
        c = cells.Cell.parse(args.cell)
        eps = load_cell(c, verified_only=not args.all)
        print("%s (%s): %d episodes, %d BC steps"
              % (c.key, c.theme, len(eps), sum(e.n_bc_steps for e in eps)))
        for ep in eps:
            print("  %-70s task=%-4s seed=%-4s r=%-4s steps=%d"
                  % (ep.name[:70], ep.task_id, ep.seed, ep.reward,
                     ep.n_bc_steps))
        return 0

    if cmd == "sharegpt":
        stats = build_sharegpt(
            _cells_arg(args.cells), args.out, think=not args.no_think,
            plan=not args.no_plan, memory=not args.no_memory,
            verified_only=not args.all, max_episodes=args.max_episodes,
            register=args.register)
        print(json.dumps(dict((k, v) for k, v in stats.items()
                              if k != "episodes"), indent=2, sort_keys=True))
        return 0

    if cmd == "cell-corpora":
        man = build_cell_corpora(
            _cells_arg(args.cells), out_dir=args.out_dir, tag=args.tag,
            think=not args.no_think, plan=not args.no_plan,
            memory=not args.no_memory, max_episodes=args.max_episodes)
        print("manifest: %s" % man["manifest"])
        for k, v in man["cells"].items():
            print("  %-10s %5d samples  %s" % (k, v["n_samples"], v["json"]))
        for k, v in man["skipped"].items():
            print("  %-10s SKIPPED: %s" % (k, v.split("\n")[0]))
        return 0

    if cmd == "conditioning":
        eps = []
        for c in _cells_arg(args.cells):
            eps.extend(load_cell(c))
        out = args.out or os.path.join(paths.OUT_CORPUS, "conditioning.json")
        attach_conditioning(eps, per_episode=not args.per_cell, out_json=out)
        print("wrote %s" % out)
        return 0

    if cmd == "filtered":
        stats = filtered_bc_corpus(args.rollout_root, args.out,
                                   min_reward=args.min_reward, era=args.era,
                                   require_answer=not args.no_require_answer,
                                   require_no_error=not args.allow_error)
        print("  wrote %s (%d samples), stats in %s"
              % (stats["out_json"], stats["n_samples"],
                 _stats_path(stats["out_json"])))
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
