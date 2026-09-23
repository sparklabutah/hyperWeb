"""One entry point for the whole project: `python -m adaptercl <stage> ...`.

The modules here need three different interpreters -- the analysis layer runs on
the system python, the modelling layer needs torch (`llamafactory`), the capture
layer needs playwright (`tw_r1_q3`), and serving needs `vllm_q35`. Getting that
wrong produces confusing ImportErrors deep inside a stage, so this dispatcher
knows which env each stage wants and **re-execs itself** under the right
interpreter instead of failing. Run it with anything; it will land in the right
place. `--where` prints the interpreter and exits, if you would rather run the
command yourself.

    python3 -m adaptercl status          # what is done, what is blocking
    python3 -m adaptercl selftest        # every check that needs no GPU
    python3 -m adaptercl cells           # the 3 x 6 grid + BC data census
    python3 -m adaptercl survey          # 4.5: layer types + injection sites
    python3 -m adaptercl style           # 6.2 precondition: era correspondence
    python3 -m adaptercl transfer --selftest

Nothing here launches GPU work, env servers or evals. The stages that *can*
(`train`, `percell run`, `eval`) are dry-run by default and require their own
explicit flag.
"""

from __future__ import print_function

import os
import subprocess
import sys

from . import paths

#: stage -> (module, interpreter attr on paths, phase, one-line help)
STAGES = [
    ("paths",       "paths",          None,       "0", "check every path this project depends on"),
    ("cells",       "cells",          None,       "0", "the 3x6 environment x era grid + BC data census"),
    ("targets",     "targets",        None,       "0", "Qwen3.5 injection sites and target-set budgets (4.5)"),
    ("survey",      "survey",         "PY_TRAIN", "0", "architecture survey incl. a meta-device module walk (8.2)"),
    ("style",       "era_style",      None,       "0", "6.2 precondition: do era styles correspond across sites?"),
    ("capture",     "capture",        "PY_BENCH", "0", "deterministic screenshot + a11y-role capture (4.2)"),
    ("scout",       "scout",          "PY_BENCH", "0", "version and task-grounded site scouting"),
    ("verify",      "verify_adapter", None,       "0", "prove the serving stack actually applies an adapter (4.5)"),
    ("census",      "bcdata",         None,       "2", "BC corpus census and corpus building"),
    ("percell",     "percell",        None,       "2", "per-cell LoRA sweep planning (Phase 1/2)"),
    ("transfer",    "transfer",       None,       "2", "the functional transfer matrix and dissociation (6.2/8.5)"),
    ("probe",       "probe",          None,       "2", "6.7 diagnostics: identity probe, collapse, controls"),
    ("encoder",     "encoder",        "PY_TRAIN", "3", "conditioning encoder: vision / structure / both (4.2)"),
    ("hypernet",    "hypernet",       "PY_TRAIN", "3", "generator family: free / mixture / basis (6.5)"),
    ("diagnostics", "diagnostics",    "PY_TRAIN", "3", "generator-specific health checks (6.7)"),
    ("train",       "train_hypernet", "PY_TRAIN", "3", "end-to-end BC training of the generator (4.4/8.4)"),
    ("eval",        "evalbridge",     None,       "*", "build and read evaluations through the repo harness"),
    ("toy",         "toy",            "PY_TRAIN", "*", "the tiny fake Qwen3.5 model used by every selftest"),
]

STAGE_BY_NAME = dict((s[0], s) for s in STAGES)

PHASE_LABEL = {
    "0": "phase 0 -- harness, survey, preconditions",
    "2": "phase 1/2 -- adapters, transfer matrix, ablations",
    "3": "phase 3 -- hypernetwork",
    "*": "any phase",
}

#: (stage, extra args) for every check that runs without a GPU
SELFTESTS = [
    ("targets", []),
    ("scout", ["selftest"]),
    ("cells", []),
    ("transfer", ["--selftest"]),
    ("probe", ["--selftest"]),
    ("toy", ["--selftest"]),
    ("encoder", ["--selftest"]),
    ("hypernet", ["--selftest"]),
    ("diagnostics", ["--selftest"]),
    ("train", ["--smoke"]),
]


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

def interpreter_for(stage):
    """Absolute python a stage needs, or None if any interpreter will do."""
    spec = STAGE_BY_NAME.get(stage)
    if spec is None:
        return None
    return getattr(paths, spec[2]) if spec[2] else None


def env_label(py):
    """'llamafactory' from '/.../envs/llamafactory/bin/python'."""
    return os.path.basename(os.path.dirname(os.path.dirname(py)))


def _same_interpreter(path):
    try:
        return os.path.samefile(path, sys.executable)
    except OSError:
        return os.path.abspath(path) == os.path.abspath(sys.executable)


def _child_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = paths.PROJECT + os.pathsep + env.get("PYTHONPATH", "")
    return env


def dispatch(stage, argv, where_only=False):
    """Run one stage's module main, re-execing under its interpreter if needed."""
    spec = STAGE_BY_NAME.get(stage)
    if spec is None:
        raise SystemExit("unknown stage %r. Run `python3 -m adaptercl --help`."
                         % (stage,))
    module = "adaptercl." + spec[1]
    want = interpreter_for(stage)
    py = want or sys.executable

    if where_only:
        print("%s -m %s %s" % (py, module, " ".join(argv)))
        return 0

    if want and not _same_interpreter(want):
        if not os.path.exists(want):
            raise SystemExit(
                "stage %r needs the interpreter %s, which is missing.\n"
                "That env holds this project's torch/transformers stack; the "
                "env map is in FINDINGS.md Appendix A.6." % (stage, want))
        cmd = [want, "-m", module] + list(argv)
        return subprocess.call(cmd, cwd=paths.PROJECT, env=_child_env())

    import runpy
    sys.argv = [module] + list(argv)
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as e:
        return e.code or 0
    return 0


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------

def adapter_verification_state():
    """(passed, detail) for the 4.5 serving-stack check.

    Deliberately reads the artefact rather than testing for its existence.
    `verify_adapter e2e` writes this file even when it only *plans* the check --
    with an honest `blocking` entry saying the vLLM leg did not run -- so
    existence alone would clear the one gate that exists to stop an ignored
    adapter being mistaken for a real result. Passing means: the file is there,
    it says `ok`, and its `blocking` list is empty.
    """
    p = os.path.join(paths.OUT_EVAL, "adapter_verification.json")
    if not os.path.exists(p):
        return False, "not run"
    try:
        import json
        with open(p) as fh:
            d = json.load(fh)
    except (ValueError, IOError) as exc:
        return False, "unreadable (%s)" % (exc,)
    blocking = d.get("blocking") or []
    if d.get("ok") and not blocking:
        checks = d.get("checks") or []
        return True, "%d check(s) passed" % len(checks)
    if blocking:
        return False, blocking[0].split(".")[0]
    return False, "reported not ok"


def _count_adapters(root):
    """Directories under `root` that are actually PEFT adapters.

    Counting bare directory entries would count the `_gen_*.yaml` plan files the
    dry-run modes emit, and report "per-cell adapters: done" for a sweep that has
    not trained anything. An adapter is a directory with an adapter_config.json.
    """
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if "adapter_config.json" in filenames:
            n += 1
            dirnames[:] = []          # do not descend into an adapter
    return n


def _count_matrices(root):
    """Saved transfer matrices (JSON with a `cells` key), not stray files."""
    import json
    n = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            if not f.endswith(".json"):
                continue
            try:
                with open(os.path.join(dirpath, f)) as fh:
                    d = json.load(fh)
            except (ValueError, IOError):
                continue
            if isinstance(d, dict) and "cells" in d and "matrix" in d:
                n += 1
    return n


def status(fh=None):
    """What is built, what has been run, and what is currently blocking."""
    fh = fh or sys.stdout
    from . import cells

    print("adapterCL project status", file=fh)
    print("=" * 70, file=fh)

    here = os.path.dirname(os.path.abspath(__file__))
    mods = sorted(f[:-3] for f in os.listdir(here)
                  if f.endswith(".py") and not f.startswith("_"))
    print("\nmodules (%d): %s" % (len(mods), ", ".join(mods)), file=fh)

    print("\ninterpreters:", file=fh)
    for label, p in (("analysis (current)", sys.executable),
                     ("torch    (llamafactory)", paths.PY_TRAIN),
                     ("serve    (vllm_q35)", paths.PY_VLLM),
                     ("bench    (tw_r1_q3)", paths.PY_BENCH),
                     ("web      (tw_web)", paths.PY_WEB)):
        print("  %-26s %-4s %s" % (label, "ok" if os.path.exists(p) else "MISS", p),
              file=fh)

    snap = paths.hf_snapshot(paths.BASE_MODEL)
    print("\nbase model %s:\n  %s" % (paths.BASE_MODEL,
                                      snap or "NOT CACHED LOCALLY"), file=fh)

    print("\nartefacts:", file=fh)
    checks = [
        ("0", "architecture survey",
         os.path.join(paths.OUT_SURVEY, "survey_Qwen-Qwen3.5-9B.json")),
        ("0", "era-style correspondence",
         os.path.join(paths.OUT_STYLE, "era_style.json")),
        ("0", "capture manifest",
         os.path.join(paths.OUT_CAPTURE, "manifest.json")),
    ]
    for phase, label, p in checks:
        done = os.path.exists(p)
        extra = ""
        if done and os.path.isdir(p):
            n = len([x for x in os.listdir(p) if not x.startswith(".")])
            extra = " (%d entr%s)" % (n, "y" if n == 1 else "ies")
            done = n > 0
        print("  phase %s  %-26s %-5s %s%s"
              % (phase, label, "done" if done else "--", p, extra), file=fh)

    # These three count *products*, not directory entries: the dry-run modes
    # emit plan files (`_gen_*.yaml`, `*.todo`) into the same trees, and
    # counting entries would report a sweep as trained when nothing has run.
    for phase, label, root, counter, one, many in (
            ("2", "per-cell adapters", paths.OUT_CELLS, _count_adapters,
             "adapter", "adapters"),
            ("2", "transfer matrix", paths.OUT_TRANSFER, _count_matrices,
             "matrix", "matrices"),
            ("3", "generated adapters", paths.OUT_HYPERNET, _count_adapters,
             "adapter", "adapters")):
        n = counter(root) if os.path.isdir(root) else 0
        print("  phase %s  %-26s %-5s %s  [%d %s]"
              % (phase, label, "done" if n else "--", root, n,
                 one if n == 1 else many), file=fh)
    verified, detail = adapter_verification_state()
    print("  phase 0  %-26s %-5s %s  [%s]"
          % ("adapter verification", "done" if verified else "--",
             os.path.join(paths.OUT_EVAL, "adapter_verification.json"), detail),
          file=fh)

    print("\nblocking:", file=fh)
    blocking = []
    # The default adapter unit is the VERSION (2), not the cell. Report the
    # version census as the blocker and the cell census only as a caveat on
    # 6.2's crossed matrix -- otherwise the two empty cells read as blocking the
    # whole project when they only affect one experiment.
    vav = cells.version_availability()
    if vav["_missing"]:
        blocking.append(
            "BC data: %d of 6 versions below %d verified episodes (%s)."
            % (len(vav["_missing"]), vav["_min_episodes"],
               ", ".join(vav["_missing"])))
    av = cells.availability()
    if av["_missing"]:
        print("  (note) %d of 18 cells are short of BC data (%s). This does NOT "
              "block the primary version-level experiment -- only 6.2's crossed "
              "transfer matrix. Fill with scripts/collect_missing_cells.sh."
              % (len(av["_missing"]), ", ".join(av["_missing"])), file=fh)
    if not verified:
        blocking.append(
            "adapter application through vLLM is UNVERIFIED (%s). 4.5's "
            "silent-failure class: an ignored adapter reads as base-model "
            "performance with no error. Run `bash scripts/run_phase0.sh "
            "VERIFY=1 GPU_JOB=<id>` before trusting any adapter eval number."
            % (detail,))
    if not os.path.exists(os.path.join(paths.OUT_STYLE, "era_style.json")):
        blocking.append(
            "6.2 precondition unchecked: run `python3 -m adaptercl style`.")
    if not blocking:
        print("  none", file=fh)
    for b in blocking:
        print("  * " + b, file=fh)
    return blocking


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def selftest(only=None, fh=None, quiet=False):
    """Run every check that needs no GPU, in the right interpreter each time."""
    fh = fh or sys.stdout
    results = []
    for stage, args in SELFTESTS:
        if only and stage not in only:
            continue
        py = interpreter_for(stage) or sys.executable
        if not os.path.exists(py):
            results.append((stage, "SKIP", "interpreter missing: %s" % py))
            continue
        cmd = [py, "-m", "adaptercl." + STAGE_BY_NAME[stage][1]] + args
        if not quiet:
            print("--- %-12s [%s]" % (stage, env_label(py)), file=fh)
        proc = subprocess.Popen(cmd, cwd=paths.PROJECT, env=_child_env(),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        out, _ = proc.communicate()
        out = out.decode("utf-8", "replace")
        if not quiet:
            tail = out.strip().splitlines()[-5:]
            for line in tail:
                print("    " + line, file=fh)
        results.append((stage, "PASS" if proc.returncode == 0 else "FAIL",
                        "rc=%d" % proc.returncode))

    print("\n" + "=" * 70, file=fh)
    n_fail = 0
    for stage, verdict, note in results:
        print("  %-14s %-5s %s" % (stage, verdict, note), file=fh)
        n_fail += 1 if verdict == "FAIL" else 0
    print("\n%d/%d passed" % (len(results) - n_fail, len(results)), file=fh)
    return n_fail


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def print_usage(fh=None):
    fh = fh or sys.stdout
    fh.write("usage: python3 -m adaptercl <stage> [args...]\n\n"
             "adapterCL -- conditioned adapter synthesis for TimeWarp web agents.\n"
             "Implements adapter_project/adapterCL.md. Each stage re-execs itself\n"
             "under the interpreter it needs, so this works from any python.\n")
    last = None
    for name, mod, env_attr, phase, help_ in STAGES:
        if phase != last:
            fh.write("\n  %s\n" % PHASE_LABEL.get(phase, phase))
            last = phase
        env = "  [%s]" % env_label(getattr(paths, env_attr)) if env_attr else ""
        fh.write("    %-12s %s%s\n" % (name, help_, env))
    fh.write("\n  meta\n"
             "    status       what is built, what has been run, what is blocking\n"
             "    selftest     run every GPU-free check across all interpreters\n"
             "\n"
             "Pass --help to any stage for its own options, or --where to print\n"
             "the interpreter and command instead of running it.\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print_usage()
        return 0
    stage, rest = argv[0], argv[1:]
    where_only = "--where" in rest
    if where_only:
        rest = [a for a in rest if a != "--where"]
    if stage == "status":
        status()
        return 0
    if stage == "selftest":
        return 1 if selftest(only=set(rest) or None) else 0
    return dispatch(stage, rest, where_only=where_only)


if __name__ == "__main__":
    sys.exit(main() or 0)
