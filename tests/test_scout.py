#!/usr/bin/env python3
"""Contract tests for the scout layer (scout_plan.md Part E).

Every check here guards a seam whose failure mode is a plausible NUMBER rather
than a crash -- which is the only kind worth a test in this project:

  1. The manual schema. All seven sections, in order, with the missing ones
     filled, or `strip_sections` compares structurally different objects across
     versions and the affordance/style ablation is not mechanical.
  2. `strip_sections` is exact: affordance drops 7 and nothing else, style keeps
     7 and nothing else.
  3. The leak checker catches a planted `v3`, a planted `2005` in an affordance
     section, and a planted task-intent 6-gram -- and does NOT fire on the same
     manual clean. A checker nobody tested returns 0 hits for the wrong reason.
     The one waiver (a forbidden token inside a UI label the crawl saw) fires
     only when the label was actually observed.
  4. **The training prompt and the eval prompt put the manual in the same
     slot.** This is the load-bearing one: arm A trains on manual-injected
     corpus samples and is evaluated through `extra_instructions`. If the two
     renderings differ, A is trained on a prompt it is never evaluated on and
     the A-vs-C comparison stops being one-variable. The test reconstructs the
     eval string from AgentLab's own prompt template and diffs it against the
     injected corpus sample.
  5. The three copies of the framing string (scout.py, bcdata.py,
     benchmark_adapter.py) agree, checked by reading the third off disk.
  6. The descriptions JSON is exactly `t2l.load_descriptions`'s format.
  7. The crawler is deterministic and budget-bounded, exercised against a fake
     in-memory site (no browser, no server), including the fixed search probe
     and the port-independence of every recorded path.
  8. `--site-manual` reaches TW_SITE_MANUAL, is recorded in the run log, and
     preflight refuses a manual that does not exist.
  9. `gen_yaml_holdout.py --dataset-prefix` pools the manual corpora and its
     post-hoc leak check still refuses the held-out version.

Run: /usr/bin/python3 adapter_project/tests/test_scout.py
No GPU, no weights, no network, no server; writes only inside a tempdir.
"""
from __future__ import print_function

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
REPO = os.path.dirname(PROJECT)
sys.path.insert(0, PROJECT)

from adaptercl import bcdata, cells, evalbridge, scout  # noqa: E402

PASS = [0]
FAIL = [0]
SKIP = [0]


def check(name, cond, extra=""):
    if cond:
        PASS[0] += 1
        print("  ok   %s%s" % (name, (" " + extra) if extra else ""))
    else:
        FAIL[0] += 1
        print("  FAIL %s%s" % (name, (" " + extra) if extra else ""))


def skip(name, why):
    SKIP[0] += 1
    print("  skip %s (%s)" % (name, why))


SAMPLE_MANUAL = """# Site manual: wiki
## 1. Purpose and top-level navigation
A left-hand column of links leads to the main page, recent changes and history.
## 2. Search
A text box in the header with a Go button next to it; results are a list of
article titles.
## 3. Page anatomy
Article pages end with a Related Pages list.
## 4. Lists, pagination, sorting, filters
Long lists continue with a next-page link at the bottom.
## 5. Forms and multi-step flows
Only the search form.
## 6. Gotchas
The language names in the header are plain text, not links.
## 7. Visual style
Nested tables, a serif face at a fixed size, roughly 2001.
"""


# --------------------------------------------------------------------------
# 1-2. schema and ablation
# --------------------------------------------------------------------------

def test_schema():
    print("[1] manual schema")
    text = scout.normalize_manual(SAMPLE_MANUAL, "wiki")
    title, body = scout.parse_manual(text)
    check("title names the env", title == "# Site manual: wiki")
    check("all 7 sections present", sorted(body) == [1, 2, 3, 4, 5, 6, 7])
    order = [int(m.group(1)) for m in re.finditer(r"^## (\d)\.", text, re.M)]
    check("sections are in schema order", order == [1, 2, 3, 4, 5, 6, 7])
    headings = re.findall(r"^## \d\. (.*)$", text, re.M)
    check("headings are the plan's, verbatim",
          headings == [h for _, h in scout.SECTIONS])

    partial = scout.normalize_manual(
        "# Site manual: shop\n## 2. Search\nA search box.\n", "shop")
    _, pbody = scout.parse_manual(partial)
    check("a summariser that skips sections still yields 7",
          sorted(pbody) == [1, 2, 3, 4, 5, 6, 7]
          and pbody[1] == scout.EMPTY_SECTION and pbody[2] == "A search box.")
    check("out-of-order input is reordered",
          [int(m.group(1)) for m in re.finditer(
              r"^## (\d)\.", scout.normalize_manual(
                  "# x\n## 7. Visual style\ns\n## 1. Purpose\np\n", "wiki"),
              re.M)] == [1, 2, 3, 4, 5, 6, 7])


def test_strip_sections():
    print("[2] mechanical ablation")
    text = scout.normalize_manual(SAMPLE_MANUAL, "wiki")
    aff, sty = scout.affordance_only(text), scout.style_only(text)
    a_secs = [int(m.group(1)) for m in re.finditer(r"^## (\d)\.", aff, re.M)]
    s_secs = [int(m.group(1)) for m in re.finditer(r"^## (\d)\.", sty, re.M)]
    check("affordance == sections 1-6", a_secs == [1, 2, 3, 4, 5, 6])
    check("style == section 7 only", s_secs == [7])
    check("affordance keeps the bodies byte-identical",
          "Related Pages list." in aff and "Nested tables" not in aff)
    check("style keeps its body byte-identical",
          "Nested tables, a serif face at a fixed size, roughly 2001." in sty)
    check("full + affordance + style partition the sections",
          set(a_secs) | set(s_secs) == set(n for n, _ in scout.SECTIONS)
          and not set(a_secs) & set(s_secs))
    check("stripping is idempotent", scout.affordance_only(aff) == aff)


# --------------------------------------------------------------------------
# 3. leak check
# --------------------------------------------------------------------------

def test_leak_check():
    print("[3] leak check (B.3)")
    intents = ["List all articles from the Related Pages sections of the "
               "Biophysics and Biochemistry pages"]
    clean = scout.normalize_manual(SAMPLE_MANUAL, "wiki")
    r = scout.leak_check(clean, sections="full", task_intents=intents,
                         version_key="v1")
    check("the clean sample passes as a full manual", not r["hits"],
          str([h["rule"] for h in r["hits"]]))

    planted = clean.replace("A left-hand", "Unlike v3, a left-hand")
    check("a planted version key is caught",
          any(h["rule"] == "1-version-key"
              for h in scout.leak_check(planted)["hits"]))
    planted = clean.replace("A left-hand", "Introduced in 2005, a left-hand")
    check("a planted year in an affordance section is caught",
          any(h["rule"] == "2-year-in-affordance"
              for h in scout.leak_check(planted, sections="affordance")["hits"]))
    check("the SAME year in section 7 is allowed",
          not any(h["rule"] == "2-year-in-affordance"
                  for h in scout.leak_check(scout.style_only(clean),
                                            sections="style")["hits"]))
    planted = clean.replace(
        "A left-hand",
        "List all articles from the Related Pages sections of the Biophysics "
        "and Biochemistry pages. A left-hand")
    check("a planted task-intent 6-gram is caught",
          any(h["rule"] == "3-task-overlap"
              for h in scout.leak_check(planted, task_intents=intents)["hits"]))
    planted = clean.replace("A left-hand", "A more modern, left-hand")
    check("a comparative term is caught",
          any(h["rule"] == "1-comparative"
              for h in scout.leak_check(planted)["hits"]))
    theme = cells.THEME_NAME["shop"][3]
    planted = clean.replace("A left-hand", "Theme %s. A left-hand" % theme)
    check("a theme name is caught",
          any(h["rule"] == "1-theme-name"
              for h in scout.leak_check(planted)["hits"]))
    planted = clean.replace("A left-hand", "The agent usually fails here. "
                                           "A left-hand")
    check("a statement about the agent is caught",
          any(h["rule"] == "4-performance"
              for h in scout.leak_check(planted)["hits"]))
    planted = clean.replace("A left-hand", "This is the newest one. A left-hand")
    check("'newest' is caught on the neutral era",
          any(h["rule"] in ("1-comparative", "5-neutral-era")
              for h in scout.leak_check(planted,
                                        version_key="v%d" % cells.NEUTRAL_ERA)["hits"]))

    # The one waiver, both directions.
    quoted = clean.replace("A left-hand",
                           "A link labelled Printable version sits below. "
                           "A left-hand")
    r_lab = scout.leak_check(quoted, labels={"printable version"})
    check("a forbidden token inside an OBSERVED label is waived",
          not r_lab["hits"] and r_lab["waived"])
    check("the same token with no such label is a hit",
          any(h["rule"] == "1-comparative"
              for h in scout.leak_check(quoted, labels=set())["hits"]))
    check("the built-in fixture agrees", scout.selftest_leak_fixture()["ok"])

    intents = scout.task_intents()
    check("the real task file yields 231 intents", len(intents) == 231,
          "(got %d)" % len(intents))


# --------------------------------------------------------------------------
# 4-5. train/eval prompt parity
# --------------------------------------------------------------------------

#: AgentLab's own template, copied from
#: AgentLab/src/agentlab/agents/dynamic_prompting.py:499-511 (the `Goal`
#: element) and the observation element that follows it. Copied rather than
#: imported: agentlab lives in tw_r1_q3 and this suite runs on the system
#: python.
def render_eval_prompt(extra_instructions):
    return ("## Extra instructions:\n\n%s\n" % extra_instructions
            + "\n# Observation of current step:")


def test_prompt_parity():
    print("[4] the manual lands in the SAME prompt slot in training and eval")
    manual = scout.normalize_manual(SAMPLE_MANUAL, "wiki")
    base_extra = ("\nIMPORTANT: You must only navigate to URLs within the "
                  "TimeWarp environment. \nDo NOT navigate to external "
                  "websites.\n")

    corpus_prompt = ("# Instructions\n\n## Goal:\nx\n\n"
                     + render_eval_prompt(base_extra))
    injected = bcdata.inject_manual_into_prompt(corpus_prompt, manual)
    eval_rendered = ("# Instructions\n\n## Goal:\nx\n\n"
                     + render_eval_prompt(base_extra + scout.manual_block(manual)))
    check("injected corpus prompt == rendered eval prompt",
          injected == eval_rendered,
          "" if injected == eval_rendered else
          "\n    train: %r\n    eval : %r"
          % (injected[-260:], eval_rendered[-260:]))

    check("injecting twice is refused",
          _raises(bcdata.inject_manual_into_prompt, injected, manual))
    check("a prompt with no extra-instructions section is refused",
          _raises(bcdata.inject_manual_into_prompt, "# Instructions\n", manual))


def _raises(fn, *a):
    try:
        fn(*a)
        return False
    except ValueError:
        return True


def test_inject_manual_roundtrip(tmp):
    print("[4b] inject_manual over a corpus fixture")
    manual = scout.normalize_manual(SAMPLE_MANUAL, "wiki")
    base = ("# Instructions\n\n## Goal:\ng\n\n"
            + render_eval_prompt("\nIMPORTANT: stay on site.\n"))
    fixture = [{"system": "sys",
                "conversations": [{"from": "human", "value": base},
                                  {"from": "gpt", "value": "<action>noop()</action>"}]}
               for _ in range(3)]
    src = os.path.join(tmp, "corpus.json")
    dst = os.path.join(tmp, "corpus_manual.json")
    with open(src, "w") as fh:
        json.dump(fixture, fh)
    info = bcdata.inject_manual(src, dst, manual)
    out = json.load(open(dst))
    check("every sample was injected exactly once",
          info["n_samples"] == 3 and info["n_injected"] == 3
          and all(s["conversations"][0]["value"].count(
              scout.MANUAL_BLOCK_HEADER) == 1 for s in out))
    check("the gpt turns are untouched",
          all(s["conversations"][1]["value"] == "<action>noop()</action>"
              for s in out))
    check("the system field survives", all(s["system"] == "sys" for s in out))

    real = os.path.join(PROJECT, "out", "corpus", "ver", "v2.json")
    if not os.path.exists(real):
        skip("real corpus injection", "%s not built" % real)
        return
    dst2 = os.path.join(tmp, "v2_manual.json")
    info2 = bcdata.inject_manual(real, dst2, manual)
    check("the real v2 corpus injects cleanly (%d samples)" % info2["n_samples"],
          info2["n_injected"] == info2["n_samples"] and info2["n_samples"] > 100)


def test_framing_strings_agree():
    print("[5] the three copies of the framing string agree")
    check("scout and bcdata agree",
          scout.manual_block("X") == bcdata.manual_block("X"))
    bench = os.path.join(PROJECT, "scripts", "benchmark_adapter.py")
    body = open(bench).read()
    check("benchmark_adapter.py reads TW_SITE_MANUAL",
          'os.environ.get("TW_SITE_MANUAL"' in body)
    # The literal in the bench script is what the eval actually appends.
    m = re.search(r'extra_instructions \+= \(\s*\n\s*"([^"]*)" \+ _manual \+ "([^"]*)"',
                  body)
    if not m:
        check("the bench script's framing literal is findable", False)
        return
    prefix = m.group(1).encode().decode("unicode_escape")
    suffix = m.group(2).encode().decode("unicode_escape")
    check("bench framing == scout.manual_block",
          prefix + "X" + suffix == scout.manual_block("X"),
          repr(prefix + "X" + suffix))
    check("the bench script fails loudly on an empty manual",
          "TW_SITE_MANUAL=%s is empty" in body)


# --------------------------------------------------------------------------
# 6. descriptions format
# --------------------------------------------------------------------------

def test_descriptions_format(tmp):
    print("[6] descriptions_*.json is t2l.load_descriptions' format")
    root = os.path.join(tmp, "crawl")
    for era in (1, 6):
        for env in cells.ENVIRONMENTS:
            d = scout.env_dir(era, env, root)
            os.makedirs(d)
            scout.write_text(os.path.join(d, "manual.md"),
                             scout.normalize_manual(SAMPLE_MANUAL, env))
    for era in (1, 6):
        scout.build_version_manuals(era, root=root)
    desc = scout.write_descriptions(root=root, eras=(1, 6))
    for name in ("full", "affordance", "style"):
        p = os.path.join(root, "descriptions_%s.json" % name)
        d = json.load(open(p))
        check("descriptions_%s.json keys are version keys" % name,
              sorted(d) == ["v1", "v6"])
        check("descriptions_%s.json values are strings" % name,
              all(isinstance(v, str) and v.strip() for v in d.values()))
    check("the version manual concatenates all three envs",
          all(desc["full"]["v1"].count("# Site manual: %s" % e) == 1
              for e in cells.ENVIRONMENTS))
    check("the affordance version manual has no section 7",
          "## 7." not in desc["affordance"]["v1"])
    check("the style version manual has only section 7",
          desc["style"]["v1"].count("## 7.") == 3
          and "## 1." not in desc["style"]["v1"])
    try:
        from adaptercl import t2l
    except Exception as e:                             # noqa: BLE001
        skip("t2l.load_descriptions round-trip", "torch not importable here (%s)"
             % type(e).__name__)
        return
    loaded = t2l.load_descriptions(
        os.path.join(root, "descriptions_full.json"))
    check("t2l.load_descriptions reads it unchanged",
          loaded == desc["full"])


# --------------------------------------------------------------------------
# 7. the crawler, against a fake site
# --------------------------------------------------------------------------

SITE = {
    "/": {"title": "Main Page",
          "links": [("Biology", "/wiki/Biology"), ("Physics", "/wiki/Physics"),
                    ("Main Page", "/"), ("Elsewhere", "http://example.com/x")],
          "inputs": [{"type": "text", "name": "search", "label": "",
                      "placeholder": "Search"}]},
    "/wiki/Biology": {"title": "Biology",
                      "links": [("Main Page", "/"),
                                ("Biophysics", "/wiki/Biophysics")]},
    "/wiki/Physics": {"title": "Physics", "links": [("Main Page", "/")]},
    "/wiki/Biophysics": {"title": "Biophysics", "links": []},
    "/search?q=Biology": {"title": "Search results",
                          "links": [("Biology", "/wiki/Biology")]},
}


#: A site whose ROOT IS A PATH and whose other routes are NOT under it -- the
#: webshop's `/abc` session (capture.LAUNCH["shop"]["suffix"]). Every product is
#: behind the search form, so this fixture is also the post-probe expansion case.
SHOP_SITE = {
    "/abc": {"title": "WebShop",
             "links": [],
             "inputs": [{"type": "text", "name": "search_query", "label": "",
                         "placeholder": "Search..."}]},
    "/search_results/abc/q/1": {
        "title": "Results",
        "links": [("B0001", "/item_page/abc/B0001/q/1/%7B%7D"),
                  ("B0002", "/item_page/abc/B0002/q/1/%7B%7D")]},
    "/item_page/abc/B0001/q/1/%7B%7D": {
        "title": "Item B0001", "links": [("Back to Search", "/abc")]},
    "/item_page/abc/B0002/q/1/%7B%7D": {
        "title": "Item B0002", "links": [("Back to Search", "/abc")]},
}


class FakePage(object):
    """The 6 playwright methods `scout.crawl_env` uses, over a static site.

    A fake rather than a real browser so this test needs no server, no chromium
    and no network -- and so the determinism check measures the CRAWLER's
    determinism (frontier order, budget, normalisation) rather than the
    renderer's, which `capture.check_determinism` already owns.
    """

    def __init__(self, base):
        self.base = base
        self.url = base
        self.n_goto = 0

    def goto(self, url, **kw):
        self.n_goto += 1
        path = scout.normalize_path(url, self.base)
        if path not in SITE:
            raise RuntimeError("404 %s" % path)
        self.url = url
        return None

    def _rec(self):
        return SITE[scout.normalize_path(self.url, self.base)]

    def evaluate(self, js, *a):
        rec = self._rec()
        if "querySelectorAll('a[href]')" in js or "a[href]" in js:
            return {"title": rec["title"],
                    "links": [{"label": l, "href": h} for l, h in rec["links"]],
                    "buttons": ["Go"] if rec.get("inputs") else [],
                    "inputs": rec.get("inputs", []),
                    "headings": ["h1: " + rec["title"]],
                    "forms": 1 if rec.get("inputs") else 0}
        if "tagRole" in js or "counts" in js:
            return {"link": len(rec["links"]), "heading": 1}
        return None

    def wait_for_timeout(self, ms):
        return None

    def wait_for_load_state(self, *a, **kw):
        return None

    def aria_snapshot(self):
        rec = self._rec()
        return "\n".join(["- heading \"%s\"" % rec["title"]]
                         + ["- link \"%s\"" % l for l, _ in rec["links"]])

    def locator(self, sel):
        return _FakeLocator(self)

    def screenshot(self, **kw):
        raise RuntimeError("no renderer in the fake page")


class _FakeLocator(object):
    def __init__(self, page):
        self.page = page
        self.value = ""

    def nth(self, i):
        return self

    def fill(self, v):
        self.value = v

    def press(self, key):
        self.page.url = self.page.base + "/search?q=" + self.value


class FakeShopPage(FakePage):
    """FakePage over SHOP_SITE, whose site root is the session path `/abc`."""

    def _rec(self):
        return SHOP_SITE[scout.normalize_path(self.url, self.base)]

    def goto(self, url, **kw):
        self.n_goto += 1
        path = scout.normalize_path(url, self.base)
        if path not in SHOP_SITE:
            raise RuntimeError("404 %s" % path)
        self.url = url
        return None

    def locator(self, sel):
        return _FakeShopLocator(self)


class _FakeShopLocator(_FakeLocator):
    def press(self, key):
        # Flask's /<session_id> route: the results page is served from the HOST
        # root with the session id as an inner segment, not under /abc. Built
        # from the page's OWN origin so the port-independence check is a check
        # on the crawler and not on this fixture.
        origin = self.page.base.rsplit("/", 1)[0]
        self.page.url = origin + "/search_results/abc/q/1"


def test_session_rooted_site():
    print("[7c] a site whose root is a path (the webshop session)")
    base = "http://localhost:5511/abc"
    check("the canonical root path is the session path",
          scout.site_root_path(base) == "/abc")
    check("_abs_url inverts normalize_path for a sub-route",
          scout._abs_url(base, "/item_page/abc/B0001/q/1/%7B%7D")
          == "http://localhost:5511/item_page/abc/B0001/q/1/%7B%7D")
    check("_abs_url does not append a slash to the session root",
          scout._abs_url(base, "/abc") == "http://localhost:5511/abc")

    cfg = scout.CrawlConfig(max_pages=10, max_depth=3, screenshots=False)
    page = FakeShopPage(base)
    c = scout.crawl_env(page, base, "shop", 6, cfg=cfg)
    paths = [p["path"] for p in c["pages"]]
    check("the landing page is the session root, not a 404",
          paths[0] == "/abc" and "error" not in c["pages"][0], str(paths))
    check("the probe reaches the result list",
          c["probe"]["result_path"] == "/search_results/abc/q/1")
    check("the crawl expands PAST the probe to the item pages",
          "/item_page/abc/B0001/q/1/%7B%7D" in paths, str(paths))
    check("no page is visited twice", len(paths) == len(set(paths)))
    check("no absolute URL leaks into the canonical crawl",
          "localhost:5511" not in json.dumps(c))

    other = FakeShopPage("http://localhost:5999/abc")
    c2 = scout.crawl_env(other, other.base, "shop", 6, cfg=cfg)
    check("the crawl hash does not depend on the port",
          c["crawl_sha256"] == c2["crawl_sha256"])

    # The budget bounds the BFS, and the post-probe expansion runs INSIDE it:
    # at max_pages=2 the landing page plus the probe's result page already fill
    # it, so no item page is reached.
    small = scout.CrawlConfig(max_pages=2, max_depth=3, screenshots=False)
    c3 = scout.crawl_env(FakeShopPage(base), base, "shop", 6, cfg=small)
    check("the post-probe expansion still respects the page budget",
          len(c3["pages"]) <= small.max_pages + 1
          and not any("item_page" in p["path"] for p in c3["pages"]),
          str([p["path"] for p in c3["pages"]]))


def test_crawler():
    print("[7] the crawler: BFS, budget, probe, determinism, port independence")
    cfg = scout.CrawlConfig(max_pages=10, max_depth=3, screenshots=False)
    page = FakePage("http://localhost:5511")
    c = scout.crawl_env(page, page.base, "wiki", 1, cfg=cfg)
    paths = [p["path"] for p in c["pages"]]
    check("BFS reaches every reachable page",
          set(paths) >= {"/", "/wiki/Biology", "/wiki/Physics",
                         "/wiki/Biophysics"}, str(paths))
    check("off-site links are not followed",
          not any("example.com" in p for p in paths))
    check("no page is visited twice", len(paths) == len(set(paths)))
    check("frontier order is BFS, root first", paths[0] == "/")

    check("the probe query comes from the site, not a task",
          c["probe"]["query"] == "Biology")
    check("the probe result is recorded",
          c["probe"]["result_path"] == "/search?q=Biology"
          and any(p.get("from_probe") for p in c["pages"]))

    # Port independence: the same site on another port must hash identically.
    other = FakePage("http://localhost:5999")
    c2 = scout.crawl_env(other, other.base, "wiki", 1, cfg=cfg)
    check("the crawl hash does not depend on the port",
          c["crawl_sha256"] == c2["crawl_sha256"])
    check("no absolute URL leaks into the canonical crawl",
          "localhost:5511" not in json.dumps(c))

    r = scout.check_determinism(FakePage("http://localhost:5511"),
                                "http://localhost:5511", "wiki", 1, n=5, cfg=cfg)
    check("5/5 crawls are byte-identical",
          r["identical"] and r["n_distinct"] == 1)

    small = scout.CrawlConfig(max_pages=2, max_depth=3, screenshots=False)
    c3 = scout.crawl_env(FakePage("http://localhost:5511"),
                         "http://localhost:5511", "wiki", 1, cfg=small)
    check("the page budget is respected",
          len([p for p in c3["pages"] if not p.get("from_probe")]) == 2)
    check("a different budget is a different crawl (G3 needs variants)",
          c3["crawl_sha256"] != c["crawl_sha256"])

    flat = scout.CrawlConfig(max_pages=10, max_depth=1, screenshots=False)
    c4 = scout.crawl_env(FakePage("http://localhost:5511"),
                         "http://localhost:5511", "wiki", 1, cfg=flat)
    check("depth 1 does not reach a depth-2 page",
          "/wiki/Biophysics" not in [p["path"] for p in c4["pages"]])

    prompt = scout.render_crawl_for_prompt(c)
    check("the summariser's input names labels and paths",
          "Biology" in prompt and "/wiki/Biology" in prompt)
    check("the summariser's input carries no task text",
          not any(h["rule"] == "3-task-overlap"
                  for h in scout.leak_check(
                      scout.normalize_manual("# x\n## 1. p\n" + prompt[:400],
                                             "wiki"),
                      task_intents=scout.task_intents())["hits"]))


def test_summarizer_contract():
    print("[7b] the summariser call is fixed-schema and side-effect free")
    seen = {}

    def fake(system, user):
        seen["system"], seen["user"] = system, user
        # A deliberately messy answer: wrong order, a missing section, prose
        # around it. normalize_manual has to fix all three.
        return ("Sure!\n# Site manual: wiki\n## 7. Visual style\nTables.\n"
                "## 1. Purpose and top-level navigation\nA nav column.\n",
                {"prompt_tokens": 11, "completion_tokens": 7})

    cfg = scout.CrawlConfig(max_pages=4, screenshots=False)
    c = scout.crawl_env(FakePage("http://localhost:5511"),
                        "http://localhost:5511", "wiki", 1, cfg=cfg)
    text, usage = scout.summarize_crawl(c, summarizer=fake)
    _, body = scout.parse_manual(text)
    check("a messy answer is normalised to the schema",
          sorted(body) == [1, 2, 3, 4, 5, 6, 7]
          and body[1] == "A nav column." and body[7] == "Tables.")
    check("the prose around the answer is dropped", "Sure!" not in text)
    check("usage is recorded for the cost line",
          usage["prompt_tokens"] == 11 and usage["prompt_chars"] > 0)
    check("the prompt states the schema and the leak rules",
          all(("## %d. %s" % (n, t)) in seen["user"] for n, t in scout.SECTIONS)
          and "Do NOT write any year" in seen["user"])
    check("no task text reaches the summariser",
          "Biophysics mentioned" not in seen["user"])


# --------------------------------------------------------------------------
# 8. evalbridge plumbing
# --------------------------------------------------------------------------

def test_evalbridge_site_manual(tmp):
    print("[8] --site-manual reaches TW_SITE_MANUAL and preflight guards it")
    man = os.path.join(tmp, "manual_full.md")
    scout.write_text(man, scout.normalize_manual(SAMPLE_MANUAL, "wiki"))

    spec = evalbridge.EvalSpec(version=1, seed=1, deterministic_judge=True,
                               extra_env={"TW_SITE_MANUAL": man, "FOO": "bar"})
    env = evalbridge.build_env(spec)
    check("TW_SITE_MANUAL is in the eval env", env.get("TW_SITE_MANUAL") == man)
    check("extra_env is applied LAST (it can override)", env.get("FOO") == "bar")
    check("as_dict records it for the run log",
          "TW_SITE_MANUAL=%s" % man in (spec.as_dict()["extra_env"] or ""))
    check("preflight accepts an existing manual",
          not any("TW_SITE_MANUAL" in p for p in evalbridge.preflight(spec)))

    bad = evalbridge.EvalSpec(version=1, seed=1,
                              extra_env={"TW_SITE_MANUAL": man + ".missing"})
    check("preflight refuses a missing manual",
          any("TW_SITE_MANUAL does not exist" in p
              for p in evalbridge.preflight(bad)))

    # The CLI is the surface the driver uses; parse it the way the call-site
    # test does rather than launching anything.
    out = _run_cli(["plan", "--version", "1", "--seed", "1",
                    "--deterministic-judge", "--site-manual", man,
                    "--env", "TW_X=1"])
    check("`eval plan --site-manual` parses and dry-runs", out is not None)
    if out is not None:
        check("the printed command carries TW_SITE_MANUAL",
              "TW_SITE_MANUAL=%s" % man in out)
        check("--env KEY=VAL reaches the command", "TW_X=1" in out)
    check("a malformed --env is refused",
          _run_cli(["plan", "--version", "1", "--env", "NOEQUALS"]) is None)


def _run_cli(argv):
    """Run `adaptercl.evalbridge` with argv; None if it exited non-zero."""
    p = subprocess.Popen(
        [sys.executable, "-m", "adaptercl.evalbridge"] + argv,
        cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=dict(os.environ, PYTHONPATH=PROJECT))
    out, _ = p.communicate()
    return out.decode("utf-8", "replace") if p.returncode == 0 else None


# --------------------------------------------------------------------------
# 9. the pooled-with-manuals YAML
# --------------------------------------------------------------------------

def test_dataset_prefix():
    print("[9] gen_yaml_holdout --dataset-prefix")
    src = os.path.join(PROJECT, "out", "cells", "ver9b6", "yaml", "_gen_v2.yaml")
    if not os.path.exists(src):
        skip("--dataset-prefix", "%s missing" % src)
        return
    tag = "_scouttest9b"
    ydir = os.path.join(PROJECT, "out", "cells", tag, "yaml")
    try:
        p = subprocess.Popen(
            [sys.executable, "scripts/gen_yaml_holdout.py", "--holdout", "v1",
             "--backbone", "9b", "--pooled", "--tag", tag,
             "--dataset-prefix", "adaptercl_scout_ver_%s"],
            cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, _ = p.communicate()
        out = out.decode("utf-8", "replace")
        check("it succeeds", p.returncode == 0, out[-300:] if p.returncode else "")
        yml = os.path.join(ydir, "_gen_pooled.yaml")
        if not os.path.exists(yml):
            check("the yaml was written", False)
            return
        body = open(yml).read()
        ds = [l for l in body.splitlines() if l.startswith("dataset:")]
        check("exactly one dataset line", len(ds) == 1, str(ds))
        check("it pools the five MANUAL corpora",
              ds and ds[0] == "dataset: " + ",".join(
                  "adaptercl_scout_ver_v%d" % i for i in (2, 3, 4, 5, 6)))
        check("the held-out version is absent", "adaptercl_scout_ver_v1" not in body)
        check("the leak check ran and passed", "leak check: OK" in out)
        check("the default is unchanged for every other caller",
              "adaptercl_ver_%s" in open(
                  os.path.join(PROJECT, "scripts", "gen_yaml_holdout.py")).read())
    finally:
        shutil.rmtree(os.path.join(PROJECT, "out", "cells", tag),
                      ignore_errors=True)


# --------------------------------------------------------------------------
# 10. launchers
# --------------------------------------------------------------------------

LAUNCHERS = ["run_fullset_scout.sh", "run_scout.sh"]


def test_shop_featured_is_deterministic_opt_in():
    import os as _os
    import importlib.util as _iu
    app_py = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
        "TimeWarp", "env", "webshop", "web_agent_site", "app.py")
    if not _os.path.exists(app_py):
        return
    src = open(app_py).read()
    check("the shop exposes an opt-in deterministic RNG for featured products",
       "_featured_rng" in src)
    check("no unseeded random.sample over the product list remains",
       "random.sample(all_products" not in src)
    check("every featured/sidebar sample goes through the gated RNG",
       src.count("_featured_rng().sample(all_products") == 3)
    check("the flag defaults OFF so existing eval numbers are untouched",
       'TW_DETERMINISTIC_FEATURED' in src and 'return random' in src)
    scout_src = open(_os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "adaptercl", "scout.py")).read()
    check("the crawl sets the flag before EnvServers starts the children",
       scout_src.index('TW_DETERMINISTIC_FEATURED')
       < scout_src.index('capture.EnvServers(era'))

def main():
    tmp = tempfile.mkdtemp(prefix="adaptercl_scout_")
    try:
        test_schema()
        test_strip_sections()
        test_leak_check()
        test_prompt_parity()
        test_inject_manual_roundtrip(tmp)
        test_framing_strings_agree()
        test_descriptions_format(tmp)
        test_crawler()
        test_session_rooted_site()
        test_summarizer_contract()
        test_evalbridge_site_manual(tmp)
        test_dataset_prefix()
        test_shop_featured_is_deterministic_opt_in()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n%d passed, %d failed, %d skipped" % (PASS[0], FAIL[0], SKIP[0]))
    sys.exit(1 if FAIL[0] else 0)


if __name__ == "__main__":
    main()
