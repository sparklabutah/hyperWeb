"""Scout agent: explore a TimeWarp version, write a site manual (scout_plan.md).

The proposal under test is that a *scout* -- something that visits the site with
no task and writes down how it works -- produces a description that either (A)
the policy can use in its prompt, or (B) a hypernetwork can turn into a better
adapter than in-context reading gives. `scout_plan.md` Part A fixes the
comparisons; this module builds the conditioning object those arms need.

Two variants; **S-crawl is the primary one and is what this module implements**:
a scripted crawler explores, and an LLM only *summarises* the crawl. 4.2's rule
is why -- the conditioning signal sets the weights, so nondeterminism in it is
nondeterminism in the policy, and Gate G1 (10/10 identical crawls) has to hold
for anything new. An LLM driving the browser (S-agent) cannot clear that bar by
construction; it is the opt-in secondary variant and is not run from here.

Facts that shaped the code, verified rather than assumed:

* **Every task starts on a landing page.** All 231 records in the task JSON have
  `start_url` in {`__WIKI__`, `__NEWS__`, `__WEBSHOP__`} (capture.py's header
  verified this). So a crawl that starts at the site root starts where every
  episode starts, and the manual is a *per-version* object -- there are only 18
  distinct initial observations. Per-episode scouting is out of scope
  (scout_plan.md B.4).
* **Ports move between runs.** `EnvServers` allocates from a base upward, so the
  absolute URL of a page is NOT stable across two crawls of the same version.
  Every URL in the canonical crawl JSON is therefore stored as a **path**
  relative to the site root; the absolute URLs live only in a side-car that is
  excluded from the determinism hash. Storing absolute URLs was the obvious
  thing to do and would have failed G1 on every version.
* **Screenshot bytes are not part of the crawl hash** either. G1 is about the
  summariser's input; the PNGs are kept only for the optional vision+text arm
  (B+V), and `capture.check_determinism` already owns the pixel question.
* **The webshop serves per-session roots** (`/abc`), wiki and news do not
  (capture.LAUNCH). Path normalisation strips the session prefix so the shop's
  paths are comparable across runs.
* **`page.accessibility` does not exist in playwright 1.61**; `page_roles`
  already handles that and is reused verbatim rather than re-implemented.

Leak rules (B.3) are enforced by `leak_check`, with one documented deviation:
the forbidden-token list contains "version", and the wiki's own navigation has a
link literally labelled "Printable version". A manual that may not quote the
labels on the page is not an affordance manual. So a forbidden token is waived
**only** when it occurs inside a UI label the crawl actually observed on that
site; every waiver is recorded in `gates.json` so the count can be read. A token
that is not a quoted label is still a hard failure.

Run (playwright lives in `tw_r1_q3` = paths.PY_BENCH):

    python3 -m adaptercl scout plan                       # what would run
    GO=1 python3 -m adaptercl scout crawl --versions 1,6
    GO=1 python3 -m adaptercl scout summarize --versions 1,6 --endpoint http://localhost:8000/v1
    python3 -m adaptercl scout ablate
    python3 -m adaptercl scout gates
"""

from __future__ import print_function

import argparse
import collections
import hashlib
import json
import os
import random
import re
import shutil
import sys

from . import capture, cells, paths

# --------------------------------------------------------------------------
# The manual schema (scout_plan.md B.2) -- fixed, so content can be ablated
# mechanically rather than by asking the model for a shorter description.
# --------------------------------------------------------------------------

#: (number, heading). Sections 1-6 are AFFORDANCE; section 7 is STYLE.
SECTIONS = (
    (1, "Purpose and top-level navigation"),
    (2, "Search"),
    (3, "Page anatomy"),
    (4, "Lists, pagination, sorting, filters"),
    (5, "Forms and multi-step flows"),
    (6, "Gotchas"),
    (7, "Visual style"),
)
AFFORDANCE_SECTIONS = (1, 2, 3, 4, 5, 6)
STYLE_SECTIONS = (7,)
SECTION_TITLE = dict(SECTIONS)

#: Written when the summariser omits a section. Never empty: `strip_sections`
#: has to produce the same section skeleton for every version or the affordance
#: and style variants are not mechanically comparable across versions.
EMPTY_SECTION = "(nothing observed for this section.)"

#: How a manual is framed when it is handed to the POLICY (arm A). This exact
#: string is duplicated -- deliberately, once -- in
#: `scripts/benchmark_adapter.py`, which cannot import this package (it runs
#: under the bench interpreter with the repo root as cwd). `tests/test_scout.py`
#: asserts the two agree, so the eval prompt and the training corpus put the
#: manual in the same slot with the same framing (scout_plan.md Part E #7).
MANUAL_BLOCK_HEADER = "# Site manuals"


def manual_block(text):
    """The exact text appended to `extra_instructions` for arm A.

    `extra_instructions` already ends in a newline, so one leading newline gives
    a single blank line before the header -- the same shape the corpus injection
    produces (`bcdata.inject_manual`).
    """
    return "\n%s\n%s\n" % (MANUAL_BLOCK_HEADER, text.strip())


# --------------------------------------------------------------------------
# Crawl configuration
# --------------------------------------------------------------------------

class CrawlConfig(object):
    """Every knob that could make two crawls of one version differ, pinned.

    Not a dataclass, for the same reason `capture.CaptureConfig` is not.
    """

    __slots__ = ("max_pages", "max_depth", "probe_search", "capture_cfg",
                 "screenshots", "max_links_per_page", "max_labels")

    def __init__(self, **kw):
        # 25 pages per env is the plan's budget. Amortised over 103 test tasks
        # this is well under one agent step per episode, which is why arm A
        # needs no step-budget control (scout_plan.md Part D).
        self.max_pages = 25
        self.max_depth = 3
        # Exactly one fixed probe query per env, typed into the first search box
        # found. The string comes from the SITE (the first nav/category label),
        # never from a task intent, so it is task-free by construction.
        self.probe_search = True
        # Screenshots are for the optional vision+text arm only and are excluded
        # from the determinism hash.
        self.screenshots = True
        # Bounds so one pathological page cannot dominate the summariser's
        # input; both are applied in DOM order, so they are deterministic.
        self.max_links_per_page = 200
        self.max_labels = 60
        self.capture_cfg = capture.CaptureConfig()
        for k, v in kw.items():
            if k not in self.__slots__:
                raise TypeError("unknown CrawlConfig option %r" % (k,))
            setattr(self, k, v)

    def as_dict(self):
        d = collections.OrderedDict()
        for k in self.__slots__:
            if k == "capture_cfg":
                continue
            d[k] = getattr(self, k)
        d["capture_fingerprint"] = self.capture_cfg.fingerprint()
        return d

    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Page extraction
# --------------------------------------------------------------------------

#: One evaluate() per page rather than N locator round-trips: the DOM order of
#: querySelectorAll is stable, a per-locator walk is not obviously so, and this
#: is ~40x faster on the shop's long listing pages.
_EXTRACT_JS = r"""
() => {
  const txt = (el) => (el.innerText || el.textContent || '')
      .replace(/\s+/g, ' ').trim().slice(0, 120);
  const out = {title: document.title || '', links: [], buttons: [],
               inputs: [], headings: [], forms: 0};
  for (const a of document.querySelectorAll('a[href]')) {
    out.links.push({label: txt(a), href: a.getAttribute('href') || ''});
  }
  for (const b of document.querySelectorAll(
        'button, input[type=submit], input[type=button]')) {
    const l = txt(b) || b.getAttribute('value') || b.getAttribute('aria-label') || '';
    out.buttons.push(String(l).replace(/\s+/g, ' ').trim().slice(0, 120));
  }
  for (const i of document.querySelectorAll('input, textarea, select')) {
    const type = (i.getAttribute('type') || i.tagName).toLowerCase();
    if (type === 'submit' || type === 'button' || type === 'hidden') continue;
    let label = i.getAttribute('aria-label') || '';
    if (!label && i.id) {
      const l = document.querySelector('label[for="' + i.id + '"]');
      if (l) label = txt(l);
    }
    if (!label) {
      const p = i.closest('label');
      if (p) label = txt(p);
    }
    out.inputs.push({type: type,
                     name: i.getAttribute('name') || '',
                     label: String(label).replace(/\s+/g, ' ').trim().slice(0, 120),
                     placeholder: i.getAttribute('placeholder') || ''});
  }
  for (const h of document.querySelectorAll('h1, h2, h3')) {
    out.headings.push(h.tagName.toLowerCase() + ': ' + txt(h));
  }
  out.forms = document.querySelectorAll('form').length;
  return out;
}
"""


def normalize_path(url, base):
    """Absolute URL -> a port-independent, HOST-ROOT-absolute path, or None if
    off-site.

    Ports are allocated per run (`capture.EnvServers`), so the absolute URL is
    not stable across crawls and must never enter the canonical JSON; the path
    is.

    The path is kept as the SERVER sees it and is deliberately NOT made relative
    to the site root. An earlier version stripped the root prefix, which is
    lossless only for a site whose root is the host root. The webshop's root is
    a per-session PATH (`capture.LAUNCH["shop"]["suffix"] == "/abc"`) while
    every other route is served from the host root with the session id as an
    INNER segment (`/item_page/abc/<asin>/...`). Stripping therefore made the
    canonical path non-invertible: `_abs_url` rebuilt `/abc/item_page/abc/...`,
    Flask answered 404, and every shop crawl collapsed to its landing page.
    The session id is fixed by `capture.LAUNCH`, not allocated per run, so
    keeping it in the path costs nothing in stability.
    """
    try:
        from urllib.parse import urlsplit, urlunsplit
    except ImportError:                                # pragma: no cover - py2
        from urlparse import urlsplit, urlunsplit
    b, u = urlsplit(base), urlsplit(url)
    if u.scheme and u.scheme not in ("http", "https"):
        return None
    if u.netloc and u.netloc != b.netloc:
        return None
    path = u.path or "/"
    if not path.startswith("/"):
        path = "/" + path
    return urlunsplit(("", "", path, u.query, ""))


def site_root_path(base):
    """The canonical path of a site's landing page (`/` or the shop's `/abc`)."""
    try:
        from urllib.parse import urlsplit
    except ImportError:                                # pragma: no cover - py2
        from urlparse import urlsplit
    return urlsplit(base).path or "/"


def _abs_url(base, path):
    """Canonical (host-root-absolute) path -> the absolute URL to navigate to.

    The exact inverse of `normalize_path`: the path already carries everything
    below the host, so it is joined to the ORIGIN and never to the base's path.
    Appending to the base instead is what produced `/abc/item_page/abc/...` and
    a 404 on every shop link -- silently, because `capture._http_ok` accepts any
    status under 500 and the readiness probe hits `/abc` (200), not `/abc/`.
    """
    try:
        from urllib.parse import urlsplit
    except ImportError:                                # pragma: no cover - py2
        from urlparse import urlsplit
    b = urlsplit(base)
    origin = "%s://%s" % (b.scheme or "http", b.netloc)
    if not path:
        path = site_root_path(base)
    if not path.startswith("/"):
        path = "/" + path
    return origin + path


def _probe_query(page_record):
    """The fixed probe string for an env: the first usable nav/category label.

    Taken from the site, never from a task intent (B.3 rule 3), and chosen by a
    rule rather than by hand so it is reproducible per version.
    """
    for link in page_record["links"]:
        label = link["label"].strip()
        if len(label) >= 3 and re.match(r"^[A-Za-z][A-Za-z '\-]+$", label):
            return label
    return "list"


def extract_page(page, base, cfg):
    """One loaded page -> the record that goes into the crawl JSON."""
    raw = page.evaluate(_EXTRACT_JS)
    roles = capture.page_roles(page, cfg.capture_cfg)
    links, seen = [], set()
    for a in raw["links"][:cfg.max_links_per_page]:
        try:
            target = normalize_path(page.url if a["href"].startswith("#")
                                    else _resolve(page.url, a["href"]), base)
        except Exception:                              # noqa: BLE001
            target = None
        if target is None:
            continue
        key = (a["label"], target)
        if key in seen:
            continue
        seen.add(key)
        links.append({"label": a["label"], "path": target})
    return collections.OrderedDict([
        ("path", None),                 # filled by the caller; kept first
        ("depth", None),
        ("title", raw["title"].strip()),
        ("headings", raw["headings"][:cfg.max_labels]),
        ("links", links),
        ("buttons", [b for b in raw["buttons"][:cfg.max_labels] if b]),
        ("inputs", raw["inputs"][:cfg.max_labels]),
        ("n_forms", raw["forms"]),
        ("role_counts", roles["counts"]),
        ("aria_snapshot", roles["aria_snapshot_text"] or ""),
    ])


def _resolve(current, href):
    try:
        from urllib.parse import urljoin
    except ImportError:                                # pragma: no cover - py2
        from urlparse import urljoin
    return urljoin(current, href)


# --------------------------------------------------------------------------
# The crawl
# --------------------------------------------------------------------------

def crawl_env(page, base, env, era, cfg=None, out_dir=None):
    """BFS from a site root; returns the canonical crawl dict.

    Deterministic by construction: the frontier is a FIFO of paths discovered in
    DOM order, `visited` bounds it, and nothing in the returned dict depends on
    a port, a timestamp or a screenshot byte.
    """
    cfg = cfg or CrawlConfig()
    pages, visited = [], set()
    # Seeded with the SITE ROOT's own canonical path, which is "/" for wiki and
    # news and the session path "/abc" for the shop.
    frontier = collections.deque([(site_root_path(base), 0)])
    shots = []
    probe = {"query": None, "typed_into": None, "result_path": None}

    def bfs():
        """Drain the frontier, appending page records. Deterministic: a FIFO of
        paths discovered in DOM order, bounded by `visited` and the budget."""
        while frontier and len(pages) < cfg.max_pages:
            path, depth = frontier.popleft()
            if path in visited:
                continue
            visited.add(path)
            url = _abs_url(base, path)
            try:
                page.goto(url, wait_until=cfg.capture_cfg.wait_until,
                          timeout=cfg.capture_cfg.nav_timeout_ms)
            except Exception as e:                     # noqa: BLE001
                pages.append(collections.OrderedDict(
                    [("path", path), ("depth", depth), ("error", str(e)[:200])]))
                continue
            page.wait_for_timeout(cfg.capture_cfg.settle_ms)
            rec = extract_page(page, base, cfg)
            rec["path"] = path
            rec["depth"] = depth
            pages.append(rec)
            if cfg.screenshots and out_dir:
                shots.append(_shoot(page, out_dir, len(pages) - 1, path, cfg))
            if depth < cfg.max_depth:
                for link in rec["links"]:
                    if link["path"] not in visited:
                        frontier.append((link["path"], depth + 1))

    bfs()

    # -- one fixed probe query, typed into the first search box found --------
    if cfg.probe_search and pages:
        probe = _run_probe(page, base, pages, cfg)
        # ...and then KEEP CRAWLING from where it landed. On a site whose
        # content is reachable only through a search form -- the webshop, where
        # the landing page is a single search box and every product lives behind
        # a result list -- the pre-probe BFS sees exactly one page, so a manual
        # written from it can say nothing about items, options or checkout: the
        # affordances the held-out-domain question is actually about. This adds
        # no form submissions (B.1 still allows exactly one); it only follows
        # links, with the same budget, depth and FIFO order as above.
        probe_pages = [r for r in pages if r.get("from_probe")]
        if probe_pages:
            last = probe_pages[-1]
            visited.add(last.get("path"))
            if last.get("depth", 0) < cfg.max_depth:
                for link in last.get("links") or []:
                    if link["path"] not in visited:
                        frontier.append((link["path"], last["depth"] + 1))
            bfs()

    crawl = collections.OrderedDict([
        ("env", env),
        ("era", int(era)),
        ("cell", cells.Cell(env, int(era)).key),
        ("config", cfg.as_dict()),
        ("probe", probe),
        ("n_pages", len(pages)),
        ("pages", pages),
    ])
    crawl["crawl_sha256"] = canonical_sha(crawl)
    if shots:
        crawl["screenshots"] = shots          # excluded from the hash (see below)
    return crawl


def _shoot(page, out_dir, index, path, cfg):
    d = os.path.join(out_dir, "pages")
    if not os.path.isdir(d):
        os.makedirs(d)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", path).strip("-")[:40] or "root"
    png = os.path.join(d, "%03d_%s.png" % (index, slug))
    try:
        page.screenshot(path=png, full_page=cfg.capture_cfg.full_page,
                        animations="disabled", caret="hide",
                        scale=cfg.capture_cfg.screenshot_scale)
    except Exception as e:                             # noqa: BLE001
        return {"path": path, "error": str(e)[:120]}
    return {"path": path, "file": os.path.relpath(png, out_dir)}


def _run_probe(page, base, pages, cfg):
    """Type the fixed probe query into the first search box and record where it
    lands. No other form is ever submitted (B.1)."""
    query = _probe_query(pages[0])
    for rec in pages:
        if "inputs" not in rec:
            continue
        for i, field in enumerate(rec["inputs"]):
            hay = " ".join([field.get("label", ""), field.get("placeholder", ""),
                            field.get("name", ""), field.get("type", "")]).lower()
            if field.get("type") not in ("text", "search") and "search" not in hay:
                continue
            try:
                page.goto(_abs_url(base, rec["path"]),
                          wait_until=cfg.capture_cfg.wait_until,
                          timeout=cfg.capture_cfg.nav_timeout_ms)
                box = page.locator("input[type=text], input[type=search]").nth(0)
                box.fill(query)
                box.press("Enter")
                page.wait_for_load_state(cfg.capture_cfg.wait_until,
                                         timeout=cfg.capture_cfg.nav_timeout_ms)
                page.wait_for_timeout(cfg.capture_cfg.settle_ms)
                res = extract_page(page, base, cfg)
                res["path"] = normalize_path(page.url, base)
                res["depth"] = rec["depth"] + 1
                res["from_probe"] = True
                pages.append(res)
                return {"query": query, "typed_into": rec["path"],
                        "field_index": i, "result_path": res["path"]}
            except Exception as e:                     # noqa: BLE001
                return {"query": query, "typed_into": rec["path"],
                        "field_index": i, "result_path": None,
                        "error": str(e)[:200]}
    return {"query": query, "typed_into": None, "result_path": None}


#: Keys excluded from the determinism hash. Screenshot bytes are a pixel
#: question (capture.check_determinism owns it) and their filenames embed an
#: index that a re-ordered crawl would change without the crawl differing.
_HASH_EXCLUDE = ("crawl_sha256", "screenshots")


def canonical_sha(crawl):
    """sha256 of the crawl with the non-canonical keys removed."""
    d = dict((k, v) for k, v in crawl.items() if k not in _HASH_EXCLUDE)
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Summariser (one chat call per (env, version), T=0, seed=0)
# --------------------------------------------------------------------------

SUMMARY_SYSTEM = (
    "You are documenting a website for someone who has to operate it without "
    "ever having seen it. You are given a machine-readable crawl of the site: "
    "page paths, titles, headings, link labels, button labels, form fields and "
    "accessibility-role counts. Write a site manual."
)

SUMMARY_RULES = """\
Rules, all mandatory:
* Emit EXACTLY these seven sections, in this order, with these headings and
  nothing before or after them:
%(schema)s
* Sections 1-6 describe what the interface DOES and where its controls are:
  labels, paths, page structure, flows, and traps. Section 7, and only section
  7, describes what it LOOKS like.
* Do NOT write any year, date or decade in sections 1-6.
* Do NOT mention any other variant of this site, and do NOT use comparative or
  temporal words: "version", "era", "modern", "older", "newer", "legacy",
  "updated", "classic". You may quote a control's label verbatim even if the
  label itself contains such a word.
* Do NOT mention tasks, goals, users, agents, difficulty, performance or
  success. Describe the interface only.
* Write only what the crawl shows. Do not invent controls or pages.
* Be concrete: name the labels and the paths. 80-150 words per section.
"""

SUMMARY_USER = """\
Site: %(env)s

%(rules)s

Crawl:
%(crawl)s

Write the manual now, starting with the line `# Site manual: %(env)s`.
"""


def schema_text():
    return "\n".join("  ## %d. %s" % (n, t) for n, t in SECTIONS)


def render_crawl_for_prompt(crawl, max_pages=25, max_chars=60000):
    """The crawl as compact text. This is the summariser's ONLY input."""
    out = ["site: %s" % crawl["env"],
           "pages crawled: %d" % crawl.get("n_pages", len(crawl["pages"]))]
    p = crawl.get("probe") or {}
    if p.get("query"):
        out.append("search probe: typed %r into the search box on %s -> %s"
                   % (p["query"], p.get("typed_into"), p.get("result_path")))
    for rec in crawl["pages"][:max_pages]:
        if "error" in rec:
            out.append("\n--- %s : FAILED TO LOAD" % rec.get("path"))
            continue
        out.append("\n--- page %s%s" % (rec["path"],
                                        "  (search results)" if rec.get("from_probe") else ""))
        out.append("title: %s" % rec["title"])
        if rec["headings"]:
            out.append("headings: " + " | ".join(rec["headings"][:20]))
        labels = [l["label"] for l in rec["links"] if l["label"]]
        if labels:
            out.append("links: " + " | ".join(labels[:40]))
        paths = []
        for l in rec["links"][:40]:
            if l["path"] not in paths:
                paths.append(l["path"])
        if paths:
            out.append("link targets: " + " ".join(paths[:25]))
        if rec["buttons"]:
            out.append("buttons: " + " | ".join(rec["buttons"][:20]))
        if rec["inputs"]:
            out.append("fields: " + " | ".join(
                "%s[%s]%s" % (f.get("label") or f.get("name") or "?",
                              f.get("type"),
                              (" ph=%s" % f["placeholder"]) if f.get("placeholder") else "")
                for f in rec["inputs"][:20]))
        out.append("forms: %d" % rec.get("n_forms", 0))
        rc = rec.get("role_counts") or {}
        if rc:
            out.append("roles: " + " ".join("%s=%d" % (k, v)
                                            for k, v in sorted(rc.items())[:25]))
    text = "\n".join(out)
    return text[:max_chars]


def chat_once(endpoint, model, system, user, max_tokens=6000, timeout=600,
              no_think=True):
    """One OpenAI-compatible chat completion at temperature 0, seed 0.

    urllib rather than the `openai` package: this has to run under the bench
    interpreter and under the system python (tests), and the payload is three
    fields. `seed` is passed because vLLM honours it and a sampler that ignores
    it at T=0 is a no-op either way.

    THINKING IS TURNED OFF, and that is not a preference. The summariser is the
    policy family (Qwen3.5), whose chat template is a REASONING template: asked
    for a manual it emits a long "Thinking Process:" deliberation first and, on
    a crawl-sized prompt, spends the whole token budget there and never reaches
    `## 1.`. The result parses to a manual with seven empty sections -- a
    treatment that is silently no treatment at all. `enable_thinking: false` is
    a vLLM chat-template kwarg; a server that rejects it gets one retry without
    it, so a plain OpenAI-compatible endpoint still works (and `max_tokens` is
    large enough for a preamble plus the manual if one thinks anyway).
    """
    try:
        from urllib.request import Request, urlopen
    except ImportError:                                # pragma: no cover - py2
        raise RuntimeError("python 3 required")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": max_tokens,
    }
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    def _post(pl):
        req = Request(endpoint.rstrip("/") + "/chat/completions",
                      data=json.dumps(pl).encode("utf-8"),
                      headers={"Content-Type": "application/json",
                               "Authorization": "Bearer EMPTY"})
        return urlopen(req, timeout=timeout).read().decode("utf-8")

    try:
        body = _post(payload)
    except Exception:                                  # noqa: BLE001
        if not no_think:
            raise
        payload.pop("chat_template_kwargs", None)
        body = _post(payload)
    d = json.loads(body)
    msg = d["choices"][0]["message"]
    text = msg.get("content") or ""
    usage = d.get("usage") or {}
    return text, usage


def summarize_crawl(crawl, endpoint=None, model=None, summarizer=None):
    """Crawl -> (normalised manual text, usage dict).

    `summarizer` is a `f(system, user) -> (text, usage)` override; the tests use
    it so nothing in this module needs a server to be exercised.
    """
    user = SUMMARY_USER % {
        "env": crawl["env"],
        "rules": SUMMARY_RULES % {"schema": schema_text()},
        "crawl": render_crawl_for_prompt(crawl),
    }
    if summarizer is not None:
        text, usage = summarizer(SUMMARY_SYSTEM, user)
    else:
        if not endpoint or not model:
            raise ValueError("summarize_crawl needs --endpoint and --model "
                             "(or a summarizer callable)")
        text, usage = chat_once(endpoint, model, SUMMARY_SYSTEM, user)
    usage = dict(usage or {})
    usage["prompt_chars"] = len(user)
    manual = normalize_manual(text, crawl["env"])
    # An all-empty manual is the one failure that looks like a result. Every
    # section reads "(nothing observed for this section.)", the file is written,
    # the eval arm runs, and it measures the baseline under a different name.
    # Fail here instead, with the raw answer, so it cannot reach a unit.
    if manual.count(EMPTY_SECTION) == len(SECTIONS):
        raise RuntimeError(
            "the summariser produced no parseable section for %r: every one of "
            "the %d sections is empty. This is usually a reasoning template "
            "spending the whole budget before `## 1.` -- check `no_think` and "
            "`max_tokens`. First 400 chars of the answer:\n%s"
            % (crawl["env"], len(SECTIONS), text[:400]))
    return manual, usage


# --------------------------------------------------------------------------
# Schema normalisation and mechanical ablation (B.2)
# --------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^##\s*(\d)\.\s*(.*)$", re.M)


def parse_manual(text):
    """-> (title line, {section number: body}). Unknown sections are dropped."""
    lines = text.strip().splitlines()
    title = lines[0].strip() if lines and lines[0].strip().startswith("#") else ""
    body = {}
    marks = list(_SECTION_RE.finditer(text))
    for i, m in enumerate(marks):
        n = int(m.group(1))
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body[n] = text[m.end():end].strip()
    return title, body


def normalize_manual(text, env):
    """Rebuild a manual to the exact schema: right title, all 7 sections, order.

    A summariser that skips "5. Forms" on the wiki would otherwise make the
    affordance variant of the wiki manual structurally different from the
    shop's, and `strip_sections` would be comparing different objects.
    """
    _, body = parse_manual(text)
    out = ["# Site manual: %s" % env]
    for n, heading in SECTIONS:
        out.append("## %d. %s" % (n, heading))
        out.append(body.get(n, "").strip() or EMPTY_SECTION)
    return "\n".join(out).strip() + "\n"


def strip_sections(text, keep):
    """Keep exactly the numbered sections in `keep`, in schema order."""
    keep = tuple(keep)
    title, body = parse_manual(text)
    out = [title or "# Site manual"]
    for n, heading in SECTIONS:
        if n not in keep:
            continue
        out.append("## %d. %s" % (n, heading))
        out.append(body.get(n, "").strip() or EMPTY_SECTION)
    return "\n".join(out).strip() + "\n"


def affordance_only(text):
    return strip_sections(text, AFFORDANCE_SECTIONS)


def style_only(text):
    return strip_sections(text, STYLE_SECTIONS)


# --------------------------------------------------------------------------
# Leak check (B.3)
# --------------------------------------------------------------------------

#: Comparative / identifying tokens. `cells.THEME_NAME` supplies the 18 theme
#: strings, which are the most direct version key there is.
FORBIDDEN_WORDS = ("era", "version", "modern", "older", "newer", "legacy",
                   "newest", "latest", "outdated", "contemporary", "retro",
                   "old-fashioned", "old-style", "dated")
VERSION_KEYS = tuple("v%d" % e for e in cells.ERAS)
#: Rule 4: no statement about the agent or how well anything performs.
PERFORMANCE_WORDS = ("success rate", "difficulty", "benchmark", "agent",
                     "reward", "episode", "task success", "score of")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")


def theme_tokens():
    out = set()
    for env in cells.THEME_NAME:
        for era in cells.THEME_NAME[env]:
            out.add(cells.THEME_NAME[env][era].lower())
    return tuple(sorted(out))


def observed_labels(crawl):
    """Every UI label the crawl saw, lowercased. Used only to waive a forbidden
    token that the manual is QUOTING (see the module docstring)."""
    out = set()
    for rec in crawl.get("pages", []):
        for l in rec.get("links", []) or []:
            if l.get("label"):
                out.add(l["label"].lower())
        for b in rec.get("buttons", []) or []:
            out.add(str(b).lower())
        for f in rec.get("inputs", []) or []:
            for k in ("label", "placeholder", "name"):
                if f.get(k):
                    out.add(str(f[k]).lower())
        # HEADINGS ARE LABELS TOO. The news front page carries `h2: Latest
        # News`, and "latest" is on FORBIDDEN_WORDS as a comparative -- so a
        # manual that says *headlines sit under an h2 heading labeled "Latest
        # News"* failed G2 for naming a section of the page it is documenting.
        # That is the same case the waiver already exists for (a manual that may
        # not name the controls on the page is not an affordance manual); a
        # heading was simply missing from the labels it is checked against.
        # The record stores them prefixed with the tag (`h2: Latest News`), so
        # the prefix is stripped -- both forms are kept, since a manual may
        # legitimately quote either.
        for h in rec.get("headings", []) or []:
            h = str(h)
            out.add(h.lower())
            out.add(re.sub(r"^h[1-6]:\s*", "", h).lower())
        if rec.get("title"):
            out.add(rec["title"].lower())
    return set(x for x in out if x.strip())


def _waived(text_low, start, end, labels):
    """True if the token at [start, end) sits inside an observed UI label."""
    for lab in labels:
        i = 0
        while True:
            i = text_low.find(lab, i)
            if i < 0:
                break
            if i <= start and end <= i + len(lab):
                return lab
            i += 1
    return None


def leak_check(text, sections="full", labels=(), task_intents=(), ngram=6,
               version_key=None, scouted_intents=(), answer_strings=(),
               warn_answer_strings=()):
    """B.3 as code. Returns {"hits": [...], "waived": [...]}; empty hits == pass.

    `sections` is "full", "affordance" or "style" and selects rule 2 (years are
    permitted in section 7 only).

    `scouted_intents` and `answer_strings` are the task-scout additions
    (scout_task_plan.md section 4 / section 11 WHAT TO BUILD (b)/(c)) and
    default to `()`, so every existing caller -- the task-free `gate_leak`,
    `selftest_leak_fixture`, `cmd_selftest` -- gets byte-identical results
    (HARD RULE 5):
    * `scouted_intents` (rule 3b) are forbidden in sections 1-7 but allowed
      inside `## 8. Worked examples` -- or anywhere in a document with no
      numbered sections at all, i.e. `paste.md`, which the task-scout plan
      names as the intents' other legitimate home.
    * `answer_strings` (rule 6) are forbidden EVERYWHERE and, unlike rule 1/5,
      carry NO observed-UI-label waiver: unlike a comparative word or an era
      name, a reference-answer string is the exact thing this rule exists to
      catch, and it is the MODAL case (not an edge case) for it to also be a
      page title, heading or link label the scout's own crawl observed --
      this task set answers with wiki article names and section headings.
      Waiving on label-match here would tautologically clear a real leak
      every time it fires. Mirrors rule 3 (task-intent overlap), which is
      likewise never waived.
    """
    hits, waived = [], []
    low = text.lower()
    labels = set(l for l in labels if l)

    def add(rule, what, where=""):
        hits.append({"rule": rule, "match": what, "where": where})

    # -- rule 1: no other version, theme name, or comparative term -----------
    for key in VERSION_KEYS:
        for m in re.finditer(r"\b%s\b" % key, low):
            add("1-version-key", key, _ctx(text, m.start()))
    for theme in theme_tokens():
        idx = low.find(theme)
        if idx >= 0:
            add("1-theme-name", theme, _ctx(text, idx))
    for w in FORBIDDEN_WORDS:
        for m in re.finditer(r"\b%s\b" % re.escape(w), low):
            lab = _waived(low, m.start(), m.end(), labels)
            if lab:
                waived.append({"rule": "1-comparative", "match": w, "label": lab})
            else:
                add("1-comparative", w, _ctx(text, m.start()))

    # -- rule 2: no years outside section 7 ---------------------------------
    if sections in ("full", "affordance"):
        _, body = parse_manual(text)
        for n in AFFORDANCE_SECTIONS:
            for m in YEAR_RE.finditer(body.get(n, "")):
                add("2-year-in-affordance", m.group(0), "section %d" % n)
    elif sections == "style":
        pass                                    # years are allowed in style

    # -- rule 3: no task content --------------------------------------------
    # An n-gram that ALSO occurs in a SCOUTED intent is waived, because section 8 and paste.md are
    # allowed to quote the scouted intent (rule 3b) and quoting it necessarily produces that n-gram.
    # The TimeWarp tasks are template-generated, so this is common rather than exotic: scouted task
    # 147 asks "Is there any news article regarding San Diego ..." and TEST task 36 asks "Is there
    # any news article regarding Salt Lake City ..." -- six identical leading words, and the city,
    # which is the whole content of the question, differs. Waiving is safe because the check does not
    # stop at the first match: any n-gram of the test intent that a scouted intent does NOT explain
    # is still a hard hit, and for two genuinely different questions those exist in quantity.
    if task_intents:
        grams = _ngrams(low, ngram)
        scout_grams = set()
        for si in (scouted_intents or ()):
            scout_grams |= _ngrams(str(si).lower(), ngram)
        for intent in task_intents:
            explained = None
            for g in _ngrams(intent.lower(), ngram):
                if g not in grams:
                    continue
                if g in scout_grams:
                    explained = g
                    continue
                add("3-task-overlap", g, intent[:60])
                break
            else:
                if explained:
                    waived.append({"rule": "3-task-overlap", "match": explained,
                                   "label": "shared question template with a scouted intent"})

    # -- rule 4: nothing about the agent or performance ---------------------
    # Same observed-label waiver as rules 1 and 5, for the same reason and on the same evidence:
    # the webshop's own account page has sections headed "Purchased", "Target" and "Reward", and a
    # manual that documents where those controls are has to be allowed to name them. An occurrence
    # OUTSIDE an observed label -- the manual talking about the agent's own reward -- still fails.
    for w in PERFORMANCE_WORDS:
        for m in re.finditer(r"\b%s\b" % re.escape(w), low):
            lab = _waived(low, m.start(), m.end(), labels)
            if lab:
                waived.append({"rule": "4-performance", "match": w, "label": lab})
            else:
                add("4-performance", w, _ctx(text, m.start()))

    # -- rule 5: era 6 is style-neutral, never "newest" ---------------------
    if version_key == "v%d" % cells.NEUTRAL_ERA:
        for w in ("newest", "latest", "most recent", "state of the art"):
            for m in re.finditer(r"\b%s\b" % re.escape(w), low):
                # Same waiver as rule 1, for the same reason and with the same
                # evidence requirement: the news front page's own h2 is
                # "Latest News", and a manual that documents where the headlines
                # are has to be allowed to name the heading they sit under.
                # An occurrence OUTSIDE an observed label -- the manual calling
                # the site itself the latest one -- is still a hard failure.
                lab = _waived(low, m.start(), m.end(), labels)
                if lab:
                    waived.append({"rule": "5-neutral-era", "match": w,
                                   "label": lab})
                else:
                    add("5-neutral-era", w, _ctx(text, m.start()))

    # -- rule 3b: a SCOUTED task's intent is allowed only in section 8 -------
    # (or anywhere in a document that has no numbered sections at all, i.e.
    # paste.md: `parse_manual` then finds no section body to check against,
    # so every occurrence there is, correctly, never flagged.)
    if scouted_intents:
        _, body = parse_manual(text)
        outside = set()
        for n, sec_text in body.items():
            if n == WORKED_EXAMPLES_SECTION[0]:
                continue
            outside |= _ngrams(sec_text.lower(), ngram)
        for intent in scouted_intents:
            for g in _ngrams(intent.lower(), ngram):
                if g in outside:
                    add("3b-scouted-outside-section8", g, intent[:60])
                    break

    # -- rule 6: no TEST or SCOUTED-task reference-answer string, anywhere ---
    # NEVER waived (see the docstring): an answer string is a hard hit even
    # when it also happens to be an observed UI label.
    # RULE 6 IS SPLIT, and the split is structural rather than a tuned heuristic (2026-09-19).
    #
    # `answer_strings` (HARD, rule 6): answers of the tasks the scout ACTUALLY ATTEMPTED. At level
    # T2 the scout was shown their recipes, so it can restate an answer it was told -- the real
    # leak this gate exists to catch, and the one a reviewer demonstrated with task 108's recipe.
    # No waiver: an observed UI label that happens to equal a scouted answer is still a restatement.
    #
    # `warn_answer_strings` (REPORTED, rule 6w): answers of the TEST tasks. The scout can never have
    # seen these -- the picker raises on any id <= 103, `run_bu_unit.py --save-history` refuses a
    # test id before a browser starts, and the scouted set is train-only -- so a test answer in the
    # manual is a COINCIDENCE of vocabulary, not a leak. Treating it as fatal failed 29 of 29 cells
    # of the first wave on the words "news", "white", "black", "zero", "atom", "information" and
    # "2024", each of which is some test task's answer AND ordinary interface vocabulary. The count
    # is reported in gates.json so a real pattern would still be visible.
    def _scan(strings, rule, sink, waivable=False):
        if not strings:
            return
        norm_text = re.sub(r"\s+", " ", low)
        for ans in strings:
            a = re.sub(r"\s+", " ", str(ans).strip().lower())
            if len(a) < 4:
                continue
            # RULE 6 IS NEVER WAIVED, even for a single token that is also an observed UI label.
            # That waiver was tried on 2026-09-19 and reverted the same hour: task 108's answer is
            # the single word "Physics", which is also a page title, and its recipe's last step
            # ("Conclude that only the Physics article contained ...") is a genuine restatement the
            # waiver would hide. A cell that trips this rule is DROPPED, which is the cheap error;
            # a contaminated arm is the expensive one.
            single = False
            idx = 0
            while True:
                idx = norm_text.find(a, idx)
                if idx < 0:
                    break
                lab = _waived(norm_text, idx, idx + len(a), labels) if single else None
                if lab:
                    waived.append({"rule": rule, "match": a, "label": lab})
                else:
                    sink(rule, a, _ctx(norm_text, idx))
                idx += 1

    warnings = []

    def warn(rule, what, where=""):
        warnings.append({"rule": rule, "match": what, "where": where})

    _scan(answer_strings, "6-answer-leak", add, waivable=True)
    _scan(warn_answer_strings, "6w-test-answer-coincidence", warn)
    return {"hits": hits, "waived": waived, "warnings": warnings}


def _ctx(text, idx, span=40):
    return text[max(0, idx - span):idx + span].replace("\n", " ")


def _ngrams(text, n):
    words = _WORD.findall(text)
    return set(" ".join(words[i:i + n]) for i in range(0, max(0, len(words) - n + 1)))


def task_intents(task_data=None):
    """Every task intent in the deterministic-judge task file (231 records)."""
    path = task_data or paths.TASK_DATA_DETERMINISTIC
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        data = json.load(fh)
    rows = data if isinstance(data, list) else data.get("tasks", [])
    out = []
    for r in rows:
        for k in ("intent", "goal", "instruction", "task"):
            if isinstance(r, dict) and isinstance(r.get(k), str):
                out.append(r[k])
                break
    return out


# --------------------------------------------------------------------------
# Output layout (B.5)
# --------------------------------------------------------------------------

OUT_SCOUT = os.path.join(paths.OUT, "scout")
OUT_CRAWL = os.path.join(OUT_SCOUT, "crawl")

VARIANTS = collections.OrderedDict([
    ("full", None),                       # all seven sections
    ("affordance", AFFORDANCE_SECTIONS),
    ("style", STYLE_SECTIONS),
])


def version_dir(era, root=None):
    return os.path.join(root or OUT_CRAWL, "v%d" % int(era))


def env_dir(era, env, root=None):
    return os.path.join(version_dir(era, root), env)


def write_json(path, obj):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
    return path


def read_text(path):
    with open(path) as fh:
        return fh.read()


def write_text(path, text):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(path, "w") as fh:
        fh.write(text)
    return path


def build_version_manuals(era, root=None, envs=None):
    """Concatenate the three env manuals into the version's three variants.

    The concatenation is deterministic (fixed env order) and is the conditioning
    unit: one manual per version, not per cell (B.4).
    """
    root = root or OUT_CRAWL
    envs = tuple(envs or cells.ENVIRONMENTS)
    parts = {}
    for name, keep in VARIANTS.items():
        chunks = []
        for env in envs:
            p = os.path.join(env_dir(era, env, root), "manual.md")
            if not os.path.exists(p):
                raise IOError("missing %s -- run `scout summarize` first" % p)
            text = read_text(p)
            chunks.append(text if keep is None else strip_sections(text, keep))
        joined = "\n\n".join(c.strip() for c in chunks) + "\n"
        write_text(os.path.join(version_dir(era, root), "manual_%s.md" % name),
                   joined)
        parts[name] = joined
    return parts


def write_descriptions(root=None, eras=None):
    """The three `descriptions_*.json` in `t2l.load_descriptions` format."""
    root = root or OUT_CRAWL
    eras = tuple(eras or cells.ERAS)
    out = {}
    for name in VARIANTS:
        d = collections.OrderedDict()
        for era in eras:
            p = os.path.join(version_dir(era, root), "manual_%s.md" % name)
            if os.path.exists(p):
                d["v%d" % era] = read_text(p)
        write_json(os.path.join(root, "descriptions_%s.json" % name), d)
        out[name] = d
    return out


# --------------------------------------------------------------------------
# Gates (Part C)
# --------------------------------------------------------------------------

def gate_leak(root=None, eras=None, envs=None, task_data=None):
    """G2 over every env manual x every variant, plus the planted-leak fixture."""
    root = root or OUT_CRAWL
    eras = tuple(eras or cells.ERAS)
    envs = tuple(envs or cells.ENVIRONMENTS)
    intents = task_intents(task_data)
    rows, n_hits, n_waived = [], 0, 0
    for era in eras:
        for env in envs:
            mp = os.path.join(env_dir(era, env, root), "manual.md")
            cp = os.path.join(env_dir(era, env, root), "crawl.json")
            if not os.path.exists(mp):
                continue
            labels = observed_labels(json.load(open(cp))) if os.path.exists(cp) else set()
            text = read_text(mp)
            for name, keep in VARIANTS.items():
                variant = text if keep is None else strip_sections(text, keep)
                r = leak_check(variant, sections=name, labels=labels,
                               task_intents=intents, version_key="v%d" % era)
                n_hits += len(r["hits"])
                n_waived += len(r["waived"])
                rows.append({"cell": cells.Cell(env, era).key, "variant": name,
                             "hits": r["hits"], "n_waived": len(r["waived"])})
    planted = selftest_leak_fixture()
    return {"gate": "G2", "n_manuals": len(rows), "n_hits": n_hits,
            "n_waived": n_waived, "planted_caught": planted["ok"],
            "planted": planted, "rows": rows,
            "pass": bool(n_hits == 0 and planted["ok"] and rows)}


def selftest_leak_fixture():
    """G2's own control: plant a `v3`, a `2005` and a task-intent 6-gram and
    confirm the checker catches all three. A leak checker nobody tested is a
    leak checker that returns 0 hits for the wrong reason."""
    intents = ["List all articles from the Related Pages sections of the "
               "Biophysics and Biochemistry pages"]
    good = normalize_manual(
        "# Site manual: wiki\n## 1. Purpose and top-level navigation\nA "
        "left-hand column of links leads to the main page and to recent "
        "changes.\n", "wiki")
    planted_version = good.replace("A left-hand", "This is v3. A left-hand")
    planted_year = good.replace("A left-hand", "Built in 2005. A left-hand")
    planted_task = good.replace(
        "A left-hand",
        "List all articles from the Related Pages sections of the "
        "Biophysics and Biochemistry pages. A left-hand")
    r_clean = leak_check(good, task_intents=intents)
    r_ver = leak_check(planted_version, task_intents=intents)
    r_year = leak_check(planted_year, task_intents=intents)
    r_task = leak_check(planted_task, task_intents=intents)
    ok = (not r_clean["hits"]
          and any(h["rule"] == "1-version-key" for h in r_ver["hits"])
          and any(h["rule"] == "2-year-in-affordance" for h in r_year["hits"])
          and any(h["rule"] == "3-task-overlap" for h in r_task["hits"]))
    return {"ok": bool(ok),
            "clean_hits": len(r_clean["hits"]),
            "version_hits": [h["rule"] for h in r_ver["hits"]],
            "year_hits": [h["rule"] for h in r_year["hits"]],
            "task_hits": [h["rule"] for h in r_task["hits"]]}


#: Above this, the manual starts competing with the AXTree for the policy's
#: prompt budget. AgentLab shrinks the OBSERVATION to fit
#: (generic_agent._get_maxes; the 2026-08-16 context-budget fix), so an
#: over-long manual does not error -- arm A simply sees less of the page than
#: arm C does, and the comparison stops being one-variable. 8000 characters is
#: ~2k tokens against `max_input_tokens = 40960 - 2048`.
MANUAL_WARN_CHARS = 8000


def gate_manual_size(root=None, eras=None):
    """Report each version manual's size, and warn before it eats the AXTree."""
    root = root or OUT_CRAWL
    rows, warn = [], []
    for era in tuple(eras or cells.ERAS):
        for name in VARIANTS:
            p = os.path.join(version_dir(era, root), "manual_%s.md" % name)
            if not os.path.exists(p):
                continue
            n = len(read_text(p))
            rows.append({"version": "v%d" % era, "variant": name, "chars": n,
                         "approx_tokens": n // 4})
            if name == "full" and n > MANUAL_WARN_CHARS:
                warn.append("v%d full manual is %d chars (~%d tokens)"
                            % (era, n, n // 4))
    return {"gate": "manual_size", "reported_not_gated": True, "rows": rows,
            "warn_over_chars": MANUAL_WARN_CHARS, "warnings": warn}


def gate_determinism(root=None, eras=None, envs=None):
    """G1 read back off disk: `determinism.json` per (env, version)."""
    root = root or OUT_CRAWL
    eras = tuple(eras or cells.ERAS)
    envs = tuple(envs or cells.ENVIRONMENTS)
    rows, ok = [], True
    for era in eras:
        for env in envs:
            p = os.path.join(env_dir(era, env, root), "determinism.json")
            if not os.path.exists(p):
                rows.append({"cell": cells.Cell(env, era).key, "status": "missing"})
                ok = False
                continue
            d = json.load(open(p))
            rows.append({"cell": cells.Cell(env, era).key, "n": d.get("n"),
                         "identical": d.get("identical"),
                         "n_distinct": d.get("n_distinct")})
            ok = ok and bool(d.get("identical"))
    return {"gate": "G1", "rows": rows, "pass": bool(ok and rows)}


def check_determinism(page, base, env, era, n=10, cfg=None, out_dir=None):
    """G1: crawl the same site n times; the canonical JSON must be identical."""
    shas, first = [], None
    for i in range(int(n)):
        c = crawl_env(page, base, env, era, cfg=cfg,
                      out_dir=out_dir if i == 0 else None)
        shas.append(c["crawl_sha256"])
        if i == 0:
            first = c
    distinct = sorted(set(shas))
    out = {"cell": cells.Cell(env, int(era)).key, "n": int(n),
           "sha256": shas, "n_distinct": len(distinct),
           "identical": len(distinct) == 1,
           "n_pages": first["n_pages"] if first else 0}
    if out_dir:
        write_json(os.path.join(out_dir, "determinism.json"), out)
    return out


# --------------------------------------------------------------------------
# G3 probe samples + G4 separability
# --------------------------------------------------------------------------

def probe_variants(era, env, root=None, budgets=(15, 20, 25, 30, 40)):
    """The manual variants written for one cell by `crawl --budget-sweep`."""
    root = root or OUT_CRAWL
    d = env_dir(era, env, root)
    out = []
    for b in budgets:
        p = os.path.join(d, "variants", "manual_b%d.md" % b)
        if os.path.exists(p):
            out.append((("b%d" % b), read_text(p)))
    return out


def build_probe_samples(root=None, eras=None, envs=None, out_path=None,
                        device="cpu", encoder_id=None):
    """G3's npz: one row per (crawl variant, env, version) x manual variant.

    Rows are per **cell**, not per version, because `probe.identity_probes`
    decomposes into era / env / joint and its labels have to parse as cell keys
    (`probe._labels_for`). That decomposition is the informative part: era
    accuracy answers "is the manual a version key", env accuracy answers "does
    it carry site identity".

    `groups` is the crawl-variant id, so `probe`'s grouped split never trains
    and tests on two manuals derived from the same crawl.
    """
    import numpy as np
    from . import t2l

    root = root or OUT_CRAWL
    eras = tuple(eras or cells.ERAS)
    envs = tuple(envs or cells.ENVIRONMENTS)
    out_path = out_path or os.path.join(root, "probe_samples.npz")

    texts, labels, groups, variants = [], [], [], []
    for era in eras:
        for env in envs:
            base = os.path.join(env_dir(era, env, root), "manual.md")
            items = []
            if os.path.exists(base):
                items.append(("b25", read_text(base)))
            items += [(g, t) for g, t in probe_variants(era, env, root)
                      if g != "b25"]
            for group, text in items:
                for name, keep in VARIANTS.items():
                    texts.append(text if keep is None
                                 else strip_sections(text, keep))
                    labels.append(cells.Cell(env, era).key)
                    groups.append(group)
                    variants.append(name)
    if not texts:
        raise IOError("no manuals under %s -- run `scout summarize` first" % root)

    tc = t2l.TextConditioner(device=device,
                             model_id=encoder_id or t2l.DEFAULT_TEXT_ENCODER)
    emb = tc.encode(texts).numpy()
    np.savez(out_path, embeddings=emb,
             labels=np.array(labels), groups=np.array(groups),
             variants=np.array(variants))
    return {"path": out_path, "n_rows": len(texts),
            "dim": int(emb.shape[1]),
            "n_groups": len(set(groups)),
            "variants": sorted(set(variants))}


def run_identity_probe(npz_path=None, root=None, variant="full", seed=0):
    """G3 on one manual variant, reported (not gated) per Part A."""
    import numpy as np
    from . import probe as probe_mod

    npz_path = npz_path or os.path.join(root or OUT_CRAWL, "probe_samples.npz")
    d = np.load(npz_path, allow_pickle=True)
    keep = np.array([str(v) for v in d["variants"]]) == variant
    if not keep.any():
        raise SystemExit("no rows for variant %r in %s" % (variant, npz_path))
    rep = probe_mod.identity_probes(
        d["embeddings"][keep],
        [str(x) for x in d["labels"][keep]],
        groups=[str(x) for x in d["groups"][keep]],
        seed=seed)
    # `identity_probes` nests the three probes under "probes"; every caller here
    # wants the accuracies, so flatten to {target: accuracy} and keep the rest.
    rep["accuracy"] = dict((t, rep["probes"][t]["test_accuracy"])
                           for t in rep["probes"])
    return rep


# --------------------------------------------------------------------------
# S-task: the task-grounded scout (scout_task_plan.md, package B)
#
# Where S-crawl (above) wanders a site with no task, S-task drops the frozen
# policy onto a single held-out-site TRAIN task (id > 103) through the SAME
# harness the eval uses, and turns what it did into a manual. This section
# implements the CONSUMER side of that pipeline: `task-pick` draws the tasks,
# `task-collect` turns a scout episode's visited pages back into a crawl
# (contract C1, produced by `buharness.run_episode(save_history_dir=...)`,
# built in parallel -- this module never edits buharness.py), `task-summarize`
# and `task-paste` build the two things that can be injected into an eval
# prompt, and `task-gates` is this level's leak checker (scout_task_plan.md
# section 4 / section 11 "WHAT TO BUILD").
#
# Every path below lives under `OUT_TASK` (C3): none of it touches
# `OUT_CRAWL`, so the task-free scout above is completely unaffected.
# --------------------------------------------------------------------------

TASK_LEVELS = ("T1", "T2")

#: Bumped whenever `pick_tasks`'s draw algorithm changes, so a `tasks.json`
#: written by an old picker is visibly stale rather than silently reused.
PICKER_VERSION = 1

#: The train/test id boundary (scout_task_plan.md fact table row 3 / C1 / C3).
#: Single source of truth so "id > 103" is never typed twice with a typo.
TASK_TRAIN_ID_MIN = 103

#: §11 "WHAT TO BUILD" (a): manual.md and paste.md each hard-fail above this.
TASK_MANUAL_SIZE_CAP = 6000

WORKED_EXAMPLES_SECTION = (8, "Worked examples")

# `ADAPTERCL_TASK_ROOT` (added 2026-09-19 for the ICL-LOSO baselines) lets a
# campaign keep its own scout cells side by side with another's -- the ICL
# campaign scouts TWICE, once per eval backbone (Qwen3.5-4B and
# Llama-3.1-8B-Instruct each write their own manual), and both would
# otherwise collide on `out/scout/task/v6/<site>/<level>_k<k>_d<d>`.
# Unset => byte-identical to the pre-campaign default.
OUT_TASK = os.environ.get("ADAPTERCL_TASK_ROOT") or os.path.join(
    OUT_SCOUT, "task")


def task_version_dir(era, root=None):
    return os.path.join(root or OUT_TASK, "v%d" % int(era))


def task_site_dir(era, site, root=None):
    return os.path.join(task_version_dir(era, root), site)


def task_draw_dir(era, site, level, k, draw, root=None):
    """C3: `out/scout/task/v<era>/<site>/<level>_k<k>_d<d>/`."""
    return os.path.join(task_site_dir(era, site, root),
                        "%s_k%d_d%d" % (level, int(k), int(draw)))


def read_jsonl(path):
    """Every line of a `.jsonl` file as a dict. `[]` if the file is absent --
    callers that need it to exist check that themselves and hard-fail (HARD
    RULE 7: a missing input must not look like a clean, empty result)."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------
# task-pick
# --------------------------------------------------------------------------

def test_intent_overlap_ids(task_ids, task_data=None, ngram=6):
    """Train task ids whose own intent shares an `ngram`-gram with ANY test
    task's intent -- i.e. exactly what `leak_check` rule 3 will flag if that
    task is scouted, because section 8 of the manual carries the scouted
    intent verbatim.

    Added 2026-09-19 for the ICL-LOSO baselines. Rule 3 is un-waivable and
    applies to the WHOLE document, section 8 included, so a train task that
    shares a test task's phrasing rule-reds its cell no matter what the
    scout does. Measured on this task set: 14/39 wiki, 12/25 news and 7/37
    shop train tasks overlap, so the gate would have killed roughly a third
    of every draw pool at random.

    Excluding them at PICK time rather than waiving them at GATE time is
    also the better experiment: a scouted train task that is a near-template
    duplicate of a test task inflates the target-domain arm specifically,
    which is the arm under test. Uses the same `_ngrams` helper rule 3 does,
    so picker and gate agree by construction.
    """
    tests = set()
    for intent in test_intents(task_data):
        tests |= _ngrams(str(intent).lower(), ngram)
    if not tests:
        return set()
    out = set()
    for tid, intent in _intents_by_id(task_ids, task_data).items():
        if _ngrams(str(intent).lower(), ngram) & tests:
            out.add(int(tid))
    return out


def _intents_by_id(task_ids, task_data=None):
    ids = set(int(t) for t in task_ids) if task_ids else None
    path = task_data or paths.TASK_DATA_DETERMINISTIC
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        data = json.load(fh)
    rows = data if isinstance(data, list) else data.get("tasks", [])
    out = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        tid = r.get("task_id")
        if not isinstance(tid, int) or (ids is not None and tid not in ids):
            continue
        for k in ("intent", "goal", "instruction", "task"):
            if isinstance(r.get(k), str):
                out[tid] = r[k]
                break
    return out


def _pick_seed(site, draw):
    """A deterministic int seed from (site, draw, picker version). Draw d is
    tied to the training seed s (C3), so the scout-task variance a training
    seed sees is reproducible from that seed alone."""
    h = hashlib.sha256(
        ("task-pick|%s|%d|v%d" % (site, int(draw), PICKER_VERSION))
        .encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _site_train_candidates(site, rows):
    """`rows` narrowed to single-site TRAIN candidates for `site`. Shared by
    `pick_tasks` and `cmd_task_pick` (the latter needs the pre-exclusion pool
    size for `tasks.json`'s `pool_size_before`), so the two can never define
    "the pool" differently."""
    return [r for r in rows
           if r.get("site") == site and r.get("split") == "train"]


def pick_tasks(site, k, draw, rows, exclude_ids=()):
    """Deterministic seeded draw of k single-site TRAIN tasks for `site`.

    Pure and file-free, so it is directly unit-testable: `rows` is a list of
    dicts, each with at least `task_id` (int), `site` (a `cells.ENVIRONMENTS`
    value, or `"multi"`/`None` for a task this site should never claim) and
    `split` (`"train"` / `"test"` / other). `intent` is carried through into
    the result if present. The real CLI builds `rows` from
    `cells.load_tasks()` + `cells.task_site_group_map()` + `cells.load_split()`
    (the task file spells the shop `"webshop"`; `task_site_group_map` is what
    turns that into `"shop"` -- this function never sees the raw spelling).

    Nesting (scout_task_plan.md section 3): for a fixed (site, draw),
    `pick_tasks(site, 1, draw, rows)[0] == pick_tasks(site, 4, draw, rows)[0]`,
    because both draw from the same seeded shuffle of the same candidate id
    list and only the truncation length differs.

    `exclude_ids` (new, optional, defaults to `()` so every existing caller
    is byte-identical -- HARD RULE 5) removes those task ids from the
    candidate pool BEFORE the seeded shuffle, so a task in it can never be
    drawn at any k or draw for this site. Applied identically regardless of
    `level` -- this function has no `level` argument at all, which is what
    makes T1 and T2 scout the SAME (site, draw) pool a structural property
    rather than something a caller has to get right (campaign rule: DOMSCOUT
    scout draw). The id-convention check below runs on the FULL split-tagged
    pool, before exclusion, so a broken id is still caught even if the
    offending row happens to also be in `exclude_ids`.

    Hard errors, never a silently short or wrong draw:
    * `site` is not a known single-site group;
    * a candidate tagged `split == "train"` for `site` has `task_id <=
      TASK_TRAIN_ID_MIN` -- the id convention the picker relies on (row 3 of
      the fact table) would be broken, and drawing anyway would silently mix
      a test task into a "train-only" scout;
    * fewer than `k` candidates remain AFTER exclusion.
    """
    if site not in cells.ENVIRONMENTS:
        raise ValueError("task-pick: unknown site %r (want one of %r)"
                         % (site, cells.ENVIRONMENTS))
    k = int(k)
    if k < 1:
        raise ValueError("task-pick: k must be >= 1, got %r" % (k,))
    candidates = _site_train_candidates(site, rows)
    for r in candidates:
        tid = r.get("task_id")
        if not isinstance(tid, int) or tid <= TASK_TRAIN_ID_MIN:
            raise ValueError(
                "task-pick: candidate task %r for site %r is tagged "
                "split='train' but task_id <= %d -- the train/test id "
                "convention is broken, refusing to draw"
                % (tid, site, TASK_TRAIN_ID_MIN))
    excluded = set(int(t) for t in (exclude_ids or ()))
    if excluded:
        candidates = [r for r in candidates if r["task_id"] not in excluded]
    if len(candidates) < k:
        raise ValueError(
            "task-pick: only %d single-site train task(s) for site %r after "
            "exclusion (excluded %d of the original pool), need k=%d"
            % (len(candidates), site, len(excluded), k))
    ordered = sorted(candidates, key=lambda r: r["task_id"])
    ids = [r["task_id"] for r in ordered]
    rng = random.Random(_pick_seed(site, draw))
    rng.shuffle(ids)
    chosen_ids = ids[:k]
    by_id = dict((r["task_id"], r) for r in ordered)
    return [collections.OrderedDict([
        ("task_id", tid),
        ("intent", by_id[tid].get("intent")),
    ]) for tid in chosen_ids]


def _train_task_rows(task_data=None):
    """`rows` for `pick_tasks`, built from disk: every task tagged with its
    `cells.task_site_group_map()` site (single-site only; multi-site tasks
    get `site=None` and can never be picked) and its `cells.load_split()`
    split."""
    smap = cells.task_site_group_map(task_data=task_data)
    splits = cells.load_split()
    rows = []
    for t in cells.load_tasks(task_data):
        tid = t.get("task_id")
        site = smap.get(tid)
        if site == "multi":
            site = None
        rows.append(collections.OrderedDict([
            ("task_id", tid),
            ("site", site),
            ("split", splits.get(tid)),
            ("intent", t.get("intent")),
        ]))
    return rows


def build_task_pick_result(site, era, level, k, draw, task_data=None,
                           exclude_leaky=False, exclude_intent_overlap=False):
    """The pure core of `task-pick`: `tasks.json`'s content, computed but
    never written. Split out from `cmd_task_pick` so a test can exercise the
    WHOLE pick (rows -> pool -> exclusion -> seeded draw) against the real
    task file without touching disk or the live `out/` tree -- the only
    file I/O here is the read-only task-data load already inside
    `_train_task_rows` / `_load_recipes` / `task_reference_answers`.

    `exclude_leaky` (DOMSCOUT_KL_RUN_PLAN.md / section 11 DO item 1) is
    computed on the WHOLE site pool, identically for every `level` -- this
    function's own exclusion step never looks at `level` at all, which is
    what makes "T1 and T2 scout the SAME (site, draw) pool" a structural
    property rather than something a caller has to get right.
    """
    exclude_leaky = bool(exclude_leaky)
    exclude_intent_overlap = bool(exclude_intent_overlap)
    rows = _train_task_rows(task_data)
    pool = _site_train_candidates(site, rows)
    pool_before = len(pool)
    excluded_ids = set()
    if exclude_leaky:
        # A picked task can never fail the answer-leak gate's rule (c) for
        # its OWN answer, so drop from the pool every candidate whose
        # answer-stripped recipe still contains one of its own
        # reference-answer strings under `leak_check` rule 6's
        # normalisation. Computed BEFORE `pick_tasks` narrows the pool.
        cand_ids = [r["task_id"] for r in pool
                   if isinstance(r.get("task_id"), int)]
        recipes = _load_recipes(cand_ids, task_data)
        answers = task_reference_answers(task_ids=cand_ids,
                                         task_data=task_data)
        excluded_ids = leaky_own_answer_ids(cand_ids, recipes, answers)
    overlap_ids = set()
    if exclude_intent_overlap:
        # Drop every candidate whose own intent shares a 6-gram with a TEST
        # task's intent. `leak_check` rule 3 is un-waivable and covers the
        # whole document, section 8 (which carries the scouted intent
        # verbatim) included, so such a task rule-reds its cell whatever the
        # scout does -- and is the near-template-duplicate case that would
        # inflate the target-domain arm specifically. See
        # `test_intent_overlap_ids`.
        overlap_ids = test_intent_overlap_ids(
            [r["task_id"] for r in pool if isinstance(r.get("task_id"), int)],
            task_data)
        excluded_ids = set(excluded_ids) | overlap_ids
    picked = pick_tasks(site, k, draw, rows, exclude_ids=excluded_ids)
    pool_after = pool_before - len(excluded_ids)
    return collections.OrderedDict([
        ("site", site), ("era", int(era)), ("level", level),
        ("k", int(k)), ("draw", int(draw)), ("seed", int(draw)),
        ("picker_version", PICKER_VERSION),
        ("exclude_leaky", exclude_leaky),
        ("exclude_intent_overlap", exclude_intent_overlap),
        ("intent_overlap_ids", sorted(overlap_ids)),
        ("pool_size_before", pool_before),
        ("pool_size_after", pool_after),
        ("excluded_ids", sorted(excluded_ids)),
        ("tasks", picked),
    ])


def cmd_task_pick(args):
    exclude_leaky = bool(getattr(args, "exclude_leaky", False))
    excl_overlap = bool(getattr(args, "exclude_intent_overlap", False))
    out = build_task_pick_result(args.site, args.era, args.level, args.k,
                                 args.draw, task_data=args.task_data,
                                 exclude_leaky=exclude_leaky,
                                 exclude_intent_overlap=excl_overlap)
    picked = out["tasks"]
    d = task_draw_dir(args.era, args.site, args.level, args.k, args.draw)
    path = write_json(os.path.join(d, "tasks.json"), out)
    print("  wrote %s (%d task(s): %s)%s"
          % (path, len(picked), ", ".join(str(t["task_id"]) for t in picked),
             ("  [pool %d -> %d, excluded %s]"
              % (out["pool_size_before"], out["pool_size_after"],
                 out["excluded_ids"]))
             if (exclude_leaky or excl_overlap) else ""))
    return 0


# --------------------------------------------------------------------------
# task-subset (DOMSCOUT_KL_RUN_PLAN.md "How many examples the scout sees" /
# section 11 DO item 5b): the k in {1, 3, 5} ablation is built from ONE set
# of k=5 scout episodes per (site, draw, level) -- this command carves a
# smaller-k CELL out of that k=5 PARENT cell, so k=1/3/5 differ only in how
# many of the same five attempts their manual distils, never in which tasks
# were attempted or how.
# --------------------------------------------------------------------------

def build_task_subset_result(parent, k):
    """Pure core of `task-subset`: `parent` is an already-loaded parent
    `tasks.json` dict (k=`from-k`, e.g. k=5); returns the new k=`k` cell's
    `tasks.json` content -- the FIRST `k` of `parent["tasks"]`, unchanged
    otherwise, plus a `subset_of` field naming the parent cell and its
    picker seed. No file I/O (the parent is already loaded); the disk side
    -- reading the parent, copying `episodes.jsonl`/`steps/`/`history/` --
    is `cmd_task_subset`'s job, so this function stays directly unit-testable
    against a synthetic `parent` dict.

    Hard errors: `k < 1`; `k` bigger than the parent's own `k` (its `from-k`);
    the parent has fewer than `k` picked tasks (a corrupt/truncated parent).
    Nesting itself (does this prefix match what `task-pick --k k` would
    really draw) is NOT checked here -- that needs `rows`/`exclude_ids`
    (disk-backed task data), so it lives in `cmd_task_subset`'s
    `_verify_subset_nesting` call instead.
    """
    k = int(k)
    if k < 1:
        raise ValueError("task-subset: k must be >= 1, got %r" % (k,))
    from_k = int(parent.get("k") or 0)
    if k > from_k:
        raise ValueError(
            "task-subset: k=%d > the parent cell's own k=%d -- a subset can "
            "only be smaller than (or equal to) its parent" % (k, from_k))
    parent_tasks = parent.get("tasks") or []
    if len(parent_tasks) < k:
        raise ValueError(
            "task-subset: parent cell has only %d picked task(s) on file, "
            "need k=%d -- the parent's tasks.json looks truncated/corrupt"
            % (len(parent_tasks), k))
    subset = [collections.OrderedDict(t) for t in parent_tasks[:k]]
    out = collections.OrderedDict(parent)
    out["k"] = k
    out["tasks"] = subset
    out["subset_of"] = collections.OrderedDict([
        ("cell", "%s_k%d_d%d" % (parent.get("level"), from_k,
                                 int(parent.get("draw") or 0))),
        ("from_k", from_k),
        ("picker_seed", _pick_seed(parent.get("site"), parent.get("draw"))),
    ])
    return out


def _verify_subset_nesting(site, k, draw, rows, exclude_ids, subset_ids):
    """Hard error if `subset_ids` (the first `k` of some k=`from-k` pick)
    are not EXACTLY what `pick_tasks(site, k, draw, rows, exclude_ids=...)`
    would draw on its own -- i.e. nesting is broken, the parent cell was
    picked under a different pool/seed than a direct `task-pick --k k` for
    this (site, draw) would use today. Pure (same contract as `pick_tasks`):
    testable against synthetic `rows` with no disk I/O."""
    expected = [t["task_id"] for t in
               pick_tasks(site, k, draw, rows, exclude_ids=exclude_ids)]
    if expected != list(subset_ids):
        raise ValueError(
            "task-subset: nesting broken for site=%r draw=%d k=%d -- "
            "task-pick --k %d would draw %r, but the parent's first %d "
            "id(s) are %r" % (site, draw, k, k, expected, k, list(subset_ids)))


def cmd_task_subset(args):
    from_k = int(args.from_k)
    k = int(args.k)
    if k > from_k:
        raise SystemExit(
            "task-subset: --k %d > --from-k %d" % (k, from_k))
    parent_dir = task_draw_dir(args.era, args.site, args.level, from_k,
                               args.draw)
    parent_tasks_path = os.path.join(parent_dir, "tasks.json")
    if not os.path.exists(parent_tasks_path):
        raise SystemExit(
            "task-subset: parent cell missing -- %s not found (run "
            "task-pick --k %d --draw %d --site %s --level %s first)"
            % (parent_tasks_path, from_k, args.draw, args.site, args.level))
    parent = json.load(open(parent_tasks_path))
    parent_episodes_path = os.path.join(parent_dir, "episodes.jsonl")
    episodes = read_jsonl(parent_episodes_path)
    result = build_task_subset_result(parent, k)
    subset_ids = [t["task_id"] for t in result["tasks"]]
    have_ids = set(e.get("task_id") for e in episodes)
    missing = [i for i in subset_ids if i not in have_ids]
    if missing:
        raise SystemExit(
            "task-subset: parent's episodes.jsonl (%s) is missing episode(s) "
            "for id(s) %r of the k=%d subset -- run the k=%d scout episodes "
            "to completion first" % (parent_episodes_path, missing, k, from_k))

    rows = _train_task_rows(args.task_data)
    exclude_ids = set(int(i) for i in (parent.get("excluded_ids") or ()))
    _verify_subset_nesting(args.site, k, args.draw, rows, exclude_ids,
                           subset_ids)

    d = task_draw_dir(args.era, args.site, args.level, k, args.draw)
    write_json(os.path.join(d, "tasks.json"), result)

    # episodes.jsonl: filtered to the subset ids, IN THE PARENT FILE'S OWN
    # ROW ORDER (not re-sorted by subset_ids) -- it is a straight subset of
    # what actually ran, not a re-derivation.
    subset_id_set = set(subset_ids)
    filtered = [e for e in episodes if e.get("task_id") in subset_id_set]
    with open(os.path.join(d, "episodes.jsonl"), "w") as fh:
        for e in filtered:
            fh.write(json.dumps(e) + "\n")

    # steps/<id>.jsonl and history/<id>.json: COPIED, never symlinked (a
    # subset cell must stay readable if the parent cell is later moved).
    n_copied = collections.OrderedDict([("steps", 0), ("history", 0)])
    for sub, ext in (("steps", ".jsonl"), ("history", ".json")):
        src_dir = os.path.join(parent_dir, sub)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(d, sub)
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        for tid in subset_ids:
            src = os.path.join(src_dir, "%s%s" % (tid, ext))
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(dst_dir, "%s%s" % (tid, ext)))
                n_copied[sub] += 1

    print("  wrote %s (k=%d subset of %s, %d/%d episode(s), steps=%d "
         "history=%d copied)"
         % (os.path.join(d, "tasks.json"), k,
            os.path.basename(parent_dir), len(filtered), k,
            n_copied["steps"], n_copied["history"]))
    return 0


# --------------------------------------------------------------------------
# task-collect
# --------------------------------------------------------------------------

def _rebase_visited(url, base):
    """A URL the scout VISITED, re-pointed at the server `task-collect` is driving.

    The scout episode ran against its own `capture.EnvServers` instance (say port 9501); this
    collect run starts a fresh instance on its own ports (say 9201). The PATH is portable -- the
    TimeWarp apps use fixed session ids, so `/item_page/abc/...` is the same page on either
    instance -- but the ORIGIN is not, and `normalize_path` correctly refuses a foreign origin.
    Without this, EVERY visited URL was rejected and every cell of the first wave wrote a crawl of
    ZERO pages, which the summariser then dutifully described as "zero pages were crawled"
    (2026-09-19; the manuals were vacuous and the gates fired on them).

    Only a LOCAL origin is rebased. A URL the scout reached off-site keeps its own origin and is
    then dropped by `normalize_path`, exactly as before -- rebasing it would silently turn an
    external page into a local path.
    """
    try:
        from urllib.parse import urlsplit, urlunsplit
    except ImportError:                                    # pragma: no cover - py2 shim
        from urlparse import urlsplit, urlunsplit          # type: ignore
    try:
        u, b = urlsplit(url), urlsplit(base)
    except Exception:                                      # noqa: BLE001
        return url
    if u.scheme not in ("http", "https") or not u.netloc or u.netloc == b.netloc:
        return url
    if (u.hostname or "").lower() not in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        return url
    return urlunsplit((b.scheme, b.netloc, u.path, u.query, ""))


def collect_task_crawl(page, base, env, era, tasks, episodes, cfg=None):
    """The C1-CONSUMING core of `task-collect`.

    Re-opens the ordered, de-duplicated URLs the scout actually visited
    (`episodes.jsonl`'s `visited` lists, C1) with the SAME `extract_page` /
    `normalize_path` the task-free crawl uses, and returns a dict in exactly
    `crawl_env`'s shape (`env, era, cell, config, probe, n_pages, pages,
    crawl_sha256`) plus `source="task"`, `tasks=[...ids]`, and a per-page
    `from_task` -- so `summarize_crawl` / `render_crawl_for_prompt` need no
    change to consume it (scout_task_plan.md section 3 / section 11 C3).

    `page` is a playwright `Page` in production; the test drives this with
    the same fake page/site fixture `tests/test_scout.py` already uses, so no
    browser or server is needed to exercise it.

    `episodes` is the parsed `episodes.jsonl` rows (C1: dicts with at least
    `task_id` and `visited`). Order is TASK order (as `tasks` lists the ids),
    then within a task the episode's own visit order; a URL that recurs --
    across tasks or within one -- keeps only its FIRST visit and FIRST
    `from_task`, matching `crawl_env`'s own `visited` de-duplication. Bounded
    by `cfg.max_pages`, same as the task-free crawl.
    """
    cfg = cfg or CrawlConfig()
    by_task = collections.OrderedDict()
    for row in episodes:
        by_task.setdefault(row["task_id"], []).extend(row.get("visited") or [])

    order, seen_paths = [], set()
    for tid in tasks:
        for url in by_task.get(tid, []):
            path = normalize_path(_rebase_visited(url, base), base)
            if path is None or path in seen_paths:
                continue
            seen_paths.add(path)
            order.append((path, tid))

    pages = []
    for path, tid in order:
        if len(pages) >= cfg.max_pages:
            break
        url = _abs_url(base, path)
        try:
            page.goto(url, wait_until=cfg.capture_cfg.wait_until,
                      timeout=cfg.capture_cfg.nav_timeout_ms)
        except Exception as e:                             # noqa: BLE001
            pages.append(collections.OrderedDict(
                [("path", path), ("depth", 0), ("error", str(e)[:200]),
                 ("from_task", tid)]))
            continue
        page.wait_for_timeout(cfg.capture_cfg.settle_ms)
        rec = extract_page(page, base, cfg)
        rec["path"] = path
        rec["depth"] = 0
        rec["from_task"] = tid
        pages.append(rec)

    crawl = collections.OrderedDict([
        ("env", env),
        ("era", int(era)),
        ("cell", cells.Cell(env, int(era)).key),
        ("config", cfg.as_dict()),
        ("probe", {"query": None, "typed_into": None, "result_path": None}),
        ("source", "task"),
        ("tasks", list(tasks)),
        ("n_pages", len(pages)),
        ("pages", pages),
    ])
    crawl["crawl_sha256"] = canonical_sha(crawl)
    return crawl


def build_task_scout_cost(tasks, episodes, steps_dir=None, crawl=None):
    """`scout_cost.json` for one task-draw dir.

    `reward` -- the verifier outcome -- is recorded HERE ONLY (C1 / section
    4): it must never reach `task-summarize`'s prompt except at level `T1v`,
    which this module does not build a prompt path for.
    """
    by_task = dict((r.get("task_id"), r) for r in episodes)
    rows = []
    for tid in tasks:
        ep = by_task.get(tid) or {}
        n_steps = ep.get("n_steps")
        if n_steps is None and steps_dir:
            sp = os.path.join(steps_dir, "%s.jsonl" % tid)
            n_steps = len(read_jsonl(sp)) if os.path.exists(sp) else None
        rows.append(collections.OrderedDict([
            ("task_id", tid),
            ("n_steps", n_steps),
            ("terminated", ep.get("terminated")),
            ("reward", ep.get("reward")),
        ]))
    return collections.OrderedDict([
        ("tasks", list(tasks)),
        ("n_episodes", len(episodes)),
        ("n_pages", (crawl or {}).get("n_pages")),
        ("rows", rows),
    ])


def cmd_task_collect(args):
    d = task_draw_dir(args.era, args.site, args.level, args.k, args.draw)
    tasks_path = os.path.join(d, "tasks.json")
    if not os.path.exists(tasks_path):
        raise SystemExit(
            "task-collect: %s missing -- run task-pick first" % tasks_path)
    picked = json.load(open(tasks_path))
    task_ids = [t["task_id"] for t in (picked.get("tasks") or [])]
    if not task_ids:
        raise RuntimeError(
            "task-collect: %s has no scouted tasks -- refusing to write an "
            "empty crawl.json (a missing draw must not look like a clean "
            "null)" % tasks_path)
    episodes_path = os.path.join(d, "episodes.jsonl")
    episodes = read_jsonl(episodes_path)
    if not episodes:
        raise RuntimeError(
            "task-collect: %s is missing or empty -- run the scout episodes "
            "(C1) first" % episodes_path)
    if not _need_go("task-collect %s (drives chromium against a fresh "
                    "EnvServers instance of v%s %s)"
                    % (d, args.era, args.site)):
        return 0
    capture._require_playwright()
    from playwright.sync_api import sync_playwright

    cfg = CrawlConfig(max_pages=args.budget, screenshots=False)
    with capture.EnvServers(int(args.era), port_base=args.port_base,
                            envs=(args.site,)) as srv:
        with sync_playwright() as pw:
            browser, ctx = capture._new_context(pw, capture.CaptureConfig())
            try:
                page = ctx.new_page()
                crawl = collect_task_crawl(page, srv.url(args.site), args.site,
                                           args.era, task_ids, episodes,
                                           cfg=cfg)
            finally:
                ctx.close()
                browser.close()
    write_json(os.path.join(d, "crawl.json"), crawl)
    cost = build_task_scout_cost(task_ids, episodes,
                                 steps_dir=os.path.join(d, "steps"),
                                 crawl=crawl)
    write_json(os.path.join(d, "scout_cost.json"), cost)
    print("  wrote %s (%d page(s) from %d task(s))"
          % (os.path.join(d, "crawl.json"), crawl["n_pages"], len(task_ids)))
    return 0


# --------------------------------------------------------------------------
# strip_answer_from_recipe -- the one thing standing between T2's recipe and
# the reference answer it ends with (section 3 / section 11 WHAT TO BUILD).
# --------------------------------------------------------------------------

_TASK_STEP_RE = re.compile(r"(?m)^(\s*\d+\.\s)")

#: The trailing sentence of a train task's `additional_instructions` that
#: hands over the answer. Deliberately narrow -- it may only match a clause
#: that (a) contains no sentence-ending period or newline of its own, so it
#: cannot reach back past an earlier sentence in the same step, (b) mentions
#: delivering something "to the user(s)", and (c) ends in a quoted string
#: right at the end of the step. Measured against all 128 real train-task
#: recipes (2026-09-18): matches ~110/128 verbatim "send/report/reply/... to
#: the user: "<answer>"." endings; the rest (multi-branch or answer-less
#: closing sentences) fall through to the conservative whole-step drop below.
_ANSWER_SENTENCE_RE = re.compile(
    r'[^.\n"]*?\bto\s+(?:the\s+)?users?\b[^."\n]*"[^"]*"\.?\s*\Z', re.I)


def strip_answer_from_recipe(text, answer_strings=()):
    """A train task's `additional_instructions`, with the final answer gone.

    Tries to isolate just the trailing "send/report/... to the user:
    "<answer>"." sentence of the LAST numbered step and drop only that,
    keeping the rest of the step. When that shape is not found -- multiple
    branches, a conditional answer, an answer with no quoted string -- it is
    conservative and drops the WHOLE last numbered step instead, per
    scout_task_plan.md section 11: "when unsure, drop the whole last numbered
    step." Never raises on unstructured text; a string with no numbered steps
    is returned trimmed but otherwise untouched (there is nothing here this
    function can identify as "the last step").

    `answer_strings` (new, optional, defaults to `()` so every existing
    caller that omits it is byte-identical -- HARD RULE 5) is this task's own
    reference-answer strings (`task_reference_answers()`'s per-task value).
    When given, the narrow sentence-strip above is not trusted on its own: an
    EARLIER sentence in the same last step routinely restates the answer
    without the "to the user" framing (e.g. "Conclude that only the Physics
    article contained the specific section." right before "Send a message to
    the user: "Physics"." -- stripping only the second sentence leaves
    "Physics" sitting in the first one). If the narrow strip's surviving text
    still contains any answer string, the WHOLE last step is dropped instead,
    and the walk continues backward, dropping each earlier step in turn for
    as long as IT ALSO contains an answer string, stopping at the first step
    that does not (measured on the real 128-task train set: this closes 44/44
    of the leaks the narrow strip alone left in place).
    """
    if not text:
        return text
    marks = list(_TASK_STEP_RE.finditer(text))
    if not marks:
        return text.rstrip()
    n = len(marks)
    last = marks[n - 1]
    head = text[:last.start()].rstrip()
    step_marker = last.group(1)
    body = text[last.end():].rstrip()
    m = _ANSWER_SENTENCE_RE.search(body)
    kept = body[:m.start()].rstrip() if m else None

    norm_answers = []
    for a in (answer_strings or ()):
        a = re.sub(r"\s+", " ", str(a).strip().lower())
        if len(a) >= 4:
            norm_answers.append(a)

    if not norm_answers:
        # Legacy behaviour (no answer strings supplied): touch only the last
        # step, exactly as before HARD RULE 5 required this addition.
        if kept:
            return head + "\n" + step_marker + kept
        return head

    def leaks(s):
        low = re.sub(r"\s+", " ", s.lower())
        return any(a in low for a in norm_answers)

    if kept is not None and not leaks(kept):
        if kept:
            return head + "\n" + step_marker + kept
        return head

    # Either the narrow shape did not match, or its surviving text still
    # leaks: the whole last step goes. Then walk backward, dropping any
    # earlier step whose body still contains an answer string, stopping at
    # the first one that does not (or when steps run out).
    boundary = marks[n - 1].start()
    idx = n - 2
    while idx >= 0:
        step_body = text[marks[idx].end():marks[idx + 1].start()]
        if not leaks(step_body):
            break
        boundary = marks[idx].start()
        idx -= 1
    return text[:boundary].rstrip()


# --------------------------------------------------------------------------
# task-summarize
# --------------------------------------------------------------------------

TASK_SUMMARY_RULES_EXTRA = (
    "* You are ALSO given the task(s) the scout was assigned when it produced "
    "this crawl: each one's INTENT, and at level T2 an answer-stripped recipe "
    "of STEPS. Use them only to understand which flows and controls matter --"
    " do not restate any task, its outcome, or any specific value it "
    "produced. Sections 1-6 still describe the interface only, the same as "
    "if no task had ever been attempted.\n"
)


def _indent(text, prefix="    "):
    return "\n".join(prefix + ln for ln in text.splitlines())


def _task_context_block(tasks_meta, level, recipes=None):
    """The scouted-task context appended to the summariser's prompt: intents
    always, answer-stripped recipe steps only at level T2. The reward and the
    answer must never appear here -- `recipes`, if passed, is expected to
    already be the `strip_answer_from_recipe` output."""
    lines = []
    for t in tasks_meta:
        lines.append("- task %s intent: %s"
                     % (t["task_id"], t.get("intent") or ""))
        if level == "T2" and recipes and recipes.get(t["task_id"]):
            lines.append("  recipe steps:")
            lines.append(_indent(recipes[t["task_id"]]))
    return "\n".join(lines)


def summarize_task_crawl(crawl, tasks_meta, level, endpoint=None, model=None,
                         summarizer=None, recipes=None):
    """`crawl.json` (+ scouted intents, + T2 answer-stripped recipes) ->
    (sections 1-7, normalised exactly like `summarize_crawl`, usage dict).

    Section 8 is NOT written here -- it is deterministic (`build_worked_
    examples`), never an LLM's to phrase, so it cannot accidentally restate
    an answer the model was told to omit. `summarizer` is a
    `f(system, user) -> (text, usage)` override, exactly like
    `summarize_crawl`'s, so a test needs no server.
    """
    if level not in TASK_LEVELS:
        raise ValueError("summarize_task_crawl: level must be one of %r, "
                         "got %r" % (TASK_LEVELS, level))
    context = _task_context_block(tasks_meta, level, recipes=recipes)
    crawl_text = render_crawl_for_prompt(crawl)
    if context:
        crawl_text = crawl_text + "\n\nScouted task(s):\n" + context
    user = SUMMARY_USER % {
        "env": crawl["env"],
        "rules": (SUMMARY_RULES % {"schema": schema_text()})
                + TASK_SUMMARY_RULES_EXTRA,
        "crawl": crawl_text,
    }
    if summarizer is not None:
        text, usage = summarizer(SUMMARY_SYSTEM, user)
    else:
        if not endpoint or not model:
            raise ValueError("summarize_task_crawl needs --endpoint and "
                             "--model (or a summarizer callable)")
        text, usage = chat_once(endpoint, model, SUMMARY_SYSTEM, user)
    usage = dict(usage or {})
    usage["prompt_chars"] = len(user)
    manual = normalize_manual(text, crawl["env"])
    if manual.count(EMPTY_SECTION) == len(SECTIONS):
        raise RuntimeError(
            "task-summarize: the summariser produced no parseable section "
            "for %r: every one of the %d sections is empty (usual cause: a "
            "reasoning template eating the token budget before `## 1.` -- "
            "check `no_think` and `max_tokens`). First 400 chars of the "
            "answer:\n%s" % (crawl["env"], len(SECTIONS), text[:400]))
    return manual, usage


def build_worked_examples(tasks_meta, steps_by_task, budget=None):
    """The deterministic body of `## 8. Worked examples`: per scouted task,
    its intent, then an affordance-level path built from `steps.jsonl`
    (action name + the VISIBLE label of its target). No answers and no
    entity names beyond what the UI itself showed (section 3).

    `budget` bounds the WHOLE section in characters, split EQUALLY between the
    k examples. This is what keeps the k ablation honest (2026-09-19): sections
    1-7 describe the interface and are identical whatever k is, while section 8
    grows with k -- measured at ~500 chars per example, so k=5 would have run
    ~1000 chars longer than k=1 and could have won on prompt volume rather than
    on content. With a fixed budget every k gets the SAME total manual size and
    k only changes how that fixed space is divided. An over-long example is cut
    at the end of its path (the later hops), never in its intent, and the cut is
    marked so a reader of the manual knows the path continued.
    """
    blocks = []
    per = None
    if budget is not None and tasks_meta:
        # The `"\n".join` below adds len(tasks_meta) - 1 separators and they
        # count against TASK_MANUAL_SIZE_CAP like any other character, so they
        # come out of the budget here. There is deliberately NO floor on `per`:
        # the old `max(120, ...)` handed every one of k blocks 120 chars even
        # when the budget could not pay for them, which is how a k=5 cell whose
        # sections 1-7 nearly filled the cap landed OVER 6000 and went red on
        # its own size gate (wiki/T2_k5_d1, 6064 chars, 2026-09-19).
        per = (int(budget) - (len(tasks_meta) - 1)) // len(tasks_meta)
    for t in tasks_meta:
        tid = t["task_id"]
        steps = steps_by_task.get(tid) or []
        hops = []
        for s in steps:
            action = s.get("action") or "?"
            target = (s.get("target") or "").strip()
            hops.append("%s(%s)" % (action, target) if target else action)
        path = " -> ".join(hops) if hops else "(no steps recorded)"
        block = "- task %s\n  intent: %s\n  path: %s" % (
            tid, t.get("intent") or "", path)
        if per is not None and len(block) > per:
            head = "- task %s\n  intent: %s\n  path: " % (tid, t.get("intent") or "")
            room = per - len(head) - len(" ... (truncated)")
            if room > 0:
                block = head + path[:room] + " ... (truncated)"
            else:
                # Not even one hop fits. The INTENT is the supervision the k
                # ablation is about and is never cut, so the PATH goes instead.
                # The old line still appended " ... (truncated)" here, which
                # made the block LONGER than `per` in exactly the case `per`
                # was too small to begin with.
                block = head + "(elided)"
        blocks.append(block)
    return "\n".join(blocks) if blocks else "(no scouted tasks)"


def _trim_sections_1_7(sections_1_7, target):
    """Shrink an over-long sections 1-7 to at most `target` characters WITHOUT
    deleting any `## ` header, so `parse_manual` still sees the full 7-section
    schema (HARD RULE 5 is about the SCHEMA, not about how much prose each
    section is allowed to carry).

    This exists because the summariser has no size budget of its own: on
    2026-09-19 `news/T2_k1_d2` came back with 6148 chars of interface
    description against a 6000 cap, so NO section 8 could fit and the cell
    went red on its own size gate with nothing left to give back. Trimming an
    interface description is a real loss, but it only ever happens where the
    alternative is losing the whole cell, and the cut is marked in the text so
    a reader of the manual can see that it happened.

    Body lines are shortened from the END: the tail of the last section is the
    least load-bearing part of an interface description, and cutting there
    leaves sections 1-3 (purpose, navigation, page anatomy) intact.
    """
    head = sections_1_7.rstrip()
    if len(head) <= target:
        return head
    MARK = " ...(trimmed)"
    lines = head.split("\n")
    i = len(lines) - 1
    while i >= 0 and len("\n".join(lines)) > target:
        ln = lines[i]
        if ln.lstrip().startswith("#") or not ln.strip():
            i -= 1
            continue
        over = len("\n".join(lines)) - target
        if len(ln) - over >= len(MARK) + 20:
            lines[i] = ln[:len(ln) - over - len(MARK)] + MARK
        else:
            lines[i] = ""
        i -= 1
    return "\n".join(lines).rstrip()


def normalize_task_manual(sections_1_7, tasks_meta, steps_by_task):
    """Append the deterministic `## 8. Worked examples` to an already
    `normalize_manual`-normalised (sections 1-7) manual, landing the WHOLE
    manual at or under TASK_MANUAL_SIZE_CAP. Never touches
    `normalize_manual`/`SECTIONS` itself, so the task-free scout's schema is
    byte-identical (HARD RULE 5).

    Three things compete for the cap and they give way in this order:

      1. each example's PATH (the later hops), then
      2. the interface description in sections 1-7 (its tail), then
      3. nothing -- every example's INTENT is inviolable, because a k-example
         whose intent is cut is not the example the k ablation names, and the
         number of examples is never reduced for the same reason.

    Measure-then-shrink rather than predict: the budget arithmetic alone was
    wrong twice on 2026-09-19 (a 120-char-per-example floor that the budget
    could not pay for, then a summariser that spent 6148 of the 6000 cap on
    sections 1-7 by itself), and both times the cell went red on its own size
    gate with no way to give the overage back.
    """
    n, title = WORKED_EXAMPLES_SECTION
    head = sections_1_7.rstrip()
    overhead = len("\n## %d. %s\n\n" % (n, title))
    budget = max(120, TASK_MANUAL_SIZE_CAP - len(head) - overhead - 32)
    manual = None
    for _ in range(12):
        body = build_worked_examples(tasks_meta, steps_by_task, budget=budget)
        manual = head + "\n## %d. %s\n%s\n" % (n, title, body)
        over = len(manual) - TASK_MANUAL_SIZE_CAP
        if over <= 0:
            return manual
        # Pay the overshoot out of the interface description first: section 8
        # is already at its floor whenever `budget` has stopped biting.
        trimmed = _trim_sections_1_7(head, len(head) - over)
        if len(trimmed) < len(head):
            head = trimmed
            budget = max(120, TASK_MANUAL_SIZE_CAP - len(head) - overhead - 32)
            continue
        if budget > 0:
            budget = max(0, budget - over)
            continue
        break
    # Still over: the intents alone do not fit under the cap at this k, and
    # sections 1-7 are down to their headers. Return it and let the size gate
    # fail the cell loudly, rather than writing a manual that silently drops
    # examples.
    return manual


def cmd_task_summarize(args):
    d = task_draw_dir(args.era, args.site, args.level, args.k, args.draw)
    tasks_path = os.path.join(d, "tasks.json")
    crawl_path = os.path.join(d, "crawl.json")
    if not os.path.exists(tasks_path):
        raise SystemExit(
            "task-summarize: %s missing -- run task-pick first" % tasks_path)
    if not os.path.exists(crawl_path):
        raise SystemExit(
            "task-summarize: %s missing -- run task-collect first" % crawl_path)
    picked = json.load(open(tasks_path))
    tasks_meta = picked.get("tasks") or []
    if not tasks_meta:
        raise RuntimeError(
            "task-summarize: %s has no scouted tasks -- refusing to write a "
            "manual whose section 8 would silently be empty" % tasks_path)
    crawl = json.load(open(crawl_path))
    steps_by_task = {}
    for t in tasks_meta:
        sp = os.path.join(d, "steps", "%s.jsonl" % t["task_id"])
        steps_by_task[t["task_id"]] = read_jsonl(sp)
    recipes = None
    if args.level == "T2":
        raw = _load_recipes([t["task_id"] for t in tasks_meta], args.task_data)
        answers = task_reference_answers(
            task_ids=[t["task_id"] for t in tasks_meta], task_data=args.task_data)
        recipes = dict((tid, strip_answer_from_recipe(
                           txt, answer_strings=answers.get(tid) or ()))
                      for tid, txt in raw.items())
    if not _need_go("task-summarize %s against %s" % (d, args.endpoint)):
        return 0
    sections_1_7, usage = summarize_task_crawl(
        crawl, tasks_meta, args.level, endpoint=args.endpoint,
        model=args.model, recipes=recipes)
    manual = normalize_task_manual(sections_1_7, tasks_meta, steps_by_task)
    write_text(os.path.join(d, "manual.md"), manual)
    cost_path = os.path.join(d, "scout_cost.json")
    cost = json.load(open(cost_path)) if os.path.exists(cost_path) else {}
    cost["summarize_usage"] = usage
    write_json(cost_path, cost)
    print("  wrote %s (%d chars)"
          % (os.path.join(d, "manual.md"), len(manual)))
    return 0


# --------------------------------------------------------------------------
# task-paste (the paste control: XP1 / XP2)
# --------------------------------------------------------------------------

def _load_recipes(task_ids, task_data=None):
    """`{task_id: additional_instructions}` for `task_ids`, straight off the
    task file -- the raw recipe, WITH the answer. Callers that may put this
    in front of a model or write it to disk MUST run it through
    `strip_answer_from_recipe` first; this loader does not."""
    path = task_data or paths.TASK_DATA_DETERMINISTIC
    ids = set(int(t) for t in task_ids)
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        data = json.load(fh)
    rows = data if isinstance(data, list) else data.get("tasks", [])
    for r in rows:
        if isinstance(r, dict) and r.get("task_id") in ids:
            ai = r.get("additional_instructions")
            if isinstance(ai, str):
                out[r["task_id"]] = ai
    return out


def build_paste_block(tasks_meta, level, recipes=None,
                      answer_strings_by_task=None):
    """paste.md: the k intents (T1) or intents + answer-stripped recipes
    (T2), as a manual-shaped block with NO scouting content -- the paste
    control (section 9 / C4 `XP1`/`XP2`). `recipes`, if passed, are stripped
    with `strip_answer_from_recipe` here (callers pass the RAW recipe text).
    `answer_strings_by_task` (optional, `{task_id: [answer strings]}`,
    typically `task_reference_answers()`'s output) is threaded through to
    `strip_answer_from_recipe` so an answer restated OUTSIDE the narrow
    "to the user" sentence is still caught; omitting it preserves the old,
    narrower behaviour byte-for-byte (HARD RULE 5)."""
    if level not in TASK_LEVELS:
        raise ValueError("build_paste_block: level must be one of %r, got %r"
                         % (TASK_LEVELS, level))
    lines = ["# Task examples"]
    for t in tasks_meta:
        lines.append("")
        lines.append("Task %s intent: %s"
                     % (t["task_id"], t.get("intent") or ""))
        if level == "T2":
            raw = (recipes or {}).get(t["task_id"]) or ""
            ans = (answer_strings_by_task or {}).get(t["task_id"]) or ()
            stripped = strip_answer_from_recipe(raw, answer_strings=ans).strip()
            if stripped:
                lines.append("Steps:")
                lines.append(stripped)
    return "\n".join(lines).strip() + "\n"


def cmd_task_paste(args):
    d = task_draw_dir(args.era, args.site, args.level, args.k, args.draw)
    tasks_path = os.path.join(d, "tasks.json")
    if not os.path.exists(tasks_path):
        raise SystemExit(
            "task-paste: %s missing -- run task-pick first" % tasks_path)
    picked = json.load(open(tasks_path))
    tasks_meta = picked.get("tasks") or []
    if not tasks_meta:
        raise RuntimeError(
            "task-paste: %s has no scouted tasks" % tasks_path)
    recipes = None
    answers = None
    if args.level == "T2":
        recipes = _load_recipes([t["task_id"] for t in tasks_meta],
                                args.task_data)
        answers = task_reference_answers(
            task_ids=[t["task_id"] for t in tasks_meta], task_data=args.task_data)
    text = build_paste_block(tasks_meta, args.level, recipes=recipes,
                             answer_strings_by_task=answers)
    path = write_text(os.path.join(d, "paste.md"), text)
    print("  wrote %s (%d chars)" % (path, len(text)))
    return 0


# --------------------------------------------------------------------------
# Leak rules (section 4): test/scouted intent split, the answer-string rule,
# and the planted-leak selftest. `leak_check` itself grows two new OPTIONAL
# keyword arguments below; every existing caller (the task-free `gate_leak`,
# `selftest_leak_fixture`, `cmd_selftest`, `tests/test_scout.py`) passes
# neither, so nothing about it changes for them (HARD RULE 5).
# --------------------------------------------------------------------------

def test_intents(task_data=None):
    """Every TEST task's intent (`task_id <= TASK_TRAIN_ID_MIN`): forbidden
    everywhere in a task-scout manual, same 6-gram rule as the task-free
    scout's `task_intents()` (section 4 rule 1)."""
    path = task_data or paths.TASK_DATA_DETERMINISTIC
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        data = json.load(fh)
    rows = data if isinstance(data, list) else data.get("tasks", [])
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        tid = r.get("task_id")
        if not isinstance(tid, int) or tid > TASK_TRAIN_ID_MIN:
            continue
        for k in ("intent", "goal", "instruction", "task"):
            if isinstance(r.get(k), str):
                out.append(r[k])
                break
    return out


def scouted_task_intents(task_ids, task_data=None):
    """The intents of exactly `task_ids` -- allowed ONLY inside
    `## 8. Worked examples` or `paste.md` (section 4 rule 1)."""
    ids = set(int(t) for t in task_ids)
    path = task_data or paths.TASK_DATA_DETERMINISTIC
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        data = json.load(fh)
    rows = data if isinstance(data, list) else data.get("tasks", [])
    out = []
    for r in rows:
        if not isinstance(r, dict) or r.get("task_id") not in ids:
            continue
        for k in ("intent", "goal", "instruction", "task"):
            if isinstance(r.get(k), str):
                out.append(r[k])
                break
    return out


def task_reference_answers(task_ids=None, task_data=None, min_len=4):
    """`{task_id: [answer strings]}` pulled from each task's `eval` field
    (section 11 WHAT TO BUILD (c)).

    `eval` is a dict in every task file measured, but this project has a
    history of a field silently arriving as a JSON-encoded string instead
    (HARD RULE 7), so a stringified `eval` is decoded rather than skipped.
    Collects `must_include`, every alternative string inside every
    `list_match` item, `fuzzy_match`, `exact_match`, and `number_match`
    value(s); NEVER `must_exclude` (that is what a WRONG answer says, and
    treating it as a leak would forbid the correct one). Candidates shorter
    than `min_len` (normalised: lowercase, whitespace collapsed) are dropped
    so bare "yes"/"no" cannot fire the gate on their own.
    """
    path = task_data or paths.TASK_DATA_DETERMINISTIC
    out = collections.OrderedDict()
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        data = json.load(fh)
    rows = data if isinstance(data, list) else data.get("tasks", [])
    want = set(int(t) for t in task_ids) if task_ids is not None else None
    for r in rows:
        if not isinstance(r, dict):
            continue
        tid = r.get("task_id")
        if want is not None and tid not in want:
            continue
        ev = r.get("eval")
        if isinstance(ev, str):
            try:
                ev = json.loads(ev)
            except ValueError:
                ev = {}
        if not isinstance(ev, dict):
            continue
        ra = ev.get("reference_answers")
        if not isinstance(ra, dict):
            continue
        strings = []
        if isinstance(ra.get("must_include"), list):
            strings.extend(str(x) for x in ra["must_include"])
        lm = ra.get("list_match")
        if isinstance(lm, dict):
            for item in (lm.get("items") or []):
                if isinstance(item, list):
                    strings.extend(str(x) for x in item)
                elif item is not None:
                    strings.append(str(item))
        if ra.get("fuzzy_match") is not None:
            strings.append(str(ra["fuzzy_match"]))
        if ra.get("exact_match") is not None:
            ex = ra["exact_match"]
            strings.extend(str(x) for x in ex) if isinstance(ex, list) \
                else strings.append(str(ex))
        nm = ra.get("number_match")
        if isinstance(nm, dict):
            if nm.get("value") is not None:
                strings.append(str(nm["value"]))
            if isinstance(nm.get("values"), list):
                strings.extend(str(x) for x in nm["values"])
        strings = [s for s in strings
                  if len(re.sub(r"\s+", " ", s.strip())) >= min_len]
        # DROP a candidate that RESTATES THE QUESTION (2026-09-19). `fuzzy_match` often holds the
        # intent itself rather than the answer -- task 152's is the whole sentence "Is there any
        # news article regarding Salt Lake City that was ...". Section 8 and paste.md are EXPLICITLY
        # allowed to quote a scouted task's intent (rule 3b), so such a candidate makes the answer
        # gate fire on the one thing the schema tells the manual to include; 15 of 24 cells failed
        # this way, every hit a quoted intent.
        #
        # The >= 5-token floor is what keeps this from swallowing a real answer. Task 108 asks which
        # of "Physics, Chemistry, and Biology" has a maths section and the answer is "Physics": the
        # word IS in the intent, but as one of three options, so naming it is the whole answer. A
        # five-token span of the question is a restatement; a single word it happened to list is not.
        intent_low = re.sub(r"\s+", " ", str(r.get("intent") or "").strip().lower())
        if intent_low:
            strings = [x for x in strings
                       if not (len(re.sub(r"\s+", " ", x.strip().lower()).split()) >= 5
                               and re.sub(r"\s+", " ", x.strip().lower()) in intent_low)]
        if strings:
            out[tid] = strings
    return out


def _flatten_answers(answer_map):
    out = []
    for v in answer_map.values():
        out.extend(v)
    return out


def gate_task_answer_strings(scouted_ids, task_data=None):
    """The flat answer-string list `task-gates` checks manual.md/paste.md
    against: every TEST task's answers plus the SPECIFIC scouted tasks'
    own answers (section 11 (c))."""
    scouted_ans = task_reference_answers(task_ids=scouted_ids,
                                         task_data=task_data)
    return _flatten_answers(scouted_ans)


def gate_test_answer_strings(task_data=None):
    """Every TEST task's answers -- REPORTED, never fatal. See leak_check's rule-6 note: the scout
    is structurally barred from seeing a test task, so a match here is shared vocabulary."""
    test_ans = task_reference_answers(task_data=task_data)
    test_ans = collections.OrderedDict(
        (tid, v) for tid, v in test_ans.items()
        if isinstance(tid, int) and tid <= TASK_TRAIN_ID_MIN)
    return _flatten_answers(test_ans)


def leaky_own_answer_ids(task_ids, recipes, answer_map):
    """`task-pick --exclude-leaky` (DOMSCOUT_KL_RUN_PLAN.md / section 11 DO
    item 1): the subset of `task_ids` that can NEVER pass `task-gates` rule
    (c) for their own answer, because `strip_answer_from_recipe` -- run with
    that task's OWN reference-answer strings, same as `cmd_task_paste` /
    `cmd_task_summarize` already do -- leaves at least one of those strings
    sitting in the recipe under the exact normalisation `leak_check` rule 6
    uses (whitespace-collapsed, lowercased, length >= 4). Typical cause: the
    answer is also a search/lookup term used mid-recipe (e.g. real task 108's
    step 1, "Type Physics in the search box", answer "Physics"), which no
    step-dropping heuristic can remove without deleting the recipe's own
    method.

    `recipes`: `{task_id: raw additional_instructions}` (`_load_recipes`'s
    shape, UNSTRIPPED). `answer_map`: `{task_id: [answer strings]}`
    (`task_reference_answers()`'s shape). A task with no recipe or no answer
    strings is never excluded -- there is nothing for it to leak. Pure: no
    file I/O, so it is directly testable both against fabricated fixtures
    and against the real 128-task train set.
    """
    excluded = set()
    for tid in task_ids:
        raw = recipes.get(tid)
        answers = answer_map.get(tid)
        if not raw or not answers:
            continue
        stripped = strip_answer_from_recipe(raw, answer_strings=answers)
        norm = re.sub(r"\s+", " ", stripped.lower())
        for a in answers:
            a_norm = re.sub(r"\s+", " ", str(a).strip().lower())
            if len(a_norm) >= 4 and a_norm in norm:
                excluded.add(tid)
                break
    return excluded


def selftest_task_leak_fixture():
    """Section 11 (d): plant a TEST intent, a TEST answer and a SCOUTED-task
    answer and confirm `leak_check`'s new arguments catch all three -- the
    task-scout counterpart of `selftest_leak_fixture`."""
    test_intents_ = ["List all articles from the Related Pages sections of "
                     "the Biophysics and Biochemistry pages"]
    scouted_intents_ = ["Compare the population of Australia's largest city "
                        "to Canada's largest city"]
    good = normalize_manual(
        "# Site manual: wiki\n## 1. Purpose and top-level navigation\nA "
        "left-hand column of links leads to the main page and to recent "
        "changes.\n", "wiki")
    good = good.rstrip() + ("\n## 8. Worked examples\n- task 106\n  intent: %s"
                            "\n  path: search -> result -> article\n"
                            % scouted_intents_[0])

    planted_test_intent = good.replace(
        "A left-hand",
        "List all articles from the Related Pages sections of the "
        "Biophysics and Biochemistry pages. A left-hand")
    planted_test_answer = good.replace(
        "A left-hand", "Bionics, Computational biology. A left-hand")
    planted_scouted_answer = good.replace(
        "A left-hand", "Around 2.5 Million. A left-hand")

    r_clean = leak_check(good, task_intents=test_intents_,
                         scouted_intents=scouted_intents_)
    r_ti = leak_check(planted_test_intent, task_intents=test_intents_,
                      scouted_intents=scouted_intents_)
    # A TEST answer goes down the REPORTED channel (rule 6w): it must be seen, and must NOT fail
    # the cell. See leak_check's rule-6 note for why that is structural, not a loosened threshold.
    r_ta = leak_check(planted_test_answer, task_intents=test_intents_,
                      scouted_intents=scouted_intents_,
                      warn_answer_strings=["Bionics, Computational biology"])
    r_sa = leak_check(planted_scouted_answer, task_intents=test_intents_,
                      scouted_intents=scouted_intents_,
                      answer_strings=["Around 2.5 Million"])
    ok = (not r_clean["hits"]
          and any(h["rule"] == "3-task-overlap" for h in r_ti["hits"])
          and not r_ta["hits"]
          and any(w["rule"] == "6w-test-answer-coincidence"
                  for w in r_ta["warnings"])
          and any(h["rule"] == "6-answer-leak" for h in r_sa["hits"]))
    return {"ok": bool(ok),
            "clean_hits": len(r_clean["hits"]),
            "test_intent_hits": [h["rule"] for h in r_ti["hits"]],
            "test_answer_hits": [h["rule"] for h in r_ta["hits"]],
            "test_answer_warnings": [w["rule"] for w in r_ta["warnings"]],
            "scouted_answer_hits": [h["rule"] for h in r_sa["hits"]]}


#: A scout episode counts as "solved" at this reward or above. Matches the
#: deterministic judge's binary pass/fail convention this codebase already
#: uses elsewhere (evalbridge's `cum_reward`); recorded, not hidden, so a
#: report reading gates.json can rebuild "scout solved n/k" with a different
#: threshold later if the judge ever stops being binary.
SCOUT_SOLVED_REWARD_THRESHOLD = 1.0


def build_scout_solved_summary(scouted_ids, episodes):
    """DOMSCOUT_KL_RUN_PLAN.md / section 11 DO item 5c: per scouted task, the
    scout's OWN outcome -- reward stays out of every PROMPT (task-summarize /
    task-paste never see it), but the report needs "k, chars, scout solved
    n/k" next to each manual, so it is recorded here, in gates.json, instead.
    Pure: `episodes` is already-loaded `episodes.jsonl` rows (a list of
    dicts with at least `task_id` and `reward`); a task with no matching
    episode row gets `reward: null` and does not count as solved."""
    by_id = dict((e.get("task_id"), e) for e in episodes)
    rewards = collections.OrderedDict()
    n_solved = 0
    for tid in scouted_ids:
        ep = by_id.get(tid) or {}
        r = ep.get("reward")
        rewards[str(tid)] = r
        if r is not None and float(r) >= SCOUT_SOLVED_REWARD_THRESHOLD:
            n_solved += 1
    return collections.OrderedDict([
        ("k", len(scouted_ids)),
        ("n_solved", n_solved),
        ("reward_threshold", SCOUT_SOLVED_REWARD_THRESHOLD),
        ("rewards", rewards),
    ])


def gate_task_size(text, name):
    n = len(text)
    return collections.OrderedDict([
        ("name", name), ("chars", n), ("cap", TASK_MANUAL_SIZE_CAP),
        ("pass", n <= TASK_MANUAL_SIZE_CAP)])


def cmd_task_gates(args):
    d = task_draw_dir(args.era, args.site, args.level, args.k, args.draw)
    tasks_path = os.path.join(d, "tasks.json")
    if not os.path.exists(tasks_path):
        raise SystemExit(
            "task-gates: %s missing -- run task-pick first" % tasks_path)
    picked = json.load(open(tasks_path))
    scouted_ids = [t["task_id"] for t in (picked.get("tasks") or [])]
    manual_path = os.path.join(d, "manual.md")
    paste_path = os.path.join(d, "paste.md")
    if not os.path.exists(manual_path) or not os.path.exists(paste_path):
        raise SystemExit(
            "task-gates: manual.md/paste.md missing under %s -- run "
            "task-summarize and task-paste first" % d)
    manual_text, paste_text = read_text(manual_path), read_text(paste_path)
    crawl_path = os.path.join(d, "crawl.json")
    labels = (observed_labels(json.load(open(crawl_path)))
             if os.path.exists(crawl_path) else set())

    size = collections.OrderedDict([
        ("manual", gate_task_size(manual_text, "manual.md")),
        ("paste", gate_task_size(paste_text, "paste.md"))])

    test_intents_ = test_intents(args.task_data)
    scouted_intents_ = scouted_task_intents(scouted_ids, args.task_data)
    answers = gate_task_answer_strings(scouted_ids, args.task_data)
    test_answers = gate_test_answer_strings(args.task_data)

    r_manual = leak_check(manual_text, sections="full", labels=labels,
                          task_intents=test_intents_,
                          scouted_intents=scouted_intents_,
                          answer_strings=answers,
                          warn_answer_strings=test_answers)
    r_paste = leak_check(paste_text, sections="full", labels=labels,
                         task_intents=test_intents_,
                         scouted_intents=scouted_intents_,
                         answer_strings=answers,
                         warn_answer_strings=test_answers)
    planted = selftest_task_leak_fixture()
    n_hits = len(r_manual["hits"]) + len(r_paste["hits"])
    n_waived = len(r_manual["waived"]) + len(r_paste["waived"])
    n_warn = len(r_manual.get("warnings") or []) + len(r_paste.get("warnings") or [])

    # k in {1,3,5} ablation (section 11 DO item 5c): the scout's OWN outcome
    # per scouted task, from episodes.jsonl (reward never reaches the manual
    # or paste prompt above -- it is read here only for the report).
    episodes = read_jsonl(os.path.join(d, "episodes.jsonl"))
    scout_solved = build_scout_solved_summary(scouted_ids, episodes)

    out = collections.OrderedDict([
        ("gate", "task-gates"),
        ("site", args.site), ("era", int(args.era)), ("level", args.level),
        ("k", int(args.k)), ("draw", int(args.draw)),
        ("size", size),
        ("leak", collections.OrderedDict([
            ("manual", r_manual), ("paste", r_paste),
            ("n_hits", n_hits), ("n_waived", n_waived),
            ("n_test_answer_warnings", n_warn)])),
        ("planted_caught", planted["ok"]), ("planted", planted),
        ("scout_solved", scout_solved),
        ("pass", bool(size["manual"]["pass"] and size["paste"]["pass"]
                      and n_hits == 0 and planted["ok"])),
    ])
    write_json(os.path.join(d, "gates.json"), out)
    print("  task-gates %s: size manual=%d/paste=%d chars, leak hits=%d "
          "waived=%d, planted=%s -> %s"
          % (d, size["manual"]["chars"], size["paste"]["chars"], n_hits,
             n_waived, planted["ok"], "PASS" if out["pass"] else "FAIL"))
    return 0 if out["pass"] else 2


# --------------------------------------------------------------------------
# Driver commands
# --------------------------------------------------------------------------

def _go():
    return os.environ.get("GO", "") == "1"


def _need_go(what):
    if _go():
        return True
    print("DRY RUN: %s\n  re-run with GO=1 to actually do it." % what)
    return False


def _parse_eras(s):
    return tuple(int(x) for x in re.split(r"[,\s]+", str(s).strip()) if x)


def cmd_plan(args):
    eras = _parse_eras(args.versions)
    cfg = CrawlConfig(max_pages=args.budget, max_depth=args.depth)
    print("scout plan (S-crawl, primary variant)")
    print("  versions      : %s" % ", ".join("v%d" % e for e in eras))
    print("  envs          : %s" % ", ".join(cells.ENVIRONMENTS))
    print("  budget        : %d pages, depth <= %d, one fixed search probe"
          % (cfg.max_pages, cfg.max_depth))
    print("  out root      : %s" % OUT_CRAWL)
    print("  crawl config  : %s" % cfg.fingerprint())
    print("  summariser    : %s @ %s (T=0, seed=0)"
          % (args.model, args.endpoint or "<--endpoint required>"))
    print("  interpreter   : %s (playwright)" % paths.PY_BENCH)
    print("")
    print("  steps: crawl -> summarize -> ablate -> probe-samples -> gates")
    return 0


def cmd_crawl(args):
    eras = _parse_eras(args.versions)
    envs = tuple(args.envs.split(",")) if args.envs else cells.ENVIRONMENTS
    if not _need_go("crawl %s x %s (starts env servers, drives chromium)"
                    % (["v%d" % e for e in eras], list(envs))):
        return 0
    capture._require_playwright()
    from playwright.sync_api import sync_playwright

    budgets = _parse_eras(args.budget_sweep) if args.budget_sweep else ()
    rc = 0
    # The shop's landing page samples its featured products with an unseeded
    # `random.sample` per REQUEST (webshop `app.py`), so two identical crawls
    # follow different item links and G1 can never pass on an era whose
    # catalogue exceeds the page budget -- measured 0/10 identical on v3/v4/v5,
    # 10/10 on v1/v2/v6 whose listings are complete rather than sampled. The
    # flag is read by the server child process, so it must be set BEFORE
    # EnvServers starts them. It is opt-in and changes nothing for evals.
    os.environ["TW_DETERMINISTIC_FEATURED"] = "1"
    for era in eras:
        with capture.EnvServers(era, port_base=args.port_base, envs=envs) as srv:
            with sync_playwright() as pw:
                browser, ctx = capture._new_context(pw, capture.CaptureConfig())
                try:
                    page = ctx.new_page()
                    for env in envs:
                        d = env_dir(era, env)
                        if not os.path.isdir(d):
                            os.makedirs(d)
                        cfg = CrawlConfig(max_pages=args.budget,
                                          max_depth=args.depth)
                        crawl = crawl_env(page, srv.url(env), env, era,
                                          cfg=cfg, out_dir=d)
                        write_json(os.path.join(d, "crawl.json"), crawl)
                        print("  %-9s %2d pages  sha %s  probe=%r"
                              % (cells.Cell(env, era).key, crawl["n_pages"],
                                 crawl["crawl_sha256"][:12],
                                 (crawl["probe"] or {}).get("query")))
                        for b in budgets:
                            if b == args.budget:
                                continue
                            c2 = crawl_env(page, srv.url(env), env, era,
                                           cfg=CrawlConfig(max_pages=b,
                                                           max_depth=args.depth,
                                                           screenshots=False),
                                           out_dir=None)
                            write_json(os.path.join(d, "variants",
                                                    "crawl_b%d.json" % b), c2)
                        if args.determinism > 1:
                            r = check_determinism(page, srv.url(env), env, era,
                                                  n=args.determinism,
                                                  cfg=cfg, out_dir=d)
                            print("    G1: %d/%d identical (%d distinct sha)"
                                  % (r["n"] if r["identical"] else 0, r["n"],
                                     r["n_distinct"]))
                            rc = rc or (0 if r["identical"] else 2)
                finally:
                    ctx.close()
                    browser.close()
    return rc


def cmd_summarize(args):
    eras = _parse_eras(args.versions)
    envs = tuple(args.envs.split(",")) if args.envs else cells.ENVIRONMENTS
    if not _need_go("summarize %s x %s against %s"
                    % (["v%d" % e for e in eras], list(envs), args.endpoint)):
        return 0
    cost = []
    for era in eras:
        for env in envs:
            d = env_dir(era, env)
            for crawl_path, dest in _summary_targets(d, args):
                crawl = json.load(open(crawl_path))
                text, usage = summarize_crawl(crawl, endpoint=args.endpoint,
                                              model=args.model)
                write_text(dest, text)
                cost.append({"cell": cells.Cell(env, era).key,
                             "crawl": os.path.basename(crawl_path),
                             "pages": crawl.get("n_pages"),
                             "prompt_tokens": usage.get("prompt_tokens"),
                             "completion_tokens": usage.get("completion_tokens")})
                print("  %-9s %-16s -> %s (%d chars)"
                      % (cells.Cell(env, era).key,
                         os.path.basename(crawl_path),
                         os.path.relpath(dest, OUT_CRAWL), len(text)))
    write_json(os.path.join(OUT_CRAWL, "scout_cost.json"), cost)
    return 0


def _summary_targets(d, args):
    """(crawl.json, manual.md) plus every budget variant that was crawled."""
    out = []
    main = os.path.join(d, "crawl.json")
    if os.path.exists(main):
        out.append((main, os.path.join(d, "manual.md")))
    vd = os.path.join(d, "variants")
    if args.variants and os.path.isdir(vd):
        for f in sorted(os.listdir(vd)):
            m = re.match(r"^crawl_(b\d+)\.json$", f)
            if m:
                out.append((os.path.join(vd, f),
                            os.path.join(vd, "manual_%s.md" % m.group(1))))
    return out


def cmd_ablate(args):
    eras = _parse_eras(args.versions)
    for era in eras:
        try:
            parts = build_version_manuals(era)
        except IOError as e:
            print("  v%d: %s" % (era, e))
            continue
        print("  v%d  full=%d  affordance=%d  style=%d chars"
              % (era, len(parts["full"]), len(parts["affordance"]),
                 len(parts["style"])))
    desc = write_descriptions(eras=eras)
    for name in VARIANTS:
        print("  wrote descriptions_%s.json (%d versions)"
              % (name, len(desc[name])))
    return 0


def cmd_probe_samples(args):
    info = build_probe_samples(device=args.device)
    print("  wrote %s: %d rows, dim %d, %d crawl groups"
          % (info["path"], info["n_rows"], info["dim"], info["n_groups"]))
    for v in info["variants"]:
        try:
            rep = run_identity_probe(npz_path=info["path"], variant=v)
        except Exception as e:                        # noqa: BLE001
            print("  probe(%s): %s" % (v, e))
            continue
        a = rep["accuracy"]
        print("  probe(%-10s) era=%.3f env=%.3f cell=%.3f"
              % (v, a["era"], a["env"], a["cell"]))
    return 0


def cmd_gates(args):
    eras = _parse_eras(args.versions)
    out = collections.OrderedDict()
    out["G1_determinism"] = gate_determinism(eras=eras)
    out["G2_leak"] = gate_leak(eras=eras)
    cost = os.path.join(OUT_CRAWL, "scout_cost.json")
    if os.path.exists(cost):
        rows = json.load(open(cost))
        # The cost that Part D's compute-matching argument is about is the cost
        # of the manual the arms ACTUALLY use -- the main crawl. The G3 budget
        # variants are a diagnostic that never reaches an eval, so counting them
        # here would overstate the scout by ~5x.
        used = [r for r in rows if r.get("crawl") == "crawl.json"]
        pages = sum(r.get("pages") or 0 for r in used)
        toks = sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
                   for r in used)
        n_v = float(max(1, len(eras)))
        out["scout_cost"] = {
            "n_calls": len(used), "n_calls_incl_variants": len(rows),
            "pages": pages, "tokens": toks,
            "pages_per_version": pages / n_v,
            # 103 test tasks per version. Well under one agent step per episode
            # is the claim Part D makes; this is the number that backs it.
            "steps_per_episode": pages / (103.0 * n_v)}
    out["manual_size"] = gate_manual_size(eras=eras)
    g4 = os.path.join(OUT_CRAWL, "g4_separability.json")
    if os.path.exists(g4):
        rep = json.load(open(g4))
        rep["gate"] = "G4"
        rep["reported_not_gated"] = True
        rep.pop("cosine", None)          # the matrix stays in its own file
        out["G4_separability"] = rep
    npz = os.path.join(OUT_CRAWL, "probe_samples.npz")
    if os.path.exists(npz):
        g3 = {}
        for v in VARIANTS:
            try:
                rep = run_identity_probe(npz_path=npz, variant=v)
                g3[v] = dict(rep["accuracy"])
                g3[v]["structure"] = rep["structure"]
            except Exception as e:                    # noqa: BLE001
                g3[v] = {"error": str(e)}
        out["G3_identity_probe"] = {"gate": "G3", "reported_not_gated": True,
                                    "accuracy": g3}
    write_json(os.path.join(OUT_CRAWL, "gates.json"), out)

    print("=" * 72)
    print("scout gates")
    print("=" * 72)
    for k in out:
        v = out[k]
        verdict = ("PASS" if v.get("pass") else
                   ("report" if v.get("reported_not_gated") else
                    ("-" if "pass" not in v else "FAIL")))
        print("  %-22s %s" % (k, verdict))
    if "G2_leak" in out:
        g = out["G2_leak"]
        print("  G2: %d manual-variant(s), %d hit(s), %d waived label quote(s), "
              "planted leaks caught=%s"
              % (g["n_manuals"], g["n_hits"], g["n_waived"], g["planted_caught"]))
    g4r = out.get("G4_separability")
    if g4r:
        print("  G4 off-diagonal cosine mean %.4f (vision %.3f, hand-written "
              "text %.3f)" % (g4r["off_diagonal_mean"], g4r["reference_vision"],
                              g4r["reference_text"]))
    ms = out.get("manual_size") or {}
    for wmsg in ms.get("warnings", []):
        print("  WARN manual size: %s -- the policy's prompt budget is shared "
              "with the AXTree, which AgentLab shrinks to fit" % wmsg)
    if "G3_identity_probe" in out:
        for v, acc in out["G3_identity_probe"]["accuracy"].items():
            if "error" not in acc:
                print("  G3 %-10s era=%.3f env=%.3f cell=%.3f  (%s)"
                      % (v, acc["era"], acc["env"], acc["cell"],
                         acc.get("structure", "-")))
    print("  wrote %s" % os.path.join(OUT_CRAWL, "gates.json"))
    blocking = [k for k in ("G1_determinism", "G2_leak")
                if not out.get(k, {}).get("pass")]
    if blocking:
        print("  BLOCKING: %s -- Part D does not start until these pass"
              % ", ".join(blocking))
        return 2
    return 0


def cmd_selftest(args):
    """Everything that needs no server, no GPU and no network."""
    ok, fail = 0, 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
            print("  ok   %s" % name)
        else:
            fail += 1
            print("  FAIL %s" % name)

    text = normalize_manual("# Site manual: wiki\n"
                            "## 1. Purpose and top-level navigation\nnav.\n"
                            "## 7. Visual style\nplain tables, roughly 2001.\n",
                            "wiki")
    _, body = parse_manual(text)
    check("normalize_manual emits all 7 sections", len(body) == 7)
    check("missing sections are filled", body[3] == EMPTY_SECTION)
    aff = affordance_only(text)
    check("affordance drops section 7", "## 7." not in aff and "## 6." in aff)
    sty = style_only(text)
    check("style keeps only section 7", "## 7." in sty and "## 1." not in sty)
    check("style may carry a year", not leak_check(sty, sections="style")["hits"])
    check("affordance may not carry a year",
          any(h["rule"] == "2-year-in-affordance"
              for h in leak_check(normalize_manual(
                  "# Site manual: wiki\n## 1. Purpose and top-level navigation\n"
                  "built in 2005.\n", "wiki"), sections="affordance")["hits"]))
    planted = selftest_leak_fixture()
    check("planted v3 / 2005 / task-6gram are all caught", planted["ok"])
    quoted = normalize_manual(
        "# Site manual: wiki\n## 1. Purpose and top-level navigation\n"
        "A link labelled Printable version sits under the article.\n", "wiki")
    r = leak_check(quoted, labels={"printable version"})
    check("a quoted UI label waives the forbidden token",
          not r["hits"] and r["waived"])
    check("the same token unquoted is still a hit",
          any(h["rule"] == "1-comparative"
              for h in leak_check(quoted, labels=set())["hits"]))
    check("manual_block framing matches the eval injector",
          manual_block("x") == "\n# Site manuals\nx\n")
    # The shop's `/abc` session prefix is KEPT on purpose (see normalize_path:
    # stripping it made the canonical path non-invertible and collapsed every
    # shop crawl to its landing page). Only the port is dropped.
    check("normalize_path strips the port and keeps the shop session",
          normalize_path("http://localhost:5310/abc/search?q=a",
                         "http://localhost:5310/abc") == "/abc/search?q=a")
    check("normalize_path refuses off-site links",
          normalize_path("http://example.com/x", "http://localhost:5310") is None)
    fake = {"env": "wiki", "era": 1, "pages": [], "n_pages": 0}
    check("canonical_sha ignores screenshots",
          canonical_sha(fake) == canonical_sha(dict(fake, screenshots=[1, 2])))
    print("\n%d passed, %d failed" % (ok, fail))
    return 0 if fail == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="scout_plan.md: crawl a TimeWarp version, write a site "
                    "manual, ablate it, and run the gates. Dry-run unless GO=1.")

    # Shared options live on a PARENT parser, not on the top-level one, so they
    # are accepted AFTER the subcommand. `capture`'s top-level-only --port-base
    # is the shape that cost a launch round (tests/test_cli_callsites.py, bug 3)
    # -- `scout crawl --versions 1,6` has to work, because that is what anyone
    # will type.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--versions", default=",".join(str(e) for e in cells.ERAS))
    common.add_argument("--envs", default=None,
                        help="comma-separated subset of wiki,news,shop")
    common.add_argument("--port-base", type=int, default=5400,
                        help="disjoint from capture's 5300 so both can run at once")
    common.add_argument("--budget", type=int, default=25, help="pages per env")
    common.add_argument("--depth", type=int, default=3)
    common.add_argument("--endpoint", default=os.environ.get("SCOUT_ENDPOINT"),
                        help="OpenAI-compatible base, e.g. http://localhost:8000/v1")
    common.add_argument("--model", default=os.environ.get("SCOUT_MODEL",
                                                          "Qwen/Qwen3.5-9B"))
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("plan", parents=[common], help="print what would run")
    p.set_defaults(func=cmd_plan)

    c = sub.add_parser("crawl", parents=[common], help="S-crawl every (env, version)")
    c.add_argument("--determinism", type=int, default=0,
                   help="also run G1 with this many repeats (10 for the gate)")
    c.add_argument("--budget-sweep", default=None,
                   help="extra page budgets for G3's variants, e.g. 15,20,30,40")
    c.set_defaults(func=cmd_crawl)

    s = sub.add_parser("summarize", parents=[common], help="crawl JSON -> manual.md (one call each)")
    s.add_argument("--variants", action="store_true",
                   help="also summarise the budget-sweep crawls (G3 samples)")
    s.set_defaults(func=cmd_summarize)

    a = sub.add_parser("ablate", parents=[common], help="manual_{full,affordance,style} + descriptions")
    a.set_defaults(func=cmd_ablate)

    b = sub.add_parser("probe-samples", parents=[common], help="G3 npz + the identity probe")
    b.add_argument("--device", default="cpu")
    b.set_defaults(func=cmd_probe_samples)

    g = sub.add_parser("gates", parents=[common], help="G1-G3 -> gates.json")
    g.set_defaults(func=cmd_gates)

    t = sub.add_parser("selftest", parents=[common], help="every check that needs nothing")
    t.set_defaults(func=cmd_selftest)

    # -- S-task: the task-grounded scout (scout_task_plan.md) ---------------
    task_common = argparse.ArgumentParser(add_help=False)
    task_common.add_argument("--site", required=True, choices=cells.ENVIRONMENTS)
    task_common.add_argument("--era", type=int, default=cells.NEUTRAL_ERA)
    task_common.add_argument("--level", choices=TASK_LEVELS, default="T1")
    task_common.add_argument("--k", type=int, default=4)
    task_common.add_argument("--draw", type=int, required=True,
                             help="draw index; tied to the training seed (C3)")
    task_common.add_argument("--task-data", default=None,
                             help="override the deterministic-judge task file")

    tp = sub.add_parser("task-pick", parents=[task_common],
                        help="seeded draw of k single-site train tasks -> tasks.json")
    tp.add_argument("--exclude-intent-overlap", action="store_true",
                    help="drop from the pool every train task whose own "
                         "intent shares a 6-gram with a TEST task's intent "
                         "(default OFF: byte-identical to the pre-campaign "
                         "pool; the ICL-LOSO campaign turns this ON because "
                         "leak_check rule 3 is un-waivable and would "
                         "rule-red such a cell no matter what the scout did)")
    tp.add_argument("--exclude-leaky", action="store_true",
                    help="drop from the pool every train task whose own "
                         "answer-stripped recipe still contains its own "
                         "reference answer (default OFF: byte-identical to "
                         "the pre-campaign pool; the DOMSCOUT campaign turns "
                         "this ON so a picked task can never fail task-gates "
                         "rule (c) for its own answer)")
    tp.set_defaults(func=cmd_task_pick)

    tsu = sub.add_parser("task-subset", parents=[task_common],
                         help="k=K cell built from a k=FROM-K parent's own "
                              "first K picked tasks (the k in {1,3,5} "
                              "ablation nesting)")
    tsu.add_argument("--from-k", type=int, default=5,
                     help="the parent cell's k (DOMSCOUT: the 5 episodes "
                          "every k in {1,3,5} cell is carved from)")
    tsu.set_defaults(func=cmd_task_subset)

    tc = sub.add_parser("task-collect", parents=[task_common],
                        help="scout episodes -> crawl.json (re-opens visited pages)")
    tc.add_argument("--budget", type=int, default=25,
                    help="page cap, same knob as `crawl --budget`")
    tc.add_argument("--port-base", type=int, default=5450,
                    help="disjoint from `crawl`'s 5400 and capture's 5300")
    tc.set_defaults(func=cmd_task_collect)

    ts = sub.add_parser("task-summarize", parents=[task_common],
                        help="crawl.json -> manual.md (sections 1-7 + worked examples)")
    ts.add_argument("--endpoint", default=os.environ.get("SCOUT_ENDPOINT"),
                    help="OpenAI-compatible base, e.g. http://localhost:8000/v1")
    ts.add_argument("--model", default=os.environ.get("SCOUT_MODEL",
                                                       "Qwen/Qwen3.5-9B"))
    ts.set_defaults(func=cmd_task_summarize)

    tpa = sub.add_parser("task-paste", parents=[task_common],
                         help="intents (+ T2 answer-stripped recipes) -> paste.md")
    tpa.set_defaults(func=cmd_task_paste)

    tg = sub.add_parser("task-gates", parents=[task_common],
                        help="size cap + leak rules -> gates.json")
    tg.set_defaults(func=cmd_task_gates)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 1
    paths.ensure_out_dirs()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
