"""Deterministic screenshot + a11y-role capture for the conditioning signal (4.2).

4.2 makes this a lock-it-before-you-collect problem: "The conditioning signal
now determines the model's *weights*, so nondeterminism in it is nondeterminism
in the policy." 7 Phase 0 turns that into an acceptance test -- "same page ->
identical conditioning embedding across 10 repeats" -- which `check_determinism`
implements.

Three pieces:

* `CaptureConfig` -- every knob that could make two captures of the same page
  differ, pinned and justified in-line.
* `capture_cell`  -- one (env, era) cell -> screenshot.png, page.html,
  axtree.json, roles.json. `roles.json` is 4.2's "structural side-channel":
  a histogram over ARIA roles in the fixed order of `ROLE_VOCAB`, so the vector
  is comparable across pages, eras and runs.
* `EnvServers`    -- starts wiki/news/webshop for one era on non-colliding
  ports, waits for health, exports the TW_* variables, and tears down by
  process group.

Facts that shaped the design, all verified rather than assumed:

* **Every TimeWarp task starts on its site's landing page.** All 231 records in
  `browsergym/timewarp/data/test.raw.json` have `start_url` in
  {`__WIKI__`, `__NEWS__`, `__WEBSHOP__`} -- a bare site-root placeholder that
  instance.py:35 resolves from `TW_WIKI` / `TW_NEWS` / `TW_WEBSHOP`. There is no
  per-task deep link. Under 4.3's per-episode granularity this means the
  conditioning observation is **the same landing page for every task in a
  cell**, so the whole conditioning input has exactly 18 distinct values. That
  is good for 3's corollary (the screenshot cannot leak the task) and bad for
  6.7's memorization probe (an 18-entry lookup table is trivially learnable).
  Capturing landing pages is therefore not a shortcut; it is the real signal.
* **`page.accessibility` does not exist in playwright 1.61** (the version in
  `tw_r1_q3`); it was removed upstream. `page.aria_snapshot()` survives and is
  tried first, with a DOM-derived implicit-role map as the always-computed
  fallback. `roles.json` records which source produced the primary vector.
* **Playwright is installed in `tw_r1_q3` only, not `tw_web`** -- contrary to
  what the project brief states. Use `paths.PY_BENCH`.
* Chromium hard-refuses a fixed list of ports with ERR_UNSAFE_PORT no matter
  what is listening; `UNSAFE_PORTS` is copied from
  `run_v6_eval_q35_sonnet.sh:98-105`, where using 5060 as a port base once
  produced 219/309 errored episodes that looked exactly like a method failure.
* The webshop URL carries a session suffix (`/abc`); wiki and news do not.

Run:  cd adapter_project && $PY_BENCH -m adaptercl.capture determinism --era 6
      cd adapter_project && $PY_BENCH -m adaptercl.capture all --eras 1,2,3,4,5,6
  where PY_BENCH = <bench-env>/bin/python
"""

from __future__ import print_function

import argparse
import collections
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time

from . import cells, paths

# --------------------------------------------------------------------------
# Capture configuration (4.2 "Capture protocol -- lock before collecting")
# --------------------------------------------------------------------------

#: Injected after load. Playwright's own `animations="disabled"` screenshot
#: option freezes CSS animations at their *end* state, but it does not stop a
#: transition that a hover/focus already started, and it does nothing to
#: `scroll-behavior: smooth` (wiki era 5 and shop era 5 both set it). Belt and
#: braces, per 4.2's explicit request.
FREEZE_CSS = ("*,*::before,*::after{animation:none!important;"
              "transition:none!important;scroll-behavior:auto!important;"
              "caret-color:transparent!important}")


class CaptureConfig(object):
    """Every source of capture-to-capture variation, pinned.

    Not a dataclass: this module must stay readable from python 3.6 tooling and
    `dataclasses` does not exist there.
    """

    __slots__ = (
        "viewport_width", "viewport_height", "device_scale_factor",
        "wait_until", "settle_ms", "nav_timeout_ms", "scroll_to_top",
        "disable_animations", "locale", "timezone_id", "color_scheme",
        "reduced_motion", "forced_colors", "full_page", "browser_args",
        "role_source", "headless", "screenshot_scale",
    )

    def __init__(self, **kw):
        # -- geometry ------------------------------------------------------
        # 1280x1024 is the browsergym/AgentLab default for TimeWarp rollouts;
        # matching it means the conditioning image is the image the policy
        # actually sees at inference, not a differently-laid-out proxy.
        self.viewport_width = 1280
        self.viewport_height = 1024
        # 1, not 2. A retina factor doubles the pixel count for no extra
        # information and makes the PNG bytes depend on the rasterizer's
        # subpixel behaviour, which is the thing we are trying to pin down.
        self.device_scale_factor = 1
        # Screenshot in CSS pixels so the file is independent of
        # device_scale_factor even if someone changes it.
        self.screenshot_scale = "css"
        # Above-the-fold only. full_page height depends on content length, so
        # two eras of the same site would produce different-shaped tensors and
        # the vision tower would see different token counts per cell -- a
        # confound between "era" and "sequence length".
        self.full_page = False

        # -- when is the page "done" ---------------------------------------
        # networkidle rather than load: the era-4/5 themes fetch fonts and
        # images after DOMContentLoaded, and a `load`-timed screenshot catches
        # them mid-swap. TimeWarp is containerised (no ads, no A/B tests, no
        # live drift, 4.2), so networkidle actually terminates here.
        self.wait_until = "networkidle"
        # Fixed extra settle *after* networkidle. Deliberately a constant and
        # not a "wait for stability" loop: a loop makes the wait depend on
        # machine load, which is exactly the nondeterminism 4.2 warns about.
        self.settle_ms = 750
        self.nav_timeout_ms = 45000

        # -- rendering state ------------------------------------------------
        # A page restored from bfcache or re-entered mid-scroll screenshots
        # from wherever it was left.
        self.scroll_to_top = True
        self.disable_animations = True

        # -- environment ----------------------------------------------------
        # Locale and timezone are pinned because the news app renders
        # timestamps and the shop app renders prices; either would make the
        # screenshot depend on when and where it was taken.
        self.locale = "en-US"
        self.timezone_id = "UTC"
        # The themes ship no dark variants, but `prefers-color-scheme` defaults
        # to the host's setting, so pin it rather than inherit it.
        self.color_scheme = "light"
        self.reduced_motion = "reduce"
        self.forced_colors = "none"

        self.headless = True
        self.browser_args = [
            # sRGB, not the display profile: colour management is the most
            # common cause of byte-level PNG differences between machines.
            "--force-color-profile=srgb",
            # Subpixel antialiasing depends on the font stack and the GPU.
            "--disable-lcd-text",
            "--font-render-hinting=none",
            # Scrollbar width differs with the GTK theme and shifts layout.
            "--hide-scrollbars",
            # Rasterize on the CPU: identical output regardless of which node
            # (and which GPU) the capture happens to land on.
            "--disable-gpu",
            # Do not screenshot a partially-composited frame.
            "--run-all-compositor-stages-before-draw",
            "--disable-new-content-rendering-timeout",
            # Lazy loading defers below-fold images by a *time*-dependent
            # heuristic; PaintHolding can serve the previous page's pixels.
            "--disable-features=LazyImageLoading,LazyFrameLoading,PaintHolding",
            "--disable-back-forward-cache",
            # CHPC compute nodes have a small /dev/shm; without this Chromium
            # crashes on large pages rather than producing a bad screenshot.
            "--disable-dev-shm-usage",
            "--no-sandbox",
        ]

        # "auto" -> aria_snapshot if available, else DOM-derived roles.
        self.role_source = "auto"

        for k, v in kw.items():
            if k not in self.__slots__:
                raise TypeError("CaptureConfig has no option %r" % (k,))
            setattr(self, k, v)

    def as_dict(self):
        return collections.OrderedDict(
            (k, getattr(self, k)) for k in sorted(self.__slots__))

    def fingerprint(self):
        """Short hash of the whole config. Written into every meta.json so a
        capture can never be silently compared against one taken under
        different settings."""
        blob = json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]


# --------------------------------------------------------------------------
# The structural side-channel (4.2 "histogram over a11y roles present")
# --------------------------------------------------------------------------

#: Fixed, sorted ARIA role vocabulary. The order defines the feature ordering
#: of the role vector and is frozen as of this commit: never reorder it and
#: never remove an entry -- only append before `_other`, and even that
#: invalidates every vector captured earlier.
#: Sourced from the WAI-ARIA 1.2 role list, restricted to roles a
#: document-shaped page can actually carry, plus `text`, which is not a
#: WAI-ARIA role but is emitted by playwright's `aria_snapshot()` for text
#: nodes and would otherwise dump a large, page-length-dependent count into
#: `_other`.
ROLE_VOCAB = (
    "alert", "alertdialog", "application", "article", "banner", "blockquote",
    "button", "caption", "cell", "checkbox", "code", "columnheader",
    "combobox", "complementary", "contentinfo", "definition", "deletion",
    "dialog", "document", "emphasis", "feed", "figure", "form", "generic",
    "grid", "gridcell", "group", "heading", "img", "insertion", "link",
    "list", "listbox", "listitem", "log", "main", "mark", "marquee", "math",
    "menu", "menubar", "menuitem", "menuitemcheckbox", "menuitemradio",
    "meter", "navigation", "none", "note", "option", "paragraph",
    "presentation", "progressbar", "radio", "radiogroup", "region", "row",
    "rowgroup", "rowheader", "scrollbar", "search", "searchbox", "separator",
    "slider", "spinbutton", "status", "strong", "subscript", "superscript",
    "switch", "tab", "table", "tablist", "tabpanel", "term", "text", "textbox",
    "time", "timer", "toolbar", "tooltip", "tree", "treegrid", "treeitem",
    "_other",
)
_ROLE_INDEX = dict((r, i) for i, r in enumerate(ROLE_VOCAB))

#: DOM -> implicit-ARIA-role map for the fallback path, evaluated in the page.
#: Kept deliberately small and explicit: a full HTML-AAM implementation would
#: be another moving part, and what we need is a *stable* signature, not a
#: spec-perfect one.
_ROLE_JS = r"""
() => {
  const inputRole = (el) => {
    const t = (el.getAttribute('type') || 'text').toLowerCase();
    const m = {checkbox:'checkbox', radio:'radio', button:'button',
               submit:'button', reset:'button', image:'button',
               search:'searchbox', range:'slider', number:'spinbutton',
               email:'textbox', tel:'textbox', url:'textbox', text:'textbox',
               password:'none', hidden:'none', file:'none', color:'none',
               date:'none', 'datetime-local':'none', month:'none',
               time:'none', week:'none'};
    return m[t] || 'textbox';
  };
  const tagRole = (el) => {
    const tag = el.tagName.toLowerCase();
    switch (tag) {
      case 'a': return el.hasAttribute('href') ? 'link' : 'generic';
      case 'area': return el.hasAttribute('href') ? 'link' : 'generic';
      case 'button': return 'button';
      case 'input': return inputRole(el);
      case 'select':
        return (el.multiple || (el.size && el.size > 1)) ? 'listbox' : 'combobox';
      case 'textarea': return 'textbox';
      case 'option': return 'option';
      case 'optgroup': return 'group';
      case 'fieldset': return 'group';
      case 'form': return 'form';
      case 'search': return 'search';
      case 'img': return el.getAttribute('alt') === '' ? 'presentation' : 'img';
      case 'svg': return 'img';
      case 'figure': return 'figure';
      case 'table': return 'table';
      case 'tr': return 'row';
      case 'td': return 'cell';
      case 'th':
        return el.getAttribute('scope') === 'row' ? 'rowheader' : 'columnheader';
      case 'thead': case 'tbody': case 'tfoot': return 'rowgroup';
      case 'caption': return 'caption';
      case 'ul': case 'ol': case 'menu': return 'list';
      case 'li': return 'listitem';
      case 'dl': return 'list';
      case 'dd': return 'definition';
      case 'dt': return 'term';
      case 'h1': case 'h2': case 'h3': case 'h4': case 'h5': case 'h6':
        return 'heading';
      case 'nav': return 'navigation';
      case 'main': return 'main';
      case 'header': return el.closest('article,aside,main,nav,section')
                            ? 'generic' : 'banner';
      case 'footer': return el.closest('article,aside,main,nav,section')
                            ? 'generic' : 'contentinfo';
      case 'aside': return 'complementary';
      case 'article': return 'article';
      case 'section':
        return (el.hasAttribute('aria-label') || el.hasAttribute('aria-labelledby'))
               ? 'region' : 'generic';
      case 'p': return 'paragraph';
      case 'blockquote': return 'blockquote';
      case 'hr': return 'separator';
      case 'dialog': return 'dialog';
      case 'progress': return 'progressbar';
      case 'meter': return 'meter';
      case 'output': return 'status';
      case 'code': return 'code';
      case 'em': return 'emphasis';
      case 'strong': return 'strong';
      case 'time': return 'time';
      case 'sub': return 'subscript';
      case 'sup': return 'superscript';
      case 'del': return 'deletion';
      case 'ins': return 'insertion';
      case 'mark': return 'mark';
      case 'div': case 'span': return 'generic';
      default: return null;
    }
  };
  const counts = {};
  const bump = (r) => { if (r) counts[r] = (counts[r] || 0) + 1; };
  const els = document.querySelectorAll('body *');
  for (const el of els) {
    const explicit = (el.getAttribute('role') || '').trim().toLowerCase();
    // An explicit role attribute may list fallbacks; the first token wins.
    bump(explicit ? explicit.split(/\s+/)[0] : tagRole(el));
  }
  // Sorting makes the JSON byte-identical across runs even though object key
  // order is only insertion-ordered.
  const out = {};
  for (const k of Object.keys(counts).sort()) out[k] = counts[k];
  return out;
}
"""


def role_vector(counts):
    """Fixed-order vector over ROLE_VOCAB. Unknown roles fold into `_other`."""
    vec = [0] * len(ROLE_VOCAB)
    for role, n in counts.items():
        vec[_ROLE_INDEX.get(role, _ROLE_INDEX["_other"])] += int(n)
    return vec


def _roles_from_aria_snapshot(text):
    """Parse `page.aria_snapshot()`'s YAML-ish tree into role counts.

    Lines look like `  - listitem:` or `- link "Main Page"` or
    `- heading "Biology" [level=1]`, so the role is the first bare token after
    the dash.
    """
    import re
    counts = collections.Counter()
    for line in text.splitlines():
        m = re.match(r"\s*-\s+([a-z]+)\b", line)
        if m:
            counts[m.group(1)] += 1
    return dict(counts)


def page_roles(page, cfg=None):
    """Role counts + fixed-order vector for a loaded page.

    Always computes the DOM-derived counts (cheap, fully deterministic, and
    independent of Chromium's accessibility-tree computation); additionally
    tries the real a11y tree and uses it as the primary source when available.
    """
    cfg = cfg or CaptureConfig()
    dom_counts = page.evaluate(_ROLE_JS)
    aria_counts, aria_text, source = None, None, "dom"

    if cfg.role_source in ("auto", "aria"):
        acc = getattr(page, "accessibility", None)
        if acc is not None:                       # playwright < 1.6x
            try:
                snap = acc.snapshot()
                counts = collections.Counter()

                def _walk(node):
                    if not node:
                        return
                    r = node.get("role")
                    if r:
                        counts[r] += 1
                    for ch in node.get("children") or []:
                        _walk(ch)

                _walk(snap)
                aria_counts, source = dict(counts), "accessibility.snapshot"
            except Exception:                     # noqa: BLE001
                aria_counts = None
        if aria_counts is None and hasattr(page, "aria_snapshot"):
            try:
                aria_text = page.aria_snapshot()
                aria_counts, source = (_roles_from_aria_snapshot(aria_text),
                                       "aria_snapshot")
            except Exception:                     # noqa: BLE001
                aria_counts, aria_text = None, None
    if cfg.role_source == "dom":
        aria_counts, source = None, "dom"

    primary = aria_counts if aria_counts is not None else dom_counts
    if aria_counts is None:
        source = "dom"
    return {
        "source": source,
        "vocab_size": len(ROLE_VOCAB),
        "counts": collections.OrderedDict(sorted(primary.items())),
        "vector": role_vector(primary),
        "counts_dom": collections.OrderedDict(sorted(dom_counts.items())),
        "vector_dom": role_vector(dom_counts),
        "aria_snapshot_text": aria_text,
    }


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------

def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright
        return sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "playwright is not importable from %s.\n"
            "  It is installed in tw_r1_q3 ONLY -- run this module as\n"
            "    cd %s && %s -m adaptercl.capture ...\n"
            "  (tw_web has flask but no playwright, despite what you may have "
            "been told.)\n  underlying error: %s"
            % (sys.executable, paths.PROJECT, paths.PY_BENCH, e))


def _new_context(pw, cfg):
    browser = pw.chromium.launch(headless=cfg.headless, args=cfg.browser_args)
    ctx = browser.new_context(
        viewport={"width": cfg.viewport_width, "height": cfg.viewport_height},
        device_scale_factor=cfg.device_scale_factor,
        locale=cfg.locale,
        timezone_id=cfg.timezone_id,
        color_scheme=cfg.color_scheme,
        reduced_motion=cfg.reduced_motion,
        forced_colors=cfg.forced_colors,
    )
    ctx.set_default_navigation_timeout(cfg.nav_timeout_ms)
    if cfg.disable_animations:
        # add_init_script runs before any page script, so the freeze is in
        # place before the first frame rather than after it.
        ctx.add_init_script(
            "document.addEventListener('DOMContentLoaded',()=>{"
            "const s=document.createElement('style');"
            "s.textContent=%s;document.head.appendChild(s);});"
            % json.dumps(FREEZE_CSS))
    return browser, ctx


def capture_cell(cell, url, out_dir, cfg=None, page=None, label="landing"):
    """Capture one page of one cell.

    Writes `screenshot.png`, `page.html`, `axtree.json`, `roles.json` and
    `meta.json` into `<out_dir>/<cell.key>/<label>/`. Returns the meta dict.

    Pass `page` to reuse an open browser (much faster for `check_determinism`
    and `capture_all`); otherwise a throwaway Chromium is launched.
    """
    cfg = cfg or CaptureConfig()
    dest = os.path.join(out_dir, cell.key, label)
    if not os.path.isdir(dest):
        os.makedirs(dest)

    if page is not None:
        return _capture_on_page(page, cell, url, dest, cfg, label)
    sync_playwright = _require_playwright()
    with sync_playwright() as pw:
        browser, ctx = _new_context(pw, cfg)
        try:
            return _capture_on_page(ctx.new_page(), cell, url, dest, cfg, label)
        finally:
            ctx.close()
            browser.close()


def _capture_on_page(page, cell, url, dest, cfg, label):
    t0 = time.time()
    page.goto(url, wait_until=cfg.wait_until, timeout=cfg.nav_timeout_ms)
    if cfg.disable_animations:
        try:
            page.add_style_tag(content=FREEZE_CSS)
        except Exception:                          # noqa: BLE001 - about:blank
            pass
    if cfg.scroll_to_top:
        page.evaluate("window.scrollTo(0, 0)")
    # Fonts must be resolved before the screenshot or the first capture of a
    # run renders in the fallback face and every later one does not -- the
    # single most common cause of "capture 1 differs, 2..10 agree".
    try:
        page.evaluate("() => document.fonts && document.fonts.ready")
    except Exception:                              # noqa: BLE001
        pass
    page.wait_for_timeout(cfg.settle_ms)

    png = os.path.join(dest, "screenshot.png")
    page.screenshot(path=png, full_page=cfg.full_page, animations="disabled",
                    caret="hide", scale=cfg.screenshot_scale)
    html = page.content()
    fh = open(os.path.join(dest, "page.html"), "w")
    try:
        fh.write(html)
    finally:
        fh.close()

    roles = page_roles(page, cfg)
    _write_json(os.path.join(dest, "roles.json"), {
        "cell": cell.as_dict(), "label": label, "url": url,
        "role_vocab": list(ROLE_VOCAB),
        "source": roles["source"],
        "counts": roles["counts"], "vector": roles["vector"],
        "counts_dom": roles["counts_dom"], "vector_dom": roles["vector_dom"],
    })
    _write_json(os.path.join(dest, "axtree.json"), {
        "cell": cell.as_dict(), "label": label, "url": url,
        "source": roles["source"],
        "aria_snapshot": roles["aria_snapshot_text"],
        "role_counts": roles["counts"],
        "note": ("playwright 1.61 removed page.accessibility; aria_snapshot is "
                 "a YAML rendering of the same tree. `role_counts` is the "
                 "machine-readable part."),
    })

    png_bytes = open(png, "rb").read()
    meta = collections.OrderedDict([
        ("cell", cell.as_dict()),
        ("label", label),
        ("url", url),
        ("dir", dest),
        ("elapsed_s", round(time.time() - t0, 3)),
        ("png_sha256", hashlib.sha256(png_bytes).hexdigest()),
        ("png_bytes", len(png_bytes)),
        ("html_sha256", hashlib.sha256(html.encode("utf-8")).hexdigest()),
        ("html_chars", len(html)),
        ("role_source", roles["source"]),
        ("role_vector_sha256",
         hashlib.sha256(json.dumps(roles["vector"]).encode()).hexdigest()),
        ("n_roles", sum(roles["vector"])),
        ("config_fingerprint", cfg.fingerprint()),
        ("config", cfg.as_dict()),
    ])
    _write_json(os.path.join(dest, "meta.json"), meta)
    return meta


def _write_json(path, obj):
    fh = open(path, "w")
    try:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
    finally:
        fh.close()


# --------------------------------------------------------------------------
# Environment servers
#
# Structure follows mutateWeb/mutateloop/serve.py::EnvServer (start_new_session
# + killpg teardown, generous startup timeout because the apps load large
# content indexes). Reused rather than reinvented; extended with the three
# things that file does not need: all three sites at once, the TW_* exports
# browsergym requires (instance.py:21), and the unsafe-port list.
# --------------------------------------------------------------------------

#: Chromium refuses these with ERR_UNSAFE_PORT regardless of what is listening.
#: Copied verbatim from run_v6_eval_q35_sonnet.sh:98-105.
UNSAFE_PORTS = frozenset(
    (5060, 5061, 6000, 6566, 6665, 6666, 6667, 6668, 6669, 6697, 10080))

DEFAULT_PORT_BASE = 5300      # clear of the apps' own 5000-5101 default band

#: env -> how to launch it. Verified against run_v6_eval_q35_sonnet.sh:136-146
#: and TimeWarp/run_all_env.sh. Note the version flag differs: wiki/news take
#: `-<V>`, webshop takes a bare `<V>`.
LAUNCH = {
    "wiki": {
        "python": paths.PY_WEB,
        "cwd": os.path.join(paths.TIMEWARP_ENV, "env", "wiki"),
        "argv": lambda era, port: ["wiki_app.py", "-%d" % era,
                                   "--port=%d" % port],
        "suffix": "",
    },
    "news": {
        "python": paths.PY_WEB,
        "cwd": os.path.join(paths.TIMEWARP_ENV, "env", "news"),
        "argv": lambda era, port: ["news_app.py", "-%d" % era,
                                   "--port=%d" % port],
        "suffix": "",
    },
    "shop": {
        "python": paths.PY_SHOP,
        "cwd": os.path.join(paths.TIMEWARP_ENV, "env", "webshop"),
        "argv": lambda era, port: ["-m", "web_agent_site.app", "%d" % era,
                                   "--port=%d" % port, "--log", "--attrs"],
        # The webshop app serves per-session URLs; browsergym uses a fixed
        # session id, so the served root is /<session> and not /.
        "suffix": "/abc",
        # The shop needs its bundled JVM (pyserini search index).
        "env": {
            "JAVA_HOME": os.path.join(paths.ENVS, "tw_webshop", "lib", "jvm"),
            "PATH": os.path.join(paths.ENVS, "tw_webshop", "bin")
                    + os.pathsep + os.environ.get("PATH", ""),
        },
    },
}

#: browsergym's TimeWarpInstance requires this even though nothing serves it
#: (instance.py:21 lists it among `required_vars`).
TW_HOME_PLACEHOLDER = "http://localhost:5100"


def free_port(start, used=None):
    """First bindable port at or above `start` that Chromium will also accept."""
    used = used or set()
    port = int(start)
    while port < 65535:
        if port in UNSAFE_PORTS or port in used:
            port += 1
            continue
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            return port
        except socket.error:
            port += 1
        finally:
            s.close()
    raise RuntimeError("no free port at or above %s" % (start,))


def _http_ok(url, timeout=5):
    """True if the URL answers with anything below 500. `urllib` rather than
    `requests` so this works in any of the project's interpreters."""
    try:
        from urllib.request import urlopen
        from urllib.error import HTTPError, URLError
    except ImportError:                            # pragma: no cover - py2
        return False
    try:
        r = urlopen(url, timeout=timeout)
        r.read(1)
        r.close()
        return True
    except HTTPError as e:
        return e.code < 500
    except (URLError, socket.error, OSError):
        return False


class EnvServers(object):
    """Start wiki + news + webshop for one era; export TW_*; tear down cleanly.

    Ports are allocated from `port_base` upward, skipping UNSAFE_PORTS. Give
    concurrent runs disjoint bases (5300, 5400, ...) -- "is anyone LISTENING"
    is not enough, two runs starting in the same second both see the same port
    free and one of them then dies in the health wait.
    """

    def __init__(self, era, port_base=DEFAULT_PORT_BASE, envs=None,
                 log_dir=None, startup_timeout=300.0, export=True):
        self.era = int(era)
        self.port_base = int(port_base)
        self.envs = tuple(envs or cells.ENVIRONMENTS)
        self.log_dir = log_dir or paths.OUT_LOGS
        self.startup_timeout = startup_timeout
        self.export = export
        self.procs = collections.OrderedDict()
        self.ports = collections.OrderedDict()
        self.urls = collections.OrderedDict()
        self._saved_env = {}

    # -- lifecycle --------------------------------------------------------
    def start(self):
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        used = set()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        try:
            for env in self.envs:
                spec = LAUNCH[env]
                if not os.path.exists(spec["python"]):
                    raise RuntimeError(
                        "missing interpreter for %s: %s" % (env, spec["python"]))
                port = free_port(self.port_base + len(used), used)
                used.add(port)
                self.ports[env] = port
                self.urls[env] = "http://localhost:%d%s" % (port, spec["suffix"])
                argv = [spec["python"]] + spec["argv"](self.era, port)
                child_env = dict(os.environ)
                child_env.update(spec.get("env") or {})
                log = os.path.join(
                    self.log_dir,
                    "env_%s_v%d_%d_%s.log" % (env, self.era, port, stamp))
                fh = open(log, "wb")
                proc = subprocess.Popen(
                    argv, cwd=spec["cwd"], env=child_env, stdout=fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True)     # own pgid -> killpg teardown
                self.procs[env] = {"proc": proc, "log": log, "fh": fh,
                                   "argv": argv}
            self._wait_healthy()
        except Exception:
            self.stop()
            raise
        if self.export:
            self._export()
        return self

    def _wait_healthy(self):
        deadline = time.time() + self.startup_timeout
        pending = list(self.envs)
        while pending and time.time() < deadline:
            for env in list(pending):
                proc = self.procs[env]["proc"]
                if proc.poll() is not None:
                    raise RuntimeError(
                        "%s app for era %d exited early (code %s); log: %s"
                        % (env, self.era, proc.returncode,
                           self.procs[env]["log"]))
                if _http_ok(self.urls[env]):
                    pending.remove(env)
            if pending:
                time.sleep(1.0)
        if pending:
            raise RuntimeError(
                "env(s) %s did not serve within %.0fs for era %d; logs: %s"
                % (pending, self.startup_timeout, self.era,
                   [self.procs[e]["log"] for e in pending]))

    def _export(self):
        for env in self.envs:
            var = cells.TW_URL_VAR[env]
            self._saved_env[var] = os.environ.get(var)
            os.environ[var] = self.urls[env]
        self._saved_env["TW_HOME"] = os.environ.get("TW_HOME")
        os.environ.setdefault("TW_HOME", TW_HOME_PLACEHOLDER)

    def stop(self):
        for env, rec in list(self.procs.items()):
            proc = rec["proc"]
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=15)
                except Exception:                  # noqa: BLE001
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:              # noqa: BLE001
                        pass
            try:
                rec["fh"].close()
            except Exception:                      # noqa: BLE001
                pass
        self.procs = collections.OrderedDict()
        for var, old in self._saved_env.items():
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old
        self._saved_env = {}

    def url(self, env):
        return self.urls[env]

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


# --------------------------------------------------------------------------
# Determinism check (7 Phase 0)
# --------------------------------------------------------------------------

def _max_pixel_diff(paths_):
    """Max absolute per-channel difference across a list of PNGs, via PIL."""
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return None, "PIL not installed in this interpreter"
    ref = Image.open(paths_[0]).convert("RGB")
    worst, worst_pair = 0, None
    for p in paths_[1:]:
        im = Image.open(p).convert("RGB")
        if im.size != ref.size:
            return None, "size mismatch: %s vs %s" % (ref.size, im.size)
        bbox = ImageChops.difference(ref, im).getextrema()
        m = max(hi for _lo, hi in bbox)
        if m > worst:
            worst, worst_pair = m, p
    return {"max_channel_diff": worst, "worst_file": worst_pair}, None


def check_determinism(cell, url, n=10, cfg=None, out_dir=None, reuse_page=True):
    """Capture the same page `n` times and report whether anything moved.

    7 Phase 0: "Capture protocol verified deterministic: same page -> identical
    conditioning embedding across 10 repeats." We check three levels, because
    they fail independently: PNG bytes (strictest -- also catches encoder
    nondeterminism), pixels (what the vision tower actually sees), and the role
    vector (4.2's structural side-channel).

    `reuse_page=True` reuses one browser and page across repeats, which is what
    a real capture run does. Pass False to also exercise cold-start variation.
    """
    cfg = cfg or CaptureConfig()
    out_dir = out_dir or os.path.join(paths.OUT_CAPTURE, "determinism")
    sync_playwright = _require_playwright()

    metas = []
    with sync_playwright() as pw:
        browser, ctx = _new_context(pw, cfg)
        page = ctx.new_page() if reuse_page else None
        try:
            for i in range(n):
                label = "rep_%02d" % i
                if reuse_page:
                    dest = os.path.join(out_dir, cell.key, label)
                    if not os.path.isdir(dest):
                        os.makedirs(dest)
                    metas.append(_capture_on_page(page, cell, url, dest, cfg,
                                                  label))
                else:
                    p = ctx.new_page()
                    dest = os.path.join(out_dir, cell.key, label)
                    if not os.path.isdir(dest):
                        os.makedirs(dest)
                    metas.append(_capture_on_page(p, cell, url, dest, cfg,
                                                  label))
                    p.close()
        finally:
            ctx.close()
            browser.close()

    png_hashes = [m["png_sha256"] for m in metas]
    role_hashes = [m["role_vector_sha256"] for m in metas]
    html_hashes = [m["html_sha256"] for m in metas]
    png_paths = [os.path.join(m["dir"], "screenshot.png") for m in metas]
    pix, pix_err = _max_pixel_diff(png_paths)

    png_same = len(set(png_hashes)) == 1
    roles_same = len(set(role_hashes)) == 1
    result = collections.OrderedDict([
        ("cell", cell.as_dict()),
        ("url", url),
        ("n", n),
        ("reuse_page", reuse_page),
        ("out_dir", os.path.join(out_dir, cell.key)),
        ("png_identical", png_same),
        ("png_distinct_hashes", sorted(set(png_hashes))),
        ("html_identical", len(set(html_hashes)) == 1),
        ("role_vector_identical", roles_same),
        ("role_source", metas[0]["role_source"]),
        ("n_roles", metas[0]["n_roles"]),
        ("pixel_diff", pix),
        ("pixel_diff_error", pix_err),
        ("config_fingerprint", cfg.fingerprint()),
        # Phase 0 asks for an identical conditioning *embedding*. The
        # embedding is a deterministic function of the pixels and the role
        # vector, so identical pixels + identical role vector is the
        # sufficient condition; identical PNG bytes is strictly stronger and
        # not required.
        ("deterministic", bool(
            roles_same and (png_same or (pix is not None
                                         and pix["max_channel_diff"] == 0)))),
    ])
    return result


# --------------------------------------------------------------------------
# Batch capture
# --------------------------------------------------------------------------

def capture_all(eras=None, out_root=None, cfg=None, port_base=DEFAULT_PORT_BASE,
                envs=None, extra_pages=None, keep_servers_log=True):
    """Capture the landing page of every site for every era.

    `extra_pages` is `{env: [relative_path, ...]}` for additional pages; it is
    empty by default on purpose. Every TimeWarp task starts at the site root
    (all 231 records use a `__WIKI__`/`__NEWS__`/`__WEBSHOP__` placeholder), so
    the landing page *is* the per-episode conditioning observation and anything
    beyond it is a separate, deliberate choice rather than a default.
    """
    eras = tuple(eras or cells.ERAS)
    envs = tuple(envs or cells.ENVIRONMENTS)
    out_root = out_root or paths.OUT_CAPTURE
    cfg = cfg or CaptureConfig()
    extra_pages = extra_pages or {}
    paths.ensure_out_dirs()

    sync_playwright = _require_playwright()
    manifest = collections.OrderedDict()
    errors = []
    with sync_playwright() as pw:
        browser, ctx = _new_context(pw, cfg)
        page = ctx.new_page()
        try:
            for era in eras:
                servers = EnvServers(era, port_base=port_base, envs=envs)
                with servers:
                    for env in envs:
                        cell = cells.Cell(env, era)
                        base = servers.url(env)
                        todo = [("landing", base)]
                        for rel in extra_pages.get(env, []):
                            todo.append((rel.strip("/").replace("/", "_")
                                         or "root",
                                         base.rstrip("/") + "/" + rel.lstrip("/")))
                        for label, url in todo:
                            try:
                                meta = capture_cell(cell, url, out_root,
                                                    cfg=cfg, page=page,
                                                    label=label)
                                manifest.setdefault(cell.key, {})[label] = meta
                            except Exception as e:     # noqa: BLE001
                                errors.append({"cell": cell.key,
                                               "label": label, "url": url,
                                               "error": "%s: %s"
                                                        % (type(e).__name__, e)})
        finally:
            ctx.close()
            browser.close()

    out = collections.OrderedDict([
        ("out_root", out_root),
        ("eras", list(eras)),
        ("envs", list(envs)),
        ("port_base", port_base),
        ("config_fingerprint", cfg.fingerprint()),
        ("config", cfg.as_dict()),
        ("role_vocab", list(ROLE_VOCAB)),
        ("n_captured", sum(len(v) for v in manifest.values())),
        ("errors", errors),
        ("cells", manifest),
    ])
    _write_json(os.path.join(out_root, "manifest.json"), out)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_eras(s):
    return tuple(int(x) for x in str(s).split(",") if x.strip())


def _cmd_capture(args):
    cfg = CaptureConfig()
    cell = cells.Cell(args.env, args.era)
    if args.url:
        meta = capture_cell(cell, args.url, args.out, cfg=cfg)
    else:
        with EnvServers(args.era, port_base=args.port_base,
                        envs=(args.env,)) as srv:
            meta = capture_cell(cell, srv.url(args.env), args.out, cfg=cfg)
    print("captured %s -> %s" % (cell.key, meta["dir"]))
    print("  png    %s (%d bytes)" % (meta["png_sha256"][:16], meta["png_bytes"]))
    print("  roles  %d elements from %s" % (meta["n_roles"], meta["role_source"]))
    print("  config %s" % meta["config_fingerprint"])
    return 0


def _cmd_determinism(args):
    cfg = CaptureConfig()
    results, bad = collections.OrderedDict(), 0
    envs = (args.env,) if args.env else cells.ENVIRONMENTS
    with EnvServers(args.era, port_base=args.port_base, envs=envs) as srv:
        for env in envs:
            cell = cells.Cell(env, args.era)
            r = check_determinism(cell, srv.url(env), n=args.n, cfg=cfg,
                                  out_dir=os.path.join(args.out, "determinism"))
            results[cell.key] = r
            bad += 0 if r["deterministic"] else 1

    print("")
    print("=" * 72)
    print("Phase 0 determinism check: %d repeats per cell, era %d"
          % (args.n, args.era))
    print("=" * 72)
    print("  %-10s %-6s %-6s %-7s %-9s %s"
          % ("cell", "png", "html", "roles", "maxpix", "verdict"))
    for k, r in results.items():
        pix = ("-" if r["pixel_diff"] is None
               else str(r["pixel_diff"]["max_channel_diff"]))
        print("  %-10s %-6s %-6s %-7s %-9s %s"
              % (k, "same" if r["png_identical"] else "DIFF",
                 "same" if r["html_identical"] else "DIFF",
                 "same" if r["role_vector_identical"] else "DIFF",
                 pix, "OK" if r["deterministic"] else "NONDETERMINISTIC"))
    if results and results[list(results)[0]]["pixel_diff_error"]:
        print("  (pixel diff unavailable: %s)"
              % results[list(results)[0]]["pixel_diff_error"])
    print("")
    print("  role source: %s   config: %s"
          % (results[list(results)[0]]["role_source"] if results else "-",
             cfg.fingerprint()))

    path = os.path.join(args.out, "determinism.json")
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    _write_json(path, results)
    print("  wrote %s" % path)
    return 0 if bad == 0 else 2


def _cmd_all(args):
    extra = {}
    for spec in args.page or []:
        env, _, rel = spec.partition("=")
        extra.setdefault(env, []).append(rel)
    out = capture_all(eras=_parse_eras(args.eras), out_root=args.out,
                      port_base=args.port_base, extra_pages=extra)
    print("captured %d page(s) across %d cell(s) into %s"
          % (out["n_captured"], len(out["cells"]), out["out_root"]))
    for key in sorted(out["cells"]):
        labels = out["cells"][key]
        print("  %-10s %s" % (key, ", ".join(
            "%s(%s)" % (l, m["png_sha256"][:8]) for l, m in sorted(labels.items()))))
    for e in out["errors"]:
        print("  ERROR %s/%s: %s" % (e["cell"], e["label"], e["error"]))
    print("  manifest: %s" % os.path.join(out["out_root"], "manifest.json"))
    return 0 if not out["errors"] else 2


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="deterministic conditioning capture for adapterCL (4.2)")
    ap.add_argument("--out", default=paths.OUT_CAPTURE)
    ap.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE,
                    help="give concurrent runs disjoint bases (5300, 5400, ...)")
    sub = ap.add_subparsers(dest="cmd")

    c = sub.add_parser("capture", help="capture one cell")
    c.add_argument("--env", required=True, choices=list(cells.ENVIRONMENTS))
    c.add_argument("--era", required=True, type=int, choices=list(cells.ERAS))
    c.add_argument("--url", default=None,
                   help="skip starting servers and capture this URL")
    c.set_defaults(func=_cmd_capture)

    d = sub.add_parser("determinism", help="Phase 0: n repeats of one page")
    d.add_argument("--era", required=True, type=int, choices=list(cells.ERAS))
    d.add_argument("--env", default=None, choices=list(cells.ENVIRONMENTS),
                   help="default: all three")
    d.add_argument("-n", type=int, default=10)
    d.set_defaults(func=_cmd_determinism)

    a = sub.add_parser("all", help="capture every cell of every era")
    a.add_argument("--eras", default=",".join(str(e) for e in cells.ERAS))
    a.add_argument("--page", action="append", default=None,
                   metavar="ENV=REL/PATH",
                   help="extra page to capture, repeatable "
                        "(e.g. --page wiki=wiki/Biology)")
    a.set_defaults(func=_cmd_all)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 1
    paths.ensure_out_dirs()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
