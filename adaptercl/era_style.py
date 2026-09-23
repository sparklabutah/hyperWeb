"""Does era N look the same on wiki, news and shop? (adapterCL.md 6.2 precondition)

6.2 is the paper's centerpiece and it has one precondition: "Wiki-era-3,
News-era-3, and Shop-era-3 must share a visual style ... If eras are
stylistically unrelated across environments, 6.2 doesn't run and the paper
loses its centerpiece." 7 Phase 0 calls the check blocking. This module answers
it with numbers instead of eyeballs, and answers it *statically* -- straight off
the theme directories on disk, no flask servers, no browser, no GPU.

What it does
------------
1. Reads each of the 18 (env, era) theme directories and builds a **style
   signature**: colour statistics, typography, CSS layout mechanisms, an HTML
   tag histogram and HTML-4-vs-HTML-5 structural markers.
2. Turns each signature into a fixed-length numeric vector (FEATURE_NAMES gives
   the exact ordering; every dimension is bounded in [0, 1] before block-wise
   L2 normalisation -- see NORMALISATION below).
3. Reports the 18x18 cosine matrix and the two contrasts 6.2 asks for:
   mean **within-era across-environment** vs mean **within-environment
   across-era** similarity.
4. Reports a per-era ranking of how well the three sites agree, the nominal-year
   disagreement from `cells.ERA_YEAR_SPREAD`, and a `verdict()` of
   corresponds / partial / does_not_correspond.

Why three similarity variants, not one
--------------------------------------
Raw cosine over non-negative style features has a high floor -- every web page
is mostly `<div>`s and hex colours -- and it is dominated by *environment*
identity, because the shop templates are 3-6x larger and have a completely
different page inventory (9 templates vs wiki's 4; see the file census in
`_theme_files`). So the raw within-env contrast is close to guaranteed to win
and tells you almost nothing. We therefore also report:

* ``grand``  -- features mean-centred over all 18 cells (removes "all web pages
  look alike").
* ``env``    -- features mean-centred **within each environment** over the five
  temporal eras. This asks the actually interesting question: *once you know
  which site you are looking at, does the residual style still line up by era?*
  Under this centring the mean cross-environment dot product is exactly zero by
  construction, so 0 is the honest null and a permutation test over era labels
  gives a real p-value. This is the variant `verdict()` uses.

Era 6 is a style-neutral baseline in all three sites (cells.py:39-42:
wiki `6-minimal`, news `6-base-minimal`, shop `classic`), so it carries no era
semantics and is excluded from every temporal statistic. It is still shown in
the 18x18 figure.

Pure stdlib -- runs on the system python (3.6.8). PIL is optional and only used
by ``--screenshots``.

Run:  cd adapter_project && python3 -m adaptercl.era_style
"""

from __future__ import print_function

import argparse
import collections
import json
import math
import os
import random
import re
import sys

from . import cells, paths

# --------------------------------------------------------------------------
# Theme file discovery
#
# Layout differs between the two web apps and the shop:
#   wiki/news : <theme>/style.css              + <theme>/*.html
#   shop      : <theme>/static/style.css       + <theme>/templates/*.html
# `style.css.backup` exists under webshop2000/static and must not be read
# (it is a dead copy and would double-count that theme's declarations).
# --------------------------------------------------------------------------

_SHOP_ENV = "shop"


def theme_dir(cell):
    """Absolute path of the theme directory backing one cell."""
    return os.path.join(cells.THEME_ROOT[cell.env], cell.theme)


def _theme_files(cell):
    """(css_paths, html_paths) for a cell. Raises if the theme dir is missing."""
    root = theme_dir(cell)
    if not os.path.isdir(root):
        raise IOError(
            "theme dir for %s does not exist: %s\n"
            "  (expected THEME_NAME[%r][%d] == %r under %s)"
            % (cell.key, root, cell.env, cell.era, cell.theme,
               cells.THEME_ROOT[cell.env]))
    if cell.env == _SHOP_ENV:
        css_dir = os.path.join(root, "static")
        html_dir = os.path.join(root, "templates")
    else:
        css_dir = html_dir = root
    css = [os.path.join(css_dir, f) for f in sorted(os.listdir(css_dir))
           if f.endswith(".css")] if os.path.isdir(css_dir) else []
    html = [os.path.join(html_dir, f) for f in sorted(os.listdir(html_dir))
            if f.endswith(".html")] if os.path.isdir(html_dir) else []
    if not css:
        raise IOError("no style.css under %s" % (css_dir,))
    if not html:
        raise IOError("no templates under %s" % (html_dir,))
    return css, html


def _read(path):
    fh = open(path, "rb")
    try:
        raw = fh.read()
    finally:
        fh.close()
    return raw.decode("utf-8", "replace")


# --------------------------------------------------------------------------
# CSS parsing
# --------------------------------------------------------------------------

_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)
#: Innermost brace blocks only. `[^{}]*` cannot span a nested block, so for
#: `@media (...) { .a { color: red } }` this matches `.a`'s body and never the
#: @media wrapper -- which is exactly what we want, since the wrapper has no
#: declarations of its own.
_CSS_BLOCK = re.compile(r"\{([^{}]*)\}", re.S)
_CSS_DECL = re.compile(r"^\s*([-a-zA-Z_][-a-zA-Z0-9_]*)\s*:\s*(.+?)\s*$", re.S)

_HEX = re.compile(r"#([0-9a-fA-F]{3,8})\b")
_RGB = re.compile(r"\brgba?\(([^)]*)\)")
_HSL = re.compile(r"\bhsla?\(([^)]*)\)")

#: The HTML-4 named colours plus the handful of CSS-2 names that actually turn
#: up in these themes. Matched only inside declaration *values*, so a class
#: called `.red-box` cannot inflate the count.
_NAMED_COLORS = {
    "black": (0, 0, 0), "silver": (192, 192, 192), "gray": (128, 128, 128),
    "grey": (128, 128, 128), "white": (255, 255, 255), "maroon": (128, 0, 0),
    "red": (255, 0, 0), "purple": (128, 0, 128), "fuchsia": (255, 0, 255),
    "magenta": (255, 0, 255), "green": (0, 128, 0), "lime": (0, 255, 0),
    "olive": (128, 128, 0), "yellow": (255, 255, 0), "navy": (0, 0, 128),
    "blue": (0, 0, 255), "teal": (0, 128, 128), "aqua": (0, 255, 255),
    "cyan": (0, 255, 255), "orange": (255, 165, 0),
    "darkblue": (0, 0, 139), "lightgray": (211, 211, 211),
    "lightgrey": (211, 211, 211), "darkgray": (169, 169, 169),
    "whitesmoke": (245, 245, 245), "gainsboro": (220, 220, 220),
    "beige": (245, 245, 220), "ivory": (255, 255, 240),
    "crimson": (220, 20, 60), "gold": (255, 215, 0),
}
_NAMED_RE = re.compile(r"\b(%s)\b" % "|".join(sorted(_NAMED_COLORS)), re.I)

_COLOR_PROPS = ("color", "background", "background-color", "border",
                "border-color", "border-top", "border-bottom", "border-left",
                "border-right", "outline", "fill", "stroke", "box-shadow",
                "text-shadow")

_SERIF = ("times", "georgia", "garamond", "palatino", "book antiqua",
          "serif", "cambria", "constantia", "didot", "bodoni", "merriweather")
_SANS = ("arial", "helvetica", "verdana", "tahoma", "geneva", "trebuchet",
         "segoe", "calibri", "lato", "roboto", "open sans", "inter",
         "sans-serif", "futura", "gill sans", "franklin")
_MONO = ("courier", "monaco", "consolas", "menlo", "monospace", "lucida console")
_SYSTEM_STACK = ("system-ui", "-apple-system", "blinkmacsystemfont",
                 "segoe ui", "ui-sans-serif", "ui-serif")


def _hex_to_rgb(h):
    if len(h) == 3:
        return tuple(int(c * 2, 16) for c in h)
    if len(h) in (6, 8):
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    return None


def _rgb_to_hsl(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    lig = (mx + mn) / 2.0
    if mx == mn:
        return 0.0, 0.0, lig
    d = mx - mn
    sat = d / (2.0 - mx - mn) if lig > 0.5 else d / (mx + mn)
    if mx == r:
        hue = ((g - b) / d) % 6.0
    elif mx == g:
        hue = (b - r) / d + 2.0
    else:
        hue = (r - g) / d + 4.0
    return hue / 6.0, sat, lig


def parse_css(text):
    """Structural summary of one CSS document. Comments are stripped first."""
    text = _CSS_COMMENT.sub(" ", text)
    n_media = len(re.findall(r"@media\b", text))
    n_keyframes = len(re.findall(r"@keyframes\b", text))
    n_custom_prop = len(re.findall(r"(?m)^\s*--[-a-zA-Z0-9_]+\s*:", text))
    n_var = len(re.findall(r"\bvar\(\s*--", text))

    decls = []
    for body in _CSS_BLOCK.findall(text):
        for chunk in body.split(";"):
            m = _CSS_DECL.match(chunk)
            if m:
                decls.append((m.group(1).lower(), m.group(2).strip()))

    colors = []          # list of (rgb, syntax) where syntax in {hex3,hex6,rgb,hsl,named}
    families = []
    size_units = collections.Counter()
    hit = collections.Counter()

    for prop, value in decls:
        v = value.lower()
        # -- colours (only from properties that can carry one) --------------
        if prop in _COLOR_PROPS or prop.startswith("border") or "color" in prop:
            for h in _HEX.findall(value):
                rgb = _hex_to_rgb(h)
                if rgb:
                    colors.append((rgb, "hex3" if len(h) == 3 else "hex6"))
            for args in _RGB.findall(v):
                parts = re.split(r"[,\s/]+", args.strip())
                nums = []
                for p in parts[:3]:
                    try:
                        nums.append(int(float(p.replace("%", ""))))
                    except ValueError:
                        break
                if len(nums) == 3:
                    colors.append((tuple(nums), "rgb"))
            for args in _HSL.findall(v):
                colors.append(((0, 0, 0), "hsl"))   # syntax only; hue parsed below
            for nm in _NAMED_RE.findall(v):
                colors.append((_NAMED_COLORS[nm.lower()], "named"))
        # -- typography ------------------------------------------------------
        if prop in ("font-family", "font"):
            first = v.split(",")[0].strip().strip("'\"")
            if first and not first[0].isdigit():
                families.append(first)
        if prop in ("font-size", "font"):
            for unit in ("rem", "em", "pt", "px", "%", "vw"):
                if re.search(r"[\d.]" + unit + r"\b", v):
                    size_units[unit] += 1
                    break
        # -- layout mechanisms ----------------------------------------------
        if prop == "float" and v not in ("none",):
            hit["float"] += 1
        if "flex" in prop or v.startswith("flex") or v.startswith("inline-flex"):
            hit["flex"] += 1
        if prop.startswith("grid") or v.startswith("grid") or v.startswith("inline-grid"):
            hit["grid"] += 1
        if prop == "position" and v == "absolute":
            hit["pos_absolute"] += 1
        if prop == "position" and v in ("fixed", "sticky"):
            hit["pos_fixed"] += 1
        if "border-radius" in prop:
            hit["border_radius"] += 1
        if "shadow" in prop:
            hit["shadow"] += 1
        if "gradient(" in v:
            hit["gradient"] += 1
        if prop.startswith("transition"):
            hit["transition"] += 1
        if prop.startswith("animation"):
            hit["animation"] += 1
        if prop == "transform" or prop.endswith("-transform") and "text" not in prop:
            hit["transform"] += 1
        if "!important" in v:
            hit["important"] += 1
        if prop.startswith("-webkit-") or prop.startswith("-moz-") or prop.startswith("-ms-"):
            hit["vendor"] += 1
        if re.search(r"[\d.](%|vw|vh|fr)\b", v) or "minmax(" in v:
            hit["fluid_unit"] += 1
        if prop in ("width", "max-width", "min-width") and re.search(r"\d+px", v):
            hit["fixed_px_width"] += 1

    return {
        "n_decls": len(decls),
        "n_media": n_media,
        "n_keyframes": n_keyframes,
        "n_custom_prop": n_custom_prop,
        "n_var": n_var,
        "colors": colors,
        "families": families,
        "size_units": size_units,
        "hits": hit,
    }


# --------------------------------------------------------------------------
# HTML parsing
#
# The templates are Jinja, so `{% ... %}` / `{{ ... }}` are stripped first --
# `{% block %}` bodies would otherwise be counted as text and `{{ url_for(...) }}`
# inside an attribute confuses the attribute scanner.
# --------------------------------------------------------------------------

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_JINJA = re.compile(r"\{[%{#].*?[%}#]\}", re.S)
_TAG = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)((?:[^>\"']|\"[^\"]*\"|'[^']*')*)>", re.S)
_ATTR = re.compile(r"([-a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=")
_DOCTYPE = re.compile(r"<!doctype([^>]*)>", re.I)
_STYLE_BLOCK = re.compile(r"<style[^>]*>(.*?)</style>", re.S | re.I)

#: Fixed tag vocabulary. Order is frozen -- it defines feature dimensions.
#: Anything outside it lands in the `_other` bucket, so the histogram is
#: complete and sums to 1.
TAG_VOCAB = (
    "a", "article", "aside", "b", "body", "br", "button", "center", "div",
    "em", "font", "footer", "form", "h1", "h2", "h3", "h4", "h5", "header",
    "hr", "i", "img", "input", "label", "li", "main", "nav", "ol", "option",
    "p", "script", "section", "select", "small", "span", "strong", "svg",
    "table", "td", "textarea", "th", "tr", "ul",
)
_SEMANTIC = ("main", "nav", "header", "footer", "section", "article", "aside",
             "figure", "figcaption")
_FORM_CONTROL = ("input", "select", "button", "textarea", "label")
#: Unambiguously presentational HTML-4 attributes. `width`/`height` are left
#: out on purpose: they are still legal on <img> in HTML 5, so counting them
#: would make every era look old.
_PRESENTATIONAL_ATTRS = (
    "bgcolor", "background", "text", "link", "vlink", "alink", "align",
    "valign", "cellpadding", "cellspacing", "hspace", "vspace", "nowrap",
    "face", "noshade", "marginwidth", "marginheight", "frameborder", "clear",
    "compact", "border",
)


def parse_html(text):
    """Tag histogram + structural markers for one template."""
    doctype = "none"
    m = _DOCTYPE.search(text)
    if m:
        d = m.group(1).lower()
        if "xhtml" in d:
            doctype = "xhtml"
        elif "html 4" in d or "html4" in d:
            doctype = "html4"
        elif d.strip() in ("html", ""):
            doctype = "html5"
        else:
            doctype = "html4"

    inline_css = "\n".join(_STYLE_BLOCK.findall(text))
    text = _STYLE_BLOCK.sub(" ", text)
    text = _HTML_COMMENT.sub(" ", text)
    text = _JINJA.sub(" ", text)

    tags = collections.Counter()
    attrs = collections.Counter()
    n_tags = 0
    n_table_presentational = 0
    for name, attrstr in _TAG.findall(text):
        name = name.lower()
        if name in ("!doctype",):
            continue
        n_tags += 1
        tags[name] += 1
        found = set(a.lower() for a in _ATTR.findall(attrstr))
        for a in found:
            attrs[a] += 1
        if name in ("table", "td", "th", "tr"):
            if found & set(_PRESENTATIONAL_ATTRS):
                n_table_presentational += 1

    return {
        "doctype": doctype,
        "inline_css": inline_css,
        "tags": tags,
        "attrs": attrs,
        "n_tags": n_tags,
        "n_table_presentational": n_table_presentational,
        "has_viewport_meta": bool(re.search(
            r'<meta[^>]+name\s*=\s*["\']viewport', text, re.I)),
        "has_charset_meta": bool(re.search(
            r'<meta[^>]+charset', text, re.I)),
    }


# --------------------------------------------------------------------------
# Signature -> feature vector
#
# NORMALISATION. Every raw quantity is mapped into [0, 1) before it reaches the
# vector, by one of three documented transforms:
#
#   frac(a, b)   = a / b               -- already a proportion (histograms,
#                                         attribute-per-tag fractions)
#   sat(x, c)    = x / (x + c)         -- saturating rational for *densities*
#                                         and counts. Monotone, unbounded input,
#                                         0.5 at the "typical" value c. Chosen
#                                         over log or hard clipping because it
#                                         needs no cap and degrades gracefully
#                                         for the shop themes, whose files are
#                                         3-6x larger than wiki's.
#   bin(x)       = 0.0 / 1.0           -- presence flags (doctype, viewport meta)
#
# The vector is then normalised **per block** to unit L2 length, so colour,
# typography, layout, tag histogram and HTML structure each contribute equally
# to the cosine regardless of how many dimensions they happen to occupy. A block
# that is entirely zero is left at zero.
# --------------------------------------------------------------------------

def _sat(x, c):
    return float(x) / (float(x) + float(c)) if (x + c) > 0 else 0.0


def _frac(a, b):
    return float(a) / float(b) if b else 0.0


_WEBSAFE = set([0x00, 0x33, 0x66, 0x99, 0xCC, 0xFF])
_N_HUE_BINS = 8


def _color_block(css):
    colors = css["colors"]
    n = len(colors)
    rgbs = [c for c, _ in colors]
    syn = collections.Counter(s for _, s in colors)
    distinct = set(rgbs)
    hsl = [_rgb_to_hsl(*c) for c in rgbs]
    sats = [s for _, s, _ in hsl]
    ligs = [l for _, _, l in hsl]
    mean_s = sum(sats) / n if n else 0.0
    mean_l = sum(ligs) / n if n else 0.0
    var_l = (sum((l - mean_l) ** 2 for l in ligs) / n) if n else 0.0
    gray = sum(1 for r, g, b in rgbs if r == g == b)
    websafe = sum(1 for r, g, b in rgbs
                  if r in _WEBSAFE and g in _WEBSAFE and b in _WEBSAFE)
    hue_hist = [0.0] * _N_HUE_BINS
    n_chroma = 0
    for h, s, _l in hsl:
        if s < 0.15:
            continue
        n_chroma += 1
        hue_hist[min(_N_HUE_BINS - 1, int(h * _N_HUE_BINS))] += 1.0
    if n_chroma:
        hue_hist = [v / n_chroma for v in hue_hist]

    out = collections.OrderedDict()
    out["color.palette_size"] = _sat(len(distinct), 40)
    out["color.mean_saturation"] = mean_s
    out["color.mean_lightness"] = mean_l
    out["color.std_lightness"] = math.sqrt(var_l)
    out["color.frac_grayscale"] = _frac(gray, n)
    out["color.frac_websafe"] = _frac(websafe, n)
    out["color.syntax_hex3"] = _frac(syn["hex3"], n)
    out["color.syntax_named"] = _frac(syn["named"], n)
    out["color.syntax_rgb_hsl"] = _frac(syn["rgb"] + syn["hsl"], n)
    for i, v in enumerate(hue_hist):
        out["color.hue_%d" % i] = v
    return out


def _font_block(css):
    fams = [f.lower() for f in css["families"]]
    n = len(fams)

    def _share(keys):
        return _frac(sum(1 for f in fams if any(k in f for k in keys)), n)

    units = css["size_units"]
    n_units = sum(units.values())
    out = collections.OrderedDict()
    out["font.n_families"] = _sat(len(set(fams)), 6)
    out["font.serif"] = _share(_SERIF)
    out["font.sans"] = _share(_SANS)
    out["font.mono"] = _share(_MONO)
    out["font.system_stack"] = _share(_SYSTEM_STACK)
    out["font.quoted_webfont"] = _frac(
        sum(1 for f in fams
            if not any(k in f for k in _SERIF + _SANS + _MONO)), n)
    out["font.size_pt"] = _frac(units["pt"], n_units)
    out["font.size_px"] = _frac(units["px"], n_units)
    out["font.size_rel"] = _frac(units["rem"] + units["em"] + units["%"], n_units)
    return out


#: (hit key, saturation constant) for the layout block. The constant is a
#: density per 100 declarations, so `c=1.0` means "one occurrence per hundred
#: declarations is the half-way point".
_LAYOUT_HITS = (
    ("float", 1.0), ("flex", 2.0), ("grid", 1.0), ("pos_absolute", 1.0),
    ("pos_fixed", 0.5), ("border_radius", 2.0), ("shadow", 1.5),
    ("gradient", 0.5), ("transition", 1.5), ("animation", 0.5),
    ("transform", 0.5), ("important", 0.5), ("vendor", 1.0),
    ("fluid_unit", 5.0), ("fixed_px_width", 2.0),
)


def _layout_block(css):
    n = max(1, css["n_decls"])
    out = collections.OrderedDict()
    for key, c in _LAYOUT_HITS:
        out["layout." + key] = _sat(100.0 * css["hits"][key] / n, c)
    out["layout.media_queries"] = _sat(css["n_media"] + css["n_keyframes"], 4)
    out["layout.custom_props"] = _sat(css["n_custom_prop"], 8)
    out["layout.var_usage"] = _sat(100.0 * css["n_var"] / n, 5.0)
    return out


def _tag_block(html):
    tags = html["tags"]
    n = max(1, html["n_tags"])
    out = collections.OrderedDict()
    covered = 0
    for t in TAG_VOCAB:
        out["tag." + t] = _frac(tags[t], n)
        covered += tags[t]
    out["tag._other"] = _frac(n - covered, n)
    return out


def _struct_block(html):
    tags, attrs = html["tags"], html["attrs"]
    n = max(1, html["n_tags"])
    out = collections.OrderedDict()
    for d in ("html5", "html4", "xhtml", "none"):
        out["struct.doctype_" + d] = 1.0 if html["doctype"] == d else 0.0
    out["struct.table_density"] = _sat(1000.0 * tags["table"] / n, 5.0)
    out["struct.table_presentational"] = _frac(
        html["n_table_presentational"], max(1, tags["table"] + tags["td"]
                                            + tags["th"] + tags["tr"]))
    out["struct.semantic_frac"] = _frac(sum(tags[t] for t in _SEMANTIC), n)
    out["struct.form_control_frac"] = _frac(
        sum(tags[t] for t in _FORM_CONTROL), n)
    out["struct.inline_style_attr"] = _frac(attrs["style"], n)
    out["struct.presentational_attr"] = _frac(
        sum(attrs[a] for a in _PRESENTATIONAL_ATTRS), n)
    out["struct.class_attr"] = _frac(attrs["class"], n)
    out["struct.id_attr"] = _frac(attrs["id"], n)
    out["struct.aria_attr"] = _frac(
        sum(v for k, v in attrs.items() if k.startswith("aria-") or k == "role"), n)
    out["struct.viewport_meta"] = 1.0 if html["has_viewport_meta"] else 0.0
    out["struct.charset_meta"] = 1.0 if html["has_charset_meta"] else 0.0
    out["struct.script_density"] = _sat(1000.0 * tags["script"] / n, 8.0)
    return out


BLOCK_ORDER = ("color", "font", "layout", "tag", "struct")


def signature(cell):
    """Style signature of one (env, era) cell, read off the theme directory.

    Returns a dict with the raw parse summaries, the per-block feature
    OrderedDicts, and the file census that produced them.
    """
    css_paths, html_paths = _theme_files(cell)
    html_parses = [parse_html(_read(p)) for p in html_paths]

    css_text = "\n".join(_read(p) for p in css_paths)
    # Inline <style> blocks are real style information (wiki era 2 keeps its
    # HTML-4 quickbar rules there, shop eras 3 and 5 keep page-specific rules),
    # so they join the CSS corpus rather than being dropped.
    css_text += "\n" + "\n".join(h["inline_css"] for h in html_parses)
    css = parse_css(css_text)

    # Only base.html declares a doctype on wiki/news -- the other templates are
    # Jinja `{% extends %}` children and have none. A plain majority over the
    # four files would therefore report "none" for every wiki theme, so vote
    # only among the files that actually declare one.
    declared = [h["doctype"] for h in html_parses if h["doctype"] != "none"]
    merged = {
        "doctype": (collections.Counter(declared).most_common(1)[0][0]
                    if declared else "none"),
        "tags": sum((h["tags"] for h in html_parses), collections.Counter()),
        "attrs": sum((h["attrs"] for h in html_parses), collections.Counter()),
        "n_tags": sum(h["n_tags"] for h in html_parses),
        "n_table_presentational": sum(h["n_table_presentational"]
                                      for h in html_parses),
        "has_viewport_meta": any(h["has_viewport_meta"] for h in html_parses),
        "has_charset_meta": any(h["has_charset_meta"] for h in html_parses),
    }

    blocks = collections.OrderedDict()
    blocks["color"] = _color_block(css)
    blocks["font"] = _font_block(css)
    blocks["layout"] = _layout_block(css)
    blocks["tag"] = _tag_block(merged)
    blocks["struct"] = _struct_block(merged)

    return {
        "cell": cell.as_dict(),
        "theme_dir": theme_dir(cell),
        "n_css_files": len(css_paths),
        "n_html_files": len(html_paths),
        "html_files": [os.path.basename(p) for p in html_paths],
        "n_css_decls": css["n_decls"],
        "n_html_tags": merged["n_tags"],
        "doctype": merged["doctype"],
        "blocks": blocks,
    }


def _l2_normalise(vals):
    n = math.sqrt(sum(v * v for v in vals))
    if n <= 0.0:
        return list(vals)
    return [v / n for v in vals]


def feature_names():
    """The frozen feature ordering. Built once from an arbitrary cell so the
    names always match what `feature_vector` emits."""
    sig = signature(cells.Cell("wiki", 1))
    names = []
    for b in BLOCK_ORDER:
        names.extend(sig["blocks"][b].keys())
    return tuple(names)


def feature_vector(sig):
    """Concatenated, block-wise-L2-normalised feature vector for a signature."""
    out = []
    for b in BLOCK_ORDER:
        out.extend(_l2_normalise(list(sig["blocks"][b].values())))
    return out


# --------------------------------------------------------------------------
# Similarity machinery
# --------------------------------------------------------------------------

def cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return num / (na * nb)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _center(vecs, groups=None):
    """Mean-centre a {key: vector} dict.

    groups=None       -> grand centring over every key.
    groups={key: g}   -> centre within each group independently.
    """
    keys = list(vecs)
    if not keys:
        return collections.OrderedDict()
    dim = len(vecs[keys[0]])
    if groups is None:
        groups = dict((k, "_all") for k in keys)
    by_group = collections.defaultdict(list)
    for k in keys:
        by_group[groups[k]].append(k)
    out = collections.OrderedDict()
    for g, gkeys in by_group.items():
        mean = [_mean([vecs[k][i] for k in gkeys]) for i in range(dim)]
        for k in gkeys:
            out[k] = [vecs[k][i] - mean[i] for i in range(dim)]
    return collections.OrderedDict((k, out[k]) for k in keys)


def similarity_matrix(vecs, keys):
    return [[cosine(vecs[a], vecs[b]) for b in keys] for a in keys]


def contrasts(mat, keys, eras=None):
    """The two means 6.2 compares, plus the residual 'different era AND different
    env' baseline that makes them interpretable."""
    idx = dict((k, i) for i, k in enumerate(keys))
    parsed = dict((k, cells.Cell.parse(k)) for k in keys)
    use = [k for k in keys
           if eras is None or parsed[k].era in eras]
    same_era, same_env, neither = [], [], []
    for a in use:
        for b in use:
            if a == b:
                continue
            ca, cb = parsed[a], parsed[b]
            v = mat[idx[a]][idx[b]]
            if ca.era == cb.era and ca.env != cb.env:
                same_era.append(v)
            elif ca.env == cb.env and ca.era != cb.era:
                same_env.append(v)
            elif ca.env != cb.env and ca.era != cb.era:
                neither.append(v)
    return {
        "within_era_across_env": _mean(same_era),
        "within_env_across_era": _mean(same_env),
        "across_both": _mean(neither),
        "n_within_era_across_env": len(same_era),
        "n_within_env_across_era": len(same_env),
        "delta": _mean(same_era) - _mean(same_env),
    }


def per_era_agreement(mat, keys, eras=None):
    """Mean of the three cross-site pairs at each era, ranked best first."""
    idx = dict((k, i) for i, k in enumerate(keys))
    eras = eras or cells.ERAS
    rows = []
    for era in eras:
        ks = ["%s_e%d" % (e, era) for e in cells.ENVIRONMENTS]
        ks = [k for k in ks if k in idx]
        pairs = []
        detail = collections.OrderedDict()
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                v = mat[idx[ks[i]]][idx[ks[j]]]
                pairs.append(v)
                detail["%s|%s" % (ks[i], ks[j])] = v
        rows.append({"era": era, "mean": _mean(pairs), "pairs": detail,
                     "year_spread": cells.ERA_YEAR_SPREAD[era]})
    rows.sort(key=lambda r: -r["mean"])
    return rows


def nearest_cross_env(mat, keys, eras=None):
    """For each cell, its most similar cell in a *different* environment.

    'Top-1 era match' is the sharpest single number in this module: if era
    styles correspond, wiki_e3's nearest news/shop neighbour should be era 3.
    Chance is 2/10 = 0.20 over the five temporal eras (10 cross-env candidates,
    2 of which share the era).
    """
    idx = dict((k, i) for i, k in enumerate(keys))
    parsed = dict((k, cells.Cell.parse(k)) for k in keys)
    use = [k for k in keys if eras is None or parsed[k].era in eras]
    rows, hits, gaps = [], 0, []
    for a in use:
        cands = [b for b in use if parsed[b].env != parsed[a].env]
        if not cands:
            continue
        best = max(cands, key=lambda b: mat[idx[a]][idx[b]])
        ok = parsed[best].era == parsed[a].era
        hits += 1 if ok else 0
        gaps.append(abs(parsed[best].era - parsed[a].era))
        rows.append({"cell": a, "nearest": best,
                     "sim": mat[idx[a]][idx[best]], "era_match": ok,
                     "era_gap": abs(parsed[best].era - parsed[a].era)})
    n_eras = len(set(parsed[k].era for k in use))
    n_env = len(set(parsed[k].env for k in use))
    chance = (n_env - 1.0) / ((n_env - 1.0) * n_eras) if n_eras else 0.0
    return {"rows": rows, "n": len(rows), "top1_era_match": _frac(hits, len(rows)),
            "chance": chance, "mean_era_gap": _mean(gaps)}


PERMUTATIONS = 10000
PERM_SEED = 20260806


def permutation_test(vecs, keys, eras, n_perm=PERMUTATIONS, seed=PERM_SEED):
    """Is the within-era across-env similarity higher than chance era pairing?

    Null: era labels are exchangeable within each environment. We permute them
    independently per environment and recompute the within-era across-env mean.
    Under the env-centred features the expected cross-env dot product is exactly
    0 (each environment's residuals sum to zero), so this test has a
    well-defined null and needs no extra assumptions.
    """
    parsed = dict((k, cells.Cell.parse(k)) for k in keys)
    use = [k for k in keys if parsed[k].era in eras]
    mat = similarity_matrix(vecs, use)
    idx = dict((k, i) for i, k in enumerate(use))
    envs = sorted(set(parsed[k].env for k in use))
    eras = sorted(eras)

    def _score(assign):
        """assign: env -> {era_slot: cell_key}."""
        vals = []
        for i in range(len(envs)):
            for j in range(i + 1, len(envs)):
                for e in eras:
                    a, b = assign[envs[i]].get(e), assign[envs[j]].get(e)
                    if a and b:
                        vals.append(mat[idx[a]][idx[b]])
        return _mean(vals)

    ident = dict((env, dict((parsed[k].era, k) for k in use
                            if parsed[k].env == env)) for env in envs)
    observed = _score(ident)

    rng = random.Random(seed)
    ge = 0
    null = []
    for _ in range(n_perm):
        assign = {}
        for env in envs:
            ks = [ident[env][e] for e in eras if e in ident[env]]
            shuffled = list(ks)
            rng.shuffle(shuffled)
            assign[env] = dict(zip(eras, shuffled))
        s = _score(assign)
        null.append(s)
        if s >= observed:
            ge += 1
    null.sort()
    return {
        "observed": observed,
        "p_value": (ge + 1.0) / (n_perm + 1.0),
        "null_mean": _mean(null),
        "null_p95": null[int(0.95 * len(null))] if null else float("nan"),
        "n_perm": n_perm,
    }


# --------------------------------------------------------------------------
# Verdict (adapterCL.md 9: "Era styles don't correspond -> fatal to centerpiece")
# --------------------------------------------------------------------------

#: Thresholds are stated up front so the verdict cannot be reverse-engineered
#: from the numbers after the fact.
V_ALPHA = 0.05                 # permutation p-value for "there is an era signal"
V_ALPHA_WEAK = 0.20            # below this but above V_ALPHA -> partial
V_TOP1_MULTIPLIER = 2.0        # nearest cross-env neighbour must beat 2x chance
V_MIN_SIM = 0.0                # env-centred within-era similarity must be > 0


def verdict(report):
    """corresponds / partial / does_not_correspond, with the numbers behind it."""
    perm = report["permutation_test"]
    near = report["nearest_cross_env"]
    con = report["contrasts"]["env"]

    sig = perm["p_value"] < V_ALPHA
    weak_sig = perm["p_value"] < V_ALPHA_WEAK
    positive = con["within_era_across_env"] > V_MIN_SIM
    top1_ok = near["top1_era_match"] >= V_TOP1_MULTIPLIER * near["chance"]

    if sig and positive and top1_ok:
        v = "corresponds"
        why = ("era-aligned cells are more similar than chance era pairings "
               "(p=%.4f) and %d/%d cells find a same-era nearest neighbour in "
               "another site (%.0f%% vs %.0f%% chance)"
               % (perm["p_value"],
                  int(round(near["top1_era_match"] * near["n"])), near["n"],
                  100 * near["top1_era_match"], 100 * near["chance"]))
    elif (sig or weak_sig) and positive:
        v = "partial"
        why = ("there is an era signal (p=%.4f, mean env-centred within-era "
               "similarity %.3f) but it is not strong enough to identify the "
               "matching era: top-1 era match %.0f%% vs %.0f%% chance"
               % (perm["p_value"], con["within_era_across_env"],
                  100 * near["top1_era_match"], 100 * near["chance"]))
    else:
        v = "does_not_correspond"
        why = ("era-aligned cells across sites are no more similar than "
               "arbitrarily paired eras (p=%.4f, mean env-centred within-era "
               "similarity %.3f, top-1 era match %.0f%% vs %.0f%% chance)"
               % (perm["p_value"], con["within_era_across_env"],
                  100 * near["top1_era_match"], 100 * near["chance"]))

    return {
        "verdict": v,
        "reason": why,
        "p_value": perm["p_value"],
        "env_centred_within_era": con["within_era_across_env"],
        "env_centred_within_env": con["within_env_across_era"],
        "top1_era_match": near["top1_era_match"],
        "top1_chance": near["chance"],
        "mean_era_gap": near["mean_era_gap"],
        "thresholds": {"alpha": V_ALPHA, "alpha_weak": V_ALPHA_WEAK,
                       "top1_multiplier": V_TOP1_MULTIPLIER,
                       "min_similarity": V_MIN_SIM},
        "consequence": (
            "6.2 dissociation runs as designed" if v == "corresponds" else
            "6.2 runs but the within-theme arm is weakened; report the "
            "correspondence numbers in the paper rather than assuming them"
            if v == "partial" else
            "6.2 is dead as specified (adapterCL.md 7 Phase 0 kill criterion); "
            "reconsider scope before Phase 2"),
    }


# --------------------------------------------------------------------------
# Screenshot (perceptual) signatures -- optional
# --------------------------------------------------------------------------

GRAY_SIDE = 16          # 16x16 downsample -> 256 dims
RGB_BINS = 4            # 4x4x4 colour histogram -> 64 dims


def _find_screenshot(root, cell):
    """Locate a cell's screenshot under a capture.py output tree."""
    cands = [
        os.path.join(root, cell.key, "landing", "screenshot.png"),
        os.path.join(root, cell.key, "screenshot.png"),
        os.path.join(root, cell.key + ".png"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    for dirpath, _dirnames, filenames in os.walk(root):
        if cell.key in dirpath.split(os.sep) and "screenshot.png" in filenames:
            return os.path.join(dirpath, "screenshot.png")
    return None


def perceptual_signature(png_path):
    """Downsampled grayscale + coarse RGB histogram. Needs PIL; returns None
    with a reason if PIL is unavailable."""
    try:
        from PIL import Image
    except ImportError:
        return None, "PIL is not installed in this interpreter"
    im = Image.open(png_path)
    im = im.convert("RGB")
    gray = im.convert("L").resize((GRAY_SIDE, GRAY_SIDE))
    g = [p / 255.0 for p in gray.getdata()]
    small = im.resize((64, 64))
    hist = [0.0] * (RGB_BINS ** 3)
    step = 256 // RGB_BINS
    for r, gg, b in small.getdata():
        hist[(r // step) * RGB_BINS * RGB_BINS + (gg // step) * RGB_BINS
             + (b // step)] += 1.0
    tot = sum(hist) or 1.0
    hist = [h / tot for h in hist]
    return _l2_normalise(g) + _l2_normalise(hist), None


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------

def analyse(vecs, keys, temporal_eras=None, n_perm=PERMUTATIONS):
    """Every similarity view + contrast for one set of feature vectors."""
    temporal_eras = tuple(temporal_eras or cells.TEMPORAL_ERAS)
    parsed = dict((k, cells.Cell.parse(k)) for k in keys)
    variants = collections.OrderedDict()
    variants["raw"] = collections.OrderedDict((k, vecs[k]) for k in keys)
    variants["grand"] = _center(variants["raw"])
    tkeys = [k for k in keys if parsed[k].era in temporal_eras]
    variants["env"] = _center(
        collections.OrderedDict((k, vecs[k]) for k in tkeys),
        groups=dict((k, parsed[k].env) for k in tkeys))

    mats, cons, eras_agree = (collections.OrderedDict(),
                              collections.OrderedDict(),
                              collections.OrderedDict())
    for name, vv in variants.items():
        kk = list(vv)
        m = similarity_matrix(vv, kk)
        mats[name] = {"keys": kk, "matrix": m}
        cons[name] = contrasts(m, kk, eras=temporal_eras)
        eras_agree[name] = per_era_agreement(m, kk, eras=temporal_eras)

    env_keys = mats["env"]["keys"]
    near = nearest_cross_env(mats["env"]["matrix"], env_keys)
    perm = permutation_test(variants["env"], env_keys, temporal_eras,
                            n_perm=n_perm)
    return {"matrices": mats, "contrasts": cons, "per_era": eras_agree,
            "nearest_cross_env": near, "permutation_test": perm,
            "temporal_eras": list(temporal_eras)}


def build_report(screenshots=None, n_perm=PERMUTATIONS):
    """Full static (and optionally perceptual) era-correspondence report."""
    keys = [c.key for c in cells.ALL_CELLS]
    sigs = collections.OrderedDict((c.key, signature(c)) for c in cells.ALL_CELLS)
    vecs = collections.OrderedDict((k, feature_vector(sigs[k])) for k in keys)

    rep = analyse(vecs, keys, n_perm=n_perm)
    rep["feature_names"] = list(feature_names())
    rep["n_features"] = len(rep["feature_names"])
    rep["block_order"] = list(BLOCK_ORDER)
    rep["signatures"] = collections.OrderedDict(
        (k, {"theme_dir": sigs[k]["theme_dir"],
             "n_css_files": sigs[k]["n_css_files"],
             "n_html_files": sigs[k]["n_html_files"],
             "html_files": sigs[k]["html_files"],
             "n_css_decls": sigs[k]["n_css_decls"],
             "n_html_tags": sigs[k]["n_html_tags"],
             "doctype": sigs[k]["doctype"]}) for k in keys)
    rep["vectors"] = collections.OrderedDict(
        (k, [round(v, 6) for v in vecs[k]]) for k in keys)
    rep["year_spread"] = dict((str(e), cells.ERA_YEAR_SPREAD[e])
                              for e in cells.ERAS)
    rep["neutral_era"] = cells.NEUTRAL_ERA
    rep["verdict"] = verdict(rep)

    if screenshots:
        rep["perceptual"] = _perceptual_report(screenshots, n_perm=n_perm)
    return rep


def _perceptual_report(root, n_perm=PERMUTATIONS):
    if not os.path.isdir(root):
        return {"skipped": "screenshot dir does not exist: %s" % root}
    pvecs, found, missing, err = collections.OrderedDict(), [], [], None
    for c in cells.ALL_CELLS:
        p = _find_screenshot(root, c)
        if p is None:
            missing.append(c.key)
            continue
        vec, why = perceptual_signature(p)
        if vec is None:
            err = why
            break
        pvecs[c.key] = vec
        found.append(c.key)
    if err:
        return {"skipped": err}
    # The contrasts are only defined if at least two sites and two temporal
    # eras are represented; below that the permutation null is degenerate.
    got = [cells.Cell.parse(k) for k in pvecs]
    n_env = len(set(c.env for c in got))
    n_era = len(set(c.era for c in got if c.is_temporal))
    if n_env < 2 or n_era < 2:
        return {"skipped": "need >=2 sites and >=2 temporal eras; found %d/%d "
                           "screenshots under %s (%d sites, %d temporal eras)"
                           % (len(pvecs), len(cells.ALL_CELLS), root,
                              n_env, n_era),
                "missing": missing}
    out = analyse(pvecs, list(pvecs), n_perm=n_perm)
    out["root"] = root
    out["n_cells"] = len(pvecs)
    out["missing"] = missing
    out["dims"] = GRAY_SIDE * GRAY_SIDE + RGB_BINS ** 3
    return out


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def _short(key):
    c = cells.Cell.parse(key)
    return "%s%d" % (c.env[0].upper(), c.era)


def print_matrix(mat, keys, fh, title):
    print("", file=fh)
    print(title, file=fh)
    print("       " + " ".join("%5s" % _short(k) for k in keys), file=fh)
    for i, k in enumerate(keys):
        print("%-6s " % _short(k)
              + " ".join("%5.2f" % v for v in mat[i]), file=fh)


def print_report(rep, fh=None, full_matrix=True):
    fh = fh or sys.stdout
    p = lambda s="": print(s, file=fh)

    p("=" * 78)
    p("adapterCL 6.2 precondition: do era styles correspond across the three sites?")
    p("=" * 78)
    p("static signatures from theme directories on disk -- no servers, no browser")
    p("%d cells, %d features (%s), block-wise L2 normalised"
      % (len(rep["signatures"]), rep["n_features"], "+".join(rep["block_order"])))
    p("")
    p("NOTE era %d is a style-NEUTRAL baseline in all three sites (wiki 6-minimal,"
      % rep["neutral_era"])
    p("     news 6-base-minimal, shop classic). It carries no era semantics, so it is")
    p("     EXCLUDED from every temporal statistic below; it is still drawn in the")
    p("     18x18 figure. Temporal eras analysed: %s" % (rep["temporal_eras"],))
    p("")

    p("theme census")
    p("  %-10s %-30s %5s %6s %7s %s" % ("cell", "theme dir", "html", "decls",
                                        "tags", "doctype"))
    for k, s in rep["signatures"].items():
        p("  %-10s %-30s %5d %6d %7d %s"
          % (k, os.path.basename(s["theme_dir"]), s["n_html_files"],
             s["n_css_decls"], s["n_html_tags"], s["doctype"]))
    p("")

    p("nominal-year disagreement (cells.ERA_YEAR_SPREAD)")
    p("  %-5s %-14s %-14s %-14s %s" % ("era", "wiki", "news", "shop", "spread"))
    for e in cells.ERAS:
        row = []
        for env in cells.ENVIRONMENTS:
            y = cells.NOMINAL_YEAR[env][e]
            row.append("%s" % ("neutral" if y is None else y))
        sp = rep["year_spread"][str(e)]
        p("  %-5d %-14s %-14s %-14s %s"
          % (e, row[0], row[1], row[2],
             "n/a" if sp is None else "%.1f yr" % sp))
    worst = max((e for e in cells.TEMPORAL_ERAS),
                key=lambda e: rep["year_spread"][str(e)])
    p("  -> era %d has the largest nominal-year spread (%.1f yr): wiki 2003-4 vs "
      "news 2008 vs shop 2010." % (worst, rep["year_spread"][str(worst)]))
    p("     The three sites are NOT depicting the same year at that era, which is")
    p("     the single biggest threat to the 6.2 crossing.")
    p("")

    if full_matrix:
        m = rep["matrices"]["raw"]
        print_matrix(m["matrix"], m["keys"], fh,
                     "18x18 raw cosine (W=wiki N=news S=shop, digit = era)")
        m = rep["matrices"]["env"]
        print_matrix(m["matrix"], m["keys"], fh,
                     "15x15 env-centred cosine (temporal eras only; null = 0.00)")

    p("")
    p("6.2 contrast: mean within-ERA across-ENV  vs  mean within-ENV across-ERA")
    p("  %-8s %22s %22s %14s %8s" % ("variant", "within-era/across-env",
                                     "within-env/across-era", "across both",
                                     "delta"))
    for name in ("raw", "grand", "env"):
        c = rep["contrasts"][name]
        p("  %-8s %22.4f %22.4f %14.4f %8.4f"
          % (name, c["within_era_across_env"], c["within_env_across_era"],
             c["across_both"], c["delta"]))
    p("  raw   = plain cosine over non-negative features; dominated by site")
    p("          identity (shop templates are 3-6x larger with a different page")
    p("          inventory), so within-env wins almost by construction.")
    p("  grand = features mean-centred over all 18 cells.")
    p("  env   = features mean-centred WITHIN each site over its 5 temporal eras;")
    p("          this is the variant that isolates 'is there an era signal at all'.")
    p("")

    p("per-era agreement of the three sites (env-centred, ranked)")
    p("  %-5s %8s %10s  %s" % ("era", "mean", "yr spread", "pairs"))
    for row in rep["per_era"]["env"]:
        pairs = "  ".join("%s=%.2f" % (k.replace("_e", "").replace("|", "/"), v)
                          for k, v in row["pairs"].items())
        p("  %-5d %8.3f %10s  %s"
          % (row["era"], row["mean"],
             "n/a" if row["year_spread"] is None else "%.1f" % row["year_spread"],
             pairs))
    p("")

    near = rep["nearest_cross_env"]
    p("nearest cross-site neighbour (env-centred)")
    p("  top-1 same-era match: %d/%d = %.0f%%   (chance %.0f%%)"
      % (int(round(near["top1_era_match"] * near["n"])), near["n"],
         100 * near["top1_era_match"], 100 * near["chance"]))
    p("  mean |era gap| to nearest cross-site cell: %.2f" % near["mean_era_gap"])
    for r in near["rows"]:
        p("    %-10s -> %-10s  sim=%+.3f  %s"
          % (r["cell"], r["nearest"], r["sim"],
             "SAME ERA" if r["era_match"] else "era gap %d" % r["era_gap"]))
    p("")

    perm = rep["permutation_test"]
    p("permutation test (era labels shuffled within each site, %d draws)"
      % perm["n_perm"])
    p("  observed within-era across-site mean: %+.4f" % perm["observed"])
    p("  null mean %+.4f, null 95th pct %+.4f  ->  p = %.4f"
      % (perm["null_mean"], perm["null_p95"], perm["p_value"]))
    p("")

    if "perceptual" in rep:
        pc = rep["perceptual"]
        p("-" * 78)
        if "skipped" in pc:
            p("perceptual (screenshot) check SKIPPED: %s" % pc["skipped"])
        else:
            p("perceptual (screenshot) check over %d cells, %d dims"
              % (pc["n_cells"], pc["dims"]))
            p("  %-8s %22s %22s %8s" % ("variant", "within-era/across-env",
                                        "within-env/across-era", "delta"))
            for name in ("raw", "grand", "env"):
                c = pc["contrasts"][name]
                p("  %-8s %22.4f %22.4f %8.4f"
                  % (name, c["within_era_across_env"],
                     c["within_env_across_era"], c["delta"]))
            p("  top-1 same-era match: %.0f%% (chance %.0f%%), p = %.4f"
              % (100 * pc["nearest_cross_env"]["top1_era_match"],
                 100 * pc["nearest_cross_env"]["chance"],
                 pc["permutation_test"]["p_value"]))
        p("")

    v = rep["verdict"]
    p("=" * 78)
    p("VERDICT: %s" % v["verdict"].upper())
    p("  %s" % v["reason"])
    p("  consequence: %s" % v["consequence"])
    p("=" * 78)
    return rep


def write_markdown(rep, path):
    """The 18x18 matrix as a markdown table -- a figure for the paper."""
    m = rep["matrices"]["raw"]
    keys, mat = m["keys"], m["matrix"]
    lines = []
    lines.append("# adapterCL 6.2 precondition: era style correspondence")
    lines.append("")
    lines.append("Static style signatures over the 18 TimeWarp theme "
                 "directories (%d features, block-wise L2 normalised). "
                 "See `adaptercl/era_style.py`." % rep["n_features"])
    lines.append("")
    lines.append("**Verdict: %s** -- %s" % (rep["verdict"]["verdict"],
                                            rep["verdict"]["reason"]))
    lines.append("")
    lines.append("Legend: `W`=wiki, `N`=news, `S`=shop; digit = era. "
                 "Era 6 is the style-neutral baseline in all three sites and is "
                 "excluded from the temporal statistics.")
    lines.append("")
    lines.append("## 18x18 raw cosine similarity")
    lines.append("")
    lines.append("| | " + " | ".join(_short(k) for k in keys) + " |")
    lines.append("|" + "---|" * (len(keys) + 1))
    for i, k in enumerate(keys):
        lines.append("| **%s** | " % _short(k)
                     + " | ".join("%.2f" % v for v in mat[i]) + " |")
    lines.append("")

    m = rep["matrices"]["env"]
    keys, mat = m["keys"], m["matrix"]
    lines.append("## 15x15 environment-centred cosine (temporal eras only)")
    lines.append("")
    lines.append("Each site's five era vectors are mean-centred within the site, "
                 "so the expected cross-site value is exactly 0. Positive = more "
                 "alike than the site's own average.")
    lines.append("")
    lines.append("| | " + " | ".join(_short(k) for k in keys) + " |")
    lines.append("|" + "---|" * (len(keys) + 1))
    for i, k in enumerate(keys):
        lines.append("| **%s** | " % _short(k)
                     + " | ".join("%+.2f" % v for v in mat[i]) + " |")
    lines.append("")

    lines.append("## Contrast (adapterCL.md 6.2)")
    lines.append("")
    lines.append("| variant | within-era across-env | within-env across-era | "
                 "across both | delta |")
    lines.append("|---|---|---|---|---|")
    for name in ("raw", "grand", "env"):
        c = rep["contrasts"][name]
        lines.append("| %s | %.4f | %.4f | %.4f | %+.4f |"
                     % (name, c["within_era_across_env"],
                        c["within_env_across_era"], c["across_both"],
                        c["delta"]))
    lines.append("")
    lines.append("Permutation test (era labels shuffled within site, %d draws): "
                 "observed %+.4f, null mean %+.4f, **p = %.4f**."
                 % (rep["permutation_test"]["n_perm"],
                    rep["permutation_test"]["observed"],
                    rep["permutation_test"]["null_mean"],
                    rep["permutation_test"]["p_value"]))
    lines.append("")
    lines.append("## Per-era agreement of the three sites (environment-centred)")
    lines.append("")
    lines.append("| era | mean cross-site similarity | nominal-year spread |")
    lines.append("|---|---|---|")
    for row in rep["per_era"]["env"]:
        lines.append("| %d | %+.3f | %s |"
                     % (row["era"], row["mean"],
                        "n/a" if row["year_spread"] is None
                        else "%.1f yr" % row["year_spread"]))
    lines.append("")

    fh = open(path, "w")
    try:
        fh.write("\n".join(lines) + "\n")
    finally:
        fh.close()
    return path


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="adapterCL 6.2 precondition check: do era styles "
                    "correspond across wiki / news / shop?")
    ap.add_argument("--screenshots", default=None,
                    help="capture.py output root; adds a perceptual check")
    ap.add_argument("--permutations", type=int, default=PERMUTATIONS)
    ap.add_argument("--out-dir", default=None,
                    help="default: paths.OUT_STYLE")
    ap.add_argument("--no-matrix", action="store_true",
                    help="suppress the 18x18 matrix in the console report")
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args(argv)

    out_dir = args.out_dir or paths.OUT_STYLE
    paths.ensure_out_dirs()
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    rep = build_report(screenshots=args.screenshots, n_perm=args.permutations)

    json_path = os.path.join(out_dir, "era_style.json")
    fh = open(json_path, "w")
    try:
        json.dump(rep, fh, indent=2, sort_keys=True, default=str)
    finally:
        fh.close()
    md_path = write_markdown(rep, os.path.join(out_dir, "era_style.md"))

    if not args.json_only:
        print_report(rep, full_matrix=not args.no_matrix)
    print("")
    print("wrote %s" % json_path)
    print("wrote %s" % md_path)
    return 0 if rep["verdict"]["verdict"] != "does_not_correspond" else 2


if __name__ == "__main__":
    sys.exit(main())
