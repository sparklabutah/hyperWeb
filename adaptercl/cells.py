"""The (environment x era) cell registry -- the experimental unit of adapterCL.

adapterCL.md 6.1/6.2 assume a crossed 3 x 6 design: three TimeWarp sites (wiki,
news, shop) rendered in six temporally-ordered UI eras. This module is the single
source of truth for that grid: which theme directory each cell corresponds to,
what year it nominally depicts, which tasks belong to it, and how the cells are
split into train/test folds.

Two facts here contradict the plan and are load-bearing:

1. **Era 6 is not the newest era.** For all three sites, theme 6 is a neutral
   "minimal"/"classic" baseline; era 5 is 2024/2025. A monotone time axis over
   1..6 is wrong. `nominal_year()` returns None for era 6 and
   `TEMPORAL_ERAS` excludes it, so extrapolation splits are built over 1..5.
2. **Era correspondence across sites is real but imperfect.** v1 and v5 line up
   well (2000/2000s/2001 and 2024/2025/2023-25); v3 does not (wiki 2003-4 vs
   news 2008 vs shop 2010). era_style.py quantifies this; `ERA_YEAR_SPREAD`
   records the nominal disagreement.

Pure stdlib -- importable from the system python (3.6+).
"""

from __future__ import print_function

import collections
import csv
import json
import os
import re

from . import paths

# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------

ENVIRONMENTS = ("wiki", "news", "shop")
ERAS = (1, 2, 3, 4, 5, 6)

#: Eras that lie on the temporal axis. Era 6 is a style-neutral baseline in all
#: three sites, so it is excluded from any ordering-based split.
TEMPORAL_ERAS = (1, 2, 3, 4, 5)
NEUTRAL_ERA = 6

#: era -> theme directory name, read off the three flask apps:
#:   TimeWarp/env/wiki/wiki_app.py:31-39
#:   TimeWarp/env/news/news_app.py:461-468
#:   TimeWarp/env/webshop/web_agent_site/app.py:40-47
THEME_NAME = {
    "wiki": {1: "1-2001", 2: "2-2002", 3: "3-2003-4", 4: "4-2005-2022",
             5: "5-2023-2025", 6: "6-minimal"},
    "news": {1: "1-2000s", 2: "2-2004s", 3: "3-2008s", 4: "4-2016s",
             5: "5-2024s", 6: "6-base-minimal"},
    "shop": {1: "webshop2000", 2: "webshop2005", 3: "webshop2010",
             4: "webshop2015", 5: "webshop2025", 6: "classic"},
}

#: Nominal year each theme depicts, parsed from the theme names above. Ranges
#: are collapsed to their midpoint. None == not on the temporal axis.
NOMINAL_YEAR = {
    "wiki": {1: 2001, 2: 2002, 3: 2003.5, 4: 2013.5, 5: 2024, 6: None},
    "news": {1: 2000, 2: 2004, 3: 2008, 4: 2016, 5: 2024, 6: None},
    "shop": {1: 2000, 2: 2005, 3: 2010, 4: 2015, 5: 2025, 6: None},
}

#: Theme roots on disk, relative to the repo. Used by era_style.py.
THEME_ROOT = {
    "wiki": os.path.join(paths.TIMEWARP_ENV, "env", "wiki", "themes"),
    "news": os.path.join(paths.TIMEWARP_ENV, "env", "news", "themes"),
    "shop": os.path.join(paths.TIMEWARP_ENV, "env", "webshop",
                         "web_agent_site", "themes"),
}

#: Env var each site's URL is published under (browsergym/timewarp/instance.py).
TW_URL_VAR = {"wiki": "TW_WIKI", "news": "TW_NEWS", "shop": "TW_WEBSHOP"}

#: The key used in the task JSON's `sites` list.
SITE_KEY = {"wiki": "wiki", "news": "news", "shop": "webshop"}
SITE_KEY_INV = dict((v, k) for k, v in SITE_KEY.items())


def nominal_year(env, era):
    return NOMINAL_YEAR[env][era]


def era_year_spread(era):
    """Max-min nominal year across the three sites for one era.

    Small spread == the three sites agree on what this era looks like, which is
    the 6.2 precondition. Returns None for the neutral era.
    """
    ys = [NOMINAL_YEAR[e][era] for e in ENVIRONMENTS]
    if any(y is None for y in ys):
        return None
    return max(ys) - min(ys)


ERA_YEAR_SPREAD = dict((e, era_year_spread(e)) for e in ERAS)


class Cell(object):
    """One (environment, era) configuration of the benchmark."""

    __slots__ = ("env", "era")

    #: Which axis this unit is a unit OF. `bcdata.load_cell` dispatches on this
    #: rather than duck-typing on `.env`: a Site's `.env` is its site name for
    #: the single-site keys and None for `multi`/`all`, so the old
    #: `getattr(unit, "env", "") is None` test would silently classify
    #: `Site("wiki")` as a Cell and select zero episodes for it.
    granularity = "cell"

    def __init__(self, env, era):
        if env not in ENVIRONMENTS:
            raise ValueError("unknown environment %r (want one of %r)"
                             % (env, ENVIRONMENTS))
        if era not in ERAS:
            raise ValueError("unknown era %r (want one of %r)" % (era, ERAS))
        self.env = env
        self.era = era

    # -- identity ---------------------------------------------------------
    @property
    def key(self):
        return "%s_e%d" % (self.env, self.era)

    @property
    def theme(self):
        return THEME_NAME[self.env][self.era]

    @property
    def year(self):
        return NOMINAL_YEAR[self.env][self.era]

    @property
    def is_temporal(self):
        return self.era in TEMPORAL_ERAS

    def __eq__(self, other):
        return isinstance(other, Cell) and (self.env, self.era) == (other.env, other.era)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.env, self.era))

    def __repr__(self):
        return "Cell(%s, era=%d, theme=%s)" % (self.env, self.era, self.theme)

    def as_dict(self):
        return {"env": self.env, "era": self.era, "key": self.key,
                "theme": self.theme, "year": self.year}

    @staticmethod
    def parse(key):
        """'wiki_e3' -> Cell('wiki', 3)."""
        m = re.match(r"^([a-z]+)_e([1-6])$", key)
        if not m:
            raise ValueError("bad cell key %r" % (key,))
        return Cell(m.group(1), int(m.group(2)))


ALL_CELLS = tuple(Cell(e, v) for e in ENVIRONMENTS for v in ERAS)
TEMPORAL_CELLS = tuple(c for c in ALL_CELLS if c.is_temporal)


def cells_for(envs=None, eras=None):
    envs = ENVIRONMENTS if envs is None else tuple(envs)
    eras = ERAS if eras is None else tuple(eras)
    return tuple(Cell(e, v) for e in envs for v in eras)


# --------------------------------------------------------------------------
# Version — the adapter unit for the primary experiment
# --------------------------------------------------------------------------

class Version(object):
    """One UI era, pooled across all three sites. **The default adapter unit.**

    2 makes version-level CL the setting ("Primary: zero-shot adaptation to
    unseen UI versions"), and an era is one coherent design language rendered on
    every site, so one adapter per era is the object the claim in 3 is actually
    about. The environment x era `Cell` grid exists for one purpose -- 6.2's
    dissociation, which has to cross the two factors to ask whether an adapter
    encodes appearance or site function -- and is not the training unit.

    Pooling also fixes the data problem outright: `wiki_e1` has 0 BC episodes
    and `shop_e1` has 2, so two of eighteen cells cannot be trained at all,
    whereas version 1 pooled has 30 episodes and every version has >= 30.
    """

    __slots__ = ("era",)

    granularity = "version"

    def __init__(self, era):
        if era not in ERAS:
            raise ValueError("unknown era %r (want one of %r)" % (era, ERAS))
        self.era = era

    @property
    def key(self):
        return "v%d" % self.era

    @property
    def env(self):
        """A Version spans every environment; kept so Cell/Version are
        duck-type compatible wherever code only needs `.key` and `.era`."""
        return None

    @property
    def cells(self):
        return tuple(Cell(e, self.era) for e in ENVIRONMENTS)

    @property
    def themes(self):
        return dict((e, THEME_NAME[e][self.era]) for e in ENVIRONMENTS)

    @property
    def year(self):
        """Mean nominal year across sites; None for the neutral era."""
        ys = [NOMINAL_YEAR[e][self.era] for e in ENVIRONMENTS]
        return None if any(y is None for y in ys) else sum(ys) / float(len(ys))

    @property
    def is_temporal(self):
        return self.era in TEMPORAL_ERAS

    def __eq__(self, other):
        return isinstance(other, Version) and other.era == self.era

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(("version", self.era))

    def __repr__(self):
        return "Version(era=%d, themes=%s)" % (self.era, self.themes)

    def as_dict(self):
        return {"era": self.era, "key": self.key, "themes": self.themes,
                "year": self.year, "cells": [c.key for c in self.cells]}

    @staticmethod
    def parse(key):
        m = re.match(r"^v([1-6])$", key)
        if not m:
            raise ValueError("bad version key %r" % (key,))
        return Version(int(m.group(1)))


ALL_VERSIONS = tuple(Version(e) for e in ERAS)
TEMPORAL_VERSIONS = tuple(v for v in ALL_VERSIONS if v.is_temporal)

# --------------------------------------------------------------------------
# Site — the adapter unit for the DOMAIN axis (domain_transfer_timewarp_plan.md)
# --------------------------------------------------------------------------

#: The site-level training units. `wiki|news|shop` are the three single-site
#: corpora; `multi` is the 50 tasks whose `sites` list has >= 2 entries (the
#: records `task_cell_map` deliberately drops and the census reports as
#: "unattributable"); `all` is the union and the pooled control.
SITE_UNITS = ("wiki", "news", "shop", "multi", "all")

#: The four post-hoc groups an eval's episodes are bucketed into. `all` is not a
#: group -- it is a corpus -- so it is absent here on purpose.
SITE_GROUPS = ("wiki", "news", "shop", "multi")


def canonical_site_key(sites):
    """'wiki+news' for {'news', 'wiki'}: sites in ENVIRONMENTS order, '+'-joined.

    One spelling for one set, used for BOTH composite training units and the
    multi-site task subgroups, so "trained on wiki+news" and "a wiki+news task"
    are the same string. Refuses anything that is not a set of >= 1 known
    environments.
    """
    s = set(sites)
    bad = s - set(ENVIRONMENTS)
    if bad or not s:
        raise ValueError("not a set of environments: %r" % (sorted(s),))
    return "+".join(e for e in ENVIRONMENTS if e in s)


#: Composite single-site unions -- the LEAVE-ONE-SITE-OUT corpora. Every pair
#: and the triple, each excluding the `multi` episodes (a composite is "the
#: single-site episodes of these sites", nothing else). `wiki+news+shop` is NOT
#: `all`: `all` also carries the multi-site episodes.
COMPOSITE_SITE_UNITS = tuple(
    canonical_site_key(c) for n in (2, 3)
    for c in __import__("itertools").combinations(ENVIRONMENTS, n))

#: The leave-one-site-out ARMS (domain-generalization design, LOSO_HANDOFF.md).
#: An arm is named by what it has NEVER seen; its adapter is trained on the
#: complementary composite. `L_multi` holds out the multi-site EPISODES while
#: seeing every single site -- the single-site -> multi-site composition test.
LOSO_ARMS = collections.OrderedDict([
    ("L_wiki", "news+shop"),
    ("L_news", "wiki+shop"),
    ("L_shop", "wiki+news"),
    ("L_multi", "wiki+news+shop"),
])
#: The in-distribution reference at the SAME dose and era: `all` (every site,
#: multi included). The retention ratio is L_x / M_all on x's column.
LOSO_REF_ARM = "M_all"
LOSO_REF_UNIT = "all"
#: The era every LOSO corpus is drawn from and every LOSO unit is evaluated on.
#: One era, so the site axis is the only thing that differs between the arms.
LOSO_ERA = 6


def loso_held_out(arm):
    """'L_shop' -> 'shop'; anything else -> None."""
    if arm in LOSO_ARMS:
        return arm[2:]
    return None


def loso_seen_sites(arm):
    """The single sites an arm's corpus contains, as a frozenset (or None)."""
    if arm not in LOSO_ARMS:
        return None
    return frozenset(LOSO_ARMS[arm].split("+"))


def unseen_fraction(task_sites, seen_sites):
    """Fraction of a task's sites the adapter never trained on.

    The x-axis of the OOD gradient: 0 for a task made only of seen sites, 1 for
    a task on a single held-out site, 1/2 and 1/3 for the multi-site tasks that
    mix seen and held-out sites.
    """
    ts = set(task_sites)
    if not ts:
        return None
    return len(ts - set(seen_sites)) / float(len(ts))


_SITE_NAMES = SITE_UNITS + COMPOSITE_SITE_UNITS
_SITE_RE = re.compile(r"^(%s)(?:_v([1-6]+))?(?:_d(\d+))?$"
                      % "|".join(re.escape(s) for s in _SITE_NAMES))


class Site(object):
    """One site, pooled across every UI era. **The domain-axis adapter unit.**

    `Version` pools sites within an era and `Cell` crosses the two; neither can
    ask the orthogonal question. FINDINGS §2 showed era adapters transfer flat
    to every other era -- they learn an output protocol, not interface
    knowledge. Wiki, News and Shop differ in *function* (search-and-read,
    browse-a-feed, search-cart-checkout), not merely in style, so the site axis
    is where a domain claim has something to detect.

    Two things about this unit are load-bearing:

    1. **`multi` is a real unit, not a leftover.** 27 train / 23 test tasks span
       two or three sites, and their teacher episodes are exactly the records
       whose `cell` is None. They are also the LONGEST episodes (~9.3 BC steps
       vs wiki's 6.3), which is why dose has to be matched on samples and not on
       episodes -- see `bcdata.load_cell(max_samples=...)`.
    2. **`draw` is part of the key, not a training seed.** Dose matching selects
       a *subset* of a site's episodes, so the subset itself is a random draw and
       three seeds means three draws. `Site("wiki", draw=1).key == "wiki_d1"`,
       which is what the adapter directory and the YAML are named after; the
       corpus membership rule ignores `draw` entirely.

    Two extensions for the leave-one-site-out design (LOSO_HANDOFF.md), both
    optional and both absent from every pre-existing key:

    3. **Composite sites.** `wiki+news` is the union of the wiki and news
       SINGLE-site episodes (never the multi-site ones), in ENVIRONMENTS order
       -- `canonical_site_key` is the one spelling. `wiki+news+shop` differs
       from `all` by exactly the multi episodes.
    4. **An era restriction.** `eras=(6,)` keeps only era-6 episodes and
       renders as `_v6` in the key (`wiki+news_v6_d1`). It is `_v`, not `_e`,
       because `wiki_e6` is already a Cell key and the grammars must stay
       disjoint. Unrestricted units render no era and keep their old keys.
    """

    __slots__ = ("site", "draw", "_eras")

    granularity = "site"

    def __init__(self, site, draw=None, eras=None):
        if "+" in str(site):
            parts = str(site).split("+")
            if len(set(parts)) < 2 or len(set(parts)) != len(parts):
                raise ValueError("a composite site needs >= 2 DISTINCT "
                                 "environments, got %r" % (site,))
            site = canonical_site_key(parts)
        if site not in _SITE_NAMES:
            raise ValueError("unknown site unit %r (want one of %r or a "
                             "'+'-joined set of %r)"
                             % (site, SITE_UNITS, ENVIRONMENTS))
        self.site = site
        self.draw = None if draw is None else int(draw)
        if eras is None:
            self._eras = ERAS
        else:
            es = tuple(sorted(set(int(e) for e in eras)))
            bad = [e for e in es if e not in ERAS]
            if bad or not es:
                raise ValueError("bad eras %r (want a non-empty subset of %r)"
                                 % (eras, ERAS))
            self._eras = es

    # -- identity ---------------------------------------------------------
    @property
    def key(self):
        k = self.site
        if self._eras != ERAS:
            k += "_v" + "".join(str(e) for e in self._eras)
        if self.draw is not None:
            k += "_d%d" % self.draw
        return k

    @property
    def env(self):
        """The single environment this unit trains on, or None.

        None for `multi`, `all` and the composites, which are not attributable
        to one site -- the same convention `Version.env` uses for "spans
        everything".
        """
        return self.site if self.site in ENVIRONMENTS else None

    @property
    def era(self):
        """None: a Site is an object OF the site axis, whatever eras it draws
        from. Consumers that want the era set read `.eras`; `.era` is kept so
        Cell/Version/Site stay duck-type compatible on `.key`/`.era`."""
        return None

    @property
    def eras(self):
        return self._eras

    @property
    def is_composite(self):
        return "+" in self.site

    @property
    def sites(self):
        """Which environments' episodes this unit's corpus can contain."""
        if self.site in ENVIRONMENTS:
            return (self.site,)
        if self.is_composite:
            return tuple(self.site.split("+"))
        return ENVIRONMENTS

    @property
    def cells(self):
        return tuple(Cell(e, v) for e in self.sites for v in self._eras)

    @property
    def theme(self):
        if len(self._eras) == 1:
            return "era %d only" % self._eras[0]
        return "pooled over eras %s" % ",".join(str(e) for e in self._eras)

    @property
    def year(self):
        return None

    @property
    def is_temporal(self):
        """A Site is not on the temporal axis at all -- it pools it away."""
        return False

    def accepts(self, rec):
        """Does this unit's corpus include a `scan_episodes` record?

        The membership rule is the whole definition of the unit and it lives
        here, once, so `bcdata.load_cell` cannot disagree with the census:

        * era filter first, when restricted: the record's `era` (or its
          cell's) must be one of `self.eras`;
        * `wiki|news|shop` -- single-site episodes, by `rec["cell"].env`;
        * `a+b[+c]`        -- single-site episodes whose env is in the set;
        * `multi`          -- exactly the episodes with `rec["cell"] is None`;
        * `all`            -- everything (the union of the version corpora).
        """
        if self._eras != ERAS:
            era = rec.get("era")
            if era is None and rec.get("cell") is not None:
                era = rec["cell"].era
            if era not in self._eras:
                return False
        if self.site == "all":
            return True
        if self.site == "multi":
            return rec.get("cell") is None
        c = rec.get("cell")
        return c is not None and c.env in self.sites

    def __eq__(self, other):
        return (isinstance(other, Site)
                and (other.site, other.draw, other._eras)
                == (self.site, self.draw, self._eras))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(("site", self.site, self.draw, self._eras))

    def __repr__(self):
        extra = ""
        if self.draw is not None:
            extra += ", draw=%d" % self.draw
        if self._eras != ERAS:
            extra += ", eras=%r" % (self._eras,)
        return "Site(%s%s)" % (self.site, extra)

    def as_dict(self):
        return {"site": self.site, "draw": self.draw, "key": self.key,
                "env": self.env, "sites": list(self.sites),
                "eras": list(self.eras), "composite": self.is_composite}

    @staticmethod
    def parse(key):
        """'wiki' -> Site('wiki'); 'wiki_d1' -> Site('wiki', draw=1);
        'wiki+news_v6_d2' -> Site('wiki+news', draw=2, eras=(6,)).

        A composite head in a non-canonical order ('news+wiki') is accepted
        and canonicalised; the rendered key is always canonical.
        """
        key = key or ""
        head, sep, rest = key.partition("_")
        if "+" in head:
            parts = head.split("+")
            try:
                if len(set(parts)) < 2 or len(set(parts)) != len(parts):
                    raise ValueError("duplicate")
                head = canonical_site_key(parts)
            except ValueError:
                raise ValueError("bad site key %r (a composite is >= 2 "
                                 "distinct environments joined by '+')"
                                 % (key,))
            key = head + sep + rest
        m = _SITE_RE.match(key)
        if not m:
            raise ValueError("bad site key %r (want one of %r or a composite "
                             "of %r, optionally suffixed _v<eras> and/or "
                             "_d<draw>)" % (key, SITE_UNITS, ENVIRONMENTS))
        eras = tuple(int(ch) for ch in m.group(2)) if m.group(2) else None
        return Site(m.group(1), int(m.group(3)) if m.group(3) else None,
                    eras=eras)


ALL_SITES = tuple(Site(s) for s in SITE_UNITS)
SINGLE_SITES = tuple(Site(s) for s in ENVIRONMENTS)


def sites_for(site_keys=None, draw=None, eras=None):
    """Site units, optionally all carrying the same episode draw and the same
    era restriction."""
    keys = SITE_UNITS if site_keys is None else tuple(site_keys)
    return tuple(Site(Site.parse(k).site if isinstance(k, str) else k.site,
                      draw=draw, eras=eras) for k in keys)


def loso_units(draw=None, era=LOSO_ERA):
    """{arm: Site} for the four LOSO arms plus the matched reference, all on
    one era and one draw -- the corpus set one LOSO seed trains."""
    out = collections.OrderedDict()
    for arm, comp in LOSO_ARMS.items():
        out[arm] = Site(comp, draw=draw, eras=(era,))
    out[LOSO_REF_ARM] = Site(LOSO_REF_UNIT, draw=draw, eras=(era,))
    return out


#: The unit an adapter is trained for. 'version' is the default and the one 2
#: describes; 'cell' is only for 6.2's crossed transfer matrix; 'site' is the
#: domain axis (domain_transfer_timewarp_plan.md).
GRANULARITIES = ("version", "cell", "site")
DEFAULT_GRANULARITY = "version"


def versions_for(eras=None):
    return tuple(Version(e) for e in (ERAS if eras is None else tuple(eras)))


def units(granularity=DEFAULT_GRANULARITY, eras=None, envs=None,
          draw=None):
    """The training units at a given granularity.

    >>> len(units("version"))   # 6 adapters, one per era
    6
    >>> len(units("cell"))      # 18, the 6.2 crossing
    18
    >>> len(units("site"))      # 5, the domain axis (incl. multi and all)
    5
    """
    if granularity == "version":
        return versions_for(eras)
    if granularity == "cell":
        return cells_for(envs, eras)
    if granularity == "site":
        return sites_for(envs, draw=draw, eras=eras)
    raise ValueError("unknown granularity %r (want one of %r)"
                     % (granularity, GRANULARITIES))


def parse_unit(key):
    """'v3' -> Version(3); 'wiki_e3' -> Cell('wiki', 3); 'wiki_d1' -> Site.

    The three key grammars are disjoint by construction, so this never has to
    guess: `v<era>`, `<site>_e<era>`, and `<site-unit>[_v<eras>][_d<draw>]`.
    Note that `shop_e3` is a Cell and `shop_d3` is a Site -- one letter apart,
    which is why the site pattern requires the literal `_d` and digits rather
    than accepting any suffix; and a Site's era restriction is spelled `_v6`,
    never `_e6`, for the same reason.
    """
    if re.match(r"^v[1-6]$", key or ""):
        return Version.parse(key)
    if "+" in (key or "") or _SITE_RE.match(key or ""):
        return Site.parse(key)
    return Cell.parse(key)


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------

_TASK_CACHE = {}


def load_tasks(task_data=None):
    """Load the TimeWarp task list. Returns list of dicts (as in test.raw.json)."""
    path = task_data or os.environ.get("TW_TASK_DATA_PATH") or paths.TASK_DATA_LLM_JUDGE
    if not os.path.isabs(path):
        path = os.path.join(paths.TASK_DATA_DIR, path)
    if path not in _TASK_CACHE:
        with open(path) as fh:
            _TASK_CACHE[path] = json.load(fh)
    return _TASK_CACHE[path]


def load_split(split_csv=None):
    """task_id -> 'train'|'test', from the benchmark metadata CSV.

    The `split` key inside the task JSON is present on only 27 records and is
    NOT the real split; the CSV is authoritative.
    """
    path = split_csv or paths.SPLIT_CSV
    out = {}
    # csv.DictReader, NOT line.split(","): the `sites` column is a QUOTED field
    # that contains commas for multi-site tasks ("news,wiki"). Splitting naively
    # misaligns every column after it, so all 50 multi-site tasks silently fell
    # out of the split map -- 23 of 103 test and 27 of 128 train. Single-site
    # rows parsed fine, which is why it went unnoticed: anything asking for
    # single_site=True got the right answer.
    with open(path) as fh:
        for row in csv.DictReader(fh):
            name = row.get("task_name") or ""
            m = re.search(r"timewarp\.(\d+)", name)
            if not m and row.get("task_id"):
                try:
                    tid = int(str(row["task_id"]).strip())
                except ValueError:
                    continue
            elif m:
                tid = int(m.group(1))
            else:
                continue
            sp = (row.get("browsergym_split") or "").strip()
            if sp:
                out[tid] = sp
    return out


def task_sites(task):
    """Normalised site names for a task record, e.g. {'wiki', 'shop'}."""
    return set(SITE_KEY_INV.get(s, s) for s in task.get("sites", []))


def tasks_for_env(env, split=None, single_site=True, task_data=None):
    """Task ids that exercise `env`.

    single_site=True keeps only tasks whose *only* site is `env`. That matters
    for the 6.2 dissociation: a task spanning wiki+news is not attributable to
    one cell, so cross-applying a wiki adapter to it confounds the comparison.
    """
    splits = load_split() if split else None
    out = []
    for t in load_tasks(task_data):
        sites = task_sites(t)
        if single_site:
            if sites != set([env]):
                continue
        elif env not in sites:
            continue
        tid = t["task_id"]
        if split and splits.get(tid) != split:
            continue
        out.append(tid)
    return sorted(out)


def site_group(task):
    """Which eval group a task record belongs to: wiki|news|shop|multi.

    This is the post-hoc grouping the site matrix's COLUMNS are built from.
    Every test unit already runs all 103 tasks, so one eval unit yields the
    whole row of the matrix and the grouping costs nothing at run time
    (domain_transfer_timewarp_plan.md 1.1).
    """
    s = task_sites(task)
    if len(s) >= 2:
        return "multi"
    if len(s) == 1:
        only = sorted(s)[0]
        if only in ENVIRONMENTS:
            return only
    return None


def task_sites_map(split=None, task_data=None):
    """task_id -> frozenset of normalised site names, for every task.

    The raw material of the OOD gradient: the fraction of a task's sites an
    adapter never saw is computed from this and the arm's seen set, so the
    grouping is per ARM and post hoc, never a filter on the run.
    """
    splits = load_split() if split else None
    out = {}
    for t in load_tasks(task_data):
        s = frozenset(task_sites(t))
        if not s:
            continue
        tid = t["task_id"]
        if split and splits.get(tid) != split:
            continue
        out[tid] = s
    return out


def multi_subgroup(task_or_sites):
    """'wiki+news' for a 2-3 site task; None for a single-site task.

    Same spelling as the composite training units (`canonical_site_key`), so
    the subgroup a task belongs to and the corpus an arm trained on can be
    compared as strings.
    """
    s = (task_sites(task_or_sites) if isinstance(task_or_sites, dict)
         else set(task_or_sites))
    if len(s) < 2:
        return None
    return canonical_site_key(s)


def multi_subgroup_sizes(task_data=None):
    """{subgroup: {"train": n, "test": n}} over the multi-site tasks.

    Measured 2026-09-09: wiki+news 10/9, wiki+news+shop 8/6, news+shop 6/3,
    wiki+shop 3/5 (train/test). Small on purpose to report; pooled over the
    three LOSO arms and three seeds in the gradient, never read per cell.
    """
    splits = load_split()
    out = collections.OrderedDict()
    for t in load_tasks(task_data):
        g = multi_subgroup(t)
        if g is None:
            continue
        d = out.setdefault(g, {"train": 0, "test": 0, "other": 0})
        sp = splits.get(t["task_id"])
        d[sp if sp in ("train", "test") else "other"] += 1
    return out


def task_site_group_map(split=None, task_data=None):
    """task_id -> 'wiki'|'news'|'shop'|'multi' for every task.

    Unlike `task_cell_map` this keeps the multi-site tasks: on the domain axis
    they are a group of their own (the composition test, 2.5), not noise to be
    dropped.
    """
    splits = load_split() if split else None
    out = {}
    for t in load_tasks(task_data):
        g = site_group(t)
        if g is None:
            continue
        tid = t["task_id"]
        if split and splits.get(tid) != split:
            continue
        out[tid] = g
    return out


def tasks_for_site_group(group, split=None, task_data=None):
    """Task ids in one site group, e.g. tasks_for_site_group('multi', 'test')."""
    if group not in SITE_GROUPS:
        raise ValueError("unknown site group %r (want one of %r)"
                         % (group, SITE_GROUPS))
    m = task_site_group_map(split=split, task_data=task_data)
    return sorted(tid for tid, g in m.items() if g == group)


def site_group_sizes(task_data=None):
    """{group: {"train": n, "test": n, "total": n}} -- the 1.1 table.

    Measured 2026-09-09: wiki 39/31, news 25/22, shop 37/27, multi 27/23,
    total 128/103. Regenerated rather than hard-coded so a task-set change
    shows up as a different table instead of a silently stale one.
    """
    splits = load_split()
    out = collections.OrderedDict()
    for g in SITE_GROUPS:
        out[g] = {"train": 0, "test": 0, "total": 0, "other": 0}
    for t in load_tasks(task_data):
        g = site_group(t)
        if g is None:
            continue
        sp = splits.get(t["task_id"])
        out[g]["total"] += 1
        if sp in ("train", "test"):
            out[g][sp] += 1
        else:
            out[g]["other"] += 1
    return out


def cell_tasks(cell, split=None, single_site=True, task_data=None):
    """Tasks belonging to a cell. Tasks are version-agnostic, so this is
    entirely determined by the cell's environment."""
    return tasks_for_env(cell.env, split=split, single_site=single_site,
                         task_data=task_data)


def task_cell_map(era, single_site=True, task_data=None):
    """task_id -> Cell for one era (only unambiguous, single-site tasks)."""
    out = {}
    for t in load_tasks(task_data):
        sites = task_sites(t)
        if single_site and len(sites) != 1:
            continue
        site = sorted(sites)[0] if sites else None
        if site in ENVIRONMENTS:
            out[t["task_id"]] = Cell(site, era)
    return out


# --------------------------------------------------------------------------
# Splits over cells (adapterCL.md 6.1)
# --------------------------------------------------------------------------

class Split(object):
    """A train/test partition of eras, applied to every environment."""

    __slots__ = ("name", "train_eras", "test_eras", "kind", "note")

    def __init__(self, name, train_eras, test_eras, kind, note=""):
        self.name = name
        self.train_eras = tuple(train_eras)
        self.test_eras = tuple(test_eras)
        self.kind = kind          # 'extrapolation' | 'interpolation' | 'neutral'
        self.note = note

    def train_cells(self, envs=None):
        return cells_for(envs, self.train_eras)

    def test_cells(self, envs=None):
        return cells_for(envs, self.test_eras)

    def as_dict(self):
        return {"name": self.name, "train_eras": list(self.train_eras),
                "test_eras": list(self.test_eras), "kind": self.kind,
                "note": self.note}

    def __repr__(self):
        return "Split(%s: train=%r test=%r)" % (self.name, self.train_eras,
                                                self.test_eras)


#: Headline + folds. Built over TEMPORAL_ERAS only; era 6 gets its own split
#: because generalising to a *style-neutral* interface is a different question
#: from extrapolating along the era axis.
SPLITS = collections.OrderedDict()


def _reg(split):
    SPLITS[split.name] = split
    return split


_reg(Split("forward", (1, 2, 3), (4, 5), "extrapolation",
           "headline: train on old eras, predict newer designs"))
_reg(Split("backward", (3, 4, 5), (1, 2), "extrapolation",
           "symmetric check: train on new eras, predict older designs"))
_reg(Split("interp", (1, 2, 5), (3, 4), "interpolation",
           "should be easier than extrapolation; if not, the encoder is broken"))
_reg(Split("fold_a", (1, 2, 4), (3, 5), "extrapolation", "rotation fold A"))
_reg(Split("fold_b", (2, 3, 5), (1, 4), "extrapolation", "rotation fold B"))
_reg(Split("fold_c", (1, 3, 5), (2, 4), "interpolation", "rotation fold C"))
_reg(Split("neutral_holdout", (1, 2, 3, 4, 5), (6,), "neutral",
           "can the generator handle a style-neutral interface it never saw?"))
_reg(Split("all", ERAS, (), "neutral", "no holdout; for oracle / sanity runs"))

HEADLINE_SPLIT = "forward"
#: The >=3 folds 6.1 asks for.
ROTATION_FOLDS = ("forward", "backward", "fold_a", "fold_b", "fold_c")

#: Sequential-arrival stream for the secondary CL setting (2). Temporal order,
#: neutral era appended last so it cannot masquerade as "the future".
SEQUENTIAL_STREAM = tuple(list(TEMPORAL_ERAS) + [NEUTRAL_ERA])


def split_by_name(name):
    if name not in SPLITS:
        raise KeyError("unknown split %r (have %r)" % (name, list(SPLITS)))
    return SPLITS[name]


# --------------------------------------------------------------------------
# BC data availability (TimeTraj rollouts on disk)
# --------------------------------------------------------------------------

EPISODE_RE = re.compile(r"timewarp\.(\d+)_(\d+)$")


def timetraj_dir(era):
    return os.path.join(paths.TIMETRAJ, "version_%d" % era)


def scan_episodes(era, root=None):
    """Enumerate TimeTraj episodes for one era.

    Returns a list of dicts: {dir, era, task_id, seed, reward, n_steps, cell}.
    `cell` is None for multi-site tasks (not attributable to one environment).

    `era` is on every record because the Site unit pools across eras: a caller
    that has a mixed list can no longer recover the era from the unit it asked
    for, and re-deriving it from the path (`version_(\d+)`) in the consumer is
    the sort of duplicated parse that drifts.
    """
    root = root or timetraj_dir(era)
    if not os.path.isdir(root):
        return []
    tmap = task_cell_map(era)
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        m = EPISODE_RE.search(name)
        if not m:
            continue
        tid, seed = int(m.group(1)), int(m.group(2))
        reward, n_steps = 0.0, 0
        si = os.path.join(d, "summary_info.json")
        if os.path.exists(si):
            try:
                with open(si) as fh:
                    info = json.load(fh)
                reward = float(info.get("cum_reward") or 0.0)
                n_steps = int(info.get("n_steps") or 0)
            except (ValueError, IOError):
                pass
        td = os.path.join(d, "training_data")
        n_dumps = 0
        if os.path.isdir(td):
            n_dumps = len([f for f in os.listdir(td)
                           if f.startswith("agent_output_step_") and f.endswith(".json")])
        out.append({"dir": d, "era": era, "task_id": tid, "seed": seed,
                    "reward": reward, "n_steps": n_steps, "n_dumps": n_dumps,
                    "cell": tmap.get(tid)})
    return out


def version_availability(eras=None, min_episodes=20):
    """Per-version BC census — the census that matters for the default unit.

    Unlike `availability()`, a multi-site task counts here: pooling across
    environments means a wiki+news task is attributable to the *version*, which
    is the whole point of training at this granularity.
    """
    eras = eras or ERAS
    stats = collections.OrderedDict()
    missing = []
    for v in versions_for(eras):
        eps = scan_episodes(v.era)
        verified = [e for e in eps if e["reward"] >= 1.0]
        stats[v.key] = {
            "era": v.era,
            "episodes": len(eps),
            "verified": len(verified),
            "steps": sum(e["n_dumps"] for e in eps),
            "attributable_cells": len([e for e in eps if e["cell"] is not None]),
        }
        stats[v.key]["ok"] = len(verified) >= min_episodes
        if not stats[v.key]["ok"]:
            missing.append(v.key)
    stats["_missing"] = missing
    stats["_min_episodes"] = min_episodes
    return stats


def site_availability(min_episodes=20, root=None, units=None):
    """Per-SITE BC census, pooled over eras -- the domain-axis unit's census.

    Counts every episode, including the multi-site ones: on this axis they are
    the `multi` unit rather than "unattributable". Reports `steps` because dose
    is matched on SAMPLES, not episodes (multi averages ~9.3 steps per episode
    against wiki's ~6.3, so equal episode counts would hand `multi` 1.6x the
    dose -- 7's first trap).

    `units` (default `ALL_SITES`) may be any Site units, composite and/or
    era-restricted; the census then scans only the eras those units draw from
    and `_matched_samples` is the smallest verified corpus among THEM -- the
    dose every arm of that set can be matched at. With the default it stays the
    smallest single-site corpus, as before.
    """
    units = tuple(ALL_SITES if units is None else units)
    eras = tuple(sorted(set(e for u in units for e in u.eras)))
    stats = collections.OrderedDict()
    for s in units:
        stats[s.key] = {"site": s.site, "episodes": 0, "verified": 0,
                        "steps": 0, "verified_steps": 0,
                        "eras": list(s.eras),
                        "per_era": collections.OrderedDict(
                            (e, {"episodes": 0, "steps": 0}) for e in s.eras)}
    missing = []
    for era in eras:
        for rec in scan_episodes(era, root=root):
            ver = (rec["reward"] is not None and float(rec["reward"]) >= 1.0)
            for s in units:
                if not s.accepts(rec):
                    continue
                st = stats[s.key]
                st["episodes"] += 1
                st["steps"] += rec["n_dumps"]
                st["per_era"][era]["episodes"] += 1
                st["per_era"][era]["steps"] += rec["n_dumps"]
                if ver:
                    st["verified"] += 1
                    st["verified_steps"] += rec["n_dumps"]
    for key, st in stats.items():
        st["steps_per_episode"] = (float(st["steps"]) / st["episodes"]
                                   if st["episodes"] else 0.0)
        st["ok"] = st["verified"] >= min_episodes
        if not st["ok"]:
            missing.append(key)
    stats["_missing"] = missing
    stats["_min_episodes"] = min_episodes
    stats["_eras"] = list(eras)
    #: The smallest corpus in verified SAMPLES among the units that can be
    #: matched: with the default unit set, the single sites (news, 771 -> the
    #: plan's 770); with an explicit set, every unit in it.
    if units == ALL_SITES:
        pool = [stats[s]["verified_steps"] for s in ENVIRONMENTS]
    else:
        pool = [stats[s.key]["verified_steps"] for s in units]
    stats["_matched_samples"] = min(pool) if pool else 0
    return stats


def availability(eras=None, min_episodes=20):
    """Per-cell BC-data census over TimeTraj-Trajectories.

    Returns {cell_key: {"episodes", "verified", "steps", "ok"}} plus a
    "_missing" list of cells that fall below `min_episodes`. Phase 2 needs all
    18 cells; era 1 is known to be short (30 episodes total, none wiki-only).
    """
    eras = eras or ERAS
    stats = collections.OrderedDict()
    for c in cells_for(None, eras):
        stats[c.key] = {"env": c.env, "era": c.era, "episodes": 0,
                        "verified": 0, "steps": 0}
    for era in eras:
        for ep in scan_episodes(era):
            if ep["cell"] is None:
                continue
            s = stats[ep["cell"].key]
            s["episodes"] += 1
            s["steps"] += ep["n_dumps"]
            if ep["reward"] >= 1.0:
                s["verified"] += 1
    missing = []
    for k, s in stats.items():
        s["ok"] = s["verified"] >= min_episodes
        if not s["ok"]:
            missing.append(k)
    stats["_missing"] = missing
    stats["_min_episodes"] = min_episodes
    return stats


def summarize(fh=None):
    """Print the grid + availability census. Used by `python -m adaptercl.cli cells`."""
    import sys
    fh = fh or sys.stdout
    print("adapterCL cell grid: %d environments x %d eras = %d cells"
          % (len(ENVIRONMENTS), len(ERAS), len(ALL_CELLS)), file=fh)
    print("", file=fh)
    hdr = "%-6s | " % "era" + " | ".join("%-22s" % e for e in ENVIRONMENTS) + " | spread"
    print(hdr, file=fh)
    print("-" * len(hdr), file=fh)
    for era in ERAS:
        row = "%-6s | " % era
        cols = []
        for env in ENVIRONMENTS:
            y = NOMINAL_YEAR[env][era]
            cols.append("%-22s" % ("%s (%s)" % (THEME_NAME[env][era],
                                                "neutral" if y is None else y)))
        spread = ERA_YEAR_SPREAD[era]
        row += " | ".join(cols) + " | " + ("n/a" if spread is None else "%.1f yr" % spread)
        print(row, file=fh)
    print("", file=fh)
    print("splits:", file=fh)
    for name, sp in SPLITS.items():
        print("  %-16s %-14s train=%s test=%s  %s"
              % (name, sp.kind, sp.train_eras, sp.test_eras, sp.note), file=fh)
    print("", file=fh)
    print("BC data by VERSION -- the default adapter unit (2):", file=fh)
    vav = version_availability()
    print("  %-6s %9s %9s %8s" % ("unit", "episodes", "verified", "steps"), file=fh)
    for k, s in vav.items():
        if k.startswith("_"):
            continue
        print("  %-6s %9d %9d %8d%s"
              % (k, s["episodes"], s["verified"], s["steps"],
                 "" if s["ok"] else "   <-- SHORT"), file=fh)
    if vav["_missing"]:
        print("  %d version(s) below %d verified episodes: %s"
              % (len(vav["_missing"]), vav["_min_episodes"],
                 ", ".join(vav["_missing"])), file=fh)
    else:
        print("  every version has enough data to train an adapter.", file=fh)

    print("", file=fh)
    print("BC data by SITE -- the DOMAIN-axis unit "
          "(domain_transfer_timewarp_plan.md):", file=fh)
    sav = site_availability()
    print("  %-8s %9s %9s %8s %9s %7s" % ("unit", "episodes", "verified",
                                          "steps", "ver.steps", "st/ep"),
          file=fh)
    for k, s in sav.items():
        if k.startswith("_"):
            continue
        print("  %-8s %9d %9d %8d %9d %7.2f%s"
              % (k, s["episodes"], s["verified"], s["steps"],
                 s["verified_steps"], s["steps_per_episode"],
                 "" if s["ok"] else "   <-- SHORT"), file=fh)
    tsz = site_group_sizes()
    print("  tasks per group (train/test): "
          + ", ".join("%s %d/%d" % (g, tsz[g]["train"], tsz[g]["test"])
                      for g in SITE_GROUPS), file=fh)
    print("  dose-matched corpus size: %d verified samples (the smallest "
          "single-site corpus)" % sav["_matched_samples"], file=fh)

    print("", file=fh)
    print("BC data by CELL -- needed only for 6.2's crossed transfer matrix:",
          file=fh)
    av = availability()
    print("  %-10s %9s %9s %8s" % ("cell", "episodes", "verified", "steps"), file=fh)
    for k, s in av.items():
        if k.startswith("_"):
            continue
        flag = "" if s["ok"] else "   <-- SHORT"
        print("  %-10s %9d %9d %8d%s"
              % (k, s["episodes"], s["verified"], s["steps"], flag), file=fh)
    if av["_missing"]:
        print("\n  %d cell(s) below %d verified episodes: %s"
              % (len(av["_missing"]), av["_min_episodes"], ", ".join(av["_missing"])),
              file=fh)
    return av


if __name__ == "__main__":
    summarize()
