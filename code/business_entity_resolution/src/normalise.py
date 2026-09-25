"""Shared normalisation for names and addresses (used by blocking, features and inference).

Deterministic and vectorised with polars; the only per-string Python call is anyascii
transliteration, applied once per distinct non-ASCII string.

Country-aware: rules are looked up by the lower-cased country label. Countries without a
table of their own (anything new in test) get the general rules only, never a US/India default.

Names   -> full_name (legal suffix canonicalised), core_name (legal suffix removed), legal (the suffix)
Address -> addr_norm, city, state, dept (France departement), numbers (digit tokens, leading zeros stripped)
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import polars as pl
from anyascii import anyascii

# ---------------------------------------------------------------------------- basic cleaning


def transliterate(s: pl.Series) -> pl.Series:
    """anyascii for non-ASCII strings (Devanagari, Tamil, accents, ...); ASCII strings untouched."""
    s = s.fill_null("")
    mask = s.str.contains(r"[^\x00-\x7F]")
    uniq = s.filter(mask).unique().to_list()
    if not uniq:
        return s
    return s.replace({u: anyascii(u) for u in uniq})


def _clean(e: pl.Expr, drop_dots: bool) -> pl.Expr:
    """Lowercase, & -> and, punctuation -> space, collapse whitespace. For names, dots and
    apostrophes are deleted (l.l.c. -> llc, nash's -> nashs); for addresses they become spaces
    (no.330 -> no 330)."""
    e = e.str.to_lowercase().str.replace_all("&", " and ")
    if drop_dots:
        e = e.str.replace_all(r"[.'`]", "")
    return e.str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()


def clean_text(text: str, drop_dots: bool = False) -> str:
    """Scalar version of transliterate + _clean (for building lookup tables and tests)."""
    s = anyascii(text).lower().replace("&", " and ")
    if drop_dots:
        s = re.sub(r"[.'`]", "", s)
    return " ".join(re.findall(r"[a-z0-9]+", s))


def _key(country: str | None) -> str:
    return (country or "").strip().lower()


# ---------------------------------------------------------------------------- name rules

# Legal suffixes: token -> canonical token. Only a trailing run of these is treated as a suffix.
LEGAL_GENERAL = {
    "inc": "inc", "incorporated": "inc", "incorporation": "inc",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc", "psc": "psc",
    "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "cos": "co",
    "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "opc": "opc",
    "gmbh": "gmbh", "bv": "bv", "srl": "srl", "spa": "spa",
    # French forms that are unambiguous as a trailing token anywhere
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "selarl": "selarl", "eirl": "eirl", "scop": "scop",
}
LEGAL_BY_COUNTRY = {
    "india": {
        # transliterations seen in S2/S3 (Devanagari / Tamil -> anyascii), e.g. "pra li" = "pvt ltd"
        "praivet": "pvt", "praibhet": "pvt", "piraivet": "pvt", "praivett": "pvt", "prayvet": "pvt", "pra": "pvt",
        "limitet": "ltd", "limited": "ltd", "limitedd": "ltd", "li": "ltd", "elelpi": "llp", "ellpi": "llp",
    },
    "france": {"sa": "sa", "sci": "sci", "snc": "snc", "ei": "ei", "sca": "sca", "scp": "scp", "gie": "gie", "scm": "scm"},
}
# Legal forms that also appear as a LEADING token after word reordering ("LLC Value Electronics").
# Only unambiguous abbreviations: never words like "company" or "limited", or short forms like "co"/"sa".
LEGAL_LEADING = {"inc", "llc", "llp", "ltd", "pvt", "corp", "plc", "pllc", "gmbh", "sarl", "sas", "sasu", "eurl", "snc"}
# multi-token spellings, collapsed before suffix detection (anchored at the end of the name)
LEGAL_MULTI = [(r" l l c$", " llc"), (r" l l p$", " llp"), (r" p l l c$", " pllc"), (r" l p$", " lp"), (r" p c$", " pc")]

# Honorific prefixes (repeated prefixes are all removed): India only, where they are noise added to names.
HONORIFICS_BY_COUNTRY = {
    "india": ["m s", "ms", "messrs", "smt", "shrimati", "shri", "shree", "sri", "sree", "kumari", "km", "mr", "mrs", "dr"],
}

# Word abbreviations, all countries (token -> expansion)
NAME_ABBREV_GENERAL = {
    "intl": "international", "mfg": "manufacturing", "mfrs": "manufacturers",
    "svc": "services", "svcs": "services", "srvcs": "services", "mgmt": "management", "mgt": "management",
    "assn": "association", "assoc": "associates", "dept": "department", "natl": "national",
    "govt": "government", "univ": "university", "hosp": "hospital", "ctr": "center", "cntr": "center",
    "centre": "center", "bros": "brothers", "engg": "engineering", "engr": "engineering", "grp": "group",
    "hldgs": "holdings", "inds": "industries", "sys": "systems", "solns": "solutions",
}
NAME_ABBREV_BY_COUNTRY = {"france": {"et": "and", "cie": "company"}}


# ---------------------------------------------------------------------------- address rules

ADDR_ABBREV_GENERAL = {
    "rd": "road", "ave": "avenue", "blvd": "boulevard", "hwy": "highway", "pkwy": "parkway",
    "bldg": "building", "apt": "apartment", "flr": "floor",
}
ADDR_ABBREV_BY_COUNTRY = {
    "us": {
        "st": "street", "str": "street", "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle", "pl": "place",
        "sq": "square", "ste": "suite", "fl": "floor", "ter": "terrace", "terr": "terrace", "trl": "trail",
        "pt": "point", "mt": "mount", "ft": "fort", "hts": "heights", "rte": "route", "fwy": "freeway",
        "expy": "expressway", "cres": "crescent",
    },
    "india": {"nr": "near", "opp": "opposite", "fl": "floor", "ngr": "nagar", "bldg": "building"},
    "france": {
        "st": "saint", "ste": "sainte", "av": "avenue", "bd": "boulevard", "bld": "boulevard", "bvd": "boulevard",
        "boul": "boulevard", "rte": "route", "chem": "chemin", "pl": "place", "imp": "impasse", "fbg": "faubourg",
        "sq": "square", "all": "allee", "r": "rue",
    },
}
LANDMARK_PREFIX = r"^(near|nr|opp|opposite|behind|beside|next to|adjacent to|in front of)\b"
COUNTRY_WORDS = {"india", "bharat", "usa", "us", "united states", "united states of america", "america", "france", "null", "none", "na"}
CITY_PREFIX = r"^(city of|town of|village of|township of|borough of|ville de)\s+"
CITY_SUFFIX = r"\s+(city|township|town|village|borough|cdp)$"

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia", "pr": "puerto rico", "gu": "guam",
    "vi": "virgin islands", "as": "american samoa", "mp": "northern mariana islands",
}
# canonical = full state name; value = codes and alternative spellings
INDIA_STATES = {
    "andhra pradesh": ["ap"], "arunachal pradesh": ["ar"], "assam": ["as"], "bihar": ["br"],
    "chhattisgarh": ["cg", "ct", "chattisgarh"], "goa": ["ga"], "gujarat": ["gj", "gujrat"],
    "haryana": ["hr"], "himachal pradesh": ["hp"], "jharkhand": ["jh"], "karnataka": ["ka"],
    "kerala": ["kl", "keralam"], "madhya pradesh": ["mp"], "maharashtra": ["mh"], "manipur": ["mn"],
    "meghalaya": ["ml"], "mizoram": ["mz"], "nagaland": ["nl"], "odisha": ["od", "or", "orissa"],
    "punjab": ["pb"], "rajasthan": ["rj"], "sikkim": ["sk"], "tamil nadu": ["tn", "tamilnadu"],
    "telangana": ["tg", "ts"], "tripura": ["tr"], "uttar pradesh": ["up"],
    "uttarakhand": ["uk", "ua", "ut", "uttaranchal"], "west bengal": ["wb"],
    "andaman and nicobar islands": ["an", "andaman and nicobar"], "chandigarh": ["ch"],
    "dadra and nagar haveli and daman and diu": ["dn", "dd", "dadra and nagar haveli", "daman and diu"],
    "delhi": ["dl", "nct of delhi", "national capital territory of delhi"], "jammu and kashmir": ["jk"],
    "ladakh": ["la"], "lakshadweep": ["ld"], "puducherry": ["py", "pondicherry"],
}
# canonical = current region; value = old regions / alternative spellings
FRANCE_REGIONS = {
    "auvergne rhone alpes": ["auvergne", "rhone alpes"], "bourgogne franche comte": ["bourgogne", "franche comte"],
    "bretagne": ["brittany"], "centre val de loire": ["centre"], "corse": ["corsica"],
    "grand est": ["alsace", "lorraine", "champagne ardenne"],
    "hauts de france": ["nord pas de calais", "picardie"], "ile de france": [],
    "normandie": ["basse normandie", "haute normandie", "normandy"],
    "nouvelle aquitaine": ["aquitaine", "limousin", "poitou charentes"],
    "occitanie": ["languedoc roussillon", "midi pyrenees"], "pays de la loire": [],
    "provence alpes cote d azur": ["paca", "provence alpes cote dazur"],
    "guadeloupe": [], "martinique": [], "guyane": ["french guiana"], "la reunion": ["reunion"], "mayotte": [],
}
# Departements (Paris omitted: as an address component it is the city)
FRANCE_DEPTS = [
    "ain", "aisne", "allier", "alpes de haute provence", "hautes alpes", "alpes maritimes", "ardeche", "ardennes",
    "ariege", "aube", "aude", "aveyron", "bouches du rhone", "calvados", "cantal", "charente", "charente maritime",
    "cher", "correze", "corse du sud", "haute corse", "cote d or", "cotes d armor", "creuse", "dordogne", "doubs",
    "drome", "eure", "eure et loir", "finistere", "gard", "haute garonne", "gers", "gironde", "herault",
    "ille et vilaine", "indre", "indre et loire", "isere", "jura", "landes", "loir et cher", "loire",
    "haute loire", "loire atlantique", "loiret", "lot", "lot et garonne", "lozere", "maine et loire", "manche",
    "marne", "haute marne", "mayenne", "meurthe et moselle", "meuse", "morbihan", "moselle", "nievre", "nord",
    "oise", "orne", "pas de calais", "puy de dome", "pyrenees atlantiques", "hautes pyrenees",
    "pyrenees orientales", "bas rhin", "haut rhin", "rhone", "haute saone", "saone et loire", "sarthe",
    "savoie", "haute savoie", "seine maritime", "seine et marne", "yvelines", "deux sevres", "somme", "tarn",
    "tarn et garonne", "var", "vaucluse", "vendee", "vienne", "haute vienne", "vosges", "yonne",
    "territoire de belfort", "essonne", "hauts de seine", "seine saint denis", "val de marne", "val d oise",
]


def _state_table(country: str) -> dict[str, str]:
    """Hand-written variant -> canonical state/region, keys cleaned like address components."""
    t: dict[str, str] = {}
    if country == "us":
        for code, name in US_STATES.items():
            t[code] = code
            t[clean_text(name)] = code
    elif country == "india":
        for name, alts in INDIA_STATES.items():
            t[clean_text(name)] = name
            for a in alts:
                t[clean_text(a)] = name
    elif country == "france":
        for name, alts in FRANCE_REGIONS.items():
            t[clean_text(name)] = name
            for a in alts:
                t[clean_text(a)] = name
    return t


# ---------------------------------------------------------------------------- normaliser


class Normaliser:
    """Holds the lookup tables. `state_aliases` ({country: {variant: canonical}}) are learned from
    training pairs with learn_state_aliases() and added on top of the hand-written tables."""

    def __init__(self, state_aliases: Mapping[str, Mapping[str, str]] | None = None):
        self.state_aliases = {_key(c): dict(m) for c, m in (state_aliases or {}).items()}

    @classmethod
    def load(cls, path: str | Path | None) -> "Normaliser":
        if path and Path(path).exists():
            return cls(json.loads(Path(path).read_text()))
        return cls()

    def states(self, country: str) -> dict[str, str]:
        return {**_state_table(country), **self.state_aliases.get(country, {})}

    # ------------------------------------------------------------------ names
    def names(self, names: pl.Series, countries: pl.Series) -> pl.DataFrame:
        df = pl.DataFrame({"n": transliterate(names), "country": countries.cast(pl.String).fill_null("")})
        df = df.with_row_index("rid").with_columns(_clean(pl.col("n"), drop_dots=True).alias("n"))
        parts = [self._names_one(g, key) for key, g in df.group_by(pl.col("country").str.to_lowercase().str.strip_chars())]
        out = pl.concat(parts) if parts else df.select("rid").with_columns(
            full_name=pl.lit(""), core_name=pl.lit(""), legal=pl.lit(""))
        return out.sort("rid").drop("rid")

    def _names_one(self, g: pl.DataFrame, key: tuple) -> pl.DataFrame:
        country = key[0]
        e = pl.col("n")
        hon = HONORIFICS_BY_COUNTRY.get(country)
        if hon:
            stripped = e.str.replace(r"^(?:(?:" + "|".join(hon) + r")\s+)+", "")
            e = pl.when(stripped != "").then(stripped).otherwise(e)
        abbrev = {**NAME_ABBREV_GENERAL, **NAME_ABBREV_BY_COUNTRY.get(country, {})}
        single = {k: v for k, v in abbrev.items() if " " not in k}
        e = e.str.split(" ").list.eval(pl.element().replace(single)).list.join(" ")
        for k, v in abbrev.items():
            if " " in k:
                e = e.str.replace_all(rf"\b{k}\b", v)
        for pat, rep in LEGAL_MULTI:
            e = e.str.replace(pat, rep)
        legal = {**LEGAL_GENERAL, **LEGAL_BY_COUNTRY.get(country, {})}
        alt = "|".join(sorted(map(re.escape, legal), key=len, reverse=True))
        pat = r"^(.*?)((?: (?:" + alt + r"))*)$"
        lead_pat = r"^((?:(?:" + "|".join(sorted(LEGAL_LEADING)) + r") )+)(.+)$"
        g = g.with_columns(e.alias("_e")).with_columns(
            pl.col("_e").str.extract(lead_pat, 1).fill_null("").alias("_lead"),
            pl.coalesce(pl.col("_e").str.extract(lead_pat, 2), pl.col("_e")).alias("_rest"),
        )
        g = g.with_columns((pl.lit(" ") + pl.col("_rest")).alias("_s"))
        g = g.with_columns(
            pl.col("_s").str.extract(pat, 1).str.strip_chars().alias("core_name"),
            pl.concat_str([pl.col("_lead"), pl.col("_s").str.extract(pat, 2)], separator=" ")
            .str.strip_chars().str.split(" ").list.eval(pl.element().replace(legal)).list.join(" ")
            .str.replace_all(r"\s+", " ").str.strip_chars().alias("legal"),
        )
        # a name that is only suffix tokens ("Company") keeps them as its core
        g = g.with_columns(
            pl.when(pl.col("core_name") == "").then(pl.col("_e")).otherwise(pl.col("core_name")).alias("core_name"),
            pl.when(pl.col("core_name") == "").then(pl.lit("")).otherwise(pl.col("legal")).alias("legal"),
        )
        full = pl.concat_str([pl.col("core_name"), pl.col("legal")], separator=" ").str.strip_chars()
        return g.select("rid", full.alias("full_name"), "core_name", "legal")

    # ------------------------------------------------------------------ addresses
    def addresses(self, addrs: pl.Series, countries: pl.Series, city_vocab: pl.DataFrame | None = None) -> pl.DataFrame:
        """city_vocab: (country, cand, freq) from fit_city_vocab(); when None it is fitted on these
        addresses. The city is the most frequent city-like component, so reordered addresses agree."""
        df = pl.DataFrame({"a": transliterate(addrs), "country": countries.cast(pl.String).fill_null("")})
        df = df.with_row_index("rid").with_columns(pl.col("country").str.to_lowercase().str.strip_chars().alias("ckey"))
        comps = self._components(df)
        vocab = city_vocab if city_vocab is not None else fit_city_vocab(comps)
        cities = (
            comps.filter(pl.col("kind") == "cand")
            .join(vocab, on=["ckey", "cand"], how="left")
            .group_by("rid")
            .agg(pl.col("cand").sort_by([pl.col("freq").fill_null(0), pl.col("pos")], descending=[True, True]).first().alias("city"))
        )
        per = comps.group_by("rid").agg(
            pl.col("c").sort_by("pos").str.join(" ").alias("addr_norm"),
            pl.col("state").drop_nulls().last().alias("state"),
            pl.col("dept").drop_nulls().last().alias("dept"),
        )
        out = (
            df.select("rid", "a")
            .join(per, on="rid", how="left")
            .join(cities, on="rid", how="left")
            .with_columns(
                pl.col("addr_norm").fill_null(""),
                _clean(pl.col("a"), drop_dots=False).str.extract_all(r"\d+")
                .list.eval(pl.element().str.replace(r"^0+(\d)", "$1")).list.unique(maintain_order=True).alias("numbers"),
            )
        )
        return out.sort("rid").select("addr_norm", "city", "state", "dept", "numbers")

    def _components(self, df: pl.DataFrame) -> pl.DataFrame:
        comps = (
            df.select("rid", "ckey", pl.col("a").str.split(",").alias("c"))
            .with_columns(pl.int_ranges(pl.col("c").list.len()).alias("pos"))
            .explode("c", "pos", empty_as_null=True)
            .with_columns(_clean(pl.col("c"), drop_dots=False).alias("c"))
            .filter(pl.col("c").is_not_null() & (pl.col("c") != ""))
        )
        parts = []
        for (ckey,), g in comps.group_by("ckey"):
            abbrev = {**ADDR_ABBREV_GENERAL, **ADDR_ABBREV_BY_COUNTRY.get(ckey, {})}
            states = self.states(ckey)
            depts = {clean_text(d): clean_text(d) for d in FRANCE_DEPTS} if ckey == "france" else {}
            # States first, on the cleaned component: US codes like "ct", "fl", "mt" would otherwise be
            # expanded to court / floor / mount. A state may carry a trailing postcode ("nc 27260").
            bare = pl.col("c").str.replace(r"(\s\d{5,6})+$", "")
            g = g.with_columns(
                bare.replace_strict(states, default=None, return_dtype=pl.String).alias("state"),
                bare.replace_strict(depts, default=None, return_dtype=pl.String).alias("dept"),
            )
            expanded = pl.col("c").str.split(" ").list.eval(pl.element().replace(abbrev)).list.join(" ")
            g = g.with_columns(pl.coalesce(pl.col("state"), pl.col("dept"), expanded).alias("c"))
            parts.append(g)
        comps = pl.concat(parts, how="vertical_relaxed")
        stripped = pl.col("c").str.replace(CITY_PREFIX, "").str.replace(CITY_SUFFIX, "")
        cand = pl.when(stripped != "").then(stripped).otherwise(pl.col("c"))
        kind = (
            pl.when(pl.col("state").is_not_null()).then(pl.lit("state"))
            .when(pl.col("dept").is_not_null()).then(pl.lit("dept"))
            .when(pl.col("c").str.contains(r"\d")).then(pl.lit("number"))
            .when(pl.col("c").str.contains(LANDMARK_PREFIX)).then(pl.lit("landmark"))
            .when(pl.col("c").is_in(list(COUNTRY_WORDS))).then(pl.lit("country"))
            .otherwise(pl.lit("cand"))
        )
        return comps.with_columns(kind.alias("kind"), cand.alias("cand"))


def fit_city_vocab(comps: pl.DataFrame) -> pl.DataFrame:
    """(ckey, cand, freq): how often each city-like component occurs per country in a corpus.
    Fitted on the records being processed (all sources), so it needs no labels and works for new countries."""
    return comps.filter(pl.col("kind") == "cand").group_by("ckey", "cand").agg(pl.len().alias("freq"))


# ---------------------------------------------------------------------------- learned state aliases


def learn_state_aliases(pairs: pl.DataFrame, min_count: int = 30, purity: float = 0.9,
                        max_s1_ratio: float = 0.02, max_city_concentration: float = 1.5) -> dict[str, dict[str, str]]:
    """Learn state spellings from true pairs (columns: country, addr_1 (S1), addr_2 (match)).

    A match-side component becomes an alias of state X when it co-occurs with S1 state X in
    >= `purity` of its >= `min_count` pairs and is (almost) never an S1 component itself.
    The last condition keeps cities out: "mumbai" always sits with Maharashtra, but S1 uses it
    as a city, whereas "mharastr" (Devanagari transliterated) never appears in S1. Old city names
    that S1 never uses ("bombay", "poona") are kept out because their pairs concentrate on one
    S1 city far more than the state's pairs do overall (top-city share > `max_city_concentration`
    x the state's own top-city share), while a real state spelling mirrors the state's city mix."""
    base = Normaliser()
    s1c = base._components(pl.DataFrame({"a": transliterate(pairs["addr_1"]), "ckey": pairs["country"].str.to_lowercase()})
                           .with_row_index("rid"))
    s1_state = s1c.filter(pl.col("state").is_not_null()).group_by("rid").agg(pl.col("state").last().alias("s1_state"))
    s1_city = base.addresses(pairs["addr_1"], pairs["country"]).select("city").with_row_index("rid")
    s1_state = s1_state.join(s1_city, on="rid", how="left")
    s1_freq = s1c.group_by("ckey", "c").agg(pl.len().alias("n_s1"))
    mc = base._components(pl.DataFrame({"a": transliterate(pairs["addr_2"]), "ckey": pairs["country"].str.to_lowercase()})
                          .with_row_index("rid"))
    unknown = mc.filter((pl.col("kind") == "cand")).select("rid", "ckey", "c").unique()
    co = unknown.join(s1_state, on="rid")
    city_share = (
        co.group_by("ckey", "c", "city").agg(pl.len().alias("nc"))
        .group_by("ckey", "c").agg((pl.col("nc").max() / pl.col("nc").sum()).alias("top_city_share"))
    )
    state_ckey = s1_state.join(s1c.select("rid", "ckey").unique("rid"), on="rid")
    state_share = (
        state_ckey.group_by("ckey", "s1_state", "city").agg(pl.len().alias("nc"))
        .group_by("ckey", "s1_state").agg((pl.col("nc").max() / pl.col("nc").sum()).alias("state_top_city_share"))
    )
    stats = (
        co.group_by("ckey", "c", "s1_state").agg(pl.len().alias("n"))
        .with_columns(pl.col("n").sum().over("ckey", "c").alias("n_c"))
        .join(s1_freq, on=["ckey", "c"], how="left")
        .join(city_share, on=["ckey", "c"], how="left")
        .join(state_share, on=["ckey", "s1_state"], how="left")
        .with_columns(pl.col("n_s1").fill_null(0))
        .filter((pl.col("n_c") >= min_count) & (pl.col("n") >= purity * pl.col("n_c"))
                & (pl.col("n_s1") <= max_s1_ratio * pl.col("n_c")) & (pl.col("top_city_share") <= max_city_concentration * pl.col("state_top_city_share")))
    )
    out: dict[str, dict[str, str]] = {}
    for ckey, c, st in stats.select("ckey", "c", "s1_state").sort("ckey", "c").iter_rows():
        out.setdefault(ckey, {})[c] = st
    return out


def save_state_aliases(aliases: Mapping[str, Mapping[str, str]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(aliases, indent=1, sort_keys=True, ensure_ascii=False))
