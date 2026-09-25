"""Shared normalisation for names and addresses (used by blocking, features and inference).

Deterministic and vectorised with polars; the only per-string Python call is anyascii
transliteration, applied once per distinct non-ASCII string.

Open-set countries: rules are looked up by the lower-cased country label. Country tables only ADD
cleaning for known countries; any other label (new in test, empty, made up) gets the generic rules
(transliteration, cleaning, generic legal suffixes and abbreviations, number tokens, city guess).

Names   -> full_name (legal suffix canonicalised), core_name (legal suffix removed), legal (the suffix)
Address -> addr_norm, city, state, dept (French departement), numbers (digit tokens, leading zeros stripped)
Fallbacks: core_name falls back to full_name; text fields are empty only when the raw value has no
letters or digits. city / state / dept are null when not found. normalise_records() keeps the raw columns.

What is learned from data (see docs/normalisation.md):
- state aliases: from TRAIN true pairs only (labels), excluding validation S1; saved to a versioned file.
- city frequency vocabulary: from train + test records (no labels), per country.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import polars as pl
from anyascii import anyascii

# Bump whenever a rule or list changes; stored in every output so cached artifacts can be checked.
NORM_VERSION = "norm-v3"  # v2: international legal forms only for France / unknown countries; v3: cie, Malayalam/Gurmukhi ltd/pvt, no dangling "and"

# ---------------------------------------------------------------------------- basic cleaning


def transliterate(s: pl.Series) -> pl.Series:
    """anyascii for non-ASCII strings (Devanagari, Tamil, accents, ...); ASCII strings untouched."""
    s = s.cast(pl.String).fill_null("")
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


def _ckey(countries: pl.Series) -> pl.Series:
    return countries.cast(pl.String).fill_null("").str.to_lowercase().str.strip_chars()


# ---------------------------------------------------------------------------- name rules (language knowledge)

# Legal suffixes: token -> canonical token. Only a trailing run (or, for LEGAL_LEADING, a leading run)
# of these is treated as a legal form; the same words mid-name are kept.
LEGAL_GENERAL = {  # all countries
    "inc": "inc", "incorporated": "inc", "incorporation": "inc",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc", "psc": "psc",
    "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "cos": "co",
    "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "opc": "opc",
}
# Continental-European forms. Applied to France and to every country WITHOUT its own table (open set),
# but not to the US or India, where "spa" / "sas" / "sl" are ordinary words ("Ramey Spa Inc", "Sas Nagar").
LEGAL_INTERNATIONAL = {
    "gmbh": "gmbh", "bv": "bv", "srl": "srl", "spa": "spa", "sl": "sl", "slu": "slu",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "selarl": "selarl", "eirl": "eirl", "scop": "scop",
    "cie": "co",  # "& Cie" = "& Co"
}
LEGAL_BY_COUNTRY = {
    "us": {},
    "india": {
        # transliterations seen in S2/S3 (Devanagari / Tamil / Kannada / Odia -> anyascii), e.g. "pra li" = "pvt ltd"
        "praivet": "pvt", "praibhet": "pvt", "piraivet": "pvt", "praivett": "pvt", "prayvet": "pvt", "pra": "pvt",
        "praivrr": "pvt",  # Malayalam
        "limitet": "ltd", "limited": "ltd", "limitedd": "ltd", "li": "ltd", "elelpi": "llp", "ellpi": "llp",
        "limirrd": "ltd",  # Malayalam
        "limtid": "ltd",  # Gurmukhi
    },
    "france": {"sa": "sa", "sci": "sci", "snc": "snc", "ei": "ei", "sca": "sca", "scp": "scp", "gie": "gie", "scm": "scm"},
}
# Legal forms that also appear as a LEADING token after word reordering ("LLC Value Electronics").
# Only unambiguous abbreviations: never words like "company" or "limited", or short forms like "co"/"sa".
LEGAL_LEADING = {"inc", "llc", "llp", "ltd", "pvt", "corp", "plc", "pllc"}
LEGAL_LEADING_INTERNATIONAL = {"gmbh", "sarl", "sas", "sasu", "eurl", "snc"}
KNOWN_ENGLISH_ONLY = {"us", "india"}  # countries whose tables replace the international forms


def _legal_for(country: str) -> dict[str, str]:
    extra = {} if country in KNOWN_ENGLISH_ONLY else LEGAL_INTERNATIONAL
    return {**LEGAL_GENERAL, **extra, **LEGAL_BY_COUNTRY.get(country, {})}


def _leading_for(country: str) -> set[str]:
    return LEGAL_LEADING if country in KNOWN_ENGLISH_ONLY else LEGAL_LEADING | LEGAL_LEADING_INTERNATIONAL
# multi-token spellings, collapsed before suffix detection (anchored at the end of the name)
LEGAL_MULTI = [(r" l l c$", " llc"), (r" l l p$", " llp"), (r" p l l c$", " pllc"), (r" l p$", " lp"), (r" p c$", " pc")]

# Honorific prefixes (repeated prefixes are all removed): India only, where they are noise added to names.
HONORIFICS_BY_COUNTRY = {
    "india": ["m s", "ms", "messrs", "smt", "shrimati", "shri", "shree", "sri", "sree", "kumari", "km", "mr", "mrs", "dr"],
}

# Word abbreviations (token -> expansion)
NAME_ABBREV_GENERAL = {
    "intl": "international", "mfg": "manufacturing", "mfrs": "manufacturers",
    "svc": "services", "svcs": "services", "srvcs": "services", "mgmt": "management", "mgt": "management",
    "assn": "association", "assoc": "associates", "dept": "department", "natl": "national",
    "govt": "government", "univ": "university", "hosp": "hospital", "ctr": "center", "cntr": "center",
    "centre": "center", "bros": "brothers", "engg": "engineering", "engr": "engineering", "grp": "group",
    "hldgs": "holdings", "inds": "industries", "sys": "systems", "solns": "solutions",
}
NAME_ABBREV_BY_COUNTRY = {"france": {"et": "and", "st": "saint", "ste": "sainte"}}


# ---------------------------------------------------------------------------- address rules (language knowledge)

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
COUNTRY_WORDS = {"india", "bharat", "usa", "us", "united states", "united states of america", "america", "france",
                 "null", "none", "na", "n a", "nil"}
CITY_PREFIX = r"^(city of|town of|village of|township of|borough of|ville de)\s+"
CITY_SUFFIX = r"\s+(city|township|town|village|borough|cdp)$"
# A component "<postcode> <words>" or "<words> <postcode>" (4-6 digits) is a city candidate once the
# postcode is removed, unless the words look like a street (then the number is a house number) or a
# unit / box ("po box 4823", "suite 1200", "pmb 12345").
POSTCODE_EDGE = r"^\d{4,6}\s+|\s+\d{4,6}$"
STREET_WORDS = (r"\b(road|street|avenue|lane|drive|boulevard|highway|parkway|court|circle|trail|terrace|way|"
                r"rue|allee|impasse|chemin|route|quai|calle|avenida|carrer|via|viale|piazza|marg|floor|building|"
                r"po|box|pmb|unit|suite|ste|apt|apartment|room|rm|flat|plot|no|door|house|block|office|shop|sector|"
                r"ward|lot|survey|khasra|gali|lane)\b"
                r"|(strasse|gasse|weg|platz)\b")

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
    """Hand-written variant -> canonical state/region, keys cleaned like address components.
    Empty for any country without a table (open set)."""
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


def _dept_table(country: str) -> dict[str, str]:
    return {clean_text(d): clean_text(d) for d in FRANCE_DEPTS} if country == "france" else {}


# ---------------------------------------------------------------------------- normaliser


class Normaliser:
    """Holds the lookup tables. `state_aliases` ({country: {variant: canonical}}) are learned from
    training pairs with learn_state_aliases() and added on top of the hand-written tables."""

    def __init__(self, state_aliases: Mapping[str, Mapping[str, str]] | None = None, version: str = NORM_VERSION):
        self.state_aliases = {_key(c): dict(m) for c, m in (state_aliases or {}).items()}
        self.version = version

    @classmethod
    def load(cls, path: str | Path | None) -> "Normaliser":
        """Load aliases saved by save_state_aliases(); a missing path gives the hand-written tables only."""
        if not path or not Path(path).exists():
            return cls()
        data = json.loads(Path(path).read_text())
        if "aliases" in data:  # versioned file
            if data.get("norm_version") != NORM_VERSION:
                raise ValueError(f"{path} was learned with {data.get('norm_version')}, code is {NORM_VERSION}")
            return cls(data["aliases"])
        return cls(data)

    def states(self, country: str) -> dict[str, str]:
        return {**_state_table(country), **self.state_aliases.get(country, {})}

    # ------------------------------------------------------------------ names
    def names(self, names: pl.Series, countries: pl.Series) -> pl.DataFrame:
        raw = names.cast(pl.String).fill_null("")
        df = pl.DataFrame({"n": transliterate(raw), "ckey": _ckey(countries)})
        df = df.with_row_index("rid").with_columns(_clean(pl.col("n"), drop_dots=True).alias("n"))
        parts = [self._names_one(g, ckey) for (ckey,), g in df.group_by("ckey")]
        if not parts:
            return pl.DataFrame(schema={"full_name": pl.String, "core_name": pl.String, "legal": pl.String})
        return pl.concat(parts).sort("rid").drop("rid")

    def _names_one(self, g: pl.DataFrame, country: str) -> pl.DataFrame:
        e = pl.col("n")
        hon = HONORIFICS_BY_COUNTRY.get(country)
        if hon:
            stripped = e.str.replace(r"^(?:(?:" + "|".join(hon) + r")\s+)+", "")
            e = pl.when(stripped != "").then(stripped).otherwise(e)
        abbrev = {**NAME_ABBREV_GENERAL, **NAME_ABBREV_BY_COUNTRY.get(country, {})}
        e = e.str.split(" ").list.eval(pl.element().replace(abbrev)).list.join(" ")
        for pat, rep in LEGAL_MULTI:
            e = e.str.replace(pat, rep)
        legal = _legal_for(country)
        alt = "|".join(sorted(map(re.escape, legal), key=len, reverse=True))
        tail_pat = r"^(.*?)((?: (?:" + alt + r"))*)$"
        lead_pat = r"^((?:(?:" + "|".join(sorted(_leading_for(country))) + r") )+)(.+)$"
        g = g.with_columns(e.alias("_e")).with_columns(
            pl.col("_e").str.extract(lead_pat, 1).fill_null("").alias("_lead"),
            pl.coalesce(pl.col("_e").str.extract(lead_pat, 2), pl.col("_e")).alias("_rest"),
        )
        g = g.with_columns((pl.lit(" ") + pl.col("_rest")).alias("_s"))
        core = pl.col("_s").str.extract(tail_pat, 1).fill_null("").str.strip_chars()
        g = g.with_columns(
            core.alias("core_name"),
            pl.concat_str([pl.col("_lead"), pl.col("_s").str.extract(tail_pat, 2).fill_null("")], separator=" ")
            .str.replace_all(r"\s+", " ").str.strip_chars().str.split(" ")
            .list.eval(pl.element().replace(legal)).list.unique(maintain_order=True)  # "pvt ltd ltd" -> "pvt ltd"
            .list.join(" ").str.strip_chars().alias("legal"),
        )
        # "Smith & Co", "Elsa & Cie SARL", "Dupont et Cie": the connector belongs to the legal form
        g = g.with_columns(
            pl.when(pl.col("legal") != "").then(pl.col("core_name").str.replace(r"(?:\s+and)+$", ""))
            .otherwise(pl.col("core_name")).alias("core_name"))
        full = pl.concat_str([pl.col("core_name"), pl.col("legal")], separator=" ").str.strip_chars()
        g = g.with_columns(full.alias("full_name"))
        # Fallbacks: a name that is only legal tokens ("LLC", "Pvt Ltd") keeps them as its core.
        only_legal = pl.col("core_name") == ""
        return g.select(
            "rid",
            "full_name",
            pl.when(only_legal).then(pl.col("full_name")).otherwise(pl.col("core_name")).alias("core_name"),
            pl.when(only_legal).then(pl.lit("")).otherwise(pl.col("legal")).alias("legal"),
        )

    # ------------------------------------------------------------------ addresses
    def addresses(self, addrs: pl.Series, countries: pl.Series, city_vocab: pl.DataFrame | None = None) -> pl.DataFrame:
        """city_vocab: (ckey, cand, freq) from city_counts(); when None it is fitted on these addresses.
        The city is the most frequent city-like component, so reordered addresses agree."""
        raw = addrs.cast(pl.String).fill_null("")
        df = pl.DataFrame({"a": transliterate(raw), "ckey": _ckey(countries)}).with_row_index("rid")
        comps = self._components(df)
        vocab = city_vocab if city_vocab is not None else _vocab(comps)
        cities = (
            comps.filter(pl.col("kind") == "cand")
            .join(vocab, on=["ckey", "cand"], how="left")
            .group_by("rid")
            .agg(pl.col("cand").sort_by([pl.col("freq").fill_null(0), pl.col("pos")], descending=[True, True]).first().alias("city"))
        )
        per = comps.group_by("rid").agg(
            pl.col("c").sort_by("pos").str.join(", ").alias("addr_norm"),  # keeps component boundaries (idempotent)
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
        """One row per non-empty comma component: rid, ckey, pos, c (normalised), state, dept, kind, cand."""
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
            # States first, on the cleaned component: US codes like "ct", "fl", "mt" would otherwise be
            # expanded to court / floor / mount. A state may carry a trailing postcode ("nc 27260").
            bare = pl.col("c").str.replace(r"(\s\d{5,6})+$", "")
            g = g.with_columns(
                bare.replace_strict(self.states(ckey), default=None, return_dtype=pl.String).alias("state"),
                bare.replace_strict(_dept_table(ckey), default=None, return_dtype=pl.String).alias("dept"),
            )
            expanded = pl.col("c").str.split(" ").list.eval(pl.element().replace(abbrev)).list.join(" ")
            g = g.with_columns(pl.coalesce(pl.col("state"), pl.col("dept"), expanded).alias("c"))
            parts.append(g)
        schema = {"rid": pl.UInt32, "ckey": pl.String, "c": pl.String, "pos": pl.Int64, "state": pl.String, "dept": pl.String}
        comps = pl.concat(parts, how="vertical_relaxed") if parts else pl.DataFrame(schema=schema)
        no_post = pl.col("c").str.replace_all(POSTCODE_EDGE, "").str.strip_chars()
        city_like = (no_post != "") & ~no_post.str.contains(r"\d") & ~no_post.str.contains(STREET_WORDS)
        stripped = no_post.str.replace(CITY_PREFIX, "").str.replace(CITY_SUFFIX, "")
        cand = pl.when(stripped != "").then(stripped).otherwise(no_post)
        kind = (
            pl.when(pl.col("state").is_not_null()).then(pl.lit("state"))
            .when(pl.col("dept").is_not_null()).then(pl.lit("dept"))
            .when(pl.col("c").str.contains(LANDMARK_PREFIX)).then(pl.lit("landmark"))
            .when(pl.col("c").is_in(list(COUNTRY_WORDS))).then(pl.lit("country"))
            .when(~pl.col("c").str.contains(r"\d")).then(pl.lit("cand"))
            .when(city_like).then(pl.lit("cand"))  # "10115 berlin", "paris 75001"
            .otherwise(pl.lit("number"))
        )
        return comps.with_columns(kind.alias("kind"), cand.alias("cand"))

    def city_counts(self, addrs: pl.Series, countries: pl.Series) -> pl.DataFrame:
        """(ckey, cand, freq) for one batch of addresses. Sum over batches (train + test, all sources,
        no labels) with merge_city_counts() to build the vocabulary passed to addresses()."""
        df = pl.DataFrame({"a": transliterate(addrs.cast(pl.String).fill_null("")), "ckey": _ckey(countries)}).with_row_index("rid")
        return _vocab(self._components(df))


def _vocab(comps: pl.DataFrame) -> pl.DataFrame:
    return comps.filter(pl.col("kind") == "cand").group_by("ckey", "cand").agg(pl.len().cast(pl.Int64).alias("freq"))


def merge_city_counts(parts: list[pl.DataFrame]) -> pl.DataFrame:
    return pl.concat(parts).group_by("ckey", "cand").agg(pl.col("freq").sum())


def normalise_records(df: pl.DataFrame, norm: Normaliser, city_vocab: pl.DataFrame | None = None,
                      name_col: str = "business_name", addr_col: str = "business_address",
                      country_col: str = "country") -> pl.DataFrame:
    """All original columns (raw name/address kept) + normalised name and address fields + norm_version."""
    n = norm.names(df[name_col], df[country_col])
    a = norm.addresses(df[addr_col], df[country_col], city_vocab)
    return pl.concat([df, n, a], how="horizontal_extend").with_columns(pl.lit(norm.version).alias("norm_version"))


# ---------------------------------------------------------------------------- learned state aliases


def learn_state_aliases(pairs: pl.DataFrame, min_count: int = 30, purity: float = 0.9,
                        max_s1_ratio: float = 0.02, max_city_concentration: float = 1.5,
                        max_with_state: float = 0.2) -> dict[str, dict[str, str]]:
    """Learn state spellings from TRAIN true pairs (columns: country, addr_1 (S1), addr_2 (match)).
    Uses labels: never pass pairs of validation S1 entities.

    A match-side city-like component becomes an alias of state X when
    - it co-occurs with S1 state X in >= `purity` of its >= `min_count` pairs,
    - it is (almost) never an S1 component (<= `max_s1_ratio`): keeps real cities such as "mumbai" out,
    - its pairs do not concentrate on one S1 city much more than the state's pairs do overall
      (<= `max_city_concentration` x): keeps old city names such as "bombay" / "poona" out,
    - it REPLACES the state: <= `max_with_state` of the match addresses containing it also contain a
      known state component. Keeps city abbreviations such as "atl" (written "ATL, GA") out."""
    base = Normaliser()
    ckey = _ckey(pairs["country"])
    s1c = base._components(pl.DataFrame({"a": transliterate(pairs["addr_1"]), "ckey": ckey}).with_row_index("rid"))
    s1_state = s1c.filter(pl.col("state").is_not_null()).group_by("rid").agg(pl.col("state").last().alias("s1_state"))
    s1_city = base.addresses(pairs["addr_1"], pairs["country"]).select("city").with_row_index("rid")
    s1_state = s1_state.join(s1_city, on="rid", how="left")
    s1_freq = s1c.group_by("ckey", "c").agg(pl.len().alias("n_s1"))
    mc = base._components(pl.DataFrame({"a": transliterate(pairs["addr_2"]), "ckey": ckey}).with_row_index("rid"))
    has_state = mc.group_by("rid").agg((pl.col("kind") == "state").any().alias("has_state"))
    unknown = mc.filter(pl.col("kind") == "cand").select("rid", "ckey", "c").unique().join(has_state, on="rid")
    co = unknown.join(s1_state, on="rid")
    per_c = co.group_by("ckey", "c").agg(pl.col("has_state").mean().alias("with_state_share"))
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
        .join(per_c, on=["ckey", "c"], how="left")
        .with_columns(pl.col("n_s1").fill_null(0))
        .filter((pl.col("n_c") >= min_count) & (pl.col("n") >= purity * pl.col("n_c"))
                & (pl.col("n_s1") <= max_s1_ratio * pl.col("n_c"))
                & (pl.col("top_city_share") <= max_city_concentration * pl.col("state_top_city_share"))
                & (pl.col("with_state_share") <= max_with_state))
    )
    out: dict[str, dict[str, str]] = {}
    for ck, c, st in stats.select("ckey", "c", "s1_state").sort("ckey", "c").iter_rows():
        out.setdefault(ck, {})[c] = st
    return out


def alias_diagnostics(pairs: pl.DataFrame, aliases: Mapping[str, Mapping[str, str]]) -> pl.DataFrame:
    """Evidence per learned alias (pair count, purity) for review."""
    ckey = _ckey(pairs["country"])
    base = Normaliser()
    s1c = base._components(pl.DataFrame({"a": transliterate(pairs["addr_1"]), "ckey": ckey}).with_row_index("rid"))
    s1_state = s1c.filter(pl.col("state").is_not_null()).group_by("rid").agg(pl.col("state").last().alias("s1_state"))
    mc = base._components(pl.DataFrame({"a": transliterate(pairs["addr_2"]), "ckey": ckey}).with_row_index("rid"))
    rows = [{"ckey": ck, "c": a, "state": s} for ck, m in aliases.items() for a, s in m.items()]
    if not rows:
        return pl.DataFrame(schema={"ckey": pl.String, "alias": pl.String, "state": pl.String, "pairs": pl.UInt32, "purity": pl.Float64})
    al = pl.DataFrame(rows)
    hits = mc.select("rid", "ckey", "c").unique().join(al, on=["ckey", "c"]).join(s1_state, on="rid", how="left")
    return (
        hits.group_by("ckey", "c", "state")
        .agg(pl.len().alias("pairs"), (pl.col("s1_state") == pl.col("state")).mean().alias("purity"))
        .rename({"c": "alias"})
        .sort("ckey", "pairs", descending=[False, True])
    )


def save_state_aliases(aliases: Mapping[str, Mapping[str, str]], path: str | Path, meta: Mapping | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {"norm_version": NORM_VERSION, "meta": dict(meta or {}), "aliases": aliases}
    Path(path).write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False))
