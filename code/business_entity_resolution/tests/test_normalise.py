"""Normalisation tests. Inputs are real strings from outputs/eda/ (e20, d17, f24 samples) unless noted."""
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normalise import NORM_VERSION, Normaliser, learn_state_aliases, normalise_records  # noqa: E402

N = Normaliser()


def names(pairs: list[tuple[str, str]]) -> list[dict]:
    raw, ctry = zip(*pairs)
    return N.names(pl.Series(raw), pl.Series(ctry)).to_dicts()


def addrs(pairs: list[tuple[str, str]], norm: Normaliser = N) -> list[dict]:
    raw, ctry = zip(*pairs)
    return norm.addresses(pl.Series(raw), pl.Series(ctry)).to_dicts()


# ------------------------------------------------------------------ names

@pytest.mark.parametrize("raw,country,full,core,legal", [
    ("Royal Food Pvt. Ltd.", "India", "royal food pvt ltd", "royal food", "pvt ltd"),
    ("Jayesh Industrial Private Limited", "India", "jayesh industrial pvt ltd", "jayesh industrial", "pvt ltd"),
    ("Great Bay Polska L.L.C.", "US", "great bay polska llc", "great bay polska", "llc"),
    ("Helios Inc.", "US", "helios inc", "helios", "inc"),
    ("Piedmont Fund Inc", "US", "piedmont fund inc", "piedmont fund", "inc"),
    ("T N & Z TÓWING EAST P.C.", "US", "t n and z towing east pc", "t n and z towing east", "pc"),
    ("Jamais Collège SARL", "France", "jamais college sarl", "jamais college", "sarl"),
    ("Bordeaux Jeunes Sasu", "France", "bordeaux jeunes sasu", "bordeaux jeunes", "sasu"),
    ("Girls Acteurs Cie Participations [SARL]", "France", "girls acteurs company participations sarl",
     "girls acteurs company participations", "sarl"),
    ("Deleves Çulturelle SNC", "France", "deleves culturelle snc", "deleves culturelle", "snc"),
])
def test_legal_suffix_full_and_core(raw, country, full, core, legal):
    r = names([(raw, country)])[0]
    assert (r["full_name"], r["core_name"], r["legal"]) == (full, core, legal)


def test_same_business_same_core_across_suffix_variants():
    r = names([("Upvan Sterling Private Limited", "India"), ("Upvan Sterling Pvt Ltd", "India"),
               ("UPVAN STERLING PRIVATE", "India")])
    assert {x["core_name"] for x in r} == {"upvan sterling"}
    assert r[0]["full_name"] == r[1]["full_name"] == "upvan sterling pvt ltd"


def test_indian_scripts_transliterated_and_suffix_found():
    r = names([("यूनिवर्सल केयर बेकरी प्रा. लि.", "India"),          # Devanagari "Pvt. Ltd."
               ("அல் லாஜிஸ்டிக்ஸ் பிரைவேட் லிமிடெட்", "India"),     # Tamil "Private Limited"
               ("लोटस टेक प्राइवेट लिमिटेड", "India"),
               ("ବେଷ୍ଟ୍ ଏକ୍ସପୋର୍ଟସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍", "India"),  # Odia
               ("बालाजी इंडस्ट्रीज एलएलपी", "India")])
    assert [x["legal"] for x in r] == ["pvt ltd", "pvt ltd", "pvt ltd", "pvt ltd", "llp"]
    assert r[1]["core_name"] == "al lajistiks"
    assert all(x["full_name"].isascii() for x in r)


def test_indian_honorifics_removed():
    r = names([("M/s Nidhi Infrastructure  Co", "India"), ("Smt JP Interior Private Limited", "India"),
               ("Shri Balaji Traders", "India"), ("M/S. Smt Kaveri Foods", "India")])
    assert [x["core_name"] for x in r] == ["nidhi infrastructure", "jp interior", "balaji traders", "kaveri foods"]


def test_honorifics_only_for_india_and_never_empty():
    r = names([("Dr Smith Dental Group", "US"), ("Shri", "India")])
    assert r[0]["core_name"] == "dr smith dental group"
    assert r[1]["core_name"] == "shri"


def test_leading_legal_form_after_word_reorder():
    r = names([("LLC Value Electronics International", "US"), ("Value Electronics International LLC", "US"),
               ("Inc Platinum Genomic Dynamics", "US")])
    assert r[0]["full_name"] == r[1]["full_name"] == "value electronics international llc"
    assert r[2]["core_name"] == "platinum genomic dynamics"


def test_short_forms_are_country_aware():
    # "sa" is a legal form only in France; "li" only as an Indian transliteration of "Ltd"
    r = names([("SAAD SA", "France"), ("Casa Sa", "US"), ("Wang Li", "US"), ("Balaji Industries Pra Li", "India")])
    assert r[0]["legal"] == "sa" and r[0]["core_name"] == "saad"
    assert r[1]["legal"] == "" and r[2]["legal"] == ""
    assert r[3]["legal"] == "pvt ltd"


def test_mid_name_legal_words_are_kept():
    r = names([("Andaman Private Fashion Limited", "India"), ("SAAD SA DÉVELOPPEMENT", "France")])
    assert r[0]["core_name"] == "andaman private fashion"
    assert r[1]["core_name"] == "saad sa developpement" and r[1]["legal"] == ""


def test_name_that_is_only_a_suffix_falls_back_to_full_name():
    r = names([("Company", "US"), ("LLC", "US"), ("Pvt Ltd", "India"), ("SARL", "France")])
    assert [x["core_name"] for x in r] == [x["full_name"] for x in r] == ["co", "llc", "pvt ltd", "sarl"]
    assert all(x["legal"] == "" for x in r)


def test_abbreviations_all_countries_including_unseen():
    r = names([("Acme Intl Mfg Corp", "Canada"), ("École et Associés SA", "France")])
    assert r[0]["full_name"] == "acme international manufacturing corp"
    assert r[1]["core_name"] == "ecole and associes"


# ------------------------------------------------------------------ addresses

def test_us_state_code_and_full_name_agree():
    r = addrs([("25224 56th Avenue, Phoenix, AZ", "US"), ("#30592 Twin Rose Ln, Princess Anne, Maryland", "US"),
               ("001006 Mcmillan Ave, Bay Minette, Alabama", "US"), ("12909 Churchill Court, KS, Wichita, Fl 0", "US")])
    assert [x["state"] for x in r] == ["az", "md", "al", "ks"]
    assert [x["city"] for x in r] == ["phoenix", "princess anne", "bay minette", "wichita"]


def test_us_state_codes_not_eaten_by_abbreviations():
    # CT / FL / MT are states as whole components, but court / floor / mount inside a street
    r = addrs([("10 Main St, Hartford, CT 06103", "US"), ("Fl. 1, Missouri, 331 Avant Drive, Hazelwood", "US")])
    assert r[0]["state"] == "ct" and "main street" in r[0]["addr_norm"]
    assert r[1]["state"] == "mo" and r[1]["addr_norm"].startswith("floor 1")


def test_st_is_street_in_us_and_saint_in_france():
    r = addrs([("4446 Woodwind St, Charlotte, NC", "US"), ("4 Rue Daurat, St-Nazaire, Pays de la Loire", "France"),
               ("2 Rue Ste Catherine, Bordeaux, Nouvelle-Aquitaine", "France")])
    assert "woodwind street" in r[0]["addr_norm"]
    assert r[1]["city"] == "saint nazaire"
    assert "rue sainte catherine" in r[2]["addr_norm"]


def test_st_left_alone_for_unknown_country():
    assert "main st" in addrs([("10 Main St, Toronto", "Canada")])[0]["addr_norm"]


def test_france_departement_separate_from_city():
    r = addrs([("Gironde, Nº 37 AVENUE CARNOT, PESSAC", "France"),
               ("5 AVENUE DES COTTAGES, NANTES, Loire-Atlantique", "France"),
               ("12 Rue Perinot, Bordeaux, Nouvelle-Aquitaine", "France")])
    assert [x["dept"] for x in r] == ["gironde", "loire atlantique", None]
    assert [x["city"] for x in r] == ["pessac", "nantes", "bordeaux"]
    assert r[2]["state"] == "nouvelle aquitaine"


def test_france_abbreviated_rue():
    r = addrs([("16 R ARAGO, BORDEAUX, Nouvelle-Aquitaine", "France")])
    assert r[0]["addr_norm"].startswith("16 rue arago") and r[0]["city"] == "bordeaux"


def test_india_state_codes_and_names_agree():
    r = addrs([("607, Street No.3 Tarnaka, Secunderabad, Hyderabad, Telangana", "India"),
               ("607, Secunderabad, Hyderabad, TG", "India"),
               ("C/o Yazdi Patel Post Gholvad, Taluka Dahanu, Thane, Palghar, MH", "India"),
               ("Roli(E), Roli (E), Mumbai, Mumbai City, Maharashtra", "India")])
    assert [x["state"] for x in r] == ["telangana", "telangana", "maharashtra", "maharashtra"]


def test_landmark_component_is_not_the_city():
    r = addrs([("510/8A, NEW HYDERABAD, NEAR WATER TANK PARK, LUCKNOW, Uttar Pradesh", "India"),
               ("Tulasi House, No.26, Opp Hp Petrol Station, New Redial Road, Bangalore North, NULL, KA", "India")])
    assert r[0]["city"] == "lucknow" and r[0]["state"] == "uttar pradesh"
    assert r[1]["state"] == "karnataka" and r[1]["city"] != "opposite hp petrol station"


def test_numbers_extracted_without_leading_zeros():
    r = addrs([("001006 Mcmillan Ave, Bay Minette, Alabama", "US"), ("H.no. 1-291/Rv/1/402, Balaji Nagar", "India"),
               ("", "US")])
    assert r[0]["numbers"] == ["1006"]
    assert r[1]["numbers"] == ["1", "291", "402"]
    assert r[2]["numbers"] == [] and r[2]["city"] is None and r[2]["addr_norm"] == ""


def test_city_suffixes_stripped():
    r = addrs([("333 Morgan Ln, Lexington CITY, North Carolina", "US"), ("NC, Lexington, 333 Morgan Lane", "US")])
    assert r[0]["city"] == r[1]["city"] == "lexington"
    assert r[0]["state"] == r[1]["state"] == "nc"


def test_reordered_address_picks_the_frequent_city():
    # S2/S3 reorder components; the city is the most frequent city-like component in the corpus
    corpus = [("1 Oak Street, Boise, ID", "US")] * 5 + [("ID, 2 Elm Street, Boise, Garden Plaza", "US")]
    assert addrs(corpus)[-1]["city"] == "boise"


def test_learned_aliases_map_transliterated_states():
    s1 = "H.no. 1-291/Rv/1/402, Balaji Nagar Miyapur, Telangana, Hyderabad"
    s2 = "H.no. 1-291/Rv/1/402, Balaji Nagar Miyapur, Hyderabad, తెలంగాణ"
    pairs = pl.DataFrame({"country": ["India"] * 40, "addr_1": [s1] * 40, "addr_2": [s2] * 40})
    aliases = learn_state_aliases(pairs, min_count=30, max_city_concentration=1.5)
    assert aliases == {"india": {"telmgan": "telangana"}}
    r = addrs([(s1, "India"), (s2, "India")], Normaliser(aliases))
    assert r[0]["state"] == r[1]["state"] == "telangana"
    assert r[0]["city"] == r[1]["city"] == "hyderabad"


def test_learned_aliases_reject_old_city_names():
    # "bombay" always sits with Maharashtra but only with one S1 city, while the state spans several
    s1 = [f"{i} Road, {c}, Maharashtra" for i, c in enumerate(["Mumbai", "Pune", "Nagpur", "Nashik"] * 10)]
    s2 = [f"{i} Road, Bombay" if c == "Mumbai" else f"{i} Road, {c}, MH" for i, c in
          enumerate(["Mumbai", "Pune", "Nagpur", "Nashik"] * 10)]
    pairs = pl.DataFrame({"country": ["India"] * 400, "addr_1": s1 * 10, "addr_2": s2 * 10})
    assert "bombay" not in learn_state_aliases(pairs).get("india", {})


def test_deterministic_and_order_preserving():
    raw = pl.Series(["Helios Inc.", "Royal Food Pvt Ltd", "Jamais Collège SARL", "Helios Inc."])
    ctry = pl.Series(["US", "India", "France", "US"])
    a, b = N.names(raw, ctry), N.names(raw, ctry)
    assert a.equals(b)
    assert a["core_name"].to_list() == ["helios", "royal food", "jamais college", "helios"]


# ------------------------------------------------------------------ Part 1: fallbacks and edge cases

def test_empty_whitespace_and_null_inputs():
    r = N.names(pl.Series(["", "   ", None]), pl.Series(["US", "India", None]))
    assert r.to_dicts() == [{"full_name": "", "core_name": "", "legal": ""}] * 3
    a = N.addresses(pl.Series(["", "  ", None]), pl.Series(["US", "", None])).to_dicts()
    assert all(x["addr_norm"] == "" and x["city"] is None and x["numbers"] == [] for x in a)


def test_numbers_and_punctuation_only_names():
    r = names([("12345", "US"), ("!!! ---", "US"), ("7-Eleven", "US")])
    assert r[0]["core_name"] == r[0]["full_name"] == "12345"
    assert r[1]["full_name"] == "" and r[1]["core_name"] == ""  # no letters or digits: empty (raw is kept)
    assert r[2]["core_name"] == "7 eleven"


def test_repeated_suffixes_collapse():
    r = names([("Royal Food Pvt Ltd Ltd", "India"), ("Acme Inc. Inc.", "US"), ("Acme LLC L.L.C.", "US")])
    assert [x["legal"] for x in r] == ["pvt ltd", "inc", "llc"]
    assert r[0]["full_name"] == "royal food pvt ltd"


def test_mixed_scripts_in_one_string():
    r = names([("Hotel कंसल्टेंट्स Private Limited", "India")])[0]
    assert r["full_name"].isascii() and r["legal"] == "pvt ltd" and r["core_name"].startswith("hotel ")


def test_very_long_strings():
    long_name = "Alpha " * 3000 + "Private Limited"
    long_addr = ", ".join(f"{i} Road" for i in range(2000)) + ", Pune, Maharashtra"
    r = names([(long_name, "India")])[0]
    assert r["legal"] == "pvt ltd" and len(r["core_name"].split()) == 3000
    a = addrs([(long_addr, "India")])[0]
    assert a["city"] == "pune" and a["state"] == "maharashtra" and len(a["numbers"]) == 2000


def test_raw_columns_kept_by_normalise_records():
    df = pl.DataFrame({"entity_id": ["S1-1", "S2-9"], "business_name": ["Helios Inc.", "M/s Nidhi Co"],
                       "business_address": ["10 Main St, Hartford, CT", ""], "country": ["US", "India"]})
    out = normalise_records(df, N)
    assert out.select(df.columns).equals(df)
    assert {"full_name", "core_name", "legal", "addr_norm", "city", "state", "dept", "numbers"} <= set(out.columns)
    assert out["norm_version"].to_list() == [NORM_VERSION] * 2


# ------------------------------------------------------------------ Part 2: aliases

def test_city_abbreviation_next_to_a_state_is_not_an_alias():
    # "atl" (Atlanta) is written next to the state ("ATL, GA"), so it does not replace the state
    s1 = [f"{i} Peachtree Street, {c}, GA" for i, c in enumerate(["Atlanta", "Macon", "Athens", "Savannah"] * 10)]
    s2 = [f"{i} Peachtree St, ATL, GA" for i in range(40)]
    pairs = pl.DataFrame({"country": ["US"] * 400, "addr_1": s1 * 10, "addr_2": s2 * 10})
    assert "atl" not in learn_state_aliases(pairs).get("us", {})


def test_versioned_alias_file_roundtrip(tmp_path):
    from normalise import save_state_aliases
    path = tmp_path / f"state_aliases_{NORM_VERSION}.json"
    save_state_aliases({"india": {"telmgan": "telangana"}}, path, {"learned_from": "test"})
    n = Normaliser.load(path)
    assert n.states("india")["telmgan"] == "telangana"
    assert Normaliser.load(tmp_path / "missing.json").state_aliases == {}


# ------------------------------------------------------------------ Part 3: open-set countries

def test_germany_generic_rules_only():
    n = names([("Müller GmbH", "Germany")])[0]
    assert n == {"full_name": "muller gmbh", "core_name": "muller", "legal": "gmbh"}
    a = addrs([("Hauptstraße 5, 10115 Berlin", "Germany")])[0]
    assert a["city"] == "berlin" and a["state"] is None and a["dept"] is None
    assert a["numbers"] == ["5", "10115"]


def test_spain_generic_rules_only():
    n = names([("Construcciones García S.L.", "Spain")])[0]
    assert n["core_name"] == "construcciones garcia" and n["legal"] == "sl"
    a = addrs([("Calle Mayor 3, Madrid", "Spain")])[0]
    assert a["city"] == "madrid" and a["numbers"] == ["3"]


@pytest.mark.parametrize("country", ["Atlantis", "", "  ", "XX-123"])
def test_unknown_country_gets_no_country_specific_rules(country):
    n = names([("Shri Balaji Traders", country), ("Casa Sa", country), ("Dr Smith Dental", country)])
    assert n[0]["core_name"] == "shri balaji traders"   # no Indian honorific removal
    assert n[1]["legal"] == ""                           # no French "sa"
    assert n[2]["core_name"] == "dr smith dental"
    a = addrs([("12 Main St, Springfield, TX", country), ("4 Rue St-Honore, Lyon", country)])
    assert "main st" in a[0]["addr_norm"] and a[0]["state"] is None   # no US "st" -> street, no US states
    assert "rue st honore" in a[1]["addr_norm"]                       # no French "st" -> saint
    assert a[0]["numbers"] == ["12"] and a[0]["city"] in {"springfield", "tx"}


def test_null_country_does_not_crash():
    r = N.names(pl.Series(["Helios Inc."]), pl.Series([None], dtype=pl.String))
    assert r["core_name"].to_list() == ["helios"]


# ------------------------------------------------------------------ Part 3: France without training data

def test_france_names():
    r = names([("Boulangerie St Michel SARL", "France"), ("SAS Dupont et Fils", "France")])
    assert r[0] == {"full_name": "boulangerie saint michel sarl", "core_name": "boulangerie saint michel", "legal": "sarl"}
    assert r[1] == {"full_name": "dupont and fils sas", "core_name": "dupont and fils", "legal": "sas"}


def test_france_addresses_postcode_city_and_departements():
    r = addrs([("12 Rue St-Honoré, 75001 Paris", "France"),
               ("3 Rue Nationale, Lille, Nord", "France"),
               ("Nord, 3 RUE NATIONALE, LILLE", "France"),
               ("7 Cours Victor Hugo, Bordeaux, Gironde, Nouvelle-Aquitaine", "France")])
    assert r[0]["addr_norm"] == "12 rue saint honore, 75001 paris"
    assert r[0]["city"] == "paris" and r[0]["dept"] is None and r[0]["numbers"] == ["12", "75001"]
    assert [x["dept"] for x in r[1:]] == ["nord", "nord", "gironde"]
    assert [x["city"] for x in r[1:]] == ["lille", "lille", "bordeaux"]
    assert r[3]["state"] == "nouvelle aquitaine"


# ------------------------------------------------------------------ Part 4: determinism and idempotence

IDEMPOTENCE_NAMES = [
    ("Royal Food Pvt. Ltd.", "India"), ("M/s Nidhi Infrastructure  Co", "India"), ("LLC Value Electronics", "US"),
    ("Great Bay Polska L.L.C.", "US"), ("अल लॉजिस्टिक्स प्रा. लि.", "India"), ("SAS Dupont et Fils", "France"),
    ("Boulangerie St Michel SARL", "France"), ("Müller GmbH", "Germany"), ("Dupont et Cie", "France"),
    ("Acme Intl Mfg Corp", "Canada"), ("Company", "US"), ("12345", "US"),
]
IDEMPOTENCE_ADDRS = [
    ("1795 Westchester Drive, High Point, NC", "US"), ("#30592 Twin Rose Ln, Princess Anne, Maryland", "US"),
    ("12 Rue St-Honoré, 75001 Paris", "France"), ("16 R ARAGO, BORDEAUX, Nouvelle-Aquitaine", "France"),
    ("510/8A, NEW HYDERABAD, NEAR WATER TANK PARK, LUCKNOW, Uttar Pradesh", "India"),
    ("Hauptstraße 5, 10115 Berlin", "Germany"), ("Fl. 1, Missouri, 331 Avant Drive, Hazelwood", "US"),
]


def test_names_idempotent():
    first = names(IDEMPOTENCE_NAMES)
    again = names([(x["full_name"], c) for x, (_, c) in zip(first, IDEMPOTENCE_NAMES)])
    assert [x["full_name"] for x in again] == [x["full_name"] for x in first]
    core_again = names([(x["core_name"], c) for x, (_, c) in zip(first, IDEMPOTENCE_NAMES)])
    assert [x["core_name"] for x in core_again] == [x["core_name"] for x in first]


def test_addresses_idempotent():
    first = addrs(IDEMPOTENCE_ADDRS)
    again = addrs([(x["addr_norm"], c) for x, (_, c) in zip(first, IDEMPOTENCE_ADDRS)])
    assert [x["addr_norm"] for x in again] == [x["addr_norm"] for x in first]
    assert [x["numbers"] for x in again] == [x["numbers"] for x in first]


SAMPLE = Path(__file__).resolve().parents[3] / "data_sample/train/train_source2.parquet"


@pytest.mark.skipif(not SAMPLE.exists(), reason="local data_sample not available")
def test_idempotent_and_deterministic_on_real_sample():
    df = pl.read_parquet(SAMPLE).sample(3000, seed=1)
    a = N.names(df["business_name"], df["country"])
    assert a.equals(N.names(df["business_name"], df["country"]))
    b = N.names(a["full_name"], df["country"])
    changed = (a["full_name"] != b["full_name"]).sum()
    assert changed == 0, a.filter(a["full_name"] != b["full_name"]).head(5)
    ad = N.addresses(df["business_address"], df["country"])
    ad2 = N.addresses(ad["addr_norm"], df["country"])
    assert (ad["addr_norm"] != ad2["addr_norm"]).sum() == 0


def test_international_forms_not_applied_to_us_or_india():
    r = names([("Ramey Spa Inc", "US"), ("Laws Frontier Spa", "US"), ("Sas Nagar Mohali Infratech Pvt Ltd", "India"),
               ("Rossi Costruzioni S.p.A.", "Italy"), ("Boulangerie Dupont SAS", "France")])
    assert (r[0]["core_name"], r[0]["legal"]) == ("ramey spa", "inc")
    assert (r[1]["core_name"], r[1]["legal"]) == ("laws frontier spa", "")
    assert (r[2]["core_name"], r[2]["legal"]) == ("sas nagar mohali infratech", "pvt ltd")
    assert (r[3]["core_name"], r[3]["legal"]) == ("rossi costruzioni", "spa")
    assert (r[4]["core_name"], r[4]["legal"]) == ("boulangerie dupont", "sas")
