# Normalisation (`src/normalise.py`, version `norm-v3`)

This stage cleans every business name and address once. The same code is used for blocking, features and
inference. It is deterministic, vectorised with polars, and **idempotent**: normalising an output again
changes nothing. It is tested on 3,000 real records.

**No external data.** Every hand-written list below is general language and geography knowledge: legal-form
abbreviations, honorifics, street abbreviations, and the official names of US states, Indian states and French
regions and départements. It is the kind of knowledge a person reading the addresses already has. Nothing is
looked up from a database, API, geocoder or business registry. Everything else is learned from the challenge's
own files, as described in [What is learned from data](#what-is-learned-from-data).

## Outputs

| Field | Meaning |
|---|---|
| `business_name`, `business_address`, `country` | raw input, always kept unchanged |
| `full_name` | cleaned name with the legal form in canonical spelling (`royal food pvt ltd`) |
| `core_name` | `full_name` without the legal form (`royal food`) |
| `legal` | the canonical legal form(s) (`pvt ltd`), or empty |
| `addr_norm` | cleaned address, comma components kept (`12 rue saint honore, 75001 paris`) |
| `city`, `state`, `dept` | extracted city, state or region (canonical), French département; null when not found |
| `numbers` | digit tokens from the address, leading zeros removed (`001006` → `1006`) |
| `norm_version` | `norm-v3`; bump it whenever a rule or list changes. v1 → v2: international legal forms no longer applied to US/India. v2 → v3: `cie`, Malayalam/Gurmukhi spellings, connector removal |

**Fallbacks.**
- If `core_name` would be empty (the name is only a legal form, such as "LLC" or "Pvt Ltd"), it falls back to
  `full_name`.
- Text fields are empty only when the raw value has no letters or digits (empty, whitespace or punctuation only).
  The raw value is still kept.
- `city`, `state` and `dept` are null when not found. They are never guessed from other countries' rules.

## Processing order

**Names**
1. Transliterate any non-ASCII text with anyascii (ISC licence): accents, Devanagari, Tamil, Kannada, Odia,
   and so on.
2. Lowercase, `&` → `and`, delete dots and apostrophes (`l.l.c.` → `llc`), other punctuation → space, collapse
   spaces.
3. Remove honorific prefixes (India only).
4. Expand word abbreviations.
5. Collapse multi-token legal spellings at the end (`l l c` → `llc`).
6. Split off the legal form: a trailing run of legal tokens, and a leading run of unambiguous ones (after word
   reordering: "LLC Value Electronics"). The same words mid-name are kept ("Andaman Private Fashion Limited" →
   core `andaman private fashion`). Repeated forms collapse (`pvt ltd ltd` → `pvt ltd`).

**Addresses**
1. Transliterate, split on commas, and clean each component (dots become spaces: `no.330` → `no 330`).
2. **State first**, on the cleaned component, optionally followed by a 5-6 digit postcode (`nc 27260`). This is
   done before abbreviations so that the US codes `ct`, `fl` and `mt` are not turned into court, floor and mount.
3. French département check (France only).
4. Country-specific and general abbreviation expansion on the remaining components.
5. Classify each component: state, département, landmark (`near…`, `opp…`), country word, city candidate, or
   number. A component such as `75001 paris` or `10115 berlin` is a city candidate once the 4-6 digit postcode
   is removed, unless the rest looks like a street or unit (`17560 ellis road`, `po box 4823`).
6. `city` = the city candidate that is most frequent in the country's city vocabulary (so reordered addresses
   agree), with `city of …`, `… city`, `… township` and `… cdp` stripped.

## Hand-written lists and rules (language knowledge)

### Legal forms (name suffixes)
- **General, all countries** (20 spellings → canonical):
  - inc/incorporated/incorporation → `inc`; `llc`, `llp`, `lp`, `plc`, `pllc`, `pc`, `psc`
  - corp/corporation → `corp`; co/company/cos → `co`
  - ltd/limited → `ltd`; pvt/private → `pvt`; `opc`
- **International, France and every country without its own table:** `gmbh`, `bv`, `srl`, `spa`, `sl`, `slu`,
  `sarl`, `sas`, `sasu`, `eurl`, `selarl`, `eirl`, `scop`, and `cie` → `co`. These are **not** applied to the US
  or India, where they are ordinary words ("Ramey Spa Inc" keeps `spa`; "Sas Nagar" is a place).
- **Connector:** when a legal form is removed, a trailing `and` goes with it ("Smith & Co" → core `smith`,
  "Elsa & Cie SARL" → core `elsa`; French `et` is mapped to `and` first).
- **India only.** Transliterated forms seen in S2/S3 after anyascii:
  - praivet/praibhet/piraivet/praivett/prayvet/praivrr (Malayalam)/pra → `pvt`
  - limitet/limitedd/limirrd (Malayalam)/limtid (Gurmukhi)/li → `ltd`
  - elelpi/ellpi → `llp`
  - `pra` and `li` count only as trailing tokens in Indian names.
- **France only.** Short forms that are ordinary words elsewhere: `sa`, `sci`, `snc`, `ei`, `sca`, `scp`, `gie`,
  `scm`.
- **Leading position** (after word reordering), unambiguous abbreviations only: inc, llc, llp, ltd, pvt, corp,
  plc, pllc. For France and countries without a table, also gmbh, sarl, sas, sasu, eurl, snc.
- **Multi-token at the end:** `l l c` → llc, `l l p` → llp, `p l l c` → pllc, `l p` → lp, `p c` → pc.

### Honorific prefixes (India only)
m/s, ms, messrs, smt, shrimati, shri, shree, sri, sree, kumari, km, mr, mrs, dr. These are removed only as
leading tokens, repeatedly ("M/s. Smt Kaveri Foods" → `kaveri foods`), and never if nothing would remain.

### Word abbreviations in names
- **General (26):** intl, mfg, mfrs, svc/svcs/srvcs, mgmt/mgt, assn, assoc, dept, natl, govt, univ, hosp,
  ctr/cntr/centre, bros, engg/engr, grp, hldgs, inds, sys, solns → full words.
- **France:** `et` → and, `st` → saint, `ste` → sainte.

### Address abbreviations
- **General, all countries (8):** rd → road, ave → avenue, blvd → boulevard, hwy → highway, pkwy → parkway,
  bldg → building, apt → apartment, flr → floor.
- **US (21):** **st → street**, str, dr → drive, ln, ct → court, cir, pl, sq, ste → suite, fl → floor, ter/terr,
  trl, pt, mt → mount, ft → fort, hts, rte, fwy, expy, cres.
- **India (5):** nr → near, opp → opposite, fl → floor, ngr → nagar, bldg.
- **France (15):** **st → saint, ste → sainte**, av → avenue, bd/bld/bvd/boul → boulevard, rte, chem, pl, imp,
  fbg, sq, all → allee, r → rue.
- `st` is deliberately left untouched for every other country.

### Geography
- **US states:** 50 states, DC and 5 territories (56 two-letter codes and full names). Canonical form = the
  lowercase code (`texas` → `tx`).
- **Indian states and union territories:** 36, with 54 codes and alternative spellings (MH, KA, TG/TS, OD/OR,
  orissa, pondicherry, ...). Canonical form = the full name.
- **French regions:** 13 metropolitan + 5 overseas, with 24 old-region names and alternative spellings (alsace
  → grand est, picardie → hauts de france, ...). Stored in `state`.
- **French départements:** 95 names (Paris excluded, because as an address component it is the city). Stored in
  `dept`, never used as the city.
- **Other lists:**
  - Landmark prefixes: near, nr, opp, opposite, behind, beside, next to, adjacent to, in front of.
  - Null and country words: india, bharat, usa, us, united states, america, france, null, none, na, n a, nil.
  - City prefixes and suffixes: city of / town of / village of / township of / borough of / ville de;
    city / township / town / village / borough / cdp.
  - Street and unit words that block the postcode-city rule (road, street, rue, calle, strasse, po box, suite,
    unit, plot, ...).

## What is learned from data

| What | Learned from | Uses labels? | Where it is stored |
|---|---|---|---|
| State aliases (e.g. Devanagari `mharastr` → maharashtra, `krnatk` → karnataka) | **Train true pairs only, excluding S1 entities in the validation split** (`role != "val"` in `g25_split.parquet`); seeded subsample of up to 3M pairs | **Yes** | `state_aliases_norm-v3.json`, with provenance; loaded at inference, never relearned |
| City frequency vocabulary (which component is the city) | Train + test records of all sources, per country | No | `city_vocab_norm-v3.parquet` |

**Alias rules.** A match-side component becomes an alias of state X only when all of these hold:
- ≥ 30 pairs, and ≥ 90% of them have S1 state X;
- it (almost) never appears in S1, which keeps real cities such as "mumbai" out;
- its pairs do not concentrate on one S1 city much more than the state's pairs do overall, which keeps old city
  names such as "bombay" and "poona" out;
- it **replaces** the state: ≤ 20% of its addresses also contain a known state. This keeps city abbreviations
  such as `atl` (written "ATL, GA") out.

## Unknown countries (open set)

The country label is lower-cased and looked up. Country tables only **add** rules for us, india and france.
Any other label gets the generic rules, never another country's rules (the international legal forms are
generic, since they are what a European or other country most likely uses). That includes Germany, Spain, a made-up
label, an empty string or null. The generic rules are:
- transliteration, cleaning, general and international legal forms, and general abbreviations;
- number tokens and the postcode-city rule;
- a city guess from the frequency vocabulary.

For an unknown country there is no honorific removal, no `st` expansion, no state or département lookup, and no
French short legal forms. Tested examples:
- "Müller GmbH" → core `muller`, legal `gmbh`
- "Hauptstraße 5, 10115 Berlin" → city `berlin`, numbers `5, 10115`
- "Construcciones García S.L." → core `construcciones garcia`, legal `sl`
- "Calle Mayor 3, Madrid" → city `madrid`

## Known limitation (agreed: handled as a feature)

About 5.7% of names still contain a legal word inside `core_name`. The generator appends noise words after the
legal form ("Willow LLC Center", "Krishna Best Investment Limited Center"), or reorders it mid-name. The suffix is
then no longer at an edge, so it is not stripped. Stripping legal words anywhere in normalisation would also hit
real names ("Andaman Private Fashion"). **Decision:** normalisation stays as it is. The feature stage adds name
similarities computed on the tokens with **all** legal words removed (any position), next to the `core_name` and
`full_name` similarities. The model then learns when that matters.
