"""Canonicalize the messy, brand-marketing `structured_caption.attributes.color`
values in apparel_dataset/metadata.json into a small controlled color
vocabulary, same convention and rationale as build_hierarchy.py did for
taxonomy_path -- see that script's docstring for the general pattern this
follows.

Why this exists: the by-facet SigLIP2 eval (docs/eval_log.md, 2026-07-29)
found color scoring a mediocre 31.57% R@1 despite being the most visually
salient, "classically easy" CLIP-style attribute. Root cause: 390 unique
raw color strings scraped verbatim from brand product pages, a large
fraction of which are marketing-name fragmentation of the same visual
color -- e.g. "black" / "core black" / "washed black" / "jet black", or
"sail" / "cloud white" / "summit white" / "cream white" / "core white" /
"off white" as six variants of near-identical whites. No vision model can
tell these apart from a photo; they're brand-naming artifacts, not real
color differences, and they were needlessly inflating the color
candidate space's effective difficulty.

Design principle (same as build_hierarchy.py): canonical color = a
visually distinct family a person would name looking at the garment from
a few feet away. Brand marketing sub-names collapse into their visual
family; genuinely distinct hues (navy vs. black vs. charcoal-gray, or tan
vs. khaki vs. cream) stay separate since apparel shoppers and the dataset
itself treat them as meaningfully different.

Methodology, in priority order per raw value:
  1. Exact match against EXPLICIT_COLOR_MAP -- hand-curated for brand
     colorway names I have real confidence about (Nike/Adidas/New Balance
     marketing terms like "photon dust", "obsidian", "sail", "gum").
  2. Substring keyword match against CANONICAL_KEYWORDS, checked in a
     specific order (more specific/compound keywords before generic
     single-word ones) so e.g. "metallic gold" resolves to gold, not a
     generic "metallic" bucket.
  3. Left unmapped and reported at the end -- per build_hierarchy.py's own
     convention, honesty about what wasn't confidently resolved beats
     guessing. Do not extend this script to force 100% coverage without
     actually verifying the unmapped names; some are genuinely ambiguous
     without seeing the product photo.

Writes:
  - docs/color_hierarchy.json     canonical color -> set of raw values folded into it
  - Adds `structured_caption.attributes.canonical_color` to every record
    (new field, list, non-destructive -- original `color` list untouched,
    same convention as canonical_taxonomy_path).
"""

import json
from collections import defaultdict
from pathlib import Path

METADATA_PATH = Path("apparel_dataset/metadata.json")
OUTPUT_PATH = Path("docs/color_hierarchy.json")


# ============================================================
# 1. Explicit mappings -- brand/marketing names I have real confidence
# about. Anything not here falls through to keyword matching, then to
# the unmapped report.
# ============================================================

EXPLICIT_COLOR_MAP = {
    # Nike/Jordan neutrals
    "sail": "cream", "phantom": "cream", "coconut milk": "cream",
    "pale ivory": "cream", "light bone": "cream", "natural": "cream",
    "photon dust": "gray", "particle grey": "gray", "wolf grey": "gray",
    "smoke grey": "gray", "dark smoke grey": "gray", "light smoke grey": "gray",
    "cool grey": "gray", "football grey": "gray", "iron grey": "gray",
    "lone star grey": "gray", "lone star gray": "gray", "mineral slate": "gray",
    "vast grey": "gray", "neptune grey": "gray", "horizon grey": "gray",
    "college grey": "gray", "grey fog": "gray", "grey matter": "gray",
    "shadow grey": "gray", "pearl grey": "gray", "mindful grey": "gray",
    "medium grey": "gray", "faded gray": "gray", "carbon heather": "gray",
    "carbon": "gray", "anthracite": "gray", "gridiron": "gray", "magnet": "gray",
    "timberwolf": "gray", "ghost": "gray", "steam": "gray",
    "obsidian": "navy", "nskm obsidian": "navy", "midnight navy": "navy",
    "college navy": "navy", "collegiate navy": "navy", "team navy": "navy",
    "mystic navy": "navy", "night sky": "navy", "night indigo": "navy",
    "world indigo": "navy", "vintage indigo": "navy", "dark indigo": "navy",
    "nb navy": "navy", "old royal": "navy",
    "gum": "gum", "gum light brown": "gum",
    "total black": "black", "off noir": "black", "faded black": "black",
    "nightshade": "black",
    # Common athletic-brand color-name shorthand
    "eqt yellow": "yellow", "varsity maize": "yellow", "amarillo": "yellow",
    "crew yellow": "yellow", "university gold": "gold", "bold gold": "gold",
    "topaz gold": "gold", "collegiate royal": "blue", "game royal": "blue",
    "royal": "blue", "hyper royal": "blue", "true blue": "blue",
    "football blue": "blue", "valor blue": "blue", "victory blue": "blue",
    "power blue": "blue", "rush blue": "blue", "work blue": "blue",
    "squadron blue": "blue", "diffused blue": "blue", "glacier blue": "blue",
    "light photo blue": "blue", "crew blue": "blue", "fairweather blue": "blue",
    "zinc blue": "blue", "light zinc blue": "blue", "blue void": "blue",
    "blue fusion": "blue", "blue glow": "blue", "blue beyond": "blue",
    "blue bird": "blue", "cobalt bliss": "blue", "denim turquoise": "teal",
    "team carolina": "blue", "oxford blue": "blue",
    "university red": "red", "team red": "red", "collegiate red": "red",
    "dark team red": "red", "gym red": "red", "picante red": "red",
    "chile red": "red", "flash crimson": "red", "team victory red": "red",
    "semi flash red": "red", "semi lucid red": "red", "speed red": "red",
    "solar red": "red", "chile red": "red", "iced carmine": "red",
    "red stardust": "red", "silt red": "red", "shadow red": "red",
    "better scarlet": "red", "noble maroon": "maroon", "burgundy ash": "maroon",
    "burgundy crush": "maroon",
    "collegiate green": "green", "pro green": "green", "lucky green": "green",
    "solar green": "green", "mean green": "green", "arctic green": "green",
    "alkaline green": "green", "green strike": "green", "green noise": "green",
    "power green": "green", "illusion green": "green", "malachite": "green",
    "quantum moss": "olive", "focus olive": "olive", "dusty olive": "olive",
    "olive strata": "olive", "olive aura": "olive", "olive flak": "olive",
    "dusty cactus": "olive", "cucumber calm": "green", "barely green": "green",
    "sage": "olive", "spruce aura": "green", "spruce fog": "green",
    "black spruce": "green", "arctic night": "green", "silver moss": "green",
    "preloved green": "green",
    "hyper pink": "pink", "laser fuchsia": "pink", "cosmic fuchsia": "pink",
    "hot fuchsia": "pink", "shock pink": "pink", "playful pink": "pink",
    "fire pink": "pink", "blush pink": "pink", "clear pink": "pink",
    "pink rise": "pink", "pink salt": "pink", "pink smoke": "pink",
    "pink spark": "pink", "pink spell": "pink", "pinksicle": "pink",
    "tropical pink": "pink", "true pink": "pink", "sandy pink": "pink",
    "medium soft pink": "pink", "pearl pink": "pink", "peony": "pink",
    "rose": "pink", "rose sugar": "pink", "stone pink": "pink",
    "light magenta": "pink", "preloved red": "red",
    "court purple": "purple", "grand purple": "purple", "hyper grape": "purple",
    "vivid purple": "purple", "persian violet": "purple", "smoked violet": "purple",
    "platinum violet": "purple", "bright violet": "purple", "light violet": "purple",
    "amethyst tint": "purple", "purple agate": "purple", "powder plum": "purple",
    "dark concord": "purple", "dark raisin": "purple",
    "aurora coffee": "brown", "aurora ivy": "green", "mosswood brown": "brown",
    "thunder brown": "brown", "fox brown": "brown", "mink brown": "brown",
    "preloved brown": "brown", "cave stone": "tan", "earth strata": "tan",
    "arid stone": "tan", "desert": "tan", "desert berry": "pink",
    "desert khaki": "khaki", "cargo khaki": "khaki", "light khaki": "khaki",
    "stone khaki": "khaki", "parachute beige": "tan", "magic beige": "tan",
    "light orewood brown": "tan", "hemp": "tan", "malt": "tan",
    "pumpernickel": "brown", "cortado": "brown", "cocoa": "brown",
    "cacao wow": "brown", "chocolate": "brown", "chestnut": "brown",
    "sequoia": "brown", "clay": "tan", "warm clay": "tan", "mushroom": "tan",
    "taupe haze": "tan", "dark taupe": "tan", "dark driftwood": "brown",
    "dark stucco": "tan", "turtle dove": "tan", "turtledove": "tan",
    "doll": "tan", "cargo khaki": "khaki",
    "safety orange": "orange", "unity orange": "orange", "laser orange": "orange",
    "campfire orange": "orange", "dusky orange": "orange", "orange pulse": "orange",
    "orange horizon": "orange", "orange peel": "orange", "orange chalk": "orange",
    "lucid tangerine": "orange", "tangerine heat": "orange", "hot lava": "orange",
    "infrared 23": "orange", "neo flame": "red",
    "action grape": "purple",
    "aluminum": "silver", "aluminum grey": "silver", "alumina": "silver",
    "chrome": "silver", "pure platinum": "silver", "platinum tint": "silver",
    "metallic platinum": "silver", "ice gold met": "gold",
    "metallic gold": "gold", "gold metallic": "gold", "rose gold": "gold",
    "metallic copper": "brown", "copper metallic": "brown",
    "metallic red bronze": "maroon", "metallic dark grey": "gray",
    "metallic cool grey": "gray", "metallic summit white": "cream",
    "light silver metallic": "silver", "dark silver metallic": "silver",
    "silver metallic": "silver", "metallic silver": "silver",
    "wonder silver": "silver", "wonder white": "cream", "wonder quartz": "tan",
    "moonbeam": "cream", "lightning": "gray", "shimmer": "silver",
    "reflection": "silver", "raincloud": "gray", "rain cloud": "gray",
    "castlerock": "gray", "cement grey": "gray", "dark grey heather": "gray",
    "heather grey": "gray", "heather": "gray", "grey heather": "gray",
    "light grey": "gray", "light gray": "gray", "dark grey": "gray",
    "slate": "gray", "slate grey": "gray", "ashen slate": "gray",
    "mineral": "gray", "cement": "gray",
    "aura": "gray",
    "afterglow": "orange", "sweet beet": "maroon", "mystic dates": "brown",
    "hydrangeas": "purple", "periwinkle": "blue", "lapis": "blue",
    "sapphire": "blue", "turquoise": "teal", "bleached turquoise": "teal",
    "pulse aqua": "teal", "artisan teal": "teal", "malachite": "green",
    "mauve": "purple", "grand purple": "purple", "real lilac": "purple",
    "bleached lilac": "purple", "lavender": "purple", "violet": "purple",
    "amarillo": "yellow", "varsity maize": "yellow", "bright citron": "yellow",
    "solar green": "green", "olive flak": "olive",
    "team orange": "orange", "team crimson": "red", "team white": "cream",
    "denim": "blue", "blue denim": "blue", "medium blue light wash": "blue",
    "light medium blue": "blue", "quicksand": "tan", "sand": "tan",
    "camouflage": "multi", "multi-color": "multi",
    "pale ivory": "cream", "chalk pearl": "cream", "chalk": "cream",
    "chalk white": "cream", "crystal linen": "cream", "crystal white": "cream",
    "alabaster": "cream", "sea salt": "cream", "linen": "cream",
    "angora": "cream", "clear sky": "blue", "crystal sky": "blue",
    "glint blue": "blue", "power blue": "blue", "bright blue": "blue",
    "aluminum grey": "silver",
    "coral": "pink", "clay": "tan", "tattoo": "red", "gridiron": "gray",
    "iron grey": "gray", "ironstone": "gray", "shadow grey": "gray",
    "olive strata": "olive", "arctic green": "green", "aurora ivy": "green",
    "pink foam": "pink", "mint foam": "green", "mint water": "green",
    "volt ice": "yellow", "light liquid lime": "yellow", "lime blast": "yellow",
    "team victory red": "red", "picante red": "red",
    "taupe": "tan", "breakfast tea": "brown", "pencil point": "gray",
    "truffle salt": "brown",
}


# ============================================================
# 2. Substring keyword fallback, checked in order (first match wins) --
# longer/more specific tokens first so e.g. "metallic gold" doesn't
# accidentally match a bare "gold" check before a more specific one, and
# "off white" doesn't match a bare "white" check meant for pure white.
# ============================================================

CANONICAL_KEYWORDS = [
    ("off white", "cream"), ("off-white", "cream"), ("cream", "cream"),
    ("ivory", "cream"), ("bone", "cream"), ("summit white", "cream"),
    ("cloud white", "cream"), ("core white", "cream"), ("pearl", "cream"),
    ("white", "cream"),
    ("gum", "gum"),
    ("obsidian", "navy"), ("navy", "navy"), ("indigo", "navy"),
    ("black", "black"), ("jet black", "black"), ("onyx", "black"),
    ("charcoal", "gray"), ("grey", "gray"), ("gray", "gray"), ("smoke", "gray"),
    ("stone", "tan"), ("concrete", "gray"), ("cement", "gray"),
    ("silver", "silver"), ("platinum", "silver"), ("chrome", "silver"),
    ("metallic gold", "gold"), ("gold", "gold"),
    ("khaki", "khaki"),
    ("beige", "tan"), ("tan", "tan"), ("camel", "tan"), ("sand", "tan"),
    ("teal", "teal"), ("turquoise", "teal"), ("aqua", "teal"),
    ("blue", "blue"), ("denim", "blue"), ("cobalt", "blue"), ("sapphire", "blue"),
    ("maroon", "maroon"), ("burgundy", "maroon"), ("wine", "maroon"),
    ("red", "red"), ("crimson", "red"), ("scarlet", "red"), ("ruby", "red"),
    ("coral", "pink"), ("fuchsia", "pink"), ("magenta", "pink"), ("blush", "pink"),
    ("pink", "pink"), ("rose", "pink"),
    ("purple", "purple"), ("violet", "purple"), ("lavender", "purple"),
    ("grape", "purple"), ("plum", "purple"), ("mauve", "purple"),
    ("lilac", "purple"), ("amethyst", "purple"), ("orchid", "purple"),
    ("olive", "olive"), ("sage", "olive"), ("moss", "olive"),
    ("green", "green"), ("mint", "green"), ("forest", "green"), ("emerald", "green"),
    ("brown", "brown"), ("chocolate", "brown"), ("cocoa", "brown"),
    ("espresso", "brown"), ("mocha", "brown"), ("walnut", "brown"),
    ("mahogany", "brown"), ("chestnut", "brown"), ("coffee", "brown"),
    ("tobacco", "brown"), ("umber", "brown"),
    ("orange", "orange"), ("tangerine", "orange"), ("citron", "yellow"),
    ("yellow", "yellow"), ("maize", "yellow"),
    ("multi", "multi"), ("camo", "multi"), ("rainbow", "multi"),
]


def canonicalize_color(raw_value):
    key = str(raw_value).strip().lower()
    if key in EXPLICIT_COLOR_MAP:
        return EXPLICIT_COLOR_MAP[key]
    for keyword, canonical in CANONICAL_KEYWORDS:
        if keyword in key:
            return canonical
    return None


def main():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    color_hierarchy = defaultdict(set)
    unmapped = defaultdict(int)
    updated = 0

    for product in metadata:
        sc = product.get("structured_caption")
        if not sc:
            continue
        attributes = sc.get("attributes") or {}
        raw_colors = attributes.get("color") or []
        if not raw_colors:
            continue

        canonical_colors = []
        for raw_value in raw_colors:
            canonical = canonicalize_color(raw_value)
            if canonical is None:
                unmapped[str(raw_value).strip().lower()] += 1
                continue
            color_hierarchy[canonical].add(str(raw_value).strip().lower())
            if canonical not in canonical_colors:
                canonical_colors.append(canonical)

        if canonical_colors:
            attributes["canonical_color"] = canonical_colors
            updated += 1

    color_hierarchy_json = {
        canonical: sorted(raw_values) for canonical, raw_values in sorted(color_hierarchy.items())
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(color_hierarchy_json, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    total_raw = sum(len(v) for v in color_hierarchy_json.values()) + len(unmapped)
    print(f"Updated {updated} products with canonical_color")
    print(f"Wrote color hierarchy to {OUTPUT_PATH}")
    print(f"\nCanonical colors ({len(color_hierarchy_json)}), raw values folded into each:")
    for canonical, raw_values in color_hierarchy_json.items():
        print(f"  {canonical:8s}  ({len(raw_values)} raw values)")

    print(f"\nCoverage: {total_raw - len(unmapped)}/{total_raw} unique raw color strings mapped "
          f"({100 * (total_raw - len(unmapped)) / total_raw:.1f}%)")

    if unmapped:
        print(f"\n{len(unmapped)} distinct raw color values had no confident mapping "
              f"(left unmapped, canonical_color omits these -- review and add to "
              f"EXPLICIT_COLOR_MAP or CANONICAL_KEYWORDS, don't guess blind):")
        for value, count in sorted(unmapped.items(), key=lambda kv: -kv[1]):
            print(f"  {count:4d}  {value!r}")


if __name__ == "__main__":
    main()
