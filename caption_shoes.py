"""
Generate CLIP-finetuning captions for every scraped shoe variant using
Azure AI Foundry (Azure OpenAI). Text-only — reads the scraped name/color/
price/product-details copy, does NOT look at images, to keep cost minimal.

Caption format (one dense line, no fluff):
    brand + model family + exact colorway + shoe type + materials +
    distinctive design details + gender/collection when relevant

Setup:
    cp .env.example .env
    # fill in AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_DEPLOYMENT
    pip install openai python-dotenv   # already done in .venv

Usage:
    python caption_shoes.py --limit 60   # test batch (only records with "details")
    python caption_shoes.py              # full run
"""

import argparse, json, os, re, time
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

DB_FILES = {
    "adidas": Path("adidas_products.json"),
    "nike": Path("nike_products.json"),
    "newbalance": Path("newbalance_products.json"),
    "skechers": Path("skechers_products.json"),
}
CHECKPOINT_EVERY = 20

# Pricing as of writing (USD per 1M tokens) — gpt-4o-mini on Azure.
# Used only to print a running cost estimate; not billed by this script.
PRICE_PER_1M_INPUT = 0.15
PRICE_PER_1M_OUTPUT = 0.60

SYSTEM_PROMPT = """You write single-line image captions for finetuning a CLIP-style vision-language model on shoe product photos.

Given structured product data, output ONE caption line in this exact structure:

{Brand} {Model family} in {exact colorway}, with {upper material}, {2-4 distinctive visual design details}.

Example captions (match this tone, density, and structure exactly):
- Adidas Gazelle Bold in Better Scarlet, Off White, and Eqt Yellow, with a suede upper, floral embroidery, platform sole, and branded tongue.
- Nike Air Force 1 '07 in white leather, with a perforated toe box, padded collar, Swoosh overlays, and rubber outsole.
- New Balance 9060 in Rain Cloud, with a mesh upper, suede overlays, sculpted midsole, and translucent heel detail.
- Skechers Slip-ins in navy canvas, with stretch laces, cushioned midsole, flexible outsole, and hands-free heel construction.

Rules:
- Keep ONLY visually verifiable, SKU-specific details — things a person could point to in THIS photo (colors, materials, silhouette, stitching, overlays, sole shape, hardware, embroidery, logo placement if visually distinctive).
- Order details from most distinctive/SKU-specific to most generic. Lead with whatever makes this exact variant recognizable, not a detail shared by every shoe of this model.
- NEVER use these words/phrases anywhere in the caption, no matter how the source data phrases things — find a concrete visible replacement or cut the clause entirely: "playful", "character-inspired", "character details", "classic silhouette", "sturdy", "stylish", "sleek", "versatile", "unique", "iconic", "signature look". These are marketing filler, not visual description — if the source text uses them, translate to the actual visible thing (a specific motif, color, shape) instead of copying the adjective.
- Do NOT include vague intent/marketing phrases ("futuristic aesthetic", "must-have", "timeless", "enjoy the comfort of", "designed for a modern aesthetic").
- Do NOT include comfort or performance claims (cushioning quality, support, breathability, all-day comfort) unless they describe something visible.
- Do NOT mention logos/branding unless the placement or execution is visually distinctive (e.g. "oversized tongue logo", "triple-stacked branding") — skip generic "adidas logo" or "Skechers logo" mentions.
- Capitalize the brand name consistently as a proper noun: "Adidas", "Nike", "New Balance", "Skechers" — never lowercase.
- Gender/collection words ("men's", "women's", "kids", "junior", "toddler") are OMITTED BY DEFAULT. The one exception: if the source name/data marks this SKU as a distinct size-tier or collection variant (e.g. "Junior", "Kids") of the same model/colorway as another SKU, INCLUDE that qualifier — it is the only thing distinguishing the two captions and is essential for retrieval.
- If two SKUs would otherwise produce near-identical captions (same model, same colorway family), actively look for ANY distinguishing visual or naming detail in the source data (closure type, silhouette variant, size-tier label, material difference) and surface it — captions for genuinely different SKUs should not be interchangeable.
- One line only, no line breaks, no bullet points.
- Keep it under 35 words.
- Output ONLY the caption text, nothing else."""


def build_prompt(brand: str, record: dict) -> str:
    lines = [
        f"Brand: {brand}",
        f"Name: {record.get('name', '')}",
        f"Color: {record.get('color_name', '')}",
    ]
    details = record.get("details", {})
    if details.get("description"):
        lines.append(f"Description: {details['description'][:600]}")
    for key in ("features", "product_details", "bullets", "materials", "key_features", "design_details"):
        if details.get(key):
            lines.append(f"{key.replace('_', ' ').title()}: {'; '.join(details[key])}")
    return "\n".join(lines)


def get_client() -> AzureOpenAI:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key or "YOUR" in api_key:
        raise SystemExit(
            "Azure credentials not set. Copy .env.example to .env and fill in "
            "AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_DEPLOYMENT."
        )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


CANONICAL_BRAND_CASE = {
    "adidas": "Adidas",
    "nike": "Nike",
    "newbalance": "New Balance",
    "skechers": "Skechers",
}


def fix_brand_casing(caption: str, brand: str) -> str:
    canonical = CANONICAL_BRAND_CASE.get(brand)
    if not canonical:
        return caption
    return re.sub(re.escape(canonical), canonical, caption, flags=re.IGNORECASE)


def caption_one(client: AzureOpenAI, deployment: str, brand: str, record: dict) -> tuple[str, int, int]:
    prompt = build_prompt(brand, record)
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=80,
        temperature=0,
    )
    caption = fix_brand_casing(resp.choices[0].message.content.strip(), brand)
    usage = resp.usage
    return caption, usage.prompt_tokens, usage.completion_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="only caption first N eligible (has details, no caption yet) records total")
    parser.add_argument("--brand", choices=list(DB_FILES), default=None,
                         help="only process this brand")
    args = parser.parse_args()

    client = get_client()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

    total_input_tokens = 0
    total_output_tokens = 0
    total_captioned = 0

    brands = [args.brand] if args.brand else list(DB_FILES)

    for brand in brands:
        db_file = DB_FILES[brand]
        if not db_file.exists():
            print(f"[skip] {brand}: {db_file} not found")
            continue
        database = json.loads(db_file.read_text())
        # Prefer records with scraped details, but still caption ones without
        # (name/color/price only) rather than leaving them uncaptioned.
        todo = [p for p in database if "caption" not in p]
        if args.limit:
            remaining = args.limit - total_captioned
            if remaining <= 0:
                break
            todo = todo[:remaining]
        if not todo:
            continue

        print(f"\n=== {brand}: {len(todo)} variants to caption ===")
        for idx, p in enumerate(todo, 1):
            try:
                caption, in_tok, out_tok = caption_one(client, deployment, brand, p)
            except Exception as e:
                print(f"  [warn] {p['product_code']}: {e}")
                time.sleep(1)
                continue

            p["caption"] = caption
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            total_captioned += 1

            cost_so_far = (total_input_tokens / 1e6 * PRICE_PER_1M_INPUT
                            + total_output_tokens / 1e6 * PRICE_PER_1M_OUTPUT)
            print(f"  [{idx}/{len(todo)}] {p['product_code']}: {caption}")

            if total_captioned % CHECKPOINT_EVERY == 0:
                db_file.write_text(json.dumps(database, indent=2, ensure_ascii=False))
                print(f"    [checkpoint] saved. running cost estimate: ${cost_so_far:.4f}")

            time.sleep(0.05)

        db_file.write_text(json.dumps(database, indent=2, ensure_ascii=False))

    total_cost = (total_input_tokens / 1e6 * PRICE_PER_1M_INPUT
                  + total_output_tokens / 1e6 * PRICE_PER_1M_OUTPUT)
    print(f"\nDone. {total_captioned} captions generated.")
    print(f"Tokens: {total_input_tokens} in / {total_output_tokens} out")
    print(f"Estimated cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
