"""
Generate structured retrieval-training captions for every apparel record in
apparel_dataset/metadata.json using the schema/prompt defined in
newLLMprompt.py (positive_texts, taxonomy_path, attributes, defining_features).

Unlike caption_shoes.py (which wrote a single free-text `caption` string),
this writes a new `structured_caption` field (a parsed JSON object) and never
touches the existing `caption` field — both coexist on every record.

Azure OpenAI (gpt-4o-mini), text-only, same .env credentials as caption_shoes.py.

Usage:
    python caption_apparel.py --limit 20     # test batch
    python caption_apparel.py                # only records missing structured_caption
    python caption_apparel.py --force        # reprocess every record
    python caption_apparel.py --category "Shorts"   # only this category
"""

import argparse, json, os, re, time
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

from newLLMprompt import prompt as PROMPT_TEMPLATE
from dataset_utils import load_records, save_records_safe

load_dotenv()

CHECKPOINT_EVERY = 20

PRICE_PER_1M_INPUT = 0.15
PRICE_PER_1M_OUTPUT = 0.60

CANONICAL_BRAND_CASE = {
    "adidas": "Adidas",
    "nike": "Nike",
    "newbalance": "New Balance",
    "skechers": "Skechers",
    "pacsun": "Pacsun",
}


def build_description(record: dict) -> str:
    details = record.get("details", {})
    parts = []
    if details.get("description"):
        parts.append(details["description"])
    for key in ("features", "product_details", "bullets", "materials", "key_features", "design_details"):
        if details.get(key):
            parts.append(f"{key.replace('_', ' ').title()}: " + "; ".join(details[key]))
    return "\n".join(parts)[:2000]


def build_user_message(record: dict) -> str:
    brand_display = CANONICAL_BRAND_CASE.get(record.get("brand", ""), record.get("brand", ""))
    return (
        PROMPT_TEMPLATE
        .replace("{{PRODUCT_TITLE}}", record.get("name", ""))
        .replace("{{BRAND}}", brand_display)
        .replace("{{SKU}}", record.get("product_code", ""))
        .replace("{{CATEGORY}}", record.get("category", ""))
        .replace("{{COLOR}}", record.get("color_name", ""))
        .replace("{{DESCRIPTION}}", build_description(record))
    )


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


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


def caption_one(client: AzureOpenAI, deployment: str, record: dict) -> tuple[dict, int, int]:
    user_msg = build_user_message(record)
    messages = [{"role": "user", "content": user_msg}]

    resp = client.chat.completions.create(
        model=deployment, messages=messages, max_tokens=900, temperature=0,
    )
    raw = resp.choices[0].message.content
    in_tok = resp.usage.prompt_tokens
    out_tok = resp.usage.completion_tokens

    try:
        return extract_json(raw), in_tok, out_tok
    except json.JSONDecodeError:
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": "That was not valid JSON. Return ONLY the single valid JSON object, no markdown, no explanations."})
        resp2 = client.chat.completions.create(
            model=deployment, messages=messages, max_tokens=900, temperature=0,
        )
        raw2 = resp2.choices[0].message.content
        in_tok += resp2.usage.prompt_tokens
        out_tok += resp2.usage.completion_tokens
        return extract_json(raw2), in_tok, out_tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="reprocess records that already have structured_caption")
    parser.add_argument("--brand", default=None)
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    client = get_client()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

    database = load_records()

    todo = database
    if not args.force:
        todo = [r for r in todo if "structured_caption" not in r]
    if args.brand:
        todo = [r for r in todo if r.get("brand") == args.brand]
    if args.category:
        todo = [r for r in todo if r.get("category") == args.category]
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(todo)} records to caption (of {len(database)} total).\n")

    total_input_tokens = 0
    total_output_tokens = 0
    total_captioned = 0
    total_failed = 0
    touched = {}

    for idx, record in enumerate(todo, 1):
        try:
            structured, in_tok, out_tok = caption_one(client, deployment, record)
        except Exception as e:
            print(f"  [warn] {record.get('product_code')}: {e}")
            total_failed += 1
            time.sleep(1)
            continue

        record["structured_caption"] = structured
        touched[record["product_code"]] = record
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        total_captioned += 1

        cost_so_far = (total_input_tokens / 1e6 * PRICE_PER_1M_INPUT
                        + total_output_tokens / 1e6 * PRICE_PER_1M_OUTPUT)
        n_pos = len(structured.get("positive_texts", []))
        print(f"[{idx}/{len(todo)}] {record.get('product_code')}: {n_pos} positive_texts, "
              f"taxonomy={structured.get('taxonomy_path')}")

        if total_captioned % CHECKPOINT_EVERY == 0:
            save_records_safe(touched)
            print(f"  [checkpoint] saved. running cost estimate: ${cost_so_far:.4f}")

        time.sleep(0.05)

    save_records_safe(touched)

    total_cost = (total_input_tokens / 1e6 * PRICE_PER_1M_INPUT
                  + total_output_tokens / 1e6 * PRICE_PER_1M_OUTPUT)
    print(f"\nDone. {total_captioned} captioned, {total_failed} failed.")
    print(f"Tokens: {total_input_tokens} in / {total_output_tokens} out")
    print(f"Estimated cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
