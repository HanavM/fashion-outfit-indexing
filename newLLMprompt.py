prompt = """
You are converting scraped fashion product data into structured training supervision for a vision-language retrieval system.

Input fields may include:

* product title
* brand
* SKU or product ID
* category
* scraped product description
* color name
* material information
* product specifications

Return exactly one valid JSON object. Do not include markdown, explanations, or text outside the JSON.

Use this schema:

{
"product_id": "string or null",
"positive_texts": ["string"],
"taxonomy_path": ["string"],
"attributes": {
"color": ["string"],
"material": ["string"],
"pattern": ["string"],
"fit": ["string"],
"length": ["string"],
"silhouette": ["string"],
"closure": ["string"],
"defining_features": [
{
"feature": "string",
"location": "string or null",
"material": "string or null",
"color": "string or null"
}
]
}
}

Rules:

1. Use only information explicitly supported by the input.
2. Do not guess visual details, materials, colors, category levels, brand, model, or SKU.
3. Omit unsupported attribute groups or return empty arrays.
4. Normalize values to short lowercase phrases, except official brand and model names.
5. Keep taxonomy categories singular and general.
6. The taxonomy must describe product type, not color, material, brand, model, or SKU.
7. Arrange taxonomy from broadest to most specific supported category.

Example taxonomy:

[
"apparel",
"footwear",
"shoe",
"sneaker",
"low-top sneaker"
]

8. Treat color, material, pattern, fit, closure, and decorative details as independent attributes, not taxonomy nodes.
9. Put product-specific construction or recognizable details under `defining_features`.

Examples:

{
"feature": "suede toe overlay",
"location": "toe",
"material": "suede",
"color": "gray"
}

{
"feature": "spider-web graphic",
"location": "front",
"material": null,
"color": null
}

10. Generate 5 to 10 positive texts at different specificity levels.
11. Each positive text must be a natural, independently valid description of the product.
12. Include a balanced selection of:

* broad category text
* category plus major attributes
* brand plus category
* brand plus model
* product-specific description
* exact SKU text when a SKU is available

13. Do not generate every possible combination of attributes.
14. Do not repeat nearly identical captions.
15. Avoid promotional language such as “premium,” “iconic,” “stylish,” or “perfect.”
16. Do not include details that are not useful for visual recognition or retrieval.
17. SKU captions must contain enough context to identify the SKU as a product identifier.

Good positive texts:

[
"a low-top sneaker",
"a white leather sneaker",
"a white Adidas sneaker with a gum sole",
"a white Adidas Samba",
"an Adidas Samba with a gray suede toe overlay",
"Adidas Samba SKU 12345"
]

Bad positive texts:

[
"white",
"gray toe",
"amazing iconic sneaker",
"white sneaker gray sneaker gum sneaker",
"SKU 12345"
]

18. If the scraped description contains uncertain wording such as “appears,” “inspired,” “style,” or “look,” do not convert it into a definite fact.
19. Preserve official brand, model, and SKU spelling when explicitly provided.
20. Do not treat gendered marketing labels such as “men’s” or “women’s” as visual attributes unless they represent an actual fit or construction difference.
21. Remove empty attribute groups from the final JSON.
22. Ensure every positive text is supported by the structured fields or explicit source description.

Input:

Product title:
{{PRODUCT_TITLE}}

Brand:
{{BRAND}}

SKU or product ID:
{{SKU}}

Category:
{{CATEGORY}}

Color:
{{COLOR}}

Scraped description:
{{DESCRIPTION}}



"""