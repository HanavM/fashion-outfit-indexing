# Fashion Outfit Indexing System — Project Specification

**Version:** 1.0  
**Date:** 2026-07-17  
**Status:** Baseline architecture for implementation and evaluation  
**Primary audience:** Engineering team and coding agents

---

## 1. Objective

Build an index over outfit images that:

1. Detects each visible fashion item.
2. Describes each item at the most specific defensible level.
3. Identifies an exact product or colorway when evidence supports it.
4. Falls back safely to broader labels when exact identification is uncertain.
5. Supports mixed image-and-text queries such as:
   - “Show me this shoe with cargo jorts.”
   - “Show me blue jeans.”
   - “Show me this exact sneaker.”
   - “Show me gray suede Adidas sneakers.”

An exact product is only one possible label. The system must preserve broader categories and attributes so exact products remain discoverable through broad queries.

---

## 2. Core design decisions

### 2.1 Use a hybrid index

Every detected item must have:

- Structured metadata and hierarchical labels
- A semantic image embedding
- A visual-identity embedding
- Confidence and provenance for every inferred fact
- Optional local patch features for reranking

Do not rely exclusively on captions, one embedding, or exact catalog matching.

### 2.2 Use two encoder roles

#### SigLIP 2 — semantic encoder

Primary responsibilities:

- Text-to-image retrieval
- Category recognition
- Attribute recognition
- Broad visual similarity
- Open-vocabulary queries
- Matching phrases such as “blue jeans,” “cargo shorts,” and “gray suede sneaker”

SigLIP 2 is the default semantic backbone because it directly aligns text and images and reports improvements in semantic understanding, retrieval, localization, and dense features over earlier SigLIP models.

#### DINOv3 — visual-identity encoder

Primary responsibilities:

- Exact or near-exact product retrieval
- Same-model and same-colorway matching
- Fine visual differences
- Local patch comparison
- Candidate reranking
- Details such as panel geometry, stitching, sole shape, pocket placement, embroidery, and logo position

DINOv3 is the default visual backbone because it is trained for strong general and dense visual representations without depending on captions.

### 2.3 FashionCLIP is a benchmark, not the default architecture

FashionCLIP must be included in evaluation because fashion-specific training may help on some tasks. It is not assumed to be the final model because:

- It is based on an older CLIP architecture.
- Its domain training is centered on fashion catalog image-text pairs.
- It does not remove the need for a separate fine-grained identity representation.
- Previous caption-pair fine-tuning degraded exact-shoe retrieval in the current dataset.

The final model choice must be determined by held-out benchmark results.

### 2.4 Structured facts are the source of truth

Free-form captions may assist search and explanation, but they must not be the authoritative database representation.

Preferred representation:

```json
{
  "category": "jeans",
  "color": ["blue"],
  "material": ["denim"],
  "fit": ["loose"],
  "features": ["faded wash"],
  "brand": {
    "value": "Gap",
    "confidence": 0.91,
    "evidence": ["visible waistband patch"]
  },
  "product_candidates": [],
  "display_label": "Gap blue jeans"
}
```

### 2.5 Specificity must depend on confidence

Example hierarchy:

```text
apparel
└── bottoms
    └── pants
        └── jeans
```

Independent facets:

```text
color = blue
material = denim
brand = Gap
fit = loose
feature = cargo pockets
product_id = optional
```

The output should back off automatically:

```text
exact product
→ model/colorway
→ brand + category
→ attributes + category
→ category
```

Never force an exact product prediction.

---

## 3. End-to-end architecture

```text
Outfit image
    ↓
Item detection and segmentation
    ↓
Per-item crop and mask
    ↓
┌───────────────────────────────────────────────┐
│ SigLIP 2 semantic embedding                  │
│ DINOv3 visual-identity embedding             │
│ Category and attribute heads                 │
│ Logo detection and OCR                       │
│ Optional VLM structured evidence extraction  │
└───────────────────────────────────────────────┘
    ↓
Confidence calibration and label backoff
    ↓
Structured item record
    ↓
Metadata index + semantic vector index + identity vector index
    ↓
Candidate retrieval
    ↓
Patch-level and multimodal reranking
    ↓
Outfit-level results
```

---

## 4. Pipeline stages

## 4.1 Image ingestion

For every source image, store:

- Source URI or file identifier
- Image hash
- Perceptual hash
- Width and height
- Source domain
- Capture type: catalog, editorial, street, social, unknown
- Ingestion timestamp
- Licensing or usage metadata where applicable

Deduplicate exact and near-duplicate images before train/test splitting.

## 4.2 Item detection and segmentation

Detect and isolate:

- Shoes
- Pants
- Shorts
- Skirts
- Tops
- Jackets
- Dresses
- Bags
- Hats
- Other supported fashion categories

Outputs:

```json
{
  "item_id": "item_123",
  "outfit_id": "outfit_456",
  "bounding_box": [x1, y1, x2, y2],
  "mask_uri": "...",
  "detection_category": "shoe",
  "detection_confidence": 0.98,
  "visibility": 0.74,
  "occlusion": 0.22
}
```

Use fashion-specific datasets and taxonomies as initial references. DeepFashion2 provides detection, segmentation, viewpoint, occlusion, and consumer-to-commercial pairs. Fashionpedia provides an apparel ontology, masks, and fine-grained attributes.

## 4.3 Semantic representation

Generate and store a normalized SigLIP 2 embedding for every item crop.

Use it for:

- Text query retrieval
- Broad semantic image retrieval
- Zero-shot prototype scoring
- Category and attribute initialization
- Candidate generation

Example text prototypes:

```text
shoe
sneaker
low-top sneaker
gray sneaker
gray suede sneaker
Adidas sneaker
Adidas Gazelle
```

Do not treat prototype similarity as calibrated probability without validation.

## 4.4 Visual-identity representation

Generate and store a normalized DINOv3-based identity embedding.

Initial implementation:

```text
DINOv3 backbone
→ pooled visual features
→ trainable projection head
→ L2-normalized identity vector
```

Later implementation:

- Partial backbone adaptation
- Multi-view aggregation
- Region or patch descriptors
- Category-specific projection heads if justified by evaluation

Use this representation for exact or near-exact visual matching, not broad text search.

## 4.5 Structured label extraction

Predict independent fields:

### Category

Examples:

```text
footwear > sneaker > low-top sneaker
bottoms > pants > jeans
bottoms > shorts > denim shorts
```

### Attributes

Initial attribute groups:

- Color
- Material
- Pattern
- Fit
- Length
- Silhouette
- Closure
- Pocket type
- Distressing
- Heel type
- Sole type
- Toe shape
- Visible decorative details

### Brand evidence

Brand must be nullable.

Evidence sources:

- Visible logo
- OCR
- Trademark pattern
- Distinctive construction
- Catalog-candidate agreement

Do not assign a brand solely from weak style resemblance.

### Product identity

Store ranked candidates:

```json
[
  {
    "product_id": "catalog_abc",
    "score": 0.84,
    "evidence": ["visual_identity", "ocr", "attribute_match"]
  }
]
```

Promote a candidate to an asserted product label only after calibration thresholds are satisfied.

## 4.6 Canonical label generation

Generate searchable labels from structured facts.

Example:

```text
jeans
blue jeans
blue denim jeans
loose blue jeans
Gap jeans
Gap blue jeans
loose Gap blue denim jeans
```

All labels point to the same item record.

Free-form descriptions may be generated for display:

```text
Loose blue denim jeans with a faded wash and visible Gap branding.
```

The description must not introduce facts absent from structured predictions.

## 4.7 Index storage

Each item should be stored approximately as:

```json
{
  "item_id": "item_123",
  "outfit_id": "outfit_456",
  "category_path": ["apparel", "bottoms", "pants", "jeans"],
  "attributes": {
    "color": [{"value": "blue", "confidence": 0.97}],
    "material": [{"value": "denim", "confidence": 0.94}],
    "fit": [{"value": "loose", "confidence": 0.78}]
  },
  "brand": {
    "value": "Gap",
    "confidence": 0.91,
    "evidence": ["ocr", "logo"]
  },
  "product_candidates": [],
  "canonical_labels": [
    "jeans",
    "blue jeans",
    "blue denim jeans",
    "Gap jeans",
    "Gap blue jeans"
  ],
  "semantic_embedding_ref": "siglip:item_123",
  "identity_embedding_ref": "dino:item_123",
  "patch_embedding_ref": null,
  "crop_uri": "...",
  "mask_uri": "...",
  "quality": {
    "visibility": 0.74,
    "occlusion": 0.22,
    "blur": 0.08
  },
  "model_versions": {
    "detector": "...",
    "semantic_encoder": "...",
    "identity_encoder": "...",
    "attribute_model": "..."
  }
}
```

Maintain separate indexes for:

- Metadata/filter search
- Semantic vectors
- Identity vectors
- Optional lexical search over canonical labels and descriptions

## 4.8 Query execution

### Text-only query: “blue jeans”

1. Parse into facets:
   - category = jeans
   - color = blue
2. Search structured metadata.
3. Search SigLIP semantic vectors.
4. Merge and rank.
5. Include exact branded products that inherit these broader facts.

### Image-only query: exact shoe

1. Detect/crop the shoe.
2. Generate SigLIP and DINOv3 embeddings.
3. Retrieve primarily through the identity index.
4. Use semantic and metadata compatibility as supporting evidence.
5. Rerank top candidates with local patch features.
6. Return exact identity only if calibrated confidence is sufficient.

### Image + text query: “this shoe with cargo jorts”

1. Identify or embed the shoe.
2. Parse “cargo jorts” into:
   - category = denim shorts
   - feature = cargo pockets
3. Search outfit records containing:
   - a strong shoe match
   - a shorts item satisfying the semantic/attribute query
4. Rank by:
   - shoe-match score
   - cargo-jorts score
   - item visibility
   - image quality
   - contradiction penalties

---

## 5. Training strategy

## 5.1 First establish frozen baselines

Evaluate without fine-tuning:

- FashionCLIP
- SigLIP 2
- DINOv3
- Combined SigLIP 2 + DINOv3

Do not modify pretrained weights until these baselines are recorded.

## 5.2 DINOv3 identity adaptation

Train from visual-product identities, not long captions.

### Positive pairs

- Different views of the same product/colorway
- Catalog and worn images of the same product
- Crops with realistic viewpoint and occlusion differences

### Hard negatives

Prioritize:

- Same model, different colorway
- Same brand, similar silhouette
- Visually similar competing models
- Same category and dominant color
- Near-identical construction with one discriminating detail

### Batch construction

Use identity-balanced batches:

```text
P identities × K images per identity
```

Require `K >= 2` so each identity has genuine in-batch positives.

### Candidate losses

Benchmark:

- Supervised contrastive loss
- ArcFace-style angular-margin loss
- Proxy-based metric losses
- Classification + metric-learning combination

### Fine-tuning order

1. Freeze DINOv3; train projection head.
2. Evaluate.
3. Unfreeze only final blocks with a lower learning rate.
4. Evaluate generalization and forgetting.
5. Continue only if held-out retrieval improves.

## 5.3 SigLIP 2 adaptation

Start frozen.

Train separate heads or projections for:

- Category hierarchy
- Multi-label attributes
- Optional fashion-specific semantic projection

Use multiple valid texts per image:

```text
sneaker
red sneaker
red suede sneaker
Adidas sneaker
Adidas Gazelle
Adidas Gazelle Bold
red Adidas Gazelle Bold with floral embroidery
```

All valid descriptions for the same product must be treated as positives. Do not use a standard one-image/one-caption loss that marks equivalent captions as negatives.

Avoid full-model fine-tuning on the current small dataset until the frozen and lightweight-adaptation baselines are complete.

## 5.4 Data augmentation

Preserve identity-critical evidence.

Recommended:

- Modest resize/crop
- Mild brightness and contrast changes
- Blur and compression simulation
- Partial occlusion
- Background variation
- Perspective changes

Avoid or limit:

- Strong hue shifts
- Grayscale
- Aggressive crops removing identity features
- Augmentations that erase logos or sole geometry
- Arbitrary horizontal flips where side-specific details matter

## 5.5 Data requirements

The current shoe dataset is useful for an initial identity experiment but is not sufficient to validate the complete system.

Required expansion:

- Worn/street images
- Catalog-to-consumer positive pairs
- Multiple viewpoints
- Hard-negative colorways
- Unknown products for open-set testing
- Non-shoe categories
- Attribute annotations
- Visibility and occlusion annotations

---

## 6. Candidate reranking

Retrieve a small candidate pool before expensive comparison.

Suggested flow:

```text
metadata candidates
+ SigLIP candidates
+ DINOv3 candidates
+ OCR/logo candidates
→ union and deduplicate
→ top 20–100 candidates
→ local visual and multimodal reranking
```

Reranking evidence:

- Patch-level DINO similarity
- Shape and part geometry
- OCR consistency
- Logo consistency
- Attribute agreement
- Contradictions
- Agreement across catalog views

A multimodal language model may explain or verify evidence among a small candidate set. It must not scan the entire database or invent product facts from memory.

---

## 7. Confidence and open-set behavior

The system must support:

- Exact match
- Model or family match
- Brand-level match
- Attribute/category match
- Unknown

Example:

```json
{
  "category": {"value": "sneaker", "confidence": 0.998},
  "brand": {"value": "Adidas", "confidence": 0.96},
  "model_family": {"value": "Gazelle", "confidence": 0.87},
  "product_id": {"value": null, "confidence": 0.42},
  "display_label": "Adidas Gazelle sneaker"
}
```

Thresholds must be calibrated on held-out known and unknown products.

Useful signals:

- Top candidate score
- Margin between top two candidates
- Agreement across views
- Semantic/visual/OCR agreement
- Input quality
- Distance from known identity distributions

---

## 8. Evaluation

## 8.1 Required data splits

### Closed-set product recognition

Products appear in training, but query images are held out.

### Catalog-to-consumer retrieval

Catalog images form the gallery; worn or real-world images form the queries.

### Unseen-product enrollment

Entire identities are excluded from training and added to the gallery only at evaluation.

### Open-set rejection

Queries include products absent from the catalog.

### Broad semantic retrieval

Queries include categories and attributes such as:

- blue jeans
- black cargo shorts
- gray suede sneakers
- striped shirt

### Outfit conjunction retrieval

Queries require two or more items in the same outfit.

Prevent source-level and near-duplicate leakage across splits.

## 8.2 Metrics

### Detection

- Bounding-box mAP
- Mask mAP
- Recall by category
- Recall by visibility and occlusion

### Semantic labeling

- Hierarchical category accuracy
- Multi-label precision, recall, and F1
- Brand precision at chosen coverage
- Calibration error
- Unsupported-fact rate

### Semantic retrieval

- Recall@1, Recall@5, Recall@10
- Mean average precision
- NDCG where graded relevance exists

### Product identity

- Recall@1 and Recall@5
- Mean average precision
- Same-model/different-colorway error rate
- Consumer-to-catalog retrieval
- Unknown-product AUROC or AUPR

### Final system

- Exact result success rate
- Broad-query coverage
- Outfit conjunction precision
- Latency
- Index size
- Cost per indexed image
- Cost per query

## 8.3 Model selection rule

No model is selected based on reputation or general benchmark claims.

Choose models using a weighted validation score reflecting product requirements:

```text
semantic retrieval
+ exact identity retrieval
+ open-set rejection
+ attribute accuracy
+ latency/cost
```

Keep per-task scores visible; do not hide weaknesses in one aggregate metric.

---

## 9. MVP implementation plan

## Phase 1 — Shoe retrieval benchmark

Deliver:

- Clean train/validation/test split
- Frozen FashionCLIP, SigLIP 2, and DINOv3 baselines
- Exact shoe Recall@K
- Text-to-shoe retrieval benchmark
- Error analysis by model family and colorway

Exit condition:

- Reliable evidence showing which model is best for each task.

## Phase 2 — Dual-encoder shoe index

Deliver:

- SigLIP semantic vector
- DINO identity vector
- Structured shoe attributes
- Combined retrieval
- Open-set rejection
- Candidate reranking

Exit condition:

- Combined system outperforms either encoder alone on the weighted validation criteria.

## Phase 3 — Real outfit images

Deliver:

- Shoe detector and cropper
- Catalog-to-consumer evaluation
- Occlusion and low-resolution handling
- Indexing of outfit images containing shoes

Exit condition:

- Stable performance on non-catalog imagery.

## Phase 4 — Additional apparel categories

Deliver:

- Jeans, shorts, tops, and jackets
- Category ontology
- Attribute schema
- Hierarchical label backoff
- Broad text retrieval

Exit condition:

- “Blue jeans” retrieves both generic and exact branded jeans correctly.

## Phase 5 — Composed outfit search

Deliver:

- Image + text query parser
- Multi-item outfit filtering
- “This shoe with cargo jorts” workflow
- Outfit-level ranking and explanation

Exit condition:

- High precision on manually labeled conjunction queries.

---

## 10. Non-goals and prohibited shortcuts

Do not:

- Treat every item as an exact catalog product.
- Force a brand when no reliable brand evidence exists.
- Use one free-form caption as the database truth.
- Use only one universal embedding without benchmarking.
- Fully fine-tune a large encoder on the current small dataset before frozen baselines.
- Randomly split near-duplicate product images across training and test.
- Evaluate only on catalog images.
- Treat cosine similarity as a calibrated probability.
- Let an LLM invent SKU-level facts.
- Permanently couple the system to one model checkpoint or vector database.

---

## 11. Initial implementation interfaces

Suggested service boundaries:

```text
ingestion_service
detector_service
semantic_encoder_service
identity_encoder_service
attribute_service
ocr_logo_service
index_writer
query_parser
candidate_retriever
candidate_reranker
confidence_calibrator
outfit_search_api
```

Required versioning:

- Dataset version
- Taxonomy version
- Model checkpoint
- Projection-head version
- Index version
- Threshold/calibration version

Every indexed fact should be reproducible from its model and pipeline versions.

---

## 12. Key experiments to run immediately

1. Compare frozen FashionCLIP, SigLIP 2, and DINOv3 on the same shoe split.
2. Compare long-caption CLIP training against multi-positive hierarchical texts.
3. Train only a DINOv3 projection head with identity-balanced batches.
4. Add same-model/different-colorway hard negatives.
5. Test combined SigLIP + DINO score fusion.
6. Add unknown shoes and calibrate identity rejection.
7. Create a small real-world worn-shoe test set.
8. Measure whether patch-level reranking improves top-1 accuracy.
9. Audit predictions for unsupported brand and material claims.
10. Record failures by visibility, view, colorway similarity, and image source.

---

## 13. Final architectural summary

```text
Detect the fashion item.
Use SigLIP 2 to understand and search its meaning.
Use DINOv3 to match its detailed visual identity.
Extract structured categories, attributes, logos, and text.
Search metadata, semantic vectors, and identity vectors together.
Rerank a small candidate set using local visual evidence.
Return the most specific label justified by calibrated confidence.
Preserve every broader parent label for future retrieval.
```

---

## 14. Research references

1. Tschannen et al., **SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features** (2025).  
   https://arxiv.org/abs/2502.14786

2. Siméoni et al., **DINOv3** (2025).  
   https://arxiv.org/abs/2508.10104

3. Chia et al., **FashionCLIP: Connecting Language and Images for Product Representations** (2022/2023).  
   https://arxiv.org/abs/2204.03972

4. Ge et al., **DeepFashion2: A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images** (CVPR 2019).  
   https://arxiv.org/abs/1901.07973

5. Jia et al., **Fashionpedia: Ontology, Segmentation, and an Attribute Localization Dataset** (ECCV 2020).  
   https://arxiv.org/abs/2004.12276

6. Wu et al., **Fashion IQ: A New Dataset Towards Retrieving Images by Natural Language Feedback** (CVPR 2021).  
   https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Fashion_IQ_A_New_Dataset_Towards_Retrieving_Images_by_Natural_Language_Feedback_CVPR_2021_paper.html

7. Deng et al., **ArcFace: Additive Angular Margin Loss for Deep Face Recognition** (CVPR 2019).  
   https://arxiv.org/abs/1801.07698

---

## 15. Decision log

### Accepted

- Hybrid structured + semantic + visual index
- SigLIP 2 as default semantic encoder
- DINOv3 as default identity encoder
- FashionCLIP retained as a benchmark
- Confidence-based specificity backoff
- Multi-channel candidate retrieval and reranking
- Exact product identity as optional, not mandatory

### Must still be validated

- Best checkpoint sizes
- Whether one shared backbone can meet cost constraints
- Best metric-learning loss
- Best detector
- Best fusion strategy
- Best vector-index implementation
- Thresholds for brand, model, and exact-product assertions
- Whether VLM reranking improves enough to justify its cost
