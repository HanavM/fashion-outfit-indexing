"""Phase 6 serving layer: a warm HTTP wrapper around the real retrieval
pipeline, on Modal.

Every entry point in this repo before this file was a CLI script that
loaded ~1GB of weights per invocation and printed to stdout. A voice
round trip has a budget closer to 2 seconds, so this keeps SigLIP2 +
DINOv3 + every index resident in one container
(`@modal.enter()` + a generous `scaledown_window`) and exposes them over
FastAPI.

## It runs the REAL pipeline file, deliberately

Same rule as `modal_app_phase4_eval.py`, for the same reason: a
hand-maintained Modal copy of `hierarchical_retrieval_pipeline.py` once
drifted 800+ lines behind and silently evaluated months-old code. So the
pipeline (and `composed_query_search.py` / `catalog_query_search.py` /
`free_text_visual_search.py`) are shipped verbatim via `add_local_file`
and pointed at the Volume with `APPAREL_DATASET_ROOT=/data/apparel_dataset`.
Nothing about retrieval is reimplemented here -- this file only does
transport, auth, and response shaping.

## Deploy / operate

    python3 -m modal secret create fashion-api-key FASHION_API_KEY=<secret>
    # ONE-OFF, only when the DINOv3 checkpoint or the catalog changed:
    python3 -m modal run modal_app_serve.py::build_indexes
    python3 -m modal deploy modal_app_serve.py

`build_indexes` exists because the cached identity index on the Volume was
built against *frozen base* DINOv3, and the pipeline's own freshness check
correctly invalidates it now that `finetuned_dinov3_identity_v1_supcon` is
present. Rebuilding ~6.5k gallery embeddings takes ~10 minutes of GPU --
acceptable as a one-off job, not acceptable inside `@modal.enter()` of a
request-serving container. Run it, let it `volume.commit()`, and every
container afterwards just loads the tensors.

## Endpoints

    GET  /health              -- no auth. models, checkpoints, index sizes.
    POST /identify            -- image -> ranked catalog products
    POST /compose             -- image + text -> primary item + companions

Auth on everything except /health: `X-API-Key: <secret>` (or
`Authorization: Bearer <secret>`) checked against the `fashion-api-key`
Modal Secret.

Both POST endpoints accept either JSON (`{"image_base64": "..."}`) or
multipart (`image=@file.jpg`). Both return a `spoken` field: one short
sentence for a voice assistant, separate from the rich list, because
Phase 7's Shortcut needs something to say out loud and should not be
re-deriving it from the JSON.

## Open-set rejection is surfaced, and is honestly uncalibrated

The pipeline computes `rejected_open_set`/`reject_threshold` and this API
passes both through untouched, plus `confidence` (the top result's raw
DINOv3 cosine score) so a client can decide for itself.

The threshold itself is NOT calibrated. `REJECT_SIMILARITY_THRESHOLD` is
`None` upstream on purpose (see its comment, and docs/eval_log.md
2026-08-03): the open-set eval split exists but has not been run, so no
false-accept rate has ever been measured and any number here would be
invented. Consequently the API default is `reject_threshold: null`, which
means `rejected_open_set` is always `false` -- the response says so
explicitly via `reject_threshold_calibrated: false`. A caller may pass
`reject_threshold` per request to opt into rejection at a threshold it
chooses, and `REJECT_THRESHOLD` can be set as an env default once the
open-set run produces a real number. Surfacing an uncalibrated flag as if
it were calibrated would be worse than surfacing nothing.

## Deploying a change: `modal deploy` alone is NOT enough

`modal deploy` updates the app definition but does NOT cycle a warm
container, and a warm container keeps running the OLD code and the OLD
secrets. This has now bitten twice on this app: a rotated API key kept
accepting the leaked value, and a fixed endpoint kept throwing its old
traceback -- both after a "successful" 2-second deploy.

If the deploy returns in a couple of seconds, assume nothing changed for
live traffic. To actually apply a change:

    modal app stop fashion-serve --yes && modal deploy modal_app_serve.py

"""

import os
from pathlib import Path

import modal

HERE = Path(__file__).parent

app = modal.App("fashion-serve")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "transformers", "pillow", "tqdm", "numpy",
        "accelerate", "safetensors", "sentencepiece",
        "fastapi[standard]", "python-multipart",
    )
    # The real files, not copies. See module docstring.
    .add_local_file(str(HERE / "hierarchical_retrieval_pipeline.py"),
                    "/root/hierarchical_retrieval_pipeline.py")
    .add_local_file(str(HERE / "composed_query_search.py"),
                    "/root/composed_query_search.py")
    .add_local_file(str(HERE / "catalog_query_search.py"),
                    "/root/catalog_query_search.py")
    .add_local_file(str(HERE / "free_text_visual_search.py"),
                    "/root/free_text_visual_search.py")
    .add_local_file(str(HERE / "docs" / "hierarchy.json"), "/root/docs/hierarchy.json")
)

volume = modal.Volume.from_name("fashion-dataset", create_if_missing=False)
hf_secret = modal.Secret.from_name("hf-token")
api_secret = modal.Secret.from_name("fashion-api-key")

ENV = {
    "APPAREL_DATASET_ROOT": "/data/apparel_dataset",
    # Images live on a network filesystem; the encode loop is bound by
    # per-file round-trip latency, not by the GPU (measured ~3.2 img/s on a
    # T4). Oversubscribing the loader is the right response to latency.
    # Same values modal_app_phase4_eval.py settled on.
    "IMAGE_LOADER_WORKERS": "32",
    "CATALOG_VERIFY_WORKERS": "64",
}


def _prepare_runtime():
    """Point the real pipeline at the Volume, then import it.

    Order matters: `hierarchical_retrieval_pipeline` reads
    APPAREL_DATASET_ROOT at module import time (it resolves DATASET_ROOT,
    METADATA_PATH and INDEX_DIR as module constants), so the env has to be
    set before the import, not after.
    """
    import sys

    os.environ.update(ENV)
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    import hierarchical_retrieval_pipeline as pipeline

    return pipeline


@app.function(image=image, gpu="A10G", volumes={"/data": volume},
              secrets=[hf_secret], timeout=90 * 60)
def build_indexes():
    """One-off: build/refresh every cached index on the Volume.

    Constructing a HierarchicalRetriever is exactly what the serving
    container does on cold start, so running it here first means the
    serving container's cold start is a tensor load rather than a 6,500-
    image GPU encode.
    """
    pipeline = _prepare_runtime()
    retriever = pipeline.HierarchicalRetriever()
    print(f"Gallery: {len(retriever.gallery_product_codes):,} products, "
          f"{len(retriever.identities):,} identities.")
    volume.commit()


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/data": volume},
    secrets=[hf_secret, api_secret],
    # The entire point of this file. A cold start is ~1GB of weights plus
    # index loads; 20 minutes of idle keeps that off the critical path for
    # any realistic interactive session.
    scaledown_window=20 * 60,
    timeout=15 * 60,
    min_containers=0,
)
@modal.concurrent(max_inputs=4)
class FashionService:
    @modal.enter()
    def load(self):
        import threading
        import time

        started = time.time()
        self.pipeline = _prepare_runtime()
        self.retriever = self.pipeline.HierarchicalRetriever()

        # Lexical/canonical half of the composed query. Cheap (pure string
        # index over metadata.json), so it is loaded eagerly too. Its
        # semantic fallback (FreeTextVisualSearch) is NOT loaded eagerly --
        # that one embeds the whole catalog and would add minutes to cold
        # start for a path most requests never take. Callers opt in per
        # request with "semantic_fallback": true.
        import catalog_query_search
        self.text_engine = catalog_query_search.CatalogQuerySearch()

        # One GPU at a time. FastAPI runs sync endpoints in a threadpool and
        # @modal.concurrent lets several requests land at once, but the
        # retriever holds mutable per-instance caches and a single device.
        self._gpu_lock = threading.Lock()

        self.default_reject_threshold = (
            float(os.environ["REJECT_THRESHOLD"])
            if os.environ.get("REJECT_THRESHOLD") else None
        )
        self.loaded_at = time.time()
        self.load_seconds = self.loaded_at - started
        print(f"Warm in {self.load_seconds:.1f}s.")

    # ---------------- helpers ----------------

    def _authorize(self, request):
        from fastapi import HTTPException

        expected = os.environ.get("FASHION_API_KEY")
        if not expected:
            # Fail closed. An unset secret must not silently become "open".
            raise HTTPException(status_code=503, detail="FASHION_API_KEY not configured")
        presented = request.headers.get("x-api-key")
        if not presented:
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                presented = auth[7:]
        import hmac
        if not presented or not hmac.compare_digest(presented, expected):
            raise HTTPException(status_code=401, detail="bad or missing API key")

    async def _read_payload(self, request):
        """Accept JSON with image_base64, or multipart with an image file."""
        import base64
        import binascii

        from fastapi import HTTPException

        content_type = (request.headers.get("content-type") or "").lower()
        if "multipart/form-data" in content_type:
            form = await request.form()
            upload = form.get("image")
            image_bytes = await upload.read() if upload is not None else None
            fields = {k: v for k, v in form.items() if k != "image"}
        else:
            body = await request.json() if await request.body() else {}
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="body must be a JSON object")
            fields = dict(body)
            encoded = fields.pop("image_base64", None)
            image_bytes = None
            if encoded:
                # Tolerate a data: URL prefix -- iOS Shortcuts' base64
                # encoding step is easy to wire up either way.
                if isinstance(encoded, str) and encoded.startswith("data:"):
                    encoded = encoded.split(",", 1)[-1]
                try:
                    image_bytes = base64.b64decode(encoded, validate=False)
                except (binascii.Error, ValueError) as error:
                    raise HTTPException(status_code=400, detail=f"image_base64 not decodable: {error}")
        return image_bytes, fields

    def _write_temp_image(self, image_bytes):
        """The pipeline's retrieve() takes a path (it does its own EXIF
        transpose + RGB convert via load_rgb_image). Rather than fork that,
        hand it a real file."""
        import tempfile

        from fastapi import HTTPException
        from PIL import Image, UnidentifiedImageError

        if not image_bytes:
            raise HTTPException(status_code=400, detail="no image supplied (image_base64 or multipart 'image')")
        handle = tempfile.NamedTemporaryFile(suffix=".img", delete=False)
        handle.write(image_bytes)
        handle.close()
        try:
            with Image.open(handle.name) as probe:
                probe.verify()
        except (UnidentifiedImageError, OSError) as error:
            os.unlink(handle.name)
            raise HTTPException(status_code=400, detail=f"not a decodable image: {error}")
        return handle.name

    def _run_retrieve(self, image_path, top_k, reject_threshold, use_category_gate):
        with self._gpu_lock:
            return self.retriever.retrieve(
                image_path,
                use_category_gate=use_category_gate,
                final_top_k=top_k,
                reject_threshold=reject_threshold,
            )

    def _shape_identity(self, raw, reject_threshold):
        display_brand = self.pipeline.display_brand
        results = [{
            "rank": entry["rank"],
            "product_code": entry["product_code"],
            "brand": display_brand(entry["brand"]),
            "name": entry["name"],
            "category": entry["category"],
            "model_identity": entry["model_identity"],
            "score": round(float(entry["dino_identity_score"]), 6),
        } for entry in raw["results"]]

        return {
            "results": results,
            "confidence": results[0]["score"] if results else None,
            "rejected_open_set": bool(raw["rejected_open_set"]),
            "reject_threshold": reject_threshold,
            # See module docstring: the threshold has never been calibrated
            # against a measured false-accept rate, so the API says so
            # rather than letting a client read `rejected_open_set: false`
            # as "the system checked and is confident".
            "reject_threshold_calibrated": False,
            "same_model_different_colorway_ambiguous": bool(raw["same_model_different_colorway_ambiguous"]),
            "predicted_category": {
                "node": raw["hsc_predicted_node"],
                "level": raw["hsc_predicted_level"],
                "confidence": round(float(raw["hsc_confidence"]), 6),
                "best_leaf": raw["hsc_best_leaf"],
                "best_leaf_probability": round(float(raw["hsc_best_leaf_probability"]), 6),
                "climbing_path": raw["hsc_climbing_path"],
            },
            "num_identity_candidates": raw["num_identity_candidates"],
        }

    @staticmethod
    def _short(brand, name, limit=52):
        label = f"{brand} {name}".strip()
        return label if len(label) <= limit else label[:limit].rsplit(" ", 1)[0] + "…"

    def _spoken_identify(self, shaped):
        results = shaped["results"]
        if not results:
            return "I couldn't find anything close to that in the catalog."
        top = results[0]
        label = self._short(top["brand"], top["name"])
        if shaped["rejected_open_set"]:
            return f"I'm not sure that's in the catalog. The closest thing is a {label}."
        if shaped["same_model_different_colorway_ambiguous"]:
            return f"That looks like a {label}, though I can't call the colorway."
        return f"That looks like a {label}."

    def _spoken_search(self, query, hits, semantic):
        """Text-search results as one sentence for TTS.

        Deliberately says when nothing matched rather than reading out the
        closest thing anyway -- a text query that misses should sound like a
        miss. The canonical index answering nothing is a real signal (the
        phrasing isn't in the catalog's label space at all), and suggesting
        the semantic fallback is more useful than a wrong product name.
        """
        if not hits:
            if not semantic:
                return (f"I didn't find anything matching {query}. "
                        "There may be results with semantic search turned on.")
            return f"I didn't find anything matching {query}."
        top = hits[0]
        label = self._short(top.get("brand", ""), top.get("name", ""))
        if len(hits) == 1:
            return f"I found one match for {query}: a {label}."
        return f"I found {len(hits)} matches for {query}. The closest is a {label}."

    # ---------------- ASGI ----------------

    @modal.asgi_app()
    def api(self):
        import time

        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        web = FastAPI(title="fashion retrieval", version="6.0")

        @web.get("/health")
        def health():
            return {
                "status": "ok",
                "device": self.pipeline.DEVICE,
                "warm_seconds": round(time.time() - self.loaded_at, 1),
                "container_load_seconds": round(self.load_seconds, 1),
                "models": {
                    "siglip2_checkpoint": self.siglip2_checkpoint,
                    "dinov3_checkpoint": self.dinov3_checkpoint,
                    "dinov3_projection_head": bool(self.retriever.dino_use_projection),
                },
                "index": {
                    "gallery_products": len(self.retriever.gallery_product_codes),
                    "semantic_identities": len(self.retriever.identities),
                    "hsc_leaves": len(self.retriever.hsc_leaf_ids),
                    "catalog_products": len(self.pipeline.CATALOG),
                    "brands": sorted({e["brand"] for e in self.pipeline.CATALOG.values()}),
                    "canonical_text_labels": len(self.text_engine.label_to_codes),
                },
                "open_set_rejection": {
                    "default_reject_threshold": self.default_reject_threshold,
                    "calibrated": False,
                    "note": ("reject_threshold is uncalibrated upstream "
                             "(REJECT_SIMILARITY_THRESHOLD is None); pass "
                             "reject_threshold per request to opt in."),
                },
            }

        @web.post("/identify")
        async def identify(request: Request):
            self._authorize(request)
            started = time.time()
            image_bytes, fields = await self._read_payload(request)
            path = self._write_temp_image(image_bytes)
            try:
                top_k = int(fields.get("top_k") or self.pipeline.FINAL_TOP_K)
                threshold = fields.get("reject_threshold", self.default_reject_threshold)
                threshold = float(threshold) if threshold not in (None, "") else None
                gate = str(fields.get("use_category_gate", "")).lower() in {"1", "true", "yes"}
                raw = self._run_retrieve(path, top_k, threshold, gate)
            finally:
                os.unlink(path)
            shaped = self._shape_identity(raw, threshold)
            shaped["spoken"] = self._spoken_identify(shaped)
            shaped["latency_ms"] = round((time.time() - started) * 1000, 1)
            return JSONResponse(shaped)

        @web.post("/compose")
        async def compose(request: Request):
            self._authorize(request)
            started = time.time()
            image_bytes, fields = await self._read_payload(request)
            text = (fields.get("text") or "").strip()
            if not text:
                return JSONResponse({"detail": "'text' is required"}, status_code=400)

            top_k = int(fields.get("top_k") or 10)
            threshold = fields.get("reject_threshold", self.default_reject_threshold)
            threshold = float(threshold) if threshold not in (None, "") else None
            gate = str(fields.get("use_category_gate", "")).lower() in {"1", "true", "yes"}
            # Semantic fallback loads a second full catalog embedding index
            # on first use; off by default so a voice request never pays it.
            semantic = str(fields.get("semantic_fallback", "")).lower() in {"1", "true", "yes"}

            primary = None
            if image_bytes:
                path = self._write_temp_image(image_bytes)
                try:
                    raw = self._run_retrieve(path, top_k, threshold, gate)
                finally:
                    os.unlink(path)
                primary = self._shape_identity(raw, threshold)

            import composed_query_search

            parsed = composed_query_search.parse_text_fragment(
                text, str(self.pipeline.METADATA_PATH))
            companions, filter_applied = self._companions(parsed, text, top_k, semantic)

            payload = {
                "primary": primary,
                "parsed_text_query": parsed,
                "companions": companions,
                "category_filter_applied": filter_applied,
                # composed_query_search.py's own standing caveat, carried to
                # the API edge verbatim in spirit: these are two independent
                # searches, not a pairing this system has evidence for.
                "note": ("primary and companions are TWO INDEPENDENT SEARCHES. "
                         "Nothing here shows these items were ever worn together "
                         "in a real photo -- that needs the outfit co-occurrence "
                         "index (roadmap Phase 8), which does not exist yet."),
                "latency_ms": round((time.time() - started) * 1000, 1),
            }
            payload["spoken"] = self._spoken_compose(primary, companions, parsed)
            return JSONResponse(payload)

        @web.post("/search")
        async def search(request: Request):
            """Text-only catalog search -- spec section 1's "Show me blue jeans"
            and "Show me gray suede Adidas sneakers".

            Two of the spec's four named query types had no endpoint at all
            even though catalog_query_search.py and free_text_visual_search.py
            were both finished and working; they were simply never wired to
            HTTP (docs/product_gap_analysis.md). This is that wiring, not new
            retrieval logic.

            `semantic_fallback` is opt-in and off by default for a real
            reason, not caution: the canonical path is a pure string index
            over metadata.json and answers in milliseconds, while the
            semantic path constructs FreeTextVisualSearch, which embeds the
            WHOLE catalog on first use. That is minutes on a cold container
            and would blow the ~2s voice budget for every caller, including
            the ones whose query the canonical index already answers.
            """
            # Local import, matching this file's convention everywhere else --
            # the enclosing api() scope imports only FastAPI/Request/JSONResponse,
            # so referencing HTTPException here without it is a NameError at
            # request time, not at import time.
            from fastapi import HTTPException

            self._authorize(request)
            started = time.time()
            # _read_payload returns (image_bytes, fields) in that order, and
            # does NOT require an image -- the "no image supplied" 400 lives
            # in _write_temp_image, which this endpoint never calls. So a
            # text-only body is accepted here by design.
            _, fields = await self._read_payload(request)

            query = (fields.get("query") or fields.get("text") or "").strip()
            if not query:
                raise HTTPException(status_code=400, detail="'query' is required")

            top_k = int(fields.get("top_k") or 15)
            semantic = str(fields.get("semantic_fallback", "")).lower() in {"1", "true", "yes"}

            # Held under the same lock as the retriever: the semantic path
            # touches the GPU, and CatalogQuerySearch memoises its engine on
            # first use, so two concurrent first-callers would otherwise race
            # to build it.
            with self._gpu_lock:
                hits = self.text_engine.search(query, top_k=top_k, canonical_only=not semantic)

            payload = {
                "query": query,
                "results": hits,
                "semantic_fallback_used": semantic and any(
                    h.get("match_type") == "semantic" for h in hits),
                "match_types": sorted({h.get("match_type") for h in hits if h.get("match_type")}),
                "latency_ms": round((time.time() - started) * 1000, 1),
            }
            payload["spoken"] = self._spoken_search(query, hits, semantic)
            return JSONResponse(payload)

        return web

    # ---------------- compose helpers ----------------

    @property
    def siglip2_checkpoint(self):
        found = self.pipeline.pick_first_existing(self.pipeline.SIGLIP2_CHECKPOINT_CANDIDATES)
        return str(found) if found else self.pipeline.SIGLIP2_BASE_MODEL_ID

    @property
    def dinov3_checkpoint(self):
        found = self.pipeline.pick_first_existing(self.pipeline.DINOV3_CHECKPOINT_CANDIDATES)
        return str(found) if found else self.pipeline.DINOV3_MODEL_ID

    def _companions(self, parsed, text, top_k, semantic):
        """The text half of composed_query_search.composed_search, reusing
        the warm CatalogQuerySearch instead of building one per call.

        Query the engine with the PARSED terms rather than the raw
        fragment, for composed_query_search.py's reason: slang the synonym
        table resolved ("jorts") is generally not a substring of any real
        canonical label, only the taxonomy term it resolved to is.
        """
        parsed_category = parsed["category"] or {}
        category_term = parsed_category.get("leaf") or parsed_category.get("category")
        attribute_keywords = [a["keyword"] for a in parsed["attributes"]]
        if category_term or attribute_keywords:
            query = " ".join(attribute_keywords + ([category_term] if category_term else []))
        else:
            query = text

        with self._gpu_lock if semantic else _null_lock():
            hits = self.text_engine.search(query, top_k=top_k * 3, canonical_only=not semantic)

        target_category = parsed_category.get("category")
        filter_applied = False
        if target_category:
            hierarchy = self.pipeline._hierarchy
            code_to_product = self.text_engine.code_to_product
            filtered = []
            for hit in hits:
                product = code_to_product.get(hit["product_code"], {})
                canonical_path = (product.get("structured_caption") or {}).get("canonical_taxonomy_path") or []
                product_category = None
                for _group, categories in hierarchy.items():
                    for category, leaves in categories.items():
                        names = {category.lower()} | {leaf.lower() for leaf in leaves}
                        if any(str(p).lower() in names for p in canonical_path):
                            product_category = category
                if product_category and product_category.lower() == target_category.lower():
                    filtered.append(hit)
            if filtered:
                hits = filtered
                filter_applied = True

        display_brand = self.pipeline.display_brand
        return [{
            "rank": rank,
            "product_code": hit["product_code"],
            "brand": display_brand(hit.get("brand", "")),
            "name": hit.get("name", ""),
            "match_type": hit["match_type"],
            "matched_label": hit.get("matched_label"),
            "score": round(float(hit.get("score") or 0.0), 6),
        } for rank, hit in enumerate(hits[:top_k], start=1)], filter_applied

    def _spoken_compose(self, primary, companions, parsed):
        if primary is None:
            head = ""
        else:
            head = self._spoken_identify(primary) + " "
        if not companions:
            wanted = (parsed.get("category") or {}).get("category") or "that"
            return (head + f"I couldn't find anything matching {wanted} in the catalog.").strip()
        top = companions[0]
        label = self._short(top["brand"], top["name"])
        return (head + f"It'd go with a {label}.").strip()


class _null_lock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@app.local_entrypoint()
def main():
    """`modal run modal_app_serve.py` == the one-off index build."""
    build_indexes.remote()
