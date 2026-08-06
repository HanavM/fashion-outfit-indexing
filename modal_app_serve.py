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
    # /query's rules-based router. Imported lazily inside _route_query, so
    # forgetting this line would fail at REQUEST time on /query only --
    # every other endpoint and the health check would look fine.
    .add_local_file(str(HERE / "query_router.py"), "/root/query_router.py")
    .add_local_file(str(HERE / "docs" / "hierarchy.json"), "/root/docs/hierarchy.json")
    # Outfit co-occurrence: the module that defines OutfitCooccurrence and
    # the index itself. Both are required -- without them _outfit_evidence
    # degrades silently to "no evidence", which looks identical to a pair
    # the index genuinely has nothing for, so the failure would be
    # invisible rather than loud.
    .add_local_file(str(HERE / "build_outfit_cooccurrence.py"),
                    "/root/build_outfit_cooccurrence.py")
    .add_local_file(str(HERE / "outfit_cooccurrence.json"),
                    "/root/outfit_cooccurrence.json")
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
            # The garment gate, unlike open-set rejection, IS calibrated --
            # AUROC 0.9994 against 507 real non-clothing photos, operating
            # point +0.010 for FR 2.50% / FA 0.39% (docs/eval_log.md
            # 2026-08-04). The pipeline reports it without enforcing so the
            # eval path is unaffected; enforcing is this layer's job.
            #
            # `looks_like_clothing: false` does NOT empty `results`. The
            # closest matches stay visible and the client decides -- same
            # discipline as open-set rejection, and it keeps a false-reject
            # (2.5%, and by measurement ALL of them are worn outfits, i.e.
            # real users) recoverable rather than a dead end.
            "garment_gate": {
                "score": round(float(raw["garment_gate"]["score"]), 6),
                "threshold": raw["garment_gate"]["threshold"],
                "looks_like_clothing": bool(raw["garment_gate"]["passed"]),
                "calibrated": True,
            },
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
        # Don't prepend the brand when the name already starts with it.
        # Many catalog names embed it ("Nike Pro Dri-FIT", "Gap x Awake
        # NY ..."), and the naive concatenation produced "Nike Nike Pro
        # Dri-FIT". Harmless in JSON, but this string is SPOKEN, and a
        # doubled brand is immediately audible as broken.
        brand = (brand or "").strip()
        name = (name or "").strip()
        if brand and name.lower().startswith(brand.lower()):
            label = name
        else:
            label = f"{brand} {name}".strip()
        return label if len(label) <= limit else label[:limit].rsplit(" ", 1)[0] + "…"

    def _spoken_identify(self, shaped):
        results = shaped["results"]
        if not results:
            return "I couldn't find anything close to that in the catalog."
        # Checked before anything else: if there is no clothing in the
        # frame, naming the closest catalog product is exactly the
        # confident-nonsense failure this gate exists to stop. Said out
        # loud rather than silently returning matches, because the spoken
        # line is the whole answer for a voice surface.
        gate = shaped.get("garment_gate") or {}
        if gate and not gate.get("looks_like_clothing", True):
            return "I don't see any clothing in that image."
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
            return JSONResponse(self._do_identify(image_bytes, fields, started))

        @web.post("/compose")
        async def compose(request: Request):
            self._authorize(request)
            started = time.time()
            image_bytes, fields = await self._read_payload(request)
            text = (fields.get("text") or "").strip()
            if not text:
                return JSONResponse({"detail": "'text' is required"}, status_code=400)
            return JSONResponse(self._do_compose(image_bytes, fields, text, started))

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

            return JSONResponse(self._do_search(fields, query, started))

        @web.post("/query")
        async def query(request: Request):
            """The single entry point -- `docs/unified_query_design.md`.

            Accepts `{image_base64?, text?, top_k?}` (or multipart with an
            `image` file), infers the query's SHAPE with cheap lexical
            rules, and dispatches to whichever of the three measured paths
            answers it. `route` is reported in the response so a caller
            can always see -- and a test can always assert -- which path
            ran and why.

            This is a ROUTING change, not a retrieval change. It calls
            `_do_identify` / `_do_compose` / `_do_search`, which are the
            exact functions `/identify` / `/compose` / `/search` call, so
            parity holds by construction rather than by a second
            implementation kept in sync by hand. `query_parity_check.py`
            asserts that against the live service.

            Deliberately NOT done here: blending the semantic and identity
            scores into one ranking. That is unifying the representation
            rather than the interface, and it measured -6.22pt R@1 with
            shortlist miss identical in both arms.
            """
            from fastapi import HTTPException

            self._authorize(request)
            started = time.time()
            image_bytes, fields = await self._read_payload(request)

            text = (fields.get("text") or fields.get("query") or "").strip()
            decision = self._route_query(bool(image_bytes), text)

            if decision["intent"] == "unroutable":
                raise HTTPException(
                    status_code=400,
                    detail="supply an image, text, or both "
                           "(image_base64 / multipart 'image', and/or 'text')")

            if decision["intent"] == "search":
                payload = self._do_search(fields, text, started)
            elif decision["intent"] == "compose":
                payload = self._do_compose(
                    image_bytes, fields, decision["companion_text"], started)
            else:
                # identify AND brand: the same retrieval, because the brand
                # IS a field on the identified catalog record. There is no
                # separate brand-reading path worth routing to -- the logo
                # detector built for that turned out to be a brand-
                # PHOTOGRAPHY-STYLE classifier (it scores 83.95% at 32x32
                # where no mark is legible, and separates catalog photos
                # from real ones better than it separates brands).
                payload = self._do_identify(image_bytes, fields, started)
                if decision["intent"] == "brand":
                    payload["spoken"] = self._spoken_brand(payload)

            payload["route"] = {
                "intent": decision["intent"],
                "equivalent_endpoint": decision["path"],
                "reason": decision["reason"],
                "signals": decision["signals"],
                "router": "rules",
            }
            # Recomputed after routing so it covers the routing decision
            # too, not just the retrieval the sub-handler timed.
            payload["latency_ms"] = round((time.time() - started) * 1000, 1)
            return JSONResponse(payload)

        return web

    # ---------------- shared endpoint implementations ----------------
    #
    # These exist so `/query` and the three original endpoints run the
    # SAME code rather than two implementations that agree until someone
    # edits one of them. Each takes an already-parsed payload and returns
    # a plain dict; the HTTP wrappers above only do auth, body parsing,
    # and their own 400s.

    def _do_identify(self, image_bytes, fields, started):
        import time

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
        return shaped

    def _do_compose(self, image_bytes, fields, text, started):
        import time

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

        # Real co-occurrence evidence, when the index has any for this
        # anchor/companion pair. Built 2026-08-04 from 6,860 outfit
        # photos; before it existed this endpoint returned a hardcoded
        # note saying it did not, which is now false.
        outfit_evidence = self._outfit_evidence(primary, parsed)

        payload = {
            "primary": primary,
            "parsed_text_query": parsed,
            "companions": companions,
            "category_filter_applied": filter_applied,
            "outfit_evidence": outfit_evidence,
            # The caveat is CONDITIONAL, because the honest answer differs
            # by case. With evidence, these items were seen together in
            # real photos -- by an unvalidated detector, so that qualifier
            # travels with the claim rather than being dropped at the API
            # edge. Without it, the old warning still holds exactly.
            "note": (
                "companions are supported by co-occurrence in real outfit photos "
                "(see outfit_evidence). Those labels come from an UNVALIDATED "
                "detector (SAM2 + zero-shot FashionCLIP) over an unlabelled "
                "corpus -- evidence of what the model saw, not measured fact."
                if outfit_evidence else
                "primary and companions are TWO INDEPENDENT SEARCHES. Nothing "
                "here shows these items were ever worn together in a real photo. "
                "The co-occurrence index exists but has no evidence for this "
                "particular pair."
            ),
            "latency_ms": round((time.time() - started) * 1000, 1),
        }
        payload["spoken"] = self._spoken_compose(primary, companions, parsed)
        return payload

    def _do_search(self, fields, query, started):
        import time

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
        return payload

    # ---------------- routing ----------------

    def _route_query(self, has_image, text):
        """Rules-based intent inference for `/query`. See `query_router`.

        The taxonomy parser is bound once and memoised: it reads the whole
        catalog metadata to build its attribute vocabulary, and doing that
        per request would put a file read in front of every voice query."""
        import query_router

        if getattr(self, "_query_parser", None) is None:
            try:
                self._query_parser = query_router.bind_parser(self.pipeline.METADATA_PATH)
            except Exception as error:  # noqa: BLE001
                # Routing degrades rather than fails: without the parser,
                # bare-"with" companion detection is off and those queries
                # route to /identify, which ignores the text instead of
                # acting on a misparse.
                print(f"[route] taxonomy parser unavailable, "
                      f"bare-'with' detection disabled: {error}")
                self._query_parser = False
        parser = self._query_parser or None
        return query_router.route(has_image, text, parser)

    def _spoken_brand(self, shaped):
        """"What brand is this" deserves a brand-shaped answer, not the
        full product name -- but it must carry the same confidence caveat
        as `/identify`, because it IS `/identify`. Confidence is not
        reliability here: on the jacket miss the WRONG answer scored 0.922
        against the right one's 0.854."""
        results = shaped.get("results") or []
        if shaped.get("rejected_open_set") or not results:
            return "I couldn't match that to anything in the catalog, so I can't name a brand."
        gate = shaped.get("garment_gate") or {}
        if gate.get("looks_like_clothing") is False:
            return "That doesn't look like clothing to me, so I'd rather not guess a brand."
        # `brand` is already display-mapped by _shape_identity.
        top = results[0]
        brand = top.get("brand") or "unknown"
        runner_up = results[1].get("brand") if len(results) > 1 else None
        answer = f"Looks like {brand}"
        if runner_up and runner_up != brand:
            answer += f", though {runner_up} is close"
        return answer + f". Best match: {top.get('name', 'unnamed product')}."

    # ---------------- compose helpers ----------------

    @property
    def siglip2_checkpoint(self):
        found = self.pipeline.pick_first_existing(self.pipeline.SIGLIP2_CHECKPOINT_CANDIDATES)
        return str(found) if found else self.pipeline.SIGLIP2_BASE_MODEL_ID

    @property
    def dinov3_checkpoint(self):
        found = self.pipeline.pick_first_existing(self.pipeline.DINOV3_CHECKPOINT_CANDIDATES)
        return str(found) if found else self.pipeline.DINOV3_MODEL_ID

    def _outfit_evidence(self, primary, parsed):
        """Co-occurrence support for this anchor->companion pair, or None.

        Needs BOTH a category for the query image and one parsed out of the
        text; with either missing there is no pair to look up. Returns None
        rather than raising if the index is absent, so the endpoint keeps
        working on a Volume that predates it.
        """
        if not primary or not parsed:
            return None

        import composed_query_search

        # BOTH sides need mapping into the index's vocabulary, and getting
        # either wrong fails SILENTLY as "no evidence" -- indistinguishable
        # from a pair the index genuinely lacks. The first version of this
        # got both wrong and looked like it worked:
        #
        #   anchor: predicted_category.best_leaf is an HSC LEAF ("shirt
        #     jacket"); the index keys on the 13 hierarchy CATEGORIES
        #     ("jacket"). _taxonomy_term_to_category() is the mapping, and
        #     product_category()'s own docstring says category level is
        #     "the only granularity the two indexes can be joined on".
        #   companion: parse_text_fragment returns category as a DICT
        #     {'leaf': 'cargo pants', 'category': 'pants', ...}, not a string.
        lookup = composed_query_search._taxonomy_term_to_category()
        predicted = primary.get("predicted_category") or {}
        anchor = None
        for term in (predicted.get("best_leaf"), predicted.get("node")):
            if term:
                anchor = lookup.get(str(term).lower())
                if anchor:
                    break

        companion_field = parsed.get("category")
        if isinstance(companion_field, dict):
            companion = companion_field.get("category")
        else:
            companion = companion_field

        if not anchor or not companion:
            return None
        index = getattr(self, "_cooccurrence", "unset")
        if index == "unset":
            # Loaded once and cached, including the None result -- a Volume
            # that predates the index should not retry a missing file on
            # every request.
            try:
                from build_outfit_cooccurrence import OutfitCooccurrence
                index = OutfitCooccurrence.load_if_available(
                    composed_query_search.COOCCURRENCE_PATH)
            except Exception as error:  # missing/unreadable -- not fatal
                print(f"outfit evidence unavailable: {type(error).__name__}: {error}")
                index = None
            self._cooccurrence = index
        if index is None:
            return None
        try:
            evidence = index.evidence_for(anchor, companion)
        except Exception as error:
            print(f"outfit evidence lookup failed: {type(error).__name__}: {error}")
            return None
        if not evidence:
            return None
        if evidence.get("cooccurrence_count", 0) < composed_query_search.MIN_EVIDENCE_OUTFITS:
            return None
        return {
            **evidence,
            "anchor_category": anchor,
            "companion_category": companion,
            "labels_are_ground_truth": False,
        }

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
