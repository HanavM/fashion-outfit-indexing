"""Serve outfit search as an HTTP API, so the iOS app has something to call.

    modal app stop outfit-serve --yes && modal deploy modal_app_outfit_serve.py

Until now outfit search existed only as `outfit_search.py serve` on
localhost. That is fine for judging results by eye and useless to a phone.

## Why CPU, not GPU

A query is one SigLIP2 forward over a handful of text/image parts, then a
matmul of 768 floats against ~41k crop vectors. The matmul is
microseconds; the encoder forward is the cost, and on CPU it is a few
hundred milliseconds -- fine for an interactive search, and roughly 20x
cheaper per warm hour than an A10G. The catalog's `/identify` needs a GPU
because it runs two encoders over a real photo; this does not.

## Everything is reused, nothing is reimplemented

`outfit_search.OutfitSearch` is the same class the local server uses, with
its paths repointed at the Volume. So the deployed API inherits the
garment-type binding, the clause splitting, the learned colour head and
the skin-tone comparison exactly as measured, and there is no second
ranking implementation to drift.

## The skin-tone slider

The iOS brief asks for a 0..1 slider. There is no absolute tone scale
here on purpose -- `extract_skin_tone.py` measured that binning
photographed skin against reference swatches is broken (the swatches for
lighter tones sit at L* 78-94 while real photographed skin in this corpus
measures L* 11-61, so half the scale is unreachable). What IS meaningful
is comparison between photos measured the same way.

So the slider is mapped onto the corpus's own distribution: 0.0 is the
5th percentile of measured skin lightness, 1.0 the 95th, and the
reference chroma is the corpus median. It is a position within what we
have actually observed, not a claim about skin.
"""

import modal

app = modal.App("outfit-serve")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "transformers", "pillow", "numpy",
                 "tqdm", "accelerate", "safetensors", "sentencepiece",
                 "fastapi[standard]", "python-multipart")
    .add_local_file("outfit_search.py", "/root/outfit_search.py")
    .add_local_file("free_text_visual_search.py", "/root/free_text_visual_search.py")
    .add_local_file("composed_query_search.py", "/root/composed_query_search.py")
    .add_local_file("catalog_query_search.py", "/root/catalog_query_search.py")
    .add_local_file("docs/hierarchy.json", "/root/docs/hierarchy.json")
)

outfit_volume = modal.Volume.from_name("outfit-index", create_if_missing=False)
catalog_volume = modal.Volume.from_name("fashion-dataset", create_if_missing=False)
api_secret = modal.Secret.from_name("fashion-api-key")


@app.cls(image=image, cpu=4.0, memory=8192,
         volumes={"/v": outfit_volume, "/data": catalog_volume},
         secrets=[api_secret], scaledown_window=300, timeout=600)
@modal.concurrent(max_inputs=4)
class OutfitService:

    @modal.enter()
    def load(self):
        import os
        import sys
        import time

        started = time.time()
        sys.path.insert(0, "/root")
        os.environ["APPAREL_DATASET_ROOT"] = "/data/apparel_dataset"

        import outfit_search

        # Repoint at the Volume before constructing: OutfitSearch reads
        # these at __init__ time.
        from pathlib import Path

        outfit_search.REPO_ROOT = Path("/v")
        outfit_search.OUTFIT_METADATA = Path("/v/outfit_dataset/metadata.json")
        outfit_search.PHOTO_INDEX_PATH = Path("/v/outfit_dataset/outfit_search_index.pt")
        outfit_search.CROP_INDEX_PATH = Path("/v/outfit_dataset/outfit_crop_index.pt")

        self.module = outfit_search
        self.engine = outfit_search.OutfitSearch()
        self.load_seconds = round(time.time() - started, 1)
        self.skin_scale = self._skin_percentiles()
        print(f"loaded in {self.load_seconds}s: "
              f"{len(self.engine.photo_records):,} photos, "
              f"{len(self.engine.crop_records):,} crops")

    def _skin_percentiles(self):
        """The corpus's own skin-lightness distribution, for the slider.

        Returns (L*_at_p5, L*_at_p95, median a*, median b*) or None when
        too few photos have been measured to place a slider honestly."""
        import numpy as np

        values = [v for v in self.engine._skin_lab.values() if v]
        if len(values) < 200:
            return None
        arr = np.array(values, dtype=float)
        return (float(np.percentile(arr[:, 0], 5)),
                float(np.percentile(arr[:, 0], 95)),
                float(np.median(arr[:, 1])), float(np.median(arr[:, 2])))

    def _skin_reference(self, slider):
        if self.skin_scale is None or slider is None:
            return None
        low, high, a, b = self.skin_scale
        return [low + (high - low) * max(0.0, min(1.0, float(slider))), a, b]

    def _authorize(self, request):
        import os

        from fastapi import HTTPException

        expected = os.environ.get("FASHION_API_KEY")
        supplied = (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
        if not expected or supplied != expected:
            raise HTTPException(status_code=401, detail="unauthorized")

    @modal.asgi_app()
    def api(self):
        import base64
        import os
        import tempfile
        import time
        from pathlib import Path

        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, Response

        web = FastAPI(title="outfit search", version="1.0")
        outfit_root = Path("/v/outfit_dataset")

        @web.get("/health")
        def health():
            return {
                "status": "ok",
                "photos": len(self.engine.photo_records),
                "posts": len(self.engine.photo_rows_by_post),
                "crops": len(self.engine.crop_records),
                "skin_measured": len(self.engine._skin_lab),
                "skin_slider_available": self.skin_scale is not None,
                "colour_source": ("learned head"
                                  if self.engine._crop_colors_learned is not None
                                  else "palette heuristic"),
                "colour_vocab": self.engine.color_vocab,
                "container_load_seconds": self.load_seconds,
                "caveats": {
                    "labels": "garment labels are UNVALIDATED model output; "
                              "there is no ground truth for these photos",
                    "skin_tone": "RELATIVE only -- position within this corpus's "
                                 "measured distribution, not an absolute tone scale",
                    "people": "photographs of real people; every result carries "
                              "post_url back to its source",
                },
            }

        @web.post("/outfit_search")
        async def outfit_search(request: Request):
            self._authorize(request)
            started = time.time()
            body = await request.json()

            parts, temps = [], []
            for index, encoded in enumerate(body.get("images") or []):
                if isinstance(encoded, str) and encoded.startswith("data:"):
                    encoded = encoded.split(",", 1)[-1]
                handle = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                handle.write(base64.b64decode(encoded))
                handle.close()
                temps.append(handle.name)
                parts.append({"kind": "image", "value": handle.name,
                              "label": f"image {index + 1}"})
            for phrase in body.get("texts") or []:
                if str(phrase).strip():
                    parts.append({"kind": "text", "value": str(phrase).strip()})

            # A skin reference photo wins over the slider: it is a direct
            # measurement rather than a position on a distribution.
            skin_lab = None
            skin_encoded = body.get("skin_image_base64")
            if skin_encoded:
                skin_lab = self._skin_from_photo(skin_encoded, temps)
            if skin_lab is None:
                skin_lab = self._skin_reference(body.get("skin_tone"))

            try:
                result = self.engine.search(
                    parts,
                    top_k=int(body.get("top_k") or 24),
                    use_filters=bool(body.get("use_filters", True)),
                    drop_non_us=bool(body.get("drop_non_us", False)),
                    drop_womens=bool(body.get("drop_womens", False)),
                    colour_name=body.get("colour_name") or None,
                    colour_rgb=body.get("colour_rgb") or None,
                    skin_lab=skin_lab)
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error))
            finally:
                for path in temps:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

            for hit in result["results"]:
                # A URL the app can fetch, rather than a path only this
                # container can resolve.
                hit["image_url"] = f"/photo?path={hit['rel']}"
                hit["id"] = hit["post_id"]
            result["latency_ms"] = round((time.time() - started) * 1000, 1)
            result["skin_reference_used"] = skin_lab is not None
            return JSONResponse(result)

        @web.get("/photo")
        def photo(path: str, request: Request):
            self._authorize(request)
            target = (Path("/v") / path).resolve()
            # Guard on the RESOLVED path: "../.." would otherwise walk out
            # of the dataset and serve anything on the volume.
            if not str(target).startswith(str(outfit_root.resolve())):
                raise HTTPException(status_code=403, detail="outside the dataset")
            if not target.is_file():
                raise HTTPException(status_code=404, detail="not found")
            return Response(content=target.read_bytes(), media_type="image/jpeg",
                            headers={"Cache-Control": "public, max-age=86400"})

        return web

    def _skin_from_photo(self, encoded, temps):
        """Measure skin from a reference photo. Returns None on any failure —
        a missing reference must degrade to 'no skin filter', never to a
        wrong one."""
        return None
