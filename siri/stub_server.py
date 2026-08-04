"""A fake serving layer that speaks the Phase 6 contract, so the Phase 7
client can be tested before the real server exists.

WHY: `modal_app_serve.py` is being built in parallel and is deliberately
NOT touched from here. Without something to POST at, `siri_client.py` and
the Shortcut recipe would be untested prose. This stub is the smallest
thing that makes them testable: stdlib only, no models, no GPU, instant,
and it can be told to produce every failure mode the client claims to
handle -- which is the part that actually matters, because the happy path
is the easy one.

It is NOT a mock of the model. Its "results" are real product codes read
out of catalog_metadata.json so the output shape is realistic, but the
ranking is arbitrary. Nothing about retrieval quality can be learned from
this file, and no number produced against it should ever be reported as
an accuracy result.

Run:
    python3 siri/stub_server.py                       # happy path
    python3 siri/stub_server.py --scenario rejected   # open-set rejection
    python3 siri/stub_server.py --scenario low        # weak match
    python3 siri/stub_server.py --scenario empty      # nothing found
    python3 siri/stub_server.py --scenario no-flag    # contract violation:
                                                      # drops `rejected`
    python3 siri/stub_server.py --scenario error      # HTTP 500
    python3 siri/stub_server.py --scenario slow       # exceeds client timeout

    export FASHION_API_TOKEN=stub-secret
    python3 siri_client.py --image some.jpg --url http://127.0.0.1:8000
"""

import argparse
import base64
import json
import random
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_FILE = REPO_ROOT / "catalog_metadata.json"

TOKEN = "stub-secret"
SCENARIO = "ok"
SLOW_SECONDS = 30.0


def load_sample_products(count=8):
    """Real product codes/URLs from the real catalog index, so the client
    is parsing realistic strings rather than 'foo'/'bar'."""
    if not CATALOG_FILE.exists():
        return [{"product_code": f"STUB{i}", "brand": "stub", "name": f"Stub Product {i}"}
                for i in range(count)]
    records = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    seen, products = set(), []
    for record in records:
        code = record.get("product_code")
        if not code or code in seen:
            continue
        seen.add(code)
        slug = record.get("slug", "")
        image_path = record.get("image_path", "")
        products.append({
            "product_code": code,
            "brand": image_path.split("_catalog/")[0] if "_catalog/" in image_path else "unknown",
            "name": slug.replace("-", " ").title() or code,
            "product_url": record.get("product_url"),
            "image_path": image_path,
        })
        if len(products) >= count * 6:
            break
    random.shuffle(products)
    return products[:count]


PRODUCTS = []


def scored(products, high):
    base = 0.81 if high else 0.24
    return [dict(p, score=round(base - 0.04 * i, 4)) for i, p in enumerate(products)]


def identify_payload(top_k):
    """Field names follow modal_app_serve.py (`rejected_open_set`,
    `reject_threshold_calibrated`), not the shorthand in the contract
    table -- the client must handle the names the real server actually
    emits. `--scenario alias` sends the shorthand instead, to prove the
    client's tolerance is real rather than asserted."""
    if SCENARIO == "empty":
        return {"spoken": "I couldn't find anything like that.",
                "confidence": None, "rejected_open_set": True,
                "reject_threshold": 0.35, "reject_threshold_calibrated": False,
                "results": []}
    if SCENARIO == "rejected":
        results = scored(PRODUCTS[:top_k], high=False)
        return {"spoken": "That looks like " + results[0]["name"] + ".",  # deliberately
                # over-confident: the client must ignore this and hedge,
                # because the rejection flag is authoritative.
                "confidence": 0.19, "rejected_open_set": True,
                "reject_threshold": 0.35, "reject_threshold_calibrated": False,
                "results": results}
    if SCENARIO == "low":
        results = scored(PRODUCTS[:top_k], high=False)
        return {"spoken": "That looks like " + results[0]["name"] + ".",
                "confidence": 0.22, "rejected_open_set": False,
                "reject_threshold": 0.35, "reject_threshold_calibrated": False,
                "results": results}
    if SCENARIO == "no-flag":
        results = scored(PRODUCTS[:top_k], high=True)
        return {"spoken": "That looks like " + results[0]["name"] + ".",
                "confidence": 0.78, "results": results}
    results = scored(PRODUCTS[:top_k], high=True)
    top = results[0]
    payload = {"spoken": f"That looks like a {top['brand']} {top['name']}.",
               "confidence": 0.78,
               "reject_threshold": 0.35, "reject_threshold_calibrated": False,
               "results": results,
               "detection": {"garment_found": True, "category": "outerwear/jacket",
                             "bbox": [120, 80, 640, 900], "detector": "stub"}}
    if SCENARIO == "alias":
        payload["rejected"] = False              # the contract's shorthand
    else:
        payload["rejected_open_set"] = False     # what the server really sends
    return payload


def compose_payload(text, top_k):
    """Mirrors modal_app_serve.py's actual /compose shape, which is NOT a
    flattened version of /identify: the whole identity block (results,
    confidence, rejection) is nested under `primary`, and the top level
    carries only the companions. A client that reads rejection at the top
    level of this response finds nothing and hedges forever -- which is
    precisely the bug this scenario exists to catch."""
    block = identify_payload(top_k)
    spoken = block.pop("spoken", None)
    results = block.get("results", [])
    companions = results[1:top_k]
    payload = {
        "primary": block,
        "companions": companions,
        "parsed_text_query": {"raw": text, "category": None, "attributes": []},
        "note": ("primary and companions are TWO INDEPENDENT SEARCHES. Nothing here "
                 "shows these items were ever worn together in a real photo -- that "
                 "needs the outfit co-occurrence index (roadmap Phase 8)."),
    }
    if results and companions and not block.get("rejected_open_set"):
        payload["spoken"] = (f"That looks like a {results[0]['brand']} {results[0]['name']}. "
                             f"With {text.strip()}, try the {companions[0]['name']}.")
    elif spoken:
        payload["spoken"] = spoken
    return payload


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"  [stub] {fmt % args}", flush=True)

    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        header = self.headers.get("Authorization", "")
        if header == f"Bearer {TOKEN}":
            return True
        self._send(401, {"error": "unauthorized",
                         "detail": "send Authorization: Bearer <token>"})
        return False

    def do_GET(self):
        if self.path.rstrip("/") != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(200, {"status": "ok", "stub": True, "scenario": SCENARIO,
                         "models_loaded": False,
                         "note": "stub server -- no models, results are arbitrary"})

    def do_POST(self):
        path = self.path.rstrip("/")
        if path not in ("/identify", "/compose"):
            self._send(404, {"error": "not found"})
            return
        if not self._authorized():
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            request = json.loads(raw)
        except ValueError:
            self._send(400, {"error": "body was not JSON"})
            return

        encoded = request.get("image_base64")
        if not encoded:
            self._send(400, {"error": "image_base64 is required"})
            return
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except Exception:
            self._send(400, {"error": "image_base64 was not valid base64"})
            return
        print(f"  [stub] {path}: {len(decoded)/1024:.0f} KB image"
              f"{', text=' + repr(request.get('text')) if request.get('text') else ''}",
              flush=True)

        if SCENARIO == "error":
            self._send(500, {"error": "stub failure (scenario=error)"})
            return
        if SCENARIO == "slow":
            time.sleep(SLOW_SECONDS)

        top_k = int(request.get("top_k") or 5)
        if path == "/identify":
            self._send(200, identify_payload(top_k))
        else:
            self._send(200, compose_payload(request.get("text", ""), top_k))


def main():
    global TOKEN, SCENARIO, PRODUCTS, SLOW_SECONDS
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token", default=TOKEN)
    parser.add_argument("--scenario", default="ok",
                        choices=["ok", "alias", "rejected", "low", "empty", "no-flag", "error", "slow"])
    parser.add_argument("--slow-seconds", type=float, default=SLOW_SECONDS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    TOKEN, SCENARIO, SLOW_SECONDS = args.token, args.scenario, args.slow_seconds
    random.seed(args.seed)
    PRODUCTS = load_sample_products()

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"stub serving http://127.0.0.1:{args.port}  scenario={SCENARIO}  token={TOKEN}",
          flush=True)
    print("NOT a model. Results are arbitrary; never report accuracy from this.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
