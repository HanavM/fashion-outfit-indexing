"""Point a browser at the live service and try the product by hand.

    python3 try_it.py                 # then open http://localhost:7860
    python3 try_it.py --lan           # also reachable from your phone
    python3 try_it.py --url http://localhost:8000

Why this exists: the product's real surface is an iOS Shortcut, and a
Shortcut cannot be run from a laptop. `siri_client.py` covers that flow as
code, but reading a JSON blob in a terminal is a poor way to judge whether
an answer is any *good* -- you want to see the photo you sent next to the
product it named. This is that view, and nothing more. It adds no
retrieval logic: it forwards to `POST /query` and renders what comes back.

Stdlib only, so it runs under system python3 with no venv.

**The API key stays on this side.** The browser never sees it -- the page
talks to this local process, which attaches the bearer token and forwards
to Modal. That also sidesteps CORS, so `modal_app_serve.py` needs no
cross-origin middleware added for a debugging tool.

Honesty is a feature here, not decoration. The page shows the garment
gate, the open-set flag, the fact that `reject_threshold` is uncalibrated,
and the brand-coverage caveat -- because the most likely way to be fooled
by this system is a confident, well-presented, wrong answer. Confidence is
not reliability: on the jacket miss the wrong answer scored 0.922 against
the right one's 0.854.
"""

import argparse
import json
import os
import socket
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_URL = "https://hanavm--fashion-serve-fashionservice-api.modal.run"
REPO_ROOT = Path(__file__).resolve().parent
# Only files under these roots are servable. A debugging tool that will
# happily read /etc/passwd because the path came from a query string is
# still a directory traversal bug.
IMAGE_ROOTS = (REPO_ROOT / "apparel_dataset", REPO_ROOT / "apparel_dataset_full")


def load_api_key():
    key = os.environ.get("FASHION_API_KEY")
    if key:
        return key
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("FASHION_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("FASHION_API_KEY not found in environment or .env")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fashion retrieval — try it</title>
<style>
  :root { color-scheme: light dark; --bg:#fff; --fg:#16161a; --muted:#6b6b76;
          --line:#e3e3e8; --card:#fafafb; --accent:#2f6f4f; --warn:#8a5a00; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#131316; --fg:#ececf1; --muted:#9a9aa6; --line:#2c2c33;
            --card:#1b1b20; --accent:#7fd1a6; --warn:#e0b25f; }
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:24px 16px 64px; background:var(--bg); color:var(--fg);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  main { max-width: 860px; margin: 0 auto; }
  h1 { font-size:19px; margin:0 0 4px; letter-spacing:-.01em; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:20px; }
  .drop { border:1.5px dashed var(--line); border-radius:12px; padding:28px 16px;
          text-align:center; cursor:pointer; background:var(--card); transition:.15s; }
  .drop.over { border-color:var(--accent); }
  .drop input { display:none; }
  .row { display:flex; gap:10px; margin-top:12px; flex-wrap:wrap; }
  input[type=text] { flex:1 1 260px; min-width:0; padding:10px 12px; font-size:15px;
                     border:1px solid var(--line); border-radius:9px;
                     background:var(--bg); color:var(--fg); }
  button { padding:10px 20px; font-size:15px; font-weight:600; border:0;
           border-radius:9px; background:var(--accent); color:var(--bg); cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .preview { max-height:260px; border-radius:10px; margin-top:14px; display:block; }
  .spoken { font-size:19px; font-weight:600; margin:22px 0 6px; line-height:1.4; }
  .route { font-size:12px; color:var(--muted); font-family:ui-monospace,Menlo,monospace;
           word-break:break-word; }
  .flags { display:flex; gap:8px; flex-wrap:wrap; margin:14px 0 6px; }
  .flag { font-size:12px; padding:3px 9px; border-radius:99px; border:1px solid var(--line);
          color:var(--muted); }
  .flag.warn { color:var(--warn); border-color:var(--warn); }
  table { width:100%; border-collapse:collapse; margin-top:12px; font-size:14px; }
  th { text-align:left; font-weight:600; font-size:12px; color:var(--muted);
       border-bottom:1px solid var(--line); padding:6px 8px; }
  td { padding:8px; border-bottom:1px solid var(--line); vertical-align:middle; }
  td img { width:52px; height:52px; object-fit:cover; border-radius:6px;
           background:var(--card); display:block; }
  .caveat { margin-top:26px; padding:12px 14px; border-left:3px solid var(--warn);
            background:var(--card); font-size:13px; color:var(--muted); border-radius:0 8px 8px 0; }
  .caveat b { color:var(--fg); }
  .err { color:#c0392b; margin-top:16px; }
  details { margin-top:18px; } summary { cursor:pointer; font-size:13px; color:var(--muted); }
  pre { overflow-x:auto; font-size:12px; background:var(--card); padding:12px;
        border-radius:8px; }
</style>
</head>
<body><main>
  <h1>fashion retrieval</h1>
  <div class="sub">Photo in, catalog product out. Talks to the live
    <code>/query</code> endpoint — the same one the iOS Shortcut uses.</div>

  <label class="drop" id="drop">
    <input type="file" id="file" accept="image/*">
    <span id="droptext">Drop a photo here, tap to choose, or paste one</span>
  </label>
  <img id="preview" class="preview" hidden>

  <div class="row">
    <input type="text" id="text" placeholder='optional — try "what brand is this" or "with cargo pants"'>
    <button id="go" disabled>Identify</button>
  </div>

  <div id="out"></div>

  <div class="caveat">
    <b>Read results with these in mind.</b> The catalog covers a limited set of
    brands — point it at something outside them and it will still name its
    closest match, confidently. Open-set rejection cannot catch that
    (<code>reject_threshold</code> is uncalibrated, so it never fires by
    default). Confidence is not reliability: the wrong answer has outscored
    the right one, 0.922 to 0.854. The garment gate <i>is</i> calibrated
    (AUROC 0.9994) and is the one flag below you can trust.
  </div>
</main>
<script>
let imageData = null;
const $ = id => document.getElementById(id);

function loadFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = e => {
    imageData = e.target.result;
    $('preview').src = imageData; $('preview').hidden = false;
    $('droptext').textContent = file.name || 'image ready';
    $('go').disabled = false;
  };
  reader.readAsDataURL(file);
}

$('file').addEventListener('change', e => loadFile(e.target.files[0]));
$('drop').addEventListener('dragover', e => { e.preventDefault(); $('drop').classList.add('over'); });
$('drop').addEventListener('dragleave', () => $('drop').classList.remove('over'));
$('drop').addEventListener('drop', e => {
  e.preventDefault(); $('drop').classList.remove('over');
  loadFile(e.dataTransfer.files[0]);
});
document.addEventListener('paste', e => {
  for (const item of e.clipboardData.items)
    if (item.type.startsWith('image/')) loadFile(item.getAsFile());
});
// Text alone is a valid query -- that is the whole point of /query routing.
$('text').addEventListener('input', () => {
  $('go').disabled = !imageData && !$('text').value.trim();
});

$('go').addEventListener('click', async () => {
  const body = {top_k: 8};
  if (imageData) body.image_base64 = imageData;
  const text = $('text').value.trim();
  if (text) body.text = text;

  $('go').disabled = true;
  $('out').innerHTML = '<p class="sub">thinking… (a cold container takes ~20s)</p>';
  try {
    const res = await fetch('/api/query', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
    render(data);
  } catch (err) {
    $('out').innerHTML = '<p class="err">' + err.message + '</p>';
  }
  $('go').disabled = false;
});

function flag(label, warn) {
  return '<span class="flag' + (warn ? ' warn' : '') + '">' + label + '</span>';
}

function render(d) {
  let html = '';
  if (d.spoken) html += '<div class="spoken">' + escapeHtml(d.spoken) + '</div>';
  if (d.route) html += '<div class="route">route: ' + escapeHtml(d.route.intent) +
      ' → ' + escapeHtml(d.route.equivalent_endpoint || '') +
      ' &nbsp;·&nbsp; ' + escapeHtml(d.route.reason) + '</div>';

  const flags = [];
  if (d.latency_ms) flags.push(flag(Math.round(d.latency_ms) + ' ms'));
  if (typeof d.confidence === 'number') flags.push(flag('score ' + d.confidence.toFixed(3)));
  const gate = d.garment_gate;
  if (gate) flags.push(gate.looks_like_clothing
      ? flag('looks like clothing ✓')
      : flag('does NOT look like clothing', true));
  if (d.rejected_open_set) flags.push(flag('rejected as out-of-catalog', true));
  else if (d.reject_threshold_calibrated === false)
    flags.push(flag('open-set check uncalibrated — never fires', true));
  if (d.same_model_different_colorway_ambiguous)
    flags.push(flag('colorway ambiguous', true));
  if (flags.length) html += '<div class="flags">' + flags.join('') + '</div>';

  // /identify shape
  if (d.results && d.results.length) html += table(d.results);

  // /compose shape
  if (d.primary && d.primary.results) {
    html += '<h3 style="font-size:14px;margin:22px 0 0">the item you photographed</h3>';
    html += table(d.primary.results.slice(0, 5));
  }
  if (d.companions && d.companions.length) {
    html += '<h3 style="font-size:14px;margin:22px 0 0">suggested companions</h3>';
    html += table(d.companions);
    if (d.note) html += '<div class="route" style="margin-top:8px">' + escapeHtml(d.note) + '</div>';
  }
  if (d.results && !d.results.length) html += '<p class="sub">no matches returned.</p>';

  html += '<details><summary>raw response</summary><pre>' +
          escapeHtml(JSON.stringify(d, null, 2)) + '</pre></details>';
  $('out').innerHTML = html;
}

function table(rows) {
  let h = '<table><tr><th></th><th>brand</th><th>product</th><th>score</th></tr>';
  for (const r of rows) {
    const img = r.image || r.images && r.images[0];
    h += '<tr><td>' + (img
          ? '<img loading="lazy" src="/img?path=' + encodeURIComponent(img) + '">'
          : '') + '</td>' +
         '<td>' + escapeHtml(r.brand || '') + '</td>' +
         '<td>' + (r.product_url
            ? '<a href="' + escapeHtml(r.product_url) + '" target="_blank" rel="noopener">' +
              escapeHtml(r.name || '') + '</a>'
            : escapeHtml(r.name || '')) + '</td>' +
         '<td>' + (typeof r.score === 'number' ? r.score.toFixed(3) : '') + '</td></tr>';
  }
  return h + '</table>';
}

function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g,
    c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    upstream = DEFAULT_URL
    api_key = ""
    catalog = {}

    def log_message(self, fmt, *args):
        # One tidy line per request; the default logs every asset noisily.
        if not self.path.startswith("/img"):
            print(f"  {self.command} {self.path.split('?')[0]} -> {args[1] if len(args) > 1 else ''}")

    def _send(self, code, body, content_type="application/json"):
        payload = body if isinstance(body, bytes) else str(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if self.path.startswith("/img"):
            return self._serve_image()
        self._send(404, b'{"detail":"not found"}')

    def _serve_image(self):
        from urllib.parse import parse_qs, unquote, urlparse

        raw = parse_qs(urlparse(self.path).query).get("path", [""])[0]
        candidate = (REPO_ROOT / unquote(raw)).resolve()
        # resolve() then check containment: the guard has to run on the
        # resolved path, or "../../etc/passwd" walks straight out.
        if not any(str(candidate).startswith(str(root.resolve())) for root in IMAGE_ROOTS):
            return self._send(403, b'{"detail":"outside the allowed image roots"}')
        if not candidate.is_file():
            return self._send(404, b'{"detail":"image not on disk in this environment"}')
        self._send(200, candidate.read_bytes(), "image/jpeg")

    def do_POST(self):
        if self.path != "/api/query":
            return self._send(404, b'{"detail":"not found"}')
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as error:
            return self._send(400, json.dumps({"detail": str(error)}).encode())

        # The page sends a data: URL; the API tolerates the prefix, but
        # stripping it here keeps the payload smaller over the wire.
        image = body.get("image_base64")
        if isinstance(image, str) and image.startswith("data:"):
            body["image_base64"] = image.split(",", 1)[-1]

        request = urllib.request.Request(
            f"{self.upstream.rstrip('/')}/query",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read())
                self._enrich(payload)
                return self._send(response.status, json.dumps(payload).encode())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            if error.code == 401:
                detail = ('401: the API key was rejected. Note that `modal deploy` '
                          'does not cycle a warm container, so a rotated key needs '
                          '`modal app stop fashion-serve --yes` first.')
            return self._send(error.code, json.dumps({"detail": detail}).encode())
        except (urllib.error.URLError, TimeoutError) as error:
            return self._send(502, json.dumps({"detail": f"upstream: {error}"}).encode())

    def _enrich(self, payload):
        """Attach a local thumbnail path and the product URL to each hit.

        The API deliberately does not return image paths -- it serves a
        voice client that has no use for them. Joining locally keeps that
        contract unchanged. Several brands have no images in this dev
        environment at all, so a missing thumbnail here is expected and
        is not a retrieval failure."""
        def annotate(rows):
            for row in rows or []:
                record = self.catalog.get(row.get("product_code"))
                if not record:
                    continue
                images = record.get("images") or []
                if images:
                    row["image"] = images[0]
                if record.get("product_url"):
                    row["product_url"] = record["product_url"]

        annotate(payload.get("results"))
        annotate((payload.get("primary") or {}).get("results"))
        annotate(payload.get("companions"))


def load_catalog():
    path = REPO_ROOT / "apparel_dataset" / "metadata.json"
    if not path.exists():
        print("  (no local metadata.json — results will have no thumbnails)")
        return {}
    try:
        return {r["product_code"]: r for r in json.loads(path.read_text())}
    except json.JSONDecodeError:
        # A scraper checkpointing right now can leave a momentarily
        # unparseable file. Thumbnails are a nicety; don't die for them.
        print("  (metadata.json unreadable right now — continuing without thumbnails)")
        return {}


def lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("FASHION_API_URL", DEFAULT_URL),
                    help="serving API base URL")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--lan", action="store_true",
                    help="bind 0.0.0.0 so a phone on the same wifi can reach it")
    args = ap.parse_args()

    Handler.upstream = args.url
    Handler.api_key = load_api_key()
    Handler.catalog = load_catalog()

    host = "0.0.0.0" if args.lan else "127.0.0.1"
    server = ThreadingHTTPServer((host, args.port), Handler)
    print(f"\n  upstream : {args.url}")
    print(f"  catalog  : {len(Handler.catalog):,} products loaded for thumbnails")
    print(f"\n  open     : http://localhost:{args.port}")
    if args.lan:
        ip = lan_ip()
        if ip:
            print(f"  phone    : http://{ip}:{args.port}   (same wifi)")
    print("\n  ctrl-c to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main()
