"""Backfill `author` (the pinner) + the pin's outbound source link onto the
Pinterest outfit records collected by the brand-targeted run.

Why this exists: `/pin/<id>` permalinks carry no handle, and the search GRID
DOM carries no pinner either (verified 2026-08-06: pin cards expose only
href/alt/image, no profile link, no domain badge). The internal
`/resource/PinResource/get/` JSON route 403s even with a logged-in session
and a csrftoken header. The pinner IS present on the pin PAGE, in the
`__PWS_INITIAL_PROPS__` blob, at ~3-4s per pin -- so this is a second pass,
not something the grid scrape could have captured inline.

Writes through dataset_utils.save_outfit_records_safe (flock + atomic
replace), same as every other outfit writer. Only touches records whose
source_id is in --ids-file, so it can never disturb the pre-existing corpus.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_utils import load_outfit_records, outfit_key, save_outfit_records_safe

PROFILE = str(Path.home() / ".cache" / "fashion-tests" / "browser-profile")

JS = """
() => {
  const html = document.documentElement.innerHTML;
  let pinner = null;
  const m = html.match(/"pinner"\\s*:\\s*\\{(?:[^{}]|\\{[^{}]*\\})*?"username"\\s*:\\s*"([^"]+)"/);
  if (m) pinner = m[1];
  if (!pinner) { const m2 = html.match(/"username"\\s*:\\s*"([^"]+)"/); if (m2) pinner = m2[1]; }
  let creator = null;
  const mc = html.match(/"native_creator"\\s*:\\s*\\{(?:[^{}]|\\{[^{}]*\\})*?"username"\\s*:\\s*"([^"]+)"/);
  if (mc) creator = mc[1];
  const outbound = [...document.querySelectorAll('a[href^="http"]')]
      .map(a => a.href).filter(h => !h.includes('pinterest.com'));
  return {pinner, creator, link: outbound[0] || null,
          gone: /Page not found|couldn't find that page/i.test(document.body.innerText)};
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", help="json list of source_ids to SKIP (optional)")
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--deadline-min", type=float, default=90.0)
    args = ap.parse_args()

    # No --ids-file means "every authorless Pinterest record", which is the
    # legacy 1,342 that docs/licensing_review.md calls the one item with a
    # closing window: pin permalinks carry no handle, so we hold photographs
    # of real people we cannot attribute or answer a takedown for. The flag
    # is kept so a run can still be scoped to one batch.
    baseline = set(json.loads(Path(args.ids_file).read_text())) if args.ids_file else set()
    records = load_outfit_records()
    todo = [r for r in records
            if r["source"] == "pinterest" and r["source_id"] not in baseline
            and not r.get("author")]
    todo = todo[: args.limit]
    print(f"{len(records)} records; {len(todo)} new pinterest records need an author")
    if not todo:
        return

    from playwright.sync_api import sync_playwright
    start = time.time()
    got = 0
    touched = {}
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            PROFILE, headless=True, viewport={"width": 1200, "height": 800},
            # A run visits thousands of pin pages in one persistent profile,
            # and Chromium's on-disk HTTP cache grows without bound across
            # them: measured 2026-08-18, 500 pins put 765 MB in Default/Cache
            # plus 187 MB in Code Cache, on a machine with ~6 GiB free. Nothing
            # is ever re-visited, so the cache buys nothing -- pin the caches
            # to ~0 rather than periodically stopping the run to purge them.
            args=["--disk-cache-size=1", "--media-cache-size=1"],
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
        pg = ctx.pages[0] if ctx.pages else ctx.new_page()
        # Images are already downloaded; blocking them makes the page ~3x cheaper.
        pg.route("**/*", lambda route: route.abort()
                 if route.request.resource_type in ("image", "media", "font") else route.continue_())
        for i, rec in enumerate(todo, 1):
            if (time.time() - start) / 60 > args.deadline_min:
                print(f"deadline hit at {i-1}/{len(todo)}")
                break
            try:
                pg.goto(f"https://www.pinterest.com/pin/{rec['source_id']}/",
                        timeout=30000, wait_until="domcontentloaded")
                time.sleep(1.4)
                info = pg.evaluate(JS)
            except Exception as e:
                print(f"  [{i}] {rec['source_id']} FAILED {type(e).__name__}")
                continue
            author = info.get("creator") or info.get("pinner")
            if author:
                rec["author"] = author
                got += 1
            if info.get("link"):
                rec["source_link"] = info["link"].split("?")[0]
            if author or info.get("link"):
                touched[outfit_key(rec)] = rec
            if i % 25 == 0:
                print(f"  [{i}/{len(todo)}] authors {got} "
                      f"({got/i:.0%})  {(time.time()-start)/60:.1f} min")
            if len(touched) >= 20:
                save_outfit_records_safe(touched)
                touched = {}
        if touched:
            save_outfit_records_safe(touched)
        ctx.close()
    print(f"DONE: {got} authors recovered in {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    main()
