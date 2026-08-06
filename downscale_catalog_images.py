"""Cap catalog image resolution in place, to stop a scrape costing 16 MB/record.

    python downscale_catalog_images.py --brand obey --dry-run
    python downscale_catalog_images.py --brand obey

Why this is safe for this pipeline, specifically: SigLIP2 consumes 384px
and DINOv3 224px, `segment_apparel.py` resizes before mask generation
anyway (that resize was itself a measured fix), and nothing downstream
reads native resolution. A 4000x5000 source carries no information any
model in this repo can use.

Why it matters: OBEY's Shopify CDN serves 3000x3750 and 4000x5000 PNGs at
2.5-6.4 MB each, so 182 records cost 2.9 GB -- roughly 20x Dickies' ~270
KB/image for no benefit. On a machine that runs chronically near-full,
that is the difference between fitting the next four brands and not.

**Prefer not needing this.** Shopify CDN URLs accept `&width=1200`, so a
scraper should ask for a sane size rather than downloading 6 MB and
shrinking it afterwards. This script is the retrofit for scrapes that
already happened.

Reversible: every record keeps `image_urls`, so any image can be
re-fetched at full resolution. Nothing here touches `metadata.json`.

Note for the retrieval index: changing image BYTES changes the
`gallery_signature` those products are keyed by, so the next
`build_indexes` re-encodes exactly the touched products (and only those).
That is correct behaviour -- a code-only check would keep a stale vector
whose backing image no longer exists.
"""

import argparse
import os
import sys
from pathlib import Path

from PIL import Image

DATASET = Path("apparel_dataset")


def iter_images(brand):
    root = DATASET / brand if brand else DATASET
    if not root.exists():
        raise SystemExit(f"{root} does not exist")
    for path in sorted(root.rglob("*.jpg")):
        yield path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brand", help="limit to one brand folder (recommended)")
    ap.add_argument("--max-edge", type=int, default=1200,
                    help="longest edge to keep (default 1200)")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--min-bytes", type=int, default=400_000,
                    help="skip files already smaller than this")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    touched = saved = skipped = failed = 0
    for path in iter_images(args.brand):
        size = path.stat().st_size
        if size < args.min_bytes:
            skipped += 1
            continue
        try:
            with Image.open(path) as image:
                if max(image.size) <= args.max_edge and image.format == "JPEG":
                    skipped += 1
                    continue
                original = image.size
                converted = image.convert("RGB")
                converted.thumbnail((args.max_edge, args.max_edge), Image.LANCZOS)
                if args.dry_run:
                    print(f"  would shrink {path}  {original} {size/1e6:.1f}MB")
                    touched += 1
                    continue
                # Write beside the target then replace: a crash mid-write
                # must not leave a truncated image where a valid one was.
                # (Same discipline the metadata writers use, for the same
                # reason -- this machine has run out of disk before.)
                temp = path.with_suffix(".tmp.jpg")
                converted.save(temp, "JPEG", quality=args.quality, optimize=True)
                new_size = temp.stat().st_size
                if new_size >= size:
                    # Already efficient; keep the original rather than
                    # trading quality for nothing.
                    temp.unlink()
                    skipped += 1
                    continue
                os.replace(temp, path)
                saved += size - new_size
                touched += 1
        except Exception as error:  # noqa: BLE001 -- one bad file must not stop the pass
            print(f"  [warn] {path}: {error}", file=sys.stderr)
            failed += 1

    verb = "would free" if args.dry_run else "freed"
    print(f"\n  {touched} images {'to shrink' if args.dry_run else 'shrunk'}, "
          f"{skipped} left alone, {failed} failed")
    if not args.dry_run:
        print(f"  {verb} {saved/1e9:.2f} GB")


if __name__ == "__main__":
    main()
