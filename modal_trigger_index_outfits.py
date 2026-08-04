"""Fire-and-forget trigger for the outfit indexing job.

Why this exists rather than just `modal run --detach`: a `modal run` client
stays attached to the container even when detached, and when that local
client process was killed on 2026-08-04 the container received a
cancellation signal and stopped at record 1588 of 2500. The checkpointed
work survived (index_outfits.py writes every 10 records, and the volume
kept it), so nothing was lost -- but the run died with the terminal that
started it, which is the wrong failure mode for a multi-hour job.

`.spawn()` submits the call and returns immediately. The container then
has no relationship to any local process, so a dropped shell, a closed
laptop or a reaped background task cannot cancel it.

    modal deploy modal_app_index_outfits.py
    python3 modal_trigger_index_outfits.py --max-images-per-record 1

Then poll with `modal app logs fashion-index-outfits`.
"""

import argparse

import modal

APP_NAME = "fashion-index-outfits"
FUNCTION_NAME = "index_outfits"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-images-per-record", type=int, default=1,
                        help="Default 1, not index_outfits.py's own 2: a second photo of "
                             "the same post is usually the same outfit from another angle, "
                             "and the co-occurrence index de-duplicates per post anyway, so "
                             "the marginal image buys recall on outfits already counted "
                             "rather than new outfits.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    function = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    call = function.spawn(source=args.source, limit=args.limit,
                          max_images_per_record=args.max_images_per_record,
                          force=args.force)
    print(f"spawned call {call.object_id}")
    print(f"logs: python3 -m modal app logs {APP_NAME}")


if __name__ == "__main__":
    main()
