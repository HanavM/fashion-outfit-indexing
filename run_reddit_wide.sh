#!/bin/zsh
# Fan reddit_outfit_scraper.py out to one process per subreddit.
#
# Why: the scraper is network-bound, not CPU-bound -- it spends its time in
# requests.get and in the politeness sleeps, so a single process leaves the
# link idle almost all the time. One process per subreddit multiplies
# throughput at near-zero CPU cost, which matters on a laptop that is also
# running model work (a headless browser here took the machine to load 68;
# eleven of these together stay under 15% of one core).
#
# Safe to run concurrently only because outfit metadata writes take an
# exclusive flock and land via os.replace (dataset_utils). Safe to re-run:
# every shard skips source_ids already in metadata.json.
#
# Caveat: each process loads its OWN in-memory phash list at start, so a
# repost that appears in two subreddits at the same moment can slip past
# cross-subreddit dedup. Within a subreddit dedup is exact. Sweep after.
#
# Targets are in IMAGES and are raised well above the module defaults --
# the corpus is ~37k posts / 60-90k images, so none of these is supply-bound
# except the two deliberately tight ones at the bottom.
set -u
cd "$(dirname "$0")"
mkdir -p logs

run() {  # run <subreddit> <image target>
  nohup .venv/bin/python -u reddit_outfit_scraper.py \
    --subreddit "$1" --target "$2" > "logs/reddit_$1.log" 2>&1 &
  echo "  started r/$1 (target $2 images) pid $!"
}

run streetwear          800
run fashion             600
run mensfashion         600
run OUTFITS             600
run streetwearfits      500
run femalefashion       500
run malefashion         400
run femalefashionadvice 300
run rawdenim            300
run japanesestreetwear  100
run techwearclothing     60

wait
