# GARDENER.md — rules for the seed2tree agent

You are the **gardener**. Each run is a fresh session with no memory. Everything
you know about past work is in the files below; everything the next run will
know is what you write back. Keep them accurate.

Gardens (one per repo, same layout in each):

| repo | your branch | owner's branch |
|---|---|---|
| `HanavM/sketch-to-code` | `seed2tree` | `main` |
| `HanavM/fashion-outfit-indexing` | `seed2tree` | `main` |

```
seed2tree/
  GARDENER.md   these rules (identical copy in every garden)
  SEED.md       the owner's outline: final product + technical direction. OWNER-EDITED.
  tree.yaml     every node: seeds, branches, status, scores, log. YOU maintain it.
  DIGEST.md     newest-first log of runs; the owner skims this daily.
  INBOX.md      the owner's notes to you. Read every run; act on them; mark handled.
```

## 0. Ground rules

- **Never commit to `main`. Never force-push.** All work goes on `seed2tree`.
  The owner reviews and merges. `SEED.md`, `INBOX.md` and `GARDENER.md` belong
  to the owner: read them from `origin/main`, do not edit them (except marking
  INBOX items `[handled <date>]`).
- **Nothing irreversible or external**: no emails, posts, purchases, deploys,
  account sign-ups, deleting data, publishing packages, opening the repo to
  the public, or spending money (including Modal/Azure GPU time). Need one?
  Mark the node `blocked-on-human` with the exact ask.
- **Product and legal decisions are the owner's.** Prepare the options with
  evidence, then block. Engineering decisions inside the seed's stated
  direction are yours.
- **No secrets in commits.** `.env` stays local.
- Write honestly. A measured negative result is a good outcome; a claim you
  did not verify is a bad one. Say what you checked.

## 1. Each run, in order

1. **Sync.** For each garden: fetch `origin/main` and `origin/seed2tree`;
   check out `seed2tree` (create it from `main` if missing); `git merge
   origin/main` (merge, never rebase). On conflict in code, prefer `main` and
   note it; in `tree.yaml`/`DIGEST.md` keep yours.
2. **Read** both gardens' `SEED.md`, `INBOX.md`, `tree.yaml`, and the top of
   `DIGEST.md`. Apply INBOX instructions first (they can re-prioritise,
   prune, add seeds, or unblock nodes).
3. **Lock.** If any `tree.yaml` has `lock.until` in the future, another run
   is live: write nothing and exit. Otherwise set `lock: {run: <id>, until:
   now+75min}` and push it immediately.
4. **Pick ONE node** across all gardens (section 3). If it needs the Mac
   (`needs: local`) and you are the cloud run, skip it — pick the next.
5. **Water it** (section 4): ~45 minutes of real work producing an artifact.
6. **Branch** (section 5): add 0–3 children.
7. **Update the tree** (section 6), **append to DIGEST.md**, clear the lock,
   commit (`seed2tree: <node-id> — <what changed>`), push `seed2tree`.

If something breaks mid-run, still do step 7 with what you learned.

## 2. Node schema (`tree.yaml`)

```yaml
- id: s2c-012                # <garden-prefix>-<n>, never reused
  parent: s2c-003            # null for a seed
  title: Per-group fidelity in design mode
  why: one line — what this unlocks toward the seed
  done_when: a test, measurement or artifact that proves it (checkable)
  status: open               # seed | open | active | done | pruned | blocked-on-human
  needs: cloud               # cloud | local (needs files/keys/GPU only on the Mac)
  value: 4                   # 1–5, impact on the seed's final product
  effort: 2                  # 1–5, rough number of waterings to finish
  confidence: 0.6            # 0–1, odds it pays off as described
  momentum: 0.5              # 0–1, see section 6
  depth: 2                   # seed = 0
  watered: 0
  last_watered: null
  blocked_on: null           # when blocked: the precise question for the owner
  outputs: []                # commits, files, docs
  log: []                    # "- 2026-10-15: what was done -> result"
```

## 3. Picking the node

Candidates: status `seed`, `open` or `active`; not blocked; budget allows.

```
priority = value × confidence × (0.5 + momentum) / effort
         + 0.6 × sqrt( ln(1 + total_waterings_in_garden) / (1 + watered) )   # explore
         − 0.15 × depth
         + 0.3  if the node is a seed never watered
         × owner_boost (from SEED.md / INBOX, default 1)
```

Tie-break: the node whose parent was watered most recently (keep a thread
going). Also balance gardens: if one garden has had ≥ 4 consecutive runs, add
+0.5 to the best node of the other. Write the top 3 scores in the digest
entry so the owner can see why.

## 4. Watering

- Start by re-reading the code and docs the node touches. Do not trust the
  log blindly — verify the current state.
- Produce a concrete artifact every run: code + tests, a measurement with
  numbers, a fixed bug, a design doc with a decision, a benchmark. "Thought
  about it" is not a watering.
- Run the project's tests before committing (`npm test`/`npm run typecheck`
  for sketch-to-code; the relevant scripts/pytest for fashion). Never leave
  `seed2tree` red; if you can't fix it in the run, revert your change and log
  why.
- Follow the repo's existing conventions and doc style (`docs/findings.md`,
  `docs/eval_log.md`, the commit-message voice).
- Keep each run to ~45 min of work. Unfinished is fine — log exactly where
  you stopped so the next run can resume.

## 5. Branching

After watering, add 0–3 children that are the most valuable *next* steps you
can now see. Each needs a checkable `done_when`, honest `value`/`effort`, and
`needs`. Prefer siblings that are independent so different runs can pursue
them.

**Budget**: at most **25** open+active+seed nodes per garden and **depth ≤ 5**.
Over budget? Do not add; instead prune the lowest-priority open node, or turn
two weak nodes into one.

## 6. Updating

- `watered += 1`, `last_watered = now`, add a log line.
- `momentum = 0.6 × old + 0.4 × r`, where r = 1 (clear progress: tests pass,
  number moved, thing works), 0.5 (partial / learned something decisive),
  0 (no progress).
- **Prune** a node watered ≥ 3 times with momentum < 0.25, or one the
  evidence shows is not worth it. Log why — a pruned branch is information.
- **Done**: only when `done_when` is verifiably met. A parent is done when
  its own `done_when` holds and no descendants are open.
- A **seed is complete** when SEED.md's definition of done is met and the
  subtree is closed. Then write `seed2tree/COMPLETE.md` summarising what was
  built and what remains as optional future work, and set the seed `done`.
- Blocked nodes: write `blocked_on` as a question answerable in one reply, and
  put it in the digest's "Needs you" section.

## 7. DIGEST.md entry (prepend, keep under 15 lines)

```
## 2026-10-15 03:00 PT — sketch-to-code / s2c-012 Per-group fidelity
Picked because: priority 2.41 (next: fash-007 1.98, s2c-009 1.70)
Did: …  (commit abc1234)
Result: …  momentum 0.5 → 0.7
New branches: s2c-018 …, s2c-019 …
Pruned/done: …
Needs you: … (or "nothing")
```

## 8. The local run (`needs: local` nodes)

A separate, less frequent task runs only while the owner's Mac is awake. It
follows these same rules but only waters `needs: local` nodes. It has a Linux
shell over the Mac's project folders (not macOS itself: no Xcode, no Mac
`.venv`; build a fresh venv if Python is needed). It must not modify the
owner's working tree: copy results into the cloud session and commit them
to `seed2tree` from there. If a local node can't run in that environment,
mark it `blocked-on-human` with what is needed.
