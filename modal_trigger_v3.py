"""Fire-and-forget trigger for the deployed v3 SigLIP2 fine-tune.

Exists because `modal run --detach modal_app_v3.py` turned out not to
survive this environment's background-shell-process lifetime (~90 min in)
even though --detach is documented to survive client disconnection -- the
run got cancelled mid stage-1-epoch-6 with no billing/quota/error in
Modal's own logs, coinciding with the local client process being reaped.
`spawn()` against an already-`modal deploy`-ed app dispatches the job and
returns immediately; the deployed app has no dependency on this process
staying alive at all, so there's nothing left to reap.

Usage: python3 modal_trigger_v3.py
"""

import modal

train_fn = modal.Function.from_name("fashion-siglip2-v3-finetune", "train")
call = train_fn.spawn()
print("Spawned function call:", call.object_id)
