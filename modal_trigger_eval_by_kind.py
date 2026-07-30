"""Fire-and-forget trigger for the deployed by-label-kind eval, same
pattern as modal_trigger_v3.py (see that file's docstring for why: `modal
run --detach` didn't reliably survive this environment's background-shell
lifetime, spawn() against a deployed app does).

Usage: python3 modal_trigger_eval_by_kind.py
"""

import modal

fn = modal.Function.from_name("fashion-siglip2-eval-by-kind", "evaluate")
call = fn.spawn()
print("Spawned function call:", call.object_id)
