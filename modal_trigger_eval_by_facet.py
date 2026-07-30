"""Fire-and-forget trigger for the deployed by-facet eval. See
modal_trigger_v3.py's docstring for why spawn()-against-a-deployed-app is
used instead of `modal run --detach`.

Usage: python3 modal_trigger_eval_by_facet.py
"""

import modal

fn = modal.Function.from_name("fashion-siglip2-eval-by-facet", "evaluate")
call = fn.spawn()
print("Spawned function call:", call.object_id)
