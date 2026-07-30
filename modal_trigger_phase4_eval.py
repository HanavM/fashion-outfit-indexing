"""Fire-and-forget trigger for the deployed Phase 4 eval. Same spawn()-
against-a-deployed-app pattern as modal_trigger_v3.py.

Usage: python3 modal_trigger_phase4_eval.py
"""

import modal

fn = modal.Function.from_name("fashion-phase4-eval", "evaluate")
call = fn.spawn()
print("Spawned function call:", call.object_id)
