"""Fire-and-forget trigger for the deployed v4 SigLIP2 fine-tune. Same
spawn()-against-a-deployed-app pattern as modal_trigger_v3.py, for the
same reliability reason (see modal_app_v4.py's docstring).

Usage: python3 modal_trigger_v4.py
"""

import modal

train_fn = modal.Function.from_name("fashion-siglip2-v4-finetune", "train")
call = train_fn.spawn()
print("Spawned function call:", call.object_id)
