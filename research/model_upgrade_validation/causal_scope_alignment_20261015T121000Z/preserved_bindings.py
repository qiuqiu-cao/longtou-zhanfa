"""Resolve one authenticated, externally superseded instruction document.

No numerical input, code, model, prediction, or registry mismatch is allowed.
The old hash remains attached to exact recovered bytes; current instructions
are bound separately. Original receipts/files are never edited or rolled back.
"""
from pathlib import Path
import short_term_thinking_dataset_builder as b

ROOT=Path(__file__).resolve().parent


def resolve(bindings):
    event=b.read(ROOT/"external_instruction_update_receipt.json")
    old_path=event["original_live_path"];archive=event["archived_exact_bytes_path"]
    if b.sha(old_path)!=event["current_sha256"] or b.sha(archive)!=event["old_sha256"]:
        raise ValueError("instruction revision changed again; reread before proceeding")
    result={}
    for path,digest in bindings.items():
        if path==old_path and digest==event["old_sha256"]:
            result[archive]=digest
        else:result[path]=digest
    result[old_path]=event["current_sha256"]
    for path in [Path(__file__),ROOT/"external_instruction_update_receipt.json",b.ROOT/event["new_template"]]:
        result[str(path.resolve())]=b.sha(path)
    return result


def verify(bindings):
    current=resolve(bindings);b.verify(current);return current
