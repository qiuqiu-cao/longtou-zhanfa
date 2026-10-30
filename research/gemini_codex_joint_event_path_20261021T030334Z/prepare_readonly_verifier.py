"""Prepare the existing saved-model replay checks for the isolated event batch."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
source = ROOT / "research/gemini_codex_joint_long_high_20261021T022449Z/verify_saved_joint_result.py"
destination = Path(__file__).resolve().parent / "verify_saved_joint_result.py"
if destination.exists():
    raise FileExistsError("Preserve original verifier")
text = source.read_text("utf-8")
assert text.count("import first_board_joint_long_high_model as model") == 1
text = text.replace("import first_board_joint_long_high_model as model", "import first_board_joint_event_path_model as model")
assert text.count('out = Path(__file__).resolve().parent\nfreeze = model.read_freeze()') == 1
text = text.replace('out = Path(__file__).resolve().parent\nfreeze = model.read_freeze()',
                    'out = Path(__file__).resolve().parent\nmodel.OUTPUT = out\nfreeze = model.read_freeze()')
assert text.count("model.features.FIELDS") == 2
text = text.replace("model.features.FIELDS", "model.ALL_FIELDS")
with destination.open("x", encoding="utf-8", newline="\n") as f:
    f.write(text)
print("READONLY_REPLAY_VERIFIER_PREPARED_NO_FITS")
