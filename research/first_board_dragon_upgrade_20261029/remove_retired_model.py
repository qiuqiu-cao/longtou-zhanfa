"""Delete only the explicitly retired 54-feature model; preserve old models."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
assert str(ROOT).lower() == r"D:\Documents\ChatGPT\龙头战法".lower()
TARGETS = [
    "first_board_dragon_engine.py",
    "audit_first_board_dragon_upgrade.py",
    "score_1028_first_boards.py",
    "test_dual_engine_opt.py",
    "tests/test_first_board_dragon_engine.py",
    "research/first_board_dragon_upgrade_20261029/first_board_dragon_engine_bundle.joblib",
    "research/first_board_dragon_upgrade_20261029/diagnose_boost.py",
    "research/first_board_diagnostic_20261029/diagnose_saved_engine.py",
    "__pycache__/first_board_dragon_engine.cpython-313.pyc",
    "tests/__pycache__/test_first_board_dragon_engine.cpython-313-pytest-9.1.1.pyc",
]

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

targets = [(ROOT / name).resolve() for name in TARGETS]
for p in targets:
    assert p.is_relative_to(ROOT) and p != ROOT
    assert not p.exists() or p.is_file()
protected = [ROOT / name for name in (
    "first_board_leader_model.py", "first_second_stage_model.py",
    "first_board_capital_risk_model.py", "relay_model.py")]
for folder in ("first_board_leader_20261020T114600Z", "first_second_stage_20261020_run1",
               "first_board_capital_risk_20261020_run1"):
    protected.extend((ROOT / "research" / folder).rglob("*.joblib"))
protected_hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
receipt = dict(recorded_at=datetime.now(timezone.utc).isoformat(),
    status="DELETION_PLANNED", user_instruction="删除54字段首板潜龙新模型，停止推进",
    files=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=digest(p))
           for p in targets if p.exists()], protected_sha256=protected_hashes)
receipt_path = Path(__file__).parent / "DELETION_RECEIPT.json"
assert not receipt_path.exists(), "Keep prior deletion receipts intact"
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
for p in targets:
    if p.exists():
        p.unlink()
assert all(not p.exists() for p in targets)
assert all(digest(ROOT / name) == sha for name, sha in protected_hashes.items())
receipt.update(status="DELETED_STOPPED_NO_RETRY", deleted_count=len(receipt["files"]),
               protected_hashes_unchanged=True, verified_at=datetime.now(timezone.utc).isoformat())
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(dict(deleted=receipt["deleted_count"], protected_files=len(protected_hashes),
                     protected_unchanged=True, remaining_targets=0)))
