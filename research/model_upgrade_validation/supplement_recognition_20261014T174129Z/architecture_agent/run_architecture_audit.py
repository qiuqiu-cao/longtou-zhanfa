"""Read-only architecture verification; no acquisition, fitting or shared edits."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import leader_role_engine as engine

files = ["leader_role_engine.py", "leader_role_review.py", "stock_forum_preread.py",
         "stock_catalyst_context.py", "stock_narrative_review.py", "stock_short_term_context.py",
         "tests/test_leader_roles.py", "tests/test_leader_role_review_draft.py",
         "tests/test_stock_forum_preread.py"]
source_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
tree = ast.parse((ROOT/"leader_role_review.py").read_text(encoding="utf-8-sig"))
imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
imports += [a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names]
prepare = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare")
calls = [ast.unparse(node.func) for node in ast.walk(prepare) if isinstance(node, ast.Call)]
row = {"code": "000001", "signal_date": "2026-10-09", "decision_at": "2026-10-09T18:00:00+08:00"}
dependency = {"anchor_code": "000002", "same_specific_theme": True,
              "anchor_identity_source": "fixture_prior_review", "anchor_stable": True,
              "anchor_start": "2026-10-06", "own_start": "2026-10-08"}
checks = {
    "same_specific_theme_dependency_accepted": engine._valid_group("dependency", dependency, row),
    "cross_industry_dependency_rejected": not engine._valid_group("dependency", {**dependency, "same_specific_theme": False}, row),
    "missing_specific_theme_dependency_rejected": not engine._valid_group("dependency", {k:v for k,v in dependency.items() if k != "same_specific_theme"}, row),
    "single_anchor_required": "anchor_code" in dependency and "anchor_codes" not in dependency,
    "formal_supplement_required_groups": list(engine.REQUIRED["SUPPLEMENT"]),
    "preread_not_imported_by_role_review": "stock_forum_preread" not in imports,
    "preread_not_called_by_prepare": not any("preread" in name for name in calls),
}
tests = ["tests.test_stock_forum_preread", "tests.test_leader_role_review_draft",
         "tests.test_leader_roles.RolesTest.test_supplement_divergence_can_precede_anchor_break_and_future_break_is_rejected",
         "tests.test_leader_roles.RolesTest.test_promotion_miss_cannot_be_rewritten_as_role_false_negative",
         "tests.test_leader_roles.RolesTest.test_role_diagnosis_preserves_multirole_abstention_and_missing_inputs"]
completed = subprocess.run([sys.executable, "-B", "-m", "unittest", *tests], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
result = {"formed_at": datetime.now(timezone.utc).isoformat(),
          "status": "READ_ONLY_SUPPLEMENT_ARCHITECTURE_AUDIT",
          "source_hashes": source_hashes, "checks": checks,
          "prepare_calls": calls, "tests": {"command": [sys.executable, "-B", "-m", "unittest", *tests],
          "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr},
          "fitting": 0, "new_sources": 0, "shared_files_edited": False,
          "architecture_checks_are_not_prediction_improvement": True}
target = OUT/"architecture_verification.json"
with target.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"checks": checks, "test_exit_code": completed.returncode,
                  "test_output": completed.stderr, "output": str(target)}, ensure_ascii=False))
