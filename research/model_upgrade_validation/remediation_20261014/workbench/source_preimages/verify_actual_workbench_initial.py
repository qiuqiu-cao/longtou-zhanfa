"""Replay existing completed diagnostic review; no forecasts or fitting."""
from html.parser import HTMLParser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import relay_review_workbench as workbench
import stock_judgement_review as original


class PageAudit(HTMLParser):
    def __init__(self):
        super().__init__()
        self.article_ids = []

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.article_ids.append(dict(attrs).get("id"))


def main():
    base = ROOT / "research/model_upgrade_validation/forum_reasoning_20261014/actual_full15"
    out = Path(__file__).resolve().parent
    facts = base / "facts_with_forum_reasoning.json"
    review = base / "completed_argument_review.json"
    core = {p: original.sha(ROOT / p) for p in ("forum_reasoning_review.py", "supplement_role_review.py",
        "leader_role_review.py", "forum_reasoning_outcome_context.py")}
    report = workbench.render(facts, out / "actual_full15", review)
    packet = workbench.compact(facts, review, out / "shared_reason_draft.json")
    expanded = workbench.expand(out / "shared_reason_draft.json", out / "expanded_argument_review.json")
    saved = original.read(review)
    fields = sum(len(e["original_locations"]) for e in packet["reason_bank"].values())
    qa = PageAudit()
    qa.feed((out / "actual_full15/review.html").read_text(encoding="utf-8"))
    checks = {
        "full_pool": len(qa.article_ids) == 15 and len(set(qa.article_ids)) == 15,
        "html_codes_exact": set(qa.article_ids) == {"stock-" + r["code"] for r in saved["rows"]},
        "rows_exact": expanded["rows"] == saved["rows"],
        "metadata_except_added_bindings_exact": all(expanded[k] == v for k, v in saved.items() if k != "source_bindings"),
        "directions_all_none": all(r["promotion_judgement"] is None for r in expanded["rows"]),
        "real_exposure": report["model_exposure"],
        "diagnostic_only": report["diagnostic_only"],
        "reason_fields": fields,
        "distinct_shared_reasons": len(packet["reason_bank"]),
        "duplicate_reuse_fields": fields - len(packet["reason_bank"]),
        "reuse_percent": round((fields - len(packet["reason_bank"])) * 100 / fields, 2),
        "forum_opinions": sum(len(r["opinions"]) for r in report["cards"]),
        "model_output_displayed": report["model_output_displayed"],
        "new_fits": 0,
        "new_forecasts": 0,
        "core_hashes_unchanged_during_run": all(original.sha(ROOT / p) == h for p, h in core.items()),
        "original_core_hashes": core,
        "source_bindings": original.bind(facts) | original.bind(review) | original.bind(ROOT / "relay_review_workbench.py") |
            original.bind(ROOT / "tests/test_relay_review_workbench.py") | original.bind(__file__),
    }
    assert all(checks[k] for k in ("full_pool", "html_codes_exact", "rows_exact", "metadata_except_added_bindings_exact",
        "directions_all_none", "diagnostic_only", "core_hashes_unchanged_during_run"))
    original.write_new(out / "actual_verification.json", checks)
    print(json.dumps({k: v for k, v in checks.items() if k not in {"source_bindings", "original_core_hashes"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
