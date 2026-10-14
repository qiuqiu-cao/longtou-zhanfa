"""Display authenticated post-freeze disclosure annotations in daily reviews.

The full update is reconstructed from its hash-bound manual input and archived
PDFs. Source hashes alone cannot authenticate an edited annotation payload.
This attachment never changes calls, rankings, probabilities or trade status.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import bind_manual_official_review_v283 as binder
import forecast_review_bridge_v277 as forecast
from holiday_official_evidence_v283 import sha
from next_session_expectation_contract_v271 import load_registered, now, stamp

VERSION = "v284-official-review-binding-1"
KEY = "post_freeze_official_evidence"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def authenticate(update_path, archive, registered_path):
    archive = Path(archive).resolve()
    update = read(update_path)
    frozen = load_registered(registered_path)
    report_path = archive / "report.json"
    report = read(report_path)
    fixed = {str(Path(p).resolve()) for p in (registered_path, report_path, binder.__file__)}
    hashes = update["source_hashes"]
    if not fixed.issubset(hashes) or len(hashes) != len(fixed) + 1:
        raise ValueError("manual update must bind exactly its input, archive, registry and binder")
    for path, wanted in hashes.items():
        if sha(path) != wanted:
            raise ValueError("manual update source hash mismatch")
    manual_path = next(iter(set(hashes) - fixed))
    manual = read(manual_path)
    binder.validate_candidate_scope(manual, frozen)
    if report["registration_hash"] != frozen["sha256"]:
        raise ValueError("official archive registration mismatch")
    for path, wanted in report["artifact_hashes"].items():
        resolved = (archive / path).resolve()
        if not resolved.is_relative_to(archive) or sha(resolved) != wanted:
            raise ValueError("official archive artifact hash mismatch")
    for path, wanted in report["source_hashes"].items():
        if sha(path) != wanted:
            raise ValueError("official archive source hash mismatch")
    recorded = update["recorded_at"]
    if stamp(recorded) < stamp(frozen["recorded_at"]):
        raise ValueError("post-freeze annotation predates the frozen registration")
    if stamp(report["collected_until"]) > stamp(recorded):
        raise ValueError("official archive not yet available at manual review")
    rows = read(archive / "fulltext_evidence.json")
    evidence = {(r["code"], r["announcement_id"]): r for r in rows}
    if len(evidence) != len(rows):
        raise ValueError("duplicate official document identity")
    updates = []
    for row in manual["candidates"]:
        item = dict(row)
        if row["announcement_id"] is None:
            if any(code == row["code"] for code, _ in evidence):
                raise ValueError("unreviewed candidate has selected official documents")
            item.update(review_status="BOUNDED_NO_SELECTED_DOCUMENT_NOT_CLEAN",
                        manually_reviewed_at=recorded)
        else:
            item.update(binder.source_review(row, evidence, recorded))
        item.update(full_risk_coverage=False, hard_veto_from_keyword_scan=False,
                    eligible_for_original_frozen_forecast=False, probability_adjustment=None,
                    original_call_rewrite_allowed=False, trade_authorized=False)
        updates.append(item)
    visual = [{**row, **binder.source_review(row, evidence, recorded)}
              for row in manual["supplementary_visual_reviews"]]
    keyword = manual["keyword_false_positive_case"]
    if keyword.get("assert_no_investigation_anywhere") is not False:
        raise ValueError("keyword context cannot certify whole-document absence of risk")
    keyword = {**keyword, **binder.source_review(keyword, evidence, recorded),
               "hard_veto_authorized": False}
    expected = dict(version="v283-hash-bound-manual-official-update-1", recorded_at=recorded,
        registration_hash=frozen["sha256"], status="POST_FREEZE_MANUAL_EVIDENCE_UPDATE",
        signal_date=frozen["plan"]["signal_date"], target_date=frozen["plan"]["target_date"],
        candidate_updates=updates, supplementary_visual_reviews=visual, keyword_context_case=keyword,
        candidate_denominator=len(updates),
        fulltext_reviewed_candidates=sum(r["announcement_id"] is not None for r in updates),
        policy=manual["policy"], accuracy_claim_allowed=False, source_hashes=hashes)
    if update != expected:
        raise ValueError("annotation payload differs from its hash-bound manual source")
    return update, frozen


def attach(review, update, frozen):
    # Run before attaching a next-session forecast to a newly authored review.
    if forecast.KEY in review:
        raise ValueError("attach evidence before forecast binding; preserve the original frozen review")
    asof = stamp(review["asof"])
    if asof > now():
        raise ValueError("review knowledge time cannot be in the future")
    if stamp(update["recorded_at"]) > asof:
        raise ValueError("manual official update unavailable at review knowledge time")
    if review["signal_date"] != frozen["plan"]["signal_date"]:
        raise ValueError("official update signal date mismatch")
    pool = review["relay_candidates"]
    current = {r["code"]: r["boards"] for r in pool}
    expected = {r["code"]: r["boards"] for r in frozen["plan"]["candidates"]}
    if len(current) != len(pool) or current != expected or review["pool_summary"]["relay"] != len(pool):
        raise ValueError("official update requires exact full candidate pool and board counts")
    review[KEY] = dict(version=VERSION, status="AUTHENTICATED_POST_FREEZE_APPENDIX",
        original_registration_hash=frozen["sha256"], original_forecast_recorded_at=frozen["recorded_at"],
        manual_update_recorded_at=update["recorded_at"], signal_date=update["signal_date"],
        target_date=update["target_date"], candidate_denominator=update["candidate_denominator"],
        fulltext_reviewed_candidates=update["fulltext_reviewed_candidates"],
        candidate_updates=deepcopy(update["candidate_updates"]),
        keyword_context_case=deepcopy(update["keyword_context_case"]),
        original_calls_rewritten=False, automatic_rank_change_allowed=False,
        probability_adjustment=None, trade_authorized=False,
        forward_accuracy_eligible=False, accuracy_claim_allowed=False,
        all_candidates_certified_clean=False)
    return review


def markdown(attachment):
    if attachment is None:
        return ""
    def safe(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["\n## 冻结后官方公告证据", "",
        f"完整候选池{attachment['candidate_denominator']}只，"
        f"其中{attachment['fulltext_reviewed_candidates']}只完成所选全文核验。",
        f"新增证据记录于 `{attachment['manual_update_recorded_at']}`，"
        f"关联原登记 `{attachment['original_registration_hash']}`。",
        "新增材料用于纠正题材依据与风险描述；原晋级判断、排序、概率和交易状态保持原记录。",
        "本附页不计为新事前预测，未取得所选公告的候选仍为未知。", "",
        "| 代码 | 核验事实 | 被否定的具体依据 | 来源 |",
        "|---|---|---|---|"]
    for row in attachment["candidate_updates"]:
        source = (f"[官方PDF]({row['official_url']})，第"
                  + "/".join(map(str, row["source_pages"])) + "页"
                  if row.get("official_url") else "本窗口无所选材料，未证实清洁")
        lines.append(f"| {row['code']} | {safe('；'.join(row['facts']))} | "
                     f"{safe('；'.join(row['invalidated_theses']) or '未记录具体否定项')} | {source} |")
    case = attachment["keyword_context_case"]
    lines += ["", f"关键词误判核验：{case['code']}，{case['finding']} "
        f"范围：{case['review_scope']}。硬否决未授权。", ""]
    return "\n".join(lines)


def run(review_path, update_path, archive, registered_path, asof, output):
    target = Path(output)
    if target.exists():
        raise FileExistsError("refusing to overwrite an official evidence appendix")
    update, frozen = authenticate(update_path, archive, registered_path)
    original = read(review_path)
    binding = forecast.audit(original, frozen)
    if binding["status"] != "BOUND":
        raise ValueError("appendix source must be the exact originally bound review")
    review = deepcopy(original)
    review.pop(forecast.KEY, None)
    review["asof"] = asof
    review["original_frozen_review_reference"] = dict(path=str(Path(review_path).resolve()),
        sha256=sha(review_path), original_review_hash=binding["review_hash"],
        registration_hash=frozen["sha256"], original_forward_registration_unchanged=True)
    attach(review, update, frozen)
    paths = [review_path, update_path, Path(archive) / "report.json", registered_path,
             Path(__file__), Path(binder.__file__), Path(forecast.__file__)]
    review["appendix_source_hashes"] = {str(Path(p).resolve()): sha(p) for p in paths}
    forecast.attach(review)
    review[forecast.KEY]["issues"] = ["EVIDENCE_APPENDIX_IS_NOT_A_NEW_FROZEN_FORECAST"]
    target.mkdir(parents=True, exist_ok=False)
    (target / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2,
                                                  allow_nan=False), encoding="utf-8")
    frozen_text = Path(review_path).with_suffix(".md")
    # The old text is quoted unchanged and explicitly labelled with its old clock.
    prior = frozen_text.read_text(encoding="utf-8") if frozen_text.exists() else ""
    content = (f"# {review['signal_date']} 长假复盘证据补充\n\n"
        f"新增复盘知识截止：`{asof}`。原登记仍等待{update['target_date']}结算。\n"
        + markdown(review[KEY]) + forecast.markdown(review[forecast.KEY])
        + ("\n## 原冻结复盘存档引用\n\n"
           f"以下为原件，原知识截止 `{original['asof']}`；新增证据不倒填其中。\n\n" + prior if prior else ""))
    (target / "review.md").write_text(content, encoding="utf-8")
    return {"status": review[KEY]["status"], "candidate_denominator": len(review["relay_candidates"]),
            "new_forecast_registered": False, "original_calls_rewritten": False,
            "trade_authorized": False, "accuracy_claim_allowed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("review", "update", "archive", "registered", "asof", "output"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.review, args.update, args.archive, args.registered,
                         args.asof, args.output), ensure_ascii=True))
