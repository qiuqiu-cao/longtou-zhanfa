"""Causal six-state context and separate saved-history/forward stratification."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

STATES = ("冰点", "修复", "升温", "加速/一致", "分歧", "退潮", "UNKNOWN")
HISTORY_MAP = {"修复": "修复", "升温": "升温", "退潮": "退潮", "加速": "加速/一致",
               "一致": "加速/一致", "加速/一致": "加速/一致", "大分歧": "分歧"}


def stamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone-aware context timestamp required")
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unknown_context(signal_date):
    return {"version": "signal-cycle-context-1", "signal_date": signal_date,
            "state": "UNKNOWN", "formed_at": datetime.now(timezone.utc).isoformat(),
            "evidence": [], "reason": "未形成截止前至少三项来源绑定的当日周期证据；不以目标日预测或结果代替"}


def validate_cycle(context, signal_date, cutoff, recorded_at=None):
    """Validate evidence values against immutable source JSON, not target labels."""
    if not context:
        return "UNKNOWN"
    if context.get("version") != "signal-cycle-context-1" or context.get("signal_date") != signal_date:
        raise ValueError("cycle context version/signal date conflict")
    state = context.get("state")
    if state not in STATES:
        raise ValueError("unsupported signal cycle state")
    formed = stamp(context["formed_at"])
    if formed > stamp(cutoff) or (recorded_at and formed > stamp(recorded_at)):
        raise ValueError("cycle context follows knowledge/registration cutoff")
    if formed < stamp(signal_date + "T15:00:00+08:00"):
        raise ValueError("closing cycle context precedes signal close")
    if state == "UNKNOWN":
        if not context.get("reason"):
            raise ValueError("UNKNOWN cycle requires gap reason")
        return state
    evidence = context.get("evidence", [])
    if len(evidence) < 3 or len({e.get("id") for e in evidence}) != len(evidence):
        raise ValueError("known cycle needs at least three distinct evidence items")
    pointers = set()
    for item in evidence:
        if not item.get("id") or item.get("signal_date") != signal_date or item.get("value") is None:
            raise ValueError("cycle evidence identity/date/value unavailable")
        observed, available = stamp(item["observed_at"]), stamp(item["available_at"])
        if observed > available or available > formed:
            raise ValueError("cycle evidence follows formation or is inconsistent")
        if observed.astimezone(stamp(signal_date+"T00:00:00+08:00").tzinfo).date().isoformat() != signal_date:
            raise ValueError("cycle observation belongs to another session")
        path = Path(item["source_path"])
        if not path.is_file() or sha(path) != item["source_sha256"]:
            raise ValueError("cycle evidence source hash conflict")
        source = json.loads(path.read_text(encoding="utf-8"))
        source_time = next((source[k] for k in ("generated_at", "recorded_at", "formed_at", "available_at") if source.get(k)), None)
        if source_time is None or stamp(source_time) > available:
            raise ValueError("cycle evidence source formation is unbound or after availability")
        pointer = item.get("source_pointer")
        if not isinstance(pointer, list) or not pointer:
            raise ValueError("cycle evidence must bind an exact JSON value")
        key = (str(path.resolve()), json.dumps(pointer, sort_keys=True))
        if key in pointers:
            raise ValueError("three evidence items cannot repeat the same value")
        pointers.add(key)
        value = source
        for part in pointer:
            if str(part) in {"target", "truth", "outcomes", "promoted_next_day"} or str(part).startswith(("trade_", "exit_", "target_", "outcome")):
                raise ValueError("target evidence cannot label signal cycle")
            value = value[part]
        if value != item["value"]:
            raise ValueError("cycle evidence value differs from source")
    return state


def metrics(rows):
    known = [r for r in rows if type(r.get("truth")) is bool and type(r.get("prediction")) is bool]
    tp = sum(r["truth"] and r["prediction"] for r in known)
    tn = sum(not r["truth"] and not r["prediction"] for r in known)
    fp = sum(not r["truth"] and r["prediction"] for r in known)
    fn = sum(r["truth"] and not r["prediction"] for r in known)
    positives = sum(r.get("truth") is True for r in rows)
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/positives if positives else None
    f1 = (2*precision*recall/(precision+recall) if precision+recall else 0.) if precision is not None and recall is not None else None
    days = len({r["signal_date"] for r in rows})
    return {"original_registered_count": len(rows), "scoreable_count": len(known),
            "coverage": len(known)/len(rows) if rows else None,
            "Accuracy": (tp+tn)/len(known) if known else None, "Precision": precision,
            "complete_positive_Recall": recall, "F1": f1,
            "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
            "complete_positive_misses": positives-tp, "positive_support": positives,
            "signal_days": days, "calibrated_Brier": None,
            "sample_note": "样本不足：少于20个信号日或20个已知正例；不能据此改分数门槛" if days < 20 or positives < 20 else "仅分层描述，不自动调阈值"}


def summarise(rows, sessions, last_mature, day_states=None):
    eligible = [d for d in sessions if d <= last_mature]
    day_states = day_states if day_states is not None else {r["signal_date"]: r["regime"] for r in rows}
    result = {}
    for label, n in (("20", 20), ("60", 60), ("all", None)):
        first_registered = min(day_states, default=last_mature)
        window = eligible[-n:] if n else [d for d in eligible if d >= first_registered]
        selected = [r for r in rows if r["signal_date"] in set(window)]
        result[label] = {"official_window_start": window[0] if window else None,
            "official_window_end": window[-1] if window else None,
            "official_session_count": len(window), "overall": metrics(selected),
            "by_state": {s: metrics([r for r in selected if r["regime"] == s]) for s in STATES},
            "by_boards": {str(b): metrics([r for r in selected if r["boards"] == b])
                          for b in sorted({r["boards"] for r in selected})}}
        result[label]["overall"]["candidate_signal_days"] = result[label]["overall"]["signal_days"]
        result[label]["overall"]["signal_days"] = sum(d in set(window) for d in day_states)
        for state, value in result[label]["by_state"].items():
            value["candidate_signal_days"] = value["signal_days"]
            value["signal_days"] = sum(s == state and d in set(window) for d, s in day_states.items())
            value["by_boards"] = {str(b): metrics([r for r in selected if r["regime"] == state and r["boards"] == b])
                                  for b in sorted({r["boards"] for r in selected if r["regime"] == state})}
    return result


def run(registry, ledger, calendar_path, output, history_root=None, asof=None):
    import current_model_diagnostics as d
    from next_session_expectation_contract_v271 import load_registered
    from hypothesis_bound_daily_runner_v298 import read_chain, current_results
    cutoff = asof or d.now()
    if stamp(cutoff) > stamp(d.now()):
        raise ValueError("future diagnostic cutoff")
    calendar = d._calendar(calendar_path)
    sessions = calendar["sessions"]
    successor = dict(zip(sessions[:-1], sessions[1:]))
    mature = [s for s, t in successor.items() if stamp(t+"T15:00:00+08:00") <= stamp(cutoff)]
    last = max(mature) if mature else "0000-00-00"
    results = {r["registration_hash"]: r for r in current_results(read_chain(ledger), cutoff)}
    rows, pending, day_states, sources = [], [], {}, {str(Path(calendar_path).resolve()): sha(calendar_path), str(Path(ledger).resolve()): sha(ledger)}
    for path in sorted(Path(registry).glob("*.json")):
        registered = load_registered(path)
        if stamp(registered["recorded_at"]) > stamp(cutoff) or registered["forecast_status"] != "FORWARD_REGISTERED":
            continue
        plan = registered["plan"]
        day, target = plan["signal_date"], plan["target_date"]
        if successor.get(day) != target:
            raise ValueError("registration target is not exact official successor")
        sources[str(path.resolve())] = sha(path)
        regime = validate_cycle(plan.get("cycle_context"), day, registered["knowledge_cutoff"], registered["recorded_at"])
        if day > last:
            pending.append({"signal_date": day, "target_date": target, "original_candidates": len(plan["candidates"]), "regime": regime})
            continue
        if day in day_states:
            raise ValueError("duplicate forward signal registrations would double count a state day")
        day_states[day] = regime
        result = results.get(registered["sha256"])
        truth_rows = {r["key"]: r for r in result["rows"] if r["layer"] == "promotion"} if result else {}
        if result and set(truth_rows) != {r["code"] for r in plan["candidates"]}:
            raise ValueError("settlement promotion roster is incomplete/extra")
        for candidate in plan["candidates"]:
            outcome = truth_rows.get(candidate["code"], {})
            prediction = candidate.get("predicted_positive")
            # V271 legacy predictions use promotion.call; current model raw calls stay authoritative.
            if prediction is None:
                prediction = candidate.get("promotion", {}).get("call")
            if prediction is None:
                prediction = {"PROMOTE": True, "FAIL": False}.get(candidate.get("promotion_call"))
            if outcome and type(prediction) is bool and outcome["call"] != prediction:
                raise ValueError("settlement call differs from original prediction")
            rows.append({"signal_date": day, "target_date": target, "code": candidate["code"],
                "boards": candidate["boards"], "regime": regime, "prediction": prediction,
                "truth": outcome.get("truth") if outcome.get("outcome_known") is True else None})
    report = {"status": "REGIME_DIAGNOSTIC_NO_THRESHOLD_CHANGE", "generated_at": d.now(), "knowledge_cutoff": cutoff,
        "true_forward": summarise(rows, sessions, last, day_states), "pending_registrations": pending,
        "forward_rows": rows, "history": None, "model_or_prediction_changed": False,
        "new_forward_samples": 0, "accuracy_change": "NO_ACCURACY_CHANGE"}
    if history_root:
        frame, _, _, bound = d._bound_history(history_root, "relay", cutoff)
        sources.update(bound)
        market, _, _, bound = d._bound_history(history_root, "market", cutoff)
        sources.update(bound)
        native_states = {r["date"]: HISTORY_MAP.get(r.get("emotion_state"), "UNKNOWN") for r in market.to_dict("records")}
        history_days = {day: native_states.get(day, "UNKNOWN") for day in sessions
                        if any(f["start"] <= day <= f["end"] for f in d.FOLDS)}
        historical = []
        for raw in frame.to_dict("records"):
            target = successor.get(raw["date"])
            truth, reason = d._truth(raw, "relay", raw["date"], target, cutoff)
            call = d._boolean(raw.get("predicted")) if raw.get("status") == "PREDICTED_UNCALIBRATED" else None
            historical.append({"signal_date": raw["date"], "target_date": target, "code": raw["code"],
                "boards": int(raw["boards"]), "regime": HISTORY_MAP.get(raw.get("emotion_state"), "UNKNOWN"),
                "native_state": raw.get("emotion_state") if isinstance(raw.get("emotion_state"), str) else None,
                "prediction": call, "truth": truth, "truth_status": reason})
        hlast = max((r["signal_date"] for r in historical), default=last)
        report["history"] = {"status": "SAVED_HISTORY_NOT_FORWARD_NOT_BLIND", "windows": summarise(historical, sessions, min(last, hlast), history_days),
            "mapping": HISTORY_MAP, "unmapped": "启动、缺失及其他状态UNKNOWN；无历史冰点认证", "rows": historical}
    sources[str(Path(__file__).resolve())] = sha(__file__)
    report["source_hashes"] = sources
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    d._write(directory/"report.json", report)
    lines = ["# 周期分层独立附页", "", "真实前向与历史保存诊断分开；没有校准概率，不计算概率Brier。", "",
             "|范围|原数量|可评分|Accuracy|Precision|完整Recall|F1|", "|---|---:|---:|---:|---:|---:|---:|"]
    for scope, group in (("真实前向", report["true_forward"]), ("历史保存", report["history"]["windows"] if report["history"] else {})):
        for window, value in group.items():
            for state, m in [("全部", value["overall"])] + list(value["by_state"].items()):
                fmt = lambda x: "未知" if x is None else f"{100*x:.2f}%"
                lines.append(f"|{scope}/{window}/{state}|{m['original_registered_count']}|{m['scoreable_count']}|{fmt(m['Accuracy'])}|{fmt(m['Precision'])}|{fmt(m['complete_positive_Recall'])}|{fmt(m['F1'])}|")
    (directory/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report
