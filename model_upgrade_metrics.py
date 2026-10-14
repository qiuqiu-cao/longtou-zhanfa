"""Pure metric and admission functions for saved model-upgrade records.

No market collection, fitting, filesystem access, or clock reads belong here.
Unknown labels and forecasts retain their original official-session denominator.
Raw model scores are diagnostics, never calibrated trading probabilities.
"""
from __future__ import annotations

from collections import Counter
from math import ceil, isfinite, log
from typing import Any, Mapping, Sequence

import numpy as np


MARKET_CLASSES = ("IMPROVE", "CONTINUE", "WORSEN")
ARMS = ("BASE", "C1", "C2")
BOOTSTRAP_SPEC = {
    "draws": 2000,
    "seed": 20261007,
    "block_official_sessions": 5,
    "family_primary_challenges": 4,
    "tails": (0.00625, 0.99375),
}


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, (str, bytes, bool)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def _binary(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    number = _finite(value)
    return int(number) if number in (0.0, 1.0) else None


def _class(value: Any) -> str | None:
    return value if isinstance(value, str) and value in MARKET_CLASSES else None


def _divide(numerator: float, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def classification_metrics(
    truth: Sequence[Any], predicted: Sequence[Any], labels: Sequence[Any],
) -> dict[str, Any]:
    """Classify only explicit label/prediction pairs; retain every original row.

    Complete recall includes known outcomes with an unknown prediction. Macro
    statistics require support for every registered class, including all three
    market classes; absent classes are not silently omitted.
    """
    if len(truth) != len(predicted) or len(set(labels)) != len(labels):
        raise ValueError("equal row counts and unique fixed classes required")
    classes = tuple(labels)
    allowed = set(classes)
    actual_known = [y in allowed for y in truth]
    called = [p in allowed for p in predicted]
    paired = [a and b for a, b in zip(actual_known, called)]
    matrix = {str(y): {str(p): 0 for p in classes} for y in classes}
    for y, p, valid in zip(truth, predicted, paired):
        if valid:
            matrix[str(y)][str(p)] += 1
    n, known, issued = sum(paired), sum(actual_known), sum(called)
    correct = sum(matrix[str(k)][str(k)] for k in classes)
    complete_support = Counter(y for y, valid in zip(truth, actual_known) if valid)
    per_class: dict[str, Any] = {}
    for k in classes:
        tp = matrix[str(k)][str(k)]
        support = sum(matrix[str(k)].values())
        calls = sum(matrix[str(y)][str(k)] for y in classes)
        fp, fn = calls - tp, support - tp
        per_class[str(k)] = {
            "known_outcome_support": int(complete_support[k]),
            "scored_support": support,
            "scored_calls": calls,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": _divide(tp, calls),
            "scored_recall": _divide(tp, support),
            "complete_recall": _divide(tp, complete_support[k]),
            "F1": _divide(2 * tp, 2 * tp + fp + fn),
        }
    all_classes_scored = all(per_class[str(k)]["scored_support"] > 0 for k in classes)
    return {
        "registered_rows": len(truth), "known_outcome_rows": known,
        "predicted_rows": issued, "scored_rows": n,
        "unknown_outcome_rows": len(truth) - known,
        "unknown_prediction_rows": len(truth) - issued,
        "label_coverage": _divide(known, len(truth)),
        "scoring_coverage": _divide(n, len(truth)),
        "valid_label_scoring_coverage": _divide(n, known),
        "correct": correct, "accuracy": _divide(correct, n),
        "confusion_matrix": matrix, "by_class": per_class,
        "all_registered_classes_scored": all_classes_scored,
        "macro_f1": float(np.mean([per_class[str(k)]["F1"] for k in classes]))
        if all_classes_scored else None,
        "balanced_accuracy": float(np.mean([per_class[str(k)]["scored_recall"] for k in classes]))
        if all_classes_scored else None,
    }


def binary_metrics(truth: Sequence[Any], predicted: Sequence[Any]) -> dict[str, Any]:
    result = classification_metrics([_binary(y) for y in truth],
                                    [_binary(p) for p in predicted], (0, 1))
    matrix = result["confusion_matrix"]
    tp, tn = matrix["1"]["1"], matrix["0"]["0"]
    fp, fn = matrix["0"]["1"], matrix["1"]["0"]
    result.update({"tp": tp, "tn": tn, "fp": fp, "fn": fn,
                   "precision": _divide(tp, tp + fp),
                   "complete_positive_recall": result["by_class"]["1"]["complete_recall"],
                   "scored_positive_recall": _divide(tp, tp + fn),
                   "nonpositive_recall": _divide(tn, tn + fp),
                   "FPR": _divide(fp, tn + fp),
                   "F1": _divide(2 * tp, 2 * tp + fp + fn)})
    return result


def raw_binary_score_metrics(truth: Sequence[Any], scores: Sequence[Any]) -> dict[str, Any]:
    if len(truth) != len(scores):
        raise ValueError("score and label row counts differ")
    pairs = []
    for actual, score in zip(truth, scores):
        y, p = _binary(actual), _finite(score)
        if y is not None and p is not None and 0 <= p <= 1:
            pairs.append((y, p))
    return {
        "registered_rows": len(truth), "raw_score_rows": len(pairs),
        "raw_score_coverage": _divide(len(pairs), len(truth)),
        "raw_score_Brier": float(np.mean([(p-y)**2 for y, p in pairs])) if pairs else None,
        "raw_score_logloss": float(np.mean([-y*log(min(max(p, 1e-15), 1-1e-15))
                                            -(1-y)*log(1-min(max(p, 1e-15), 1-1e-15))
                                            for y, p in pairs])) if pairs else None,
        "probability_calibration": "NOT_CALIBRATED_INTERNAL_SCORE",
    }


def raw_market_score_metrics(
    truth: Sequence[Any], scores: Sequence[Mapping[str, Any] | None],
) -> dict[str, Any]:
    if len(truth) != len(scores):
        raise ValueError("score and label row counts differ")
    brier, losses = [], []
    for actual, score in zip(truth, scores):
        if _class(actual) is None or not isinstance(score, Mapping):
            continue
        if set(score) != set(MARKET_CLASSES):
            continue
        p = [_finite(score[k]) for k in MARKET_CLASSES]
        if any(value is None or not 0 <= value <= 1 for value in p):
            continue
        if not np.isclose(sum(p), 1.0, rtol=0, atol=1e-8):
            continue
        brier.append(sum((value-int(k == actual))**2 for k, value in zip(MARKET_CLASSES, p)))
        losses.append(-log(min(max(float(score[actual]), 1e-15), 1-1e-15)))
    return {
        "registered_rows": len(truth), "raw_score_rows": len(brier),
        "raw_score_coverage": _divide(len(brier), len(truth)),
        "raw_score_Brier": float(np.mean(brier)) if brier else None,
        "raw_score_logloss": float(np.mean(losses)) if losses else None,
        "Brier_definition": "mean sum over all three classes of (raw_score-one_hot_truth)^2",
        "probability_calibration": "NOT_CALIBRATED_INTERNAL_SCORE",
    }


def block_interval(values: Sequence[Any]) -> list[float] | None:
    """Family-four 95% intervals on the retained official-session grid.

    Missing dates remain NaN before circular blocks are sampled. This estimates
    a conditional paired mean and does not convert unknown outcomes to errors.
    """
    data = np.asarray([np.nan if _finite(v) is None else float(v) for v in values], dtype=float)
    if not len(data) or not np.isfinite(data).any():
        return None
    spec = BOOTSTRAP_SPEC
    rng = np.random.default_rng(spec["seed"])
    block = spec["block_official_sessions"]
    samples = []
    for _ in range(spec["draws"]):
        starts = rng.integers(0, len(data), size=ceil(len(data)/block))
        indexes = np.concatenate([(s+np.arange(block)) % len(data) for s in starts])[:len(data)]
        sampled = data[indexes]
        finite = sampled[np.isfinite(sampled)]
        if len(finite):
            samples.append(float(finite.mean()))
    return np.quantile(samples, spec["tails"]).tolist() if samples else None


def _calendar(calendar_sessions: Sequence[str]) -> list[str]:
    days = list(calendar_sessions)
    if not days or days != sorted(set(days)) or any(not isinstance(d, str) for d in days):
        raise ValueError("a complete ordered unique official calendar is required")
    # ISO date strings sort chronologically; reject shortened or timestamp keys.
    from datetime import date
    if any(date.fromisoformat(d).isoformat() != d for d in days):
        raise ValueError("official session keys must be YYYY-MM-DD")
    return days


def _records(module: str, records: Any, calendar_sessions: Sequence[str]) -> tuple[list[dict], list[str]]:
    if module not in ("market", "relay"):
        raise ValueError("module must be market or relay")
    calendar = _calendar(calendar_sessions)
    if hasattr(records, "to_dict"):
        records = records.to_dict(orient="records")
    result, seen = [], set()
    positions = {day: i for i, day in enumerate(calendar)}
    for item in records:
        if not isinstance(item, Mapping) or item.get("module") != module:
            raise ValueError("every record must identify the requested module")
        row = dict(item)
        day = row.get("signal_date")
        if day not in positions or day in seen:
            raise ValueError("duplicate or nonofficial signal date")
        expected = calendar[positions[day]+1] if positions[day]+1 < len(calendar) else None
        if row.get("target_date") != expected:
            raise ValueError("record does not bind the exact next official session")
        if not isinstance(row.get("arms", {}), Mapping) or set(row.get("arms", {})) - set(ARMS):
            raise ValueError("unregistered model arm")
        for arm, value in row.get("arms", {}).items():
            if not isinstance(value, Mapping) or value.get("status") not in ("KNOWN", "UNKNOWN"):
                raise ValueError("explicit KNOWN/UNKNOWN prediction status required")
        row["_record_available"] = True
        result.append(row)
        seen.add(day)
    return sorted(result, key=lambda r: r["signal_date"]), calendar


def _align(rows: Sequence[dict], calendar: Sequence[str], dates: Sequence[str]) -> list[dict]:
    by_day = {r["signal_date"]: r for r in rows}
    positions = {day: i for i, day in enumerate(calendar)}
    result = []
    for day in dates:
        if day in by_day:
            result.append(by_day[day])
        else:
            i = positions[day]
            result.append({"signal_date": day, "target_date": calendar[i+1] if i+1 < len(calendar) else None,
                           "arms": {}, "outcomes": {}, "naive": {}, "_record_available": False})
    return result


def _arm(row: Mapping, arm: str) -> Mapping:
    value = row.get("arms", {}).get(arm, {})
    return value if value.get("status") == "KNOWN" else {}


def _market_day(row: Mapping, training_majority: Any = None) -> dict:
    outcome = row.get("outcomes") or {}
    actual = _class(outcome.get("market_target"))
    naive = row.get("naive") or {}
    majority = _class(naive.get("majority"))
    if majority is None:
        majority = _class(training_majority.get(row["signal_date"])) if isinstance(training_majority, Mapping) else _class(training_majority)
    return {
        "actual": actual,
        "arms": {arm: {"prediction": _class(_arm(row, arm).get("prediction")),
                        "raw_scores": _arm(row, arm).get("raw_scores")} for arm in ARMS},
        "naive": {"majority": majority, "persistence": _class(naive.get("persistence"))},
    }


def _code(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 6 or not value.isdecimal():
        raise ValueError("six-digit original security codes required")
    return value


def _relay_day(row: Mapping) -> dict:
    available = bool(row.get("_record_available", True)) and "original_codes" in row
    codes = [_code(c) for c in row.get("original_codes", [])]
    if len(codes) != len(set(codes)):
        raise ValueError("duplicate original candidate code")
    original = set(codes)
    maps, facts = {}, {}
    for arm in ARMS:
        value = row.get("arms", {}).get(arm, {})
        candidates = value.get("candidates") or []
        if not isinstance(candidates, list):
            raise ValueError("candidate records must be a list")
        mapped = {}
        for candidate in candidates:
            code = _code(candidate.get("code"))
            if code not in original or code in mapped:
                raise ValueError("candidate set differs from the original roster")
            mapped[code] = candidate
            word = _binary(candidate.get("actual_one_word_daily"))
            boards = _finite(candidate.get("boards"))
            if boards is not None and (boards < 2 or not boards.is_integer()):
                raise ValueError("relay original candidates must be at least two boards")
            fact = (boards, word)
            if code in facts and facts[code] != fact:
                raise ValueError("arms disagree on original signal-day candidate facts")
            facts[code] = fact
        if value.get("status") == "KNOWN" and set(mapped) != original:
            raise ValueError("KNOWN arm must preserve the complete original candidate group")
        maps[arm] = mapped if value.get("status") == "KNOWN" else {}
    identity_known = available and all(code in facts and facts[code][1] is not None for code in codes)
    nonword = [code for code in codes if code in facts and facts[code][1] == 0]
    factual_eligible = identity_known and bool(nonword)
    known_no_nonword = identity_known and not nonword
    outcome = row.get("outcomes") or {}
    outcomes = outcome.get("candidates") or {}
    if not isinstance(outcomes, Mapping) or set(outcomes) - original:
        raise ValueError("outcome code set differs from the original roster")
    truths = {code: _binary(outcomes.get(code)) for code in codes}
    complete = available and bool(codes) and all(truths[c] is not None for c in codes)
    eligible = factual_eligible and complete
    arm_data = {}
    for arm in ARMS:
        mapped = maps[arm]
        scored = {code: _finite(mapped.get(code, {}).get("raw_score")) for code in codes}
        scored = {c: p if p is not None and 0 <= p <= 1 else None for c, p in scored.items()}
        predicted = {code: _binary(mapped.get(code, {}).get("predicted_positive")) for code in codes}
        rank_known = factual_eligible and all(scored[c] is not None for c in nonword)
        ranked = sorted(nonword, key=lambda c: (-scored[c], c)) if rank_known else []
        positive_known = rank_known and all(predicted[c] is not None for c in nonword)
        positive = [c for c in ranked if predicted[c] == 1] if positive_known else []
        rank_code = ranked[0] if ranked else None
        positive_code = positive[0] if positive else None
        arm_data[arm] = {
            "candidate_predictions": predicted, "candidate_scores": scored,
            "rank_top1_code": rank_code, "rank_top1_outcome": truths[rank_code] if eligible and rank_code else None,
            "positive_top1_code": positive_code,
            "positive_top1_status": "CALLED" if positive_code else "NO_POSITIVE" if positive_known else "UNKNOWN",
            "positive_top1_outcome": truths[positive_code] if eligible and positive_code else None,
        }
    naive_code = (row.get("naive") or {}).get("highest_nonword")
    naive_valid = eligible and naive_code in nonword
    return {"codes": codes, "facts": facts, "truths": truths,
            "roster_available": available, "identity_known": identity_known,
            "confirmed_nonword": factual_eligible, "known_no_nonword": known_no_nonword,
            "unknown_eligibility": not identity_known,
            "complete_original_outcomes": complete, "effective_eligible": eligible,
            "arms": arm_data, "naive_outcome": truths[naive_code] if naive_valid else None}


def _summarize_aligned(module: str, rows: Sequence[dict], training_majority: Any = None) -> dict:
    days = [r["signal_date"] for r in rows]
    n = len(rows)
    source_count = sum(bool(r.get("_record_available", True)) for r in rows)
    forward_count = sum(r.get("forward") is True for r in rows if r.get("_record_available", True))
    result = {"module": module, "signal_dates": days, "registered_dates": n,
              "source_record_dates": source_count, "source_record_coverage": _divide(source_count, n),
              "forward_record_dates": forward_count,
              "evidence_class": "REAL_FORWARD_SAVED_RECORDS" if source_count and forward_count == source_count else "HISTORICAL_OR_MIXED_DIAGNOSTIC",
              "scores_are_calibrated_probabilities": False, "trade_returns_estimated": False,
              "arms": {}, "naive": {}}
    if module == "market":
        derived = [_market_day(r, training_majority) for r in rows]
        actual = [d["actual"] for d in derived]
        for arm in ARMS:
            predicted = [d["arms"][arm]["prediction"] for d in derived]
            result["arms"][arm] = {**classification_metrics(actual, predicted, MARKET_CLASSES),
                                    "raw_scores": raw_market_score_metrics(actual, [d["arms"][arm]["raw_scores"] for d in derived])}
        for name in ("majority", "persistence"):
            result["naive"][name] = classification_metrics(actual, [d["naive"][name] for d in derived], MARKET_CLASSES)
    else:
        derived = [_relay_day(r) for r in rows]
        confirmed = sum(d["confirmed_nonword"] for d in derived)
        unknown = sum(d["unknown_eligibility"] for d in derived)
        denominator = confirmed + unknown
        result.update({"confirmed_nonword_dates": confirmed,
                       "effective_eligible_dates": sum(d["effective_eligible"] for d in derived),
                       "known_no_nonword_dates": sum(d["known_no_nonword"] for d in derived),
                       "unknown_eligibility_dates": unknown,
                       "eligibility_knowledge_coverage": _divide(n-unknown, n),
                       "primary_coverage_denominator": denominator,
                       "complete_original_outcome_dates": sum(d["complete_original_outcomes"] for d in derived),
                       "primary_definition": "Actual signal-day OHLC non-one-price ranked Top1 next-session closing promotion; not an executable return",
                       "ranking_rule": "raw_score descending, code ascending; positive Top1 separately requires predicted_positive=True"})
        for arm in ARMS:
            primary = [d["arms"][arm]["rank_top1_outcome"] for d in derived]
            positive = [d["arms"][arm]["positive_top1_outcome"] for d in derived]
            valid = [p for p in primary if p is not None]
            called = sum(d["arms"][arm]["positive_top1_status"] == "CALLED" for d in derived)
            candidate_truth, candidate_pred, candidate_scores = [], [], []
            boards = {"two_board": ([], [], []), "three_plus": ([], [], [])}
            for d in derived:
                for code in d["codes"]:
                    truth = d["truths"][code]
                    pred = d["arms"][arm]["candidate_predictions"][code]
                    score = d["arms"][arm]["candidate_scores"][code]
                    candidate_truth.append(truth); candidate_pred.append(pred); candidate_scores.append(score)
                    board = d["facts"].get(code, (None, None))[0]
                    category = "two_board" if board == 2 else "three_plus" if board is not None and board >= 3 else None
                    if category:
                        boards[category][0].append(truth); boards[category][1].append(pred); boards[category][2].append(score)
            result["arms"][arm] = {
                "registered_dates": n, "rank_scored_dates": len(valid), "rank_successes": sum(valid),
                "primary_accuracy": _divide(sum(valid), len(valid)),
                "primary_scoring_coverage": _divide(len(valid), denominator),
                "rank_original_grid_coverage": _divide(len(valid), n),
                "positive_top1_called_dates": called,
                "positive_top1_call_coverage": _divide(called, denominator),
                "positive_top1_scored_dates": sum(p is not None for p in positive),
                "positive_top1_precision": _divide(sum(p for p in positive if p is not None), sum(p is not None for p in positive)),
                "all_candidate": binary_metrics(candidate_truth, candidate_pred),
                "raw_scores": raw_binary_score_metrics(candidate_truth, candidate_scores),
                "by_board": {k: {"all_candidate": binary_metrics(v[0], v[1]),
                                  "raw_scores": raw_binary_score_metrics(v[0], v[2])} for k, v in boards.items()},
            }
        values = [d["naive_outcome"] for d in derived]
        valid = [v for v in values if v is not None]
        result["naive"]["highest_nonword"] = {"scored_dates": len(valid), "successes": sum(valid),
                                                 "accuracy": _divide(sum(valid), len(valid)),
                                                 "scoring_coverage": _divide(len(valid), denominator)}
    return result


def summarize_records(module, records, calendar_sessions, training_majority=None) -> dict:
    """Describe all saved records without interpreting history as admission."""
    rows, calendar = _records(module, records, calendar_sessions)
    positions = {day: i for i, day in enumerate(calendar)}
    dates = calendar[positions[rows[0]["signal_date"]]:positions[rows[-1]["signal_date"]]+1] if rows else []
    return _summarize_aligned(module, _align(rows, calendar, dates), training_majority)


def _protocol(protocol: Mapping) -> None:
    expected = {"forward_sessions": 120, "min_market_pairs": 108, "min_class_support": 20,
                "min_relay_pairs": 60, "min_coverage": .9, "min_improvement": .03}
    for name, value in expected.items():
        if not np.isclose(protocol.get(name, np.nan), value, rtol=0, atol=1e-12):
            raise ValueError("protocol differs from the approved fixed gate: " + name)
    bootstrap = protocol.get("bootstrap", {})
    if (bootstrap.get("draws") != 2000 or bootstrap.get("seed") != 20261007
            or bootstrap.get("block_sessions") != 5 or bootstrap.get("tails") != [.00625, .99375]):
        raise ValueError("fixed family-four official-session bootstrap required")
    if protocol.get("preferred_arm") not in ("C1", "C2"):
        raise ValueError("fixed candidate priority required")


def _closed_node(module: str, rows: Sequence[dict], calendar: Sequence[str], protocol: Mapping) -> bool:
    if not rows:
        return False
    target = rows[-1].get("target_date")
    if target is None or target not in calendar:
        return False
    # The controller certifies the real clock. Optional explicit closure facts
    # distinguish a mature source failure from a still-future target.
    closed = protocol.get("closed_through_date")
    if closed is not None:
        return isinstance(closed, str) and target <= closed and closed in calendar
    if not rows[-1].get("_record_available") or rows[-1].get("forward") is not True:
        return False
    if rows[-1].get("target_matured") is False:
        return False
    if rows[-1].get("target_matured") is True:
        return True
    outcome = rows[-1].get("outcomes") or {}
    if module == "market":
        return _class(outcome.get("market_target")) is not None
    return any(_binary(v) is not None for v in (outcome.get("candidates") or {}).values())


def _fixed_window(module, records, protocol, calendar_sessions, size, anchor=None):
    rows, calendar = _records(module, records, calendar_sessions)
    forward = [r for r in rows if r.get("forward") is True]
    start = anchor or (forward[0]["signal_date"] if forward else None)
    if start is None:
        return [], calendar, False, "NOT_STARTED"
    if start not in calendar:
        raise ValueError("fixed window start is outside the official calendar")
    i = calendar.index(start)
    dates = calendar[i:i+size]
    aligned = _align(forward, calendar, dates)
    mature = len(dates) == size and _closed_node(module, aligned, calendar, protocol)
    return aligned, calendar, mature, "MATURE_FIXED_NODE" if mature else "NOT_MATURE"


def _comparison(module: str, rows: Sequence[dict], arm: str) -> dict:
    if module == "market":
        daily = [_market_day(r) for r in rows]
        truth = [d["actual"] for d in daily]
        base = [d["arms"]["BASE"]["prediction"] for d in daily]
        challenge = [d["arms"][arm]["prediction"] for d in daily]
        valid = [y is not None and b is not None and c is not None for y, b, c in zip(truth, base, challenge)]
        bv = [float(b == y) if ok else None for y, b, ok in zip(truth, base, valid)]
        cv = [float(c == y) if ok else None for y, c, ok in zip(truth, challenge, valid)]
        supports = {k: sum(ok and y == k for y, ok in zip(truth, valid)) for k in MARKET_CLASSES}
        naive = {name: [float(d["naive"][name] == d["actual"]) if ok and d["naive"][name] is not None else None
                        for d, ok in zip(daily, valid)] for name in ("majority", "persistence")}
        denominator = len(rows)
    else:
        daily = [_relay_day(r) for r in rows]
        bv = [d["arms"]["BASE"]["rank_top1_outcome"] for d in daily]
        cv = [d["arms"][arm]["rank_top1_outcome"] for d in daily]
        valid = [b is not None and c is not None for b, c in zip(bv, cv)]
        bv = [b if ok else None for b, ok in zip(bv, valid)]
        cv = [c if ok else None for c, ok in zip(cv, valid)]
        supports = None
        naive = {"highest_nonword": [d["naive_outcome"] if ok else None for d, ok in zip(daily, valid)]}
        denominator = sum(d["confirmed_nonword"] or d["unknown_eligibility"] for d in daily)
    differences = [float(c-b) if ok else None for b, c, ok in zip(bv, cv, valid)]
    paired = sum(valid)
    cmean = _divide(sum(v for v in cv if v is not None), paired)
    bmean = _divide(sum(v for v in bv if v is not None), paired)
    naive_comparisons = {}
    for name, values in naive.items():
        covered = sum(v is not None for v in values)
        mean = _divide(sum(v for v in values if v is not None), covered)
        naive_comparisons[name] = {"paired_dates": covered, "required_pair_dates": paired,
                                  "accuracy": mean,
                                  "candidate_not_lower": paired > 0 and covered == paired and cmean is not None and mean is not None and cmean >= mean}
    windows = []
    for start in range(0, len(rows), 40):
        chunk = differences[start:start+40]
        finite = [v for v in chunk if v is not None]
        windows.append({"start_signal_date": rows[start]["signal_date"],
                        "end_signal_date": rows[min(start+39, len(rows)-1)]["signal_date"],
                        "official_signal_dates": len(chunk), "paired_dates": len(finite),
                        "primary_difference": _divide(sum(finite), len(finite))})
    return {"arm": arm, "registered_dates": len(rows), "paired_dates": paired,
            "primary_coverage_denominator": denominator, "pair_coverage": _divide(paired, denominator),
            "baseline_primary": bmean, "challenger_primary": cmean,
            "primary_difference": cmean-bmean if paired else None,
            "paired_class_support": supports, "naive_comparisons": naive_comparisons,
            "fixed_40_session_windows": windows, "_differences": differences}


def _not_lower(new: Any, old: Any) -> bool:
    return new is not None and old is not None and new >= old - 1e-12


def acceptance(module, records, protocol, calendar_sessions) -> dict:
    """Admit only at the single mature 120-official-signal-day forward node."""
    _protocol(protocol)
    rows, calendar, mature, status = _fixed_window(module, records, protocol, calendar_sessions, 120,
                                                  protocol.get("forward_start_date", protocol.get("first_signal_date")))
    summary = _summarize_aligned(module, rows)
    gates = {}
    for arm in ("C1", "C2"):
        comparison = _comparison(module, rows, arm)
        differences = comparison.pop("_differences")
        interval = block_interval(differences) if mature else None
        comparison["family4_primary_interval"] = interval
        base, candidate = summary["arms"]["BASE"], summary["arms"][arm]
        delta = comparison["primary_difference"]
        checks = {
            "mature_fixed_120_node": mature,
            "all_available_records_are_forward": all(r.get("forward") is True for r in rows if r.get("_record_available")),
            "pair_coverage_at_least90pct": comparison["pair_coverage"] is not None and comparison["pair_coverage"] >= .9,
            "minimum_meaningful_improvement3pp": delta is not None and delta >= .03-1e-12,
            "family4_interval_lower_above0": interval is not None and interval[0] > 0,
            "each_fixed_40_session_window_not_lower": len(comparison["fixed_40_session_windows"]) == 3 and all(w["official_signal_dates"] == 40 and w["primary_difference"] is not None and w["primary_difference"] >= -1e-12 for w in comparison["fixed_40_session_windows"]),
            "all_naive_baselines_not_lower": all(v["candidate_not_lower"] for v in comparison["naive_comparisons"].values()),
        }
        if module == "market":
            checks.update({"paired_dates_at_least108": comparison["paired_dates"] >= 108,
                           "each_class_support_at_least20": all(v >= 20 for v in comparison["paired_class_support"].values()),
                           "scoring_coverage_not_lower": _not_lower(candidate["scoring_coverage"], base["scoring_coverage"]),
                           "macro_f1_not_lower": _not_lower(candidate["macro_f1"], base["macro_f1"]),
                           "balanced_accuracy_not_lower": _not_lower(candidate["balanced_accuracy"], base["balanced_accuracy"])})
            for klass in MARKET_CLASSES:
                checks[klass+"_complete_recall_not_lower"] = _not_lower(candidate["by_class"][klass]["complete_recall"], base["by_class"][klass]["complete_recall"])
        else:
            cb, bb = candidate["all_candidate"], base["all_candidate"]
            checks.update({"paired_dates_at_least60": comparison["paired_dates"] >= 60,
                           "all_candidate_accuracy_not_lower": _not_lower(cb["accuracy"], bb["accuracy"]),
                           "complete_positive_recall_not_lower": _not_lower(cb["complete_positive_recall"], bb["complete_positive_recall"]),
                           "all_candidate_scoring_coverage_not_lower": _not_lower(cb["scoring_coverage"], bb["scoring_coverage"]),
                           "raw_score_Brier_not_worse": candidate["raw_scores"]["raw_score_Brier"] is not None and base["raw_scores"]["raw_score_Brier"] is not None and candidate["raw_scores"]["raw_score_Brier"] <= base["raw_scores"]["raw_score_Brier"]+1e-12,
                           "raw_score_coverage_not_lower": _not_lower(candidate["raw_scores"]["raw_score_coverage"], base["raw_scores"]["raw_score_coverage"]),
                           "positive_top1_call_coverage_not_lower": _not_lower(candidate["positive_top1_call_coverage"], base["positive_top1_call_coverage"]),
                           "primary_scoring_coverage_not_lower": _not_lower(candidate["primary_scoring_coverage"], base["primary_scoring_coverage"])})
        gates[arm] = {"checks": checks, "passed": all(checks.values()), "comparison": comparison}
    preferred = protocol["preferred_arm"]
    priority = (preferred, "C2" if preferred == "C1" else "C1")
    selected = next((arm for arm in priority if gates[arm]["passed"]), "BASE")
    return {"module": module, "status": status, "selected_arm": selected,
            "upgrade_eligible": selected != "BASE", "gates": gates, "summary": summary,
            "official_window_size": 120, "uncertainty_family": "four predeclared primary challenges",
            "controller_real_clock_confirmation_required": True,
            "automatic_deployment_performed": False, "scores_are_calibrated_probabilities": False}


def rollback_check(module, records, protocol, calendar_sessions) -> dict:
    """Safety decision only at a fixed mature 60-session deployed window.

    The controller supplies the current nonoverlapping slice and active arm.
    Missing evidence produces a hold, never a fabricated model deterioration.
    """
    _protocol(protocol)
    rule = protocol.get("rollback", {})
    if rule != {"sessions": 60, "primary_drop": .05, "recall_drop": .10}:
        raise ValueError("fixed approved rollback rule required")
    active = protocol.get("active_arm", protocol.get("deployed_arm", protocol["preferred_arm"]))
    if active not in ARMS:
        raise ValueError("active arm is not registered")
    rows, calendar, mature, status = _fixed_window(module, records, protocol, calendar_sessions, 60,
                                                  protocol.get("rollback_window_start"))
    summary = _summarize_aligned(module, rows)
    if active == "BASE":
        return {"module": module, "status": "BASE_ACTIVE_NOT_APPLICABLE", "rollback": False,
                "selected_arm": "BASE", "summary": summary, "automatic_deployment_performed": False}
    comparison = _comparison(module, rows, active)
    comparison.pop("_differences")
    base, candidate = summary["arms"]["BASE"], summary["arms"][active]
    if module == "market":
        recall_differences = {}
        for klass in MARKET_CLASSES:
            old = base["by_class"][klass]["complete_recall"]
            new = candidate["by_class"][klass]["complete_recall"]
            recall_differences[klass] = new-old if new is not None and old is not None else None
        # Retain the original WORSEN scalar for readers of older diagnostics.
        recall_difference = recall_differences["WORSEN"]
    else:
        old_recall = base["all_candidate"]["complete_positive_recall"]
        new_recall = candidate["all_candidate"]["complete_positive_recall"]
        recall_difference = new_recall-old_recall if new_recall is not None and old_recall is not None else None
        recall_differences = {"positive": recall_difference}
    known_recall_differences = [d for d in recall_differences.values() if d is not None]
    delta = comparison["primary_difference"]
    # The fixed 60-session safety node has no extra admission-size/coverage
    # threshold. Known paired deterioration must not be bypassed by scarcity.
    enough = mature and comparison["paired_dates"] > 0
    triggers = {"primary_drop_at_least5pp": enough and delta is not None and delta <= -.05+1e-12,
                "recall_drop_at_least10pp": enough and any(d <= -.10+1e-12 for d in known_recall_differences)}
    revert = any(triggers.values())
    return {"module": module, "status": "MATURE_FIXED_NODE" if enough else status if not mature else "INSUFFICIENT_EVIDENCE",
            "rollback": revert, "selected_arm": "BASE" if revert else active,
            "active_arm": active, "hold_due_to_missing_evidence": not enough,
            "checks": {"mature_fixed_60_node": mature, "sufficient_pair_evidence": enough},
            "triggers": triggers, "recall_difference": recall_difference,
            "recall_differences": recall_differences,
            "recall_rollback_scope": "any registered market class complete Recall" if module == "market" else "all-candidate complete positive Recall",
            "comparison": comparison, "summary": summary,
            "controller_real_clock_confirmation_required": True, "automatic_deployment_performed": False}
