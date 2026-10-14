"""Evidence-only changes in a supplied, explicitly scoped competitor universe."""
import math
from leader_holding_review_v21 import stamp


def validate(snapshot, asof):
    cutoff = stamp(asof)
    observed, available = stamp(snapshot["observed_at"]), stamp(snapshot["available_at"])
    if observed > available or available > cutoff:
        raise ValueError("future or inconsistent competitor snapshot")
    if not snapshot.get("scope_id"):
        raise ValueError("explicit competitor scope required")
    codes = [r["code"] for r in snapshot["candidates"]]
    if len(codes) != len(set(codes)):
        raise ValueError("duplicate competitor")
    for row in snapshot["candidates"]:
        height = float(row["boards"])
        if not math.isfinite(height) or height < 0 or not height.is_integer():
            raise ValueError("nonnegative integer board height required")
        if row["seal_state"] not in {"sealed", "open", "unknown"}:
            raise ValueError("invalid seal state")
    return (snapshot.get("data_status") == "verified" and bool(snapshot.get("source"))
            and snapshot.get("universe_complete") is True
            and observed.date() == cutoff.date() and (cutoff-observed).total_seconds() <= 300)


def compare(code, current, asof, previous=None):
    fresh = validate(current, asof)
    result = dict(status="DIAGNOSTIC_ONLY", code=code, scope_id=current["scope_id"],
                  asof=stamp(asof).isoformat(), leader_identity_override=None,
                  target_probability=None, events=[], automatic_trade_action=False)
    if not fresh:
        return dict(result, evidence_state="DATA_LIMITED")
    now = {r["code"]: r for r in current["candidates"]}
    if code not in now:
        return dict(result, evidence_state="SUBJECT_NOT_IN_SCOPE")
    subject = now[code]
    peers = [r for r in now.values() if r["code"] != code]
    higher = sorted(r["code"] for r in peers if float(r["boards"]) > float(subject["boards"]))
    same = sorted(r["code"] for r in peers if float(r["boards"]) == float(subject["boards"]))
    result.update(evidence_state="CURRENT_ONLY", higher_board_peers=higher, same_board_peers=same,
                  subject_seal_state=subject["seal_state"])
    if previous is None:
        return result
    if stamp(previous["available_at"]) > stamp(asof):
        raise ValueError("previous snapshot was not available at current decision")
    if stamp(previous["observed_at"]) >= stamp(current["observed_at"]):
        raise ValueError("previous observation must precede current")
    if previous["scope_id"] != current["scope_id"]:
        return dict(result, evidence_state="SCOPE_CHANGED_NOT_COMPARABLE")
    if not validate(previous, previous["available_at"]):
        return dict(result, evidence_state="PREVIOUS_DATA_LIMITED")
    old = {r["code"]: r for r in previous["candidates"]}
    if code not in old:
        return dict(result, evidence_state="PREVIOUS_SUBJECT_MISSING")
    prior = old[code]
    events = result["events"]
    if prior["seal_state"] == "sealed" and subject["seal_state"] == "open":
        events.append(dict(type="SUBJECT_OPENED", interpretation="recheck_absorption_not_automatic_identity_loss"))
    if prior["seal_state"] == "open" and subject["seal_state"] == "sealed":
        events.append(dict(type="SUBJECT_RESEALED", interpretation="seal_observed_not_fill_or_next_day_guarantee"))
    old_higher = {c for c, r in old.items() if c != code and float(r["boards"]) > float(prior["boards"])}
    for peer in higher:
        if peer not in old_higher:
            events.append(dict(type="NEW_HIGHER_BOARD_PEER", code=peer))
    for peer in sorted(set(now) & set(old) - {code}):
        if old[peer]["seal_state"] == "sealed" and now[peer]["seal_state"] == "open":
            events.append(dict(type="PEER_OPENED", code=peer,
                               interpretation="could_be_rotation_or_sector_weakness_requires_theme_evidence"))
    for peer in sorted(set(old)-set(now)-{code}):
        events.append(dict(type="PEER_LEFT_SCOPE", code=peer, interpretation="not_proof_of_failure"))
    result["evidence_state"] = "COMPARABLE"
    return result
