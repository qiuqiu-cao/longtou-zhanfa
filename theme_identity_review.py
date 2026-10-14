"""Candidate-specific theme membership evidence, not causal leader identity."""
import pandas as pd

from leader_holding_review_v21 import stamp


def review(registry, code, signal_date, asof, mode="forward"):
    if mode not in {"forward", "historical_reconstruction"}:
        raise ValueError("invalid evidence mode")
    cutoff = stamp(asof)
    date = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    if date > cutoff.strftime("%Y-%m-%d"):
        raise ValueError("future signal date")
    code = str(code).zfill(6)
    usable, exclusions = [], []
    required = {"signal_date", "code", "theme_group_id", "theme", "theme_granularity",
                "source_ref", "source_published_at", "timestamp_precision", "collected_at",
                "evidence_status", "evidence_kind", "membership_basis"}
    if not required.issubset(registry):
        raise ValueError("theme evidence registry missing required columns")
    # Legacy registry times are explicitly Shanghai-local; new output is zoned.
    def local(value):
        value = pd.Timestamp(value)
        if pd.isna(value):
            raise ValueError("missing evidence time")
        return value.tz_localize("Asia/Shanghai") if value.tzinfo is None else value.tz_convert("Asia/Shanghai")
    for row in registry[registry.signal_date.astype(str).eq(date)].to_dict("records"):
        member = str(row["code"]).zfill(6)
        reason = None
        try:
            public = local(row["source_published_at"])
            precision = row["timestamp_precision"]
            if precision == "date":
                public = public.normalize()+pd.Timedelta(hours=23,minutes=59,seconds=59)
            elif precision != "exact":
                raise ValueError("unknown publication time precision")
            captured = local(row["collected_at"])
            if captured < public:
                raise ValueError("capture before conservative public availability")
            known = max(public,captured) if mode == "forward" else public
            if known > cutoff:
                reason = "NOT_KNOWN_AT_DECISION"
        except (ValueError,TypeError):
            reason = "INVALID_TIME_EVIDENCE"
        if reason is None:
            if row["evidence_status"] != "verified" or not isinstance(row["source_ref"],str) or not row["source_ref"].strip():
                reason = "UNVERIFIED_SOURCE"
            elif row["evidence_kind"] != "MEMBERSHIP" or row["membership_basis"] != "verified_concept_membership":
                reason = "NOT_MEMBERSHIP_EVIDENCE"
            elif row["theme_granularity"] != "细分题材":
                reason = "BROAD_OR_UNSPECIFIED_THEME"
            elif not isinstance(row["theme_group_id"],str) or not row["theme_group_id"].strip():
                reason = "MISSING_GROUP_ID"
        if reason:
            exclusions.append(dict(code=member, reason=reason, group_id=row.get("theme_group_id")))
        else:
            usable.append(dict(code=member, group_id=row["theme_group_id"], theme=row["theme"],
                               source=row["source_ref"], public_at=public.isoformat(),
                               captured_at=captured.isoformat(), known_at=known.isoformat()))
    groups = []
    own = {r["group_id"] for r in usable if r["code"] == code}
    for group_id in sorted(own):
        members = [r for r in usable if r["group_id"] == group_id]
        names = {r["theme"] for r in members}
        if len(names) != 1:
            raise ValueError("conflicting theme names for group id")
        peers = sorted({r["code"] for r in members if r["code"] != code})
        groups.append(dict(group_id=group_id, theme=next(iter(names)), observed_peers=peers,
                           observed_peer_count=len(peers), roster_complete=False, evidence=members,
                           leader_role="UNVERIFIED", current_support_rate=None))
    return dict(status="DIAGNOSTIC_ONLY", code=code, signal_date=date, asof=cutoff.isoformat(), mode=mode,
                evidence_state="OBSERVED_MEMBERSHIP_ONLY" if groups else "UNAVAILABLE_NOT_PROVEN_ABSENT",
                groups=groups, excluded_candidate_evidence=[r for r in exclusions if r["code"]==code],
                confirmed_price_driver=None, leader_probability=None,
                limitations=["declared narrow theme and verified membership still require source review",
                             "observed peers are not a complete roster or proof of synchronized support",
                             "membership does not establish pioneer, core, leader, current catalyst or buy point",
                             "historical reconstruction cannot be used as forward-captured evidence",
                             "no membership evidence does not prove isolated trading or justify automatic rejection"])
