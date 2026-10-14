"""An independently archived human hold keeps production BASE while research runs.

This policy has no model fitting or automatic release path. Its implementation
overlay preserves the two earlier correction receipts and their source copies.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HOLD_NAME = "production_hold_20261007.json"
LAYER_NAME = "implementation_policy_HOLD.json"
HOLD_VERSION = "model-upgrade-user-production-hold-1"
LAYER_VERSION = "model-upgrade-policy-hold-implementation-1"


class ProductionHoldError(ValueError):
    """A human production hold is not a data/model failure or research stop."""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stamp(value):
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("policy timestamp must include timezone")
    return parsed


def load_hold(protocol_path, current=None):
    protocol_path = Path(protocol_path).resolve()
    path = protocol_path.parent / HOLD_NAME
    layer = protocol_path.parent / LAYER_NAME
    if not path.exists() and not layer.exists():
        return None
    hold = _read(path)
    clock = _stamp(current or datetime.now(timezone.utc))
    if (hold.get("version") != HOLD_VERSION or hold.get("status") != "HOLD_ORIGINAL_BASE"
            or Path(hold["protocol_path"]).resolve() != protocol_path
            or hold.get("protocol_sha256") != _sha(protocol_path)
            or _stamp(hold["formed_at"]) > clock
            or set(hold.get("modules", [])) != {"market", "relay"}
            or len(hold["modules"]) != 2 or not hold.get("user_instruction")
            or type(hold.get("numerical_refits")) is not int or hold["numerical_refits"] != 0
            or hold.get("feature_or_parameter_changed") is not False):
        raise ValueError("human production hold binding/status/time conflict")
    for key in ("allow_promote", "allow_apply", "allow_upgraded_freeze"):
        if hold.get(key) is not False:
            raise ValueError("human production hold cannot silently release: " + key)
    for key in ("allow_shadow_prediction", "allow_shadow_settlement", "allow_fixed_review"):
        if hold.get(key) is not True:
            raise ValueError("additive-only hold must preserve research: " + key)
    return {**hold, "path": str(path), "sha256": _sha(path)}


def require_allowed(protocol_path, module, action, current=None):
    hold = load_hold(protocol_path, current)
    if hold is None:
        raise ProductionHoldError("NO_EXPLICIT_NEW_REPLACEMENT_AUTHORIZATION: production remains BASE")
    if module not in hold["modules"] or action not in {"promote", "apply", "upgraded_freeze"}:
        raise ValueError("unknown module/production action under human hold")
    raise ProductionHoldError("USER_PRODUCTION_HOLD: original BASE remains; " + action + " is forbidden for " + module)


def verify_implementation_layer(protocol_path, parent_path, prior_sources, current=None):
    protocol_path, parent_path = Path(protocol_path).resolve(), Path(parent_path).resolve()
    layer_path = protocol_path.parent / LAYER_NAME
    hold = load_hold(protocol_path, current)
    if not layer_path.exists():
        if hold is not None:
            raise ValueError("human hold implementation source receipt not yet archived")
        return None
    layer = _read(layer_path)
    expected_parent = protocol_path.parent / "implementation_correction_BEFORE_FORWARD.json"
    if (not expected_parent.is_file() or parent_path != expected_parent.resolve()
            or layer.get("version") != LAYER_VERSION
            or Path(layer["parent_path"]).resolve() != parent_path
            or layer.get("parent_sha256") != _sha(parent_path)
            or layer.get("protocol_sha256") != _sha(protocol_path)
            or Path(layer["hold_path"]).resolve() != Path(hold["path"])
            or layer.get("hold_sha256") != hold["sha256"]):
        raise ValueError("policy implementation original chain/hold hash conflict")
    formed, clock = _stamp(layer["formed_at"]), _stamp(current or datetime.now(timezone.utc))
    if formed < max(_stamp(_read(parent_path)["formed_at"]), _stamp(hold["formed_at"])) or formed > clock:
        raise ValueError("policy implementation outside actual formation boundary")
    if (type(layer.get("numerical_refits")) is not int or layer["numerical_refits"] != 0
            or layer.get("feature_or_parameter_changed") is not False):
        raise ValueError("policy overlay cannot refit or change model formulas/parameters")
    shadow_path = str(Path(__file__).with_name("model_upgrade_shadow.py").resolve())
    policy_path = str(Path(__file__).resolve())
    sources = layer.get("sources", {})
    if not isinstance(sources, dict) or set(sources) != set(prior_sources) | {policy_path}:
        raise ValueError("policy implementation must preserve full six-file group plus policy")
    for source, expected in sources.items():
        if source not in (shadow_path, policy_path) and expected != prior_sources[source]:
            raise ValueError("policy overlay cannot change other model/research implementation")
        if not Path(source).is_absolute() or _sha(source) != expected:
            raise ValueError("policy implementation current source hash conflict: " + source)
    preimage = Path(layer["preimage_path"]).resolve()
    try:
        preimage.relative_to(protocol_path.parent)
    except ValueError as exc:
        raise ValueError("policy original shadow snapshot must remain in its study directory") from exc
    if layer.get("preimage_sha256") != prior_sources[shadow_path] or _sha(preimage) != prior_sources[shadow_path]:
        raise ValueError("policy original shadow preimage hash conflict")
    return {"path": str(layer_path), "sha256": _sha(layer_path), "hold": hold, "sources": sources}
