"""Bind unchanged forward gates to the independent H1/H2 protocol namespace."""
from pathlib import Path
from types import FunctionType, SimpleNamespace

import relay_context_research as common
import relay_context_shadow as original
import relay_expert_research as research


def runtime(protocol_path):
    protocol = research.load_protocol(protocol_path)
    facade = SimpleNamespace(**{k: v for k, v in vars(common).items() if not k.startswith("__")})
    facade.VERSION = research.VERSION
    facade.ARMS = {"H1": common.RELAY_FIELDS, "H2": common.RELAY_FIELDS}
    facade.ENV, facade.INTERACTIONS = [], []
    facade.load_protocol = research.load_protocol
    # All inputs already rebuild from the original 29-field, source-bound archive.
    # H1/H2 introduce no context features and never read target outcomes here.
    facade.context_features = lambda relay, market, sessions: relay.copy()

    def approved(filename):
        p = research.load_protocol(filename)
        if Path(filename).resolve() != Path(protocol_path).resolve():
            raise ValueError("architecture namespace cannot migrate to another protocol")
        root = Path(filename).parent
        result = common.read(root / "report.json")
        if result["protocol_sha256"] != common.sha(filename):
            raise ValueError("architecture historical protocol binding changed")
        common.verify({str(root / "full_candidate_comparison.csv"): result["predictions_sha256"]})
        survivors = [arm for arm in ("H1", "H2") if all(c[arm]["historical_screen_passed"] for c in result["comparisons"].values())]
        if survivors != result["survivors"] or not survivors:
            raise ValueError("no fixed historical architecture survivor; shadow disabled")
        return p, result, survivors

    namespace = dict(vars(original))
    namespace["r"] = facade
    # Clone functions to bind the isolated protocol and arm identifiers. Original
    # E1/E2 module, experiment files, acceptance policy and globals stay intact.
    for name, value in vars(original).items():
        if isinstance(value, FunctionType) and value.__module__ == original.__name__:
            namespace[name] = FunctionType(value.__code__, namespace, value.__name__, value.__defaults__, value.__closure__)
    namespace["approved_research"] = approved
    return SimpleNamespace(**{k: v for k, v in namespace.items() if not k.startswith("__")})
