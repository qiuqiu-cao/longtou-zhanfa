"""Complete fifteen-hypothesis template; all other V292 fields stay explicit."""
import argparse
import json
from pathlib import Path

from daily_research_template_v292 import draft_extension as previous_draft
from hypothesis_review_v298 import VERSION, pending_hypotheses


def draft_extension(plan):
    result = previous_draft(plan)
    result["hypothesis_policy"] = VERSION
    result["hypotheses"] = pending_hypotheses()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.plan)
    if "expectation_registry" in source.parts:
        raise ValueError("do not add new daily fields to an existing registry original")
    plan = json.loads(source.read_text(encoding="utf-8"))
    if "daily_research" in plan or "hypothesis_binding" in plan:
        raise ValueError("do not overwrite existing daily research")
    plan["daily_research"] = draft_extension(plan)
    from cycle_regime_diagnostics import unknown_context
    if "cycle_context" not in plan:
        plan["cycle_context"] = unknown_context(plan["signal_date"])
    payload = json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    print(json.dumps({"status": "DRAFT_NOT_REGISTERED", "hypotheses": 15, "output": str(output.resolve())}, ensure_ascii=True))


if __name__ == "__main__":
    main()
