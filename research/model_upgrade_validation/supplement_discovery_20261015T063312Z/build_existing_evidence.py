"""Bind only existing exact-date pools; no network requests or training."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
import stock_judgement_review as o

out = Path(__file__).parent
chosen = [
    ("2026-09-30", "public_outcome_sources_20261002_v1"),
    ("2026-10-08", "public_outcome_sources_20261008T092005Z"),
    ("2026-10-09", "public_outcome_sources_20261009T091716Z"),
    ("2026-10-12", "independent_outcome_20261012T073313Z"),
    ("2026-10-13", "independent_outcome_20261013T073137Z"),
]
bank_path = Path("research/model_upgrade_validation/supplement_recognition_20261014T174129Z/qualified_relation_bank_v2.json")
bindings = o.bind(bank_path)
items, excluded = [], []
for day, folder in chosen:
    root = Path("data/v293") / folder
    manifest = root / "manifest.json"
    raw = root / "raw" / (day + "_limit_up.json")
    bindings |= o.bind(manifest) | o.bind(raw)
    items.append(dict(date=day, path=str(raw.resolve()), sha256=o.sha(raw),
        manifest=str(manifest.resolve()), kind="EASTMONEY_CLOSING_POOL"))
    for request in o.read(manifest)["requests"]:
        if request.get("kind") != "limit_up" or not request.get("raw_path"):
            continue
        p = Path(request["raw_path"])
        if p.is_file() and str(o.read(p).get("data", {}).get("qdate")) != request["date"].replace("-", ""):
            excluded.append(dict(requested_date=request["date"], response_qdate=o.read(p).get("data", {}).get("qdate"),
                path=str(p), sha256=o.sha(p), decision="EXCLUDED_DATE_MISMATCH"))
alt = Path("results/daily_review_20261014T1118Z/alternate_full_pool_facts_ladder_corrected.json")
bindings |= o.bind(alt) | o.bind(o.read(alt)["source_raw_path"]) | o.bind(o.read(bank_path)["calendar_path"])
items.append(dict(date="2026-10-14", path=str(alt.resolve()), sha256=o.sha(alt), kind="ALTERNATE_FULL_POOL"))
spec = dict(version="supplement-discovery-evidence-1", formed_at=o.now().isoformat(),
    signal_date="2026-10-14", target_date="2026-10-15", relation_bank_path=str(bank_path.resolve()),
    pool_snapshots=items, source_bindings=bindings,
    scope="CURRENT_KNOWLEDGE_RETROSPECTIVE_IDENTITY_DIAGNOSTIC_ALREADY_MODEL_EXPOSED",
    historical_asof_training_authorized=False, new_requests=0)
o.write_new(out / "existing_pool_evidence.json", spec)
o.write_new(out / "excluded_date_mismatch.json", dict(rows=excluded, repaired_or_backfilled=False))
print(dict(snapshots=len(items), excluded_date_mismatches=len(excluded)))
