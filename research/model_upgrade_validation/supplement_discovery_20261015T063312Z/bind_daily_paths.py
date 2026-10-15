"""Add saved, raw-bound unadjusted price paths to existing pool evidence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import stock_judgement_review as o

out=Path(__file__).parent
spec=o.read(out/'existing_pool_evidence.json')
normalized=Path('results/daily_review_20261014T1118Z/focused_daily/normalized_daily.json')
facts=o.read('research/model_upgrade_validation/forum_reasoning_20261014/actual_full15/facts_with_forum_reasoning.json')
codes=sorted(r['code'] for r in facts['rows'])
manifests=[Path('results/daily_review_20261014T1118Z/focused_daily/manifest.json'),
           Path('data/v293/outcome_review_20261014T1116Z/manifest.json')]
spec['source_bindings']|=o.bind(normalized)
for p in manifests:spec['source_bindings']|=o.bind(p)
daily=o.read(normalized)
for code in codes:spec['source_bindings']|=o.bind(daily[code]['raw_path'])
spec.update(formed_at=o.now().isoformat(),daily_archive=dict(normalized_path=str(normalized.resolve()),
    manifest_paths=[str(p.resolve()) for p in manifests],codes=codes))
o.write_new(out/'existing_pool_and_daily_evidence.json',spec)
print(dict(saved_daily_candidates=len(codes),new_requests=0))
