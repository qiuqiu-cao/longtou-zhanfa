"""Independent audit of the peer's original zero-fit report; never trains."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
p = ROOT / 'event_group_features_9245.csv'
d = pd.read_csv(p, dtype={'code': str})
assert len(d) == 9245 and not d.duplicated(['date', 'code']).any()
groups = []
for matched, part in d.groupby('event_cohort_matched'):
    groups.append(dict(matched=int(matched), rows=len(part), known=int(part.target.notna().sum()),
                       unknown=int(part.target.isna().sum()), positives=int(part.target.eq(1).sum()),
                       known_outcome_rate=float(part.target.mean())))
receipt = dict(at_utc=datetime.now(timezone.utc).isoformat(), fits=0,
               source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
               rows=len(d), known=int(d.target.notna().sum()), unknown=int(d.target.isna().sum()),
               positives=int(d.target.eq(1).sum()), groups=groups,
               correction='427/948 matched known and 3276/8230 unmatched known; neither total951/8294 nor fabricated428/3302 are the observed-label denominators/numerators.',
               interpretation='Descriptive association in already seen history, not model improvement, real theme verification, or causal capital recognition.')
with (OUT / 'INITIAL_NUMBERS_INDEPENDENT_AUDIT.json').open('x', encoding='utf-8') as f:
    json.dump(receipt, f, ensure_ascii=False, indent=2)
print(json.dumps(receipt, ensure_ascii=True))
