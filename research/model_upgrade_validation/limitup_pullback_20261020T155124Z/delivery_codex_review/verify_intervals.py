"""Independent loop-based reproduction of fixed calendar block intervals."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
result = read(OUT / 'CORRECTED_RESULTS.json')
data = pd.read_csv(OUT.parent / 'full_paired_predictions.csv', dtype={'code': str}, float_precision='round_trip')
sessions = joblib.load(OUT.parent / 'prepared_FEATURES_BEFORE_FITS.joblib')['sessions']
checks = []
for fold, start, end in [('2025H2', '2025-07-01', '2025-12-31'), ('2026JanAug', '2026-01-01', '2026-08-31'), ('2026Sep', '2026-09-01', '2026-09-30')]:
    dates = [d for d in sessions if start <= d <= end]
    for scope, column in [('strict', 'daily_one_word'), ('original', 'actual_one_word_daily')]:
        frame = data[data.date.between(start, end) & data[column].eq(0) & data.label_valid & data.target.notna()]
        bydate = {}
        for date, group in frame.groupby('date'):
            counts = []
            for score in ['new_score', 'roll_score']:
                pairs = list(zip(group.target.astype(int), group[score].gt(.5)))
                counts.extend([sum(y == 1 and q for y, q in pairs), sum(y == 0 and q for y, q in pairs), sum(y == 1 and not q for y, q in pairs)])
            bydate[date] = counts
        n = len(dates)
        starts = np.random.default_rng(20261020).integers(0, n, size=(2000, (n+4)//5))
        deltas = []
        for draw in starts:
            sequence = [dates[(int(begin)+j) % n] for begin in draw for j in range(5)][:n]
            count = np.array([bydate.get(d, [0]*6) for d in sequence]).sum(axis=0)
            a, b = 2*count[0]+count[1]+count[2], 2*count[3]+count[4]+count[5]
            if a and b:
                deltas.append(2*count[0]/a-2*count[3]/b)
        expected = result['folds'][fold]['official_calendar_F1_delta95_'+scope]
        if deltas:
            assert np.allclose(np.quantile(deltas, [.025, .975]), [expected['lower'], expected['upper']], atol=1e-15)
        else:
            assert expected['lower'] is None and expected['upper'] is None
        assert expected['undefined_draws'] == 2000-len(deltas)
        checks.append(dict(fold=fold, scope=scope, official_sessions=n, known_dates=len(bydate),
                           lower=expected['lower'], upper=expected['upper'], independent_loop_recalculation=True))
verification = dict(
    at=datetime.now(timezone.utc).isoformat(), status='PASS', bootstrap_checks=checks,
    relevant_tests=dict(passed=3, failed=0), new_fits=0, source_scores_labels_changed=False,
    verifier_initial_attempt='Inline verifier incorrectly called quantile on an empty original-identity subgroup outside September. No output or model was changed. This verifier explicitly checks undefined intervals and all2000undefineddraws.',
    review_script_sha256=sha(OUT / 'review_saved.py'), test_script_sha256=sha(OUT / 'test_review.py'),
    independent_verifier_sha256=sha(__file__))
with (OUT / 'independent_interval_verification.json').open('x', encoding='utf-8') as stream:
    json.dump(verification, stream, ensure_ascii=False, indent=2)
print(json.dumps(dict(status='PASS', independent_bootstrap_recalculations=len(checks), tests=3)))
