import json
import unittest
from pathlib import Path
from archive_long_interval_v290 import ROOT,FIELDS,sha
from long_interval_layers_v290 import load


class ActualLayeredArchiveTests(unittest.TestCase):
    def test_later_recovery_preserves_full_scope_and_parent_forecast(self):
        archive=ROOT/'data/v290/full_long_interval_20261002_v1'
        audit=ROOT/'results/v290/long_interval_audit_20261002_v1/report.json'
        recovery=ROOT/'data/v290/long_interval_recovery_20261002_v1'
        registry=ROOT/'data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json'
        before=(sha(archive/'report.json'),sha(registry))
        scope,checks,bars,days,_=load(archive,audit,recovery)
        parent=json.loads((archive/'report.json').read_text(encoding='utf-8'))
        self.assertEqual(len(scope),parent['candidate_rows'])
        self.assertEqual(set(scope.columns),set(FIELDS))
        self.assertEqual(len(checks),len(scope))
        self.assertEqual(set((r['date'],r['code']) for r in checks),set(map(tuple,scope[['date','code']].to_numpy())))
        self.assertTrue(all(r['price_reconciled'] for r in checks))
        self.assertTrue(all(r['interval_minutes']==30 for r in checks))
        self.assertFalse(any(r['eligible_for_frozen_forward'] or r['actual_fill_verified'] for r in checks))
        self.assertIsNone(bars);self.assertIsNone(days)
        self.assertEqual((sha(archive/'report.json'),sha(registry)),before)


if __name__=='__main__':unittest.main()
