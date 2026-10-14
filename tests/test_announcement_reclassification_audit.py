import unittest
import pandas as pd
from announcement_reclassification_audit import compare


class AuditTests(unittest.TestCase):
    def row(self, **kw):
        base=dict(code='600001', title='收到监管警示的决定', announcement_category='',
                  published_at='2026-08-01 23:59:59',source_timestamp='2026-09-07 17:30:00',
                  risk_type='其他公告',risk_hard=0,url='source')
        base.update(kw)
        return base

    def test_newly_recognized_old_risk_is_not_active(self):
        changes, missing=compare(pd.DataFrame([self.row()]),'2026-09-07','2026-09-07 17:33')
        self.assertEqual(changes[0]['new_label'],'historical_notice_expired')
        self.assertEqual(changes[0]['new_hard'],0)
        self.assertFalse(missing)

    def test_date_only_same_day_is_not_known(self):
        changes, missing=compare(pd.DataFrame([self.row(published_at='2026-09-07 23:59:59')]),
                                '2026-09-07','2026-09-07 17:33')
        self.assertFalse(changes[0]['known_at_cutoff'])
        self.assertEqual(missing[0]['reason'],'NOT_KNOWN_AT_CUTOFF')

    def test_later_receipt_cannot_backfill_cutoff(self):
        changes, missing=compare(pd.DataFrame([self.row(published_at='2026-09-04 23:59:59',
                                source_timestamp='2026-09-08 09:00')]),'2026-09-07','2026-09-07 17:33')
        self.assertFalse(changes[0]['known_at_cutoff'])
        self.assertEqual(len(missing),1)

    def test_invalid_time_reported_not_zero_risk(self):
        changes, missing=compare(pd.DataFrame([self.row(published_at='')]),'2026-09-07','2026-09-07 17:33')
        self.assertEqual(changes,[])
        self.assertEqual(missing[0]['reason'],'INVALID_TIMESTAMP')
