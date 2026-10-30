"""Meaningful clock/source failure tests, with no external requests."""
from datetime import datetime,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import collector

class ContractTests(unittest.TestCase):
    def test_calendar_windows(self):
        cases=[('2026-10-25T03:00:00+08:00','NON_TRADING_DAY','2026-10-23',False),
               ('2026-10-26T08:50:00+08:00','BEFORE_OPEN','2026-10-23',False),
               ('2026-10-26T09:25:00+08:00','TRADING_WINDOW','2026-10-26',True),
               ('2026-10-26T12:00:00+08:00','SESSION_BREAK','2026-10-26',False),
               ('2026-10-26T15:11:00+08:00','AFTER_CLOSE','2026-10-26',False)]
        for stamp,state,day,active in cases:
            with self.subTest(stamp=stamp):
                c=collector.session_context(datetime.fromisoformat(stamp))
                self.assertEqual((c['state'],c['target_day'],c['active']),(state,day,active))
    def test_calendar_outside_range(self):
        self.assertIsNone(collector.session_context(datetime.fromisoformat('2027-01-04T10:00:00+08:00'))['target_day'])
    def quote(self,date='2026-10-23',time='15:00:00'):
        fields=['测试','10','10','11','11','10','11','11','100','1100']+['0']*20+[date,time,'00']
        return ('var hq_str_sh603089="'+','.join(fields)+'";').encode('gb18030')
    def test_old_quotes_not_today(self):
        rows=collector.parse_quotes(self.quote(),'2026-10-26',datetime.fromisoformat('2026-10-26T10:00:00+08:00'))
        self.assertEqual(rows[0]['status'],'STALE_OTHER_SESSION');self.assertFalse(rows[0]['usable_for_current_session'])
    def test_weekend_quote_not_live(self):
        rows=collector.parse_quotes(self.quote(),'2026-10-23',datetime.fromisoformat('2026-10-25T03:00:00+08:00'))
        self.assertEqual(rows[0]['status'],'EXACT_SESSION_CLOSED_QUOTE_NOT_LIVE')
    def test_future_quote_rejected(self):
        rows=collector.parse_quotes(self.quote('2026-10-26','10:03:00'),'2026-10-26',datetime.fromisoformat('2026-10-26T10:00:00+08:00'))
        self.assertEqual(rows[0]['status'],'INVALID_FUTURE_TIMESTAMP')
    def test_previous_day_packet_not_reused(self):
        c=collector.Collector();c.last={'requested_day':'2026-10-23','status':'SUCCESS_PUBLIC_DATA_ONLY','pools':[{'code':'603089'}],'quotes':[{}]}
        with patch.object(collector,'session_context',return_value={'target_day':'2026-10-26'}):s=c.status()
        self.assertEqual(s['data']['status'],'CURRENT_SESSION_NOT_CAPTURED');self.assertEqual(s['data']['pools'],[])
    def test_network_failure_no_fake_empty_success(self):
        with TemporaryDirectory() as d,patch.object(collector.requests,'get',side_effect=TimeoutError('test timeout')):
            receipt,rows=collector.capture_pool('limit_up','2026-10-26',Path(d))
        self.assertEqual(receipt['status'],'UNAVAILABLE_OR_DATE_REJECTED');self.assertEqual(rows,[])
    def test_wrong_pool_date_rejected(self):
        response=unittest.mock.Mock();response.content=b'{"data":{"qdate":20261023,"tc":0,"pool":[]}}';response.status_code=200;response.json.return_value={'data':{'qdate':20261023,'tc':0,'pool':[]}}
        with TemporaryDirectory() as d,patch.object(collector.requests,'get',return_value=response):r,rows=collector.capture_pool('limit_up','2026-10-26',Path(d))
        self.assertEqual(r['status'],'UNAVAILABLE_OR_DATE_REJECTED');self.assertEqual(rows,[])

if __name__=='__main__':unittest.main()
