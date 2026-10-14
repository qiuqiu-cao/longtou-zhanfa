import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
import archive_august_minutes_v289 as engine


class AugustReferenceTests(unittest.TestCase):
    def payload(self,rows):return {'data':{'sz000678':{'day':rows}}}

    def test_adjusted_only_day_bars_are_not_substituted(self):
        with self.assertRaisesRegex(ValueError,'unadjusted day list'):
            engine.daily_parse({'data':{'sz000678':{'qfqday':[['2026-08-03',10,10,10,10,100]]}}},'sz000678')

    def test_raw_duplicate_or_unordered_dates_rejected_before_filter(self):
        row=['2026-07-30',10,10,10,10,100]
        with self.assertRaisesRegex(ValueError,'duplicate or unordered'):
            engine.daily_parse(self.payload([row,row,['2026-08-03',10,10,10,10,100]]),'sz000678')
        with self.assertRaisesRegex(ValueError,'duplicate or unordered'):
            engine.daily_parse(self.payload([['2026-08-04',10,10,10,10,100],['2026-08-03',10,10,10,10,100]]),'sz000678')

    def test_nonfinite_and_invalid_bounds_rejected(self):
        for field,value in [(2,float('nan')),(3,9),(5,-1)]:
            row=['2026-08-03',10,10,10,10,100];row[field]=value
            with self.assertRaisesRegex(ValueError,'invalid or unavailable'):
                engine.daily_parse(self.payload([row]),'sz000678')

    def test_reference_range_retains_exact_boundary_and_native_volume(self):
        rows=[['2026-07-31',10,10,10,10,100],['2026-08-03',10,10,10,10,123.45],
              ['2026-09-01',11,11,11,11,200],['2026-09-02',12,12,12,12,300]]
        frame=engine.daily_parse(self.payload(rows),'sz000678')
        self.assertEqual(frame.date.tolist(),['2026-08-03','2026-09-01'])
        self.assertEqual(frame.volume.tolist(),[123.45,200.])

    def test_missing_minute_response_preserves_independent_daily_evidence(self):
        payload=self.payload([['2026-08-03',10,10,10,10,100]])
        class Response:
            def __init__(self,text):
                self.text=text;self.content=text.encode();self.status_code=200;self.url='https://example.test'
            def raise_for_status(self):pass
            def json(self):return json.loads(self.text)
        with tempfile.TemporaryDirectory(prefix='august_minute_source_') as temporary:
            target=Path(temporary)
            for name in ['minute_raw','daily_raw']:(target/name).mkdir()
            with patch.object(engine.requests,'get',side_effect=[Response('var _data=(null);'),Response(json.dumps(payload))]):
                audits,frames=engine.fetch('000678',target)
            self.assertEqual([r['status'] for r in audits],['UNAVAILABLE','RAW_PARSED'])
            self.assertEqual(set(frames),{'daily'})
            self.assertTrue((target/'minute_raw/000678.response').exists())


if __name__=='__main__':unittest.main()
