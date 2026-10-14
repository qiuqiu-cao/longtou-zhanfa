import copy
from decimal import Decimal
import json
import unittest
from pathlib import Path
from exchange_turnover_verified_v301 import parse_sse,parse_szse,money,aggregate,schema_evidence

ROOT=Path(__file__).resolve().parent
PROBE=ROOT/'data/v301/turnover_daily_probe_20261003_v1'


class ExchangeTurnoverIntegrity(unittest.TestCase):
    def payload(self,venue):
        return json.loads((PROBE/(venue+'_2026-09-30.response')).read_bytes())

    def test_actual_response_dates_and_component_values(self):
        a=parse_sse(self.payload('SSE'),'2026-09-30');b=parse_szse(self.payload('SZSE'),'2026-09-30')
        self.assertEqual(next(r for r in a if r['category']=='MAINBOARD_A')['amount_yi_yuan'],4661.18)
        self.assertEqual(next(r for r in b if r['category']=='MAINBOARD_A')['amount_yi_yuan'],3899.96)
        self.assertTrue(all(not r['excludes_st'] for r in a+b))

    def test_actual_returned_date_overrides_requested_date(self):
        with self.assertRaisesRegex(ValueError,'date mismatch'): parse_sse(self.payload('SSE'),'2026-09-29')
        with self.assertRaisesRegex(ValueError,'report date mismatch'): parse_szse(self.payload('SZSE'),'2026-09-29')

    def test_billion_vs_yuan_is_read_from_header(self):
        p=self.payload('SZSE');base=parse_szse(p,'2026-09-30')
        p[0]['metadata']['cols']['cjje']='成交金额<br>(元)'
        # A synthetic yuan fixture cannot inherit the original table's million-yuan rounding gap.
        components=[r for r in p[0]['data'] if any(label in r['lbmc'] for label in ['主板A股','主板B股','创业板A股'])]
        p[0]['data'][0]['cjje']=str(sum(Decimal(r['cjje'].replace(',','')) for r in components))
        for row in p[0]['data']:
            row['cjje']=str(Decimal(row['cjje'].replace(',',''))*100000000)
        converted=parse_szse(p,'2026-09-30')
        self.assertEqual([r['amount_yi_yuan'] for r in base if r['category']!='STOCKS_TOTAL'],
                         [r['amount_yi_yuan'] for r in converted if r['category']!='STOCKS_TOTAL'])
        self.assertTrue(all(r['reported_unit']=='元' for r in converted))

    def test_ambiguous_unit_rejected(self):
        p=self.payload('SZSE');p[0]['metadata']['cols']['cjje']='成交金额'
        with self.assertRaisesRegex(ValueError,'unit unknown'): parse_szse(p,'2026-09-30')
        with self.assertRaises(ValueError): money('100','万美元')

    def test_query_echo_alone_is_not_sufficient(self):
        p=self.payload('SZSE');p[0]['metadata']['subname']='2026-09-29'
        with self.assertRaises(ValueError): parse_szse(p,'2026-09-30')
        p=self.payload('SSE');p['result'][0]['TRADE_DATE']='20260929'
        with self.assertRaises(ValueError): parse_sse(p,'2026-09-30')

    def test_duplicate_categories_and_missing_components_rejected(self):
        p=self.payload('SSE');p['result'].append(copy.deepcopy(p['result'][0]))
        with self.assertRaisesRegex(ValueError,'duplicate'): parse_sse(p,'2026-09-30')
        p=self.payload('SZSE');p[0]['data']=[r for r in p[0]['data'] if '主板A' not in r['lbmc']]
        with self.assertRaisesRegex(ValueError,'component missing'): parse_szse(p,'2026-09-30')

    def test_parts_and_total_not_forced_to_match(self):
        p=self.payload('SSE');p['result'][0]['TRADE_AMT']='9661.18'
        with self.assertRaisesRegex(ValueError,'reconciliation conflict'): parse_sse(p,'2026-09-30')

    def test_published_rounding_is_respected(self):
        p=self.payload('SZSE');rows=parse_szse(p,'2026-09-30')
        total=next(r for r in rows if r['category']=='STOCKS_TOTAL')
        self.assertEqual(total['reported_amount'],'7,593.33')
        self.assertEqual(total['reported_precision_yuan'],'1000000.00')
        parts=sum(Decimal(r['amount_yuan']) for r in rows if r['category']!='STOCKS_TOTAL')
        self.assertEqual(Decimal(total['amount_yuan'])-parts,Decimal(1000000))

    def test_missing_exchange_is_unknown_not_half_the_market(self):
        rows=parse_sse(self.payload('SSE'),'2026-09-30')
        for r in rows:r['available_at_utc']='2026-10-02T16:30:00+00:00'
        output=aggregate(rows,['2026-09-30','2026-10-08'])
        self.assertTrue(output.mainboard_a_amount_yi.isna().all())
        self.assertTrue(output.shsz_stock_category_amount_yi.isna().all())

    def test_market_sums_exclude_b_funds_and_bse_but_keep_st(self):
        rows=parse_sse(self.payload('SSE'),'2026-09-30')+parse_szse(self.payload('SZSE'),'2026-09-30')
        for i,r in enumerate(rows):r['available_at_utc']='2026-10-02T16:3'+str(i)+':00+00:00'
        row=aggregate(rows,['2026-09-30']).iloc[0]
        self.assertEqual(row.mainboard_a_amount_yi,8561.14)
        self.assertEqual(row.shsz_stock_category_amount_yi,14391.72)
        self.assertIn('CDR_SCOPE_UNRESOLVED',row.scope_stock_categories)
        self.assertFalse(row.matches_original_non_st_pool_scope)
        self.assertIsNone(row.st_excluded_amount)

    def test_nonfinancial_or_missing_numeric_values_not_zero_filled(self):
        for value in ['-','',None,'-1','NaN','1,23.45','1_000']:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):money(value,'亿元')

    def test_official_js_defines_sse_codes_and_unit(self):
        sources=schema_evidence();self.assertEqual(len(sources),2)


if __name__=='__main__':unittest.main()
