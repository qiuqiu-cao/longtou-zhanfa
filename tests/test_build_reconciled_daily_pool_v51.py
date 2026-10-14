import unittest
import pandas as pd
from build_reconciled_daily_pool_v51 import build

class ReconcileTests(unittest.TestCase):
    def raw(self, code, name='甲', boards=1):
        return {'代码':code,'名称':name,'连板数':boards,'首次封板时间':93000,'最后封板时间':93100,
                '炸板次数':1,'最新价':11.0,'成交额':1e8,'换手率':5.0,'所属行业':'X','涨停价':11.0}
    def quote(self,code,name='甲'):
        return {'code':code,'name':name,'close':11.0,'amount_wan':10000,'turnover':5.0}
    def test_quote_close_moves_raw_failed_to_closing(self):
        up=pd.DataFrame([self.raw('600001')]); failed=pd.DataFrame([self.raw('600002')])
        qu=pd.DataFrame([self.quote('600001'),self.quote('600002')]);qf=pd.DataFrame(columns=['code','name'])
        prior=pd.DataFrame([{'代码':'600002','连板数':1}])
        closing,out_failed,report=build(up,failed,qu,qf,prior,'2026-09-08')
        self.assertEqual(set(closing.code),{'600001','600002'})
        self.assertEqual(closing.set_index('code').loc['600002','boards'],2)
        self.assertTrue(pd.isna(closing.set_index('code').loc['600002','last_seal_time']))
        self.assertEqual(len(out_failed),0)
        self.assertEqual(report['full_market_completeness'],'NOT_PROVEN_ARCHIVED_ROSTER_ONLY')
    def test_overlap_rejected(self):
        up=pd.DataFrame([self.raw('600001')]); failed=pd.DataFrame([self.raw('600002')])
        qu=pd.DataFrame([self.quote('600001')]);qf=pd.DataFrame([self.quote('600001')])
        with self.assertRaises(ValueError): build(up,failed,qu,qf,pd.DataFrame(columns=['代码','连板数']),'2026-09-08')
