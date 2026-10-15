import unittest

import numpy as np
import pandas as pd

import historical_symbolic_context as h


class SymbolicContextTests(unittest.TestCase):
    def setUp(self):
        self.days=["2025-01-02","2025-01-03","2025-01-06","2025-01-07"]
        self.a=pd.DataFrame([dict(signal_date=d,code=f"00000{i}",name="华"+x,board_height=n,pool_assignment="CLOSING_LIMIT_UP")
            for d in self.days for i,x,n in [(1,"甲",3),(2,"乙",2),(3,"丙",1),(4,"丁",1)]])
        self.c=self.a[self.a.board_height.ge(2)].rename(columns={"signal_date":"date","board_height":"boards"})[["date","code","name","boards"]]
        self.names=pd.DataFrame([dict(code=f"00000{i}",name="华"+x,valid_from=self.days[0],valid_to=self.days[-1],data_status="verified_import",interval_id=str(i),source_timestamp="2026-10-15T08:00:00+00:00",source_ref="local-official-history") for i,x in [(1,"甲"),(2,"乙"),(3,"丙"),(4,"丁")]])

    def run_engine(self,c=None,a=None,complete=None,names=None,industry=None):
        return h.build_causal(self.c if c is None else c,self.a if a is None else a,self.names if names is None else names,self.days,
            self.days if complete is None else complete,industry_evidence=industry)

    def test_first_support_and_multi_role_are_counted_without_business_claim(self):
        f,clusters,_=self.run_engine()
        leaders=f[f.code.eq("000001")]
        self.assertTrue(leaders.symbolic_primary_first_support.eq(2).all())
        self.assertTrue(leaders.symbolic_role_theme_leader_proxy.eq(1).all())
        self.assertTrue(all(not c["active_trading_theme_verified"] for c in clusters))
        self.assertTrue(f.independent_role_truth.eq(False).all())

    def test_archived_current_name_cannot_override_earlier_interval(self):
        c=self.c.copy();c.loc[c.code.eq("000001"),"name"]="未来新名"
        f,_,changes=self.run_engine(c=c)
        self.assertTrue(f[f.code.eq("000001")].historical_name.eq("华甲").all())
        # The source row is preserved, and the separately resolved name differs.
        self.assertTrue(f[f.code.eq("000001")].name.eq("未来新名").all())

    def test_future_suffix_and_future_targets_cannot_change_prefix(self):
        full,clusters,_=self.run_engine()
        cutoff=self.days[1]
        c=self.c[self.c.date.le(cutoff)].assign(target=999.,next_open_gap=100.)
        a=self.a[self.a.signal_date.le(cutoff)]
        prefix,pc,_=self.run_engine(c=c,a=a)
        pd.testing.assert_frame_equal(full[full.date.le(cutoff)].reset_index(drop=True),prefix.reset_index(drop=True))
        self.assertEqual([x for x in clusters if x["date"]<=cutoff],pc)
        self.assertNotIn("target",prefix)

    def test_incomplete_first_pool_keeps_structural_counts_missing(self):
        f,_,_=self.run_engine(complete=[])
        self.assertTrue(f.symbolic_primary_first_support.isna().all())
        self.assertTrue(f.symbolic_role_theme_seed_proxy.isna().all())
        self.assertTrue(f.symbolic_cluster_count.isna().all())

    def test_missing_official_day_is_not_skipped_for_change_or_streak(self):
        f,_,_=self.run_engine(complete=[self.days[0],self.days[2],self.days[3]])
        later=f[f.date.eq(self.days[2])]
        self.assertTrue(later.symbolic_primary_first_change_1.isna().all())
        self.assertTrue(later.symbolic_qualified_streak.eq(1).all())

    def test_two_parallel_seeds_can_coexist(self):
        c=self.c.copy();c["boards"]=2
        a=self.a.copy();a.loc[a.board_height.eq(3),"board_height"]=2
        f,_,_=self.run_engine(c=c,a=a)
        self.assertTrue(f.symbolic_role_theme_seed_proxy.eq(1).all())

    def test_industry_evidence_never_moves_to_another_date(self):
        ind=pd.DataFrame([dict(date=self.days[-1],code="000001",industry="行业代理")])
        f,_,_=self.run_engine(industry=ind)
        self.assertEqual(f.symbolic_industry_evidence_available.sum(),1)
        self.assertTrue(f[f.date.lt(self.days[-1])].industry_name.isna().all())

    def test_overlap_and_unbounded_name_intervals_are_rejected(self):
        names=pd.concat([self.names,self.names.iloc[:1]],ignore_index=True)
        with self.assertRaisesRegex(ValueError,"overlapping"):
            self.run_engine(names=names)

    def test_known_historical_st_cannot_supply_symbolic_first_board_support(self):
        names=self.names.copy();names["board"]="MAINBOARD";names["price_limit_regime"]="MAINBOARD_10";names["listing_status"]="LISTED";names["st_status"]="NORMAL"
        names.loc[names.code.eq("000003"),["st_status","price_limit_regime"]]=["ST","ST_5"]
        f,clusters,_=self.run_engine(names=names)
        self.assertFalse(clusters)  # Only3 eligible supporters, below frozen4.
        self.assertTrue(f.symbolic_cluster_count.eq(0).all())


if __name__=="__main__":unittest.main()
