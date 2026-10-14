import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import absorption_forward_registry_v141 as registry
from absorption_forward_protocol_v139 import protocol


class V141RegistryTest(unittest.TestCase):
    signal_date = (date.today() + timedelta(days=1)).isoformat()
    trade_date = (date.today() + timedelta(days=2)).isoformat()
    settle_date = (date.today() + timedelta(days=7)).isoformat()

    def setup(self, directory):
        root=Path(directory); (root/"protocol.json").write_text(json.dumps(protocol()),encoding="utf-8")
        with patch("absorption_forward_registry_v141.DEPENDENCIES",[]): registry.initialize(root)
        decision=root/"decision.json"
        rows=[{"code":"600001","name":"甲","boards":1,"board_group":"FIRST_BOARD",
               "minute_data_status":"VERIFIED_PUBLIC_MINUTE_BAR_FRESH","close_to_vwap_proxy":.01,
               "absorption_shadow_gate_v139":"SHADOW_PASS_ABOVE_OR_AT_VWAP"},
              {"code":"000002","name":"乙","boards":2,"board_group":"CONSECUTIVE_BOARD",
               "minute_data_status":"VERIFIED_PUBLIC_MINUTE_BAR_FRESH","close_to_vwap_proxy":-.01,
               "absorption_shadow_gate_v139":"SHADOW_VETO_BELOW_VWAP"}]
        decision.write_text(json.dumps({"signal_date":self.signal_date,"trade_date":self.trade_date,"asof":"10:00",
            "direct_entry_authorized":False,"absorption_shadow_gate_live_effect":"NONE_RESEARCH_ONLY",
            "primary_choice":{"code":"600001"},"first_board_candidates":[rows[0]],"relay_candidates":[rows[1]]}),encoding="utf-8")
        registry.snapshot_decision(root,decision,f"{self.trade_date}T10:01:00+08:00")
        return root

    def test_primary_bound_and_exact_settlement(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.setup(directory); snapshot=next((root/"snapshots").rglob("*.json"))
            saved=json.loads(snapshot.read_text(encoding="utf-8")); self.assertEqual(saved["primary_code"],"600001")
            outcomes=root/"outcomes.csv"
            pd.DataFrame([{"code":"600001","outcome_status":"OBSERVED","b1_net_close_mark":.01,
                "b2_net_close_mark":.02,"b3_net_close_mark":.03,"execution_state":"CONSERVATIVE_SIM_FILL",
                "actual_fill_verified":False,"exit_complete":True,"entry_price":10,"exit_price":10.3,
                "net_return_after_costs":.025,"provenance":"FROZEN_SIMULATOR"},
                {"code":"000002","outcome_status":"OBSERVED","b1_net_close_mark":-.01,
                "b2_net_close_mark":-.02,"b3_net_close_mark":-.03,"execution_state":"NO_FROZEN_ENTRY",
                "actual_fill_verified":False,"exit_complete":False,"entry_price":None,"exit_price":None,
                "net_return_after_costs":None,"provenance":"MARKET_OUTCOME_ONLY"}]).to_csv(outcomes,index=False)
            result=registry.settle(root,self.trade_date,outcomes,f"{self.settle_date}T16:00:00+08:00")
            self.assertEqual(result["candidate_count"],2); self.assertEqual(registry.verify(root)["status"],"PASS")

    def test_nonprimary_cannot_be_reclassified_as_trade(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.setup(directory); outcomes=root/"bad.csv"
            pd.DataFrame([{"code":"600001","outcome_status":"UNAVAILABLE","execution_state":"NO_FILL"},
                          {"code":"000002","outcome_status":"UNAVAILABLE","execution_state":"ACTUAL_FILL"}]).to_csv(outcomes,index=False)
            with self.assertRaises(ValueError):
                registry.settle(root,self.trade_date,outcomes,f"{self.settle_date}T16:00:00+08:00")
