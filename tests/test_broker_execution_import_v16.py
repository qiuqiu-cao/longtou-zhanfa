import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import broker_execution_import_v16 as broker


class BrokerExecutionImportV16Tests(unittest.TestCase):
    @staticmethod
    def order_frame(status="已成", time="09:36:00", quantity="100"):
        return pd.DataFrame([{
            "委托编号": "O1", "证券代码": "600001", "证券名称": "样本",
            "买卖方向": "买入", "委托时间": time, "委托价格": "11.00",
            "委托数量": quantity, "委托状态": status,
        }])

    @staticmethod
    def fill_frame(time="09:36:01", quantity="100", price="10.98"):
        return pd.DataFrame([{
            "成交编号": "F1", "委托编号": "O1", "证券代码": "600001",
            "证券名称": "样本", "成交方向": "买入", "成交时间": time,
            "成交价格": price, "成交数量": quantity, "佣金": "5",
            "印花税": "0", "过户费": "0.1", "其他费用": "0",
        }])

    @staticmethod
    def decision(status="ENTER", authorized=True):
        return {
            "trade_date": "2026-09-07", "asof": "09:35", "status": status,
            "direct_entry_authorized": authorized, "order_submitted": False,
            "primary_choice": {"code": "600001", "maximum_acceptable_price": 11.00},
        }

    def test_authorized_post_confirmation_fill_is_strategy_eligible(self):
        orders = broker.normalize_orders(self.order_frame(), "2026-09-07")
        fills = broker.normalize_fills(self.fill_frame(), "2026-09-07")
        result = broker.reconcile(self.decision(), orders, fills, "2026-09-07T10:00:00+08:00")
        self.assertEqual(result["strategy_eligible_filled_orders"], 1)
        self.assertEqual(result["orders"][0]["classification"], "STRATEGY_COMPLIANT_FULL_FILL")
        self.assertFalse(result["complete_trade_created"])

    def test_watch_fill_is_recorded_as_off_plan_not_strategy_success(self):
        orders = broker.normalize_orders(self.order_frame(), "2026-09-07")
        fills = broker.normalize_fills(self.fill_frame(), "2026-09-07")
        result = broker.reconcile(self.decision("WATCH_PUBLIC_EVIDENCE_ONLY", False), orders, fills,
                                  "2026-09-07T10:00:00+08:00")
        self.assertEqual(result["off_plan_filled_orders"], 1)
        self.assertEqual(result["strategy_eligible_filled_orders"], 0)

    def test_preconfirmation_order_is_off_plan_even_if_filled_later(self):
        orders = broker.normalize_orders(self.order_frame(time="09:34:59"), "2026-09-07")
        fills = broker.normalize_fills(self.fill_frame(), "2026-09-07")
        result = broker.reconcile(self.decision(), orders, fills, "2026-09-07T10:00:00+08:00")
        self.assertFalse(result["orders"][0]["strategy_sample_eligible"])

    def test_partial_and_no_fill_are_not_hidden(self):
        orders = broker.normalize_orders(self.order_frame(quantity="200"), "2026-09-07")
        fills = broker.normalize_fills(self.fill_frame(quantity="100"), "2026-09-07")
        partial = broker.reconcile(self.decision(), orders, fills, "2026-09-07T10:00:00+08:00")
        self.assertEqual(partial["partial_fill_orders"], 1)
        no_fills = broker.normalize_fills(self.fill_frame().iloc[0:0], "2026-09-07")
        no_fill = broker.reconcile(self.decision(), orders, no_fills, "2026-09-07T10:00:00+08:00")
        self.assertEqual(no_fill["no_fill_orders"], 1)

    def test_fill_cannot_exceed_order(self):
        orders = broker.normalize_orders(self.order_frame(), "2026-09-07")
        fills = broker.normalize_fills(self.fill_frame(quantity="200"), "2026-09-07")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            broker.reconcile(self.decision(), orders, fills, "2026-09-07T10:00:00+08:00")

    def test_run_hashes_normalized_execution_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = root / "decision.json"
            orders = root / "orders.csv"
            fills = root / "fills.csv"
            decision.write_text(json.dumps(self.decision()), encoding="utf-8")
            self.order_frame().to_csv(orders, index=False)
            self.fill_frame().to_csv(fills, index=False)
            result = broker.run(
                decision, orders, fills, "2026-09-07T10:00:00+08:00", "test export", root / "out",
            )
            for item in result["normalized_files"].values():
                path = Path(item["path"])
                self.assertTrue(path.exists())
                self.assertEqual(broker.sha256_file(path), item["sha256"])
            stored = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(stored["import_id"], result["import_id"])


if __name__ == "__main__":
    unittest.main()
