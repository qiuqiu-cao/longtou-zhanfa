import unittest
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import historical_reseal_replay_v48 as replay


class HistoricalResealReplayV48Tests(unittest.TestCase):
    def test_postclose_reconciliation_cannot_change_frozen_morning_selection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = root / 'cohort.csv'
            pd.DataFrame([dict(code='600000', provider_code='sh.600000',
                signal_date='2024-01-01', entry_date='2024-01-02', boards=1,
                start_known=True)]).to_csv(cohort, index=False)
            daily = root / 'daily.csv'
            pd.DataFrame([dict(code='600000', date=day, open=10, high=10,
                low=10, close=10, pre_close=10, volume=100)
                for day in ['2024-01-01', '2024-01-02']]).to_csv(daily, index=False)
            calendar = root / 'calendar.csv'
            pd.DataFrame({'date':['2024-01-01','2024-01-02']}).to_csv(calendar,index=False)
            raw = self.zero_archive()
            for row in raw['rows']:
                row[3:9] = ['10','10','10','10','100','1000']
            outputs = []
            for index, status in enumerate(['DAILY_OHLC_MATCH_ONLY', 'DAILY_OHLC_CONTRADICTION']):
                target = root / str(index)
                with patch.object(replay, '_read_archives', return_value=(
                        {('sh.600000','2024-01-02'):raw}, {})), patch.object(
                        replay, 'inspect', return_value={'status':status, 'differences':{}}):
                    report = replay.run([cohort], [], daily, calendar, None, target)
                outputs.append((target / 'signals_BEFORE_OUTCOMES.json').read_text())
                self.assertEqual(report['reconciled_validation_ready'], index == 0)
            self.assertEqual(outputs[0], outputs[1])

    @staticmethod
    def bar(clock, open_, high, low, close, volume=1000):
        return {"time": "20240102" + clock, "open": open_, "high": high,
                "low": low, "close": close, "volume": volume}

    @staticmethod
    def zero_archive(day="2024-01-02"):
        times = [t.strftime("%Y%m%d%H%M%S") + "000"
                 for start, end in (("09:35", "11:30"), ("13:05", "15:00"))
                 for t in pd.date_range(day + " " + start, day + " " + end, freq="5min")]
        fields = ["date", "time", "code", "open", "high", "low", "close",
                  "volume", "amount", "adjustflag"]
        rows = [[day, clock, "sh.600000", "0", "0", "0", "0", "0", "0", "3"]
                for clock in times]
        return {"requested_date": day, "code": "sh.600000", "error_code": "0",
                "fields": fields, "rows": rows}

    def test_complete_zero_bar_archive_is_nontradable_evidence(self):
        archive = self.zero_archive()
        self.assertTrue(replay._all_zero_no_trade_archive(archive))
        archive["rows"][-1][3:7] = ["10.0"] * 4
        self.assertTrue(replay._all_zero_no_trade_archive(archive))
        archive["rows"][0][3] = "1.0"
        self.assertFalse(replay._all_zero_no_trade_archive(archive))

    def test_future_bars_cannot_change_signal(self):
        prefix = [self.bar("093500000", 9.8, 10.0, 9.7, 9.9),
                  self.bar("094000000", 9.9, 10.0, 9.8, 10.0)]
        first = replay.causal_reseal_signal(prefix, "2024-01-02", 10.0)
        altered_future = prefix + [self.bar("100500000", 10.0, 10.0, 8.0, 8.5)]
        second = replay.causal_reseal_signal(altered_future, "2024-01-02", 10.0)
        self.assertEqual(first, second)
        self.assertEqual(first["signal_type"], "TOUCHED_PULLBACK_RESEAL")

    def test_no_same_bar_fill_claim(self):
        bars = [self.bar("093500000", 9.8, 10.0, 9.7, 9.9),
                self.bar("094000000", 9.9, 10.0, 9.8, 10.0),
                self.bar("094500000", 10.0, 10.0, 9.85, 9.9)]
        signal = replay.causal_reseal_signal(bars, "2024-01-02", 10.0)
        label = replay.next_bar_entry_label(bars, signal)
        self.assertEqual(label["state"], "BELOW_LIMIT_ENTRY_OPPORTUNITY_PROXY")
        self.assertFalse(label["actual_fill_verified"])
        self.assertFalse(label["simulated_fill"])

    def test_limit_only_next_bar_keeps_queue_unresolved(self):
        bars = [self.bar("093500000", 9.8, 10.0, 9.7, 9.9),
                self.bar("094000000", 9.9, 10.0, 9.8, 10.0),
                self.bar("094500000", 10.0, 10.0, 10.0, 10.0)]
        signal = replay.causal_reseal_signal(bars, "2024-01-02", 10.0)
        self.assertEqual(replay.next_bar_entry_label(bars, signal)["state"], "LIMIT_QUEUE_UNRESOLVED")

    def test_entry_label_rejects_unordered_full_session(self):
        bars = [self.bar("093500000", 9.8, 10.0, 9.7, 9.9),
                self.bar("094500000", 10.0, 10.0, 9.9, 10.0),
                self.bar("094000000", 9.9, 10.0, 9.8, 10.0)]
        signal = {"signal": True, "confirmation_index": 1, "entry_price_cap": 10.0}
        with self.assertRaisesRegex(ValueError, "ordered and unique"):
            replay.next_bar_entry_label(bars, signal)

    def test_cutoff_policy_requires_still_sealed_at_1000(self):
        bars = [self.bar("093500000", 9.8, 10.0, 9.7, 9.9),
                self.bar("094000000", 9.9, 10.0, 9.8, 10.0),
                self.bar("100000000", 10.0, 10.0, 9.8, 9.9),
                self.bar("100500000", 9.9, 10.0, 9.8, 10.0)]
        signal = replay.stable_cutoff_signal(bars, "2024-01-02", 10.0)
        self.assertFalse(signal["signal"])
        bars[2] = self.bar("100000000", 9.9, 10.0, 9.8, 10.0)
        stable = replay.stable_cutoff_signal(bars, "2024-01-02", 10.0)
        self.assertTrue(stable["signal"])
        self.assertEqual(stable["confirmation_time"], "20240102100000000")

    def test_one_per_day_selection_uses_no_outcomes(self):
        rows = [
            {"code": "000002", "entry_date": "2024-01-02", "boards": 2,
             "candidate_scope_eligible": True,
             "signal": {"signal": True, "confirmation_time": "20240102094000000"}},
            {"code": "000001", "entry_date": "2024-01-02", "boards": 3,
             "candidate_scope_eligible": True,
             "signal": {"signal": True, "confirmation_time": "20240102094000000"},
             "future_profit": -1},
            {"code": "000003", "entry_date": "2024-01-02", "boards": 9,
             "candidate_scope_eligible": True,
             "signal": {"signal": True, "confirmation_time": "20240102094500000"},
             "future_profit": 99},
        ]
        selected = replay.choose_one_per_day(rows)
        self.assertEqual([row["code"] for row in selected], ["000001"])

    def test_selection_fails_closed_when_any_frozen_candidate_is_missing(self):
        rows = [{"code": "000001", "entry_date": "2024-01-02", "boards": 2,
                 "candidate_scope_eligible": True,
                 "signal": {"signal": True, "confirmation_time": "20240102094000000"},
                 "cutoff_signal_1000": {"signal": True,
                                         "confirmation_time": "20240102100000000"}}]
        selected, cutoff, status = replay.select_with_coverage(
            rows, [{"code": "000002", "state": "MISSING_ARCHIVE"}])
        self.assertEqual(selected, [])
        self.assertEqual(cutoff, [])
        self.assertEqual(status, "BLOCKED_INCOMPLETE_CANDIDATE_UNIVERSE")
        selected, _, status = replay.select_with_coverage(
            rows, [{"code": "000002", "state": "MISSING_ARCHIVE"}], True)
        self.assertEqual(selected[0]["code"], "000001")
        self.assertEqual(status, "INCOMPLETE_DEVELOPMENT_OVERRIDE")

    def test_only_exact_or_contained_daily_reconciliation_is_usable(self):
        self.assertTrue(replay.reconciliation_usable("DAILY_OHLC_MATCH_ONLY"))
        self.assertTrue(replay.reconciliation_usable("DAILY_OHLC_RANGE_CONSISTENT"))
        self.assertFalse(replay.reconciliation_usable("DAILY_OHLC_CONTRADICTION"))
        self.assertFalse(replay.reconciliation_usable("INVALID_IDENTITY_OR_COVERAGE"))

    def test_left_censored_board_chain_is_not_selectable(self):
        rows = [
            {"code": "000001", "entry_date": "2024-01-02", "boards": 6,
             "candidate_scope_eligible": False,
             "signal": {"signal": True, "confirmation_time": "20240102094000000"}},
            {"code": "000002", "entry_date": "2024-01-02", "boards": 2,
             "candidate_scope_eligible": True,
             "signal": {"signal": True, "confirmation_time": "20240102094500000"}},
        ]
        self.assertEqual(replay.choose_one_per_day(rows)[0]["code"], "000002")
        self.assertEqual(replay.board_group({"boards": 6, "start_known": False}),
                         "UNKNOWN_LEFT_CENSORED")

    def test_holding_marks_preserve_missing_and_blocked(self):
        index = pd.MultiIndex.from_tuples([("000001", "2024-01-03"),
                                            ("000001", "2024-01-04")])
        daily = pd.DataFrame([
            {"close": 9.0, "pre_close": 10.0, "volume": 100},
            {"close": 9.5, "pre_close": 9.0, "volume": 100},
        ], index=index)
        row = {"code": "000001", "entry_date": "2024-01-02",
               "signal": {"entry_price_cap": 10.0}}
        result = replay.holding_marks(row, daily,
                                      ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        self.assertEqual(result["b1_state"], "CLOSE_MARK_SELL_BLOCKED_PROXY")
        self.assertEqual(result["b3_state"], "MISSING_DAILY_BAR")
        self.assertIsNone(result["realized_return"])

    def test_strict_v47_import_bridge_accepts_only_complete_five_minute_case(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            day = "2024-01-02"
            times = [t.strftime("%Y%m%d%H%M%S") + "000"
                     for start, end in (("09:35", "11:30"), ("13:05", "15:00"))
                     for t in pd.date_range(day + " " + start, day + " " + end, freq="5min")]
            frame = pd.DataFrame({
                "code": ["000001"] * 48, "date": [day] * 48, "time": times,
                "open": [10.0] * 48, "high": [10.0] * 48, "low": [10.0] * 48,
                "close": [10.0] * 48, "volume": [100] * 48, "amount": [1000] * 48,
            })
            normalized = root / "normalized.csv"
            frame.to_csv(normalized, index=False)
            digest = hashlib.sha256(normalized.read_bytes()).hexdigest()
            manifest = {
                "interval": 5, "adjustment_declared": "unadjusted",
                "normalized_sha256": digest,
                "coverage": [{"code": "000001", "date": day, "bars": 48,
                              "row_count_complete_only": True}],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            cases, hashes = replay._read_imported_archives([root])
            case = cases[("sz.000001", day)]
            self.assertEqual(len(case["rows"]), 48)
            self.assertEqual(case["fields"][-1], "adjustflag")
            self.assertEqual(case["rows"][0][-1], "3")
            self.assertIn(str(normalized.resolve()), hashes)

    def test_import_bridge_rejects_non_five_minute_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            normalized = root / "normalized.csv"
            normalized.write_text("code,date,time,open,high,low,close,volume,amount\n", encoding="utf-8")
            manifest = {"interval": 1, "adjustment_declared": "unadjusted",
                        "normalized_sha256": hashlib.sha256(normalized.read_bytes()).hexdigest(),
                        "coverage": []}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "five-minute"):
                replay._read_imported_archives([root])


if __name__ == "__main__":
    unittest.main()
