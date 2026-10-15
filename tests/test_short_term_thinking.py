"""Causal contracts for new retrospective research; no performance assertions."""
import copy
import unittest

import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b


class CausalThinkingTests(unittest.TestCase):
    def setUp(self):
        self.days = pd.bdate_range("2025-01-02", periods=9).strftime("%Y-%m-%d").tolist()
        c, bars, attempts, market = [], [], [], []
        for i, day in enumerate(self.days):
            market.append(dict(date=day, max_board=min(i+1, 6), limit_count=20, failed_rate=.4,
                down_limit_count=3, yesterday_premium=.01, promotion_mid=.4, one_word_share=.2))
            for code, name, boards in (("000001", "九阳股份", min(i+1, 6)), ("000002", "百洋股份", max(1, i-2))):
                if i >= 6 and code == "000001":
                    # A complete daily observation on the first broken day.
                    bars.append(dict(date=day, code=code, open=10., high=10., low=9., close=9., volume=200.))
                    continue
                r = dict(date=day, code=code, name=name, boards=boards, price=10., signal_amount=100.,
                    turnover=.1, amplitude=5., signal_float_cap=1000., industry="", theme="")
                if boards >= 2:
                    c.append(r)
                attempts.append(dict(signal_date=day, code=code, name=name, pool_assignment="CLOSING_LIMIT_UP", board_height=boards, close=10.))
                bars.append(dict(date=day, code=code, open=9.5, high=10., low=9., close=10., volume=100.))
        self.c = pd.DataFrame(c)
        self.bars = pd.DataFrame(bars)
        self.attempts = pd.DataFrame(attempts)
        self.market = pd.DataFrame(market)

    def run_engine(self, candidates=None, market=None, attempts=None, bars=None):
        return b.causal_features(self.c if candidates is None else candidates,
            self.market if market is None else market, self.attempts if attempts is None else attempts,
            self.bars if bars is None else bars, self.days)

    def test_future_suffix_cannot_change_earlier_features_or_parent_edges(self):
        cutoff = self.days[5]
        full, full_edges = self.run_engine()
        truncated, edges = self.run_engine(self.c[self.c.date.le(cutoff)], self.market[self.market.date.le(cutoff)],
            self.attempts[self.attempts.signal_date.le(cutoff)], self.bars[self.bars.date.le(cutoff)])
        pd.testing.assert_frame_equal(full[full.date.le(cutoff)].reset_index(drop=True), truncated.reset_index(drop=True))
        self.assertEqual([e for e in full_edges if e["date"] <= cutoff], edges)

    def test_outcome_columns_and_adversarial_future_prices_are_not_features(self):
        expected, edges = self.run_engine()
        candidates = self.c.assign(target=np.arange(len(self.c)) % 2, next_open_gap=99., terminal_board=999., net_return=100.)
        actual, more = self.run_engine(candidates=candidates)
        pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(edges, more)
        self.assertFalse(set(actual) & {"target", "next_open_gap", "terminal_board", "net_return"})

    def test_one_cent_conflict_is_not_a_nonword_observation(self):
        self.assertIsNone(b.price_state(dict(open=9, high=10, low=9, close=10), 10.01)["one_word"])
        self.assertEqual(b.price_state(dict(open=10, high=10, low=10, close=10), 10)["one_word"], 1)
        self.assertEqual(b.price_state(dict(open=9.5, high=10, low=9.5, close=10), 10)["one_word"], 0)

    def test_missing_parent_day_never_infers_first_break(self):
        removed = self.bars[~((self.bars.date == self.days[6]) & (self.bars.code == "000001"))]
        _, edges = self.run_engine(bars=removed)
        e = [x for x in edges if x["date"] == self.days[6] and x["anchor_code"] == "000001"]
        self.assertTrue(e)
        self.assertTrue(all(x["parent_anchor_state"] == "ANCHOR_UNOBSERVED" and x["handover_on_anchor_break"] == 0 for x in e))

    def test_full_same_height_denominator_and_coexistence(self):
        day = self.days[5]
        c = self.c.copy(); c.loc[c.date.eq(day), "boards"] = 3
        a = pd.concat([self.attempts, pd.DataFrame([dict(signal_date=day, code="000003", name="失败", pool_assignment="FAILED_BOARD_OBSERVATION", board_height=3, close=9.)])], ignore_index=True)
        f, _ = self.run_engine(candidates=c, attempts=a)
        same = f[f.date.eq(day)]
        self.assertEqual(len(same), 2)
        self.assertTrue(same.same_height_success_count.eq(2).all())
        self.assertTrue(same.same_height_elimination_rate.eq(1/3).all())

    def test_unknown_shape_and_no_anchor_are_different_states(self):
        f, _ = self.run_engine(bars=self.bars.iloc[:0])
        self.assertEqual(f.daily_identity_observed.sum(), 0)
        self.assertFalse(f.primary_role.isna().any())
        self.assertTrue(f.verified_turnover_boards.isna().all())
        self.assertTrue(f.parent_anchor_state.isin(["NO_ACTIVE_PARENT_IN_ARCHIVE", "ANCHOR_UNOBSERVED", "NO_MATCHED_PARENT", "ANCHOR_CONTINUING"]).all())

    def test_new_rule_roles_are_not_independent_identity_gold(self):
        f, _ = self.run_engine()
        self.assertTrue(f.role_identity_accuracy.isna().all())
        self.assertTrue(f.role_status.str.contains("NOT_INDEPENDENT_TRUTH").all())
        self.assertFalse(f.primary_role.eq("UNKNOWN").any())

    def test_closing_only_pool_does_not_mean_zero_failed_boards(self):
        f,_=self.run_engine()
        self.assertTrue(f.same_height_elimination_rate.isna().all())
        a=self.attempts.copy();a.attrs["failed_scope_complete_dates"]=self.days
        known,_=self.run_engine(attempts=a)
        self.assertTrue(known.same_height_elimination_rate.eq(0).all())

    def test_v215_one_cent_tolerance_is_removed_only_in_new_overlay(self):
        original = pd.DataFrame([dict(code="000001", high=10., close=9.99, limit_price_10=10., pool_assignment="CLOSING_LIMIT_UP"),
            dict(code="000002", high=10., close=9.9, limit_price_10=10., pool_assignment="FAILED_BOARD_OBSERVATION")])
        before = original.copy(deep=True)
        exact, audit = b.exact_attempts(original)
        self.assertEqual(exact.code.tolist(), ["000002"])
        self.assertEqual(audit["exact_price_or_assignment_excluded"], 1)
        pd.testing.assert_frame_equal(original, before)


if __name__ == "__main__":
    unittest.main()
