import unittest
import pandas as pd
from leader_episode_inventory_v20 import inventory_one


class InventoryTests(unittest.TestCase):
    def fixture(self):
        dates = pd.bdate_range("2026-01-01", periods=5).strftime("%Y-%m-%d").tolist()
        frame = pd.DataFrame(dict(date=dates, code=["000017"]*5, close=[10,11,12.1,13.31,12],
                                  pre_close=[10,10,11,12.1,13.31], status_verified=True, st=False, new_listing=False))
        return frame, dates

    def test_failures_are_in_denominator_and_terminal_features_separate(self):
        obs, eps = inventory_one(*self.fixture())
        self.assertEqual(len(obs), 2)
        self.assertEqual(eps[0]["termination"], "BROKEN_BOARD")
        self.assertNotIn("termination", obs[0])
        self.assertTrue(obs[0]["start_known"])

    def test_future_does_not_change_observations(self):
        frame, dates = self.fixture()
        left, _ = inventory_one(frame.iloc[:4], dates)
        right, _ = inventory_one(frame, dates)
        self.assertEqual(left, right)

    def test_gap_cannot_be_consecutive_boards(self):
        frame, dates = self.fixture()
        obs, _ = inventory_one(frame.drop(index=2), dates)
        self.assertEqual(obs, [])

    def test_unverified_status_is_not_assumed_nonst(self):
        frame, dates = self.fixture()
        frame["status_verified"] = False
        self.assertEqual(inventory_one(frame, dates), ([], []))

    def test_missing_st_is_not_nonst(self):
        frame, dates = self.fixture()
        frame["st"] = None
        self.assertEqual(inventory_one(frame, dates), ([], []))

    def test_full_pool_includes_first_board(self):
        frame, dates = self.fixture()
        obs, _ = inventory_one(frame, dates, min_boards=1)
        self.assertEqual([r["boards"] for r in obs], [1, 2, 3])
        self.assertTrue(obs[0]["start_known"])

    def test_truncated_history_not_known_first_board(self):
        frame, dates = self.fixture()
        obs, _ = inventory_one(frame.iloc[1:], dates, min_boards=1)
        self.assertFalse(obs[0]["start_known"])

    def test_full_pool_future_invariance(self):
        frame, dates = self.fixture()
        self.assertEqual(inventory_one(frame.iloc[:4], dates, min_boards=1)[0],
                         inventory_one(frame, dates, min_boards=1)[0])

    def test_full_pool_gap_does_not_claim_new_cycle(self):
        frame, dates = self.fixture()
        obs, _ = inventory_one(frame.drop(index=2), dates, min_boards=1)
        self.assertTrue(obs[0]["start_known"])
        self.assertFalse(obs[1]["start_known"])


if __name__ == "__main__":
    unittest.main()
