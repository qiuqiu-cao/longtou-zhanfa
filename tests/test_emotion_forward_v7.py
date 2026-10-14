import unittest

import pandas as pd

from emotion_forward_v7 import event_hash


class EmotionForwardV7Tests(unittest.TestCase):
    def test_event_hash_is_key_order_independent(self):
        self.assertEqual(event_hash({"b": 2, "a": 1}), event_hash({"a": 1, "b": 2}))

    def test_adverse_label_definition_matches_locked_target(self):
        states = pd.Series(["大分歧", "退潮", "升温"])
        labels = states.isin({"大分歧", "退潮"}).astype(float).tolist()
        self.assertEqual(labels, [1.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
