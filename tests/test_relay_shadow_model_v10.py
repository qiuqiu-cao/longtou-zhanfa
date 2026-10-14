import unittest

import numpy as np

from relay_shadow_model_v10 import _logit, canonical_json


class RelayShadowModelV10Tests(unittest.TestCase):
    def test_canonical_json_is_key_order_independent(self):
        self.assertEqual(canonical_json({"b": 2, "a": 1}), canonical_json({"a": 1, "b": 2}))

    def test_logit_clips_binary_extremes(self):
        values = _logit(np.array([0.0, 0.5, 1.0]))
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(values[1, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
