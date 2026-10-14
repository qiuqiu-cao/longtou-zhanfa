import unittest
import numpy as np
import pandas as pd
from target_validation_ablation_v25 import score_full_pool


class FixedModel:
    def predict_proba(self, frame):
        p=frame["score"].to_numpy()
        return np.column_stack([1-p,p])


class FullPoolTests(unittest.TestCase):
    def test_missing_label_cannot_remove_highest_ranked_stock(self):
        pool=pd.DataFrame([dict(code="000001",signal_date="2025-08-01",score=.9,episode_id="a"),
                           dict(code="000002",signal_date="2025-08-01",score=.5,episode_id="b")])
        labels=pd.DataFrame([dict(code="000002",signal_date="2025-08-01",proxy_target=1)])
        result=score_full_pool(FixedModel(),["score"],pool,labels)
        self.assertEqual(result.iloc[0].code,"000001")
        self.assertTrue(pd.isna(result.iloc[0].proxy_target))

    def test_labels_do_not_change_ranking(self):
        pool=pd.DataFrame([dict(code="000001",signal_date="2025-08-01",score=.9,episode_id="a"),
                           dict(code="000002",signal_date="2025-08-01",score=.5,episode_id="b")])
        labels=pool[["code","signal_date"]].assign(proxy_target=[0,1])
        self.assertEqual(score_full_pool(FixedModel(),["score"],pool,labels).iloc[0].code,"000001")


if __name__ == "__main__":
    unittest.main()
