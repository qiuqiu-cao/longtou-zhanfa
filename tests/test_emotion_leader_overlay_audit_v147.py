import unittest

import pandas as pd

from emotion_leader_overlay_audit_v147 import audit


def frames(high_bad=True):
    top, emotion = [], []
    for index in range(30):
        band = "HIGH" if index < 10 else "LOW" if index < 20 else "MIXED"
        if band == "HIGH":
            value, loss = (-.10, 1) if high_bad else (.10, 0)
        elif band == "LOW":
            value, loss = (.08, 0) if high_bad else (-.08, 1)
        else:
            value, loss = (0, 0)
        date = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=index)).strftime("%Y-%m-%d")
        top.append({"signal_date": date, "fill_proxy_target": 1,
                    "b2_net_close_mark_LABEL_ONLY": value,
                    "b2_big_loss_mark_LABEL_ONLY": loss})
        emotion.append({"signal_date": date, "risk_band": band,
                        "risk_off_probability": {"HIGH": .8, "MIXED": .5, "LOW": .2}[band]})
    return pd.DataFrame(top), pd.DataFrame(emotion)


class EmotionLeaderOverlayAuditTests(unittest.TestCase):
    def test_strong_synthetic_separation_supports_narrow_policies(self):
        top, emotion = frames(True)
        joined, report = audit(top, emotion)
        self.assertEqual(len(joined), 30)
        self.assertTrue(report["policy_evidence"]["high_risk_top1_veto_supported"])
        self.assertTrue(report["policy_evidence"]["low_risk_positive_rank_bonus_supported"])
        self.assertFalse(report["policy_evidence"]["stock_reranking_allowed"])

    def test_reversed_relationship_fails_both_claims(self):
        top, emotion = frames(False)
        _, report = audit(top, emotion)
        self.assertFalse(report["policy_evidence"]["high_risk_top1_veto_supported"])
        self.assertFalse(report["policy_evidence"]["low_risk_positive_rank_bonus_supported"])

    def test_duplicate_date_and_bad_band_fail_closed(self):
        top, emotion = frames()
        with self.assertRaises(ValueError):
            audit(pd.concat([top, top.iloc[[0]]]), emotion)
        emotion.loc[0, "risk_band"] = "UNKNOWN"
        with self.assertRaises(ValueError):
            audit(top, emotion)


if __name__ == "__main__":
    unittest.main()
