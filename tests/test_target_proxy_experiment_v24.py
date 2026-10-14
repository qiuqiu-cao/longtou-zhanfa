import unittest
import pandas as pd
from target_proxy_experiment_v24 import proxy_labels, selection_stats, audit_proxy_labels


class ProxyExperimentTests(unittest.TestCase):
    def fixture(self):
        calendar = ["2026-01-05","2026-01-06","2026-01-07","2026-01-08"]
        features = pd.DataFrame([dict(code="000017",signal_date=d,episode_id="e") for d in calendar[:3]])
        episodes = pd.DataFrame([dict(episode_id="e",start_known=True,start_date=calendar[0],
                                      last_limit_date=calendar[2],terminal_date=calendar[3],termination="BROKEN_BOARD")])
        events=[dict(code="000017",event_date=calendar[2])]
        return features,episodes,events,calendar

    def test_excludes_signals_after_target_and_labels_only_proxy(self):
        result=proxy_labels(*self.fixture())
        self.assertEqual(len(result),2)
        self.assertTrue(result.proxy_target.eq(1).all())
        self.assertTrue(result.label_status.str.contains("NOT_VERIFIED").all())

    def test_missing_event_is_only_proxy_negative(self):
        f,e,_,c=self.fixture()
        result=proxy_labels(f,e,[],c)
        self.assertTrue(result.proxy_target.eq(0).all())

    def test_incomplete_episode_does_not_become_failure(self):
        f,e,_,c=self.fixture()
        e["termination"]="CENSORED_END_OF_DATA"
        self.assertTrue(proxy_labels(f,e,[],c).empty)

    def test_reason_audit_retains_every_candidate(self):
        f,e,events,c=self.fixture()
        result=audit_proxy_labels(f,e,events,c)
        self.assertEqual(len(result),len(f))
        self.assertEqual(result.iloc[-1].label_status,"TARGET_ALREADY_OCCURRED_PROXY")

    def test_string_false_start_is_unknown_not_true(self):
        f,e,events,c=self.fixture()
        e["start_known"]="False"
        result=audit_proxy_labels(f,e,events,c)
        self.assertTrue(result.label_status.eq("START_UNKNOWN").all())

    def test_selected_episode_is_not_counted_twice(self):
        f,_,_,_=self.fixture()
        f["proxy_target"]=[0,1,1]
        result=selection_stats(f)
        self.assertEqual(result["selected_episodes"],1)
        self.assertEqual(result["proxy_hit_rate"],0)

    def test_unresolved_first_choice_is_not_dropped_from_denominator(self):
        frame=pd.DataFrame([dict(code="000017",signal_date="2026-01-05",episode_id="a",proxy_target=None),
                            dict(code="000018",signal_date="2026-01-06",episode_id="b",proxy_target=1)])
        result=selection_stats(frame)
        self.assertEqual(result["selected_episodes"],2)
        self.assertIsNone(result["proxy_hit_rate"])
        self.assertEqual(result["proxy_hit_rate_lower"],.5)
        self.assertEqual(result["proxy_hit_rate_upper"],1)


if __name__ == "__main__":
    unittest.main()
