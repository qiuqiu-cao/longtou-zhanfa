import unittest
from copy import deepcopy
from relay_cohort_output import summarize


def row(code, score, boards=2, call="UNKNOWN", actual=0):
    return dict(code=code, name=code, boards=boards, raw_score=score,
                model_positive=score>.5, independent_call=call, actual_one_word_daily=actual)


class CohortTests(unittest.TestCase):
    def test_all_negative_first_does_not_become_positive(self):
        group=summarize([row("000001", .49, call="PROMOTE"), row("000002", .2)])[0]
        self.assertTrue(group["all_model_negative"])
        self.assertFalse(group["rank_first_original_model_positive"])
        self.assertIsNone(group["nonword_positive_first"])
        self.assertTrue(group["members"][0]["independent_conflict"])

    def test_complete_groups_and_fixed_tie_order(self):
        values=[row("000002", .4), row("000001", .4), row("000003", .8, boards=3)]
        before=deepcopy(values); groups=summarize(values)
        self.assertEqual(groups[0]["boards"],3)
        self.assertEqual([r["code"] for r in groups[1]["members"]],["000001","000002"])
        self.assertEqual(sum(g["candidate_count"] for g in groups),3)
        self.assertEqual(values,before)

    def test_true_identity_and_unknown_distinct(self):
        group=summarize([row("000001",.9,actual=1),row("000002",.8,actual=None),row("000003",.7)])[0]
        self.assertEqual(group["relative_rank_first"],"000001")
        self.assertEqual(group["nonword_positive_first"],"000003")

    def test_singleton_keeps_failure_path_and_no_score_gap(self):
        group=summarize([row("000001",.6)])[0]
        self.assertIsNone(group["top_two_original_display_gap"])
        self.assertIn("JOINT_FAILURE",group["paths_retained"])
        self.assertFalse(group["entry_selection_authorized"])

    def test_duplicate_and_changed_model_call_rejected(self):
        with self.assertRaisesRegex(ValueError,"duplicate"):
            summarize([row("000001",.6),row("000001",.6)])
        r=row("000001",.49); r["model_positive"]=True
        with self.assertRaisesRegex(ValueError,"direction"):
            summarize([r])


if __name__ == "__main__": unittest.main()
