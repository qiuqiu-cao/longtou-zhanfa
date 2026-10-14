import unittest
from datetime import date,timedelta

from complete_trade_split_audit import audit


class CompleteTradeSplitTests(unittest.TestCase):
    def sessions(self,n=420):
        start=date(2025,1,1)
        return [(start+timedelta(days=i)).isoformat() for i in range(n)]

    def split(self,s):
        return dict(train_start=s[0],train_end=s[149],calibration_start=s[151],calibration_end=s[210],
                    validation_start=s[212],validation_end=s[271],blind_start=s[273],blind_end=s[419],
                    embargo_sessions=1)

    def row(self,i,signal,closed,profit=True,large=False):
        return dict(decision_id="d"+str(i),signal_date=signal,closed_at=closed+"T15:00:00+08:00",
                    training_eligible=True,complete_trade_profit_label=profit,large_loss_label=large)

    def dataset(self,rows):
        return dict(generalization_scope="FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY",training_ready=True,rows=rows)

    def test_cross_boundary_holding_is_purged(self):
        s=self.sessions()
        rows=[self.row(1,s[149],s[151]),self.row(2,s[148],s[149])]
        result=audit(self.dataset(rows),s,self.split(s),s[-1]+"T16:00:00+08:00")
        self.assertEqual(result["purged"]["train"],["d1"])
        self.assertEqual(result["partitions"]["train"],["d2"])

    def test_embargo_rows_are_never_assigned(self):
        s=self.sessions()
        result=audit(self.dataset([self.row(1,s[150],s[150])]),s,self.split(s),s[-1]+"T16:00:00+08:00")
        self.assertEqual(result["embargo_row_count"],1)
        self.assertTrue(all(not values for values in result["partitions"].values()))

    def test_blind_count_is_visible_but_labels_are_not_summarized(self):
        s=self.sessions()
        rows=[self.row(i,s[273+i],s[273+i],profit=bool(i%2),large=not bool(i%3)) for i in range(100)]
        result=audit(self.dataset(rows),s,self.split(s),s[-1]+"T16:00:00+08:00")
        self.assertEqual(result["blind_count"],100)
        self.assertIsNone(result["blind_label_metrics"])
        self.assertTrue(result["blind_labels_not_read_by_audit"])
        self.assertEqual(result["blind_label_access_control"],"NOT_ENFORCED_SOURCE_DATASET_CONTAINS_LABELS")

    def test_insufficient_embargo_and_future_blind_rejected(self):
        s=self.sessions()
        split=self.split(s)
        split["calibration_start"]=s[150]
        with self.assertRaisesRegex(ValueError,"embargo"):
            audit(self.dataset([]),s,split,s[-1]+"T16:00:00+08:00")
        with self.assertRaisesRegex(ValueError,"fully observable"):
            audit(self.dataset([]),s,self.split(s),s[-2]+"T16:00:00+08:00")

    def test_unlabeled_or_ineligible_rows_do_not_enter_partitions(self):
        s=self.sessions()
        row=self.row(1,s[10],s[11])
        row["training_eligible"]=False
        result=audit(self.dataset([row]),s,self.split(s),s[-1]+"T16:00:00+08:00")
        self.assertEqual(sum(result["kept_counts"].values()),0)

    def test_upstream_not_ready_prevents_model_development(self):
        s=self.sessions()
        dataset=self.dataset([])
        dataset["training_ready"]=False
        result=audit(dataset,s,self.split(s),s[-1]+"T16:00:00+08:00")
        self.assertEqual(result["status"],"DIAGNOSTIC_ONLY")
        self.assertIn("UPSTREAM_COMPLETE_TRADE_DATASET_NOT_READY",result["issues"])


if __name__=="__main__": unittest.main()
