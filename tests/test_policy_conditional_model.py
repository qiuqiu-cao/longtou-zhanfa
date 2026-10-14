import copy
import unittest

from policy_conditional_model import train


class PolicyConditionalModelTests(unittest.TestCase):
    def fixture(self):
        partitions={"train":[],"calibration":[],"validation":[],"blind":[]}
        rows=[]
        counts={"train":100,"calibration":30,"validation":30,"blind":100}
        index=0
        for segment,count in counts.items():
            for offset in range(count):
                decision_id=f"d{index}"
                partitions[segment].append(decision_id)
                profit=bool(offset%2)
                large=bool(offset%5==0)
                rows.append(dict(decision_id=decision_id,training_eligible=True,
                                 complete_trade_profit_label=profit,large_loss_label=large,
                                 signal_date=f"2026-01-{(index%28)+1:02d}",
                                 board_group="FIRST_BOARD" if index%2 else "CONSECUTIVE_BOARD",
                                 emotion_active=True,complete_trade_net_return=.02 if profit else -.08 if large else -.01,
                                 selected_features={"x":float(offset%7),"z":float(profit)-float(large)}))
                index+=1
        blind_ids=list(partitions.pop("blind"))
        dataset=dict(training_ready=True,generalization_scope="FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY",
                     feature_allowlist=["x","z"],policy_id="p1",blind_rows_removed=True,
                     blind_label_fields_present=False,
                     rows=[row for row in rows if row["decision_id"] not in blind_ids])
        split=dict(status="READY_FOR_MODEL_DEVELOPMENT",blind_labels_not_read_by_audit=True,partitions=partitions,
                   blind_access_control="SEPARATE_FEATURE_AND_LABEL_VAULT",
                   source_dataset_with_blind_labels_not_required_for_training=True,
                   blind_count=len(blind_ids),blind_decision_ids_sha256="a"*64,
                   blind_labels_sha256_commitment="b"*64)
        return dataset,split

    def test_trainer_uses_development_only_and_reports_no_blind_metrics(self):
        dataset,split=self.fixture()
        card,bundle=train(dataset,split)
        self.assertEqual(card["status"],"DEVELOPMENT_MODEL_ONLY")
        self.assertFalse(card["deployment_allowed"])
        self.assertIsNone(card["blind_metrics"])
        self.assertFalse(bundle["blind_feature_matrix_built"])
        self.assertFalse(bundle["source_payload_including_blind_rows_parsed"])
        self.assertTrue(bundle["out_of_distribution_gate_frozen"])
        self.assertEqual(len(bundle["trained_decision_ids"]),160)
        self.assertFalse(card["can_rank_unselected_full_pool"])

    def test_model_choice_uses_validation_and_no_threshold(self):
        card,_=train(*self.fixture())
        self.assertIn(card["profit_model"]["chosen"],{"regularized_logistic","hist_gradient_boosting"})
        self.assertEqual(card["profit_model"]["candidates"][card["profit_model"]["chosen"]]["n"],30)
        self.assertFalse(card["threshold_selected"])

    def test_blind_overlap_is_rejected(self):
        dataset,split=self.fixture()
        split["partitions"]["blind"]=[split["partitions"]["train"][0]]
        with self.assertRaisesRegex(ValueError,"must not expose"):
            train(dataset,split)

    def test_future_or_missing_feature_matrix_is_rejected(self):
        dataset,split=self.fixture()
        dataset["rows"][0]["selected_features"].pop("x")
        with self.assertRaisesRegex(ValueError,"feature matrix"):
            train(dataset,split)

    def test_unready_upstream_is_rejected(self):
        dataset,split=self.fixture()
        dataset["training_ready"]=False
        with self.assertRaisesRegex(ValueError,"not ready"):
            train(dataset,split)


if __name__=="__main__": unittest.main()
