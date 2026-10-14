import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from relay_price_shape_research import shape, attach, claim_once


class PriceShapeTests(unittest.TestCase):
    def row(self, **changes):
        source = {"date": "2026-08-03", "code": "600001", "open": 10., "high": 10., "low": 10., "close": 10., **changes}
        return {"date": "2026-08-03", "code": "600001", "original_price": 10.,
            "sources": [source], "provisional_one_price_session": True, "training_truth_authorized": False}

    def test_pattern_and_one_cent_conflict(self):
        self.assertIs(shape(self.row()), True)
        self.assertIs(shape(self.row(low=9.99, open=9.99)), False)
        self.assertIsNone(shape(self.row(high=10.01)))

    def test_target_day_rejected(self):
        with self.assertRaisesRegex(ValueError, "target-day"):
            shape(self.row(date="2026-08-04"))

    def test_disagreement_and_missing_not_imputed_from_proxy(self):
        row = self.row()
        row["sources"].append(self.row(low=9.99)["sources"][0])
        self.assertIsNone(shape(row))
        row["sources"] = []
        self.assertIsNone(shape(row))

    def test_changed_claim_or_training_authorization_rejected(self):
        full = pd.DataFrame([{"date": "2026-08-03", "code": "600001", "price": 10., "signal_one_word": 0., "target": 1.}])
        for change in [{"provisional_one_price_session": False}, {"training_truth_authorized": True}]:
            row = {**self.row(), **change}
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "authorization"):
                attach(full, {"rows": [row]})
        result = attach(full, {"rows": [self.row()]})
        self.assertEqual(result.signal_one_word.iloc[0], 1.)
        self.assertEqual(result.target.iloc[0], full.target.iloc[0])

    def test_deleted_candidate_rejected(self):
        full = pd.DataFrame([{"date": "2026-08-03", "code": "600001", "price": 10., "signal_one_word": 0.}])
        with self.assertRaisesRegex(ValueError, "complete"):
            attach(full, {"rows": []})

    def test_same_context_lock_survives_protocol_copy_and_new_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            facts = root / "facts.json"
            original, copy, receipt = root / "protocol.json", root / "copy.json", root / "receipt.json"
            original.write_text(json.dumps({"context_path": str(facts)}))
            copy.write_text(original.read_text())
            receipt.write_text("{}")
            claim_once(original, receipt, root / "first")
            with self.assertRaises(FileExistsError):
                claim_once(copy, receipt, root / "second")


if __name__ == "__main__":
    unittest.main()
