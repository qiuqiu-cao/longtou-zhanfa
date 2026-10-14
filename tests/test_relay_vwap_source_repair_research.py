import unittest
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from relay_vwap_source_repair_research import make,spec,FIELD
from current_model_diagnostics import RELAY_FIELDS


class SourceRepairIntegrityTests(unittest.TestCase):
    def test_original_transform_and_missingness_exactly_preserved(self):
        x=pd.DataFrame(np.arange(29*8,dtype=float).reshape(8,29),columns=RELAY_FIELDS)
        x.loc[0,"turnover"]=np.nan
        original=StandardScaler().fit_transform(SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True).fit_transform(x))
        x[FIELD]=[.01,np.nan,.03,.02,.04,.06,.05,.01]
        transformed=make()[0].fit_transform(x)
        np.testing.assert_array_equal(transformed[:,:original.shape[1]],original)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transformed.shape[1],original.shape[1]+2)

    def test_evaluation_values_cannot_refit_training_preprocessing(self):
        x=pd.DataFrame(np.ones((5,29)),columns=RELAY_FIELDS);x[FIELD]=np.arange(5)/100
        prep=make()[0].fit(x)
        a=prep.transform(x.iloc[:1])
        extreme=x.iloc[:1].copy();extreme[FIELD]=1e6
        np.testing.assert_array_equal(prep.transform(pd.concat([x.iloc[:1],extreme]))[:1],a)

    def test_fixed_repair_definition_keeps_old_failure_and_no_formal_source_claim(self):
        d=spec();self.assertTrue(d["old_V294_failure_preserved"])
        self.assertFalse(d["deployment_authorized"])
        self.assertEqual(d["maximum_fits"],3)
        self.assertEqual(d["BASE_refits"],0)
        self.assertNotIn("target",d["fields"])
        self.assertIn("strict original-time source qualification NOT passed",d["source_policy"])


if __name__=="__main__":unittest.main()
