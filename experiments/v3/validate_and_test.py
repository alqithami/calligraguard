#!/usr/bin/env python3
"""Regression tests are separate from the full-data integrity gate and scientific results."""
import argparse, json, unittest
from pathlib import Path
import numpy as np
import benchmark_common as b

class EvaluatorTests(unittest.TestCase):
    def normal(self,n=200):
        return [{'id':str(i),'parent_id':str(i),'family_group':'calibration_source','source_family':'calibration_source','label':0} for i in range(n)]
    def test_small_calibration_abstains(self):
        threshold,n=b.calibration_threshold(self.normal(20),np.arange(20.),.01)
        self.assertIsNone(threshold);self.assertEqual(n,20)
    def test_parent_maximum_not_number_of_views(self):
        rows=self.normal();scores=list(np.arange(200.))
        rows += [{**r,'id':r['id']+'b'} for r in self.normal()];scores+=list(np.arange(200.)+1)
        threshold,n=b.calibration_threshold(rows,np.asarray(scores),.01)
        self.assertEqual(n,200);self.assertEqual(threshold,199.)
    def test_missing_calibration_prediction_rejected(self):
        with self.assertRaises(ValueError):b.calibration_threshold(self.normal(),np.zeros(199),.01)
    def test_nonfinite_rejected(self):
        scores=np.zeros(200);scores[0]=np.nan
        with self.assertRaises(ValueError):b.calibration_threshold(self.normal(),scores,.01)
    def test_duplicate_rejected(self):
        rows=self.normal();rows[1]['id']=rows[0]['id']
        with self.assertRaises(ValueError):b.calibration_threshold(rows,np.zeros(200),.01)
    def test_defective_calibration_rejected(self):
        rows=self.normal();rows[0]['label']=1
        with self.assertRaises(ValueError):b.calibration_threshold(rows,np.zeros(200),.01)
    def test_strict_threshold_ties(self):
        cal=self.normal();rows=[{'id':'e0','parent_id':'e0','source_family':'evaluation_source','family_group':'evaluation_source','label':0,'type':'clean'},
                               {'id':'e1','parent_id':'e1','source_family':'evaluation_source','family_group':'evaluation_source','label':1,'type':'spur'}]
        report=b.detection_metrics(cal,np.zeros(200),rows,[0.,1.],bootstrap=False)
        self.assertEqual(report['operating_points']['0.01']['actual_image_fpr'],0.)
        self.assertEqual(report['operating_points']['0.01']['recall'],1.)
    def test_family_overlap_rejected(self):
        cal=self.normal();rows=[{**cal[0],'type':'clean'},{**cal[1],'label':1,'type':'spur'}]
        with self.assertRaises(ValueError):b.detection_metrics(cal,np.zeros(200),rows,[0.,1.],bootstrap=False)
    def test_test_access_guard(self):
        with self.assertRaises(ValueError):b.BenchmarkData('/nonexistent','test_family')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EvaluatorTests))
    if not result.wasSuccessful():raise SystemExit(1)
    report=b.validate_dataset(a.root,a.out);report['evaluator_regression_tests_passed']=result.testsRun;b.dump(a.out,report)
    print('FULL_DATA_AND_EVALUATOR_GATE_PASS',report['primary'],flush=True)
