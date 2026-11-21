import itertools
import unittest

import numpy as np

from analysis_common import align_predictions, counts
from paired_significance import aggregate_clusters, holm, paired_randomization


class StatisticsTests(unittest.TestCase):
    def test_exact_matches_independent_enumeration(self):
        d = [5, -2, 1, 8]
        expected = sum(abs(sum(x*y for x,y in zip(signs,d))) >= abs(sum(d))
                       for signs in itertools.product([-1,1], repeat=4)) / 16
        p, _, n, method = paired_randomization(d, 16, 7)
        self.assertEqual((p,n,method), (expected,16,"exact"))
        self.assertEqual(p, paired_randomization([-x for x in d],16,7)[0])

    def test_no_difference_and_single_cluster(self):
        self.assertEqual(paired_randomization([0,0,0],100,0)[0],1)
        self.assertEqual(paired_randomization([200],100,0)[0],1)

    def test_monte_carlo_plus_one_and_reproducibility(self):
        a = paired_randomization([1]*30, 99, 42)
        self.assertEqual(a, paired_randomization([1]*30,99,42))
        self.assertEqual(a[0],(a[1]+1)/100)
        self.assertGreater(a[0],0)

    def test_holm_known_values_and_ties(self):
        np.testing.assert_allclose(holm([.01,.04,.03,.002]), [.03,.06,.06,.008])
        np.testing.assert_allclose(holm([.02,.02,1]), [.06,.06,1])

    def test_cluster_keeps_related_clips_together(self):
        manifest=[dict(text="same",speaker="a"),dict(text="same",speaker="b"),dict(text="other",speaker="a")]
        np.testing.assert_array_equal(aggregate_clusters(manifest,[2,3,10],"test_cv"),[12,3])
        np.testing.assert_array_equal(aggregate_clusters(manifest,[2,3,10],"test_fleurs"),[5,10])

    def test_denominators_empty_hypothesis_and_weighting(self):
        a=counts("one two three",""); b=counts("four","four")
        self.assertEqual(a[:2],(3,3))
        self.assertEqual(a[2:4],(13,13))
        self.assertEqual((a[0]+b[0])/(a[1]+b[1]),.75)

    def test_alignment_checks_and_reorders(self):
        m=[dict(id="1",text="a"),dict(id="2",text="b")]
        p=[dict(id="2",reference="b",raw_hypothesis="B!",hypothesis="b"),
           dict(id="1",reference="a",raw_hypothesis="",hypothesis="")]
        self.assertEqual([r["id"] for r in align_predictions(m,p,"test")],["1","2"])
        for bad in (p[:1],p+[p[0]], [dict(p[0],reference="wrong"),p[1]]):
            with self.assertRaises(ValueError):align_predictions(m,bad,"test")


if __name__ == "__main__":
    unittest.main()
