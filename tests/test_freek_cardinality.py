from __future__ import annotations

import unittest

from hypersphere_dpp.model import HypersphereFreeKModel


class FreeKCardinalityTest(unittest.TestCase):
    def test_hysteresis_keeps_bank_size_for_small_ema_fluctuations(self):
        ema, stable_k = HypersphereFreeKModel._stabilize_cardinality(
            raw_estimated_k=48,
            previous_ema=50.0,
            previous_k=50,
            ema_momentum=0.9,
            hysteresis=2,
            min_clusters=2,
            max_clusters=100,
            num_samples=256,
            positive_eigenvalues=120,
        )
        self.assertAlmostEqual(ema, 49.8)
        self.assertEqual(stable_k, 50)

    def test_hysteresis_allows_a_persistent_large_change(self):
        ema, stable_k = HypersphereFreeKModel._stabilize_cardinality(
            raw_estimated_k=70,
            previous_ema=70.0,
            previous_k=50,
            ema_momentum=0.9,
            hysteresis=2,
            min_clusters=2,
            max_clusters=100,
            num_samples=256,
            positive_eigenvalues=120,
        )
        self.assertEqual(ema, 70.0)
        self.assertEqual(stable_k, 70)


if __name__ == "__main__":
    unittest.main()
