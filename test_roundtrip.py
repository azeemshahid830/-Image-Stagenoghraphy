"""Small, dependency-light roundtrip checks. Run: python -m unittest discover -s tests"""
import unittest
import numpy as np
from run_experiment import embed, extract, forest_spanning_path


class RoundTripTests(unittest.TestCase):
    def test_exact_recovery_and_bounded_pixel_change(self):
        rng = np.random.default_rng(123)
        cover = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
        bits = rng.integers(0, 2, 1200, dtype=np.uint8)
        path = forest_spanning_path(20, 30)
        stego, usage = embed(cover, bits, path)
        extracted = extract(stego, len(bits), path)
        np.testing.assert_array_equal(extracted, bits)
        self.assertEqual(sum(usage.values()), len(bits))
        delta = np.abs(stego.astype(np.int16) - cover.astype(np.int16))
        self.assertLessEqual(int(delta.max()), 1)

    def test_single_channel(self):
        cover = np.full((8, 8, 3), 255, dtype=np.uint8)
        bits = np.arange(40, dtype=np.uint8) % 2
        path = forest_spanning_path(8, 8)
        stego, _ = embed(cover, bits, path)
        np.testing.assert_array_equal(extract(stego, len(bits), path), bits)
        np.testing.assert_array_equal(stego[:, :, 1:], cover[:, :, 1:])


if __name__ == '__main__':
    unittest.main()
