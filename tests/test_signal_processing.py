import unittest

import numpy as np

from src import uav_acoustic_direction_finding as doa


class AngleUtilityTests(unittest.TestCase):
    def test_wrap180_maps_angles_to_expected_range(self):
        values = np.array([-540.0, -181.0, -180.0, 0.0, 180.0, 181.0, 540.0])
        expected = np.array([-180.0, 179.0, -180.0, 0.0, -180.0, -179.0, -180.0])
        np.testing.assert_allclose(doa.wrap180(values), expected)

    def test_angular_distance_crosses_boundary(self):
        self.assertAlmostEqual(float(doa.ang_dist_deg(179.0, -179.0)), 2.0)

    def test_direction_vectors_follow_documented_convention(self):
        np.testing.assert_allclose(doa.direction_unit_vector(0.0), [0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(doa.direction_unit_vector(90.0), [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(doa.direction_unit_vector(180.0), [0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(doa.direction_unit_vector(270.0), [-1.0, 0.0], atol=1e-12)


class InputValidationTests(unittest.TestCase):
    def test_select_mics_returns_four_requested_channels(self):
        samples = np.arange(60, dtype=np.float32).reshape(10, 6)
        selected = doa.select_mics(samples, [1, 2, 3, 4])
        np.testing.assert_array_equal(selected, samples[:, [1, 2, 3, 4]])

    def test_select_mics_rejects_insufficient_channels(self):
        samples = np.zeros((10, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            doa.select_mics(samples, [0, 1, 2, 3])


class ConfigurationTests(unittest.TestCase):
    def test_circle_configuration(self):
        doa.configure_mode("circle")
        self.assertEqual(doa.MODE, "circle")
        self.assertAlmostEqual(doa.SNAPSHOT_SEC, 0.05)

    def test_hover_configuration(self):
        doa.configure_mode("hover")
        self.assertEqual(doa.MODE, "hover")
        self.assertAlmostEqual(doa.SNAPSHOT_SEC, 0.10)

    def tearDown(self):
        doa.configure_mode("circle")


if __name__ == "__main__":
    unittest.main()
