import unittest

import numpy as np

from sevir_world.domain.vil_palette import (
    CASCAST_VIL_LEVELS,
    CASCAST_VIL_COLORS_RGB,
    colorize_vil,
)


class VilPaletteTests(unittest.TestCase):
    def test_palette_has_cascast_bins(self):
        self.assertEqual(CASCAST_VIL_LEVELS.tolist(), [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0])
        self.assertEqual(CASCAST_VIL_COLORS_RGB.shape, (10, 3))

    def test_colorization_uses_discrete_bins_and_black_missing(self):
        values = np.asarray([[0, 16, 74, 133, 219, 254, 255]], dtype=np.float32)
        result = colorize_vil(values)
        self.assertTrue(np.array_equal(result[0, 0], [77, 77, 77]))
        self.assertTrue(np.array_equal(result[0, 1], [40, 190, 40]))
        self.assertTrue(np.array_equal(result[0, 4], [231, 0, 255]))
        self.assertTrue(np.array_equal(result[0, 5], [231, 0, 255]))
        self.assertTrue(np.array_equal(result[0, 6], [0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
