import unittest

from app.analytics.density import DensityClassifier


class TestDensityClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = DensityClassifier()

    def test_low_density(self):
        self.assertEqual(self.classifier.classify(0), "LOW")
        self.assertEqual(self.classifier.classify(10), "LOW")

    def test_medium_density(self):
        self.assertEqual(self.classifier.classify(11), "MEDIUM")
        self.assertEqual(self.classifier.classify(25), "MEDIUM")

    def test_high_density(self):
        self.assertEqual(self.classifier.classify(26), "HIGH")
        self.assertEqual(self.classifier.classify(100), "HIGH")

    def test_negative_count_rejected(self):
        with self.assertRaises(ValueError):
            self.classifier.classify(-1)

    def test_non_integer_count_rejected(self):
        for value in (1.5, "10", True):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    self.classifier.classify(value)

    def test_invalid_thresholds_rejected(self):
        with self.assertRaises(ValueError):
            DensityClassifier(low_max=25, medium_max=10)


if __name__ == "__main__":
    unittest.main()
