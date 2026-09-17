import unittest

from filedecay.config import Thresholds
from filedecay.decay import decide_from_active, should_delete, should_progress_to_quarantine


class TestDecideFromActive(unittest.TestCase):
    def setUp(self):
        self.t = Thresholds(
            compress_after_days=7,
            archive_after_days=21,
            quarantine_after_days=45,
            delete_after_days=15,
        )

    def test_stays_active_when_fresh(self):
        self.assertIsNone(decide_from_active(1, self.t))
        self.assertIsNone(decide_from_active(6.9, self.t))

    def test_compresses_past_compress_threshold(self):
        self.assertEqual(decide_from_active(7, self.t), "compressed")
        self.assertEqual(decide_from_active(20, self.t), "compressed")

    def test_archives_past_archive_threshold(self):
        self.assertEqual(decide_from_active(21, self.t), "archived")
        self.assertEqual(decide_from_active(44, self.t), "archived")

    def test_quarantines_past_quarantine_threshold(self):
        self.assertEqual(decide_from_active(45, self.t), "quarantined")
        self.assertEqual(decide_from_active(1000, self.t), "quarantined")

    def test_jumps_straight_to_deepest_stage(self):
        # a file nobody scanned for a year should not have to crawl through
        # compress -> archive -> quarantine one scan at a time
        self.assertEqual(decide_from_active(365, self.t), "quarantined")


class TestQuarantineProgression(unittest.TestCase):
    def setUp(self):
        self.t = Thresholds(
            compress_after_days=7,
            archive_after_days=21,
            quarantine_after_days=45,
            delete_after_days=15,
        )

    def test_should_delete(self):
        self.assertFalse(should_delete(14.9, self.t))
        self.assertTrue(should_delete(15, self.t))

    def test_should_progress_to_quarantine(self):
        self.assertFalse(should_progress_to_quarantine(44.9, self.t))
        self.assertTrue(should_progress_to_quarantine(45, self.t))


if __name__ == "__main__":
    unittest.main()
