import unittest

from traffic_analysis.helmet_rider import RiderTracker, iou


class RiderTrackerTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_stable_key_for_moving_rider_and_new_key_for_other(self):
        t = RiderTracker(max_age=6)
        (k1,) = t.match([(100, 100, 200, 300)], 3)
        t.record(k1, 5, (100, 100, 200, 300), (105, 95, 195, 190), "helmet", 3)
        k2, k3 = t.match([(104, 102, 204, 302), (600, 100, 700, 300)], 6)
        self.assertEqual(k2, k1)
        self.assertNotEqual(k3, k1)

    def test_key_not_reused_after_expiry(self):
        t = RiderTracker(max_age=6)
        (k1,) = t.match([(100, 100, 200, 300)], 3)
        t.record(k1, 5, (100, 100, 200, 300), (105, 95, 195, 190), "helmet", 3)
        (k2,) = t.match([(100, 100, 200, 300)], 40)
        self.assertNotEqual(k1, k2)

    def test_helmet_shown_by_majority_unknown_never_shown(self):
        t = RiderTracker(max_age=6)
        head = (105, 95, 195, 190)
        t.record(1, 5, (100, 100, 200, 300), head, "unknown", 3)
        self.assertEqual(t.visible(3), [])
        t.record(1, 5, (100, 100, 200, 300), head, "helmet", 6)
        self.assertEqual([v[2] for v in t.visible(6)], ["helmet"])

    def test_unconfirmed_no_helmet_not_shown_confirmed_is(self):
        t = RiderTracker(max_age=6)
        head = (105, 95, 195, 190)
        t.record(1, 5, (100, 100, 200, 300), head, "no_helmet", 3)
        self.assertEqual(t.visible(3), [])
        t.mark_confirmed(1)
        self.assertEqual([v[2] for v in t.visible(3)], ["no_helmet"])

    def test_stale_rider_not_shown(self):
        t = RiderTracker(max_age=6)
        t.record(1, 5, (100, 100, 200, 300), (105, 95, 195, 190), "helmet", 3)
        self.assertEqual(t.visible(20), [])


if __name__ == "__main__":
    unittest.main()
