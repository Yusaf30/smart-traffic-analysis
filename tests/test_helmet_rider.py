import unittest

from traffic_analysis.helmet_rider import (
    RiderConfig,
    ViolationConfirmer,
    associate,
    classify_head,
    head_region,
)

CFG = RiderConfig()

# Real boxes copied from diag_helmet/rider_association.csv (1280x720 frames).
F189_AUTO_PERSON = (118.39, 258.65, 217.09, 349.34)   # auto-rickshaw passenger
F189_AUTO_BIKE = (16.88, 286.99, 309.27, 482.19)      # box COCO calls "motorcycle"
F237_AUTO_PERSON = (525.29, 253.91, 565.85, 316.54)
F237_AUTO_BIKE = (508.99, 305.88, 698.49, 427.86)
F189_RIDER1_PERSON = (290.84, 237.98, 445.68, 481.18)
F189_RIDER1_BIKE = (220.47, 337.18, 559.05, 518.26)
F189_RIDER2_PERSON = (682.12, 240.93, 779.76, 414.56)
F189_RIDER2_BIKE = (684.82, 312.21, 921.85, 481.88)


class AssociationTests(unittest.TestCase):
    def test_f189_auto_passenger_rejected(self):
        (a,) = associate([F189_AUTO_PERSON], [F189_AUTO_BIKE], CFG)
        self.assertFalse(a.accepted)
        self.assertEqual(a.reason, "person_ends_above_vehicle")

    def test_f237_auto_passenger_rejected(self):
        (a,) = associate([F237_AUTO_PERSON], [F237_AUTO_BIKE], CFG)
        self.assertFalse(a.accepted)
        self.assertEqual(a.reason, "person_ends_above_vehicle")

    def test_real_riders_accepted(self):
        for person, bike in (
            (F189_RIDER1_PERSON, F189_RIDER1_BIKE),
            (F189_RIDER2_PERSON, F189_RIDER2_BIKE),
        ):
            (a,) = associate([person], [bike], CFG)
            self.assertTrue(a.accepted, a)
            self.assertEqual(a.bike_idx, 0)

    def test_rider_next_to_auto_in_same_frame(self):
        persons = [F189_AUTO_PERSON, F189_RIDER1_PERSON]
        bikes = [F189_AUTO_BIKE, F189_RIDER1_BIKE]
        res = associate(persons, bikes, CFG)
        self.assertFalse(res[0].accepted)
        self.assertTrue(res[1].accepted)
        self.assertEqual(res[1].bike_idx, 1)

    def test_no_bike_rejected(self):
        (a,) = associate([F189_RIDER1_PERSON], [], CFG)
        self.assertFalse(a.accepted)
        self.assertEqual(a.reason, "no_bike_detected")

    def test_ambiguous_between_two_similar_bikes(self):
        person = (100, 100, 160, 260)
        bikes = [(90, 200, 170, 300), (95, 205, 175, 305)]
        (a,) = associate([person], bikes, CFG)
        self.assertFalse(a.accepted)
        self.assertEqual(a.reason, "ambiguous_multiple_bikes")

    def test_over_capacity(self):
        bike = (100, 200, 300, 300)
        persons = [
            (180, 100, 220, 260),
            (150, 100, 190, 260),
            (210, 100, 250, 260),
            (240, 100, 280, 260),
        ]
        res = associate(persons, [bike], CFG)
        self.assertEqual(sum(a.accepted for a in res), 3)
        self.assertEqual(res[3].reason, "over_capacity")

    def test_rider_head_lies_above_motorcycle_box(self):
        # Root cause of the crop approach: real riders' heads are above the bike box.
        helmet_cy_rider1 = 259.3   # CSV: With Helmet at (391.2, 259.3)
        helmet_cy_rider2 = 251.4   # CSV: With Helmet at (733.6, 251.4)
        self.assertLess(helmet_cy_rider1, F189_RIDER1_BIKE[1])
        self.assertLess(helmet_cy_rider2, F189_RIDER2_BIKE[1])


class HeadTests(unittest.TestCase):
    def test_head_region_contains_known_helmet_center(self):
        x1, y1, x2, y2 = head_region(F189_RIDER1_PERSON, CFG)
        self.assertTrue(x1 <= 391.2 <= x2 and y1 <= 259.3 <= y2)

    def test_classify_no_helmet(self):
        head = head_region(F189_RIDER1_PERSON, CFG)
        dets = [("Without Helmet", 0.80, (370, 240, 410, 280))]
        self.assertEqual(classify_head(dets, head, CFG)[0], "no_helmet")

    def test_classify_helmet(self):
        head = head_region(F189_RIDER1_PERSON, CFG)
        dets = [("With Helmet", 0.60, (370, 240, 410, 280))]
        self.assertEqual(classify_head(dets, head, CFG)[0], "helmet")

    def test_classify_conflict_is_unknown(self):
        head = head_region(F189_RIDER1_PERSON, CFG)
        dets = [
            ("With Helmet", 0.60, (370, 240, 410, 280)),
            ("Without Helmet", 0.55, (372, 242, 408, 278)),
        ]
        self.assertEqual(classify_head(dets, head, CFG)[0], "unknown")

    def test_detection_outside_head_ignored(self):
        head = head_region(F189_RIDER1_PERSON, CFG)
        dets = [("Without Helmet", 0.90, (900, 240, 940, 280))]
        self.assertEqual(classify_head(dets, head, CFG)[0], "unknown")


class ConfirmerTests(unittest.TestCase):
    def test_confirms_once(self):
        c = ViolationConfirmer(CFG)
        results = [c.update(7, "no_helmet") for _ in range(6)]
        self.assertEqual(results, [False, False, True, False, False, False])

    def test_unknown_and_single_observation_never_confirm(self):
        c = ViolationConfirmer(CFG)
        self.assertFalse(c.update(1, "no_helmet"))
        for _ in range(10):
            self.assertFalse(c.update(1, "unknown"))
        self.assertNotIn(1, c.reported)

    def test_mixed_evidence_does_not_confirm(self):
        c = ViolationConfirmer(CFG)
        seq = ["no_helmet", "helmet", "no_helmet", "helmet", "no_helmet", "helmet"]
        self.assertFalse(any(c.update(2, s) for s in seq))

    def test_keys_are_independent(self):
        c = ViolationConfirmer(CFG)
        for _ in range(3):
            c.update(1, "no_helmet")
        self.assertIn(1, c.reported)
        self.assertNotIn(2, c.reported)


if __name__ == "__main__":
    unittest.main()
