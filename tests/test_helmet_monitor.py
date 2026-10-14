import unittest

import numpy as np

from traffic_analysis.helmet_monitor import HelmetRiderMonitor


class _V:
    def __init__(self, v):
        self.v = v

    def tolist(self):
        return list(self.v)

    def item(self):
        return self.v


class _Box:
    def __init__(self, xyxy, cls=0, conf=0.9):
        self.xyxy = [_V(xyxy)]
        self.cls = _V(cls)
        self.conf = _V(conf)


class _Res:
    def __init__(self, boxes):
        self.boxes = boxes


class FakePersonModel:
    def __init__(self, persons):
        self.persons = persons
        self.calls = 0

    def predict(self, frame, **kw):
        self.calls += 1
        return [_Res([_Box(p) for p in self.persons])]


class FakeHelmetModel:
    """Always reports one detection in the middle of whatever crop it receives."""
    names = {0: "With Helmet", 1: "Without Helmet"}

    def __init__(self, cls, conf):
        self.cls, self.conf, self.calls = cls, conf, 0

    def predict(self, crop, **kw):
        self.calls += 1
        h, w = crop.shape[:2]
        return [_Res([_Box((w / 2 - 10, h / 2 - 10, w / 2 + 10, h / 2 + 10), self.cls, self.conf)])]


FRAME = np.zeros((720, 1280, 3), np.uint8)
# Real boxes from rider_association.csv, frame 189.
AUTO_PASSENGER = (118.39, 258.65, 217.09, 349.34)
AUTO_BIKE = (16.88, 286.99, 309.27, 482.19)
RIDER = (290.84, 237.98, 445.68, 481.18)
RIDER_BIKE = (220.47, 337.18, 559.05, 518.26)


class MonitorTests(unittest.TestCase):
    def test_auto_passenger_never_checked_or_flagged(self):
        helmet = FakeHelmetModel(cls=1, conf=0.95)   # screams "Without Helmet"
        mon = HelmetRiderMonitor(helmet, FakePersonModel([AUTO_PASSENGER]))
        confirmed = []
        for f in (3, 6, 9, 12):
            confirmed += mon.update(FRAME, [(3, AUTO_BIKE)], f)
        self.assertEqual(confirmed, [])
        self.assertEqual(helmet.calls, 0)
        canvas = FRAME.copy()
        mon.draw(canvas, 12)
        self.assertEqual(int(canvas.sum()), 0)       # nothing drawn

    def test_real_rider_without_helmet_confirmed_once_and_drawn(self):
        helmet = FakeHelmetModel(cls=1, conf=0.90)
        mon = HelmetRiderMonitor(helmet, FakePersonModel([RIDER]))
        results = [mon.update(FRAME, [(7, RIDER_BIKE)], f) for f in (3, 6, 9, 12, 15)]
        self.assertEqual(results, [[], [], [7], [], []])
        canvas = FRAME.copy()
        mon.draw(canvas, 15)
        self.assertGreater(int(canvas[:, :, 2].sum()), 0)   # red channel drawn
        self.assertEqual(int(canvas[:, :, 1].sum()), 0)     # no green

    def test_rider_and_auto_in_same_frame(self):
        helmet = FakeHelmetModel(cls=1, conf=0.90)
        mon = HelmetRiderMonitor(helmet, FakePersonModel([AUTO_PASSENGER, RIDER]))
        confirmed = []
        for f in (3, 6, 9):
            confirmed += mon.update(FRAME, [(3, AUTO_BIKE), (7, RIDER_BIKE)], f)
        self.assertEqual(confirmed, [7])

    def test_helmeted_rider_drawn_green_never_confirmed(self):
        helmet = FakeHelmetModel(cls=0, conf=0.80)
        mon = HelmetRiderMonitor(helmet, FakePersonModel([RIDER]))
        confirmed = []
        for f in (3, 6, 9, 12):
            confirmed += mon.update(FRAME, [(7, RIDER_BIKE)], f)
        self.assertEqual(confirmed, [])
        canvas = FRAME.copy()
        mon.draw(canvas, 12)
        self.assertGreater(int(canvas[:, :, 1].sum()), 0)
        self.assertEqual(int(canvas[:, :, 2].sum()), 0)

    def test_skips_off_frames_without_running_models(self):
        persons = FakePersonModel([RIDER])
        mon = HelmetRiderMonitor(FakeHelmetModel(1, 0.9), persons)
        self.assertEqual(mon.update(FRAME, [(7, RIDER_BIKE)], 4), [])
        self.assertEqual(persons.calls, 0)


if __name__ == "__main__":
    unittest.main()
