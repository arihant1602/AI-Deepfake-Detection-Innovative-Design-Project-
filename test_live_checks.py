"""
Unit tests for the live engine's screen-flash reflection check (Gate 1b, advisory).
"""

import random
import unittest

import numpy as np

import host_integrity as h
from live_engine import FLASH_PAIRS, FLASH_SLOTS, analyze_screen_flash


def flash_session(gain, lag_s=0.1, fps=30.0, seed=0):
    """Frames around a flash plan: the face box takes on `gain` x the shown colour after `lag_s`."""
    rng = np.random.default_rng(seed)
    plus, minus = FLASH_PAIRS[0]
    plan = {"t0": 100.0, "slot_s": 0.4, "seq": h.make_challenge_sequence(FLASH_SLOTS, 5, random.Random(seed)),
            "colours": ["#a", "#b"], "rgb": [plus, minus]}
    base = rng.uniform(70, 160, (120, 160, 3))
    items = []
    for i in range(int((FLASH_SLOTS * 0.4 + 0.5) * fps)):
        t = plan["t0"] - 0.05 + i / fps
        k = int(np.floor((t - lag_s - plan["t0"]) / 0.4))
        f = base + rng.normal(0, 2.0, base.shape) + 0.5 * i / fps  # sensor noise + auto-exposure drift
        if 0 <= k < FLASH_SLOTS:
            shown = np.array((plus if plan["seq"][k] > 0 else minus)[::-1], float) / 255.0  # BGR
            f[30:90, 50:110] += gain * (shown - 0.5)
        items.append((t, np.clip(f, 0, 255)))
    return items, plan


class TestScreenFlash(unittest.TestCase):
    face = (50, 30, 60, 60)

    def test_reflection_on_the_face_passes(self):
        items, plan = flash_session(gain=6.0)
        r = analyze_screen_flash(items, plan, self.face)
        self.assertEqual(r["verdict"], "PASS", r)
        self.assertAlmostEqual(r["lag_ms"], 100, delta=40)

    def test_no_reflection_is_reported(self):
        for seed in range(5):
            items, plan = flash_session(gain=0.0, seed=seed)
            self.assertEqual(analyze_screen_flash(items, plan, self.face)["verdict"], "NO_REFLECTION")

    def test_no_face_is_inconclusive(self):
        items, plan = flash_session(gain=6.0)
        self.assertEqual(analyze_screen_flash(items, plan, None)["verdict"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
