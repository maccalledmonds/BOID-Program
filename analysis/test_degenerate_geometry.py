"""
Regression test for the degenerate-geometry crash in the flocking rules.

pygame's Vector3.scale_to_length() and normalize_ip() raise
    ValueError: Cannot scale a vector with zero length
for any vector shorter than pygame's internal epsilon (1e-6), not only for
exactly-zero vectors. The rules originally guarded with `length() > 0`, which
lets a tiny-but-nonzero vector through and crashes the simulation.

This happens whenever a bird sits (almost) exactly on the centroid of its
neighbours -- rare per frame, but essentially certain over a long run.

Run:  python analysis/test_degenerate_geometry.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame

import flight_model as fm


def _boid(pos, vel):
    b = fm.Boid(0, 0, (255, 255, 255))
    b.pos = pygame.math.Vector3(*pos)
    b.vel = pygame.math.Vector3(*vel)
    return b


def test_cohesion_at_centroid():
    """A bird exactly at its neighbours' centroid must not crash cohesion()."""
    centre = _boid((1000, 1000, 0), (10, 0, 0))
    ring = [
        _boid((1000 + d * 50, 1000, 0), (10, 0, 0))
        for d in (-1, 1)
    ] + [
        _boid((1000, 1000 + d * 50, 0), (10, 0, 0))
        for d in (-1, 1)
    ]
    boids = [centre] + ring
    out = centre.cohesion(boids)
    assert out.length() == 0.0, f"expected zero steer at centroid, got {out}"


def test_cohesion_just_below_epsilon():
    """Offset smaller than pygame's epsilon but greater than zero."""
    centre = _boid((1000, 1000, 0), (10, 0, 0))
    others = [
        _boid((1000 + 50, 1000, 0), (10, 0, 0)),
        _boid((1000 - 50 + 1e-9, 1000, 0), (10, 0, 0)),
    ]
    centre.cohesion([centre] + others)


def test_separation_and_alignment_coincident():
    """Coincident birds with opposing velocities exercise both other rules."""
    a = _boid((1000, 1000, 0), (10, 0, 0))
    b = _boid((1000, 1000, 0), (-10, 0, 0))
    a.separation([a, b])
    a.alignment([a, b])


def test_long_run_no_crash():
    """The crash originally surfaced only over many steps; run a real sim."""
    random.seed(7)
    fm.RADIUS_MULTIPLIER = 8.0
    boids = [
        fm.Boid(random.uniform(0, fm.DEPTH), random.uniform(0, fm.DEPTH), (255, 255, 255))
        for _ in range(40)
    ]
    for _ in range(1500):
        for x in boids:
            x.apply_force(x.separation(boids) * fm.SEPARATION_WEIGHT)
            x.apply_force(x.alignment(boids) * fm.ALIGNMENT_WEIGHT)
            x.apply_force(x.cohesion(boids) * fm.COHESION_WEIGHT)
        for x in boids:
            x.update()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
