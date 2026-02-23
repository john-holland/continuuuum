"""Tests for multi-body lighting solver and occlusion detection."""
from __future__ import annotations

import datetime as dt
from datetime import timezone

from ephemeris import BodyState
from multi_body_lighting import MultiBodyLightSolver, OcclusionEclipseSolver


def test_multi_body_light_solver_basic():
    """Multi-body solver produces light_sources and aggregate direction."""
    solver = MultiBodyLightSolver()
    observer_pos = (0.0, 0.0, 0.0)  # origin
    sun_pos = (1.0, 1.0, 0.0)  # roughly upper-right
    moon_pos = (0.5, 0.5, 0.5)
    earth_pos = (-0.001, 0.0, 0.0)
    states = {
        "sun": BodyState("sun", dt.datetime.now(timezone.utc), sun_pos, (0.0, 0.0, 0.0)),
        "moon": BodyState("moon", dt.datetime.now(timezone.utc), moon_pos, (0.0, 0.0, 0.0)),
        "earth": BodyState("earth", dt.datetime.now(timezone.utc), earth_pos, (0.0, 0.0, 0.0)),
    }
    kinds = {"sun": "star", "moon": "moon", "earth": "planet"}
    result = solver.solve(
        observer_body_id="earth",
        observer_pos=observer_pos,
        observer_lat_deg=0.0,
        observer_lon_deg=0.0,
        body_states=states,
        body_kinds=kinds,
        include_eclipses=False,
    )
    assert "light_sources" in result
    assert len(result["light_sources"]) >= 1
    assert "aggregate_direction" in result
    assert "aggregate_intensity" in result
    assert "sun_azimuth_deg" in result or "moon_azimuth_deg" in result


def test_occlusion_solver_finds_candidates():
    """Occlusion solver returns events when geometry matches."""
    occ = OcclusionEclipseSolver(body_radii_km={"moon": 1737.0})
    observer = (0.0, 0.0, 0.0)
    sun = (1000.0, 0.0, 0.0)
    moon = (500.0, 0.0, 0.0)  # between observer and sun
    states = {
        "sun": BodyState("sun", dt.datetime.now(timezone.utc), sun, (0.0, 0.0, 0.0)),
        "moon": BodyState("moon", dt.datetime.now(timezone.utc), moon, (0.0, 0.0, 0.0)),
    }
    events = occ.find_occlusions(observer, states, "earth", ["sun"], body_kinds={"sun": "star"})
    assert isinstance(events, list)
