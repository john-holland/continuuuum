"""
Multi-body lighting solver: per-source visibility, aggregate irradiance, occlusion/eclipse detection.
Uses spatial broadphase (octree-like) for candidate occluder queries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

try:
    from .ephemeris import BodyState
except ImportError:
    from ephemeris import BodyState  # noqa: PLC0415 - top-level when run from continuum dir


def _norm(v: tuple[float, float, float]) -> tuple[float, float, float]:
    mag = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if mag <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    d = _vec_sub(a, b)
    return math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])


@dataclass
class LightSourceInfo:
    """Per-source lighting contribution."""

    body_id: str
    kind: str  # star, moon, planet
    direction: tuple[float, float, float]
    distance_km: float
    irradiance_factor: float  # 0..1, 1 for unobstructed primary star
    visibility: float  # 0..1 after occlusion
    is_primary: bool = False
    azimuth_deg: float | None = None
    elevation_deg: float | None = None


@dataclass
class OcclusionEvent:
    """Detected occlusion/eclipse."""

    source_body_id: str
    target_body_id: str
    occluder_body_id: str
    occlusion_ratio: float
    eclipse_type: str  # partial, annular, total, planet_occludes_planet, planet_occludes_star


@dataclass
class OctreeCell:
    """Simple 3D AABB for spatial broadphase."""

    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    bodies: list[tuple[str, tuple[float, float, float], float]] = field(default_factory=list)  # (body_id, pos, radius_km)

    def contains_point(self, x: float, y: float, z: float) -> bool:
        return (
            self.min_x <= x <= self.max_x
            and self.min_y <= y <= self.max_y
            and self.min_z <= z <= self.max_z
        )

    def intersects_segment(
        self,
        ax: float, ay: float, az: float,
        bx: float, by: float, bz: float,
    ) -> bool:
        """Conservative AABB-segment intersection."""
        t0, t1 = 0.0, 1.0
        for (lo, hi, a, b) in [
            (self.min_x, self.max_x, ax, bx),
            (self.min_y, self.max_y, ay, by),
            (self.min_z, self.max_z, az, bz),
        ]:
            d = b - a
            if abs(d) < 1e-12:
                if a < lo or a > hi:
                    return False
            else:
                t_lo = (lo - a) / d
                t_hi = (hi - a) / d
                if t_lo > t_hi:
                    t_lo, t_hi = t_hi, t_lo
                t0 = max(t0, t_lo)
                t1 = min(t1, t_hi)
                if t0 > t1:
                    return False
        return True


def _build_spatial_index(
    body_states: dict[str, BodyState],
    body_radii_km: dict[str, float],
) -> list[OctreeCell]:
    """Build coarse spatial grid of body positions for broadphase. Returns flat list of cells."""
    if not body_states:
        return []
    all_pos = [(bid, s.position, body_radii_km.get(bid, 0.0)) for bid, s in body_states.items()]
    if not all_pos:
        return []
    xs = [p[1][0] for p in all_pos]
    ys = [p[1][1] for p in all_pos]
    zs = [p[1][2] for p in all_pos]
    pad = max(1.0, max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) * 0.1
    mnx, mxx = min(xs) - pad, max(xs) + pad
    mny, mxy = min(ys) - pad, max(ys) + pad
    mnz, mxz = min(zs) - pad, max(zs) + pad
    cell = OctreeCell(mnx, mny, mnz, mxx, mxy, mxz)
    cell.bodies = all_pos
    return [cell]


def _sphere_occludes_line(
    observer: tuple[float, float, float],
    source: tuple[float, float, float],
    occluder_pos: tuple[float, float, float],
    occluder_radius_km: float,
) -> float:
    """
    Compute occlusion ratio (0..1) of source by occluder sphere from observer.
    Uses cone geometry: angular radius of occluder vs angular separation from source.
    """
    to_source = _vec_sub(source, observer)
    to_occluder = _vec_sub(occluder_pos, observer)
    d_obs_source = math.sqrt(_dot(to_source, to_source))
    d_obs_occ = math.sqrt(_dot(to_occluder, to_occluder))
    if d_obs_source < 1e-6 or d_obs_occ < occluder_radius_km:
        return 0.0
    dir_source = _norm(to_source)
    dir_occ = _norm(to_occluder)
    cos_sep = _dot(dir_source, dir_occ)
    if cos_sep < 0:
        return 0.0
    # Angular radius of occluder as seen from observer
    sin_occ = occluder_radius_km / d_obs_occ if d_obs_occ > 1e-6 else 0.0
    if sin_occ >= 1.0:
        return 1.0
    ang_occ = math.asin(min(1.0, sin_occ))
    # Angular separation between source and occluder center
    sep = math.acos(min(1.0, max(-1.0, cos_sep)))
    if sep >= ang_occ + 0.01:
        return 0.0
    if sep + ang_occ < 0.01:
        return 1.0
    # Overlap: simplified - total if source behind occluder, partial otherwise
    if sep < ang_occ - 0.001:
        return min(1.0, (ang_occ * ang_occ) / max(1e-12, (sep + ang_occ) ** 2))
    return max(0.0, min(1.0, 1.0 - (sep - ang_occ) / (2 * ang_occ)))


class OcclusionEclipseSolver:
    """Detect occlusions using spatial broadphase and geometric narrowphase."""

    def __init__(self, body_radii_km: dict[str, float] | None = None):
        self._radii = dict(body_radii_km or {})
        self._default_radius = 6371.0  # Earth km

    def _radius(self, body_id: str) -> float:
        return self._radii.get(body_id, self._default_radius)

    def find_occlusions(
        self,
        observer_pos: tuple[float, float, float],
        body_states: dict[str, BodyState],
        observer_body_id: str,
        source_body_ids: list[str],
        body_kinds: dict[str, str] | None = None,
    ) -> list[OcclusionEvent]:
        """Find occlusions of sources from observer. Excludes observer body from occluders."""
        events: list[OcclusionEvent] = []
        candidates = _build_spatial_index(body_states, self._radii)
        for src_id in source_body_ids:
            if src_id not in body_states or src_id == observer_body_id:
                continue
            src_pos = body_states[src_id].position
            for cell in candidates:
                for occ_id, occ_pos, _ in cell.bodies:
                    if occ_id == observer_body_id or occ_id == src_id:
                        continue
                    if occ_id not in body_states:
                        continue
                    if not cell.intersects_segment(
                        observer_pos[0], observer_pos[1], observer_pos[2],
                        src_pos[0], src_pos[1], src_pos[2],
                    ):
                        continue
                    ratio = _sphere_occludes_line(
                        observer_pos,
                        src_pos,
                        occ_pos,
                        self._radius(occ_id),
                    )
                    if ratio > 0.01:
                        kinds = body_kinds or {}
                        src_kind = kinds.get(src_id, "planet")
                        eclipse_type = "planet_occludes_star" if src_kind == "star" else "planet_occludes_planet"
                        events.append(OcclusionEvent(
                            source_body_id=src_id,
                            target_body_id=observer_body_id,
                            occluder_body_id=occ_id,
                            occlusion_ratio=ratio,
                            eclipse_type=eclipse_type,
                        ))
        return events

    def visibility_for_source(
        self,
        observer_pos: tuple[float, float, float],
        source_pos: tuple[float, float, float],
        body_states: dict[str, BodyState],
        exclude_body_ids: set[str],
    ) -> float:
        """Return visibility factor 0..1 after all occlusions."""
        visibility = 1.0
        candidates = _build_spatial_index(body_states, self._radii)
        for cell in candidates:
            for occ_id, occ_pos, _ in cell.bodies:
                if occ_id in exclude_body_ids:
                    continue
                ratio = _sphere_occludes_line(observer_pos, source_pos, occ_pos, self._radius(occ_id))
                visibility *= max(0.0, 1.0 - ratio)
        return visibility


class MultiBodyLightSolver:
    """
    Compute per-emitter irradiance contributions, per-source visibility, aggregate lighting.
    Output backward-compatible earth/sun/moon fields plus generalized light_sources array.
    """

    def __init__(
        self,
        body_radii_km: dict[str, float] | None = None,
        occlusion_solver: OcclusionEclipseSolver | None = None,
    ):
        self._radii = dict(body_radii_km or {})
        self._occlusion = occlusion_solver or OcclusionEclipseSolver(body_radii_km)

    def _sun_irradiance_factor(self, distance_km: float) -> float:
        au_km = 149597870.7
        return (au_km / max(1.0, distance_km)) ** 2

    def _reflectance_factor(self, body_kind: str) -> float:
        if body_kind == "moon":
            return 0.12
        if body_kind == "planet":
            return 0.3
        return 0.1

    def solve(
        self,
        observer_body_id: str,
        observer_pos: tuple[float, float, float],
        observer_lat_deg: float,
        observer_lon_deg: float,
        body_states: dict[str, BodyState],
        body_kinds: dict[str, str],  # body_id -> star|moon|planet|barycenter
        include_eclipses: bool = True,
    ) -> dict[str, Any]:
        """
        Compute multi-body lighting at observer. Returns dict with light_sources, eclipses,
        aggregate_direction, aggregate_intensity, and backward-compatible sun/moon fields.
        """
        light_sources: list[dict[str, Any]] = []
        eclipses: list[dict[str, Any]] = []
        primary_sun: dict | None = None
        primary_moon: dict | None = None

        # Identify emitters: stars (primary), moons/planets (reflected)
        emitters = [
            (bid, "star") for bid, kind in body_kinds.items()
            if kind == "star" and bid in body_states
        ]
        emitters += [
            (bid, kind) for bid, kind in body_kinds.items()
            if kind in ("moon", "planet") and bid in body_states
        ]

        source_ids = [e[0] for e in emitters]
        if include_eclipses:
            occ_events = self._occlusion.find_occlusions(
                observer_pos, body_states, observer_body_id, source_ids, body_kinds=body_kinds
            )
            for ev in occ_events:
                eclipses.append({
                    "source_body_id": ev.source_body_id,
                    "target_body_id": ev.target_body_id,
                    "occluder_body_id": ev.occluder_body_id,
                    "occlusion_ratio": round(ev.occlusion_ratio, 4),
                    "eclipse_type": ev.eclipse_type,
                })

        exclude_for_vis = {observer_body_id}
        for bid, kind in emitters:
            state = body_states[bid]
            to_source = _vec_sub(state.position, observer_pos)
            dist = math.sqrt(_dot(to_source, to_source))
            if dist < 1e-6:
                continue
            direction = _norm(to_source)
            visibility = self._occlusion.visibility_for_source(
                observer_pos, state.position, body_states, exclude_for_vis | {bid}
            ) if include_eclipses else 1.0

            if kind == "star":
                irrad = self._sun_irradiance_factor(dist) * visibility
            else:
                irrad = self._reflectance_factor(kind) * visibility / max(1.0, dist ** 2) * 1e10

            az, el = _enu_to_azimuth_elevation(direction, observer_lat_deg, observer_lon_deg)
            ls = {
                "body_id": bid,
                "kind": kind,
                "direction": [round(direction[0], 6), round(direction[1], 6), round(direction[2], 6)],
                "distance_km": round(dist, 2),
                "irradiance_factor": round(min(1.0, irrad), 6),
                "visibility": round(visibility, 4),
                "azimuth_deg": round(az, 2),
                "elevation_deg": round(el, 2),
            }
            if kind == "star":
                ls["is_primary"] = primary_sun is None
                if primary_sun is None:
                    primary_sun = ls
            if kind == "moon" and observer_body_id.lower() == "earth":
                if primary_moon is None:
                    primary_moon = ls
            light_sources.append(ls)

        agg_dir = (0.0, 0.0, 0.0)
        agg_intensity = 0.0
        for ls in light_sources:
            d = tuple(ls["direction"])
            w = ls["irradiance_factor"]
            agg_dir = (
                agg_dir[0] + d[0] * w,
                agg_dir[1] + d[1] * w,
                agg_dir[2] + d[2] * w,
            )
            agg_intensity += w
        if agg_intensity > 1e-12:
            n = _norm(agg_dir)
            agg_dir = (n[0], n[1], n[2])

        result: dict[str, Any] = {
            "light_sources": light_sources,
            "eclipses": eclipses,
            "aggregate_direction": [round(agg_dir[0], 6), round(agg_dir[1], 6), round(agg_dir[2], 6)],
            "aggregate_intensity": round(agg_intensity, 6),
        }
        if primary_sun:
            result["sun_azimuth_deg"] = primary_sun["azimuth_deg"]
            result["sun_elevation_deg"] = primary_sun["elevation_deg"]
            result["sun_direction_vector_world"] = primary_sun["direction"]
            result["sun_visibility"] = primary_sun["visibility"] > 0.01
        if primary_moon:
            result["moon_azimuth_deg"] = primary_moon["azimuth_deg"]
            result["moon_elevation_deg"] = primary_moon["elevation_deg"]
            result["moon_direction_vector_world"] = primary_moon["direction"]
            result["moon_visibility"] = primary_moon["visibility"] > 0.01
        return result


def _enu_to_azimuth_elevation(
    direction: tuple[float, float, float],
    lat_deg: float,
    lon_deg: float,
) -> tuple[float, float]:
    """Convert ENU/ICRF direction to local azimuth/elevation (simplified)."""
    x, y, z = direction
    elev = math.degrees(math.asin(max(-1.0, min(1.0, y))))
    az = math.degrees(math.atan2(x, z))
    if az < 0:
        az += 360.0
    return (az, elev)
