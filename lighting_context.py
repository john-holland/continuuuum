from __future__ import annotations

import datetime as dt
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .ephemeris import IEphemerisProvider


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _norm(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    mag = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    if mag <= 1e-9:
        return (0.0, 0.0, 0.0)
    return (vec[0] / mag, vec[1] / mag, vec[2] / mag)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _parse_vec3(value: object) -> tuple[float, float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        try:
            return (float(value["x"]), float(value["y"]), float(value["z"]))
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _parse_datetime_utc(raw: object) -> dt.datetime | None:
    if isinstance(raw, dt.datetime):
        return raw.astimezone(dt.timezone.utc) if raw.tzinfo else raw.replace(tzinfo=dt.timezone.utc)
    if isinstance(raw, str):
        txt = raw.strip()
        if not txt:
            return None
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(txt)
        except ValueError:
            return None
        return parsed.astimezone(dt.timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    return None


def _to_bool(raw: object, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class WeatherSnapshot:
    provider: str
    cloud_cover_pct: float | None = None
    visibility_m: float | None = None
    precipitation_mm: float | None = None
    wind_speed_mps: float | None = None

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "cloud_cover_pct": self.cloud_cover_pct,
            "visibility_m": self.visibility_m,
            "precipitation_mm": self.precipitation_mm,
            "wind_speed_mps": self.wind_speed_mps,
        }


class WeatherAdapter:
    name = "none"

    def get_snapshot(self, *, lat: float, lon: float, when_utc: dt.datetime) -> WeatherSnapshot:
        raise NotImplementedError


class OpenMeteoWeatherAdapter(WeatherAdapter):
    name = "open-meteo"

    def get_snapshot(self, *, lat: float, lon: float, when_utc: dt.datetime) -> WeatherSnapshot:
        # Keep request small and keyless; select nearest hourly point for the target timestamp.
        base = "https://api.open-meteo.com/v1/forecast"
        qs = urllib.parse.urlencode(
            {
                "latitude": f"{lat:.6f}",
                "longitude": f"{lon:.6f}",
                "timezone": "UTC",
                "hourly": "cloud_cover,visibility,precipitation,wind_speed_10m",
                "forecast_days": 2,
                "past_days": 1,
            }
        )
        req = urllib.request.Request(f"{base}?{qs}", headers={"User-Agent": "ContinuumLighting/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            payload = json.loads(r.read().decode("utf-8"))
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            return WeatherSnapshot(provider=self.name)
        target_dt = when_utc.replace(minute=0, second=0, microsecond=0)
        best_idx = 0
        best_dist = 10**9
        for i, t in enumerate(times):
            try:
                parsed_t = dt.datetime.fromisoformat(str(t).replace("Z", "+00:00"))
                if parsed_t.tzinfo is None:
                    parsed_t = parsed_t.replace(tzinfo=dt.timezone.utc)
                dist = abs((parsed_t - target_dt).total_seconds())
            except ValueError:
                dist = float(i)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        def _pick(name: str) -> float | None:
            series = hourly.get(name) or []
            if best_idx < len(series):
                try:
                    return float(series[best_idx])
                except (TypeError, ValueError):
                    return None
            return None
        vis_km = _pick("visibility")
        wind_kmh = _pick("wind_speed_10m")
        return WeatherSnapshot(
            provider=self.name,
            cloud_cover_pct=_pick("cloud_cover"),
            visibility_m=(vis_km * 1000.0) if vis_km is not None else None,
            precipitation_mm=_pick("precipitation"),
            wind_speed_mps=(wind_kmh / 3.6) if wind_kmh is not None else None,
        )


class SunPositionCalculator:
    @staticmethod
    def compute(*, lat_deg: float, lon_deg: float, when_utc: dt.datetime) -> dict[str, float | tuple[float, float, float]]:
        # Approximate NOAA solar position.
        n = when_utc.timetuple().tm_yday
        hour = when_utc.hour + (when_utc.minute / 60.0) + (when_utc.second / 3600.0)
        gamma = 2.0 * math.pi / 365.0 * (n - 1 + (hour - 12.0) / 24.0)
        eqtime = 229.18 * (
            0.000075
            + 0.001868 * math.cos(gamma)
            - 0.032077 * math.sin(gamma)
            - 0.014615 * math.cos(2 * gamma)
            - 0.040849 * math.sin(2 * gamma)
        )
        decl = (
            0.006918
            - 0.399912 * math.cos(gamma)
            + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma)
            + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma)
            + 0.00148 * math.sin(3 * gamma)
        )
        time_offset = eqtime + 4.0 * lon_deg
        tst = hour * 60.0 + time_offset
        ha = math.radians((tst / 4.0) - 180.0)
        lat = math.radians(lat_deg)
        cos_zenith = _clamp(math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha), -1.0, 1.0)
        zenith = math.acos(cos_zenith)
        elevation = 90.0 - math.degrees(zenith)
        az = math.degrees(
            math.atan2(
                math.sin(ha),
                math.cos(ha) * math.sin(lat) - math.tan(decl) * math.cos(lat),
            )
        )
        azimuth = (az + 180.0) % 360.0
        # ENU world vector: X East, Y Up, Z North.
        az_r = math.radians(azimuth)
        el_r = math.radians(elevation)
        direction = _norm(
            (
                math.sin(az_r) * math.cos(el_r),
                math.sin(el_r),
                math.cos(az_r) * math.cos(el_r),
            )
        )
        return {"sun_azimuth_deg": azimuth, "sun_elevation_deg": elevation, "sun_direction_vector_world": direction}


class MoonPositionCalculator:
    @staticmethod
    def compute(*, lat_deg: float, lon_deg: float, when_utc: dt.datetime) -> dict[str, float | tuple[float, float, float] | str]:
        # Approximate moon position (sufficient for lighting-context estimation).
        days = (when_utc - dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)).total_seconds() / 86400.0
        mean_long = (218.316 + 13.176396 * days) % 360.0
        mean_anomaly = (134.963 + 13.064993 * days) % 360.0
        arg_lat = (93.272 + 13.229350 * days) % 360.0
        lam = mean_long + 6.289 * math.sin(math.radians(mean_anomaly))
        beta = 5.128 * math.sin(math.radians(arg_lat))
        eps = 23.439 - 0.0000004 * days

        lam_r = math.radians(lam)
        beta_r = math.radians(beta)
        eps_r = math.radians(eps)
        ra = math.atan2(
            math.sin(lam_r) * math.cos(eps_r) - math.tan(beta_r) * math.sin(eps_r),
            math.cos(lam_r),
        )
        dec = math.asin(
            math.sin(beta_r) * math.cos(eps_r) + math.cos(beta_r) * math.sin(eps_r) * math.sin(lam_r)
        )

        jd = 2451545.0 + days
        t = (jd - 2451545.0) / 36525.0
        gmst = (
            280.46061837
            + 360.98564736629 * (jd - 2451545.0)
            + 0.000387933 * t * t
            - (t * t * t) / 38710000.0
        ) % 360.0
        lst = math.radians((gmst + lon_deg) % 360.0)
        ha = lst - ra
        while ha < -math.pi:
            ha += 2 * math.pi
        while ha > math.pi:
            ha -= 2 * math.pi

        lat_r = math.radians(lat_deg)
        sin_el = math.sin(dec) * math.sin(lat_r) + math.cos(dec) * math.cos(lat_r) * math.cos(ha)
        elevation = math.degrees(math.asin(_clamp(sin_el, -1.0, 1.0)))
        az = math.degrees(
            math.atan2(
                math.sin(ha),
                math.cos(ha) * math.sin(lat_r) - math.tan(dec) * math.cos(lat_r),
            )
        )
        azimuth = (az + 180.0) % 360.0

        az_r = math.radians(azimuth)
        el_r = math.radians(elevation)
        direction = _norm(
            (
                math.sin(az_r) * math.cos(el_r),
                math.sin(el_r),
                math.cos(az_r) * math.cos(el_r),
            )
        )

        synodic_phase = ((days % 29.53058867) / 29.53058867)
        illum = (1.0 - math.cos(2.0 * math.pi * synodic_phase)) * 0.5
        if illum < 0.03:
            phase = "new"
        elif illum < 0.22:
            phase = "crescent"
        elif illum < 0.78:
            phase = "quarter_gibbous"
        elif illum < 0.97:
            phase = "gibbous"
        else:
            phase = "full"

        return {
            "moon_azimuth_deg": azimuth,
            "moon_elevation_deg": elevation,
            "moon_direction_vector_world": direction,
            "moon_illumination_fraction": _clamp(illum, 0.0, 1.0),
            "moon_phase_name": phase,
        }


class SunPlaneSurfaceAnalyzer:
    """
    Fallback 2C estimator:
    infer sun direction from available surface/shadow hints in asset metadata.
    """

    def infer(self, metadata: dict) -> tuple[tuple[float, float, float] | None, float, list[str]]:
        reasons: list[str] = []
        # Highest confidence: explicit inferred direction already present.
        direct = _parse_vec3(metadata.get("inferred_sun_direction_vector"))
        if direct:
            return _norm(direct), 0.9, ["metadata_inferred_sun_direction_vector"]

        shadow = _parse_vec3(metadata.get("shadow_vector") or metadata.get("dominant_shadow_direction"))
        plane_normal = _parse_vec3(metadata.get("sun_plane_normal") or metadata.get("dominant_surface_normal"))
        if shadow and plane_normal:
            # light roughly opposite shadow and above surface.
            shadow_n = _norm(shadow)
            normal_n = _norm(plane_normal)
            candidate = _norm((-shadow_n[0], abs(normal_n[1]) + 0.2, -shadow_n[2]))
            return candidate, 0.72, ["shadow_vector", "surface_normal"]

        shadows = metadata.get("shadow_vectors")
        normals = metadata.get("surface_normals")
        if isinstance(shadows, list) and isinstance(normals, list) and shadows and normals:
            samples: list[tuple[float, float, float]] = []
            for i in range(min(len(shadows), len(normals))):
                sv = _parse_vec3(shadows[i])
                nv = _parse_vec3(normals[i])
                if not sv or not nv:
                    continue
                s = _norm(sv)
                n = _norm(nv)
                samples.append(_norm((-s[0], abs(n[1]) + 0.2, -s[2])))
            if samples:
                avg = _norm(
                    (
                        sum(v[0] for v in samples) / len(samples),
                        sum(v[1] for v in samples) / len(samples),
                        sum(v[2] for v in samples) / len(samples),
                    )
                )
                reasons.append("aggregate_shadow_surface_samples")
                return avg, _clamp(0.45 + 0.05 * len(samples), 0.45, 0.8), reasons
        return None, 0.0, reasons


class LightingValidator:
    def __init__(self, ml_estimator: Callable[[dict], dict] | None = None):
        self._ml_estimator = ml_estimator

    def validate(
        self,
        *,
        expected_direction: tuple[float, float, float],
        chosen_direction: tuple[float, float, float],
        weather: WeatherSnapshot | None,
        metadata: dict,
    ) -> dict[str, object]:
        flags: list[str] = []
        checks: list[float] = []

        alignment = (_dot(_norm(expected_direction), _norm(chosen_direction)) + 1.0) * 0.5
        checks.append(alignment)
        if alignment < 0.35:
            flags.append("sun_direction_misaligned_with_ephemeris")

        contrast = metadata.get("frame_contrast")
        if isinstance(contrast, (int, float)) and weather and weather.cloud_cover_pct is not None:
            cc = float(weather.cloud_cover_pct)
            cval = float(contrast)
            plausible = 1.0 - min(1.0, abs((100.0 - cc) / 100.0 - cval))
            checks.append(plausible)
            if plausible < 0.3:
                flags.append("weather_contrast_mismatch")

        ml_score = None
        if self._ml_estimator is not None:
            ml = self._ml_estimator(metadata) or {}
            ml_dir = _parse_vec3(ml.get("direction_vector"))
            ml_clarity = ml.get("clarity_score")
            if ml_dir:
                ml_alignment = (_dot(_norm(chosen_direction), _norm(ml_dir)) + 1.0) * 0.5
                checks.append(ml_alignment)
                if ml_alignment < 0.35:
                    flags.append("ml_direction_mismatch")
            if isinstance(ml_clarity, (int, float)):
                ml_score = _clamp(float(ml_clarity), 0.0, 1.0)
                checks.append(ml_score)

        score = sum(checks) / len(checks) if checks else 0.0
        return {
            "lighting_validity_score": round(_clamp(score, 0.0, 1.0), 4),
            "lighting_validation_flags": flags,
            "ml_clarity_score": ml_score,
            "rule_alignment_score": round(alignment, 4),
        }


class LightingContextService:
    def __init__(
        self,
        *,
        weather_adapters: dict[str, WeatherAdapter] | None = None,
        default_weather_adapter: str = "open-meteo",
        ml_estimator: Callable[[dict], dict] | None = None,
        ephemeris_provider: IEphemerisProvider | None = None,
        body_catalog: dict[str, str] | None = None,  # body_id -> kind
        body_radii_km: dict[str, float] | None = None,
    ):
        adapters = weather_adapters or {"open-meteo": OpenMeteoWeatherAdapter()}
        self._weather_adapters = adapters
        self._default_weather_adapter = default_weather_adapter if default_weather_adapter in adapters else next(iter(adapters))
        self._sun = SunPositionCalculator()
        self._moon = MoonPositionCalculator()
        self._analyzer = SunPlaneSurfaceAnalyzer()
        self._validator = LightingValidator(ml_estimator=ml_estimator)
        self._ephemeris_provider = ephemeris_provider
        self._body_catalog = dict(body_catalog or {})
        self._body_radii = dict(body_radii_km or {})

    @staticmethod
    def _extract_datetime(metadata: dict) -> dt.datetime:
        for key in ("capture_datetime_utc", "capture_datetime", "timestamp_utc", "recorded_at"):
            parsed = _parse_datetime_utc(metadata.get(key))
            if parsed is not None:
                return parsed
        return dt.datetime.now(tz=dt.timezone.utc)

    @staticmethod
    def _extract_metadata_direction(metadata: dict) -> tuple[tuple[float, float, float] | None, float]:
        vec = _parse_vec3(metadata.get("sun_direction_vector_world") or metadata.get("sun_direction_vector"))
        conf = metadata.get("sun_direction_confidence")
        confidence = float(conf) if isinstance(conf, (int, float)) else (0.85 if vec else 0.0)
        return (_norm(vec) if vec else None), _clamp(confidence, 0.0, 1.0)

    def compute(self, *, lat: float, lon: float, altitude_m: float | None = None, metadata: dict | None = None) -> dict:
        metadata = dict(metadata or {})
        rollout = metadata.get("lighting_rollout_flags")
        if not isinstance(rollout, dict):
            rollout = {}
        enable_surface_inference = _to_bool(rollout.get("enable_surface_inference"), True)
        enable_ml_validation = _to_bool(rollout.get("enable_ml_validation"), True)
        enable_lunar_context = _to_bool(rollout.get("enable_lunar_context"), True)
        when_utc = self._extract_datetime(metadata)
        sun = self._sun.compute(lat_deg=lat, lon_deg=lon, when_utc=when_utc)
        expected_direction = sun["sun_direction_vector_world"]
        moon = self._moon.compute(lat_deg=lat, lon_deg=lon, when_utc=when_utc) if enable_lunar_context else None

        md_vec, md_conf = self._extract_metadata_direction(metadata)
        inferred_vec, inferred_conf, inferred_reasons = ((None, 0.0, []) if not enable_surface_inference else self._analyzer.infer(metadata))

        chosen = md_vec or inferred_vec or expected_direction
        source = "metadata" if md_vec else ("inferred_surface" if inferred_vec else "calculated")
        direction_conf = md_conf if md_vec else (inferred_conf if inferred_vec else 0.55)

        weather_adapter_name = str(metadata.get("weather_adapter") or self._default_weather_adapter)
        weather_adapter = self._weather_adapters.get(weather_adapter_name) or self._weather_adapters[self._default_weather_adapter]
        weather = weather_adapter.get_snapshot(lat=lat, lon=lon, when_utc=when_utc)

        validation = self._validator.validate(
            expected_direction=expected_direction,
            chosen_direction=chosen,
            weather=weather,
            metadata=metadata if enable_ml_validation else {k: v for k, v in metadata.items() if k != "ml_lighting_estimate"},
        )
        sun_elevation = float(sun["sun_elevation_deg"])
        sun_visible = bool(sun_elevation > 0.0)
        moon_elevation = float(moon["moon_elevation_deg"]) if moon is not None else None
        moon_visible = bool(moon_elevation is not None and moon_elevation > 0.0)

        return {
            "sun_azimuth_deg": round(float(sun["sun_azimuth_deg"]), 4),
            "sun_elevation_deg": round(float(sun["sun_elevation_deg"]), 4),
            "sun_visibility": sun_visible,
            "sun_direction_source": source,
            "sun_direction_vector_world": [round(chosen[0], 6), round(chosen[1], 6), round(chosen[2], 6)],
            "sun_direction_confidence": round(direction_conf, 4),
            "inferred_sun_direction_vector": [round(inferred_vec[0], 6), round(inferred_vec[1], 6), round(inferred_vec[2], 6)]
            if inferred_vec
            else None,
            "inferred_sun_direction_confidence": round(inferred_conf, 4) if inferred_vec else 0.0,
            "inferred_sun_direction_reasons": inferred_reasons,
            "weather_snapshot": weather.to_dict(),
            "capture_datetime_utc": when_utc.isoformat().replace("+00:00", "Z"),
            "asset_location": {"lat": lat, "lon": lon, "altitude_m": altitude_m},
            "moon_azimuth_deg": round(float(moon["moon_azimuth_deg"]), 4) if moon is not None else None,
            "moon_elevation_deg": round(float(moon["moon_elevation_deg"]), 4) if moon is not None else None,
            "moon_direction_vector_world": (
                [
                    round(float(moon["moon_direction_vector_world"][0]), 6),
                    round(float(moon["moon_direction_vector_world"][1]), 6),
                    round(float(moon["moon_direction_vector_world"][2]), 6),
                ]
                if moon is not None
                else None
            ),
            "moon_illumination_fraction": round(float(moon["moon_illumination_fraction"]), 4) if moon is not None else None,
            "moon_phase_name": str(moon["moon_phase_name"]) if moon is not None else None,
            "moon_direction_source": "calculated" if moon is not None else "none",
            "moon_direction_confidence": 0.7 if moon is not None else 0.0,
            "moon_visibility": moon_visible,
            "lighting_rollout_flags": {
                "enable_surface_inference": enable_surface_inference,
                "enable_ml_validation": enable_ml_validation,
                "enable_lunar_context": enable_lunar_context,
            },
            **validation,
        }

    def query_lighting(
        self,
        *,
        observer_body_id: str = "earth",
        lat: float,
        lon: float,
        altitude_m: float | None = None,
        datetime_utc: dt.datetime | str | None = None,
        sources: list[str] | None = None,
        include_eclipses: bool = True,
        metadata: dict | None = None,
    ) -> dict:
        """
        Generalized N-body lighting query.
        Uses ephemeris + multi-body solver when provider is configured; otherwise falls back to compute().
        Preserves earth/moon field compatibility.
        """
        when_utc = _parse_datetime_utc(datetime_utc) if datetime_utc else self._extract_datetime(metadata or {})
        if when_utc is None:
            when_utc = dt.datetime.now(tz=dt.timezone.utc)

        if self._ephemeris_provider is None or observer_body_id.lower() != "earth":
            return self.compute(
                lat=lat,
                lon=lon,
                altitude_m=altitude_m,
                metadata={**(metadata or {}), "capture_datetime_utc": when_utc.isoformat()},
            )

        try:
            try:
                from .multi_body_lighting import MultiBodyLightSolver
                from .ephemeris import BodyState
            except ImportError:
                from multi_body_lighting import MultiBodyLightSolver  # noqa: PLC0415
                from ephemeris import BodyState  # noqa: PLC0415

            body_ids = sources or ["sun", "moon", "earth"]
            states: dict[str, BodyState] = {}
            for bid in body_ids:
                s = self._ephemeris_provider.get_body_state(bid, when_utc, "J2000")
                if s:
                    states[bid] = s

            if not states:
                return self.compute(lat=lat, lon=lon, altitude_m=altitude_m, metadata={"capture_datetime_utc": when_utc.isoformat()})

            earth_state = states.get("earth")
            if not earth_state:
                return self.compute(lat=lat, lon=lon, altitude_m=altitude_m, metadata={"capture_datetime_utc": when_utc.isoformat()})

            earth_radius_km = self._body_radii.get("earth", 6371.0)
            lat_r = math.radians(lat)
            lon_r = math.radians(lon)
            alt_km = (altitude_m or 0) / 1000.0
            r = earth_radius_km + alt_km
            obs_x = earth_state.position[0] + r * math.cos(lat_r) * math.cos(lon_r)
            obs_y = earth_state.position[1] + r * math.cos(lat_r) * math.sin(lon_r)
            obs_z = earth_state.position[2] + r * math.sin(lat_r)
            observer_pos = (obs_x, obs_y, obs_z)

            kinds = self._body_catalog or {"sun": "star", "moon": "moon", "earth": "planet"}
            solver = MultiBodyLightSolver(body_radii_km=self._body_radii)
            multi = solver.solve(
                observer_body_id=observer_body_id,
                observer_pos=observer_pos,
                observer_lat_deg=lat,
                observer_lon_deg=lon,
                body_states=states,
                body_kinds=kinds,
                include_eclipses=include_eclipses,
            )

            base = self.compute(lat=lat, lon=lon, altitude_m=altitude_m, metadata={"capture_datetime_utc": when_utc.isoformat()})
            base["light_sources"] = multi.get("light_sources", [])
            base["eclipses"] = multi.get("eclipses", [])
            base["aggregate_direction"] = multi.get("aggregate_direction", base.get("sun_direction_vector_world", [0, 1, 0]))
            base["aggregate_intensity"] = multi.get("aggregate_intensity", 1.0)
            if multi.get("sun_azimuth_deg") is not None:
                base["sun_azimuth_deg"] = multi["sun_azimuth_deg"]
                base["sun_elevation_deg"] = multi["sun_elevation_deg"]
                base["sun_direction_vector_world"] = multi.get("sun_direction_vector_world", base["sun_direction_vector_world"])
                base["sun_visibility"] = multi.get("sun_visibility", base["sun_visibility"])
            if multi.get("moon_azimuth_deg") is not None:
                base["moon_azimuth_deg"] = multi["moon_azimuth_deg"]
                base["moon_elevation_deg"] = multi["moon_elevation_deg"]
                base["moon_direction_vector_world"] = multi.get("moon_direction_vector_world", base.get("moon_direction_vector_world"))
                base["moon_visibility"] = multi.get("moon_visibility", base.get("moon_visibility"))
            return base
        except Exception:
            return self.compute(lat=lat, lon=lon, altitude_m=altitude_m, metadata={"capture_datetime_utc": when_utc.isoformat()})
