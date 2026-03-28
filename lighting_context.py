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


def _camera_facing_to_azimuth_deg(facing: str | float | None) -> float:
    """Map camera facing to world azimuth (0=north, 90=east, 180=south, 270=west)."""
    if facing is None:
        return 0.0
    if isinstance(facing, (int, float)):
        return float(facing) % 360.0
    s = str(facing).strip().lower()
    if s == "north":
        return 0.0
    if s == "east":
        return 90.0
    if s == "south":
        return 180.0
    if s == "west":
        return 270.0
    return 0.0


class ShadowGeometrySolver:
    """
    Pure logic for shadow-direction <-> time mapping.
    Shadows point away from the sun. Sun rises east (~90°), sets west (~270°).
    Northern hemisphere: facing north, morning -> shadows left (west); afternoon -> shadows right (east).
    """

    def __init__(self, sun_calculator: type = SunPositionCalculator):
        self._sun = sun_calculator

    def sunrise_sunset_utc(
        self,
        *,
        lat_deg: float,
        lon_deg: float,
        date_utc: dt.datetime,
    ) -> tuple[dt.datetime, dt.datetime]:
        """Return (sunrise_utc, sunset_utc) for the given date at the location."""
        day_start = date_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        if day_start.tzinfo is None:
            day_start = day_start.replace(tzinfo=dt.timezone.utc)
        # Binary search for elevation crossing 0
        lo, hi = 0.0, 24.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            t = day_start + dt.timedelta(hours=mid)
            el = float(self._sun.compute(lat_deg=lat_deg, lon_deg=lon_deg, when_utc=t)["sun_elevation_deg"])
            if el < 0:
                lo = mid
            else:
                hi = mid
        sunrise_h = (lo + hi) / 2.0
        lo, hi = 12.0, 24.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            t = day_start + dt.timedelta(hours=mid)
            el = float(self._sun.compute(lat_deg=lat_deg, lon_deg=lon_deg, when_utc=t)["sun_elevation_deg"])
            if el > 0:
                lo = mid
            else:
                hi = mid
        sunset_h = (lo + hi) / 2.0
        return (
            day_start + dt.timedelta(hours=sunrise_h),
            day_start + dt.timedelta(hours=sunset_h),
        )

    def shadow_direction_for_time(
        self,
        *,
        lat_deg: float,
        lon_deg: float,
        when_utc: dt.datetime,
    ) -> float:
        """Shadow azimuth (deg, world) at given time. Shadow points away from sun."""
        sun = self._sun.compute(lat_deg=lat_deg, lon_deg=lon_deg, when_utc=when_utc)
        sun_az = float(sun["sun_azimuth_deg"])
        return (sun_az + 180.0) % 360.0

    def estimate_time_from_shadow_direction(
        self,
        *,
        lat_deg: float,
        lon_deg: float,
        date_utc: dt.datetime,
        shadow_azimuth_world_deg: float,
        camera_facing_deg: float = 0.0,
        tolerance_deg: float = 15.0,
    ) -> dt.datetime | None:
        """
        Find when_utc on date such that predicted shadow direction matches detected.
        Returns estimated capture time or None if no plausible match.
        """
        sunrise, sunset = self.sunrise_sunset_utc(lat_deg=lat_deg, lon_deg=lon_deg, date_utc=date_utc)
        day_start = date_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        if day_start.tzinfo is None:
            day_start = day_start.replace(tzinfo=dt.timezone.utc)
        sr_h = (sunrise - day_start).total_seconds() / 3600.0
        ss_h = (sunset - day_start).total_seconds() / 3600.0
        if sr_h >= ss_h:
            return None
        best_t = None
        best_err = 360.0
        # Sample hours through the day
        steps = max(24, int((ss_h - sr_h) * 2))
        for i in range(steps + 1):
            h = sr_h + (ss_h - sr_h) * i / max(1, steps)
            t = day_start + dt.timedelta(hours=h)
            pred_shadow = self.shadow_direction_for_time(lat_deg=lat_deg, lon_deg=lon_deg, when_utc=t)
            err = min(
                abs((pred_shadow - shadow_azimuth_world_deg) % 360.0),
                abs((shadow_azimuth_world_deg - pred_shadow) % 360.0),
            )
            if err < best_err:
                best_err = err
                best_t = t
        if best_t is not None and best_err <= tolerance_deg:
            return best_t
        return best_t if best_t and best_err <= 45.0 else None


def _detect_shadow_direction_from_image(
    image_path: str | Path | None = None,
    image_array: object = None,
    *,
    crop_top_ratio: float = 0.3,
    camera_facing_deg: float = 0.0,
) -> tuple[float, float] | None:
    """
    Analyze upper region of image for edge/contrast to infer shadow direction.
    Returns (dominant_azimuth_world_deg, confidence) or None.
    Image frame: 0° = right, 90° = down. Map to world using camera_facing_deg (0=north).
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    img = None
    if image_path is not None:
        path = Path(image_path) if not isinstance(image_path, Path) else image_path
        if path.is_file():
            try:
                img = Image.open(path).convert("L")
            except Exception:
                return None
    elif image_array is not None and hasattr(image_array, "shape"):
        try:
            arr = np.asarray(image_array)
            if len(arr.shape) >= 2:
                if len(arr.shape) == 3:
                    arr = np.mean(arr, axis=2)
                img = Image.fromarray(arr.astype(np.uint8))
        except Exception:
            return None
    if img is None:
        return None
    w, h = img.size
    crop_h = int(h * crop_top_ratio)
    if crop_h < 4:
        return None
    top = np.array(img.crop((0, 0, w, crop_h)))
    # Simple gradient magnitude and direction
    gy = np.zeros_like(top, dtype=float)
    gx = np.zeros_like(top, dtype=float)
    gy[:-1, :] = np.diff(top.astype(float), axis=0)
    gx[:, :-1] = np.diff(top.astype(float), axis=1)
    mag = np.sqrt(gx * gx + gy * gy)
    if mag.max() < 1e-6:
        return None
    # Dominant edge direction: perpendicular to gradient points along edge
    angles_deg = np.degrees(np.arctan2(gy, gx))
    angles_deg = (angles_deg + 90.0) % 360.0
    hist, bin_edges = np.histogram(angles_deg.flatten(), bins=36, range=(0, 360), weights=mag.flatten())
    dominant_bin = int(np.argmax(hist))
    dominant_angle_image = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2.0
    confidence = min(0.7, 0.2 + 0.1 * (hist[dominant_bin] / (mag.sum() / 36 + 1e-9)))
    # Map image angle to world: image 0°=right (east if camera north), 90°=down (south)
    # Camera facing north: image right = east = 90°, image left = west = 270°
    # Shadow direction in image: dominant_angle_image is edge direction; shadow points opposite to light
    shadow_image_deg = (dominant_angle_image + 180.0) % 360.0
    world_offset = camera_facing_deg
    world_az = (shadow_image_deg + (90.0 - world_offset)) % 360.0
    return (world_az, _clamp(confidence, 0.0, 1.0))


class SunPlaneSurfaceAnalyzer:
    """
    Fallback 2C estimator:
    infer sun direction from available surface/shadow hints in asset metadata.
    Supports datetime-aware sun position and shadow-geometry time inference.
    """

    def __init__(
        self,
        sun_calculator: type = SunPositionCalculator,
        shadow_solver: ShadowGeometrySolver | None = None,
    ):
        self._sun = sun_calculator
        self._shadow_solver = shadow_solver or ShadowGeometrySolver(sun_calculator=sun_calculator)

    def _extract_lat_lon_when(self, metadata: dict) -> tuple[float | None, float | None, dt.datetime | None]:
        loc = metadata.get("asset_location") or {}
        if isinstance(loc, dict):
            lat = loc.get("lat")
            lon = loc.get("lon")
        else:
            lat = metadata.get("lat")
            lon = metadata.get("lon")
        when = _parse_datetime_utc(
            metadata.get("capture_datetime_utc")
            or metadata.get("capture_datetime")
            or metadata.get("asset_date")
            or metadata.get("timestamp_utc")
        )
        lat_f = float(lat) if lat is not None else None
        lon_f = float(lon) if lon is not None else None
        return lat_f, lon_f, when

    def infer(
        self,
        metadata: dict,
        *,
        lat: float | None = None,
        lon: float | None = None,
        when_utc: dt.datetime | None = None,
    ) -> tuple[tuple[float, float, float] | None, float, list[str], dict]:
        """
        Infer sun direction. Returns (direction_vector, confidence, reasons, extras).
        extras may include: daytime_lit, estimated_capture_time_utc, time_inference_source.
        """
        reasons: list[str] = []
        extras: dict = {}
        lat = lat if lat is not None else (metadata.get("asset_location") or {}).get("lat") if isinstance(metadata.get("asset_location"), dict) else metadata.get("lat")
        lon = lon if lon is not None else (metadata.get("asset_location") or {}).get("lon") if isinstance(metadata.get("asset_location"), dict) else metadata.get("lon")
        when = when_utc or _parse_datetime_utc(metadata.get("capture_datetime_utc") or metadata.get("capture_datetime") or metadata.get("asset_date"))

        # Datetime + lat/lon: use sun position
        if lat is not None and lon is not None and when is not None:
            sun = self._sun.compute(lat_deg=float(lat), lon_deg=float(lon), when_utc=when)
            elevation = float(sun["sun_elevation_deg"])
            extras["daytime_lit"] = elevation > 0.0
            direction = sun["sun_direction_vector_world"]
            reasons.append("sun_position_from_datetime")
            return _norm(direction), 0.85 if elevation > 0 else 0.5, reasons, extras

        # Highest confidence: explicit inferred direction already present.
        direct = _parse_vec3(metadata.get("inferred_sun_direction_vector"))
        if direct:
            return _norm(direct), 0.9, ["metadata_inferred_sun_direction_vector"], extras

        shadow = _parse_vec3(metadata.get("shadow_vector") or metadata.get("dominant_shadow_direction"))
        plane_normal = _parse_vec3(metadata.get("sun_plane_normal") or metadata.get("dominant_surface_normal"))
        if shadow and plane_normal:
            shadow_n = _norm(shadow)
            normal_n = _norm(plane_normal)
            candidate = _norm((-shadow_n[0], abs(normal_n[1]) + 0.2, -shadow_n[2]))
            return candidate, 0.72, ["shadow_vector", "surface_normal"], extras

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
                return avg, _clamp(0.45 + 0.05 * len(samples), 0.45, 0.8), reasons, extras
        return None, 0.0, reasons, extras

    def infer_with_image(
        self,
        metadata: dict,
        *,
        image_path: str | Path | None = None,
        image_array: object = None,
        lat: float | None = None,
        lon: float | None = None,
        when_utc: dt.datetime | None = None,
        enable_shadow_time_inference: bool = True,
        enable_image_edge_analysis: bool = True,
        camera_facing_default: str = "north",
    ) -> tuple[tuple[float, float, float] | None, float, list[str], dict]:
        """
        Infer sun direction, optionally using image edge analysis and shadow geometry time estimation.
        Returns (direction_vector, confidence, reasons, extras).
        extras: daytime_lit, estimated_capture_time_utc, time_inference_source.
        """
        lat_f, lon_f, when = self._extract_lat_lon_when(metadata)
        lat = lat or lat_f
        lon = lon or lon_f
        when = when or when_utc
        extras: dict = {}
        camera_facing = _camera_facing_to_azimuth_deg(
            metadata.get("camera_facing_deg") or metadata.get("camera_facing") or camera_facing_default
        )

        # Image edge analysis
        detected_shadow_az: float | None = None
        if enable_image_edge_analysis and (image_path or image_array):
            det = _detect_shadow_direction_from_image(
                image_path=image_path,
                image_array=image_array,
                crop_top_ratio=0.3,
                camera_facing_deg=camera_facing,
            )
            if det:
                detected_shadow_az, edge_conf = det
                extras["daytime_lit"] = True
                extras["shadow_detection_confidence"] = edge_conf

        # Time estimation from shadow when date known but time nominal
        estimated_when: dt.datetime | None = None
        if (
            enable_shadow_time_inference
            and lat is not None
            and lon is not None
            and when is not None
            and detected_shadow_az is not None
        ):
            date_utc = when.replace(hour=12, minute=0, second=0, microsecond=0)
            est = self._shadow_solver.estimate_time_from_shadow_direction(
                lat_deg=lat,
                lon_deg=lon,
                date_utc=date_utc,
                shadow_azimuth_world_deg=detected_shadow_az,
                camera_facing_deg=camera_facing,
            )
            if est is not None:
                estimated_when = est
                extras["estimated_capture_time_utc"] = est.isoformat().replace("+00:00", "Z")
                extras["time_inference_source"] = "shadow_geometry"
                when = est

        # Primary inference with lat/lon/when
        direction, conf, reasons, infer_extras = self.infer(metadata, lat=lat, lon=lon, when_utc=when)
        extras.update(infer_extras)

        if detected_shadow_az is not None and "shadow_detection_confidence" not in extras:
            extras["shadow_detection_confidence"] = 0.4

        if estimated_when is not None and direction is None and lat is not None and lon is not None:
            sun = self._sun.compute(lat_deg=lat, lon_deg=lon, when_utc=estimated_when)
            direction = sun["sun_direction_vector_world"]
            reasons.append("sun_position_from_shadow_time_estimate")
            conf = max(conf, 0.6)

        return direction, conf, reasons, extras


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

    def compute(self, *, lat: float, lon: float, altitude_m: float | None = None, metadata: dict | None = None, image_path: str | Path | None = None) -> dict:
        metadata = dict(metadata or {})
        rollout = metadata.get("lighting_rollout_flags")
        if not isinstance(rollout, dict):
            rollout = {}
        enable_surface_inference = _to_bool(rollout.get("enable_surface_inference"), True)
        enable_ml_validation = _to_bool(rollout.get("enable_ml_validation"), True)
        enable_lunar_context = _to_bool(rollout.get("enable_lunar_context"), True)
        enable_shadow_time_inference = _to_bool(rollout.get("enable_shadow_time_inference"), True)
        enable_image_edge_analysis = _to_bool(rollout.get("enable_image_edge_analysis"), True)
        camera_facing_default = str(rollout.get("camera_facing_default") or "north")

        when_utc = self._extract_datetime(metadata)
        existing_loc = metadata.get("asset_location") or {}
        if not isinstance(existing_loc, dict):
            existing_loc = {}
        meta_with_loc = {**metadata, "asset_location": {**existing_loc, "lat": lat, "lon": lon}}
        meta_with_loc["capture_datetime_utc"] = when_utc.isoformat().replace("+00:00", "Z")

        sun = self._sun.compute(lat_deg=lat, lon_deg=lon, when_utc=when_utc)
        expected_direction = sun["sun_direction_vector_world"]
        moon = self._moon.compute(lat_deg=lat, lon_deg=lon, when_utc=when_utc) if enable_lunar_context else None

        md_vec, md_conf = self._extract_metadata_direction(metadata)
        inferred_vec, inferred_conf, inferred_reasons, infer_extras = (None, 0.0, [], {})
        if enable_surface_inference:
            img_path = image_path
            if img_path is None and (metadata.get("image_path") or metadata.get("file_path")):
                img_path = Path(metadata.get("image_path") or metadata.get("file_path") or "")
            if img_path is not None and not isinstance(img_path, Path):
                img_path = Path(img_path)
            if (enable_shadow_time_inference or enable_image_edge_analysis) and img_path is not None and Path(img_path).is_file():
                result = self._analyzer.infer_with_image(
                    meta_with_loc,
                    image_path=img_path,
                    lat=lat,
                    lon=lon,
                    when_utc=when_utc,
                    enable_shadow_time_inference=enable_shadow_time_inference,
                    enable_image_edge_analysis=enable_image_edge_analysis,
                    camera_facing_default=camera_facing_default,
                )
                inferred_vec, inferred_conf, inferred_reasons, infer_extras = result
            else:
                result = self._analyzer.infer(meta_with_loc, lat=lat, lon=lon, when_utc=when_utc)
                inferred_vec, inferred_conf, inferred_reasons, infer_extras = result

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
        daytime_lit = infer_extras.get("daytime_lit")
        if daytime_lit is None:
            daytime_lit = sun_visible
        moon_elevation = float(moon["moon_elevation_deg"]) if moon is not None else None
        moon_visible = bool(moon_elevation is not None and moon_elevation > 0.0)

        out: dict = {
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
            "daytime_lit": bool(daytime_lit),
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
                "enable_shadow_time_inference": enable_shadow_time_inference,
                "enable_image_edge_analysis": enable_image_edge_analysis,
                "camera_facing_default": camera_facing_default,
            },
            **validation,
        }
        if infer_extras.get("estimated_capture_time_utc"):
            out["estimated_capture_time_utc"] = infer_extras["estimated_capture_time_utc"]
        if infer_extras.get("time_inference_source"):
            out["time_inference_source"] = infer_extras["time_inference_source"]
        if infer_extras.get("shadow_detection_confidence") is not None:
            out["shadow_detection_confidence"] = round(float(infer_extras["shadow_detection_confidence"]), 4)
        return out

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
