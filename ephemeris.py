"""
Ephemeris provider abstraction for multi-body astral lighting.
SPICE-first with Horizons flat-file fallback for offline/bootstrap.
"""

from __future__ import annotations

import datetime as dt
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unified_semantic_archiver.db import ContinuumDb


@dataclass
class BodyState:
    """Position and velocity of a body at an epoch."""

    body_id: str
    epoch_utc: dt.datetime
    position: tuple[float, float, float]  # km, frame-dependent (typically J2000/ICRF)
    velocity: tuple[float, float, float]  # km/s
    frame_id: str = "J2000"

    def position_km(self) -> tuple[float, float, float]:
        return self.position

    def velocity_km_s(self) -> tuple[float, float, float]:
        return self.velocity


def _parse_epoch_utc(value: str) -> dt.datetime:
    """Parse ISO or Horizons-style epoch string to UTC datetime."""
    value = (value or "").strip()
    if not value:
        raise ValueError("empty epoch")
    # Horizons: "2451545.00000000 = A.D. 2000-Jan-01 12:00:00.0000 TDB"
    match = re.search(r"A\.D\.\s+(\d{4})-(\w{3})-(\d{1,2})\s+(\d{2}):(\d{2}):(\d{2})", value)
    if match:
        months = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                  "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
        y, mon_str, d, h, m, s = match.groups()
        mon = months.get(mon_str, 1)
        return dt.datetime(int(y), mon, int(d), int(h), int(m), int(s), tzinfo=dt.timezone.utc)
    # ISO
    if "Z" in value or "+" in value or "-" in value[-6:]:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = dt.datetime.fromisoformat(value + "+00:00")
    return parsed.astimezone(dt.timezone.utc)


class IEphemerisProvider(ABC):
    """Abstract ephemeris provider."""

    @abstractmethod
    def get_body_state(self, body_id: str, epoch_utc: dt.datetime, frame_id: str = "J2000") -> BodyState | None:
        """Return body state at epoch, or None if unavailable."""
        ...

    def get_many_body_states(
        self,
        body_ids: list[str],
        epoch_utc: dt.datetime,
        frame_id: str = "J2000",
    ) -> dict[str, BodyState]:
        """Return states for multiple bodies. Default implementation calls get_body_state per body."""
        out: dict[str, BodyState] = {}
        for bid in body_ids:
            state = self.get_body_state(bid, epoch_utc, frame_id)
            if state is not None:
                out[bid] = state
        return out


# SPICE provider (optional)
try:
    import spiceypy  # type: ignore[import-untyped]
    _SPICE_AVAILABLE = True
except ImportError:
    _SPICE_AVAILABLE = False
    spiceypy = None  # type: ignore[assignment]


def _body_id_to_spice_naif(body_id: str) -> int:
    """Map common body IDs to NAIF IDs. Extend as needed."""
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html
    mapping = {
        "sun": 10,
        "mercury": 199,
        "venus": 299,
        "earth": 399,
        "moon": 301,
        "mars": 499,
        "jupiter": 599,
        "saturn": 699,
        "uranus": 799,
        "neptune": 899,
    }
    key = body_id.lower().strip()
    if key in mapping:
        return mapping[key]
    try:
        return int(body_id)
    except ValueError:
        raise ValueError(f"Unknown body_id: {body_id}")


class SpiceEphemerisProvider(IEphemerisProvider):
    """SPICE-backed ephemeris provider. Requires spiceypy and loaded kernels."""

    def __init__(self, kernel_paths: list[str | Path] | None = None):
        if not _SPICE_AVAILABLE:
            raise RuntimeError("spiceypy is not installed; pip install spiceypy")
        self._loaded: set[str] = set()
        for p in (kernel_paths or []):
            path = str(Path(p).resolve())
            if path not in self._loaded:
                spiceypy.furnsh(path)  # type: ignore[union-attr]
                self._loaded.add(path)

    def load_kernel(self, path: str | Path) -> None:
        path = str(Path(path).resolve())
        if path not in self._loaded:
            spiceypy.furnsh(path)  # type: ignore[union-attr]
            self._loaded.add(path)

    def get_body_state(self, body_id: str, epoch_utc: dt.datetime, frame_id: str = "J2000") -> BodyState | None:
        try:
            naif_id = _body_id_to_spice_naif(body_id)
            et = spiceypy.utc2et(epoch_utc.strftime("%Y-%m-%dT%H:%M:%S"))  # type: ignore[union-attr]
            state, _ = spiceypy.spkezr(str(naif_id), et, frame_id, "NONE", "0")  # type: ignore[union-attr]
            pos = (float(state[0]), float(state[1]), float(state[2]))
            vel = (float(state[3]), float(state[4]), float(state[5]))
            return BodyState(body_id=body_id, epoch_utc=epoch_utc, position=pos, velocity=vel, frame_id=frame_id)
        except Exception:
            return None


class HorizonsFlatFileProvider(IEphemerisProvider):
    """
    Parse JPL Horizons flat-file vectors ($$SOE ... $$EOE format).
    Use for offline/bootstrap when SPICE is unavailable.
    """

    def __init__(self, file_path: str | Path | None = None, body_id: str = "earth"):
        self._file_path = Path(file_path) if file_path else None
        self._body_id = body_id
        self._cache: dict[str, BodyState] = {}  # epoch_iso -> BodyState

    def load_from_path(self, path: str | Path, body_id: str | None = None) -> int:
        """Parse a Horizons output file. Returns count of loaded epochs."""
        path = Path(path)
        if not path.is_file():
            return 0
        bid = body_id or self._body_id
        count = 0
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        # $$SOE ... $$EOE blocks
        soe = re.findall(
            r"\$\$SOE\s+(.*?)\s+\$\$EOE",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        for block in soe:
            # Epoch line: "2451545.00000000 = A.D. 2000-Jan-01 12:00:00.0000 TDB"
            epoch_match = re.search(
                r"(\d+\.?\d*)\s*=\s*(A\.D\.\s+\d{4}-\w{3}-\d{1,2}\s+\d{2}:\d{2}:\d{2}[^\s]*)",
                block,
            )
            # X,Y,Z and VX,VY,VZ
            pos_match = re.search(
                r"X\s*=\s*([-\d.Ee+]+)\s+Y\s*=\s*([-\d.Ee+]+)\s+Z\s*=\s*([-\d.Ee+]+)",
                block,
            )
            vel_match = re.search(
                r"VX\s*=\s*([-\d.Ee+]+)\s+VY\s*=\s*([-\d.Ee+]+)\s+VZ\s*=\s*([-\d.Ee+]+)",
                block,
            )
            if epoch_match and pos_match and vel_match:
                try:
                    epoch_str = epoch_match.group(2).strip()
                    epoch_dt = _parse_epoch_utc(epoch_str)
                    px, py, pz = float(pos_match.group(1)), float(pos_match.group(2)), float(pos_match.group(3))
                    vx, vy, vz = float(vel_match.group(1)), float(vel_match.group(2)), float(vel_match.group(3))
                    key = epoch_dt.isoformat()
                    self._cache[key] = BodyState(
                        body_id=bid,
                        epoch_utc=epoch_dt,
                        position=(px, py, pz),
                        velocity=(vx, vy, vz),
                        frame_id="J2000",
                    )
                    count += 1
                except (ValueError, TypeError):
                    continue
        return count

    def get_body_state(self, body_id: str, epoch_utc: dt.datetime, frame_id: str = "J2000") -> BodyState | None:
        if body_id != self._body_id and not self._cache:
            return None
        # Exact match
        key = epoch_utc.isoformat()
        if key in self._cache:
            s = self._cache[key]
            if body_id == s.body_id:
                return s
        # Nearest epoch (linear interpolation if two neighbors)
        best_before: BodyState | None = None
        best_after: BodyState | None = None
        t = epoch_utc.timestamp()
        for s in self._cache.values():
            if s.body_id != body_id:
                continue
            st = s.epoch_utc.timestamp()
            if st <= t and (best_before is None or st > best_before.epoch_utc.timestamp()):
                best_before = s
            if st >= t and (best_after is None or st < best_after.epoch_utc.timestamp()):
                best_after = s
        if best_before is None and best_after is None:
            return None
        if best_before is None:
            return best_after
        if best_after is None:
            return best_before
        # Interpolate
        t0, t1 = best_before.epoch_utc.timestamp(), best_after.epoch_utc.timestamp()
        if t1 <= t0:
            return best_before
        alpha = (t - t0) / (t1 - t0)
        p0, p1 = best_before.position, best_after.position
        v0, v1 = best_before.velocity, best_after.velocity
        pos = (
            p0[0] + alpha * (p1[0] - p0[0]),
            p0[1] + alpha * (p1[1] - p0[1]),
            p0[2] + alpha * (p1[2] - p0[2]),
        )
        vel = (
            v0[0] + alpha * (v1[0] - v0[0]),
            v0[1] + alpha * (v1[1] - v0[1]),
            v0[2] + alpha * (v1[2] - v0[2]),
        )
        return BodyState(body_id=body_id, epoch_utc=epoch_utc, position=pos, velocity=vel, frame_id=frame_id)


class DbEphemerisProvider(IEphemerisProvider):
    """Read from USC ephemeris_samples with interpolation."""

    def __init__(self, db: ContinuumDb, tenant_id: str = "default"):
        self._db = db
        self._tenant_id = tenant_id

    def get_body_state(self, body_id: str, epoch_utc: dt.datetime, frame_id: str = "J2000") -> BodyState | None:
        epoch_str = epoch_utc.strftime("%Y-%m-%dT%H:%M:%S")
        rows = self._db.ephemeris_sample_list_near_epoch(
            body_id=body_id,
            epoch_utc=epoch_str,
            tenant_id=self._tenant_id,
            limit=5,
        )
        if not rows:
            return None
        # Find before/after
        t = epoch_utc.timestamp()
        best_before = None
        best_after = None
        for r in rows:
            try:
                et = dt.datetime.fromisoformat(str(r["epoch_utc"]).replace("Z", "+00:00"))
                if et.tzinfo is None:
                    et = et.replace(tzinfo=dt.timezone.utc)
                st = et.timestamp()
                pos = (float(r["position_x"]), float(r["position_y"]), float(r["position_z"]))
                vel = (
                    float(r.get("velocity_x") or 0),
                    float(r.get("velocity_y") or 0),
                    float(r.get("velocity_z") or 0),
                )
                s = BodyState(body_id=body_id, epoch_utc=et, position=pos, velocity=vel, frame_id=str(r.get("frame_id") or frame_id))
                if st <= t and (best_before is None or st > best_before.epoch_utc.timestamp()):
                    best_before = s
                if st >= t and (best_after is None or st < best_after.epoch_utc.timestamp()):
                    best_after = s
            except (KeyError, ValueError, TypeError):
                continue
        if best_before is None and best_after is None:
            return None
        if best_before is None:
            return best_after
        if best_after is None:
            return best_before
        t0, t1 = best_before.epoch_utc.timestamp(), best_after.epoch_utc.timestamp()
        if t1 <= t0:
            return best_before
        alpha = (t - t0) / (t1 - t0)
        p0, p1 = best_before.position, best_after.position
        v0, v1 = best_before.velocity, best_after.velocity
        pos = (
            p0[0] + alpha * (p1[0] - p0[0]),
            p0[1] + alpha * (p1[1] - p0[1]),
            p0[2] + alpha * (p1[2] - p0[2]),
        )
        vel = (
            v0[0] + alpha * (v1[0] - v0[0]),
            v0[1] + alpha * (v1[1] - v0[1]),
            v0[2] + alpha * (v1[2] - v0[2]),
        )
        return BodyState(body_id=body_id, epoch_utc=epoch_utc, position=pos, velocity=vel, frame_id=frame_id)


class ChainedEphemerisProvider(IEphemerisProvider):
    """Try providers in order: SPICE -> DB -> Horizons."""

    def __init__(self, providers: list[IEphemerisProvider]):
        self._providers = providers

    def get_body_state(self, body_id: str, epoch_utc: dt.datetime, frame_id: str = "J2000") -> BodyState | None:
        for p in self._providers:
            state = p.get_body_state(body_id, epoch_utc, frame_id)
            if state is not None:
                return state
        return None


def create_default_ephemeris_provider(
    db: ContinuumDb | None = None,
    kernel_paths: list[str | Path] | None = None,
    horizons_path: str | Path | None = None,
    tenant_id: str = "default",
) -> IEphemerisProvider:
    """
    Create SPICE-first provider with DB and Horizons fallbacks.
    """
    providers: list[IEphemerisProvider] = []
    if _SPICE_AVAILABLE and kernel_paths:
        try:
            providers.append(SpiceEphemerisProvider(kernel_paths))
        except Exception:
            pass
    if db:
        providers.append(DbEphemerisProvider(db, tenant_id))
    if horizons_path and Path(horizons_path).is_file():
        h = HorizonsFlatFileProvider()
        if h.load_from_path(horizons_path, "earth") > 0:
            providers.append(h)
    if not providers:
        # Last-resort: Horizons with empty cache (will return None until loaded)
        providers.append(HorizonsFlatFileProvider(body_id="earth"))
    return ChainedEphemerisProvider(providers)
