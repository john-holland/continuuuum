"""Center hub for daisy topology: receives guesses from petals, aggregates, produces authoritative center value."""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CenterState:
    """Current center guess and prior-round entropy for XOR collapse avoidance."""
    center_value: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    prior_entropy_hash: bytes = field(default_factory=bytes)


class CenterObjectTarget:
    """
    Daisy center hub. Receives center-ring guesses from petals, aggregates,
    produces authoritative center value for next round. Mixes entropy from
    yesterday (prior_entropy_hash) to avoid XOR smoke stack.
    """

    def __init__(self, initial_center: bytes | None = None):
        self._state = CenterState()
        if initial_center:
            self._state.center_value = initial_center
        self._guesses: dict[str, Any] = {}

    def accept_guess(self, node_id: str, guess_payload: dict[str, Any]) -> None:
        """Receive guess from a petal. guess_payload can include 'guess_bytes', 'rtt_mean_ms', etc."""
        self._guesses[node_id] = guess_payload

    def get_authoritative_center(self) -> bytes:
        """Return current center value for handoff to nodes."""
        return self._state.center_value

    def mix_entropy_yesterday(self, prior_entropy_hash: bytes) -> None:
        """Mix historical entropy to avoid XOR collapse (guides away from degenerate output)."""
        self._state.prior_entropy_hash = prior_entropy_hash
        # Fold prior hash into center to decorrelate from predictable paths
        if prior_entropy_hash:
            h = hashlib.sha256(self._state.center_value + prior_entropy_hash).digest()
            self._state.center_value = bytes(a ^ b for a, b in zip(self._state.center_value, h[:32]))

    def advance_round(self) -> bytes:
        """
        Aggregate current guesses into new center value, clear guesses, return new center.
        Call after all petals have submitted for this round. Prior entropy hash is
        stored for next round (entropy from yesterday).
        """
        old_center = self._state.center_value
        if not self._guesses:
            self._state.center_value = secrets.token_bytes(32)
        else:
            parts = []
            for node_id, payload in sorted(self._guesses.items()):
                raw = payload.get("guess_bytes")
                if isinstance(raw, bytes):
                    parts.append(raw)
                elif isinstance(raw, str) and len(raw) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in raw):
                    try:
                        parts.append(bytes.fromhex(raw))
                    except ValueError:
                        parts.append(raw.encode("utf-8"))
                elif isinstance(raw, str):
                    parts.append(raw.encode("utf-8"))
                elif isinstance(raw, (int, float)):
                    parts.append(str(raw).encode("utf-8"))
                else:
                    parts.append(node_id.encode("utf-8"))
            combined = b"".join(parts) + self._state.center_value
            if self._state.prior_entropy_hash:
                combined += self._state.prior_entropy_hash
            self._state.center_value = hashlib.sha256(combined).digest()
        self._state.prior_entropy_hash = hashlib.sha256(old_center).digest()
        self._guesses.clear()
        return self._state.center_value

    def get_prior_entropy_hash(self) -> bytes:
        """Return prior-round entropy hash for nodes to use in their predictor."""
        return self._state.prior_entropy_hash
