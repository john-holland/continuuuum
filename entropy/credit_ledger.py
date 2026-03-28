"""Credit earn/spend/balance accounting for entropy consumer interface."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from unified_semantic_archiver.db import ContinuumDb


CREDITS_PER_REQUEST = 1  # Cost per request_random call
CREDITS_PER_BYTE = 0.03  # Alternative: ceil(bytes/32) * 1


def _cost_for_bytes(size_bytes: int) -> int:
    """Compute credit cost for a random request of given size."""
    return max(CREDITS_PER_REQUEST, int(size_bytes / 32) + 1)


class CreditLedger:
    """Earn/spend/balance for entropy credits."""

    def __init__(self, db: ContinuumDb):
        self._db = db

    def get_credits(self, tenant_id: str) -> dict[str, Any]:
        """Return balance, earned_total, spent_total for tenant."""
        tenant = (tenant_id or "").strip() or "default"
        row = self._db.entropy_credits_get(tenant)
        if not row:
            return {"balance": 0, "earned_total": 0, "spent_total": 0}
        earned = int(row.get("earned", 0))
        spent = int(row.get("spent", 0))
        return {"balance": earned - spent, "earned_total": earned, "spent_total": spent}

    def earn(self, tenant_id: str, amount: int) -> None:
        """Award credits (e.g. when node contributes to a round)."""
        self._db.entropy_credits_earn(tenant_id or "default", amount)

    def spend(self, tenant_id: str, amount: int) -> bool:
        """Debit credits. Returns True if successful, False if insufficient."""
        return self._db.entropy_credits_spend(tenant_id or "default", amount)

    def cost_for_random(self, size_bytes: int) -> int:
        """Credit cost for a random request of given size."""
        return _cost_for_bytes(size_bytes)
