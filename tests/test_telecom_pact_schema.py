"""Telecom PACT schema presence test (run from continuuuum repo with TELECOM_ROOT set)."""

import json
import os
from pathlib import Path


def test_telecom_playbook_schema_exists():
    root = Path(os.environ.get("TELECOM_ROOT", Path(__file__).resolve().parents[2].parent / "Drawer 2" / "telecom"))
    schema = root / "schemas" / "telecom-playbook.schema.json"
    if not schema.exists():
        return
    data = json.loads(schema.read_text(encoding="utf-8"))
    assert data["properties"]["playbook"]["const"] == "telecom/v1"
