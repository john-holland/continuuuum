"""Unit tests for spatial bucket ID parsing and tree building."""

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from spatial_bucket_parser import build_bucket_tree, parse_bucket_id


def test_parse_quad():
    p = parse_bucket_id("Q2.1.3")
    assert p is not None
    assert p["kind"] == "quad"
    assert p["bucketId"] == "Q2.1.3"
    assert p["segments"] == ["Q2", "Q2.1", "Q2.1.3"]


def test_parse_oct():
    p = parse_bucket_id("O2.1.7")
    assert p is not None
    assert p["kind"] == "oct"
    assert p["segments"][-1] == "O2.1.7"


def test_parse_4d():
    p = parse_bucket_id("S3.O2.1.7")
    assert p is not None
    assert p["kind"] == "4d"
    assert p["sliceIndex"] == 3
    assert p["bucketId"] == "S3.O2.1.7"


def test_build_tree_merges_buckets():
    tree = build_bucket_tree(
        ["Q2.1.3", "Q2.1.5"],
        bucket_annotations={
            "Q2.1.3": {"maskSynonyms": ["ladder"]},
            "Q2.1.5": {"maskSynonyms": ["door"]},
        },
        tree_type="quad",
    )
    assert tree["name"] == "root"
    assert len(tree["children"]) == 1
    q2 = tree["children"][0]
    assert q2["name"] == "Q2"
    leaves = [c for c in q2.get("children", []) if c.get("bucketId")]
    assert any(c["bucketId"] == "Q2.1.3" and "ladder" in c["maskSynonyms"] for c in _all_nodes(tree))


def _all_nodes(node):
    yield node
    for c in node.get("children") or []:
        yield from _all_nodes(c)


def test_tree_type_filter():
    tree = build_bucket_tree(["Q1", "O1", "S1.O1"], tree_type="oct")
    ids = [n.get("bucketId") for n in _all_nodes(tree) if n.get("bucketId")]
    assert ids == ["O1"]
