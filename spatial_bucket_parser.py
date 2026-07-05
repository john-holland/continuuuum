"""Parse BedogaGenerator bucket IDs (Q*, O*, S*.O*) into nested trees for d3."""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable


_QUAD_RE = re.compile(r"^Q([\d.]+)$")
_OCT_RE = re.compile(r"^O([\d.]+)$")
_SLICE_OCT_RE = re.compile(r"^S(\d+)\.O([\d.]+)$")


def parse_bucket_id(bucket_id: str) -> dict[str, Any] | None:
    """Return { kind, bucketId, segments, sliceIndex? } or None if unrecognized."""
    if not bucket_id:
        return None
    bid = bucket_id.strip()
    m = _SLICE_OCT_RE.match(bid)
    if m:
        oct_path = [int(x) for x in m.group(2).split(".") if x]
        return {
            "kind": "4d",
            "bucketId": bid,
            "sliceIndex": int(m.group(1)),
            "segments": [f"S{m.group(1)}"] + [f"O{p}" for p in _expand_path(oct_path, "O")],
            "pathIndices": oct_path,
        }
    m = _OCT_RE.match(bid)
    if m:
        path = [int(x) for x in m.group(1).split(".") if x]
        return {
            "kind": "oct",
            "bucketId": bid,
            "segments": _expand_path(path, "O"),
            "pathIndices": path,
        }
    m = _QUAD_RE.match(bid)
    if m:
        path = [int(x) for x in m.group(1).split(".") if x]
        return {
            "kind": "quad",
            "bucketId": bid,
            "segments": _expand_path(path, "Q"),
            "pathIndices": path,
        }
    return None


def _expand_path(indices: list[int], prefix: str) -> list[str]:
    """Build cumulative segment names: Q2, Q2.1, Q2.1.3 from [2,1,3]."""
    out: list[str] = []
    parts: list[str] = []
    for idx in indices:
        parts.append(str(idx))
        out.append(prefix + ".".join(parts))
    return out


def _node_key(segments: list[str], up_to: int) -> str:
    return "/".join(segments[: up_to + 1])


def build_bucket_tree(
    bucket_ids: Iterable[str],
    *,
    bucket_annotations: dict[str, dict[str, Any]] | None = None,
    tree_type: str | None = None,
) -> dict[str, Any]:
    """
    Build nested tree for d3.hierarchy.

    bucket_annotations: bucket_id -> { maskSynonyms: [], assetIds: [], uscAssetIds: [] }
    tree_type: 'quad' | 'oct' | '4d' | None (all)
    """
    annotations = bucket_annotations or {}
    nodes: dict[str, dict[str, Any]] = {}

    def ensure_node(key: str, name: str, bucket_id: str | None, kind: str | None) -> dict[str, Any]:
        if key not in nodes:
            nodes[key] = {
                "name": name,
                "id": bucket_id or key,
                "bucketId": bucket_id,
                "kind": kind,
                "maskSynonyms": set(),
                "assetIds": set(),
                "uscAssetIds": set(),
                "children": {},
            }
        return nodes[key]

    root = {"name": "root", "id": "root", "bucketId": None, "kind": None, "children": {}}

    for raw in bucket_ids:
        parsed = parse_bucket_id(raw)
        if not parsed:
            continue
        if tree_type and parsed["kind"] != tree_type:
            continue
        ann = annotations.get(raw, {})
        synonyms = ann.get("maskSynonyms") or ann.get("mask_synonyms") or []
        asset_ids = ann.get("assetIds") or ann.get("asset_ids") or []
        usc_ids = ann.get("uscAssetIds") or ann.get("usc_asset_ids") or []

        segments = parsed["segments"]
        parent_children = root["children"]
        path_keys: list[str] = []
        for i, seg in enumerate(segments):
            path_keys.append(seg)
            key = _node_key(segments, i)
            is_leaf = i == len(segments) - 1
            if key not in parent_children:
                parent_children[key] = ensure_node(
                    key,
                    seg,
                    parsed["bucketId"] if is_leaf else None,
                    parsed["kind"] if is_leaf else None,
                )
            node = parent_children[key]
            if is_leaf:
                node["bucketId"] = parsed["bucketId"]
                node["kind"] = parsed["kind"]
                for s in synonyms:
                    node["maskSynonyms"].add(s)
                for a in asset_ids:
                    node["assetIds"].add(str(a))
                for u in usc_ids:
                    node["uscAssetIds"].add(str(u))
            parent_children = node["children"]

    def finalize(branch: dict[str, Any]) -> dict[str, Any]:
        child_map = branch.get("children") or {}
        if isinstance(child_map, dict) and child_map:
            children = [finalize(child_map[k]) for k in sorted(child_map.keys())]
        else:
            children = []
        out: dict[str, Any] = {
            "name": branch.get("name", "root"),
            "id": branch.get("id") or branch.get("name", "root"),
            "bucketId": branch.get("bucketId"),
            "kind": branch.get("kind"),
            "maskSynonyms": sorted(branch.get("maskSynonyms") or []),
            "assetIds": sorted(branch.get("assetIds") or []),
            "uscAssetIds": sorted(branch.get("uscAssetIds") or []),
        }
        if children:
            out["children"] = children
        return out

    return finalize(root)


def collect_bucket_ids_for_tree_type(bucket_ids: Iterable[str], tree_type: str | None) -> list[str]:
    out: list[str] = []
    for bid in bucket_ids:
        parsed = parse_bucket_id(bid)
        if not parsed:
            continue
        if tree_type and parsed["kind"] != tree_type:
            continue
        out.append(parsed["bucketId"])
    return sorted(set(out))
