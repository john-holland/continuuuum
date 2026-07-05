"""Read-only spatial API: vocabulary_render_masks, episode_assets, bucket trees."""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

from flask import Flask, jsonify, request

from spatial_bucket_parser import build_bucket_tree, parse_bucket_id


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _mask_row(row: sqlite3.Row, buckets: list[str]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "tenantId": row["tenant_id"],
        "assetSynonym": row["asset_synonym"],
        "episodeId": row["episode_id"],
        "bucketIds": buckets,
    }


def _load_masks(
    conn: sqlite3.Connection,
    *,
    tenant_id: str,
    episode_id: str | None,
    synonym: str | None,
    mask_id: str | None = None,
) -> list[dict[str, Any]]:
    if not _table_exists(conn, "vocabulary_render_masks"):
        return []
    sql = """
        SELECT m.id, m.tenant_id, m.asset_synonym, m.episode_id
        FROM vocabulary_render_masks m
        WHERE m.tenant_id = ?
    """
    params: list[Any] = [tenant_id]
    if mask_id:
        sql += " AND m.id = ?"
        params.append(mask_id)
    if episode_id:
        sql += " AND (m.episode_id = ? OR m.episode_id IS NULL)"
        params.append(episode_id)
    if synonym:
        sql += " AND m.asset_synonym LIKE ?"
        params.append(f"%{synonym}%")
    sql += " ORDER BY m.asset_synonym"
    rows = conn.execute(sql, params).fetchall()
    out: list[dict[str, Any]] = []
    has_buckets = _table_exists(conn, "vocabulary_render_mask_buckets")
    for row in rows:
        buckets: list[str] = []
        if has_buckets:
            bcur = conn.execute(
                "SELECT bucket_id FROM vocabulary_render_mask_buckets WHERE mask_id = ? ORDER BY bucket_id",
                (row["id"],),
            )
            buckets = [r["bucket_id"] for r in bcur.fetchall()]
        out.append(_mask_row(row, buckets))
    return out


def _load_episode_assets(conn: sqlite3.Connection, episode_id: str) -> list[dict[str, Any]]:
    if not _table_exists(conn, "episode_assets"):
        return []
    cur = conn.execute(
        """SELECT id, episode_id, usc_asset_id, asset_type, role, causality_leaf_id
           FROM episode_assets WHERE episode_id = ? ORDER BY usc_asset_id""",
        (episode_id,),
    )
    return [
        {
            "id": r["id"],
            "episodeId": r["episode_id"],
            "uscAssetId": r["usc_asset_id"],
            "assetType": r["asset_type"],
            "role": r["role"],
            "causalityLeafId": r["causality_leaf_id"],
        }
        for r in cur.fetchall()
    ]


def _build_annotations(
    masks: list[dict[str, Any]],
    assets: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    ann: dict[str, dict[str, Any]] = {}

    def touch(bucket_id: str) -> dict[str, Any]:
        if bucket_id not in ann:
            ann[bucket_id] = {
                "maskSynonyms": set(),
                "assetIds": set(),
                "uscAssetIds": set(),
            }
        return ann[bucket_id]

    for m in masks:
        syn = m.get("assetSynonym") or ""
        for bid in m.get("bucketIds") or []:
            touch(bid)["maskSynonyms"].add(syn)

    for a in assets:
        leaf = a.get("causalityLeafId")
        if not leaf:
            continue
        entry = touch(leaf)
        entry["assetIds"].add(a.get("id", ""))
        entry["uscAssetIds"].add(str(a.get("uscAssetId", "")))

    # Convert sets to lists for JSON downstream
    return {
        bid: {
            "maskSynonyms": sorted(v["maskSynonyms"]),
            "assetIds": sorted(x for x in v["assetIds"] if x),
            "uscAssetIds": sorted(x for x in v["uscAssetIds"] if x),
        }
        for bid, v in ann.items()
    }


def register_spatial_routes(app: Flask, get_db_path: Callable[[], str]) -> None:
    @app.route("/api/spatial/render-masks", methods=["GET"])
    def list_render_masks():
        tenant_id = request.args.get("tenantId", "default")
        episode_id = request.args.get("episodeId")
        synonym = request.args.get("synonym")
        try:
            conn = _connect(get_db_path())
            items = _load_masks(
                conn,
                tenant_id=tenant_id,
                episode_id=episode_id,
                synonym=synonym,
            )
            conn.close()
            return jsonify({"items": items}), 200
        except sqlite3.Error as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/spatial/render-masks/<mask_id>", methods=["GET"])
    def get_render_mask(mask_id: str):
        tenant_id = request.args.get("tenantId", "default")
        try:
            conn = _connect(get_db_path())
            items = _load_masks(
                conn,
                tenant_id=tenant_id,
                episode_id=None,
                synonym=None,
                mask_id=mask_id,
            )
            conn.close()
            if not items:
                return jsonify({"error": "not found"}), 404
            return jsonify(items[0]), 200
        except sqlite3.Error as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/spatial/episode-assets", methods=["GET"])
    def list_episode_assets():
        episode_id = request.args.get("episodeId")
        if not episode_id:
            return jsonify({"error": "episodeId required"}), 400
        try:
            conn = _connect(get_db_path())
            items = _load_episode_assets(conn, episode_id)
            conn.close()
            return jsonify({"items": items}), 200
        except sqlite3.Error as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/spatial/bucket-tree", methods=["GET"])
    def get_bucket_tree():
        episode_id = request.args.get("episodeId")
        tree_type = request.args.get("treeType")  # quad | oct | 4d | None
        tenant_id = request.args.get("tenantId", "default")
        if tree_type and tree_type not in ("quad", "oct", "4d"):
            return jsonify({"error": "treeType must be quad, oct, or 4d"}), 400
        try:
            conn = _connect(get_db_path())
            masks = _load_masks(
                conn,
                tenant_id=tenant_id,
                episode_id=episode_id,
                synonym=None,
            ) if episode_id else []
            assets = _load_episode_assets(conn, episode_id) if episode_id else []
            conn.close()

            bucket_ids: set[str] = set()
            for m in masks:
                for bid in m.get("bucketIds") or []:
                    bucket_ids.add(bid)
            for a in assets:
                if a.get("causalityLeafId"):
                    bucket_ids.add(a["causalityLeafId"])

            annotations = _build_annotations(masks, assets)
            tree = build_bucket_tree(
                sorted(bucket_ids),
                bucket_annotations=annotations,
                tree_type=tree_type,
            )
            return jsonify(
                {
                    "episodeId": episode_id,
                    "treeType": tree_type,
                    "bucketCount": len(bucket_ids),
                    "tree": tree,
                    "bucketIds": sorted(bucket_ids),
                }
            ), 200
        except sqlite3.Error as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/spatial/resolve-highlight", methods=["GET"])
    def resolve_highlight():
        """Map highlight token (doc id, prefab-id, causality leaf) to library doc + bucket."""
        token = request.args.get("token", "").strip()
        episode_id = request.args.get("episodeId")
        if not token:
            return jsonify({"error": "token required"}), 400
        try:
            conn = _connect(get_db_path())
            result: dict[str, Any] = {
                "token": token,
                "libraryDocumentId": None,
                "causalityLeafId": None,
                "bucketId": None,
            }
            # Direct numeric doc id
            if token.isdigit():
                result["libraryDocumentId"] = int(token)
            # Episode asset lookup
            if episode_id and _table_exists(conn, "episode_assets"):
                row = conn.execute(
                    """SELECT usc_asset_id, causality_leaf_id FROM episode_assets
                       WHERE episode_id = ? AND (usc_asset_id = ? OR causality_leaf_id = ?)
                       LIMIT 1""",
                    (episode_id, token, token),
                ).fetchone()
                if row:
                    usc = str(row["usc_asset_id"])
                    if usc.isdigit():
                        result["libraryDocumentId"] = int(usc)
                    if row["causality_leaf_id"]:
                        result["causalityLeafId"] = row["causality_leaf_id"]
                        result["bucketId"] = row["causality_leaf_id"]
            if parse_bucket_id(token):
                result["bucketId"] = token
                result["causalityLeafId"] = token
            conn.close()
            return jsonify(result), 200
        except sqlite3.Error as e:
            return jsonify({"error": str(e)}), 500
