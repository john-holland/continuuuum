#!/usr/bin/env python3
"""
One-shot rebrand: continuum / Continuum / CONTINUUM -> continuuuum / Continuuuum / CONTINUUUUM.

Phases (deepest paths first):
  1. Rename files whose names contain the token
  2. Rename directories whose names contain the token
  3. Replace content in text files (three case-specific passes)

Usage:
  python tools/rebrand_continuum_to_continuuuum.py [--dry-run] [repo_root ...]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SKIP_DIR_NAMES = {
    "node_modules",
    "Library",
    "PackageCache",
    "site-packages",
    ".git",
    "__pycache__",
    ".venv",
    ".venv312",
    "Artifacts",
    "ArtifactDB",
    "Temp",
}

SKIP_DIR_PREFIXES = (".venv",)

TEXT_EXTENSIONS = {
    ".py", ".cs", ".js", ".jsx", ".ts", ".tsx", ".json", ".yaml", ".yml", ".md",
    ".html", ".css", ".sql", ".sh", ".ps1", ".mjs", ".cjs", ".asmdef", ".unity",
    ".meta", ".txt", ".svg", ".xml", ".sln", ".slnx", ".toml", ".cfg", ".ini",
    ".env", ".example", ".asmref", ".gradle", ".properties", ".vue", ".dockerfile",
    ".gitignore", ".babelrc", ".editorconfig", ".npmrc", ".nuspec", ".props",
    ".targets", ".workflow", ".gitattributes", ".cursorignore", ".cursorrules",
    ".shader", ".compute", ".hlsl", ".cginc", ".prefab", ".asset", ".mat",
    ".controller", ".anim", ".inputactions", ".uxml", ".uss", ".tscn", ".gd",
    ".rs", ".go", ".java", ".kt", ".swift", ".rb", ".php", ".erb", ".h", ".cpp",
    ".hpp", ".cc", ".mm", ".plist", ".manifest", ".lock", ".sum", ".mod",
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".tga", ".psd",
    ".fbx", ".obj", ".blend", ".wav", ".mp3", ".ogg", ".aiff", ".ttf", ".otf",
    ".woff", ".woff2", ".eot", ".pdf", ".zip", ".gz", ".tar", ".7z", ".rar",
    ".dll", ".exe", ".so", ".dylib", ".pdb", ".mdb", ".db", ".sqlite", ".sqlite3",
    ".unitypackage", ".jar", ".class", ".pyc", ".pyo", ".bin", ".dat", ".wasm",
    ".mp4", ".mov", ".avi", ".mkv", ".glb", ".gltf", ".hdr", ".exr", ".cubemap",
    ".aac", ".ogv",
}

SKIP_RENAME_NAMES = {
    "rebrand_continuum_to_continuuuum.py",
}


def rebrand_segment(name: str) -> str:
    if name in SKIP_RENAME_NAMES:
        return name
    if "CONTINUUM" in name:
        name = name.replace("CONTINUUM", "CONTINUUUUM")
    if "Continuum" in name:
        name = name.replace("Continuum", "Continuuuum")
    if "continuum" in name:
        name = name.replace("continuum", "continuuuum")
    return name


def rebrand_content(text: str) -> str:
    text = text.replace("CONTINUUM", "CONTINUUUUM")
    text = text.replace("Continuum", "Continuuuum")
    text = text.replace("continuum", "continuuuum")
    return text


def should_skip_dir(path: Path) -> bool:
    name = path.name
    if name in SKIP_DIR_NAMES:
        return True
    for prefix in SKIP_DIR_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


def iter_all_paths(root: Path):
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if not should_skip_dir(Path(dirpath) / d)]
        base = Path(dirpath)
        for fn in filenames:
            yield base / fn
        for dn in dirnames:
            yield base / dn


def paths_needing_rename(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in iter_all_paths(root):
        if p.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            continue
        new_name = rebrand_segment(p.name)
        if new_name != p.name:
            out.append(p)
    out.sort(key=lambda x: len(x.parts), reverse=True)
    return out


def is_text_file(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in BINARY_EXTENSIONS:
        return False
    if ext in TEXT_EXTENSIONS:
        return True
    if ext == "" and path.name.startswith("."):
        return True
    return ext in {".md", ".json", ".yaml", ".yml", ".html", ".js", ".css", ".py", ".cs"}


def in_git_repo(root: Path) -> bool:
    return (root / ".git").is_dir()


def git_mv(src: Path, dst: Path, cwd: Path, dry_run: bool) -> None:
    rel_src = src.relative_to(cwd)
    rel_dst = dst.relative_to(cwd)
    if dry_run:
        print(f"  git mv {rel_src} {rel_dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "mv", str(rel_src), str(rel_dst)],
        cwd=str(cwd),
        check=True,
    )


def safe_rename(src: Path, dst: Path, repo_root: Path, dry_run: bool) -> None:
    if src == dst:
        return
    if dst.exists():
        raise FileExistsError(f"Target already exists: {dst}")
    print(f"  rename {src.relative_to(repo_root)} -> {dst.relative_to(repo_root)}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    use_git = in_git_repo(repo_root)
    if use_git:
        try:
            git_mv(src, dst, repo_root, dry_run=False)
            return
        except subprocess.CalledProcessError:
            pass
    src.rename(dst)


def rename_paths(root: Path, dry_run: bool) -> int:
    count = 0
    for p in paths_needing_rename(root):
        new_name = rebrand_segment(p.name)
        dst = p.with_name(new_name)
        safe_rename(p, dst, root, dry_run)
        count += 1
    return count


def replace_file_contents(root: Path, dry_run: bool) -> int:
    count = 0
    for p in iter_all_paths(root):
        if not p.is_file():
            continue
        if not is_text_file(p):
            continue
        try:
            raw = p.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:8192]:
            continue
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            continue
        if "continuum" not in text and "Continuum" not in text and "CONTINUUM" not in text:
            continue
        new_text = rebrand_content(text)
        if new_text == text:
            continue
        rel = p.relative_to(root)
        print(f"  content {rel}")
        count += 1
        if not dry_run:
            p.write_text(new_text, encoding="utf-8" if enc == "utf-8-sig" else enc)
    return count


def process_repo(root: Path, dry_run: bool) -> None:
    root = root.resolve()
    if not root.is_dir():
        print(f"Skip missing repo: {root}")
        return
    print(f"\n=== {root} ===")
    n_paths = rename_paths(root, dry_run)
    n_content = replace_file_contents(root, dry_run)
    print(f"  paths renamed: {n_paths}, files content-updated: {n_content}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebrand continuum -> continuuuum")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "repos",
        nargs="*",
        default=[
            r"c:\Users\John\unified-semantic-compressor",
            r"c:\Users\John\continuuuum",
            r"c:\Users\John\Drawer 2",
            r"D:\Development\resaurce",
            r"D:\Development\log-view-machine",
        ],
    )
    args = parser.parse_args()
    for repo in args.repos:
        process_repo(Path(repo), args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
