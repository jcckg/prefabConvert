#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_user_or_command(value: str) -> Path | None:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return candidate.resolve()
    resolved = shutil.which(value)
    if resolved:
        return Path(resolved).resolve()
    return None


def _platform_blender_candidates() -> list[Path]:
    if sys.platform == "darwin":
        return [
            Path("/Applications/Blender.app/Contents/MacOS/Blender"),
            Path("/Applications/Blender.app/Contents/MacOS/blender"),
            Path("~/Applications/Blender.app/Contents/MacOS/Blender").expanduser(),
            Path("~/Applications/Blender.app/Contents/MacOS/blender").expanduser(),
        ]
    if sys.platform.startswith("win"):
        roots = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        ]
        candidates: list[Path] = []
        for root in roots:
            if not root:
                continue
            base = Path(root) / "Blender Foundation"
            if not base.exists():
                continue
            for version_dir in sorted(base.glob("Blender*"), reverse=True):
                exe = version_dir / "blender.exe"
                if exe.exists():
                    candidates.append(exe)
        return candidates
    return [
        Path("/usr/bin/blender"),
        Path("/usr/local/bin/blender"),
        Path("/snap/bin/blender"),
        Path("/var/lib/flatpak/exports/bin/org.blender.Blender"),
    ]


def resolve_blender_binary(blender_bin: str = "") -> Path:
    if blender_bin:
        candidate = Path(blender_bin).expanduser()
        if not candidate.is_absolute():
            raise RuntimeError(f"--blender-path must be absolute: {blender_bin}")
        if candidate.exists():
            return candidate.resolve()
        raise RuntimeError(f"Blender binary not found: {candidate}")

    for env_var in ("BLENDER_BIN", "BLENDER_PATH"):
        env_value = os.environ.get(env_var, "").strip()
        if not env_value:
            continue
        resolved = _resolve_user_or_command(env_value)
        if resolved:
            return resolved

    for command in ("blender", "Blender"):
        resolved = shutil.which(command)
        if resolved:
            return Path(resolved).resolve()

    for candidate in _platform_blender_candidates():
        if candidate.exists():
            return candidate.resolve()

    raise RuntimeError(
        "Blender binary not found. Install Blender, add it to PATH, set BLENDER_BIN, or pass --blender-path."
    )


def run_blender_merge(blender_bin: str, input_root: Path, output_fbx: Path) -> None:
    blender_path = resolve_blender_binary(blender_bin)
    script_path = Path(__file__).resolve()
    output_fbx.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(blender_path),
        "--background",
        "--python",
        str(script_path),
        "--",
        "--mode",
        "merge",
        "--input-root",
        str(input_root),
        "--output",
        str(output_fbx),
    ]
    subprocess.run(cmd, check=True)


def run_blender_batch(blender_bin: str, input_root: Path, output_root: Path) -> None:
    blender_path = resolve_blender_binary(blender_bin)
    script_path = Path(__file__).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(blender_path),
        "--background",
        "--python",
        str(script_path),
        "--",
        "--mode",
        "batch",
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
    ]
    subprocess.run(cmd, check=True)


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description="OBJ to FBX conversion helper for Blender background mode")
    parser.add_argument("--mode", choices=["merge", "batch"], required=True)
    parser.add_argument("--input-root", required=True, help="Root folder containing OBJ files")
    parser.add_argument("--output", default="", help="Output FBX path for merge mode")
    parser.add_argument("--output-root", default="", help="Output FBX root for batch mode")
    return parser.parse_args(argv)


def _reset_scene(bpy) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_obj(bpy, filepath: Path) -> None:
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(filepath))
    else:
        bpy.ops.import_scene.obj(filepath=str(filepath))


def _export_fbx(bpy, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(filepath),
        path_mode="COPY",
        embed_textures=True,
        add_leaf_bones=False,
        use_mesh_modifiers=True,
        use_tspace=True,
        axis_forward="-Z",
        axis_up="Y",
    )


def _blender_main() -> int:
    try:
        import bpy
    except Exception as exc:
        raise RuntimeError("This helper must run inside Blender.") from exc

    args = _parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.exists():
        raise RuntimeError(f"Input root not found: {input_root}")

    obj_files = sorted(input_root.rglob("*.obj"))
    if not obj_files:
        raise RuntimeError(f"No OBJ files found in: {input_root}")

    if args.mode == "merge":
        output = Path(args.output).expanduser().resolve()
        if not args.output:
            raise RuntimeError("--output is required for merge mode")
        _reset_scene(bpy)
        for obj_file in obj_files:
            _import_obj(bpy, obj_file)
        _export_fbx(bpy, output)
        print(f"Merged {len(obj_files)} OBJ file(s) -> {output}")
        return 0

    output_root = Path(args.output_root).expanduser().resolve()
    if not args.output_root:
        raise RuntimeError("--output-root is required for batch mode")

    total = len(obj_files)
    for index, obj_file in enumerate(obj_files, start=1):
        rel = obj_file.relative_to(input_root)
        output_fbx = (output_root / rel).with_suffix(".fbx")
        _reset_scene(bpy)
        _import_obj(bpy, obj_file)
        _export_fbx(bpy, output_fbx)
        print(f"[{index}/{total}] {rel.as_posix()} -> {output_fbx.relative_to(output_root).as_posix()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_blender_main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
