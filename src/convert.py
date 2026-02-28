#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from blender_backend import run_blender_batch


IDENTITY4 = np.eye(4, dtype=np.float64)


@dataclass
class TransformData:
    game_object_id: int
    parent_transform_id: int
    local_position: tuple[float, float, float]
    local_rotation: tuple[float, float, float, float]
    local_scale: tuple[float, float, float]


@dataclass
class MeshData:
    name: str
    positions: np.ndarray
    uvs: np.ndarray
    submesh_triangles: list[list[tuple[int, int, int]]]


@dataclass
class MaterialData:
    name: str
    color: tuple[float, float, float, float]
    texture_path: Path | None


@dataclass
class MeshInstance:
    name: str
    mesh_guid: str
    material_guids: list[str]
    world_matrix: np.ndarray


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return cleaned or "object"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Unity prefab content to OBJ, with optional FBX export."
    )
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subdirectories for prefab files",
    )
    parser.add_argument(
        "--project-root",
        default="",
        help="Unity project root containing Assets (auto-detected when omitted)",
    )
    parser.add_argument(
        "--blender-path",
        "--blender",
        dest="blender_path",
        default="",
        help="Absolute path to Blender binary (only used with --fbx; auto-detected when omitted)",
    )
    parser.add_argument(
        "--fbx",
        action="store_true",
        help="Also export FBX files using Blender",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum prefabs to process")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in sorted prefab list")
    return parser.parse_args()


def resolve_project_root(input_path: Path, explicit: str) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "Assets").exists():
            raise RuntimeError(f"Project root does not contain Assets: {root}")
        return root

    current = input_path if input_path.is_dir() else input_path.parent
    for candidate in [current, *current.parents]:
        if (candidate / "Assets").exists():
            return candidate
    raise RuntimeError("Could not detect project root. Provide --project-root.")


def collect_prefabs(input_path: Path, recursive: bool) -> tuple[list[Path], Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".prefab":
            raise RuntimeError(f"Expected .prefab file, got: {input_path}")
        return [input_path], input_path.parent

    prefabs = sorted(input_path.rglob("*.prefab")) if recursive else sorted(input_path.glob("*.prefab"))
    return prefabs, input_path


def parse_unity_yaml_documents(path: Path) -> list[tuple[int, int, dict]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    documents: list[tuple[int, int, dict]] = []
    current_lines: list[str] = []
    class_id = 0
    file_id = 0
    active = False

    for line in text.splitlines():
        if line.startswith("%YAML") or line.startswith("%TAG"):
            continue
        if line.startswith("--- !u!"):
            if active and current_lines:
                parsed = yaml.safe_load("\n".join(current_lines))
                if parsed:
                    documents.append((class_id, file_id, parsed))
            match = re.match(r"--- !u!(\d+)\s+&(\d+)", line)
            if not match:
                active = False
                current_lines = []
                continue
            class_id = int(match.group(1))
            file_id = int(match.group(2))
            current_lines = []
            active = True
            continue
        if active:
            current_lines.append(line)

    if active and current_lines:
        parsed = yaml.safe_load("\n".join(current_lines))
        if parsed:
            documents.append((class_id, file_id, parsed))
    return documents


def parse_guid_from_meta(meta_path: Path) -> str:
    for line in meta_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("guid:"):
            return stripped.split(":", 1)[1].strip()
    return ""


def build_guid_index(assets_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for meta_path in assets_root.rglob("*.meta"):
        guid = parse_guid_from_meta(meta_path)
        if not guid or guid in mapping:
            continue
        asset_path = meta_path.with_suffix("")
        if asset_path.exists():
            mapping[guid] = asset_path
    return mapping


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_vec3(data: dict | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(data, dict):
        return default
    return (
        parse_float(data.get("x"), default[0]),
        parse_float(data.get("y"), default[1]),
        parse_float(data.get("z"), default[2]),
    )


def parse_quat(data: dict | None) -> tuple[float, float, float, float]:
    if not isinstance(data, dict):
        return (0.0, 0.0, 0.0, 1.0)
    return (
        parse_float(data.get("x")),
        parse_float(data.get("y")),
        parse_float(data.get("z")),
        parse_float(data.get("w"), 1.0),
    )


def quat_to_matrix(quat: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] = 1.0 - 2.0 * (yy + zz)
    matrix[0, 1] = 2.0 * (xy - wz)
    matrix[0, 2] = 2.0 * (xz + wy)
    matrix[1, 0] = 2.0 * (xy + wz)
    matrix[1, 1] = 1.0 - 2.0 * (xx + zz)
    matrix[1, 2] = 2.0 * (yz - wx)
    matrix[2, 0] = 2.0 * (xz - wy)
    matrix[2, 1] = 2.0 * (yz + wx)
    matrix[2, 2] = 1.0 - 2.0 * (xx + yy)
    return matrix


def compose_matrix(
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
    scale: tuple[float, float, float],
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_to_matrix(rotation) @ np.diag(np.array(scale, dtype=np.float64))
    matrix[:3, 3] = np.array(position, dtype=np.float64)
    return matrix


def parse_prefab_instances(prefab_path: Path) -> list[MeshInstance]:
    documents = parse_unity_yaml_documents(prefab_path)
    game_object_names: dict[int, str] = {}
    transform_by_id: dict[int, TransformData] = {}
    transform_id_by_game_object: dict[int, int] = {}
    mesh_guid_by_game_object: dict[int, str] = {}
    renderer_materials_by_game_object: dict[int, list[str]] = {}
    renderer_mesh_by_game_object: dict[int, str] = {}

    for class_id, file_id, data in documents:
        if class_id == 1 and "GameObject" in data:
            payload = data["GameObject"]
            game_object_names[file_id] = str(payload.get("m_Name", f"GameObject_{file_id}"))
        elif class_id == 4 and "Transform" in data:
            payload = data["Transform"]
            game_object_id = int(payload.get("m_GameObject", {}).get("fileID", 0))
            parent_id = int(payload.get("m_Father", {}).get("fileID", 0))
            transform = TransformData(
                game_object_id=game_object_id,
                parent_transform_id=parent_id,
                local_position=parse_vec3(payload.get("m_LocalPosition"), (0.0, 0.0, 0.0)),
                local_rotation=parse_quat(payload.get("m_LocalRotation")),
                local_scale=parse_vec3(payload.get("m_LocalScale"), (1.0, 1.0, 1.0)),
            )
            transform_by_id[file_id] = transform
            transform_id_by_game_object[game_object_id] = file_id
        elif class_id == 33 and "MeshFilter" in data:
            payload = data["MeshFilter"]
            game_object_id = int(payload.get("m_GameObject", {}).get("fileID", 0))
            mesh_ref = payload.get("m_Mesh", {})
            mesh_guid_by_game_object[game_object_id] = str(mesh_ref.get("guid", ""))
        elif class_id == 23 and "MeshRenderer" in data:
            payload = data["MeshRenderer"]
            game_object_id = int(payload.get("m_GameObject", {}).get("fileID", 0))
            material_guids = [
                str(entry.get("guid", ""))
                for entry in payload.get("m_Materials", [])
                if isinstance(entry, dict)
            ]
            renderer_materials_by_game_object[game_object_id] = material_guids
        elif class_id == 137 and "SkinnedMeshRenderer" in data:
            payload = data["SkinnedMeshRenderer"]
            game_object_id = int(payload.get("m_GameObject", {}).get("fileID", 0))
            mesh_ref = payload.get("m_Mesh", {})
            renderer_mesh_by_game_object[game_object_id] = str(mesh_ref.get("guid", ""))
            material_guids = [
                str(entry.get("guid", ""))
                for entry in payload.get("m_Materials", [])
                if isinstance(entry, dict)
            ]
            renderer_materials_by_game_object[game_object_id] = material_guids

    world_matrix_cache: dict[int, np.ndarray] = {}

    def world_matrix_for_transform(transform_id: int) -> np.ndarray:
        if transform_id in world_matrix_cache:
            return world_matrix_cache[transform_id]
        transform = transform_by_id.get(transform_id)
        if transform is None:
            return IDENTITY4
        local = compose_matrix(transform.local_position, transform.local_rotation, transform.local_scale)
        if transform.parent_transform_id and transform.parent_transform_id in transform_by_id:
            result = world_matrix_for_transform(transform.parent_transform_id) @ local
        else:
            result = local
        world_matrix_cache[transform_id] = result
        return result

    instances: list[MeshInstance] = []
    for game_object_id, material_guids in renderer_materials_by_game_object.items():
        mesh_guid = renderer_mesh_by_game_object.get(game_object_id) or mesh_guid_by_game_object.get(
            game_object_id, ""
        )
        if not mesh_guid or mesh_guid.startswith("0000000000000000"):
            continue
        transform_id = transform_id_by_game_object.get(game_object_id, 0)
        matrix = world_matrix_for_transform(transform_id) if transform_id else IDENTITY4
        name = game_object_names.get(game_object_id, f"GameObject_{game_object_id}")
        instances.append(
            MeshInstance(
                name=name,
                mesh_guid=mesh_guid,
                material_guids=material_guids,
                world_matrix=matrix,
            )
        )
    return instances


COMPONENT_SIZE = {
    0: 4,
    1: 2,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 1,
    7: 1,
    8: 2,
    9: 2,
    10: 4,
    11: 4,
}


def decode_component(raw: bytes, offset: int, fmt: int) -> float:
    if fmt == 0:
        return float(struct.unpack_from("<f", raw, offset)[0])
    if fmt == 1:
        return float(struct.unpack_from("<e", raw, offset)[0])
    if fmt == 2:
        return float(raw[offset] / 255.0)
    if fmt == 3:
        value = struct.unpack_from("<b", raw, offset)[0]
        return max(-1.0, float(value) / 127.0)
    if fmt == 4:
        return float(struct.unpack_from("<H", raw, offset)[0] / 65535.0)
    if fmt == 5:
        value = struct.unpack_from("<h", raw, offset)[0]
        return max(-1.0, float(value) / 32767.0)
    if fmt == 6:
        return float(raw[offset])
    if fmt == 7:
        return float(struct.unpack_from("<b", raw, offset)[0])
    if fmt == 8:
        return float(struct.unpack_from("<H", raw, offset)[0])
    if fmt == 9:
        return float(struct.unpack_from("<h", raw, offset)[0])
    if fmt == 10:
        return float(struct.unpack_from("<I", raw, offset)[0])
    if fmt == 11:
        return float(struct.unpack_from("<i", raw, offset)[0])
    return 0.0


def decode_vertex_channel(
    raw: bytes,
    vertex_index: int,
    channel: dict,
    streams: list[dict],
) -> tuple[float, ...]:
    stream_index = int(channel.get("stream", 0))
    dimension = int(channel.get("dimension", 0))
    fmt = int(channel.get("format", 0))
    component_size = COMPONENT_SIZE.get(fmt, 4)
    if stream_index < 0 or stream_index >= len(streams) or dimension <= 0:
        return tuple()
    stream = streams[stream_index]
    stride = int(stream.get("stride", 0))
    stream_offset = int(stream.get("offset", 0))
    channel_offset = int(channel.get("offset", 0))
    if stride <= 0:
        return tuple()
    base = stream_offset + vertex_index * stride + channel_offset
    values = []
    for component_idx in range(dimension):
        offset = base + component_idx * component_size
        if offset + component_size > len(raw):
            return tuple()
        values.append(decode_component(raw, offset, fmt))
    return tuple(values)


def decode_mesh_asset(mesh_asset_path: Path) -> MeshData | None:
    raw_text = mesh_asset_path.read_text(encoding="utf-8", errors="ignore")
    documents = parse_unity_yaml_documents(mesh_asset_path)
    mesh_payload = None
    for class_id, _, data in documents:
        if class_id == 43 and "Mesh" in data:
            mesh_payload = data["Mesh"]
            break
    if mesh_payload is None:
        return None

    vertex_data = mesh_payload.get("m_VertexData", {})
    vertex_count = int(vertex_data.get("m_VertexCount", 0))
    channels = vertex_data.get("m_Channels", [])
    streams = vertex_data.get("m_Streams", [])
    typeless_match = re.search(r"^\s*_typelessdata:\s*([0-9a-fA-F]+)\s*$", raw_text, re.MULTILINE)
    if typeless_match:
        raw_hex = typeless_match.group(1)
    else:
        raw_hex = str(vertex_data.get("_typelessdata", "")).strip()
    if not raw_hex or vertex_count <= 0:
        return None
    raw = bytes.fromhex(raw_hex)

    positions = np.zeros((vertex_count, 3), dtype=np.float64)
    uvs = np.zeros((vertex_count, 2), dtype=np.float64)

    position_channel = channels[0] if len(channels) > 0 else {}
    uv_channel = channels[3] if len(channels) > 3 else {}

    for index in range(vertex_count):
        position = decode_vertex_channel(raw, index, position_channel, streams)
        uv = decode_vertex_channel(raw, index, uv_channel, streams)
        if len(position) >= 3:
            positions[index] = [position[0], position[1], position[2]]
        if len(uv) >= 2:
            uvs[index] = [uv[0], uv[1]]

    index_match = re.search(r"^\s*m_IndexBuffer:\s*([0-9a-fA-F]+)\s*$", raw_text, re.MULTILINE)
    if index_match:
        index_buffer_hex = index_match.group(1)
    else:
        index_buffer_hex = str(mesh_payload.get("m_IndexBuffer", "")).strip()
    if not index_buffer_hex:
        return None
    index_bytes = bytes.fromhex(index_buffer_hex)
    index_format = int(mesh_payload.get("m_IndexFormat", 0))
    if index_format == 1:
        bytes_per_index = 4
        unpack_fmt = "<I"
    else:
        bytes_per_index = 2
        unpack_fmt = "<H"

    if len(index_bytes) % bytes_per_index != 0:
        return None
    indices = [
        int(struct.unpack_from(unpack_fmt, index_bytes, offset)[0])
        for offset in range(0, len(index_bytes), bytes_per_index)
    ]

    submesh_triangles: list[list[tuple[int, int, int]]] = []
    for submesh in mesh_payload.get("m_SubMeshes", []):
        topology = int(submesh.get("topology", 0))
        if topology != 0:
            submesh_triangles.append([])
            continue
        first_byte = int(submesh.get("firstByte", 0))
        index_count = int(submesh.get("indexCount", 0))
        start = first_byte // bytes_per_index
        sub_indices = indices[start : start + index_count]
        triangles: list[tuple[int, int, int]] = []
        for tri_offset in range(0, len(sub_indices) - 2, 3):
            a = sub_indices[tri_offset]
            b = sub_indices[tri_offset + 1]
            c = sub_indices[tri_offset + 2]
            triangles.append((a, b, c))
        submesh_triangles.append(triangles)

    return MeshData(
        name=str(mesh_payload.get("m_Name", mesh_asset_path.stem)),
        positions=positions,
        uvs=uvs,
        submesh_triangles=submesh_triangles,
    )


def parse_material_file(material_path: Path, guid_index: dict[str, Path]) -> MaterialData:
    text = material_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    name = material_path.stem
    color = (0.8, 0.8, 0.8, 1.0)
    texture_guid = ""

    color_capture = False
    tex_capture = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("m_Name:"):
            name = stripped.split(":", 1)[1].strip() or name
        if stripped == "name: _Color":
            color_capture = True
            continue
        if stripped == "name: _MainTex":
            tex_capture = True
            continue

        if color_capture and stripped.startswith("second:"):
            match = re.search(
                r"r:\s*([-+0-9.eE]+),\s*g:\s*([-+0-9.eE]+),\s*b:\s*([-+0-9.eE]+),\s*a:\s*([-+0-9.eE]+)",
                stripped,
            )
            if match:
                color = (
                    float(match.group(1)),
                    float(match.group(2)),
                    float(match.group(3)),
                    float(match.group(4)),
                )
            color_capture = False

        if tex_capture and "m_Texture:" in stripped:
            match = re.search(r"guid:\s*([0-9a-fA-F]{32})", stripped)
            if match:
                texture_guid = match.group(1).lower()
            tex_capture = False

    texture_path = guid_index.get(texture_guid) if texture_guid else None
    if texture_path is not None and not texture_path.exists():
        texture_path = None
    return MaterialData(name=name, color=color, texture_path=texture_path)


def load_material_by_guid(
    material_guid: str,
    guid_index: dict[str, Path],
    material_cache: dict[str, MaterialData],
) -> MaterialData:
    if material_guid in material_cache:
        return material_cache[material_guid]
    default = MaterialData(
        name=f"mat_{material_guid[:8] or 'default'}",
        color=(0.8, 0.8, 0.8, 1.0),
        texture_path=None,
    )
    material_path = guid_index.get(material_guid)
    if material_path is None or material_path.suffix.lower() != ".mat":
        material_cache[material_guid] = default
        return default
    try:
        material = parse_material_file(material_path, guid_index)
    except Exception:
        material = default
    material_cache[material_guid] = material
    return material


def write_prefab_obj(
    prefab_path: Path,
    output_obj_path: Path,
    instances: list[MeshInstance],
    mesh_cache: dict[str, MeshData],
    guid_index: dict[str, Path],
    material_cache: dict[str, MaterialData],
) -> tuple[int, int]:
    output_obj_path.parent.mkdir(parents=True, exist_ok=True)
    output_mtl_path = output_obj_path.with_suffix(".mtl")

    material_name_by_guid: dict[str, str] = {}
    material_definition_by_name: dict[str, MaterialData] = {}

    for instance in instances:
        for material_guid in instance.material_guids:
            if material_guid in material_name_by_guid:
                continue
            material = load_material_by_guid(material_guid, guid_index, material_cache)
            material_name = slug(f"{material.name}_{material_guid[:8]}")
            if not material_name:
                material_name = f"mat_{len(material_name_by_guid)}"
            material_name_by_guid[material_guid] = material_name
            material_definition_by_name[material_name] = material

    if not material_name_by_guid:
        default_name = "default_mat"
        material_name_by_guid[""] = default_name
        material_definition_by_name[default_name] = MaterialData(
            name=default_name, color=(0.8, 0.8, 0.8, 1.0), texture_path=None
        )

    with output_mtl_path.open("w", encoding="utf-8") as mtl_file:
        for material_name, material in material_definition_by_name.items():
            r, g, b, a = material.color
            mtl_file.write(f"newmtl {material_name}\n")
            mtl_file.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
            mtl_file.write(f"d {a:.6f}\n")
            if material.texture_path is not None:
                relative_texture = os.path.relpath(material.texture_path, output_obj_path.parent)
                mtl_file.write(f"map_Kd {Path(relative_texture).as_posix()}\n")
            mtl_file.write("\n")

    total_meshes = 0
    total_faces = 0
    vertex_offset = 1
    uv_offset = 1

    with output_obj_path.open("w", encoding="utf-8") as obj_file:
        obj_file.write(f"# Source prefab: {prefab_path.as_posix()}\n")
        obj_file.write(f"mtllib {output_mtl_path.name}\n")

        for instance_index, instance in enumerate(instances):
            mesh = mesh_cache.get(instance.mesh_guid)
            if mesh is None:
                continue

            positions = mesh.positions
            uvs = mesh.uvs
            homogeneous = np.concatenate(
                [positions, np.ones((positions.shape[0], 1), dtype=np.float64)], axis=1
            )
            transformed = (homogeneous @ instance.world_matrix.T)[:, :3]

            object_name = slug(f"{instance.name}_{instance_index}_{mesh.name}")
            obj_file.write(f"o {object_name}\n")

            for vertex in transformed:
                obj_file.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
            for uv in uvs:
                obj_file.write(f"vt {uv[0]:.8f} {uv[1]:.8f}\n")

            for submesh_index, triangles in enumerate(mesh.submesh_triangles):
                if not triangles:
                    continue
                material_guid = ""
                if submesh_index < len(instance.material_guids):
                    material_guid = instance.material_guids[submesh_index]
                elif instance.material_guids:
                    material_guid = instance.material_guids[0]
                material_name = material_name_by_guid.get(material_guid) or material_name_by_guid.get("", "default_mat")
                obj_file.write(f"usemtl {material_name}\n")
                for a, b, c in triangles:
                    if a >= len(positions) or b >= len(positions) or c >= len(positions):
                        continue
                    va = vertex_offset + a
                    vb = vertex_offset + b
                    vc = vertex_offset + c
                    ta = uv_offset + a
                    tb = uv_offset + b
                    tc = uv_offset + c
                    obj_file.write(f"f {va}/{ta} {vb}/{tb} {vc}/{tc}\n")
                    total_faces += 1

            vertex_offset += len(positions)
            uv_offset += len(uvs)
            total_meshes += 1

    return total_meshes, total_faces


def run_conversion_pipeline(args: argparse.Namespace, input_path: Path, output_path: Path) -> int:
    prefabs, prefab_base = collect_prefabs(input_path, args.recursive)
    if args.start_index:
        prefabs = prefabs[args.start_index :]
    if args.limit > 0:
        prefabs = prefabs[: args.limit]
    if not prefabs:
        raise RuntimeError("No prefab files found.")

    project_root = resolve_project_root(input_path, args.project_root)
    assets_root = project_root / "Assets"

    if output_path.suffix.lower() == ".fbx":
        raise RuntimeError("Conversion outputs multiple files. Use an output directory.")

    output_path.mkdir(parents=True, exist_ok=True)
    obj_root = output_path / "obj"
    obj_root.mkdir(parents=True, exist_ok=True)

    print(f"Building GUID index from {assets_root} ...")
    guid_index = build_guid_index(assets_root)
    print(f"GUID entries: {len(guid_index)}")

    mesh_cache: dict[str, MeshData] = {}
    material_cache: dict[str, MaterialData] = {}

    converted_prefabs = 0
    skipped_prefabs = 0
    total_meshes = 0
    total_faces = 0

    for index, prefab_path in enumerate(prefabs, start=1):
        relative = prefab_path.relative_to(prefab_base)
        output_obj_path = (obj_root / relative).with_suffix(".obj")
        try:
            instances = parse_prefab_instances(prefab_path)
            if not instances:
                skipped_prefabs += 1
                continue

            filtered_instances: list[MeshInstance] = []
            for instance in instances:
                if instance.mesh_guid.startswith("0000000000000000"):
                    continue
                if instance.mesh_guid not in mesh_cache:
                    mesh_path = guid_index.get(instance.mesh_guid)
                    if mesh_path is None:
                        continue
                    mesh = decode_mesh_asset(mesh_path)
                    if mesh is None:
                        continue
                    mesh_cache[instance.mesh_guid] = mesh
                filtered_instances.append(instance)

            if not filtered_instances:
                skipped_prefabs += 1
                continue

            mesh_count, face_count = write_prefab_obj(
                prefab_path,
                output_obj_path,
                filtered_instances,
                mesh_cache,
                guid_index,
                material_cache,
            )
            if mesh_count == 0:
                skipped_prefabs += 1
                continue

            converted_prefabs += 1
            total_meshes += mesh_count
            total_faces += face_count
            print(
                f"[{index}/{len(prefabs)}] {relative.as_posix()} -> "
                f"{output_obj_path.relative_to(output_path).as_posix()} "
                f"(meshes={mesh_count}, faces={face_count})"
            )
        except Exception as exc:
            skipped_prefabs += 1
            print(f"[{index}/{len(prefabs)}] Failed {relative.as_posix()}: {exc}")

    print(
        f"OBJ export done. converted={converted_prefabs}, skipped={skipped_prefabs}, "
        f"mesh_cache={len(mesh_cache)}, total_meshes={total_meshes}, total_faces={total_faces}"
    )

    if converted_prefabs == 0:
        raise RuntimeError("No prefabs converted to OBJ.")

    if not args.fbx:
        print("Skipping FBX export. Use --fbx to generate FBX files.")
        return 0

    fbx_root = output_path / "fbx"
    print("Running Blender batch FBX export ...")
    run_blender_batch(args.blender_path, obj_root, fbx_root)
    print(f"FBX export done: {fbx_root}")

    return 0


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise RuntimeError(f"Input path not found: {input_path}")

    return run_conversion_pipeline(args, input_path, output_path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
