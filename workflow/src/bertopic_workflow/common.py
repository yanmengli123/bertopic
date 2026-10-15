from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import uuid
from datetime import datetime, timezone

import numpy as np


class WorkflowError(Exception):
    def __init__(self, code, message, *, details=None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class Cancelled(WorkflowError):
    def __init__(self):
        super().__init__("CANCELLED", "任务已取消；已完成的阶段可通过 resume 复用。")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def json_bytes(value):
    return (json.dumps(native(value), ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(native(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, str) and re.match(r"^[\s\uFEFF]*[=+\-@]", value):
        return "'" + value
    return value


def write_table(directory, name, rows, columns=None):
    rows = native(rows)
    write_json(Path(directory) / (name + ".json"), rows)
    columns = columns or list(rows[0]) if rows else columns or []
    with (Path(directory) / (name + ".csv")).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: csv_cell(row.get(key)) for key in columns} for row in rows)


def safe_relative(base, relative):
    base = Path(base).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise WorkflowError("UNSAFE_PATH", "产物路径必须位于任务目录内。")
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise WorkflowError("UNSAFE_PATH", "产物路径指向任务目录之外。")
    return path


def file_entries(directory, *, exclude=()):
    directory = Path(directory).resolve()
    result = []
    for file in sorted(directory.rglob("*")):
        if file.is_symlink() or not file.resolve().is_relative_to(directory):
            raise WorkflowError("UNSAFE_PATH", "产物目录中不允许符号链接。")
        if not file.is_file() or file.relative_to(directory).as_posix() in exclude:
            continue
        result.append({"path": file.relative_to(directory).as_posix(),
                       "bytes": file.stat().st_size, "sha256": sha256(file)})
    return result


def verify_entries(directory, entries):
    seen = set()
    for entry in entries:
        if entry["path"] in seen:
            raise WorkflowError("INTEGRITY", "产物清单出现重复路径。")
        seen.add(entry["path"])
        file = safe_relative(directory, entry["path"])
        if not file.is_file() or file.stat().st_size != entry["bytes"] or sha256(file) != entry["sha256"]:
            raise WorkflowError("INTEGRITY", "产物已缺失或被修改。", details={"path": entry["path"]})


def runtime_versions():
    packages = ["dsh-bertopic-workflow", "numpy", "pandas", "bertopic", "scikit-learn",
                "scipy", "umap-learn", "hdbscan", "jieba", "plotly", "matplotlib",
                "pydantic", "requests", "tokenizers", "safetensors", "openpyxl", "xlrd",
                "huggingface-hub", "sentence-transformers", "transformers", "torch", "pycountry", "psutil"]
    packages += ['numba', 'llvmlite', 'pynndescent']
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    from . import __version__
    versions['dsh-bertopic-workflow'] = __version__
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions}


def dependency_issues(extras=()):
    """Verify installed distributions against the generated runtime schema.

    ``extras`` pulls in optional-dependency groups (e.g. ``('local',)`` adds the
    local-inference stack); the base check covers only the API-capable path so a
    remote-backend install is not forced to carry torch.
    """
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    expected = read_json(Path(__file__).parent / 'schemas/runtime_requirements.json')
    pending = [Requirement(r) for r in expected['dependencies']]
    for group in extras:
        pending.extend(Requirement(r) for r in expected.get('extras', {}).get(group, []))
    seen, issues = set(), []
    environment = {**default_environment(), 'extra': ''}
    while pending:
        requirement = pending.pop()
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        name = canonicalize_name(requirement.name)
        try:
            distribution = importlib.metadata.distribution(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            issues.append({'name': name, 'required': str(requirement.specifier), 'installed': None,
                           'fix': f'python -m pip install "{requirement}"'})
            continue
        if not requirement.specifier.contains(distribution.version, prereleases=True):
            issues.append({'name': name, 'required': str(requirement.specifier),
                           'installed': distribution.version,
                           'fix': f'python -m pip install "{requirement}"'})
        if name not in seen:
            seen.add(name)
            pending.extend(Requirement(r) for r in distribution.requires or [])
    return issues


# Modules whose code participates in each phase's stage key. Changing a module
# only invalidates the stages that actually execute it; unrelated edits (e.g.
# plot tweaks after a two-hour embed) no longer force a full restart.
PHASE_MODULES = {
    'ingest': ('ingest.py', 'metadata.py', 'common.py', 'config.py', '__init__.py'),
    'embed': ('embeddings.py', 'common.py', 'config.py', '__init__.py'),
    'model': ('modeling.py', 'metadata.py', 'common.py', 'config.py', '__init__.py'),
    'report': ('plots.py', 'metadata.py', 'common.py', 'config.py', '__init__.py'),
    'runtime': ('runtime.py', 'validation.py', 'cli.py', 'sdk.py', 'common.py', 'config.py', '__init__.py'),
}


def code_fingerprint(scope=None):
    root = Path(__file__).resolve().parent
    if scope is not None and scope not in PHASE_MODULES:
        raise WorkflowError("FINGERPRINT_SCOPE", "未知的指纹作用域。", details={"scope": scope})
    wanted = PHASE_MODULES.get(scope) if scope is not None else None
    files = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix in {".py", ".json"}
             and (wanted is None or p.name in wanted)}
    return hashlib.sha256(json_bytes(files)).hexdigest()
