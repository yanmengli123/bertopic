"""Offline figure build contracts, transactional publication and integrity checks."""
from __future__ import annotations

import ctypes
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

SCHEMA_VERSION = 2


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def parse_selection(value):
    if not value:
        return list(range(1, 16))
    tokens = [part.strip() for part in value.split(",")]
    if any(not token.isdigit() or not 1 <= int(token) <= 15 for token in tokens):
        raise ValueError("--only 必须是 1–15 的逗号分隔编号，例如 1,4,9")
    return list(dict.fromkeys(map(int, tokens)))


def process_alive(pid):
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: conservatively active
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class BuildLock:
    def __init__(self, directory, recover=False):
        self.path = Path(directory) / ".figure_build.lock"
        self.recover = recover

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.recover:
            existing = json.loads(self.path.read_text(encoding="utf-8"))
            if not process_alive(int(existing["pid"])):
                self.path.unlink()
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError("已有绘图进程锁；确认进程结束后可使用 --recover-lock 恢复") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, stream)
        return self

    def __exit__(self, *_):
        self.path.unlink(missing_ok=True)


def input_fingerprint(root, directory):
    files = list(sorted(directory.glob("*.json")))
    files = [path for path in files if not path.name.startswith(("figure", "metadata_", "topic_lexical"))]
    files += list(sorted(directory.glob("*.npy")))
    files += [root / "topic_names.json", root / "outputs/data_manifest.json",
              root / "outputs/joint_bge_m3_embeddings.npy",
              root / "outputs/joint_bge_m3_embedding_metadata.json", root / "text_utils.py"]
    # Schema/CSV files are derivative exports, not analysis inputs.
    files = [path for path in files if not path.name.endswith(".schema.json")]
    hashes = {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in files}
    data_manifest = json.loads((root / "outputs/data_manifest.json").read_text(encoding="utf-8"))
    for filename, expected in data_manifest["source_sha256"].items():
        raw_path = Path(data_manifest["input_directory"]) / filename
        raw_hash = sha256(raw_path)
        if raw_hash != expected:
            raise ValueError(f"{filename} 已改变：拒绝按旧记录顺序关联元数据")
        hashes["source:" + filename] = raw_hash
    for filename in ("document_topics.json", "topic_labels.json"):
        extra = root / "outputs/chinese_bge_zh" / filename
        hashes[extra.relative_to(root).as_posix()] = sha256(extra)
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return digest, hashes


def preflight(ctx, root):
    data = json.loads((root / "outputs/data_manifest.json").read_text(encoding="utf-8"))
    embedding = json.loads((root / "outputs/joint_bge_m3_embedding_metadata.json").read_text(encoding="utf-8"))
    df = ctx.df
    if not df.record_id.is_unique or len(df) != data["rows"]:
        raise ValueError("文档 ID 不唯一或文档数与数据清单不一致")
    if df.record_id.tolist() != embedding["record_ids"]:
        raise ValueError("嵌入矩阵与文档顺序不一致")
    expected_sources = {row["source"]: row["records"] for row in data["sources"]}
    if df.source.value_counts().to_dict() != expected_sources:
        raise ValueError("数据库记录数与数据清单不一致")
    assigned = df.loc[df.topic.ge(0)].topic.value_counts().sort_index().to_dict()
    if ctx.summary.set_index("topic")["count"].sort_index().to_dict() != assigned:
        raise ValueError("主题汇总与逐文档计数不一致")
    if set(ctx.ids) != set(assigned) or set(ctx.ids) != set(ctx.keywords.topic):
        raise ValueError("主题 ID、关键词或主题汇总不完整")
    n = len(ctx.ids)
    if ctx.sim.shape != (n, n) or not np.allclose(ctx.sim, ctx.sim.T, atol=1e-6):
        raise ValueError("主题相似度矩阵不是对称方阵")
    if not np.isfinite(ctx.sim).all() or not np.allclose(np.diag(ctx.sim), 1, atol=1e-5):
        raise ValueError("主题相似度矩阵包含无效值")
    if ctx.linkage.shape != (n - 1, 4):
        raise ValueError("主题层次矩阵尺寸不正确")
    vectors = np.load(root / "outputs/joint_bge_m3_embeddings.npy", mmap_mode="r")
    if vectors.shape != (len(df), embedding["dimensions"]):
        raise ValueError("嵌入矩阵尺寸不正确")
    for start in range(0, len(vectors), 2048):
        chunk = vectors[start:start + 2048]
        if not np.isfinite(chunk).all() or not np.allclose(np.linalg.norm(chunk, axis=1), 1, atol=1e-5):
            raise ValueError("嵌入包含无效值或未归一化")
    if not np.isfinite(df[["umap_1", "umap_2"]].to_numpy()).all():
        raise ValueError("UMAP 坐标包含无效值")
    return {"records": len(df), "topics": n, "source_counts": expected_sources,
            "assigned": int(df.topic.ge(0).sum()), "outliers": int(df.topic.eq(-1).sum()),
            "embedding_alignment": "passed", "embedding_finite_and_norms": "passed"}


def artifact_info(base, path):
    relative = path.relative_to(base).as_posix()
    item = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
    if path.suffix == ".png":
        with Image.open(path) as picture:
            picture.verify()
        with Image.open(path) as picture:
            item["width"], item["height"] = picture.size
        if item["width"] < 400 or item["height"] < 250:
            raise ValueError(f"图片尺寸异常: {relative}")
    elif path.suffix == ".svg":
        tree = ET.parse(path)
        if not tree.getroot().tag.endswith("svg"):
            raise ValueError(f"无效 SVG: {relative}")
        item["contains_raster"] = any(node.tag.endswith("image") for node in tree.iter())
    elif path.suffix == ".pdf":
        if not path.read_bytes().startswith(b"%PDF-"):
            raise ValueError(f"无效 PDF: {relative}")
    elif path.suffix == ".html":
        page = path.read_text(encoding="utf-8")
        if "<html" not in page or "</html>" not in page:
            raise ValueError(f"不完整 HTML: {relative}")
        if "Plotly.newPlot" in page and "plotly.min.js" not in page:
            raise ValueError(f"离线 Plotly 依赖缺失: {relative}")
    elif path.suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"))
    return item


def validate_release(base, manifest, check_hash=True):
    errors = []
    listed = set()
    for item in manifest["artifacts"]:
        relative = item["path"]
        path = (base / relative).resolve()
        if not path.is_relative_to(base.resolve()) or relative in listed:
            errors.append(f"异常或重复文件路径: {relative}")
            continue
        listed.add(relative)
        if not path.is_file():
            errors.append(f"文件缺失: {relative}")
        elif check_hash and (path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]):
            errors.append(f"文件校验失败: {relative}")
    for chart in manifest["charts"]:
        for suffix in ("png", "svg", "pdf"):
            relative = f"figures/{chart['id']}.{suffix}"
            if relative not in listed:
                errors.append(f"主图格式缺失: {relative}")
    topics = manifest["topic_details"]
    if len(topics) != manifest["topics"] or len({row["topic"] for row in topics}) != len(topics):
        errors.append("独立主题图覆盖不完整")
    for row in topics:
        for relative in row["files"]:
            if relative not in listed:
                errors.append(f"主题独立图未列入产物清单: {relative}")
    if manifest["interactive_contracts"]["fig05_document_map"]["documents"] != manifest["records"]:
        errors.append("文档交互图记录覆盖不完整")
    if any(path.startswith("visualizations/") and path.endswith(".html") for path in listed):
        if "visualizations/plotly.min.js" not in listed:
            errors.append("离线共享 JS 未列入清单")
    if errors:
        raise ValueError("; ".join(errors))
    return {"status": "passed", "files": len(listed), "topics": len(topics),
            "interactive_documents": manifest["records"], "hashes_checked": check_hash}


def make_gallery(base, manifest, catalog, extras_catalog):
    escape = html.escape
    available = {item["path"] for item in manifest["artifacts"]}
    cards = []
    for name, title, description in catalog:
        files = [(f"figures/{name}.{suffix}", suffix.upper()) for suffix in ("png", "svg", "pdf")]
        files.append((f"visualizations/{name}.html", "交互图"))
        for extra, label in extras_catalog.get(name, []):
            files.append((f"visualizations/{extra}.html", label))
            files += [(f"figures/{extra}.{suffix}", label + " " + suffix.upper()) for suffix in ("png", "svg", "pdf")]
        links = " · ".join(f'<a href="{escape(path)}">{escape(label)}</a>' for path, label in files if path in available)
        cards.append(f'<section class="card" data-search="{escape(title + description)}"><h2>{escape(title)}</h2><p>{escape(description)}</p><nav>{links}</nav><a href="figures/{name}.png"><img loading="lazy" src="figures/{name}.png" alt="{escape(title)}"></a></section>')
    rows = []
    for topic in manifest["topic_details"]:
        links = " · ".join(f'<a href="{escape(path)}">{escape(Path(path).stem.split("_")[-1] + " " + Path(path).suffix[1:].upper())}</a>' for path in topic["files"])
        rows.append(f'<tr data-search="{escape(topic["label"])}"><th scope="row">T{topic["topic"]}</th><td>{escape(topic["label"])}</td><td>{topic["records"]:,}</td><td>{links}</td></tr>')
    methods = "".join(f'<li>{escape(value)}</li>' for value in manifest["method_notes"])
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>四库联合 BERTopic 图库</title><style>
    *{{box-sizing:border-box}}body{{margin:0;font:15px/1.7 "Microsoft YaHei",Arial,sans-serif;color:#173242;background:#f3f6f7}}header{{background:#153b4d;color:white;padding:30px max(4vw,20px)}}h1{{font-size:27px;margin:0}}main{{max-width:1320px;margin:auto;padding:20px}}a{{color:#176b87}}header a{{color:#dbf3ff}}.card,.methods,.details{{background:white;border:1px solid #dbe3e7;border-radius:8px;padding:22px;margin:18px 0}}h2{{margin:0;font-size:21px}}p{{margin:8px 0}}nav{{margin:10px 0}}img{{width:100%;height:auto}}input{{width:100%;padding:13px;font:inherit;border:1px solid #91a5b1;border-radius:6px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dbe3e7}}.scroll{{overflow:auto}}[hidden]{{display:none!important}}.muted{{color:#61727e}}footer{{padding:20px;text-align:center;font-size:12px}}a:focus-visible,input:focus-visible{{outline:3px solid #d17a45;outline-offset:3px}}
    </style></head><body><header><h1>四库联合 BERTopic · 15 类主题分析图</h1><p>{manifest['records']:,} 条保留记录 · {manifest['topics']} 个主题 · 离群 {manifest['outliers']:,} 条</p><p><a href="../analysis_report.html">完整分析报告</a> · <a href="figures_manifest.json">产物与 SHA-256 清单</a> · <a href="figures_validation.json">自动校验结果</a></p></header><main><label for="search">搜索图或主题</label><input id="search" type="search" placeholder="例如：时间、T2、淀粉" aria-describedby="result"><p id="result" aria-live="polite"></p><section class="methods"><h2>数据口径与解释边界</h2><ul>{methods}</ul><p class="muted">PNG 300 dpi；SVG/PDF 保留矢量文字与线条，词云及文档散点可能含栅格层。无需联网即可查看交互图。</p></section>{''.join(cards)}<section class="details"><h2>全部 {manifest['topics']} 个主题的独立图</h2><p>每个主题均提供关键词条形图、真实分词频次词云的 PNG / SVG / PDF；保留主题编号与完整名称。</p><div class="scroll"><table><thead><tr><th>主题</th><th>解释性名称</th><th>记录数</th><th>下载</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section></main><footer>生成时间 {escape(manifest['generated_at'])} · 数据指纹 {manifest['data_fingerprint'][:16]}</footer><script>const search=document.getElementById('search');search.addEventListener('input',()=>{{const query=search.value.trim().toLocaleLowerCase();let visible=0;document.querySelectorAll('[data-search]').forEach(el=>{{el.hidden=!el.textContent.toLocaleLowerCase().includes(query);if(!el.hidden)visible++;}});document.getElementById('result').textContent=query?'找到 '+visible+' 项':'';}});</script></body></html>'''
    (base / "figure_gallery.html").write_text(page, encoding="utf-8")


def runtime_versions():
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "plotly", "wordcloud", "Pillow", "pycountry"):
        versions[package] = importlib.metadata.version(package)
    return versions


def publish(stage, destination, files):
    """Publish validated files; rollback every replaced file if publication fails.

    The manifest is the final commit marker. Readers/checkers verify its hashes.
    A process/power loss during the commit is detected by --check-only; rerunning
    rebuilds the suite. Rendering failures never enter this publication phase.
    """
    backup = Path(tempfile.mkdtemp(prefix=".figure_rollback_", dir=destination))
    replaced = []
    try:
        ordered = [value for value in files if value != "figures_manifest.json"] + ["figures_manifest.json"]
        for relative in ordered:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            old = backup / relative
            if target.exists():
                old.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, old)
            os.replace(stage / relative, target)
            replaced.append(relative)
    except BaseException:
        for relative in reversed(replaced):
            target, old = destination / relative, backup / relative
            if old.exists():
                os.replace(old, target)
            else:
                target.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(backup)
