"""Bundle results, portable and complete models, and reproducibility files."""
from pathlib import Path
import json
import os
import zipfile

ROOT = Path(__file__).resolve().parent
target = ROOT.parent / "four_database_bertopic_results.zip"
temporary = target.with_suffix(".zip.tmp")
support = ["README.md", "requirements-core.txt", "run.ps1", "run_pipeline.py", "prepare_and_embed.py", "text_utils.py", "analyze.py", "report.py", "export_tables.mjs", "finalize_models.py", "verify_outputs.py", "package_results.py", "topic_names.json", "make_figures.py", "figure_ops.py", "test_figure_ops.py"]
files = [ROOT / name for name in support] + sorted(path for path in (ROOT / "outputs").rglob("*") if path.is_file() and not any(part.startswith(".figure_") for part in path.relative_to(ROOT / "outputs").parts))
assert target.resolve().parent == ROOT.parent.resolve()
with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
    for file in files:
        archive.write(file, str(Path("bertopic_analysis") / file.relative_to(ROOT)))
with zipfile.ZipFile(temporary) as archive:
    assert len(archive.infolist()) == len(files)
    assert archive.testzip() is None, "Archive CRC validation failed"
os.replace(temporary, target)
print(json.dumps({"archive": str(target), "files": len(files), "bytes": target.stat().st_size, "crc_validation": "passed"}))
