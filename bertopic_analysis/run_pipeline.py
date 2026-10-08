"""Reproduce all four-database analysis stages without writing API credentials."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="四库联合 BERTopic 分析")
    parser.add_argument("--offline", action="store_true", help="Use existing matching embedding matrices without API access")
    parser.add_argument("--report-only", action="store_true", help="Regenerate report and CSV files from completed model results")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--node", type=Path, default=Path("C:/Users/32110/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"))
    args = parser.parse_args()
    env = os.environ.copy()
    env.update(USE_TORCH="0", USE_TF="0", NUMBA_NUM_THREADS="16", TOKENIZERS_PARALLELISM="false")

    def run(command):
        subprocess.run([str(value) for value in command], cwd=ROOT, env=env, check=True)

    if not args.report_only:
        prepare = [sys.executable, "-u", ROOT / "prepare_and_embed.py"]
        if args.offline:
            prepare.append("--prepare-only")
        else:
            prepare.extend(["--workers", args.workers, "--batch-size", args.batch_size])
        run(prepare)
        # analyze.py rejects stale vectors if any document text or order changed.
        run([sys.executable, "-u", ROOT / "analyze.py"])
    run([sys.executable, "-u", ROOT / "report.py"])
    run([sys.executable, "-u", ROOT / "make_figures.py"])
    run([sys.executable, "-u", ROOT / "finalize_models.py"])
    if not args.node.is_file():
        raise FileNotFoundError("Bundled Node runtime unavailable; provide --node with its absolute path")
    run([args.node, ROOT / "export_tables.mjs"])
    run([sys.executable, "-u", ROOT / "verify_outputs.py"])
    run([sys.executable, "-u", ROOT / "make_figures.py", "--check-only"])
    print("Completed: " + str(ROOT / "outputs" / "analysis_report.html"))


if __name__ == "__main__":
    main()
