"""Versioned JSONL subprocess interface. stdout is reserved for protocol events."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from . import __version__
from .common import Cancelled, WorkflowError, dependency_issues, read_json, runtime_versions, write_json
from .config import FIELDS, InputFile, ImportOptions, WorkflowConfig, load_config
from .runtime import Events, cancel, package_run, run


def capabilities():
    return {'engine': 'dsh-bertopic-workflow', 'engine_version': __version__, 'protocol_version': 1,
            'commands': ['capabilities', 'schema', 'doctor', 'inspect', 'validate', 'run', 'status', 'cancel', 'verify', 'package'],
            'sources': ['VIP', 'WanFang', 'CNKI', 'WoS', 'PubMed', 'Generic'],
            'mapping_fields': sorted(FIELDS),
            'formats': ['csv', 'tsv', 'xlsx', 'xls', 'html_table', 'medline', 'nbib', 'ris', 'pubmed_xml'],
            'embedding_backends': ['siliconflow', 'openai_compatible', 'local', 'precomputed'],
            'stages': ['ingest', 'embed', 'model', 'report', 'validation'],
            'cancellation': 'cooperative; clustering/reduction stops at the next stage boundary',
            'resume': 'config/input/code fingerprint and sealed-stage hash checks',
            'figures': 15, 'missing_metadata': 'not_applicable with reason',
            'gpu_required': False, 'official_dsh_registration': 'pending host SDK specification'}


def inspect_file(path, source='auto', **options):
    from .ingest import read_input
    file = Path(path).resolve()
    if not file.is_file() or file.stat().st_size > 512 * 1024**2:
        raise WorkflowError('INPUT_INVALID', '文件不存在或超过默认检查上限。')
    _, metadata = read_input(InputFile(path=str(file), source=source, **options), ImportOptions())
    return metadata


def validate_config(path):
    from .ingest import read_input
    config = load_config(path)
    files = []
    for item in config.inputs:
        _, metadata = read_input(item, config.import_options)
        files.append({'file': Path(item.path).name, **metadata})
    return {'status': 'valid', 'input_files': files, 'input_records': sum(f['records'] for f in files),
            'network_requests': 0, 'credential_environment': config.embedding.api_key_env
            if config.embedding.backend in {'siliconflow', 'openai_compatible'} else None}


def parser():
    root = argparse.ArgumentParser(description='可追溯的多数据库 BERTopic 工作流；stdout 为 JSONL 事件。')
    root.add_argument('--version', action='version', version=__version__)
    commands = root.add_subparsers(dest='command', required=True)
    commands.add_parser('capabilities')
    schema = commands.add_parser('schema')
    schema.add_argument('--output', type=Path)
    commands.add_parser('doctor')
    inspect = commands.add_parser('inspect')
    inspect.add_argument('--input', type=Path, required=True)
    inspect.add_argument('--source', default='auto', choices=['auto', 'VIP', 'WanFang', 'CNKI', 'WoS', 'PubMed', 'Generic'])
    inspect.add_argument('--sheet')
    inspect.add_argument('--header-row', type=int)
    inspect.add_argument('--encoding')
    inspect.add_argument('--delimiter')
    validate = commands.add_parser('validate')
    validate.add_argument('--config', type=Path, required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--config', type=Path, required=True)
    execute.add_argument('--run-id')
    execute.add_argument('--resume', action='store_true')
    execute.add_argument('--recover-lock', action='store_true')
    execute.add_argument('--until', choices=['ingest', 'embed', 'model', 'report'])
    execute.add_argument('--offline', action='store_true')
    for name in ['status', 'cancel', 'verify']:
        cmd = commands.add_parser(name)
        cmd.add_argument('--run-dir', type=Path, required=True)
    package = commands.add_parser('package')
    package.add_argument('--run-dir', type=Path, required=True)
    package.add_argument('--destination', type=Path, required=True)
    return root


def main(argv=None):
    # The JSONL protocol and diagnostics must survive non-UTF-8 Windows consoles.
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass
    args = parser().parse_args(argv)
    events = Events()
    try:
        if args.command == 'capabilities':
            events.emit('capabilities', **capabilities())
        elif args.command == 'schema':
            schema = WorkflowConfig.model_json_schema()
            if args.output:
                write_json(args.output, schema)
                events.emit('schema_written', path=str(args.output.resolve()))
            else:
                events.emit('schema', schema=schema)
        elif args.command == 'doctor':
            import importlib.metadata
            versions = runtime_versions()
            issues = dependency_issues()
            local_issues = dependency_issues(('local',))
            try:
                metadata_version = importlib.metadata.version('dsh-bertopic-workflow')
            except importlib.metadata.PackageNotFoundError:
                metadata_version = None
            versions.update(status='passed' if not issues else 'dependency_issues', issues=issues,
                            local_extra_issues=local_issues,
                            version_source='distribution-metadata' if metadata_version == __version__
                            else 'source-fallback',
                            version_consistent=metadata_version is None or metadata_version == __version__,
                            gpu_required=False, credential_values_logged=False)
            events.emit('doctor', **versions)
            return 0 if not issues else 2
        elif args.command == 'inspect':
            events.emit('inspected', **inspect_file(args.input, args.source, sheet=args.sheet,
                        header_row=args.header_row, encoding=args.encoding, delimiter=args.delimiter))
        elif args.command == 'validate':
            events.emit('config_validated', **validate_config(args.config))
        elif args.command == 'run':
            run(args.config, run_id=args.run_id, resume=args.resume, recover_lock=args.recover_lock,
                until=args.until, offline=args.offline)
        elif args.command == 'status':
            events.emit('job_status', **read_json(args.run_dir / 'job.json'))
        elif args.command == 'cancel':
            events.emit('cancel_requested', **cancel(args.run_dir))
        elif args.command == 'verify':
            from .validation import verify_release
            events.emit('release_verified', **verify_release(args.run_dir))
        elif args.command == 'package':
            events.emit('package_created', **package_run(args.run_dir, args.destination))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if isinstance(error, (Cancelled, KeyboardInterrupt)):
            events.emit('error', code='CANCELLED', message='任务已取消。')
            return 5
        code = error.code if isinstance(error, WorkflowError) else 'INTERNAL'
        events.emit('error', code=code, message=str(error) if isinstance(error, WorkflowError)
                    else '任务失败，请检查配置、依赖或已保存的任务状态。',
                    details=error.details if isinstance(error, WorkflowError) else {'type': type(error).__name__})
        return 4 if code == 'JOB_LOCKED' else 2 if code.startswith(('CONFIG', 'INPUT', 'HEADER', 'MAPPING')) else 3


if __name__ == '__main__':
    sys.exit(main())
