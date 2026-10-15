"""Isolated jobs, cooperative cancellation, sealed stages and final commit marker."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
import threading
import uuid
import zipfile

import psutil

from . import __version__
from .common import (Cancelled, WorkflowError, code_fingerprint, dependency_issues, file_entries, json_bytes,
                     read_json, runtime_versions, sha256, utc_now, verify_entries, write_json)
from .config import load_config

PHASES = ['ingest', 'embed', 'model', 'report']
RUN_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$')


class Events:
    def __init__(self, run_id=None, path=None, callback=None):
        self.run_id, self.path, self.callback, self.sequence = run_id, path, callback, 0
        self.stream, self.lock = sys.stdout, threading.RLock()

    def emit(self, event, **data):
        with self.lock:
            self.sequence += 1
            item = {'schema_version': 1, 'event': event, 'run_id': self.run_id,
                    'sequence': self.sequence, 'at': utc_now(), 'data': data}
            line = json.dumps(item, ensure_ascii=False, allow_nan=False)
            if self.path:
                with Path(self.path).open('a', encoding='utf-8') as stream:
                    stream.write(line + '\n')
            if self.callback:
                self.callback(item)
            else:
                self.stream.write(line + '\n')
                self.stream.flush()


class JobLock:
    def __init__(self, run_dir, recover=False):
        self.path = Path(run_dir) / '.run.lock'
        self.recover = recover
        self.token = uuid.uuid4().hex

    def __enter__(self):
        if self.path.exists() and self.recover:
            owner = read_json(self.path)
            if owner['host'] != socket.gethostname():
                raise WorkflowError('JOB_LOCKED', '不能自动解除其他主机的锁。')
            try:
                process = psutil.Process(owner['pid'])
                active = abs(process.create_time() - owner['process_started']) < .01
            except psutil.NoSuchProcess:
                active = False
            except psutil.AccessDenied:
                active = True
            if active:
                raise WorkflowError('JOB_LOCKED', '任务仍在运行，不能恢复锁。')
            self.path.unlink()
        try:
            with self.path.open('x', encoding='utf-8') as stream:
                json.dump({'pid': os.getpid(), 'process_started': psutil.Process().create_time(),
                           'host': socket.gethostname(), 'token': self.token, 'at': utc_now()}, stream)
        except FileExistsError:
            raise WorkflowError('JOB_LOCKED', '同一任务已有运行者；异常退出后可显式 recover-lock。') from None
        return self

    def __exit__(self, *_):
        if self.path.exists() and read_json(self.path).get('token') == self.token:
            self.path.unlink()


def input_hashes(config):
    return {str(i.path): sha256(i.path) for i in config.inputs}


def external_hashes(config, phase):
    if phase != 'embed':
        return {}
    fields = ['vectors_path', 'metadata_path'] if config.embedding.backend == 'precomputed' else ['tokenizer_path']
    result = {}
    for field in fields:
        value = getattr(config.embedding, field)
        if value:
            if not Path(value).is_file():
                raise WorkflowError('EMBEDDING_INPUT', '嵌入依赖文件不存在。', details={'field': field})
            result[field] = sha256(value)
    return result


def run(config_path, *, run_id=None, resume=False, recover_lock=False, until=None, offline=False, on_event=None):
    config = load_config(config_path)
    run_id = run_id or uuid.uuid4().hex
    if not RUN_ID.fullmatch(run_id):
        raise WorkflowError('RUN_ID', 'run_id 只允许字母、数字、连字符和下划线。')
    if until is not None and until not in PHASES:
        raise WorkflowError('PHASE', '未知阶段。')
    root = Path(config.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / run_id
    directory = candidate.resolve()
    if directory != candidate or directory.parent != root or candidate.is_symlink():
        raise WorkflowError('UNSAFE_PATH', '任务目录不安全。')
    existed = directory.exists()
    if existed and not resume:
        raise WorkflowError('RUN_EXISTS', '任务已存在；使用新的 run_id 或显式 resume。')
    if resume and not existed:
        raise WorkflowError('RUN_NOT_FOUND', '要恢复的任务不存在。')
    directory.mkdir(exist_ok=True)
    events = Events(run_id, directory / 'events.jsonl', on_event)
    state_file = directory / 'job.json'
    with JobLock(directory, recover_lock):
        identity = {'config': config.model_dump(mode='json'), 'input_sha256': input_hashes(config),
                    'engine_version': __version__, 'runtime': runtime_versions()}
        fingerprint = hashlib.sha256(json_bytes(identity)).hexdigest()
        if existed:
            provenance = read_json(directory / 'provenance.json')
            if provenance['fingerprint'] != fingerprint:
                # Version equality is the hard gate; per-stage reuse below is
                # additionally guarded by phase-scoped code fingerprints.
                raise WorkflowError('RESUME_MISMATCH', '配置、输入或引擎版本已改变；请创建新任务。')
        else:
            write_json(directory / 'config.json', config.model_dump(mode='json'))
            write_json(directory / 'provenance.json', {**identity, 'fingerprint': fingerprint,
                                                       'engine_sha256': code_fingerprint(),
                                                       'runtime': runtime_versions(), 'at': utc_now()})
        state = read_json(state_file) if state_file.exists() else {'run_id': run_id, 'stages': {}, 'created_at': utc_now()}
        def save(status, **data):
            state.update(status=status, updated_at=utc_now(), **data)
            write_json(state_file, state)
        def cancelled():
            if (directory / 'cancel.request.json').exists():
                raise Cancelled()
        if (directory / 'result.json').exists():
            from .validation import verify_release
            verify_release(directory)
            save('completed')
            result = read_json(directory / 'result.json')
            events.emit('run_completed', **result, reused=True)
            return result
        if resume and (directory / 'cancel.request.json').exists():
            token = directory / 'cancel.request.json'
            os.replace(token, directory / ('cancel.acknowledged.' + uuid.uuid4().hex + '.json'))
        save('running', phase=None)
        events.emit('run_started', run_directory=str(directory), phases=PHASES, resumed=resume, offline=offline)
        runtime_code = code_fingerprint('runtime')
        try:
            if until != 'ingest':
                required_extras = ('local',) if config.embedding.backend == 'local' else ()
                issues = dependency_issues(required_extras)
                if issues:
                    raise WorkflowError('DEPENDENCIES', '运行依赖缺失或不兼容；已在任何 API 请求前停止。', details={'issues': issues})
            previous = fingerprint
            for number, phase in enumerate(PHASES):
                cancelled()
                extras = external_hashes(config, phase)
                phase_code = code_fingerprint(phase)
                stage_key = hashlib.sha256(json_bytes({'previous': previous, 'phase': phase,
                                                       'external': extras, 'code': phase_code})).hexdigest()
                target = directory / phase
                if target.exists():
                    if target.resolve() != target or target.is_symlink():
                        raise WorkflowError('UNSAFE_PATH', '阶段目录不能指向其他位置。')
                    stage = read_json(target / '_stage_manifest.json')
                    if stage['input_key'] != stage_key:
                        raise WorkflowError('STAGE_MISMATCH',
                            '阶段输入或该阶段引擎代码已改变，拒绝复用；删除该阶段目录后 resume 可重建，或新建任务。',
                            details={'phase': phase})
                    verify_entries(target, stage['artifacts'])
                    events.emit('stage_completed', phase=phase, reused=True, result=stage['result'])
                else:
                    staging_root = directory / '.staging'
                    staging_root.mkdir(exist_ok=True)
                    temporary = Path(tempfile.mkdtemp(prefix=phase + '-', dir=staging_root))
                    save('running', phase=phase)
                    events.emit('stage_started', phase=phase, overall_progress=number / len(PHASES))
                    def progress(phase=phase, **data):
                        events.emit('stage_progress', phase=phase, **data)
                    with contextlib.redirect_stdout(sys.stderr):
                        if phase == 'ingest':
                            from .ingest import prepare
                            result = prepare(config, temporary, cancelled, progress)
                        elif phase == 'embed':
                            from .embeddings import embed
                            operational = config.model_copy(deep=True)
                            if offline:
                                operational.embedding.offline = True
                            result = embed(operational, directory / 'ingest', temporary, cancelled, progress)
                        elif phase == 'model':
                            from .modeling import model_corpus
                            result = model_corpus(config, directory / 'ingest', directory / 'embed', temporary, cancelled, progress)
                        else:
                            from .plots import render
                            result = render(config, directory / 'ingest', directory / 'model', temporary, cancelled, progress)
                    cancelled()
                    if input_hashes(config) != identity['input_sha256'] or external_hashes(config, phase) != extras:
                        raise WorkflowError('INPUT_CHANGED', '阶段执行过程中输入文件发生改变。')
                    if code_fingerprint(phase) != phase_code:
                        raise WorkflowError('ENGINE_CHANGED', '该阶段引擎文件在执行中被修改；本次产物不发布，可直接重试。')
                    stage = {'phase': phase, 'input_key': stage_key, 'result': result,
                             'artifacts': file_entries(temporary), 'at': utc_now()}
                    write_json(temporary / '_stage_manifest.json', stage)
                    verify_entries(temporary, stage['artifacts'])
                    # The target has never been published. Rename the entire same-volume directory.
                    os.replace(temporary, target)
                    events.emit('stage_completed', phase=phase, result=result, reused=False)
                previous = sha256(target / '_stage_manifest.json')
                state['stages'][phase] = {'manifest_sha256': previous, 'status': 'completed'}
                save('running', phase=phase)
                if until == phase:
                    save('prepared', phase=phase)
                    events.emit('run_prepared', phase=phase, run_directory=str(directory))
                    return {'status': 'prepared', 'run_id': run_id, 'run_directory': str(directory), 'phase': phase}
            cancelled()
            from .validation import verify_content
            events.emit('validation_started')
            with contextlib.redirect_stdout(sys.stderr):
                validation = verify_content(directory, config)
            if input_hashes(config) != identity['input_sha256']:
                raise WorkflowError('INPUT_CHANGED', '验收前原始输入已改变。')
            if code_fingerprint('runtime') != runtime_code:
                raise WorkflowError('ENGINE_CHANGED', '验收前运行时文件已改变，请重试验收。')
            write_json(directory / 'validation.json', validation)
            files = []
            for phase in PHASES:
                files.extend({**row, 'path': phase + '/' + row['path']} for row in file_entries(directory / phase))
            for name in ['config.json', 'provenance.json', 'validation.json']:
                file = directory / name
                files.append({'path': name, 'sha256': sha256(file), 'bytes': file.stat().st_size})
            write_json(directory / 'release_manifest.json', {'schema_version': 1, 'engine_version': __version__,
                       'fingerprint': fingerprint, 'artifacts': files, 'validation': validation, 'at': utc_now()})
            verify_entries(directory, files)
            cancelled()
            result = {'status': 'completed', 'run_id': run_id, 'run_directory': str(directory),
                      'report': 'report/report.html', 'manifest': 'release_manifest.json',
                      'manifest_sha256': sha256(directory / 'release_manifest.json'), 'records': validation['records'],
                      'topics': validation['topics'], 'outliers': validation['outliers'], 'at': utc_now()}
            write_json(directory / 'result.json', result)
            save('completed', phase=None)
            events.emit('run_completed', **result)
            return result
        except (Exception, KeyboardInterrupt) as error:
            is_cancelled = isinstance(error, (Cancelled, KeyboardInterrupt))
            code = 'CANCELLED' if is_cancelled else error.code if isinstance(error, WorkflowError) else 'INTERNAL'
            save('cancelled' if is_cancelled else 'failed', error={'code': code, 'type': type(error).__name__})
            events.emit('run_cancelled' if is_cancelled else 'run_failed', code=code,
                        message=str(error) if isinstance(error, WorkflowError) else '阶段失败；已完成的阶段仍可验收与恢复。',
                        error_type=type(error).__name__)
            if isinstance(error, KeyboardInterrupt):
                raise Cancelled() from None
            raise


def cancel(run_dir):
    directory = Path(run_dir).resolve()
    state = read_json(directory / 'job.json')
    if state['status'] == 'completed':
        raise WorkflowError('JOB_COMPLETED', '已完成的任务不能取消。')
    write_json(directory / 'cancel.request.json', {'requested_at': utc_now()})
    return {'status': 'cancel_requested', 'run_id': state['run_id']}


def package_run(run_dir, destination):
    from .validation import verify_release
    directory, destination = Path(run_dir).resolve(), Path(destination).resolve()
    verify_release(directory)
    if destination.exists() or destination.suffix.lower() != '.zip' or destination.is_relative_to(directory):
        raise WorkflowError('PACKAGE_PATH', 'ZIP 需位于任务目录之外，且不能覆盖已有文件。')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
    manifest = read_json(directory / 'release_manifest.json')
    paths = [r['path'] for r in manifest['artifacts']] + ['release_manifest.json', 'result.json']
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in paths:
            archive.write(directory / relative, relative)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None or len(archive.infolist()) != len(paths):
            raise WorkflowError('PACKAGE_CRC', '压缩包校验失败。')
        for row in manifest['artifacts']:
            digest = hashlib.sha256()
            with archive.open(row['path']) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != row['sha256']:
                raise WorkflowError('PACKAGE_INTEGRITY', '打包期间产物发生变化。')
    os.replace(temporary, destination)
    digest = sha256(destination)
    destination.with_suffix('.zip.sha256').write_text(digest + '  ' + destination.name + '\n', encoding='utf-8')
    return {'path': str(destination), 'files': len(paths), 'sha256': digest}
