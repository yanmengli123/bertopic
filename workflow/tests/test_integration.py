"""Real BERTopic integration with controlled vectors, not a research validation."""
import shutil
import os

import numpy as np
import pytest

from bertopic_workflow.common import WorkflowError, read_json, sha256, write_json
from bertopic_workflow.runtime import package_run, run
from bertopic_workflow.validation import verify_release


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp('workflow-integration')
    source = root / 'synthetic.csv'
    subjects = [('cell membrane protein signaling', 'Cell biology'),
                ('quantum photon optical laser', 'Optics'),
                ('气候变化 降水 温度 碳排放 环境', '环境科学')]
    text = ['title,abstract,year,journal,country,institution']
    for words, journal in subjects:
        for i in range(24):
            text.append(f'{words} {i},{words} measurement mechanism sample {i},{2010 + i % 15},{journal},China,Example University')
    source.write_text('\n'.join(text), encoding='utf-8')
    cfg = root / 'project.json'
    write_json(cfg, {'inputs': [{'path': str(source)}], 'output_root': str(root / 'runs'),
                    'cache_root': str(root / 'cache'),
                    'embedding': {'backend': 'precomputed', 'model': 'controlled-test-vectors', 'dimension': 12,
                                  'vectors_path': str(root / 'vectors.npy'), 'metadata_path': str(root / 'vectors.json')},
                    'model': {'min_cluster_sizes': [5, 9], 'min_samples': 2, 'n_neighbors': 10,
                              'stability_seeds': [7], 'silhouette_sample': 60},
                    'report': {'year_end': 2026, 'incomplete_year': 2026, 'dpi': 100, 'topic_details': 1,
                               'font_path': os.environ.get('WORKFLOW_TEST_FONT')}})
    events = []
    run(cfg, run_id='integration', until='ingest', on_event=events.append)
    directory = root / 'runs/integration'
    corpus = read_json(directory / 'ingest/corpus.json')
    rng = np.random.default_rng(42)
    vectors = np.zeros((72, 12), dtype=np.float32)
    for cluster in range(3):
        vectors[cluster * 24:(cluster + 1) * 24, cluster] = 1
    vectors += rng.normal(0, .025, vectors.shape).astype(np.float32)
    np.save(root / 'vectors.npy', vectors, allow_pickle=False)
    write_json(root / 'vectors.json', {'record_ids': [r['record_id'] for r in corpus],
               'text_sha256': [r['text_sha256'] for r in corpus], 'model': 'controlled-test-vectors'})
    result = run(cfg, run_id='integration', resume=True, on_event=events.append)
    return root, cfg, directory, result, events


def test_complete_workflow_and_protocol(completed):
    root, cfg, directory, result, events = completed
    assert result['status'] == 'completed' and result['records'] == 72 and result['topics'] >= 2
    verified = verify_release(directory)
    assert verified['status'] == 'passed'
    catalog = read_json(directory / 'report/figures_catalog.json')
    assert catalog['generated'] == 15
    assert events[-1]['event'] == 'run_completed'
    from jsonschema import validate
    from importlib.resources import files
    schema = read_json(files('bertopic_workflow') / 'schemas/event.schema.json')
    for event in events:
        validate(event, schema)


def test_completed_resume_preserves_report_and_rejects_tampering(completed, tmp_path):
    root, cfg, directory, result, _ = completed
    before = sha256(directory / 'report/report.html')
    assert run(cfg, run_id='integration', resume=True, on_event=lambda event: None)['status'] == 'completed'
    assert sha256(directory / 'report/report.html') == before
    copied = tmp_path / 'copied'; shutil.copytree(directory, copied)
    (copied / 'report/report.html').write_text('tampered', encoding='utf-8')
    with pytest.raises(WorkflowError, match='修改'):
        verify_release(copied)


def test_final_zip_is_verified_and_not_overwritten(completed):
    root, _, directory, _, _ = completed
    result = package_run(directory, root / 'delivery.zip')
    assert result['files'] > 50 and result['sha256']
    with pytest.raises(WorkflowError, match='覆盖'):
        package_run(directory, root / 'delivery.zip')


def test_partial_stage_integrity_and_vector_order(tmp_path):
    from test_core import config
    cfg = config(tmp_path, embedding={'backend': 'precomputed', 'model': 'test', 'dimension': 4,
                 'vectors_path': str(tmp_path / 'vectors.npy'), 'metadata_path': str(tmp_path / 'vectors.json')})
    run(cfg, run_id='case', until='ingest', on_event=lambda event: None)
    directory = tmp_path / 'runs/case'
    corpus = read_json(directory / 'ingest/corpus.json')
    np.save(tmp_path / 'vectors.npy', np.ones((5, 4)))
    write_json(tmp_path / 'vectors.json', {'record_ids': [r['record_id'] for r in corpus][::-1],
               'text_sha256': [r['text_sha256'] for r in corpus], 'model': 'test'})
    with pytest.raises(WorkflowError, match='逐条匹配'):
        run(cfg, run_id='case', resume=True, on_event=lambda event: None)
    assert not (directory / 'result.json').exists() and not (directory / 'embed').exists()
    (directory / 'ingest/corpus.json').write_text('[]')
    with pytest.raises(WorkflowError, match='修改'):
        run(cfg, run_id='case', resume=True, on_event=lambda event: None)


def test_all_noise_without_years_is_a_valid_diagnostic_result(tmp_path):
    source = tmp_path / 'noise.csv'
    source.write_text('title,abstract\n' + '\n'.join(f'Document alpha {i},beta gamma mechanism {i}' for i in range(12)))
    cfg = tmp_path / 'noise.json'
    write_json(cfg, {'inputs': [{'path': str(source)}], 'output_root': str(tmp_path / 'runs'),
                    'cache_root': str(tmp_path / 'cache'),
                    'embedding': {'backend': 'precomputed', 'model': 'test', 'dimension': 8,
                       'vectors_path': str(tmp_path / 'vectors.npy'), 'metadata_path': str(tmp_path / 'vectors.json')},
                    'model': {'min_cluster_sizes': [12], 'min_samples': 2, 'n_neighbors': 5,
                              'n_components': 3, 'stability_seeds': []},
                    'report': {'year_end': 2026, 'incomplete_year': 2026, 'dpi': 100, 'topic_details': 0,
                               'font_path': os.environ.get('WORKFLOW_TEST_FONT')}})
    run(cfg, run_id='noise', until='ingest', on_event=lambda event: None)
    directory = tmp_path / 'runs/noise'
    corpus = read_json(directory / 'ingest/corpus.json')
    np.save(tmp_path / 'vectors.npy', np.random.default_rng(2).normal(size=(12, 8)).astype(np.float32))
    write_json(tmp_path / 'vectors.json', {'record_ids': [r['record_id'] for r in corpus],
               'text_sha256': [r['text_sha256'] for r in corpus], 'model': 'test'})
    result = run(cfg, run_id='noise', resume=True, on_event=lambda event: None)
    assert result['topics'] == 0 and result['outliers'] == 12
    assert verify_release(directory)['status'] == 'passed'
    catalog = read_json(directory / 'report/figures_catalog.json')
    assert next(r for r in catalog['charts'] if r['number'] == 8)['status'] == 'not_applicable'
    assert read_json(directory / 'report/evidence/fig15.json')['dimension'] == 'database'
