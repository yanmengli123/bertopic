import json
import os
import socket
import zipfile

import numpy as np
import pytest

from bertopic_workflow.common import WorkflowError, read_json, write_json, write_table
from bertopic_workflow.config import EmbeddingOptions, ImportOptions, InputFile, load_config
from bertopic_workflow.embeddings import Cache, normalize, request_vectors, retry_seconds, split_text
from bertopic_workflow.ingest import prepare, read_input
from bertopic_workflow.metadata import address_groups
from bertopic_workflow.runtime import JobLock, run


def config(tmp_path, text='title,abstract,year\nAlpha,alpha beta,2020\nBeta,beta gamma,2021\nGamma,gamma delta,2022\nDelta,delta epsilon,2023\nEpsilon,epsilon omega,2024\n', **extra):
    source = tmp_path / 'data.csv'
    source.write_text(text, encoding='utf-8')
    target = tmp_path / 'project.json'
    data = {'inputs': [{'path': str(source)}], 'output_root': str(tmp_path / 'runs'),
            'cache_root': str(tmp_path / 'cache'), **extra}
    write_json(target, data)
    return target


def test_content_sniff_html_xls_and_row_provenance(tmp_path):
    file = tmp_path / 'CNKI.xls'
    file.write_text('<html><table><tr><td>Title-题名</td><td>Summary-摘要</td></tr>'
                    '<tr><td>细胞</td><td>摘要内容</td></tr></table></html>', encoding='utf-8')
    rows, metadata = read_input(InputFile(path=str(file)), ImportOptions())
    assert metadata['source'] == 'CNKI' and metadata['format'] == 'html_table'
    assert rows[0]['title'] == '细胞' and rows[0]['_row'] == 2


@pytest.mark.parametrize('source,header', [('VIP', '题名,摘要,刊名'), ('WanFang', '题名,摘要,刊名'),
                                         ('WoS', 'Article Title,Abstract,Source Title'), ('PubMed', 'Title,Abstract,Journal/Book')])
def test_vendor_headers(source, header, tmp_path):
    file = tmp_path / 'export.csv'
    file.write_text(header + '\nA paper,An abstract,A journal\n', encoding='utf-8')
    rows, metadata = read_input(InputFile(path=str(file), source=source), ImportOptions())
    assert rows[0]['title'] == 'A paper' and metadata['source'] == source


def test_mapping_ambiguity_and_explicit_override(tmp_path):
    file = tmp_path / 'data.csv'
    file.write_text('Title,Article Title,Abstract\nA,B,Text\n', encoding='utf-8')
    with pytest.raises(WorkflowError, match='多个列'):
        read_input(InputFile(path=str(file)), ImportOptions())
    rows, _ = read_input(InputFile(path=str(file), mapping={'title': 'Article Title'}), ImportOptions())
    assert rows[0]['title'] == 'B'


def test_medline_continuations_and_ris(tmp_path):
    file = tmp_path / 'pubmed.nbib'
    file.write_text('PMID- 00123\nTI  - A title\nAB  - first part\n      second part\nDP  - 2020 Jan\nAID - 10.1234/example [doi]\n', encoding='utf-8')
    rows, meta = read_input(InputFile(path=str(file)), ImportOptions())
    assert rows[0]['pmid'] == '00123' and rows[0]['abstract'] == 'first part second part'
    assert rows[0]['doi'] == '10.1234/example' and meta['source'] == 'PubMed'
    file = tmp_path / 'data.ris'
    file.write_text('TY  - JOUR\nTI  - A title\nAB  - abstract\nER  -\n', encoding='utf-8')
    assert len(read_input(InputFile(path=str(file)), ImportOptions())[0]) == 1


def test_safe_xml_rejects_entities(tmp_path):
    from defusedxml.common import EntitiesForbidden
    file = tmp_path / 'bad.xml'
    file.write_text('<!DOCTYPE a [<!ENTITY x "evil">]><PubmedArticleSet><PubmedArticle>&x;</PubmedArticle></PubmedArticleSet>')
    with pytest.raises(EntitiesForbidden):
        read_input(InputFile(path=str(file)), ImportOptions())


def test_quarantine_dedup_and_counting(tmp_path):
    path = config(tmp_path, text='title,abstract,doi\nA,alpha beta,10.1234/a\nA,alpha beta,10.1234/a\n,,10.1234/empty\nB,beta gamma,\nC,gamma delta,\nD,delta epsilon,\nE,epsilon omega,\n')
    cfg = load_config(path)
    cfg.import_options.deduplication = 'doi'
    target = tmp_path / 'stage'; target.mkdir()
    manifest = prepare(cfg, target)
    assert manifest['read_records'] == 7 and manifest['retained_records'] == 5
    assert manifest['quarantined_records'] == manifest['deduplicated_records'] == 1
    assert len(read_json(target / 'duplicate_lineage.json')) == 1


def test_csv_formula_escaping_does_not_modify_json(tmp_path):
    write_table(tmp_path, 'safe', [{'title': '=HYPERLINK("evil")', 'year': -1}])
    assert read_json(tmp_path / 'safe.json')[0]['title'].startswith('=')
    assert "'=HYPERLINK" in (tmp_path / 'safe.csv').read_text(encoding='utf-8-sig')


def test_invalid_config_does_not_echo_secret(tmp_path):
    path = config(tmp_path, embedding={'api_key': 'sensitive-do-not-echo'})
    with pytest.raises(WorkflowError) as error:
        load_config(path)
    assert 'sensitive-do-not-echo' not in str(error.value) + json.dumps(error.value.details)


def test_https_and_precomputed_contract():
    with pytest.raises(ValueError):
        EmbeddingOptions(base_url='http://remote.example/v1')
    with pytest.raises(ValueError):
        EmbeddingOptions(backend='precomputed')
    with pytest.raises(ValueError):
        EmbeddingOptions(base_url='https://user:secret@example.org/v1')


def test_normalization_rejects_nan_and_zero():
    for vector in [[[0., 0.]], [[np.nan, 1.]]]:
        with pytest.raises(WorkflowError): normalize(vector)
    assert np.allclose(np.linalg.norm(normalize([[3., 4.]]), axis=1), 1)


def test_cache_roundtrip(tmp_path):
    cache = Cache(tmp_path / 'cache.sqlite')
    try:
        assert cache.get('missing') is None
        cache.put('id', [1., 2., 3.])
        assert np.array_equal(cache.get('id'), [1., 2., 3.])
    finally: cache.close()


def test_long_text_roundtrip_with_real_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0, 'alpha': 1, 'beta': 2}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    text = 'alpha beta ' * 40
    parts = split_text(text, tokenizer, 8)
    assert ''.join(p for p, _ in parts) == text
    assert all(len(tokenizer.encode(p, add_special_tokens=False).ids) <= 8 for p, _ in parts)


def test_chinese_token_offsets_preserve_every_character():
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.normalizers import BertNormalizer
    from tokenizers.pre_tokenizers import BertPreTokenizer
    tokenizer = Tokenizer(WordPiece({'[UNK]': 0, '细': 1, '胞': 2, '膜': 3, '蛋': 4, '白': 5}, unk_token='[UNK]'))
    tokenizer.normalizer = BertNormalizer(handle_chinese_chars=True)
    tokenizer.pre_tokenizer = BertPreTokenizer()
    text = '细胞膜蛋白 ' * 40
    pieces = split_text(text, tokenizer, 8)
    assert ''.join(p for p, _ in pieces) == text and len(pieces) > 1
    assert all(n <= 8 for _, n in pieces)


def test_dependency_gate_stops_before_embeddings(monkeypatch, tmp_path):
    path = config(tmp_path)
    monkeypatch.setattr('bertopic_workflow.runtime.dependency_issues', lambda *args: [{'name': 'missing'}])
    with pytest.raises(WorkflowError, match='任何 API 请求前停止'):
        run(path, run_id='no-deps', on_event=lambda event: None)
    assert not (tmp_path / 'runs/no-deps/embed').exists()


class Response:
    status_code = 200
    headers = {}
    def __init__(self, data): self.data = data
    def json(self): return self.data


def test_api_index_validation_and_no_redirect(monkeypatch):
    monkeypatch.setenv('SILICONFLOW_API_KEY', 'test-credential')
    class Session:
        def post(self, url, **kwargs):
            assert kwargs['allow_redirects'] is False
            return Response({'model': 'BAAI/bge-m3', 'data': [{'index': 0, 'embedding': [1, 2]}, {'index': 0, 'embedding': [3, 4]}]})
    with pytest.raises(WorkflowError, match='响应'):
        request_vectors(EmbeddingOptions(), ['a', 'b'], lambda: None, Session())
    assert retry_seconds('not-a-date', 2) == 4


def test_address_author_semicolons_and_usa_suffix():
    countries, institutions = address_groups('[A; B] Example Univ, Boston, MA 02115 USA; Other Univ, Tokyo, Japan')
    assert countries == ['Japan', 'United States'] and institutions == ['Example Univ', 'Other Univ']
    assert address_groups('Example Univ, Unknown Territory')[0] == []


def test_lock_conflict_and_live_recovery_refused(tmp_path):
    with JobLock(tmp_path):
        with pytest.raises(WorkflowError):
            with JobLock(tmp_path): pass
        with pytest.raises(WorkflowError):
            with JobLock(tmp_path, recover=True): pass
    assert not (tmp_path / '.run.lock').exists()


def test_stale_lock_recovery(tmp_path):
    write_json(tmp_path / '.run.lock', {'pid': 2147483000, 'process_started': 0, 'host': socket.gethostname()})
    with JobLock(tmp_path, recover=True):
        assert read_json(tmp_path / '.run.lock')['pid'] == os.getpid()


def test_resume_rejects_modified_source_and_path_traversal(tmp_path):
    path = config(tmp_path)
    result = run(path, run_id='case', until='ingest', on_event=lambda event: None)
    assert result['status'] == 'prepared'
    (tmp_path / 'data.csv').write_text('title,abstract\nchanged,text\n')
    with pytest.raises(WorkflowError, match='已改变'):
        run(path, run_id='case', until='ingest', resume=True, on_event=lambda event: None)
    with pytest.raises(WorkflowError):
        run(path, run_id='../escape', on_event=lambda event: None)


def test_cancel_and_resume_prepared_stage(tmp_path):
    path = config(tmp_path)
    from bertopic_workflow.runtime import cancel
    from bertopic_workflow.common import Cancelled
    def callback(event):
        if event['event'] == 'run_started':
            cancel(event['data']['run_directory'])
    with pytest.raises(Cancelled):
        run(path, run_id='cancelled', on_event=callback)
    result = run(path, run_id='cancelled', resume=True, until='ingest', on_event=lambda event: None)
    assert result['status'] == 'prepared'


def test_xlsx_zip_size_guard(tmp_path):
    file = tmp_path / 'bad.xlsx'
    with zipfile.ZipFile(file, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('large.xml', 'x' * (1024**2 + 1))
    with pytest.raises(WorkflowError, match='解压大小'):
        read_input(InputFile(path=str(file)), ImportOptions(max_uncompressed_mb=1))


def test_version_single_source_and_doctor_consistency():
    import importlib.metadata
    from bertopic_workflow import __version__
    try:
        installed = importlib.metadata.version('dsh-bertopic-workflow')
    except importlib.metadata.PackageNotFoundError:
        installed = None
    assert installed is None or installed == __version__


def test_phase_scoped_code_fingerprint():
    from bertopic_workflow.common import PHASE_MODULES, WorkflowError, code_fingerprint
    full, embed, report = code_fingerprint(), code_fingerprint('embed'), code_fingerprint('report')
    assert full != embed != report
    assert code_fingerprint('embed') == embed and code_fingerprint('report') == report
    assert set(PHASE_MODULES) == {'ingest', 'embed', 'model', 'report', 'runtime'}
    with pytest.raises(WorkflowError):
        code_fingerprint('nonexistent')


def test_dependency_check_layers_local_extra():
    from pathlib import Path
    from bertopic_workflow.common import dependency_issues, read_json
    schema = read_json(Path(__import__('bertopic_workflow.common', fromlist=['__file__']).__file__).parent
                       / 'schemas/runtime_requirements.json')
    local_specs = schema.get('extras', {}).get('local', [])
    assert local_specs, 'local extra 应至少包含 sentence-transformers/transformers/torch'
    base_names = {i['name'] for i in dependency_issues()}
    assert 'torch' not in base_names, 'torch 不应出现在基础依赖检查中'
    assert all('fix' in i for i in dependency_issues(('local',)))


def test_cli_survives_non_utf8_console():
    import json as jsonlib
    import os
    import subprocess
    import sys
    from pathlib import Path
    source_root = str(Path(__file__).resolve().parents[1] / 'src')
    result = subprocess.run([sys.executable, '-m', 'bertopic_workflow', 'capabilities'],
                            capture_output=True, env={**os.environ, 'PYTHONIOENCODING': 'gbk',
                                                      'PYTHONPATH': source_root})
    lines = [l for l in result.stdout.decode('utf-8').splitlines() if l.strip()]
    assert result.returncode == 0 and lines
    event = jsonlib.loads(lines[0])
    assert event['event'] == 'capabilities' and event['data']['figures'] == 15
