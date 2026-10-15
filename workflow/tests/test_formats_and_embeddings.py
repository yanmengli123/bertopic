from xml.sax.saxutils import escape
import zipfile

import numpy as np
import pytest

from bertopic_workflow.common import WorkflowError
from bertopic_workflow.config import ImportOptions, InputFile, load_config
from bertopic_workflow.embeddings import Cache, embed
from bertopic_workflow.ingest import prepare, read_input


def xlsx_fixture(path, tables):
    # Minimal OOXML fixture; no Excel authoring dependency is needed by the test.
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    relns = 'http://schemas.openxmlformats.org/package/2006/relationships'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('[Content_Types].xml', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                   + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, len(tables) + 1)) + '</Types>')
        z.writestr('_rels/.rels', f'<Relationships xmlns="{relns}"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
                   + ''.join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i, (name, rows) in enumerate(tables.items(), 1)) + '</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', f'<Relationships xmlns="{relns}">'
                   + ''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(tables) + 1)) + '</Relationships>')
        for i, rows in enumerate(tables.values(), 1):
            content = ''.join('<row r="' + str(j) + '">' + ''.join(f'<c r="{chr(65 + k)}{j}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>' for k, value in enumerate(row)) + '</row>' for j, row in enumerate(rows, 1))
            z.writestr(f'xl/worksheets/sheet{i}.xml', f'<worksheet xmlns="{ns}"><sheetData>{content}</sheetData></worksheet>')


def test_vip_xlsx_and_multi_sheet_selection(tmp_path):
    file = tmp_path / '维普.xlsx'
    xlsx_fixture(file, {'Data': [['导出说明'], ['题名', '摘要', '刊名'], ['细胞研究', '细胞膜蛋白', '示例期刊']],
                        'Other': [['题名', '摘要'], ['other', 'text']]})
    with pytest.raises(WorkflowError, match='多个表格'):
        read_input(InputFile(path=str(file), source='VIP'), ImportOptions())
    rows, meta = read_input(InputFile(path=str(file), source='VIP', sheet='Data'), ImportOptions())
    assert meta['header_row'] == 2 and rows[0]['_row'] == 3 and rows[0]['title'] == '细胞研究'


def test_gb18030_tsv_and_quoted_newline(tmp_path):
    file = tmp_path / 'data.tsv'
    file.write_bytes('题名\t摘要\n文章\t"第一行\n第二行"\n'.encode('gb18030'))
    rows, meta = read_input(InputFile(path=str(file), delimiter='\t'), ImportOptions())
    assert len(rows) == 1 and rows[0]['abstract'] == '第一行 第二行' and meta['encoding'] == 'gb18030'


def test_publication_type_is_not_wos_document_type(tmp_path):
    file = tmp_path / 'wos.csv'
    file.write_text('Article Title,Abstract,Publication Type,Document Type,WoS Categories,Research Areas\nTitle,Text,J,Review,Cell Biology,Biology\n')
    rows, _ = read_input(InputFile(path=str(file)), ImportOptions())
    assert rows[0]['document_type'] == 'Review' and rows[0]['publication_type'] == 'J'
    assert rows[0]['category'] == 'Cell Biology' and rows[0]['research_areas'] == 'Biology'


def test_remote_batches_and_offline_cache(monkeypatch, tmp_path):
    from test_core import config
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0, 'alpha': 1}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / 'tokenizer.json'))
    path = config(tmp_path, embedding={'backend': 'openai_compatible', 'model': 'test-model', 'dimension': 4,
                  'max_tokens': 32, 'tokenizer_path': str(tmp_path / 'tokenizer.json'), 'batch_size': 16})
    cfg = load_config(path)
    ingest = tmp_path / 'ingest'; ingest.mkdir(); prepare(cfg, ingest)
    destination = tmp_path / 'embed'; destination.mkdir()
    calls = []
    def fake_request(options, texts, cancelled):
        calls.append(texts)
        return np.array([[1., 2., 3., len(t)] for t in texts])
    monkeypatch.setattr('bertopic_workflow.embeddings.request_vectors', fake_request)
    embed(cfg, ingest, destination)
    assert len(calls) == 1 and len(calls[0]) == 5
    cfg.embedding.offline = True
    again = tmp_path / 'again'; again.mkdir()
    result = embed(cfg, ingest, again)
    assert result['cache_hits'] == 5 and len(calls) == 1
    assert np.array_equal(np.load(destination / 'embeddings.npy'), np.load(again / 'embeddings.npy'))
    cfg.embedding.model = 'changed-model'
    missing = tmp_path / 'missing'; missing.mkdir()
    with pytest.raises(WorkflowError, match='离线缓存缺少'):
        embed(cfg, ingest, missing)


def test_cache_bit_corruption_is_detected(tmp_path):
    cache = Cache(tmp_path / 'cache.sqlite')
    try:
        cache.put('id', [1., 2.])
        cache.connection.execute('UPDATE vectors SET data=? WHERE key=?', (np.array([3., 4.], dtype='<f4').tobytes(), 'id'))
        cache.connection.commit()
        with pytest.raises(WorkflowError, match='缓存向量损坏'):
            cache.get('id')
    finally:
        cache.close()


def test_address_state_abbreviation_is_not_a_country():
    from bertopic_workflow.metadata import address_groups
    assert address_groups('Example University, Los Angeles, CA')[0] == []


def test_nested_csv_values_are_canonical(tmp_path):
    import csv
    import json
    from bertopic_workflow.common import write_table, read_json, csv_cell
    write_table(tmp_path, 'nested', [{'raw': {'z': 'last', 'a': 'first'}}])
    with (tmp_path / 'nested.csv').open(encoding='utf-8-sig', newline='') as stream:
        actual = list(csv.DictReader(stream))[0]['raw']
    expected = csv_cell(read_json(tmp_path / 'nested.json')[0]['raw'])
    assert actual == expected and json.loads(actual)['a'] == 'first'


def test_local_offline_backend_can_compute_from_cached_assets(monkeypatch, tmp_path):
    import sys
    import types
    from test_core import config
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    model_dir = tmp_path / 'local-model'; model_dir.mkdir()
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(model_dir / 'tokenizer.json'))
    class LocalModel:
        max_seq_length = 512
        def __init__(self, name, **kwargs):
            assert kwargs['local_files_only'] is True and kwargs['trust_remote_code'] is False
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)
    monkeypatch.setitem(sys.modules, 'sentence_transformers', types.SimpleNamespace(SentenceTransformer=LocalModel))
    path = config(tmp_path, embedding={'backend': 'local', 'model': str(model_dir), 'dimension': 4, 'offline': True})
    cfg = load_config(path)
    ingest = tmp_path / 'ingest'; ingest.mkdir(); prepare(cfg, ingest)
    output = tmp_path / 'embedded'; output.mkdir()
    assert embed(cfg, ingest, output)['requests'] == 0


def test_credential_environment_cannot_be_an_os_variable():
    from bertopic_workflow.config import EmbeddingOptions
    with pytest.raises(ValueError):
        EmbeddingOptions(api_key_env='PATH')
