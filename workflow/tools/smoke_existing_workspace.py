"""Optional acceptance against existing files and saved BGE vectors; no API use."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from bertopic_workflow.common import read_json, sha256, write_json, write_table  # noqa: E402
from bertopic_workflow.config import load_config  # noqa: E402
from bertopic_workflow.runtime import run  # noqa: E402
from bertopic_workflow.validation import verify_release  # noqa: E402


def main():
    parser = ArgumentParser()
    parser.add_argument('--legacy-root', type=Path, required=True)
    args = parser.parse_args()
    legacy = args.legacy_root.resolve()
    output = ROOT / '.validation' / ('existing-' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    original = {name: sha256(legacy / name) for name in ['CNKI.xls', 'WanFangdata.xlsx', 'pubmed_merged_bertopic.csv', 'wos.csv']}
    project = load_config(ROOT / 'examples/current_workspace.json').model_dump(mode='json')
    project['inputs'] = [{'path': str(legacy / name), 'source': source} for name, source in
                         [('CNKI.xls', 'CNKI'), ('WanFangdata.xlsx', 'WanFang'), ('pubmed_merged_bertopic.csv', 'PubMed'), ('wos.csv', 'WoS')]]
    project['output_root'], project['cache_root'] = str(output / 'runs'), str(output / 'cache')
    project['embedding'] = {'backend': 'precomputed', 'model': 'BAAI/bge-m3', 'dimension': 1024,
                            'vectors_path': str(legacy / 'bertopic_analysis/outputs/joint_bge_m3_embeddings.npy'),
                            'metadata_path': str(output / 'full_vectors.json')}
    write_json(output / 'full_import.json', project)
    run(output / 'full_import.json', run_id='full-import', until='ingest')
    full = read_json(output / 'runs/full-import/ingest/corpus.json')
    previous = read_json(legacy / 'bertopic_analysis/outputs/joint_bge_m3/document_topics.json')
    vectors = np.load(legacy / 'bertopic_analysis/outputs/joint_bge_m3_embeddings.npy', mmap_mode='r', allow_pickle=False)
    bank = {r['text_sha256']: i for i, r in enumerate(previous)}
    full_cache_aligned = [r['text_sha256'] for r in full] == [r['text_sha256'] for r in previous]
    if full_cache_aligned:
        write_json(output / 'full_vectors.json', {'record_ids': [r['record_id'] for r in full],
                   'text_sha256': [r['text_sha256'] for r in full], 'model': 'BAAI/bge-m3'})
        run(output / 'full_import.json', run_id='full-import', resume=True, until='embed')
    assignments = {r['text_sha256']: r['topic'] for r in previous}
    picked, seen = [], set()
    for topic in [0, 1, 2]:
        candidates = [r for r in full if r['text_sha256'] in bank and assignments[r['text_sha256']] == topic]
        for row in candidates[:24]:
            if row['record_id'] not in seen:
                picked.append(row); seen.add(row['record_id'])
    for source in ['CNKI', 'WanFang', 'PubMed', 'WoS']:
        for row in [r for r in full if r['source'] == source and r['text_sha256'] in bank and r['record_id'] not in seen][:5]:
            picked.append(row); seen.add(row['record_id'])
    inputs = []
    for source in sorted({r['source'] for r in picked}):
        rows = [{field: r[field] for field in ['title', 'abstract', 'year', 'doi', 'journal', 'institutions', 'countries', 'category']}
                for r in picked if r['source'] == source]
        write_table(output, 'sample_' + source, rows)
        inputs.append({'path': str(output / ('sample_' + source + '.csv')), 'source': source})
    project.update(inputs=inputs, embedding={'backend': 'precomputed', 'model': 'BAAI/bge-m3', 'dimension': 1024,
                   'vectors_path': str(output / 'vectors.npy'), 'metadata_path': str(output / 'vectors.json')},
                   model={'min_cluster_sizes': [5, 9], 'min_samples': 2, 'n_neighbors': 12,
                          'stability_seeds': [7, 11], 'silhouette_sample': 100},
                   report={'year_end': 2026, 'incomplete_year': 2026, 'dpi': 160, 'topic_details': 3})
    write_json(output / 'sample.json', project)
    run(output / 'sample.json', run_id='real-smoke', until='ingest')
    directory = output / 'runs/real-smoke'
    corpus = read_json(directory / 'ingest/corpus.json')
    # Safe CSV presentation escaping is reversible here for these source fields.
    assert all(r['text_sha256'] in bank for r in corpus), 'A text changed; do not borrow its previous vector'
    np.save(output / 'vectors.npy', np.stack([vectors[bank[r['text_sha256']]] for r in corpus]), allow_pickle=False)
    write_json(output / 'vectors.json', {'record_ids': [r['record_id'] for r in corpus],
               'text_sha256': [r['text_sha256'] for r in corpus], 'model': 'BAAI/bge-m3'})
    result = run(output / 'sample.json', run_id='real-smoke', resume=True)
    verification = verify_release(directory)
    assert all(sha256(legacy / name) == digest for name, digest in original.items())
    acceptance = {'full_import_records': len(full), 'source_counts': dict(Counter(r['source'] for r in full)),
                  'cached_text_matches': sum(r['text_sha256'] in bank for r in full),
                  'real_bge_sample_records': len(corpus), 'result': result, 'verification': verification,
                  'full_precomputed_vectors_aligned': full_cache_aligned,
                  'original_inputs_unchanged': True, 'api_requests': 0}
    write_json(output / 'acceptance.json', acceptance)
    write_json(ROOT / '.validation/latest_existing_acceptance.json', acceptance)
    print('ACCEPTANCE ' + str(output / 'acceptance.json'))


if __name__ == '__main__':
    main()
