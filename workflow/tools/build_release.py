"""Build wheel, source bundle, dependency inventory and checksums; never package data."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tomllib
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def main():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    version = project['version']
    dist = ROOT / 'dist'; dist.mkdir(exist_ok=True)
    subprocess.run([sys.executable, '-B', '-m', 'build', '--no-isolation', '--outdir', str(dist)], cwd=ROOT, check=True)
    pending, inventory = [Requirement(r) for r in project['dependencies']], {}
    environment = {**default_environment(), 'extra': ''}
    while pending:
        requirement = pending.pop()
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        name = canonicalize_name(requirement.name)
        if name in inventory:
            continue
        distribution = importlib.metadata.distribution(requirement.name)
        inventory[name] = distribution.version
        pending.extend(Requirement(r) for r in distribution.requires or [])
    (dist / 'requirements.lock.txt').write_text(''.join(f'{name}=={value}\n' for name, value in sorted(inventory.items())), encoding='utf-8')
    sbom = {'bomFormat': 'CycloneDX', 'specVersion': '1.5', 'version': 1,
            'metadata': {'component': {'type': 'application', 'name': project['name'], 'version': version}},
            'components': [{'type': 'library', 'name': name, 'version': value, 'purl': f'pkg:pypi/{name}@{value}'} for name, value in sorted(inventory.items())]}
    (dist / 'sbom.cdx.json').write_text(json.dumps(sbom, indent=2) + '\n', encoding='utf-8')
    bundle = dist / f'DSH_BERTopic_Workflow_{version}.zip'
    sources = [p for p in ROOT.rglob('*') if p.is_file() and p.suffix in {'.py', '.mjs', '.json', '.md', '.toml', '.ps1', '.in', '.yml'}
               and not any(part in {'dist', 'build', '.validation', '.cache', 'runs', '__pycache__', '.pytest_cache', '.ruff_cache'} or part.endswith('.egg-info') for part in p.relative_to(ROOT).parts)]
    support = [dist / 'requirements.lock.txt', dist / 'sbom.cdx.json',
               dist / f'dsh_bertopic_workflow-{version}-py3-none-any.whl', dist / f'dsh_bertopic_workflow-{version}.tar.gz']
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(sources):
            archive.write(path, 'workflow/' + path.relative_to(ROOT).as_posix())
        for path in support:
            archive.write(path, 'install/' + path.name)
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        assert all(not any(part in {'runs', '.cache', '.validation', '__pycache__'} for part in Path(p).parts) for p in archive.namelist())
    artifacts = [*support, bundle]
    (dist / 'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n' for p in artifacts), encoding='utf-8')
    manifest = {'version': version, 'python': sys.version.split()[0], 'files': [{'path': p.name, 'bytes': p.stat().st_size,
                  'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts],
                  'dependency_components': len(inventory), 'private_data_included': False, 'crc_validation': 'passed'}
    (dist / 'build_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
