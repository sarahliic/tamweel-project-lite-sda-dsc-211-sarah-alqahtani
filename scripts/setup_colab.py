"""Prepare the CPU workspace and verify every released input before use."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

OWNER_REPO = 'almiyead-rgb/sda-dsc-211-student-template'
MODULES = {'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy',
           'scikit-learn': 'sklearn', 'matplotlib': 'matplotlib', 'xgboost': 'xgboost',
           'lightgbm': 'lightgbm', 'shap': 'shap', 'numba': 'numba', 'optuna': 'optuna'}


def fetch_verified(url: str, destination: Path, expected_sha: str,
                   *, attempts: int = 3, opener=None) -> str:
    """Keep a valid local copy; retry transient network failures, never trust corrupt bytes."""
    opener = opener or urllib.request.urlopen
    destination = Path(destination)
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected_sha:
        return 'verified local copy'
    if not url.startswith('https://'):
        raise ValueError('Downloads require HTTPS.')
    last_error = None
    for attempt in range(attempts):
        try:
            with opener(url, timeout=45) as response:
                content = response.read(32 * 1024 * 1024 + 1)
            if len(content) > 32 * 1024 * 1024:
                raise ValueError('Download is larger than the allowed teaching file size.')
            if hashlib.sha256(content).hexdigest() != expected_sha:
                raise ValueError(f'Checksum mismatch for {destination.name}. Do not use this file.')
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + '.download')
            temporary.write_bytes(content)
            temporary.replace(destination)
            return 'downloaded and verified'
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(attempt + 1)
    raise RuntimeError(f'Could not download {destination.name}. Retry the setup cell. '
                       'You can also upload the unchanged course files from GitHub to this workspace; '
                       'their checksums will still be verified. No subscription is required.') from last_error


def prepare(root: str | Path, source_ref: str, manifest_sha256: str,
            *, install: bool = True) -> dict:
    started = time.perf_counter()
    if not re.fullmatch(r'[0-9a-f]{40}', source_ref):
        raise ValueError('Use the 40-character course revision from this notebook.')
    if not re.fullmatch(r'[0-9a-f]{64}', manifest_sha256):
        raise ValueError('A pinned manifest SHA256 is required.')
    if sys.version_info[:2] not in [(3, 12), (3, 13)]:
        raise RuntimeError('This release supports Python 3.12 and 3.13. Use the current CPU Colab runtime.')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base_url = f'https://raw.githubusercontent.com/{OWNER_REPO}/{source_ref}/'
    manifest_path = root / 'data/data_manifest.json'
    fetch_verified(base_url + 'data/data_manifest.json', manifest_path, manifest_sha256)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    sources = {}
    for relative, entry in manifest['files'].items():
        destination = (root / relative).resolve()
        if not destination.is_relative_to(root) or '\\' in relative:
            raise ValueError(f'Unsafe course file path: {relative}')
        sources[relative] = fetch_verified(base_url + relative, destination, entry['sha256'])
    requirements = {}
    for line in (root / 'requirements-colab.txt').read_text(encoding='utf-8').splitlines():
        if line and not line.startswith('#'):
            name, version = line.split('==')
            requirements[name] = version
    missing = []
    for name, version in requirements.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        if actual != version:
            module = sys.modules.get(MODULES[name])
            if module is not None:
                raise RuntimeError(f'{name} was already imported with a different version. '
                                   'Choose Runtime → Restart session, then run the setup cell first.')
            missing.append(f'{name}=={version}')
    if missing:
        if not install:
            raise RuntimeError('Missing required versions: ' + ', '.join(missing))
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                                 '--quiet', '-r', str(root / 'requirements-colab.txt'),
                                 '-c', str(root / 'constraints.txt')], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Package installation did not finish. Retry on a CPU runtime.\n' + result.stderr[-4000:])
        importlib.invalidate_caches()
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = '2'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['MPLCONFIGDIR'] = str(root / '.cache/matplotlib')
    random.seed(211)
    import numpy as np
    np.random.seed(211)
    for folder in ['artifacts', 'reports', 'submission']:
        (root / folder).mkdir(exist_ok=True)
    scripts = str(root / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    versions = {name: metadata.version(name) for name in requirements}
    for name, expected in requirements.items():
        if versions[name] != expected:
            raise RuntimeError(f'{name}: expected {expected}, found {versions[name]}. Restart the runtime.')
    report = {'python': platform.python_version(), 'platform': platform.system(),
              'course_revision': source_ref, 'dataset_version': manifest['dataset_version'],
              'seed': 211, 'n_jobs': 2, 'fast_mode': True, 'full_mode': False,
              'versions': versions, 'sources': sources,
              'setup_seconds': round(time.perf_counter() - started, 2)}
    (root / 'artifacts/environment.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Package                 Version')
    for name, version in versions.items():
        print(f'{name:23} {version}')
    print(f"Python {report['python']} | CPU | seed=211 | n_jobs=2 | FAST_MODE=True")
    print('Verified course files. Next: run the data and compatibility checks.')
    return report
