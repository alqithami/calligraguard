#!/usr/bin/env python3
"""Verify and publish the exact approved CalligraGuard package.

Default: local checks only. --publish creates/resumes a draft, verifies every
uploaded asset by downloading it, then publishes a pre-release. No training,
server operations, journal submission, credential extraction, or asset deletion.
Requires Python 3.9+ and an authenticated GitHub CLI for --publish only.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
REPO = 'alqithami/calligraguard'
TAG = 'source-balanced-2026-10-31'
ROOT = 'CalligraGuard_Completed_20261031'
PACKAGE = 'CalligraGuard_Complete_Package_20261031.zip'
FORBIDDEN = {'.ttf', '.otf', '.ttc', '.woff', '.woff2', '.sfd', '.glyph', '.glyphs'}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def safe_name(name: str) -> bool:
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name


def check_file(path: Path, expected: dict) -> None:
    require(path.is_file(), f'Missing file: {path}')
    require(path.stat().st_size == expected['bytes'], f'Size mismatch: {path.name}')
    require(digest(path) == expected['sha256'], f'SHA-256 mismatch: {path.name}')


def verify_archive(path: Path, expected: dict) -> int:
    check_file(path, expected)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), 'Duplicate archive members')
        require(sum(i.file_size for i in z.infolist()) < 1_000_000_000,
                'Unexpected expanded archive size')
        for i in z.infolist():
            require(safe_name(i.filename), 'Unsafe archive path')
            require(not stat.S_ISLNK(i.external_attr >> 16), 'Symlink in archive')
            require(PurePosixPath(i.filename).suffix.lower() not in FORBIDDEN,
                    'A font or original font-source file must not be published')
        manifest = json.loads(z.read(ROOT + '/PACKAGE_MANIFEST.json'))
        require(len(manifest['files']) == 395, 'Unexpected package manifest')
        listed = {ROOT + '/' + n for n in manifest['files']}
        require(set(names) == listed | {ROOT + '/PACKAGE_MANIFEST.json'},
                'Unmanifested or missing archive members')
        for name, item in manifest['files'].items():
            require(safe_name(name), 'Unsafe manifest path')
            content = z.read(ROOT + '/' + name)
            require(len(content) == item['bytes'], f'Member size mismatch: {name}')
            require(hashlib.sha256(content).hexdigest() == item['sha256'],
                    f'Member SHA-256 mismatch: {name}')
    return len(manifest['files'])


def stage_assets(archive: Path, manifest: dict, destination: Path) -> dict[str, Path]:
    verify_archive(archive, manifest['assets'][PACKAGE])
    destination.mkdir(parents=True, exist_ok=True)
    assets = {PACKAGE: archive}
    with zipfile.ZipFile(archive) as z:
        for name, item in manifest['assets'].items():
            require(PurePosixPath(name).name == name and safe_name(name), 'Unsafe asset name')
            if name == PACKAGE:
                continue
            target = destination / name
            if 'package_member' in item:
                content = z.read(item['package_member'])
            else:
                content = (HERE / name).read_bytes()
            if target.exists():
                require(target.read_bytes() == content, f'Refusing to overwrite {target}')
            else:
                target.write_bytes(content)
            check_file(target, item)
            assets[name] = target
    for name, content in {
        'ASSET_MANIFEST.json': (HERE / 'ASSET_MANIFEST.json').read_bytes(),
        'SHA256SUMS.txt': ''.join(f"{v['sha256']}  {n}\n" for n, v in sorted(manifest['assets'].items())).encode()
    }.items():
        target = destination / name
        if target.exists():
            require(target.read_bytes() == content, f'Refusing to overwrite {target}')
        else:
            target.write_bytes(content)
        assets[name] = target
    return assets


def gh(*args: str, allow_404: bool = False) -> str | None:
    p = subprocess.run(['gh', *args], capture_output=True, text=True, timeout=900)
    if p.returncode:
        if allow_404 and 'HTTP 404' in p.stderr:
            return None
        raise RuntimeError('GitHub CLI failed: ' + p.stderr[-1500:])
    return p.stdout.strip()


def public_hash(url: str, maximum: int) -> str:
    request = urllib.request.Request(url, headers={'User-Agent': 'CalligraGuard-release-verifier/1'})
    h = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=90) as r:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            require(size <= maximum, 'Unexpected public asset size')
            h.update(chunk)
    require(size == maximum, 'Incomplete public asset download')
    return h.hexdigest()


def publish(assets: dict[str, Path], target: str, output: Path) -> dict:
    require(bool(re.fullmatch(r'[0-9a-f]{40}', target)), 'Provide an exact --target-commit SHA')
    require(shutil.which('gh') is not None, 'GitHub CLI is not installed')
    gh('auth', 'status', '--hostname', 'github.com')
    commit = json.loads(gh('api', f'repos/{REPO}/commits/{target}'))
    require(commit['sha'] == target, 'Unexpected target commit')
    raw = gh('api', f'repos/{REPO}/releases/tags/{TAG}', allow_404=True)
    if raw is None:
        gh('release', 'create', TAG, '--repo', REPO, '--target', target,
           '--title', 'CalligraGuard source-balanced study: approved 31 October 2026 package',
           '--notes-file', str(HERE / 'RELEASE_NOTES.md'), '--draft', '--prerelease', '--latest=false')
        raw = gh('api', f'repos/{REPO}/releases/tags/{TAG}')
    release = json.loads(raw)
    require(release['tag_name'] == TAG and release['target_commitish'] == target,
            'Existing release has a different target; no overwrite attempted')
    present = {a['name']: a for a in release['assets']}
    require(set(present) <= set(assets), 'Unexpected assets in existing release')
    missing = [str(path) for name, path in assets.items() if name not in present]
    require(release['draft'] or not missing, 'Published release is incomplete; refusing modification')
    if missing:
        gh('release', 'upload', TAG, '--repo', REPO, *missing)
    with tempfile.TemporaryDirectory(prefix='cg-readback-') as tmp:
        for name, path in assets.items():
            gh('release', 'download', TAG, '--repo', REPO, '--pattern', name, '--dir', tmp)
            require(digest(Path(tmp) / name) == digest(path), f'Uploaded bytes differ: {name}')
    if release['draft']:
        gh('release', 'edit', TAG, '--repo', REPO, '--draft=false', '--prerelease', '--latest=false')
    release = json.loads(gh('api', f'repos/{REPO}/releases/tags/{TAG}'))
    require(not release['draft'], 'Release is still a draft')
    remote_assets = {a['name']: a for a in release['assets']}
    require(set(remote_assets) == set(assets), 'Final asset inventory mismatch')
    for name, path in assets.items():
        require(public_hash(remote_assets[name]['browser_download_url'], path.stat().st_size) == digest(path),
                f'Unauthenticated public download did not verify: {name}')
    result = {'status': 'published_prerelease_all_assets_download_hash_verified',
              'verified_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'repository': REPO, 'tag': TAG, 'target_commit': target,
              'url': release['html_url'], 'asset_count': len(assets),
              'assets': {n: {'bytes': p.stat().st_size, 'sha256': digest(p)} for n, p in assets.items()},
              'large_raw_transfer_archive_published': False,
              'journal_submission_made': False, 'new_experiments': False}
    receipt = output / 'PUBLICATION_VERIFIED.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        require(all(old.get(k) == result[k] for k in ('url', 'assets', 'target_commit')),
                'Existing publication receipt differs; no overwrite attempted')
        return old
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=Path.home() / 'Downloads' / PACKAGE)
    p.add_argument('--stage', type=Path, default=Path.home() / 'Projects/CalligraGuard_20261013/publication_20261101/assets')
    p.add_argument('--publish', action='store_true')
    p.add_argument('--target-commit')
    args = p.parse_args()
    manifest = json.loads((HERE / 'ASSET_MANIFEST.json').read_text())
    require(manifest['repository'] == REPO and manifest['planned_tag'] == TAG, 'Unexpected release destination')
    if args.publish:
        require(args.target_commit is not None, '--publish requires --target-commit')
    assets = stage_assets(args.archive.resolve(), manifest, args.stage.resolve())
    result = {'status': 'local_assets_verified_not_published', 'assets': len(assets),
              'manifest_members_verified': 395, 'journal_submission_made': False}
    if args.publish:
        result = publish(assets, args.target_commit, args.stage.resolve())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
