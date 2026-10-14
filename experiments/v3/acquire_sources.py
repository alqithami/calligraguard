#!/usr/bin/env python3
"""Acquire public OFL font inputs at a pinned revision. Never package font binaries."""
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
import argparse, hashlib, json, re, time

def sha(data):
    return hashlib.sha256(data).hexdigest()

def fetch(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'CFDefect-reproducible-benchmark'}), timeout=90) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, default=Path('source_plan.json'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(exist_ok=True, parents=True)
    lock = args.out / 'sources.json'
    if lock.exists():
        saved = json.loads(lock.read_text())
        if saved['plan_sha256'] != sha(args.plan.read_bytes()):
            raise ValueError('Source plan changed; create a new versioned build')
        for source in saved['fonts']:
            if sha((args.out / source['file']).read_bytes()) != source['sha256']:
                raise ValueError('Font input checksum mismatch')
        print('SOURCES_VERIFIED', len(saved['fonts']), flush=True)
        return
    tasks = [(name, split) for split in ['train', 'development', 'calibration', 'test_family', 'test_latin'] for name in plan[split]]
    def acquire(task):
        name, split = task
        base = 'https://raw.githubusercontent.com/google/fonts/' + plan['google_fonts_commit'] + '/ofl/' + name + '/'
        try:
            metadata = fetch(base + 'METADATA.pb')
            notice = fetch(base + 'OFL.txt')
            if b'SIL OPEN FONT LICENSE' not in notice.upper():
                raise ValueError('Expected OFL notice unavailable')
            if split != 'test_latin' and not re.search(r'subsets:\s*"arabic"', metadata.decode()):
                raise ValueError('Source does not advertise Arabic coverage')
            names = re.findall(r'filename:\s*"([^"]+\.ttf)"', metadata.decode())
            preferred = [f for f in names if 'Regular' in f or ('[' in f and 'Italic' not in f)]
            filename = (preferred or names)[0]
            content = fetch(base + quote(filename, safe=''))
            destination = args.out / name
            destination.mkdir(exist_ok=True)
            (destination / filename).write_bytes(content)
            (destination / 'OFL.txt').write_bytes(notice)
            (destination / 'METADATA.pb').write_bytes(metadata)
            from fontTools.ttLib import TTFont
            with TTFont(destination / filename) as font:
                cmap = font.getBestCmap()
                source = {'source_directory': name, 'family_name': font['name'].getDebugName(16) or font['name'].getDebugName(1),
                          'family_group': plan['ancestry_groups'].get(name, name), 'split': split,
                          'file': name + '/' + filename, 'sha256': sha(content), 'source_url': base + quote(filename, safe=''),
                          'version': font['name'].getDebugName(5), 'license_url': base + 'OFL.txt',
                          'license_sha256': sha(notice), 'git_commit': plan['google_fonts_commit'],
                          'arabic_block_coverage': sum(0x620 <= cp <= 0x6ff for cp in cmap)}
            print('SOURCE_READY', name, split, flush=True)
            return source, None
        except Exception as exc:
            error = {'source_directory': name, 'split': split, 'error': str(exc)}
            print('SOURCE_UNAVAILABLE', json.dumps(error), flush=True)
            return None, error
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(acquire, tasks))
    fonts = [source for source, error in results if source]
    errors = [error for source, error in results if error]
    group_splits = {}
    for source in fonts:
        group_splits.setdefault(source['family_group'], set()).add(source['split'])
    if any(len(v) != 1 for v in group_splits.values()):
        raise ValueError('Source ancestry group crosses partitions')
    counts = {split: sum(s['split'] == split for s in fonts) for split in ['train', 'development', 'calibration', 'test_family', 'test_latin']}
    report = {'plan_sha256': sha(args.plan.read_bytes()), 'fonts': fonts, 'unavailable_sources': errors, 'source_counts': counts,
              'ancestry_note': plan['limitations'], 'font_binaries_excluded_from_release': True}
    minimum = {'train': 16, 'development': 3, 'calibration': 3, 'test_family': 3, 'test_latin': 3}
    if any(counts[s] < minimum[s] for s in minimum):
        (args.out / 'SOURCE_FAILURE.json').write_text(json.dumps(report, indent=2))
        raise RuntimeError('Source coverage floor not met; study must not silently shrink')
    lock.write_text(json.dumps(report, indent=2) + '\n')
    print('SOURCES_COMPLETE', json.dumps(counts), flush=True)

if __name__ == '__main__':
    main()
