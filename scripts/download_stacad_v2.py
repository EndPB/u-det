"""Download the pinned public STACAD v2 package with disk and checksum guards."""
import hashlib
import json
import shutil
from pathlib import Path
import requests
import time

ROOT = Path(__file__).resolve().parents[1] / 'd-det/data/stacad_v2'
RECORD = '22809889'

def digest(path):
    h = hashlib.md5()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ROOT).free < 30 * 1024**3:
        raise RuntimeError('Keep at least 30 GiB free before downloading')
    api = requests.get(f'https://zenodo.org/api/records/{RECORD}', timeout=30)
    api.raise_for_status()
    j = api.json()
    (ROOT/'zenodo_record.json').write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
    for spec in j['files']:
        dst = ROOT / spec['key']
        expected = spec['checksum'].split(':')[1]
        if dst.exists() and digest(dst) == expected:
            print('verified existing',dst.name,flush=True)
            continue
        tmp = dst.with_suffix(dst.suffix+'.part')
        got = 0
        url = spec['links']['self']
        # Zenodo serves the large archive reliably through its download URL;
        # the API self link may wait indefinitely on some Windows clients.
        if spec['key'] == 'stacad-v2.zip':
            url = f'https://zenodo.org/records/{RECORD}/files/stacad-v2.zip?download=1'
        # Zenodo's large-file endpoint can keep a normal GET open without
        # yielding bytes on Windows.  Use bounded HTTP ranges so a stalled
        # connection cannot hold the whole download and retries are cheap.
        if spec['key'] == 'stacad-v2.zip':
            total = int(spec['size'])
            chunk = 8 * 1024 * 1024
            with tmp.open('wb') as f:
                f.truncate(total)
            with tmp.open('r+b') as f:
                for start in range(0, total, chunk):
                    end = min(total - 1, start + chunk - 1)
                    for attempt in range(5):
                        try:
                            with requests.get(url, headers={'Range': f'bytes={start}-{end}'},
                                              timeout=(20, 60), allow_redirects=True) as r:
                                r.raise_for_status()
                                cr = r.headers.get('content-range', '')
                                if not cr.startswith(f'bytes {start}-{end}/'):
                                    raise IOError(f'unexpected content-range: {cr}')
                                data = r.content
                            expected_len = end - start + 1
                            if len(data) != expected_len:
                                raise IOError(f'range length {len(data)} != {expected_len}')
                            f.seek(start); f.write(data)
                            got = end + 1
                            break
                        except Exception:
                            if attempt == 4: raise
                            time.sleep(1.5 * (attempt + 1))
                    if got % (32*1024*1024) < chunk or got == total:
                        print(dst.name,round(got/1e6,1),'MB',flush=True)
        else:
            with requests.get(url, stream=True, timeout=(20,120), allow_redirects=True) as r:
                r.raise_for_status()
                with tmp.open('wb') as f:
                    for b in r.iter_content(1024*1024):
                        if b:
                            f.write(b); got += len(b)
                            if got % (32*1024*1024) < len(b):
                                print(dst.name,round(got/1e6,1),'MB',flush=True)
        assert got == spec['size'], (got,spec['size'])
        assert digest(tmp) == expected, 'Source checksum mismatch'
        tmp.replace(dst)
        print('downloaded and verified',dst.name,got,flush=True)

if __name__ == '__main__': main()
