"""Verify the frozen A/B/C bundle without opening test or running models."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'d-det/data/mainline_abc_200k_2026-10-12'
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
 return h.hexdigest()
def main():
 m=json.loads((DATA/'manifest.json').read_text(encoding='utf8'))
 for rel,item in m['files'].items():
  p=DATA/rel; assert p.is_file() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],rel
 assert not list(DATA.rglob('test*'))
 print(json.dumps({'status':'BUNDLE_OK','files':len(m['files']),'tasks':list(m['tasks']),'test_opened':False},ensure_ascii=False))
if __name__=='__main__': main()
