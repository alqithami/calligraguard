#!/usr/bin/env python3
"""Verify the pinned public release and summary arithmetic, not raw-data reproduction.
No training, inference, server access, or large archive extraction. Python 3.9+.
"""
from __future__ import annotations
import argparse,ast,datetime,hashlib,io,json,math,statistics,tarfile,urllib.request
from pathlib import Path,PurePosixPath
COMMIT='c2aa1af2b7f816b710d96d84f3637755006d8da2'
BASE=f'https://raw.githubusercontent.com/alqithami/calligraguard/{COMMIT}/'
DOMAINS={'test_family','test_renderer','test_resolution128','test_resolution512','test_latin'}
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(b):return hashlib.sha256(b).hexdigest()
def safe(n):return not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts and '\\' not in n
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkout',type=Path);ap.add_argument('--output',type=Path);args=ap.parse_args()
 def read(name):
  require(safe(name),'Unsafe path')
  if args.checkout:return (args.checkout/name).read_bytes()
  req=urllib.request.Request(BASE+name,headers={'User-Agent':'CalligraGuard-release-check/1'})
  with urllib.request.urlopen(req,timeout=40) as r:b=r.read(8000001)
  require(len(b)<=8000000,'Unexpected file size');return b
 prefix='reproducibility/20261024/';manifest=json.loads(read(prefix+'MANIFEST.json'));require(len(manifest['archives'])==5,'Expected five bundles');members={};checked=[];code=0
 for name,m in manifest['archives'].items():
  b=read(prefix+name);require(len(b)==m['bytes'] and sha(b)==m['sha256'],'SHA-256 mismatch '+name)
  require(hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==m['git_blob_sha1'],'Git blob mismatch')
  with tarfile.open(fileobj=io.BytesIO(b),mode='r:xz') as t:
   rr=[x for x in t.getmembers() if x.isfile()];require(len(rr)==m['members'],'Member count');require(sum(x.size for x in rr)<20000000,'Expanded size')
   for x in t:
    require(safe(x.name) and not x.issym() and not x.islnk(),'Unsafe member')
    if not x.isfile():continue
    require(Path(x.name).suffix.lower() not in {'.ttf','.otf','.ttc','.woff','.woff2','.glyph','.glyphs'},'Font or source input in bundle')
    k=(name,x.name);require(k not in members,'Duplicate member');v=t.extractfile(x).read();members[k]=v
    if x.name.endswith('.py'):ast.parse(v,filename=x.name);code+=1
  checked.append({'name':name,'sha256':sha(b),'members':len(rr)})
 def unique(n):
  found=[v for (_,p),v in members.items() if PurePosixPath(p).name==n];require(len(found)==1,'Unique '+n);return found[0]
 s=json.loads(unique('RESULTS_SUMMARY.json'));require(len(s['methods'])==6 and sum(s['replicates'].values())==22,'Models/seeds');arithmetic=0;learned=0;classical=0
 for method,dd in s['methods'].items():
  require(set(dd)==DOMAINS,'Learned domains')
  for metrics in dd.values():
   learned+=1
   for r in metrics.values():
    v=r['values'];require(len(v)==s['replicates'][method] and all(math.isfinite(x) for x in v),'Seed data')
    require(math.isclose(statistics.mean(v),r['mean'],rel_tol=1e-10,abs_tol=1e-12),'Mean mismatch');require(math.isclose(statistics.stdev(v),r['sample_std'],rel_tol=1e-8,abs_tol=1e-12),'SD mismatch');arithmetic+=2
 require(set(s['classical'])==DOMAINS,'Classical domains')
 for mm in s['classical'].values():
  require(len(mm)==12,'Classical count')
  for metrics in mm.values():require(all(math.isfinite(x) for x in metrics.values()),'Nonfinite classical');classical+=1
 require(learned==30 and classical==60,'Incomplete grid')
 official=json.loads(unique('DIFF3_VERIFIED_COUNTS.json'));require(len(official['rows'])==16 and all(r['success'] for r in official['rows']),'Official summary')
 o={r['id']:r for r in official['rows']};require(o['urdu_historical']['word_image_records']==1032 and o['urdu_all_contours_reversed']['word_image_records']==44,'Native summary counts')
 n=0
 for line in read('experiments/v3/SOURCE_SHA256SUMS').decode().splitlines():
  if not line.strip():continue
  digest,name=line.split(maxsplit=1);name=name.strip().lstrip('*');require(sha(read('experiments/v3/'+name))==digest,'Frozen source '+name);n+=1
 require(n==14,'Source count')
 result={'status':'public_release_integrity_and_summary_arithmetic_passed','verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pinned_commit':COMMIT,'fresh_public_downloads':args.checkout is None,'archives':checked,'python_files_syntax_checked':code,'frozen_v3_scientific_files':n,'learned_method_condition_rows':learned,'classical_method_condition_rows':classical,'seed_mean_sd_checks':arithmetic,'official_summary_rows':16,'not_verified':['raw-score AUROC/FPR recalculation','pixel intersections','training or inference replay','all historical traces','full scientific data access','human or production validation']}
 if args.output:
  args.output.parent.mkdir(parents=True,exist_ok=True)
  with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
