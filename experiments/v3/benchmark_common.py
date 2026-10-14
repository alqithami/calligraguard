#!/usr/bin/env python3
"""Dataset, integrity, calibration, and multi-task metrics shared by all v3 methods."""
from __future__ import annotations
import base64, collections, functools, gzip, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

TYPES=['spur','notch','local_deformation','missing_mark','misplaced_mark']
TEST_SPLITS=['test_family','test_renderer','test_resolution128','test_resolution512','test_latin']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def dump(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(p)
def readrows(path):
    with Path(path).open() as f:return [json.loads(l) for l in f if l.strip()]
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def state_hash(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@functools.lru_cache(maxsize=64)
def reference_arrays(path):
    with np.load(path,allow_pickle=False) as z:return z['image'].copy(),z['geometry'].astype(np.float32)

class BenchmarkData(Dataset):
    def __init__(self,root,split,normal_only=False,allow_test=False):
        if split in TEST_SPLITS and not allow_test:raise ValueError('Locked test cannot be used in training/development loader')
        self.root=Path(root);self.split=split;self.rows=[r for r in readrows(self.root/'meta.jsonl') if r['split']==split and (not normal_only or r['label']==0)]
        if not self.rows:raise ValueError('Empty dataset partition: '+split)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];ref,geo=reference_arrays(str(self.root/row['reference_file']))
        with np.load(self.root/row['file'],allow_pickle=False) as z:
            query=z['query'].copy();mask=z['mask'].copy()
            if 'reference_at_size' in z:ref=z['reference_at_size'].copy()
        q=torch.tensor(query,dtype=torch.float32)/255.;r=torch.tensor(ref,dtype=torch.float32)/255.
        if q.shape!=r.shape:raise ValueError('Query/reference dimensions do not agree')
        x=torch.stack([(q-.5)/.25,(r-.5)/.25,abs(q-r)/.25,(q-r)/.25])
        geometry=torch.from_numpy(geo)
        if geometry.shape[-2:]!=q.shape:geometry=F.interpolate(geometry[None],size=q.shape,mode='bilinear',align_corners=False)[0]
        return x,geometry,torch.tensor(mask,dtype=torch.float32)[None],torch.tensor(float(row['label'])),torch.tensor(TYPES.index(row['type']) if row['label'] else -1),row['id']

class NormalRGB(Dataset):
    def __init__(self,dataset,size=256,indices=None):self.dataset=dataset;self.size=size;self.indices=list(range(len(dataset))) if indices is None else list(indices)
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        x,geo,mask,y,t,sid=self.dataset[self.indices[i]];q=(x[0]*.25+.5).clamp(0,1)
        rgb=q[None].expand(3,-1,-1)
        if rgb.shape[-2:]!=(self.size,self.size):rgb=F.interpolate(rgb[None],size=(self.size,self.size),mode='bilinear',align_corners=False)[0]
        mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
        return (rgb-mean)/std,sid

def validate_dataset(root,out):
    root=Path(root);allrows=readrows(root/'all_records.jsonl');rows=readrows(root/'meta.jsonl');only=readrows(root/'observability_only.jsonl')
    if len({r['id'] for r in allrows})!=len(allrows):raise ValueError('Duplicate IDs')
    if {r['id'] for r in rows}&{r['id'] for r in only}:raise ValueError('Observation partitions overlap')
    if {r['id'] for r in rows+only}!={r['id'] for r in allrows}:raise ValueError('Lost generated record')
    broad=lambda s:'test' if s in TEST_SPLITS else s
    group_splits=collections.defaultdict(set);parent_splits=collections.defaultdict(set);refs={};failed=[]
    for i,r in enumerate(allrows):
        group_splits[r['family_group']].add(broad(r['split']));parent_splits[r['parent_id']].add(broad(r['split']))
        p=root/r['file'];ref=root/r['reference_file']
        if sha(p)!=r['file_sha256']:failed.append([r['id'],'query hash mismatch']);continue
        if str(ref) not in refs:
            refs[str(ref)]=sha(ref)
            with np.load(ref,allow_pickle=False) as z:
                if z['geometry'].shape!=(4,256,256) or not np.isfinite(z['geometry']).all():failed.append([r['id'],'invalid reference geometry'])
        if refs[str(ref)]!=r['reference_sha256']:failed.append([r['id'],'reference hash mismatch'])
        with np.load(p,allow_pickle=False) as z:
            q=z['query'];mask=z['mask'];area=int(mask.sum())
            if q.dtype!=np.uint8 or q.shape!=mask.shape or q.shape!=(r['render_size'],r['render_size']):failed.append([r['id'],'shape/dtype'])
            if not np.isin(mask,[0,1]).all() or area!=r['mask_pixels']:failed.append([r['id'],'invalid mask'])
            if r['label']==0 and area:failed.append([r['id'],'clean record contains mask'])
            if r['observation_label']!=int(r['label']==1 and area>0):failed.append([r['id'],'observation label mismatch'])
            if r['label'] and not (q<128).any():failed.append([r['id'],'whole glyph deleted'])
        if (i+1)%5000==0:print('VALIDATE',i+1,'/',len(allrows),flush=True)
    if any(len(v)>1 for v in group_splits.values()):failed.append(['partition','source group leakage'])
    if any(len(v)>1 for v in parent_splits.values()):failed.append(['partition','parent leakage'])
    if any(r['label'] and not r['observation_label'] for r in rows):failed.append(['partition','unobservable positive in primary data'])
    cal=[r for r in rows if r['split']=='calibration'];normal_parents=len({r['parent_id'] for r in cal})
    if any(r['label'] for r in cal):failed.append(['calibration','contains a defective example'])
    minima={'train':20000,'development':1000,'calibration':1000,'test_family':1000,'test_latin':500}
    counts=dict(collections.Counter(r['split'] for r in rows))
    for s,n in minima.items():
        if counts.get(s,0)<n:failed.append([s,'coverage floor not reached: '+str(counts.get(s,0))])
    if normal_parents<150:failed.append(['calibration','fewer than 150 distinct base glyphs'])
    report={'status':'PASS' if not failed else 'FAIL','generated':len(allrows),'primary':len(rows),'observation_only':len(only),'counts':counts,
        'calibration_base_glyphs':normal_parents,'distinct_training_base_glyphs':len({r['parent_id'] for r in rows if r['split']=='train'}),
        'source_group_partitions':{k:sorted(v) for k,v in group_splits.items()},'failures':failed,'meta_sha256':sha(root/'meta.jsonl'),
        'statement':'Automated geometric/data integrity only; not expert realism or full genealogical independence'}
    dump(out,report)
    if failed:raise RuntimeError('Dataset validity failed: '+str(failed[:8]))
    return report

def calibration_threshold(rows,scores,alpha):
    if len(rows)!=len(scores) or not np.isfinite(scores).all():raise ValueError('Invalid calibration predictions')
    if len({r['id'] for r in rows})!=len(rows) or any(r['label'] for r in rows):raise ValueError('Calibration IDs/labels invalid')
    byparent=collections.defaultdict(list)
    for r,s in zip(rows,scores):byparent[r['parent_id']].append(float(s))
    ordered=np.sort([max(v) for v in byparent.values()]);rank=math.ceil((len(ordered)+1)*(1-alpha))
    return (float(ordered[rank-1]) if rank<=len(ordered) else None),len(ordered)

def intervals(rows,flags,seed=81013,reps=1000):
    # Resample base-glyph groups within observed source families. Conditional on those families.
    families=collections.defaultdict(lambda:collections.defaultdict(lambda:np.zeros(4,dtype=np.int64)))
    for row,f in zip(rows,flags):
        a=families[row['source_family']][row['parent_id']]
        if row['label']:a[0]+=int(f);a[1]+=1
        else:a[2]+=int(f);a[3]+=1
    matrices=[np.stack(list(d.values())) for d in families.values()];rng=np.random.default_rng(seed);values=[]
    for _ in range(reps):
        counts=sum((a[rng.integers(len(a),size=len(a))].sum(0) for a in matrices),start=np.zeros(4,dtype=np.int64))
        values.append([counts[0]/max(1,counts[1]),counts[2]/max(1,counts[3])])
    lo,hi=np.quantile(values,[.025,.975],axis=0)
    return {'recall_95pct':[float(lo[0]),float(hi[0])],'image_fpr_95pct':[float(lo[1]),float(hi[1])],
            'resampling':'base glyphs within observed source families; conditional, not a confidence claim for new families','replicates':reps}

def detection_metrics(calrows,cal_scores,rows,scores,bootstrap=True):
    if set(r['family_group'] for r in calrows)&set(r['family_group'] for r in rows):raise ValueError('Calibration/evaluation family overlap')
    y=np.asarray([r['label'] for r in rows],np.int8);s=np.asarray(scores,float)
    if len(y)!=len(s) or len({r['id'] for r in rows})!=len(rows) or not np.isfinite(s).all():raise ValueError('Incomplete, duplicate or nonfinite prediction set')
    result={'n':len(y),'normal':int((y==0).sum()),'defective':int((y==1).sum()),'auroc':float(roc_auc_score(y,s)),
        'average_precision':float(average_precision_score(y,s)),'standardized_partial_auc_fpr005':float(roc_auc_score(y,s,max_fpr=.05)),
        'scope':'Strict thresholds from normal calibration parent maxima; no distribution-shift guarantee','operating_points':{}}
    for alpha in [.01,.05]:
        t,n=calibration_threshold(calrows,np.asarray(cal_scores),alpha);flag=s>t if t is not None else np.zeros(len(s),bool)
        parents=collections.defaultdict(list)
        for r,f in zip(rows,flag):
            if not r['label']:parents[r['parent_id']].append(bool(f))
        op={'threshold':t,'threshold_is_infinite':t is None,'calibration_parent_count':n,'recall':float(flag[y==1].mean()),
            'actual_image_fpr':float(flag[y==0].mean()),'actual_parent_any_view_fpr':float(np.mean([any(v) for v in parents.values()])),
            'per_type_recall':{typ:float(flag[[r['type']==typ for r in rows]].mean()) for typ in TYPES if any(r['type']==typ for r in rows)}}
        if bootstrap:op['conditional_group_bootstrap']=intervals(rows,flag)
        result['operating_points'][str(alpha)]=op
    return result

def pack_mask(mask):return base64.b64encode(np.packbits(np.asarray(mask,dtype=np.uint8).ravel()).tobytes()).decode('ascii')

def summarize_dense(rows,pred):
    positive=[(r,pred[r['id']]) for r in rows if r['label']];negative=[pred[r['id']] for r in rows if not r['label']]
    out={'mask_threshold':.5,'positive_count':len(positive),'mean_dice':float(np.mean([p['dice'] for r,p in positive])),
         'mean_iou':float(np.mean([p['iou'] for r,p in positive])),
         'clean_predicted_pixel_fraction':float(np.mean([p['predicted_mask_fraction'] for p in negative])),
         'per_type_dice':{t:float(np.mean([p['dice'] for r,p in positive if r['type']==t])) for t in TYPES if any(r['type']==t for r,p in positive)}}
    if all('type_logits' in p for r,p in positive):
        true=[TYPES.index(r['type']) for r,p in positive];guessed=[int(np.argmax(p['type_logits'])) for r,p in positive]
        out['type_top1']=float(np.mean(np.asarray(true)==guessed));out['type_macro_f1']=float(f1_score(true,guessed,labels=sorted(set(true)),average='macro',zero_division=0))
        out['type_top3']=float(np.mean([TYPES.index(r['type']) in np.argsort(p['type_logits'])[-3:] for r,p in positive]))
    return out
