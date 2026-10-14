#!/usr/bin/env python3
"""Versioned multi-task data build with new families, explicit observation accounting,
independent renderer shift, Latin transfer, and unchanged frozen v2 geometry helpers.
No model predictions participate in generation, acceptance, or splitting.
"""
from __future__ import annotations
import argparse, collections, concurrent.futures, hashlib, io, json, math, os, time, unicodedata
from pathlib import Path
import numpy as np
import cv2, skia, pathops
from PIL import Image, ImageDraw
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.transformPen import TransformPen
from scipy.spatial import cKDTree
import geometry_helpers as g

HELPER_HASH='cb24f1de80bd5408e69f01ed6788bffe726099b3c1cf0e263b49429232f92bab'
TYPES=['spur','notch','local_deformation','missing_mark','misplaced_mark']
MARKS=[0x064e,0x064f,0x0650]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def rows_write(p,rows):
    with Path(p).open('w') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')

class SkiaPen(BasePen):
    def __init__(self):super().__init__(None);self.path=skia.Path()
    def _moveTo(self,p):self.path.moveTo(*p)
    def _lineTo(self,p):self.path.lineTo(*p)
    def _qCurveToOne(self,a,b):self.path.quadTo(*a,*b)
    def _curveToOne(self,a,b,c):self.path.cubicTo(*a,*b,*c)
    def _closePath(self):self.path.close()
    def _endPath(self):pass

def render(path,size=256,renderer='cairo',shift=(0.,0.),aa=1,gamma=1.):
    if renderer=='cairo':return g.render(path,size,shift,aa,gamma)
    if renderer!='skia':raise ValueError(renderer)
    surface=skia.Surface(size*aa,size*aa);canvas=surface.getCanvas();canvas.clear(skia.ColorWHITE)
    canvas.scale(size*aa/256,size*aa/256);canvas.translate(*shift)
    pen=SkiaPen();path.draw(pen);canvas.drawPath(pen.path,skia.Paint(Color=skia.ColorBLACK,AntiAlias=True))
    array=surface.makeImageSnapshot().toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)[:,:,0]
    if aa>1:array=np.array(Image.fromarray(array).resize((size,size),Image.Resampling.LANCZOS))
    if gamma!=1:array=np.rint(255*(array.astype(np.float64)/255)**gamma).clip(0,255).astype(np.uint8)
    return array

def latin_shape(fontfile,cp):
    with TTFont(fontfile) as font:
        location={a.axisTag:a.defaultValue for a in font['fvar'].axes} if 'fvar' in font else {}
        glyphs=font.getGlyphSet(location=location);name=font.getBestCmap()[cp];p=pathops.Path()
        rec=DecomposingRecordingPen(glyphs);glyphs[name].draw(rec);rec.replay(p.getPen())
        x0,y0,x1,y1=p.bounds;scale=200/max(x1-x0,y1-y0)
        base=pathops.Path();p.draw(TransformPen(base.getPen(),(scale,0,0,-scale,128-(x0+x1)*scale/2,128+(y0+y1)*scale/2)))
        base=pathops.simplify(base)
    return base,pathops.Path(),pathops.Path(base),[{'role':'base','glyph_name':name}]

def reference_fields(path,reference):
    # Geometry comes only from the trusted outline, never the injected edit or mask.
    tokens,cids,candidates=g.contour_tokens(path,64)
    points=np.array([r[0] for r in candidates]);tangents=np.array([r[1] for r in candidates])
    yy,xx=np.mgrid[:256,:256];distance,index=cKDTree(points).query(np.c_[xx.ravel()+.5,yy.ravel()+.5])
    sign=np.where(reference.ravel()<128,-1.,1.);sdf=np.clip(distance*sign/32,-1,1).reshape(256,256)
    tangent=tangents[index].reshape(256,256,2);boundary=np.exp(-distance.reshape(256,256)/4)
    fields=np.stack([sdf,tangent[:,:,0]*boundary,tangent[:,:,1]*boundary,boundary]).astype(np.float16)
    return fields,tokens,cids

def svg_text(path):return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><path fill="black" d="'+g.svg_path(path)+'"/></svg>'

def family_build(task):
    source,fontroot,out,cap,limit=task;name=source['source_directory'];split=source['split'];file=fontroot/source['file']
    folder=out/name;folder.mkdir(parents=True,exist_ok=True);records=[];ledger=[];start=time.time()
    with TTFont(file) as font:
        cmap=font.getBestCmap();seen=set();chosen=[]
        lo,hi=(0x41,0x17f) if split=='test_latin' else (0x620,0x6ff)
        cps=[cp for cp in cmap if lo<=cp<=hi and unicodedata.category(chr(cp)).startswith('L')]
        for cp in sorted(cps,key=lambda c:g.sha(f'benchmark-v3:{c}'.encode())):
            gid=font.getGlyphID(cmap[cp])
            if gid and gid not in seen:seen.add(gid);chosen.append((cp,gid))
        chosen=chosen[:cap]
    base_count=0;shape_count=0
    for cp,gid in chosen:
        base_id=f'{name}:glyph{gid}';parent_used=False
        for mark in ([None] if split=='test_latin' else MARKS):
            instance=f'{base_id}:mark{mark}';rng=g.rng_for('benchmark-v3',instance)
            try:
                base,markpath,clean,components=latin_shape(file,cp) if mark is None else g.shape(file,cp,mark)
                ref=render(clean);baseim=render(base)
                if np.count_nonzero(baseim<128)<30:raise ValueError('Empty/tiny base')
                if mark is not None and np.count_nonzero(abs(ref.astype(int)-baseim.astype(int))>1)<3:raise ValueError('Combining mark not observable')
                fields,tokens,cids=reference_fields(clean,ref);_,_,points=g.contour_tokens(base,32)
                roundtrip=pathops.Path();clean.draw(roundtrip.getPen())
                if not np.array_equal(ref,render(roundtrip)):raise ValueError('Reference serialization is not raster-equivalent')
            except Exception as exc:
                ledger.append({'parent_id':base_id,'instance':instance,'status':'shape_rejected','reason':str(exc)});continue
            shape_count+=1;parent_used=True;shape_id=g.sha(instance.encode())[:24]
            reference_file=folder/(shape_id+'_reference.npz')
            np.savez_compressed(reference_file,image=ref,geometry=fields,tokens=tokens,contour_ids=np.asarray(cids,np.int16))
            (folder/(shape_id+'_reference.svg')).write_text(svg_text(clean))
            profiles=g.profiles(instance)
            variants=[('clean',0,pathops.Path(clean),{},0.)]
            # Calibration contains only independently generated benign observations.
            if split!='calibration':
                for kind in (TYPES[:3] if mark is None else TYPES):
                    severities=[0.] if kind=='missing_mark' else [.15,.5,.9]
                    for level,severity in enumerate(severities):
                        accepted=False
                        for attempt in range(16):
                            try:
                                candidate,edit=g.edited(base,markpath,kind,severity,rng,points)
                                candidate_image=render(candidate);delta=ref.astype(int)-candidate_image.astype(int)
                                area=int(np.count_nonzero(abs(delta)>1))
                                if area<2:raise ValueError('Geometric outcome not observable in canonical 256px render')
                                if area>max(1000,int((ref<128).sum()*.4)):raise ValueError('Intervention too extensive')
                                if np.count_nonzero(candidate_image<128)<30:raise ValueError('Whole-glyph deletion forbidden')
                                if kind=='spur' and np.count_nonzero(delta < -2)>3:raise ValueError('Protrusion removed ink')
                                if kind in ['notch','missing_mark'] and np.count_nonzero(delta>2)>3:raise ValueError('Deletion added ink')
                                variants.append((kind,level,candidate,edit,severity));accepted=True
                                ledger.append({'parent_id':base_id,'instance':instance,'type':kind,'level':level,'attempt':attempt,'status':'accepted','canonical_changed_pixels':area});break
                            except Exception as exc:ledger.append({'parent_id':base_id,'instance':instance,'type':kind,'level':level,'attempt':attempt,'status':'attempt_rejected','reason':str(exc)})
                        if not accepted:ledger.append({'parent_id':base_id,'instance':instance,'type':kind,'level':level,'status':'variant_unavailable'})
            for kind,level,candidate,edit,severity in variants:
                variant=g.sha(f'{instance}:{kind}:{level}'.encode())[:24]
                vector_file=folder/(variant+'_candidate.svg');vector_file.write_text(svg_text(candidate))
                selected=range(5) if kind=='clean' else [int(g.rng_for(instance,kind,level,'nuisance').integers(5))]
                for profile_index in selected:
                    nuisance=profiles[profile_index];kwargs={k:nuisance[k] for k in ['shift','aa','gamma']}
                    domains=[(split,'cairo',256)]
                    if split=='test_family':
                        domains.append(('test_renderer','skia',256))
                        # Independent resolution-shift evaluation on every held-out base.
                        if mark==MARKS[0]:domains.extend([('test_resolution128','cairo',128),('test_resolution512','cairo',512)])
                    for domain,renderer,size in domains:
                        query=render(candidate,size,renderer,**kwargs);counterfactual=render(clean,size,renderer,**kwargs)
                        target=(abs(query.astype(int)-counterfactual.astype(int))>1).astype(np.uint8)
                        sid=g.sha(f'v3:{variant}:{profile_index}:{domain}'.encode())[:28]
                        dest=folder/(sid+'.npz')
                        if size==256:np.savez_compressed(dest,query=query,mask=target)
                        else:np.savez_compressed(dest,query=query,mask=target,reference_at_size=render(clean,size,'cairo'))
                        record={'id':sid,'parent_id':base_id,'shape_id':shape_id,'family_group':source['family_group'],'source_family':name,
                            'split':domain,'declared_source_split':split,'file':name+'/'+dest.name,'reference_file':name+'/'+reference_file.name,
                            'candidate_svg':name+'/'+vector_file.name,'reference_svg':name+'/'+shape_id+'_reference.svg',
                            'label':int(kind!='clean'),'type':kind,'severity':severity,'severity_level':level,'unicode':f'U+{cp:04X}',
                            'mark_unicode':None if mark is None else f'U+{mark:04X}','renderer':renderer,'render_size':size,'nuisance':nuisance,
                            'mask_pixels':int(target.sum()),'observation_label':int(kind!='clean' and target.any()),
                            'file_sha256':sha(dest),'reference_sha256':sha(reference_file),'shape_components':components,'intervention':edit,
                            'source_font_sha256':source['sha256']}
                        records.append(record)
            if limit and shape_count>=limit:break
        base_count+=int(parent_used)
        if limit and shape_count>=limit:break
    rows_write(folder/'records.jsonl',records);rows_write(folder/'attempts.jsonl',ledger)
    summary={'source_family':name,'source_split':split,'base_glyphs':base_count,'marked_shapes':shape_count,'records':len(records),
        'observation_only':sum(r['label'] and not r['observation_label'] for r in records),'seconds':time.time()-start}
    dump(folder/'family_summary.json',summary);print('FAMILY_COMPLETE',json.dumps(summary),flush=True);return summary

def finalize(out,sources):
    records=[];ledgers=[];family=[]
    for s in sources['fonts']:
        d=out/s['source_directory'];records.extend(json.loads(l) for l in (d/'records.jsonl').read_text().splitlines());ledgers.extend(json.loads(l) for l in (d/'attempts.jsonl').read_text().splitlines());family.append(json.loads((d/'family_summary.json').read_text()))
    if len({r['id'] for r in records})!=len(records):raise ValueError('Duplicate sample ID')
    primary=[r for r in records if not r['label'] or r['observation_label']];only=[r for r in records if r['label'] and not r['observation_label']]
    for filename,values in [('meta.jsonl',primary),('all_records.jsonl',records),('observability_only.jsonl',only),('attempt_ledger.jsonl',ledgers)]:rows_write(out/filename,values)
    counts={split:{'records':sum(r['split']==split for r in primary),'parents':len({r['parent_id'] for r in primary if r['split']==split}),
                   'sources':sorted({r['source_family'] for r in primary if r['split']==split})} for split in sorted({r['split'] for r in primary})}
    dump(out/'FONT_SOURCES.json',sources)
    summary={'status':'build_complete_not_scientific_validation','families':family,'counts':counts,'generated_records':len(records),'primary_records':len(primary),'observability_only':len(only),
        'generated_by_domain':dict(collections.Counter(r['split'] for r in records)),'observation_only_by_domain_and_type':dict(collections.Counter(r['split']+':'+r['type'] for r in only)),
        'meta_sha256':sha(out/'meta.jsonl'),'all_records_sha256':sha(out/'all_records.jsonl'),'builder_sha256':sha(__file__),'helper_sha256':sha(g.__file__),
        'operator_types':TYPES,'mark_codes':MARKS,'geometry_scope':'reference-derived outline-distance/tangent features; no new-information claim over the trusted reference',
        'limitations':['synthetic interventions, not real font production defects','isolated base-plus-mark shaping, not full text-line QA','known ancestry grouping only','no novel-method superiority established']}
    dump(out/'BUILD_COMPLETE.json',summary)
    # Blind model-independent preview restricted to training sources; no test-image inspection.
    preview=[]
    for typ in ['clean']+TYPES:
        eligible=sorted([r for r in primary if r['split']=='train' and r['type']==typ],key=lambda r:g.sha(('preview:'+r['id']).encode()))
        preview.extend(eligible[:3])
    board=Image.new('RGB',(768,180*len(preview)),'white');draw=ImageDraw.Draw(board)
    for i,row in enumerate(preview):
        with np.load(out/row['reference_file']) as a:ref=a['image'].copy()
        with np.load(out/row['file']) as a:query=a['query'].copy();mask=a['mask'].copy()
        for j,im in enumerate([ref,query,mask*255]):board.paste(Image.fromarray(im).resize((160,160)),(j*256,i*180+18))
        draw.text((0,i*180),row['type']+' | '+row['source_family']+' | '+row['nuisance']['name'],fill='black')
    board.save(out/'training_preview.png');dump(out/'training_preview_ids.json',preview)
    print('BUILD_COMPLETE',json.dumps(counts),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fonts',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--base-cap',type=int,default=96);ap.add_argument('--workers',type=int,default=10);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if sha(g.__file__)!=HELPER_HASH:raise ValueError('Frozen geometry helper changed')
    sources=json.loads((a.fonts/'sources.json').read_text());a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'BUILD_COMPLETE.json').exists():raise FileExistsError('Completed corpus is immutable; use a new output directory')
    construction={'builder_sha256':sha(__file__),'helper_sha256':sha(g.__file__),'sources_sha256':sha(a.fonts/'sources.json'),'base_cap':a.base_cap,'smoke':a.smoke}
    state=a.out/'BUILD_CONFIG.json'
    if state.exists():
        if json.loads(state.read_text())!=construction:raise ValueError('Partial build source/configuration changed; use a new data directory')
    else:dump(state,construction)
    if a.smoke:sources={**sources,'fonts':[sources['fonts'][0],next(s for s in sources['fonts'] if s['split']=='test_family'),next(s for s in sources['fonts'] if s['split']=='test_latin')]}
    tasks=[(s,a.fonts,a.out,2 if a.smoke else a.base_cap,2 if a.smoke else 0) for s in sources['fonts']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as executor:list(executor.map(family_build,tasks))
    finalize(a.out,sources)
if __name__=='__main__':main()
