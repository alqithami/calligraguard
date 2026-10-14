#!/usr/bin/env python3
"""CFDefect-v2a development corpus: native outline operations, explicit mark identity.
All final-test work is out of scope. Fonts remain in a private acquisition cache.
"""
from __future__ import annotations
import argparse, collections, concurrent.futures, hashlib, io, json, math, time, unicodedata
from pathlib import Path
import numpy as np
import cv2, cairosvg, pathops, uharfbuzz as hb
from PIL import Image, ImageDraw
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
from fontTools.pens.recordingPen import DecomposingRecordingPen, RecordingPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.svgPathPen import SVGPathPen

from observation_partition import partition_observability

TYPES=['spur','notch','local_deformation','missing_mark','misplaced_mark']

def sha(data): return hashlib.sha256(data).hexdigest()
def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def rng_for(*xs): return np.random.default_rng(int(sha(':'.join(map(str,xs)).encode())[:16],16))
def union(paths):
    result=pathops.Path()
    for p in paths: result=pathops.op(result,p,pathops.PathOp.UNION)
    return result

def svg_path(path):
    pen=SVGPathPen(None); path.draw(pen); return pen.getCommands()

def render(path,size=256,shift=(0.,0.),aa=1,gamma=1.):
    d=svg_path(path)
    text=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><g transform="translate({shift[0]:.8f} {shift[1]:.8f})"><path d="{d}" fill="black" fill-rule="nonzero"/></g></svg>'
    b=cairosvg.svg2png(bytestring=text.encode(),output_width=size*aa,output_height=size*aa,background_color='white')
    im=Image.open(io.BytesIO(b)).convert('L')
    if aa>1: im=im.resize((size,size),Image.Resampling.LANCZOS)
    a=np.array(im)
    if gamma!=1: a=np.rint(255*(a.astype(np.float64)/255)**gamma).clip(0,255).astype(np.uint8)
    return a

class OutlineSamples(BasePen):
    def __init__(self): super().__init__(None); self.contours=[]; self.current=[]
    def _moveTo(self,p):
        if self.current: self._endPath()
        self.current=[tuple(p)]
    def _lineTo(self,p): self.current.append(tuple(p))
    def _qCurveToOne(self,p1,p2):
        p0=np.array(self._getCurrentPoint()); p1=np.array(p1); p2=np.array(p2)
        for t in np.linspace(0,1,13)[1:]: self.current.append(tuple((1-t)**2*p0+2*(1-t)*t*p1+t*t*p2))
    def _curveToOne(self,p1,p2,p3):
        p0=np.array(self._getCurrentPoint()); p1=np.array(p1); p2=np.array(p2); p3=np.array(p3)
        for t in np.linspace(0,1,17)[1:]: self.current.append(tuple((1-t)**3*p0+3*(1-t)**2*t*p1+3*(1-t)*t*t*p2+t**3*p3))
    def _closePath(self):
        if self.current:
            if self.current[0]!=self.current[-1]: self.current.append(self.current[0])
            self.contours.append(np.asarray(self.current,dtype=np.float64)); self.current=[]
    def _endPath(self): self._closePath()

def contour_tokens(path,n=32):
    pen=OutlineSamples(); path.draw(pen)
    if pen.current: pen._closePath()
    candidates=[]
    for cid,c in enumerate(pen.contours):
        if len(c)<3: continue
        delta=np.diff(c,axis=0); lengths=np.linalg.norm(delta,axis=1); total=lengths.sum()
        if total<1e-4: continue
        cumulative=np.r_[0,np.cumsum(lengths)]
        for t in np.linspace(0,total,max(8,int(total/2)),endpoint=False):
            j=min(np.searchsorted(cumulative,t,side='right')-1,len(lengths)-1)
            tangent=delta[j]/max(lengths[j],1e-12); point=c[j]+tangent*(t-cumulative[j])
            candidates.append((point,tangent,cid,total,t/max(total,1e-9)))
    if not candidates: raise ValueError('No nondegenerate contours')
    # Deterministic spatial farthest-point sampling: only trusted reference geometry.
    points=np.asarray([x[0] for x in candidates]); chosen=[int(np.argmin(points[:,0]+points[:,1]))]
    distances=np.linalg.norm(points-points[chosen[0]],axis=1)
    while len(chosen)<n:
        j=int(np.argmax(distances)); chosen.append(j); distances=np.minimum(distances,np.linalg.norm(points-points[j],axis=1))
    features=[]; ids=[]
    for j in chosen:
        p,t,cid,length,s=candidates[j]; features.append([p[0]/128-1,p[1]/128-1,t[0],t[1],length/1024,s]); ids.append(cid)
    return np.asarray(features,np.float32),ids,candidates

def shape(font_path,cp,mark_cp):
    font=TTFont(font_path); location={a.axisTag:float(a.defaultValue) for a in font['fvar'].axes} if 'fvar' in font else {}
    if 'wght' in location: location['wght']=400.
    glyphset=font.getGlyphSet(location=location); order=font.getGlyphOrder()
    face=hb.Face(Path(font_path).read_bytes()); hf=hb.Font(face); hf.scale=(face.upem,face.upem); hb.ot_font_set_funcs(hf)
    if location: hf.set_variations(location)
    buf=hb.Buffer(); buf.add_str(chr(cp)+chr(mark_cp)); buf.direction='rtl'; buf.script='arab'; buf.language='ar'; buf.cluster_level=hb.BufferClusterLevel.CHARACTERS
    hb.shape(hf,buf)
    parts=[]; x=y=0
    for info,pos in zip(buf.glyph_infos,buf.glyph_positions):
        if info.codepoint==0: raise ValueError('Missing shaped glyph')
        p=pathops.Path(); rec=DecomposingRecordingPen(glyphset); glyphset[order[info.codepoint]].draw(rec)
        rec.replay(TransformPen(p.getPen(),(1,0,0,1,x+pos.x_offset,y+pos.y_offset)))
        parts.append({'path':p,'role':'mark' if info.cluster==1 else 'base','glyph_id':int(info.codepoint),'cluster':int(info.cluster)})
        x+=pos.x_advance; y+=pos.y_advance
    base=[r['path'] for r in parts if r['role']=='base']; marks=[r['path'] for r in parts if r['role']=='mark']
    if not base or not marks: raise ValueError('No separately identified mark cluster')
    b=union(base); m=union(marks); combined=union([b,m]); x0,y0,x1,y1=combined.bounds
    if x1-x0<1 or y1-y0<1: raise ValueError('Empty shaped outline')
    scale=200/max(x1-x0,y1-y0); dx=128-(x0+x1)*scale/2; dy=128+(y0+y1)*scale/2
    def trans(p):
        q=pathops.Path(); p.draw(TransformPen(q.getPen(),(scale,0,0,-scale,dx,dy))); return pathops.simplify(q)
    b,m=trans(b),trans(m); full=union([b,m]); font.close()
    return b,m,full,[{k:v for k,v in r.items() if k!='path'} for r in parts]

def triangle(points):
    p=pathops.Path(); p.moveTo(*points[0]); p.lineTo(*points[1]); p.lineTo(*points[2]); p.close(); return p

def edited(base,mark,kind,severity,rng,points):
    if kind=='missing_mark': return pathops.Path(base),{'component':'shaped combining mark','base_preserved':True}
    if kind=='misplaced_mark':
        angle=rng.uniform(0,2*np.pi); mag=2+5*severity; dx=math.cos(angle)*mag; dy=math.sin(angle)*mag
        m=pathops.Path(); mark.draw(TransformPen(m.getPen(),(1,0,0,1,dx,dy)))
        return union([base,m]),{'component':'shaped combining mark','base_preserved':True,'translation':[dx,dy]}
    p,t,cid,_,_=points[int(rng.integers(len(points)))]; normal=np.array([-t[1],t[0]])
    # Determine the outward normal using a local reference-only occupancy probe.
    occupancy=render(base,512)<128
    def inside(q):
        x,y=np.rint(q*2).astype(int)
        return bool(occupancy[np.clip(y,0,511),np.clip(x,0,511)])
    if inside(p+normal*1.5) and not inside(p-normal*1.5): normal=-normal
    if inside(p+normal*1.5)==inside(p-normal*1.5): raise ValueError('Ambiguous boundary normal')
    width=0.8+2.0*severity; depth=1.2+6*severity
    if kind=='spur':
        primitive=triangle([p-normal*.75+t*width,p-normal*.75-t*width,p+normal*depth])
        altered=pathops.op(base,primitive,pathops.PathOp.UNION)
    elif kind=='notch':
        primitive=triangle([p+normal*1.2+t*width,p+normal*1.2-t*width,p-normal*depth])
        altered=pathops.op(base,primitive,pathops.PathOp.DIFFERENCE)
    elif kind=='local_deformation':
        rec=RecordingPen(); base.draw(rec); warped=RecordingPen()
        direction=normal*(.5+2.5*severity); radius=6+6*severity
        for op,coords in rec.value:
            out=[]
            for q in coords:
                if q is None: out.append(None); continue
                z=np.array(q); z=z+direction*math.exp(-float(np.sum((z-p)**2))/(2*radius**2)); out.append(tuple(z))
            getattr(warped,op)(*out)
        altered=pathops.Path(); warped.replay(altered.getPen()); altered=pathops.simplify(altered)
    else: raise ValueError(kind)
    return union([altered,mark]),{'source_contour_id':cid,'source_point':p.tolist(),'outward_normal':normal.tolist(),'severity':severity,'mark_preserved':True}

def profiles(seed):
    r=rng_for(seed,'render_nuisance')
    return [dict(name='identity',shift=[0.,0.],aa=1,gamma=1.),
        dict(name='subpixel',shift=r.uniform(-.65,.65,2).tolist(),aa=1,gamma=1.),
        dict(name='supersample',shift=[0.,0.],aa=2,gamma=1.),
        dict(name='gamma',shift=[0.,0.],aa=1,gamma=float(r.uniform(.8,1.25))),
        dict(name='combined',shift=r.uniform(-.65,.65,2).tolist(),aa=2,gamma=float(r.uniform(.8,1.25)))]

def build_family(task):
    source,fontroot,out,parents=task; begin=time.time(); family=source['family_group']; split=source['split']; fontpath=fontroot/source['file']
    font=TTFont(fontpath); cmap=font.getBestCmap(); seen=set(); cps=[]
    candidates=[cp for cp in sorted(cmap) if 0x620<=cp<=0x6ff and unicodedata.category(chr(cp)).startswith('L')]
    # Seeded ordering independent of defect outcomes, recorded even for exclusions.
    candidates=sorted(candidates,key=lambda cp:sha(f'20261013:{cp}'.encode()))
    for cp in candidates:
        gid=font.getGlyphID(cmap[cp])
        if gid not in seen and gid!=0: seen.add(gid); cps.append(cp)
    font.close(); records=[]; ledger=[]; accepted_parents=0
    directory=out/family; directory.mkdir(parents=True,exist_ok=True)
    for cp in cps:
        if accepted_parents>=parents: break
        parent=f'{family}:U+{cp:04X}'; rnd=rng_for(parent); markcp=0x64e if int(rnd.integers(2))==0 else 0x650
        try:
            base,mark,clean,components=shape(fontpath,cp,markcp); ref256=render(clean,256); ref64=render(clean,64)
            if np.count_nonzero(render(base,256)<128)<30: raise ValueError('Empty/tiny base')
            markmask=np.abs(ref256.astype(int)-render(base,256).astype(int))>1
            if markmask.sum()<3: raise ValueError('Mark not separately visible at 256')
            geo,cids,_=contour_tokens(clean,32); _,_,points=contour_tokens(base,32)
            q=pathops.Path(); clean.draw(q.getPen())
            if not np.array_equal(ref256,render(q,256)): raise ValueError('Clean round-trip changes raster')
        except Exception as e:
            ledger.append({'parent':parent,'status':'parent_rejected','reason':str(e)}); continue
        ptag=sha(parent.encode())[:20]; (directory/(ptag+'_clean.svg')).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><path d="{svg_path(clean)}"/></svg>')
        nuisance=profiles(parent); variations=[('clean',0,clean,{})]
        for kind in TYPES:
            for level,severity in enumerate([.2] if kind=='missing_mark' else [.2,.8]):
                success=False
                for attempt in range(12):
                    try:
                        candidate,info=edited(base,mark,kind,severity,rnd,points)
                        image=render(candidate,256); difference=ref256.astype(int)-image.astype(int); area=int(np.count_nonzero(abs(difference)>1))
                        if area<2: raise ValueError('No observable 256px outcome')
                        if area>max(1000,int((ref256<128).sum()*.40)): raise ValueError('Intervention too extensive')
                        if np.count_nonzero(image<128)<30: raise ValueError('Whole glyph deleted')
                        if kind=='spur' and np.count_nonzero(difference < -2)>3: raise ValueError('Spur removed filled region')
                        if kind in ['notch','missing_mark'] and np.count_nonzero(difference > 2)>3: raise ValueError('Deletion added filled region')
                        variations.append((kind,level,candidate,info)); success=True
                        ledger.append({'parent':parent,'type':kind,'level':level,'attempt':attempt,'status':'accepted','changed_pixels256':area}); break
                    except Exception as e: ledger.append({'parent':parent,'type':kind,'level':level,'attempt':attempt,'status':'attempt_rejected','reason':str(e)})
                if not success: ledger.append({'parent':parent,'type':kind,'level':level,'status':'variant_unavailable'})
        # Two levels for mark removal are identical interventions; retain just one.
        variations=[v for v in variations if not(v[0]=='missing_mark' and v[1]==1)]
        perm=rnd.permutation(5).tolist(); counter=0
        for kind,level,candidate,info in variations:
            if kind=='clean': selected=list(range(5))
            else: selected=[perm[counter%5]]; counter+=1
            same256=render(candidate,256); same64=render(candidate,64)
            visible64=bool(np.any(same64!=ref64)); changed=int(np.count_nonzero(abs(same256.astype(int)-ref256.astype(int))>1))
            for profile_id in selected:
                pr=nuisance[profile_id]; kw={k:pr[k] for k in ['shift','aa','gamma']}
                query256=render(candidate,256,**kw); counterfactual256=render(clean,256,**kw); query64=render(candidate,64,**kw)
                mask=(np.abs(query256.astype(int)-counterfactual256.astype(int))>1).astype(np.uint8)
                sid=sha(f'{parent}:{kind}:{level}:{pr["name"]}:v2a'.encode())[:24]
                file=directory/(sid+'.npz')
                np.savez_compressed(file,query256=query256,reference256=ref256,query64=query64,reference64=ref64,mask256=mask,geometry=geo)
                records.append({'id':sid,'parent_id':parent,'family_group':family,'split':split,'file':family+'/'+file.name,
                    'label':int(kind!='clean'),'type':kind,'severity_level':level,'nuisance':pr,'visible64_without_nuisance':visible64,
                    'changed_pixels256_without_nuisance':changed,'mask_pixels256':int(mask.sum()),'file_sha256':sha(file.read_bytes()),
                    'mark_unicode':f'U+{markcp:04X}','shape_components':components,'contour_ids':cids,'intervention':info})
                if kind!='clean' and int(mask.sum())==0: records[-1]['observation_label256']=0
                else: records[-1]['observation_label256']=int(kind!='clean')
                if accepted_parents<2:
                    (directory/(sid+'.svg')).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><path d="{svg_path(candidate)}"/></svg>')
        accepted_parents+=1
    print('FAMILY_BUILT',family,'parents',accepted_parents,'records',len(records),'seconds',round(time.time()-begin,1),flush=True)
    return records,ledger,{'family':family,'split':split,'parents':accepted_parents,'candidate_letter_glyphs':len(cps)}

def review(out,rows):
    # Fixed IDs chosen by hash; no model outputs are involved.
    picks=[]
    for kind in ['clean']+TYPES:
        candidates=sorted([r for r in rows if r['type']==kind],key=lambda r:sha(('review:'+r['id']).encode()))
        picks.extend(candidates[:3])
    board=Image.new('RGB',(3*256, len(picks)*180),'white'); draw=ImageDraw.Draw(board)
    for i,r in enumerate(picks):
        d=np.load(out/r['file']); a=Image.fromarray(d['reference256']).resize((160,160)); b=Image.fromarray(d['query256']).resize((160,160)); c=Image.fromarray(d['mask256']*255).resize((160,160))
        for j,im in enumerate([a,b,c]): board.paste(im,(j*256,i*180+18))
        draw.text((0,i*180),f'{r["type"]} | {r["family_group"]} | {r["nuisance"]["name"]}',fill='black')
    board.save(out/'blind_review.png'); dump(out/'blind_review_ids.json',picks)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fonts',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--parents',type=int,default=48); ap.add_argument('--workers',type=int,default=8); ap.add_argument('--families',type=int,default=16); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False); begin=time.time(); sources=json.loads((a.fonts/'sources.json').read_text()); selected=sources['fonts'][:a.families]
    tasks=[(s,a.fonts,a.out,a.parents) for s in selected]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool: parts=list(pool.map(build_family,tasks))
    rows=[r for rs,_,_ in parts for r in rs]; ledger=[r for _,ls,_ in parts for r in ls]
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('Duplicate sample ID')
    all_rows = rows
    rows, observation_only = partition_observability(all_rows)
    # Keep original intervention labels and files; no resampling, relabeling, or threshold change.
    for r in observation_only:
        ledger.append({'id':r['id'],'parent':r['parent_id'],'type':r['type'],
                       'status':'observation_only','reason':'empty mask after declared nuisance rendering; intervention label preserved'})
    for name,values in [('meta.jsonl',rows),('all_records.jsonl',all_rows),('observability_only.jsonl',observation_only),('attempt_ledger.jsonl',ledger)]:
        with (a.out/name).open('w') as f:
            for r in values: f.write(json.dumps(r,allow_nan=False)+'\n')
    stats={'status':'generated_development_corpus_not_final_benchmark','families':[s for _,_,s in parts],'n':len(rows),'parent_count':len({r['parent_id'] for r in rows}),
        'split_counts':dict(collections.Counter(r['split'] for r in rows)),'type_counts':dict(collections.Counter(r['type'] for r in rows)),
        'empty_observed_positive_masks':sum(r['label'] and not r['mask_pixels256'] for r in rows),
        'invisible64_positive_count':sum(r['label'] and not r['visible64_without_nuisance'] for r in rows),
        'attempt_counts':dict(collections.Counter(r['status'] for r in ledger)),
        'normal_controls':'identity, subpixel translation, supersampling, gamma, combined; not human judgments of acceptable design changes',
        'operator_scope':'finite-area spur, local notch (not claimed full stroke gap), local outline deformation, identified combining-mark removal/displacement',
        'source_code_sha256':sha(Path(__file__).read_bytes()),'meta_sha256':sha((a.out/'meta.jsonl').read_bytes()),'build_seconds':time.time()-begin,
        'limits':['screen is development evaluation, not the final locked test','source-family ancestry expert review pending','no real-defect/expert validation yet']}
    stats.update({'corpus_version':'v2b_observation_partition', 'n_generated':len(all_rows),
        'n_observation_only':len(observation_only), 'observation_only_ids':[r['id'] for r in observation_only],
        'observation_only_by_split':dict(collections.Counter(r['split'] for r in observation_only)),
        'mask_rule_unchanged':'absolute uint8 query/counterfactual difference > 1',
        'primary_population':'all clean controls and verified 256px-observable positives',
        'all_records_sha256':sha((a.out/'all_records.jsonl').read_bytes()),
        'observability_only_sha256':sha((a.out/'observability_only.jsonl').read_bytes())})
    dump(a.out/'BUILD_SUMMARY.json',stats); dump(a.out/'FONT_SOURCES.json',sources); review(a.out,rows)
    print('BUILD_COMPLETE',json.dumps(stats),flush=True)
if __name__=='__main__': main()
