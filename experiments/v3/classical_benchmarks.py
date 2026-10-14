#!/usr/bin/env python3
"""Explicit classical controls. Same candidate/reference access; no oracle nuisance parameters."""
from __future__ import annotations
import argparse, collections, gzip, json
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import minimize_scalar
import benchmark_common as b

NAMES=['raw_max','raw_mean','raw_top1pct','aligned_max','aligned_top1pct','aligned_gamma_max','aligned_gamma_top1pct','ssim_max','ssim_top1pct','boundary_distance_max','boundary_distance_top1pct','legacy_filtered_area']

def topmean(array):
    flat=array.ravel();n=max(1,int(np.ceil(.01*len(flat))));return float(np.partition(flat,-n)[-n:].mean())

def ssim_residual(q,r):
    q=q.astype(np.float64)/255;r=r.astype(np.float64)/255
    muq=cv2.GaussianBlur(q,(11,11),1.5);mur=cv2.GaussianBlur(r,(11,11),1.5)
    varq=cv2.GaussianBlur(q*q,(11,11),1.5)-muq*muq;varr=cv2.GaussianBlur(r*r,(11,11),1.5)-mur*mur;cov=cv2.GaussianBlur(q*r,(11,11),1.5)-muq*mur
    s=((2*muq*mur+.01**2)*(2*cov+.03**2))/((muq*muq+mur*mur+.01**2)*(varq+varr+.03**2))
    return np.clip((1-s)/2,0,1)

def maps(q,r):
    qf=q.astype(np.float64)/255;rf=r.astype(np.float64)/255;raw=abs(qf-rf)
    shift,response=cv2.phaseCorrelate((255-r).astype(np.float32),(255-q).astype(np.float32))
    dx,dy=shift
    if not np.isfinite([dx,dy]).all() or max(abs(dx),abs(dy))>4*q.shape[0]/256:dx=dy=0
    aligned=cv2.warpAffine(q,np.array([[1,0,-dx],[0,1,-dy]],np.float32),(q.shape[1],q.shape[0]),flags=cv2.INTER_LINEAR,borderValue=255).astype(np.float64)/255
    edge=((q>0)&(q<255))|((r>0)&(r<255))
    def fit_gamma(gamma):
        residual=abs(aligned[edge]**gamma-rf[edge]);n=max(1,int(.95*len(residual)))
        return float(np.partition(residual,n-1)[:n].mean()) if len(residual) else 0.
    gamma=float(minimize_scalar(fit_gamma,bounds=(.65,1.55),method='bounded',options={'maxiter':20}).x) if edge.any() else 1.
    aligned_delta=abs(aligned-rf);gamma_delta=abs(aligned**gamma-rf)
    structural=ssim_residual(q,r)
    qi=(q<128).astype(np.uint8);ri=(r<128).astype(np.uint8);kernel=np.ones((3,3),np.uint8)
    qe=qi-cv2.erode(qi,kernel);re=ri-cv2.erode(ri,kernel)
    d_q=cv2.distanceTransform(1-qe,cv2.DIST_L2,3);d_r=cv2.distanceTransform(1-re,cv2.DIST_L2,3)
    boundary=np.maximum(d_r*qe,d_q*re)/(q.shape[0]/256)
    legacy=(raw>=16/255).astype(np.uint8);legacy=cv2.morphologyEx(cv2.morphologyEx(legacy,cv2.MORPH_CLOSE,kernel),cv2.MORPH_OPEN,kernel)
    n,labels,stats,_=cv2.connectedComponentsWithStats(legacy,connectivity=8);keep=np.zeros(n,np.uint8)
    for i in range(1,n):keep[i]=stats[i,cv2.CC_STAT_AREA]>=max(10,int(.002*q.size))
    filtered=keep[labels]
    arrays={'raw_max':raw,'raw_mean':raw,'raw_top1pct':raw,'aligned_max':aligned_delta,'aligned_top1pct':aligned_delta,
            'aligned_gamma_max':gamma_delta,'aligned_gamma_top1pct':gamma_delta,'ssim_max':structural,'ssim_top1pct':structural,
            'boundary_distance_max':boundary,'boundary_distance_top1pct':boundary,'legacy_filtered_area':filtered.astype(np.float64)}
    scores={name:(float(a.mean()) if name in ['raw_mean','legacy_filtered_area'] else topmean(a) if 'top1pct' in name else float(a.max())) for name,a in arrays.items()}
    return scores,arrays

def load(root,row):
    with np.load(root/row['file'],allow_pickle=False) as z:q=z['query'].copy();gt=z['mask'].astype(bool);r=z['reference_at_size'].copy() if 'reference_at_size' in z else None
    if r is None:
        with np.load(root/row['reference_file'],allow_pickle=False) as z:r=z['image'].copy()
    return q,r,gt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=b.readrows(a.root/'meta.jsonl');cal=[r for r in rows if r['split']=='calibration'];cal_scores={};pixels=collections.defaultdict(list)
    for i,r in enumerate(cal):
        q,ref,gt=load(a.root,r);scores,arrays=maps(q,ref);cal_scores[r['id']]=scores
        for n in NAMES:pixels[n].append(arrays[n][::8,::8].ravel())
        if (i+1)%500==0:print('CLASSICAL_CALIBRATION',i+1,'/',len(cal),flush=True)
    thresholds={n:float(np.quantile(np.concatenate(pixels[n]),.999)) for n in NAMES}
    b.dump(a.out/'CALIBRATION.json',{'mask_thresholds':thresholds,'mask_threshold_definition':'99.9th percentile of stride-eight normal map pixels','scores':cal_scores})
    del pixels
    all_results={}
    for split in b.TEST_SPLITS:
        group=[r for r in rows if r['split']==split];pred={n:{} for n in NAMES}
        with gzip.open(a.out/(split+'_predictions.jsonl.gz'),'wt') as f:
            for i,r in enumerate(group):
                q,ref,gt=load(a.root,r);scores,arrays=maps(q,ref);payload={'id':r['id'],'methods':{}}
                for name in NAMES:
                    mask=arrays[name]>thresholds[name];inter=int((mask & gt).sum());pa=int(mask.sum());ga=int(gt.sum())
                    item={'id':r['id'],'score':scores[name],'dice':2*inter/(pa+ga) if pa+ga else 1.,'iou':inter/(pa+ga-inter) if pa+ga-inter else 1.,'predicted_mask_fraction':pa/mask.size}
                    pred[name][r['id']]=item;payload['methods'][name]=item
                f.write(json.dumps(payload)+'\n')
                if (i+1)%1000==0:print('CLASSICAL_EVALUATION',split,i+1,'/',len(group),flush=True)
        report={}
        for name in NAMES:
            result=b.detection_metrics(cal,[cal_scores[r['id']][name] for r in cal],group,[pred[name][r['id']]['score'] for r in group])
            result['localization']=b.summarize_dense(group,pred[name]);result['localization']['mask_threshold']=thresholds[name]
            report[name]=result
        all_results[split]=report;b.dump(a.out/'RESULTS.json',all_results)
    b.dump(a.out/'COMPLETE.json',{'status':'complete','configurations':len(NAMES),'results_sha256':b.sha(a.out/'RESULTS.json'),'source_sha256':b.sha(__file__)})
if __name__=='__main__':main()
