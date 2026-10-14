#!/usr/bin/env python3
"""Exercise pinned author APIs before long training. Uses training normals only.
These checks validate execution and checkpoint serialization, not performance.
"""
from pathlib import Path
import argparse, gc, json, tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import benchmark_common as b
import normal_ad as ad

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4);b.seed_all(81431)
    normals=b.BenchmarkData(args.root,'train',normal_only=True)
    if len(normals)<4:raise ValueError('Insufficient normal inputs for API checks')
    report={'scope':'implementation API preflight, not a baseline result','training_normal_ids':[normals.rows[i]['id'] for i in range(4)],'checks':[]}
    ad.vendor('patchcore')
    import patchcore.patchcore,patchcore.common,patchcore.sampler,patchcore.backbones
    backbone=patchcore.backbones.load('wideresnet50');backbone.name='wideresnet50';device=torch.device('cuda')
    model=patchcore.patchcore.PatchCore(device)
    model.load(backbone=backbone,layers_to_extract_from=['layer2','layer3'],device=device,input_shape=(3,256,256),pretrain_embed_dimension=1024,target_embed_dimension=1024,
               patchsize=3,anomaly_score_num_nn=1,featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(.01,device),nn_method=patchcore.common.FaissNN(False,4))
    rgb=b.NormalRGB(normals,256,list(range(4)))
    dl=DataLoader(ad.ImagesOnly(rgb),batch_size=2,num_workers=0)
    model.fit(dl)
    batch=torch.stack([rgb[i][0] for i in range(2)]).cuda()
    scores,masks=model.predict(batch)
    if len(scores)!=2 or np.asarray(masks).shape!=(2,256,256) or not np.isfinite(scores).all():raise AssertionError('PatchCore output invalid')
    with tempfile.TemporaryDirectory(prefix='patchcore_preflight_',dir=args.out.parent) as temp:
        model.save_to_path(temp)
        loaded=patchcore.patchcore.PatchCore(device)
        loaded.load_from_path(load_path=temp,device=device,nn_method=patchcore.common.FaissNN(False,4))
        scores2,masks2=loaded.predict(batch)
        if not np.allclose(scores,scores2,rtol=1e-5,atol=1e-6):raise AssertionError('PatchCore save/reload scores differ')
    report['checks'].append({'method':'patchcore','fit_predict_reload':'PASS','author_commit':ad.PINS['patchcore'],'preflight_normal_images':4})
    del model,loaded,backbone,batch;gc.collect();torch.cuda.empty_cache()
    model,trainable=ad.make_dinomaly()
    from utils import global_cosine_hm_percent,cal_anomaly_maps,WarmCosineScheduler
    from optimizers import StableAdamW
    rgb=b.NormalRGB(normals,392,[0,1]);batch=torch.stack([rgb[i][0] for i in range(2)]).cuda()
    optimizer=StableAdamW([{'params':trainable.parameters()}],lr=.002,betas=(.9,.999),weight_decay=.0001,amsgrad=True,eps=1e-10)
    scheduler=WarmCosineScheduler(optimizer,base_value=.002,final_value=.0002,total_iters=10000,warmup_iters=100)
    model.train();en,de=model(batch);loss=global_cosine_hm_percent(en,de,p=.0,factor=.1)
    if not torch.isfinite(loss):raise AssertionError('Dinomaly loss is not finite')
    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(trainable.parameters(),.1,error_if_nonfinite=True);optimizer.step();scheduler.step()
    model.eval()
    with torch.no_grad():
        en,de=model(batch);maps,_=cal_anomaly_maps(en,de,out_size=392)
    if maps.shape!=(2,1,392,392) or not torch.isfinite(maps).all():raise AssertionError('Dinomaly map shape/finiteness failed')
    report['checks'].append({'method':'dinomaly','forward_backward_optimizer_and_map':'PASS','author_commit':ad.PINS['dinomaly'],'preflight_optimizer_steps':1})
    report['status']='PASS';report['source_sha256']=b.sha(__file__);b.dump(args.out,report)
    print('NORMAL_AD_IMPLEMENTATION_PREFLIGHT_PASS',json.dumps(report),flush=True)

if __name__=='__main__':main()
