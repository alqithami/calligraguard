#!/usr/bin/env python3
"""Adapters around pinned author implementations, using normal training images only.
Input adaptations and scoring are recorded; no defect-type output is invented.
"""
from __future__ import annotations
import argparse, functools, gzip, json, os, random, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from scipy.ndimage import gaussian_filter
import benchmark_common as b

PINS={'patchcore':'fcaa92f124fb1ad74a7acf56726decd4b27cbcad','dinomaly':'1f252be03a918789b19848f0ca37166e7c28dada'}

def vendor(name):
    path=Path(__file__).resolve().parent/'vendor'/name
    actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if actual!=PINS[name]:raise ValueError('Author implementation revision changed')
    if subprocess.check_output(['git','-C',str(path),'diff','--name-only'],text=True).strip():raise ValueError('Tracked author source modified')
    sys.path.insert(0,str(path/'src' if name=='patchcore' else path));return path

class ImagesOnly(Dataset):
    def __init__(self,data):self.data=data
    def __len__(self):return len(self.data)
    def __getitem__(self,i):return self.data[i][0]

def make_dinomaly():
    vendor('dinomaly')
    from models.uad import ViTill
    from models import vit_encoder
    from models.vision_transformer import Block,bMlp,LinearAttention2
    from dinov1.utils import trunc_normal_
    encoder=vit_encoder.load('dinov2reg_vit_base_14')
    bottleneck=nn.ModuleList([bMlp(768,3072,768,drop=.2)])
    decoder=nn.ModuleList([Block(dim=768,num_heads=12,mlp_ratio=4.,qkv_bias=True,norm_layer=functools.partial(nn.LayerNorm,eps=1e-8),attn=LinearAttention2) for _ in range(8)])
    model=ViTill(encoder=encoder,bottleneck=bottleneck,decoder=decoder,target_layers=[2,3,4,5,6,7,8,9],mask_neighbor_size=0,
                fuse_layer_encoder=[[0,1,2,3],[4,5,6,7]],fuse_layer_decoder=[[0,1,2,3],[4,5,6,7]])
    trainable=nn.ModuleList([bottleneck,decoder])
    for m in trainable.modules():
        if isinstance(m,nn.Linear):
            trunc_normal_(m.weight,std=.01,a=-.03,b=.03)
            if m.bias is not None:nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.LayerNorm):nn.init.constant_(m.bias,0);nn.init.constant_(m.weight,1)
    return model.cuda(),trainable

def fit(args):
    out=args.out;out.mkdir(parents=True,exist_ok=True);b.seed_all(args.seed);torch.set_num_threads(8)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    vendor(args.method)
    data=b.BenchmarkData(args.root,'train',normal_only=True)
    config={'method':args.method,'seed':args.seed,'source_commit':PINS[args.method],'adapter_sha256':b.sha(__file__),'data_sha256':b.sha(args.root/'meta.jsonl'),
            'regime':'normal-only candidate RGB; no clean reference at inference','training_normal_count':len(data),'type_predictions':'not supported',
            'preprocessing':'grayscale expanded to RGB, ImageNet normalization; full-frame resize, no center crop'}
    if (out/'COMPLETE.json').exists():
        previous=json.loads((out/'config.json').read_text())
        if any(previous[k]!=v for k,v in config.items()):raise RuntimeError('Configuration changed for completed baseline')
        print('NORMAL_BASELINE_ALREADY_COMPLETE',args.method,args.seed,flush=True);return
    if args.method=='patchcore':
        import patchcore.patchcore,patchcore.backbones,patchcore.common,patchcore.sampler
        # Source-stratified, label-blind normal image budget, recorded exactly.
        by_source={}
        for i,r in enumerate(data.rows):by_source.setdefault(r['source_family'],[]).append(i)
        rng=np.random.default_rng(args.seed)
        for v in by_source.values():rng.shuffle(v)
        indices=[]
        while len(indices)<min(1024,len(data)):
            for name in sorted(by_source):
                if by_source[name] and len(indices)<1024:indices.append(by_source[name].pop())
        rgb=b.NormalRGB(data,256,indices);trainloader=DataLoader(ImagesOnly(rgb),batch_size=16,num_workers=8,pin_memory=True)
        backbone=patchcore.backbones.load('wideresnet50');backbone.name='wideresnet50';device=torch.device('cuda')
        model=patchcore.patchcore.PatchCore(device)
        model.load(backbone=backbone,layers_to_extract_from=['layer2','layer3'],device=device,input_shape=(3,256,256),pretrain_embed_dimension=1024,target_embed_dimension=1024,
                   patchsize=3,anomaly_score_num_nn=1,featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(.01,device),nn_method=patchcore.common.FaissNN(False,8))
        config.update({'normal_image_budget':len(indices),'coreset_ratio':.01,'feature_layers':['layer2','layer3'],'backbone':'wide_resnet50_2','search':'official CPU FAISS, eight threads',
                       'normal_training_ids':[data.rows[i]['id'] for i in indices]})
        b.dump(out/'config.json',config);start=time.time();model.fit(trainloader);(out/'model').mkdir(exist_ok=True);model.save_to_path(str(out/'model'))
        files={str(p.relative_to(out)):b.sha(p) for p in (out/'model').rglob('*') if p.is_file()}
        b.dump(out/'COMPLETE.json',{'status':'complete','method':args.method,'seconds':time.time()-start,'saved_files_sha256':files,'locked_test_evaluated':False})
        return
    model,trainable=make_dinomaly()
    from optimizers import StableAdamW
    from utils import global_cosine_hm_percent,WarmCosineScheduler
    config.update({'total_iterations':10000,'batch_size':16,'input_size':392,'pretrained_encoder':'dinov2reg_vit_base_14',
                   'optimizer':'author StableAdamW','learning_rate':.002,'final_lr':.0002,'warmup_iterations':100,
                   'initial_trainable_sha256':b.state_hash(trainable),'trainable_parameters':sum(p.numel() for p in trainable.parameters()),
                   'inference_map':'author cal_anomaly_maps, Gaussian sigma=4; image maximum after smoothing','selection':'last fixed iteration, no defective labels'})
    optimizer=StableAdamW([{'params':trainable.parameters()}],lr=2e-3,betas=(.9,.999),weight_decay=1e-4,amsgrad=True,eps=1e-10)
    scheduler=WarmCosineScheduler(optimizer,base_value=2e-3,final_value=2e-4,total_iters=10000,warmup_iters=100)
    rgb=b.NormalRGB(data,392);generator=torch.Generator().manual_seed(args.seed)
    loader=DataLoader(ImagesOnly(rgb),batch_size=16,shuffle=True,drop_last=True,num_workers=8,persistent_workers=True,pin_memory=True,generator=generator)
    if not len(loader):raise RuntimeError('Insufficient normal training data')
    step=0;start=time.time();epoch_index=0;resume_batch=0;elapsed_before=0.
    if (out/'config.json').exists():
        old_config=json.loads((out/'config.json').read_text())
        if old_config!=config:raise RuntimeError('Dinomaly resume configuration/source mismatch')
    else:b.dump(out/'config.json',config)
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        step=saved['iteration'];epoch_index=saved['epoch_index'];resume_batch=saved['batches_completed_in_epoch'];elapsed_before=saved['elapsed_seconds']
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        print('DINOMALY_RESUME_VERIFIED',step,epoch_index,resume_batch,flush=True)
    while step<10000:
        generator.manual_seed(args.seed+epoch_index*100003)
        model.train()
        for batch_index,x in enumerate(loader):
            if batch_index<resume_batch:continue
            x=x.cuda(non_blocking=True);en,de=model(x);loss=global_cosine_hm_percent(en,de,p=min(.9*step/1000,.9),factor=.1)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite Dinomaly loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(trainable.parameters(),.1,error_if_nonfinite=True);optimizer.step();scheduler.step();step+=1
            if step%100==0:
                status={'method':'dinomaly','seed':args.seed,'iteration':step,'iterations':10000,'loss':float(loss.detach()),'seconds':time.time()-start}
                b.dump(out/'PROGRESS.json',status);print('DINOMALY',json.dumps(status),flush=True)
            if step%1000==0 or step==10000:
                temp=out/'last.pt.tmp'
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'iteration':step,'epoch_index':epoch_index,
                            'batches_completed_in_epoch':batch_index+1,'elapsed_seconds':elapsed_before+time.time()-start,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'config':config},temp)
                temp.replace(out/'last.pt')
            if step>=10000:break
        epoch_index+=1;resume_batch=0
    temp=out/'best.pt.tmp';torch.save({'model':model.state_dict(),'iteration':step,'config':config},temp);temp.replace(out/'best.pt')
    b.dump(out/'COMPLETE.json',{'status':'complete','iterations':step,'checkpoint_sha256':b.sha(out/'best.pt'),'seconds':time.time()-start,'locked_test_evaluated':False})

@torch.no_grad()
def maps(model,method,dataset):
    size=256 if method=='patchcore' else 392
    rgb=b.NormalRGB(dataset,size);loader=DataLoader(rgb,batch_size=8,num_workers=8,pin_memory=True)
    for x,ids in loader:
        x=x.cuda(non_blocking=True)
        if method=='patchcore':scores,arrays=model.predict(x)
        else:
            from utils import cal_anomaly_maps
            en,de=model(x);amap,_=cal_anomaly_maps(en,de,out_size=size)
            if torch.is_tensor(amap):arrays=amap.detach().cpu().numpy()
            else:arrays=np.asarray(amap)
            arrays=np.asarray([gaussian_filter(a.squeeze(),sigma=4) for a in arrays]);scores=arrays.reshape(len(ids),-1).max(1)
        for sid,score,array in zip(ids,scores,arrays):
            if not np.isfinite(score) or not np.isfinite(array).all():raise ValueError('Nonfinite AD output')
            yield sid,float(score),np.asarray(array,dtype=np.float32).squeeze()

def evaluate(args):
    vendor(args.method);out=args.out
    if args.method=='patchcore':
        import patchcore.patchcore,patchcore.common
        model=patchcore.patchcore.PatchCore(torch.device('cuda'))
        model.load_from_path(load_path=str(out/'model'),device=torch.device('cuda'),nn_method=patchcore.common.FaissNN(False,8))
    else:
        model,_=make_dinomaly();model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=False)['model']);model.eval()
    cal=b.BenchmarkData(args.root,'calibration',normal_only=True);cal_scores={};normal_pixel_samples=[]
    with gzip.open(out/'calibration_predictions.jsonl.gz','wt') as f:
        for sid,score,array in maps(model,args.method,cal):
            cal_scores[sid]=score;normal_pixel_samples.append(array[::8,::8].ravel());f.write(json.dumps({'id':sid,'score':score})+'\n')
    mask_threshold=float(np.quantile(np.concatenate(normal_pixel_samples),.999));results={}
    for split in b.TEST_SPLITS:
        data=b.BenchmarkData(args.root,split,allow_test=True);lookup={r['id']:r for r in data.rows};pred={}
        with gzip.open(out/(split+'_predictions.jsonl.gz'),'wt') as f:
            for sid,score,array in maps(model,args.method,data):
                row=lookup[sid]
                with np.load(args.root/row['file'],allow_pickle=False) as z:gt=z['mask'].astype(bool)
                resized=F.interpolate(torch.from_numpy(array)[None,None],size=gt.shape,mode='bilinear',align_corners=False)[0,0].numpy();pm=resized>mask_threshold
                inter=int((pm & gt).sum());pa=int(pm.sum());ga=int(gt.sum());item={'id':sid,'score':score,'dice':2*inter/(pa+ga) if pa+ga else 1.,
                    'iou':inter/(pa+ga-inter) if pa+ga-inter else 1.,'predicted_mask_fraction':pa/pm.size,'mask_packbits_base64':b.pack_mask(pm),'mask_shape':list(pm.shape)}
                pred[sid]=item;f.write(json.dumps(item)+'\n')
        result=b.detection_metrics(cal.rows,[cal_scores[r['id']] for r in cal.rows],data.rows,[pred[r['id']]['score'] for r in data.rows])
        result['localization']=b.summarize_dense(data.rows,pred);result['localization']['mask_threshold']=mask_threshold
        result['localization']['threshold_definition']='99.9th percentile of stride-eight normal-calibration map pixels; not supervised threshold tuning'
        results[split]=result;b.dump(out/'LOCKED_RESULTS.json',results)
    b.dump(out/'EVALUATION_COMPLETE.json',{'status':'complete','results_sha256':b.sha(out/'LOCKED_RESULTS.json')})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['fit','evaluate']);ap.add_argument('--method',choices=['patchcore','dinomaly'],required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=1701);a=ap.parse_args()
    a.root=a.root.resolve();a.out=a.out.resolve()
    if a.action=='fit':fit(a)
    else:evaluate(a)
