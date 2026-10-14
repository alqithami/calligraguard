#!/usr/bin/env python3
"""Full-resolution pretrained multi-task comparisons with isolated seeds and resumable fits.
All architectures use the same explicit detection, binary-mask and five-type objectives.
Reference-geometry conditioning is an experimental control, not a novelty assertion.
"""
from __future__ import annotations
import argparse, contextlib, gzip, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.metrics import roc_auc_score, f1_score
import benchmark_common as b

ARCHS=['unet34_ref','deeplab50_ref','segformer_b2_ref','unet34_geometry']

class MultiTask(nn.Module):
    def __init__(self,name,pretrained=True):
        super().__init__();self.name=name;self.geometry=name.endswith('_geometry')
        kwargs={'in_channels':4,'classes':1,'activation':None,'encoder_weights':'imagenet' if pretrained else None,
                'aux_params':{'classes':6,'pooling':'avg','dropout':.2,'activation':None}}
        if name.startswith('unet34'):self.net=smp.Unet(encoder_name='resnet34',**kwargs)
        elif name=='deeplab50_ref':self.net=smp.DeepLabV3Plus(encoder_name='resnet50',**kwargs)
        elif name=='segformer_b2_ref':self.net=smp.Segformer(encoder_name='mit_b2',**kwargs)
        else:raise ValueError(name)
        if self.geometry:
            old=self.net.encoder.conv1
            new=nn.Conv2d(8,old.out_channels,old.kernel_size,old.stride,old.padding,bias=old.bias is not None)
            with torch.no_grad():
                new.weight.zero_();new.weight[:,:4].copy_(old.weight)
                if old.bias is not None:new.bias.copy_(old.bias)
            self.net.encoder.conv1=new
            self.net.encoder._in_channels=8
    def forward(self,x,geometry):
        inputs=torch.cat([x,geometry],1) if self.geometry else x
        mask,aux=self.net(inputs)
        if mask.shape[-2:]!=x.shape[-2:]:mask=F.interpolate(mask,size=x.shape[-2:],mode='bilinear',align_corners=False)
        return mask,aux[:,0],aux[:,1:]

def objective(outputs,target,label,typ):
    mask,score,types=outputs
    det=F.binary_cross_entropy_with_logits(score.float(),label.float())
    seg=F.binary_cross_entropy_with_logits(mask.float(),target.float(),pos_weight=torch.tensor(20.,device=mask.device))
    prob=mask.float().sigmoid();inter=(prob*target).sum((1,2,3));den=prob.sum((1,2,3))+target.sum((1,2,3))
    positive=label>0
    dice=(1-(2*inter+1)/(den+1))[positive].mean() if positive.any() else mask.sum()*0
    typed=F.cross_entropy(types[positive].float(),typ[positive]) if positive.any() else types.sum()*0
    total=det+seg+dice+.5*typed
    return total,{'detection':float(det.detach()),'mask_bce':float(seg.detach()),'dice_loss':float(dice.detach()),'type_ce':float(typed.detach())}

def loader(dataset,batch,shuffle=False,seed=0):
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(dataset,batch_size=batch,shuffle=shuffle,num_workers=8,pin_memory=True,persistent_workers=True,
                      generator=generator,drop_last=shuffle)

@torch.no_grad()
def predict(model,dataloader,path=None,save_masks=False):
    model.eval();pred={};handle=gzip.open(path,'wt') if path else None
    positive_hist=np.zeros(4096,np.int64);negative_hist=np.zeros(4096,np.int64)
    start=time.time()
    try:
        for x,geometry,target,y,typ,ids in dataloader:
            x=x.cuda(non_blocking=True);geometry=geometry.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):m,s,t=model(x,geometry)
            probabilities=m.float().sigmoid().cpu().numpy()[:,0];scores=s.float().cpu().tolist();types=t.float().cpu().tolist();truth=target.numpy()[:,0]
            for j,sid in enumerate(ids):
                if sid in pred:raise ValueError('Duplicate predictions')
                binary=probabilities[j]>=.5;gt=truth[j]>0;intersection=int(np.count_nonzero(binary & gt));pa=int(binary.sum());ga=int(gt.sum())
                row={'id':sid,'score':float(scores[j]),'type_logits':types[j],
                     'dice':2*intersection/(pa+ga) if pa+ga else 1.,'iou':intersection/(pa+ga-intersection) if pa+ga-intersection else 1.,
                     'predicted_mask_fraction':pa/binary.size,'ground_truth_mask_pixels':ga,'height':binary.shape[0],'width':binary.shape[1]}
                if not np.isfinite(row['score']) or not np.isfinite(types[j]).all():raise RuntimeError('Nonfinite prediction')
                pred[sid]=row
                idx=np.minimum((probabilities[j]*4095).astype(np.int32),4095)
                positive_hist+=np.bincount(idx[gt],minlength=4096);negative_hist+=np.bincount(idx[~gt],minlength=4096)
                if handle:handle.write(json.dumps({**row,**({'mask_packbits_base64':b.pack_mask(binary)} if save_masks else {})})+'\n')
    finally:
        if handle:handle.close()
    report={'seconds_including_data_loading':time.time()-start,'records':len(pred),'pixel_score_histogram_bins':4096,
            'pixel_auroc_histogram_approximation':None}
    if positive_hist.sum() and negative_hist.sum():
        tpr=np.r_[0,np.cumsum(positive_hist[::-1])/positive_hist.sum()];fpr=np.r_[0,np.cumsum(negative_hist[::-1])/negative_hist.sum()]
        report['pixel_auroc_histogram_approximation']=float(np.trapz(tpr,fpr))
    if path:b.dump(Path(str(path)+'.pixel_histogram.json'),{'positive':positive_hist.tolist(),'negative':negative_hist.tolist(),**report})
    return pred,report

def dev_score(rows,pred):
    y=[r['label'] for r in rows];score=[pred[r['id']]['score'] for r in rows]
    auc=float(roc_auc_score(y,score));dense=b.summarize_dense(rows,pred)
    selection=.5*auc+.3*dense['mean_dice']+.2*dense['type_macro_f1']
    return selection,{'auroc':auc,**dense,'checkpoint_selection_score':selection}

def save_checkpoint(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.tmp');torch.save(value,tmp);tmp.replace(p)

def train(args):
    root=args.root.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((root.parent/'study_plan.json').read_text());config={'architecture':args.method,'seed':args.seed,'max_epochs':40,'min_epochs':20,'patience':8,
        'microbatch':16,'accumulation':2,'effective_batch':32,'training_drop_last':True,'drop_last_reason':'prevent singleton BatchNorm batch; shuffled final 0-15 records omitted per epoch','optimizer':'AdamW','lr':3e-4,'weight_decay':1e-4,'amp':'bf16',
        'selection':'.5 development AUROC + .3 positive Dice + .2 type macro-F1','mask_threshold':.5,'mask_positive_weight':20,
        'mask_dice_loss':'positive examples only','type_loss':'five-category CE on defective examples only, coefficient .5',
        'detection_loss':'BCE on all image labels','source_sha256':b.sha(__file__),'common_sha256':b.sha(b.__file__),
        'data_sha256':b.sha(root/'meta.jsonl'),'smp_version':smp.__version__,'torch_version':torch.__version__,
        'geometry_source':'trusted-reference outline features only' if args.method.endswith('geometry') else 'no explicit geometry input',
        'no_novelty_claim':True}
    if (out/'COMPLETE.json').exists():
        c=json.loads((out/'config.json').read_text())
        if any(c[k]!=config[k] for k in config):raise RuntimeError('Cannot reuse completed run under changed config')
        print('ALREADY_COMPLETE',out,flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required; no silent CPU fallback')
    b.seed_all(args.seed);torch.set_num_threads(8);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=MultiTask(args.method).cuda();config['parameter_count']=sum(p.numel() for p in model.parameters());config['initialization_sha256']=b.state_hash(model)
    config['encoder_initialization']='ImageNet pretrained weights shared by design; task-head initialization and minibatch order depend on seed'
    config['optimization_rng_seed']=args.seed+1000000
    b.seed_all(args.seed+1000000)
    training=b.BenchmarkData(root,'train');development=b.BenchmarkData(root,'development')
    trainloader=loader(training,16,True,args.seed);devloader=loader(development,16)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,40,eta_min=3e-6)
    best=-1.;best_epoch=0;start_epoch=1;steps=0;elapsed_before=0.;history=[]
    if (out/'last.pt').exists():
        old=json.loads((out/'config.json').read_text())
        if old!=config:raise RuntimeError('Resume source/configuration changed')
        ckpt=torch.load(out/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(ckpt['state_dict']);optimizer.load_state_dict(ckpt['optimizer']);scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch=ckpt['epoch']+1;steps=ckpt['optimizer_steps'];best=ckpt['best_score'];best_epoch=ckpt['best_epoch'];elapsed_before=ckpt['elapsed_seconds']
        random.setstate(ckpt['python_rng']);np.random.set_state(ckpt['numpy_rng']);torch.set_rng_state(ckpt['torch_rng']);torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        trainloader.generator.set_state(ckpt['loader_rng']);history=b.readrows(out/'epochs.jsonl')
    else:
        b.dump(out/'config.json',config)
    start=time.time()
    for epoch in range(start_epoch,41):
        model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.;examples=0;losses=collections_default();step_time=time.time()
        for i,(x,geo,mask,y,typ,ids) in enumerate(trainloader):
            x,geo,mask,y,typ=[t.cuda(non_blocking=True) for t in [x,geo,mask,y,typ]]
            with torch.autocast('cuda',dtype=torch.bfloat16):value,components=objective(model(x,geo),mask,y,typ)
            if not torch.isfinite(value):raise RuntimeError('Nonfinite training objective')
            # Final incomplete accumulation group uses its own number of microbatches.
            group_size=min(2,len(trainloader)-2*(i//2));(value/group_size).backward()
            if (i+1)%2==0 or i+1==len(trainloader):
                nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1
            examples+=len(ids);loss_sum+=float(value.detach())*len(ids)
            for k,v in components.items():losses[k]+=v*len(ids)
            if (i+1)%100==0:
                progress={'stage':'training','method':args.method,'seed':args.seed,'epoch':epoch,'microbatch':i+1,'microbatches_per_epoch':len(trainloader),'optimizer_steps':steps,'epoch_elapsed_seconds':time.time()-step_time}
                b.dump(out/'PROGRESS.json',progress);print('PROGRESS',json.dumps(progress),flush=True)
        scheduler.step();pred,timing=predict(model,devloader);selection,metrics=dev_score(development.rows,pred)
        elapsed=elapsed_before+time.time()-start
        record={'epoch':epoch,'train_loss':loss_sum/examples,'loss_components':{k:v/examples for k,v in losses.items()},'development':metrics,'optimizer_steps':steps,'elapsed_seconds':elapsed,'epoch_seconds':time.time()-step_time}
        if selection>best+1e-6:best=selection;best_epoch=epoch;save_checkpoint(out/'best.pt',{'state_dict':model.state_dict(),'epoch':epoch,'config':config,'development':metrics})
        checkpoint={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'optimizer_steps':steps,'best_score':best,'best_epoch':best_epoch,'elapsed_seconds':elapsed,
                    'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':trainloader.generator.get_state(),'config':config}
        save_checkpoint(out/'last.pt',checkpoint)
        with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        history.append(record);print('EPOCH_COMPLETE',json.dumps(record),flush=True)
        if epoch>=20 and epoch-best_epoch>=8:print('EARLY_STOP_AFTER_MINIMUM',epoch,'best_epoch',best_epoch,flush=True);break
    b.dump(out/'COMPLETE.json',{'status':'fit_complete','epochs':history[-1]['epoch'],'optimizer_steps':steps,'selected_epoch':best_epoch,'checkpoint_sha256':b.sha(out/'best.pt'),
        'config_sha256':b.sha(out/'config.json'),'history_sha256':b.sha(out/'epochs.jsonl'),'locked_test_evaluated':False,'elapsed_seconds':elapsed_before+time.time()-start})
    print('FIT_COMPLETE',args.method,args.seed,'steps',steps,flush=True)

def collections_default():
    from collections import defaultdict
    return defaultdict(float)

def evaluate(args):
    out=args.out;checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model=MultiTask(args.method,pretrained=False).cuda();model.load_state_dict(checkpoint['state_dict'])
    cal=b.BenchmarkData(args.root,'calibration',normal_only=True);calpred,_=predict(model,loader(cal,16),out/'calibration_predictions.jsonl.gz')
    results={}
    for split in b.TEST_SPLITS:
        ds=b.BenchmarkData(args.root,split,allow_test=True);batch=4 if split.endswith('512') else 16
        pred,timing=predict(model,loader(ds,batch),out/(split+'_predictions.jsonl.gz'),save_masks=True)
        result=b.detection_metrics(cal.rows,[calpred[r['id']]['score'] for r in cal.rows],ds.rows,[pred[r['id']]['score'] for r in ds.rows])
        result['dense_and_type']=b.summarize_dense(ds.rows,pred);result['timing_and_pixel_histogram']=timing
        result['per_source_family']={name:{'n':len(rr),'auroc':float(roc_auc_score([r['label'] for r in rr],[pred[r['id']]['score'] for r in rr]))} for name in sorted({r['source_family'] for r in ds.rows}) for rr in [[r for r in ds.rows if r['source_family']==name]]}
        results[split]=result;b.dump(out/'LOCKED_RESULTS.json',results)
    b.dump(out/'EVALUATION_COMPLETE.json',{'status':'complete','results_sha256':b.sha(out/'LOCKED_RESULTS.json'),'checkpoint_sha256':b.sha(out/'best.pt')})


def software_tests(out):
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    results=[]
    for name in ARCHS:
        with torch.random.fork_rng(devices=[0]):
            b.seed_all(81013);m=MultiTask(name).cuda();m.train();x=torch.rand(2,4,256,256,device='cuda');geo=torch.rand(2,4,256,256,device='cuda')
            mask=torch.zeros(2,1,256,256,device='cuda');mask[1,0,100:106,100:106]=1;y=torch.tensor([0.,1.],device='cuda');typ=torch.tensor([-1,0],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):outputs=m(x,geo);loss,parts=objective(outputs,mask,y,typ)
            loss.backward()
            if outputs[0].shape!=mask.shape or not torch.isfinite(loss):raise AssertionError('Invalid multitask forward/backward')
            if name.endswith('geometry') and not (m.net.encoder.conv1.weight.grad[:,4:].abs().sum()>0):raise AssertionError('Geometry conditioning is disconnected')
            results.append({'architecture':name,'parameters':sum(p.numel() for p in m.parameters()),'forward_backward':'PASS','loss_finite':True})
        del m;torch.cuda.empty_cache()
    b.dump(out,{'status':'PASS','models':results,'source_sha256':b.sha(__file__),'warning':'software tests are not performance results'})
    print('ALL_MULTITASK_SOFTWARE_TESTS_PASS',json.dumps(results),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['test','train','evaluate']);ap.add_argument('--root',type=Path);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--method',choices=ARCHS,default=ARCHS[0]);ap.add_argument('--seed',type=int,default=1701);a=ap.parse_args()
    if a.action=='test':software_tests(a.out)
    elif a.action=='train':train(a)
    else:evaluate(a)
