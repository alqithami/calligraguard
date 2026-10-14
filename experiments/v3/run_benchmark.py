#!/usr/bin/env python3
"""Bounded, resumable coordinator. Existing v2 data and experiments are read-only.
No test scoring occurs until every configured fit has completed and its checkpoint is frozen.
"""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, json, os, shutil, signal, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parent
PY=ROOT/'.venv/bin/python'
DATA=ROOT/'data_v3'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(name,obj):
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def idle():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if text:raise RuntimeError('GPU already occupied; existing processes will not be interrupted: '+text)
def disk_guard():
    free=shutil.disk_usage(ROOT).free
    if free<8*1024**3:raise RuntimeError('Less than 8 GiB free; safely paused before next stage')
def call(arguments,log,stage,timeout_hours=16):
    disk_guard();path=ROOT/log;path.parent.mkdir(parents=True,exist_ok=True)
    command=[str(x) for x in arguments]
    print('START',now(),stage,command,flush=True)
    with path.open('a',buffering=1) as f:
        p=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8'})
        dump('STATUS.json',{'status':'running','stage':stage,'child_pid':p.pid,'command':command,'log':log,'updated_utc':now()})
        try:rc=p.wait(timeout=timeout_hours*3600)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            raise RuntimeError('Stage exceeded bounded wall-time; saved epoch checkpoints retained: '+stage)
    if rc:raise RuntimeError(f'{stage} failed (exit {rc}); inspect {log}')
    print('END',now(),stage,flush=True)

def verify_sources(lock):
    for name,digest in lock['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Source changed after study lock: '+name)
    if sha(DATA/'meta.jsonl')!=lock['data_meta_sha256']:raise RuntimeError('Locked dataset manifest changed')
    for name,revision in lock['vendor_commits'].items():
        actual=subprocess.check_output(['git','-C',str(ROOT/'vendor'/name),'rev-parse','HEAD'],text=True).strip()
        if actual!=revision:raise RuntimeError('Vendor revision changed: '+name)
        if subprocess.check_output(['git','-C',str(ROOT/'vendor'/name),'diff','--name-only'],text=True).strip():raise RuntimeError('Vendor source modified: '+name)

def main():
    os.chdir(ROOT);lockfile=(ROOT/'PROCESS.lock').open('a')
    try:fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Benchmark coordinator is already running')
    if (ROOT/'STUDY_COMPLETE.json').exists():print('ALREADY_COMPLETE');return
    plan=json.loads((ROOT/'study_plan.json').read_text());idle();disk_guard();started=time.time()
    if (ROOT/'FAILED.json').exists():
        previous=json.loads((ROOT/'FAILED.json').read_text());stamp=str(int(time.time()));dump('previous_failures/'+stamp+'.json',previous);(ROOT/'FAILED.json').unlink()
    call([PY,'acquire_sources.py','--out','private_fonts'],'logs/source_acquisition.log','verify sources',1)
    if not (DATA/'BUILD_COMPLETE.json').exists():
        call([PY,'build_benchmark.py','--fonts','private_fonts','--out',DATA,'--base-cap',str(plan['data']['base_glyph_cap_per_source']),'--workers','10'],'logs/build.log','build expanded corpus',8)
    call([PY,'validate_and_test.py','--root',DATA,'--out',ROOT/'DATA_AND_EVALUATOR_TESTS.json'],'logs/validation.log','validate full corpus and evaluator',2)
    if not (ROOT/'MULTITASK_SOFTWARE_TESTS.json').exists():
        idle();call([PY,'train_multitask.py','test','--out',ROOT/'MULTITASK_SOFTWARE_TESTS.json'],'logs/model_tests.log','pretrained multi-task forward/backward tests',2)
    if not (ROOT/'NORMAL_AD_SOFTWARE_TESTS.json').exists():
        idle();call([PY,'preflight_normal_ad.py','--root',DATA,'--out',ROOT/'NORMAL_AD_SOFTWARE_TESTS.json'],'logs/normal_ad_tests.log','canonical anomaly-detection API checks',3)
    if not (ROOT/'EXPERIMENT_LOCK.json').exists():
        names=[p.name for p in ROOT.glob('*.py')]+['study_plan.json','source_plan.json','requirements.txt']
        revisions={item['method']:item['commit'] for item in plan['normal_only']}
        lock={'locked_utc':now(),'source_sha256':{name:sha(ROOT/name) for name in sorted(names)},'data_meta_sha256':sha(DATA/'meta.jsonl'),
              'all_records_sha256':sha(DATA/'all_records.jsonl'),'vendor_commits':revisions,'test_results_opened':False,
              'scope':'Source-defined synthetic family, renderer, resolution and script evaluations; real defects and expert validation not included'}
        dump('EXPERIMENT_LOCK.json',lock)
        (ROOT/'environment.freeze.txt').write_text(subprocess.check_output([str(PY),'-m','pip','freeze'],text=True))
        (ROOT/'gpu_at_launch.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    lock=json.loads((ROOT/'EXPERIMENT_LOCK.json').read_text());verify_sources(lock)
    fits=[]
    for item in plan['supervised']:
        for seed in item['seeds']:fits.append(('supervised',item['method'],seed))
    for item in plan['normal_only']:
        for seed in item['seeds']:fits.append(('normal_only',item['method'],seed))
    completed=[]
    for track,method,seed in fits:
        verify_sources(lock);run=ROOT/'runs'/method/f'seed_{seed}'
        if not (run/'COMPLETE.json').exists():
            idle()
            script='train_multitask.py' if track=='supervised' else 'normal_ad.py'
            action='train' if track=='supervised' else 'fit'
            call([PY,script,action,'--root',DATA,'--out',run,'--method',method,'--seed',str(seed)],f'logs/{method}_seed{seed}.log',f'{method} seed {seed}')
        done=json.loads((run/'COMPLETE.json').read_text())
        if 'checkpoint_sha256' in done and sha(run/'best.pt')!=done['checkpoint_sha256']:raise RuntimeError('Completed checkpoint checksum mismatch: '+str(run))
        if 'config_sha256' in done and sha(run/'config.json')!=done['config_sha256']:raise RuntimeError('Completed config checksum mismatch: '+str(run))
        if 'history_sha256' in done and sha(run/'epochs.jsonl')!=done['history_sha256']:raise RuntimeError('Completed history checksum mismatch: '+str(run))
        for relative,digest in done.get('saved_files_sha256',{}).items():
            if sha(run/relative)!=digest:raise RuntimeError('Saved baseline artifact checksum mismatch: '+relative)
        completed.append({'track':track,'method':method,'seed':seed,'completion':done})
        dump('COMPLETED_FITS.json',completed)
    registry={'created_utc':now(),'fits':completed,'source_lock_sha256':sha(ROOT/'EXPERIMENT_LOCK.json'),'checkpoint_files':{}}
    for track,method,seed in fits:
        run=ROOT/'runs'/method/f'seed_{seed}'
        if (run/'best.pt').exists():registry['checkpoint_files'][str((run/'best.pt').relative_to(ROOT))]=sha(run/'best.pt')
        if (run/'model').exists():
            for p in (run/'model').rglob('*'):
                if p.is_file():registry['checkpoint_files'][str(p.relative_to(ROOT))]=sha(p)
    if (ROOT/'MODEL_REGISTRY_LOCK.json').exists():
        old=json.loads((ROOT/'MODEL_REGISTRY_LOCK.json').read_text())
        if old['checkpoint_files']!=registry['checkpoint_files']:raise RuntimeError('Checkpoint changed after evaluation registry lock')
    else:dump('MODEL_REGISTRY_LOCK.json',registry)
    for track,method,seed in fits:
        verify_sources(lock);run=ROOT/'runs'/method/f'seed_{seed}'
        if not (run/'EVALUATION_COMPLETE.json').exists():
            idle();script='train_multitask.py' if track=='supervised' else 'normal_ad.py'
            call([PY,script,'evaluate','--root',DATA,'--out',run,'--method',method,'--seed',str(seed)],f'logs/evaluate_{method}_seed{seed}.log',f'locked evaluation {method} seed {seed}',8)
    if not (ROOT/'classical/COMPLETE.json').exists():
        call([PY,'classical_benchmarks.py','--root',DATA,'--out',ROOT/'classical'],'logs/classical.log','classical locked comparisons',8)
    summary={}
    for track,method,seed in fits:
        result=json.loads((ROOT/'runs'/method/f'seed_{seed}'/'LOCKED_RESULTS.json').read_text());summary.setdefault(method,[]).append({'seed':seed,'track':track,'results':result})
    dump('ALL_RESULTS.json',summary);verify_sources(lock)
    dump('STUDY_COMPLETE.json',{'status':'computational_benchmark_complete','completed_utc':now(),'fits':len(fits),'elapsed_this_session_seconds':time.time()-started,
        'all_results_sha256':sha(ROOT/'ALL_RESULTS.json'),'classical_results_sha256':sha(ROOT/'classical/RESULTS.json'),
        'final_paper_ready':False,'remaining':['real-defect validation','font-ancestry adjudication','methodological novelty beyond standard baselines','external human utility study']})
    dump('STATUS.json',{'status':'computational_benchmark_complete','fits_complete':len(fits),'updated_utc':now()})
    print('COMPUTATIONAL_BENCHMARK_COMPLETE',len(fits),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump('FAILED.json',{'status':'failed','updated_utc':now(),'error':str(exc),'traceback':traceback.format_exc()})
        dump('STATUS.json',{'status':'failed','error':str(exc),'updated_utc':now()})
        raise
