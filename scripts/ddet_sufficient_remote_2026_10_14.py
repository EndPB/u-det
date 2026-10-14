"""Explicit single-batch packaging and isolated AutoDL execution."""
from pathlib import Path
import argparse,hashlib,json,os,sys,zipfile

ROOT=Path(__file__).resolve().parents[1]
NAME='flagship_sufficient_2026-10-14'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def pack():
    data=ROOT/'d-det/data'/NAME;out=ROOT/'d-det/artifacts'/NAME
    paths={}
    for name in ['ddet_two_tasks_local_2026_10_14.py','ddet_flagship_local_2026_10_14.py','ddet_flagship_clean_local_2026_10_14.py','ddet_t5_local_fast_2026_10_14.py','ddet_sufficient_local_2026_10_14.py','ddet_sufficient_remote_2026_10_14.py']:
        paths['scripts/'+name]=ROOT/'scripts'/name
    for name in ['pilot.jsonl.gz','F_B0_resume.pt']:paths['d-det/data/'+NAME+'/'+name]=data/name
    for name in ['protocol.json','F_B0_training.json','attention_equivalence_and_benchmark.json','interpretation_plan.json','reserved_and_excluded.json']:
        paths['d-det/artifacts/'+NAME+'/'+name]=out/name
    pinned=ROOT/'d-det/artifacts/flagship_two_tasks_clean_2026-10-14'
    paths['d-det/artifacts/flagship_two_tasks_clean_2026-10-14/broad/protocol.json']=pinned/'broad/protocol.json'
    hashes={key:sha(path) for key,path in paths.items()};dest=data/'autodl_encoder_bundle.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        for key,path in paths.items():archive.write(path,key)
        archive.writestr('BUNDLE_MANIFEST.json',json.dumps(hashes,indent=2)+'\n')
    assert all(sha(path)==hashes[key] for key,path in paths.items()),'checkpoint changed during packaging; regenerate bundle'
    receipt={'file':str(dest),'bytes':dest.stat().st_size,'sha256':sha(dest),'files':len(paths),'model_sha256':sha(ROOT/'d-det/models/codet5-small/pytorch_model.bin')}
    (out/'upload_bundle_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt),flush=True)


def bootstrap(archive_path,repo_path):
    repo=Path(repo_path).resolve();runtime=repo/'.codex-runs'/NAME
    assert repo.is_dir();runtime.mkdir(parents=True,exist_ok=True)
    data=repo/'d-det/data'/NAME;out=repo/'d-det/artifacts'/NAME
    data.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    for link,target in [(runtime/'d-det/data'/NAME,data),(runtime/'d-det/artifacts'/NAME,out),(runtime/'d-det/models/codet5-small',repo/'d-det/models/codet5-small')]:
        link.parent.mkdir(parents=True,exist_ok=True)
        if link.exists():assert link.resolve()==target.resolve()
        else:link.symlink_to(target,target_is_directory=True)
    (runtime/'d-det/docx').mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        hashes=json.loads(archive.read('BUNDLE_MANIFEST.json'))
        for name in archive.namelist():
            path=(runtime/name).resolve();assert path.is_relative_to(repo),'archive escapes intended repository'
        marker=runtime/'JOB_IDENTITY.json'
        if marker.exists():assert json.loads(marker.read_text())['data_sha256']==hashes['d-det/data/'+NAME+'/pilot.jsonl.gz']
        archive.extractall(runtime)
    for name,digest in hashes.items():assert sha(runtime/name)==digest,name
    model=runtime/'d-det/models/codet5-small/pytorch_model.bin'
    assert sha(model)=='968fb0f45e1efc8cf3dd50012d1f82ad82098107cbadde2c0fdd8e61bac02908'
    marker.write_text(json.dumps({'data_sha256':hashes['d-det/data/'+NAME+'/pilot.jsonl.gz'],'runtime':str(runtime)},indent=2)+'\n')
    print(json.dumps({'runtime':str(runtime),'files_verified':len(hashes),'model_matches_local':True}),flush=True)


def run():
    import torch
    import transformers
    import ddet_sufficient_local_2026_10_14 as t
    t.setup();torch.set_num_threads(8)
    # Preserve the effective batch32 objective; only the micro-batch and inference batching change.
    t.MICRO=32
    original=t.encode
    def encode(*args,**kwargs):kwargs.setdefault('batch',64);return original(*args,**kwargs)
    t.encode=encode
    resume=t.DATA/'F_B0_resume.pt';best=t.DATA/'F_B0_best.pt'
    if not best.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        assert state['epoch']==state['selected'],'best checkpoint must be supplied separately if selection differs'
        torch.save({'encoder':state['encoder'],'head':state['head'],'epoch':state['selected']},best);del state
    (t.OUT/'remote_execution.json').write_text(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0),'micro_batch':32,'effective_batch':32,'inference_batch':64,'resume_checkpoint_sha256':sha(resume),'local_first_epoch_then_remote_resume':True,'runtime':str(ROOT)},indent=2)+'\n')
    # Smoke-check encoder output and gradient equivalence in this installed Transformers version.
    import ddet_t5_local_fast_2026_10_14 as fast
    fast.main()
    sys.argv=[sys.argv[0]];t.main()


def start(repo_path):
    import subprocess
    repo=Path(repo_path).resolve();runtime=repo/'.codex-runs'/NAME;out=repo/'d-det/artifacts'/NAME
    marker=out/'remote_pid.json'
    if marker.exists():
        prior=json.loads(marker.read_text());pid=prior['pid']
        if Path(f'/proc/{pid}/cmdline').exists() and b'ddet_sufficient_remote' in Path(f'/proc/{pid}/cmdline').read_bytes():
            print(json.dumps({'already_running':pid}),flush=True);return
    logfile=(out/'remote_train.log.txt').open('a',encoding='utf-8');env=os.environ.copy();env.update(HF_ENDPOINT='https://hf-mirror.com',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    child=subprocess.Popen([sys.executable,str(runtime/'scripts/ddet_sufficient_remote_2026_10_14.py'),'run'],cwd=runtime,stdout=logfile,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
    marker.write_text(json.dumps({'pid':child.pid,'runtime':str(runtime)},indent=2)+'\n');print(json.dumps({'started_pid':child.pid,'log':str(out/'remote_train.log.txt')}),flush=True)


def start_views(repo_path):
    import subprocess
    repo=Path(repo_path).resolve();runtime=repo/'.codex-runs'/NAME;out=repo/'d-det/artifacts'/NAME
    (runtime/'.gitignore').write_text('*\n')
    marker=out/'remote_views_pid.json'
    if marker.exists():
        pid=json.loads(marker.read_text())['pid'];cmd=Path(f'/proc/{pid}/cmdline')
        if cmd.exists() and b'ddet_sufficient_remote_views' in cmd.read_bytes():print(json.dumps({'already_running':pid}),flush=True);return
    logfile=(out/'remote_views.log.txt').open('a',encoding='utf-8');env=os.environ.copy();env.update(HF_ENDPOINT='https://hf-mirror.com',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    child=subprocess.Popen([sys.executable,str(runtime/'scripts/ddet_sufficient_remote_views_2026_10_14.py')],cwd=runtime,stdout=logfile,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
    marker.write_text(json.dumps({'pid':child.pid,'runtime':str(runtime)},indent=2)+'\n');print(json.dumps({'views_pid':child.pid}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['pack','bootstrap','run','start','start-views']);parser.add_argument('--archive');parser.add_argument('--repo',default='/root/autodl-tmp/u-det');args=parser.parse_args()
    if args.mode=='pack':pack()
    elif args.mode=='bootstrap':bootstrap(args.archive,args.repo)
    elif args.mode=='start':start(args.repo)
    elif args.mode=='start-views':start_views(args.repo)
    else:run()


if __name__=='__main__':main()
