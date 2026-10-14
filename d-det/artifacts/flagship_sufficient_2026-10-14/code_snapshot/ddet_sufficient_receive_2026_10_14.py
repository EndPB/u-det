"""Receive this remote batch, then complete CPU/low-VRAM analysis locally."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];NAME='flagship_sufficient_2026-10-14';OUT=ROOT/'d-det/artifacts'/NAME;DATA=ROOT/'d-det/data'/NAME


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',required=True);parser.add_argument('--port',required=True);args=parser.parse_args()
    endpoint='root@'+args.host;remote='/root/autodl-tmp/u-det';runtime=remote+'/.codex-runs/'+NAME
    def ssh(command):
        result=subprocess.run(['ssh','-F','NUL','-o','BatchMode=yes','-o','ConnectTimeout=8','-p',args.port,endpoint,command],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
        if result.returncode:raise RuntimeError(result.stderr.strip())
        return result.stdout
    deadline=time.monotonic()+43200;failures=0
    while time.monotonic()<deadline:
        try:
            state=json.loads(ssh('/root/miniconda3/envs/udet/bin/python '+runtime+'/scripts/ddet_sufficient_remote_views_2026_10_14.py --status'));failures=0
            for key,fit in state['fits'].items():(OUT/(key+'_training.json')).write_text(json.dumps(fit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            (OUT/'remote_status_snapshot.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            from ddet_sufficient_finish_2026_10_14 import progress
            progress('AutoDL正在训练或编码；本机等待本批次完成后统一接收，再执行条件对照、绘图和复算。')
            if state['inventory_ready']:break
        except (RuntimeError,subprocess.TimeoutExpired,json.JSONDecodeError) as error:
            failures+=1;print('REMOTE_STATUS_RETRY',failures,str(error),flush=True)
            if failures>=12:raise
        time.sleep(45)
    else:raise TimeoutError('remote batch monitoring deadline reached')
    inventory=json.loads(ssh('cat '+remote+'/d-det/artifacts/'+NAME+'/remote_views_done.json'))
    (OUT/'remote_views_done.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    localbench=OUT/'attention_equivalence_and_benchmark.json'
    if localbench.exists() and not (OUT/'attention_equivalence_local.json').exists():(OUT/'attention_equivalence_local.json').write_bytes(localbench.read_bytes())
    def receive(entry):
        base=OUT if entry['kind']=='artifact' else DATA;target=base/entry['name'];assert target.resolve().is_relative_to(base.resolve())
        if target.exists() and sha(target)==entry['sha256']:return entry['name']
        temp=target.with_name(target.name+'.receiving')
        result=subprocess.run(['scp','-F','NUL','-o','BatchMode=yes','-o','ConnectTimeout=8','-P',args.port,endpoint+':'+entry['remote_path'],str(temp)],capture_output=True,text=True,timeout=3600)
        if result.returncode:raise RuntimeError(result.stderr)
        assert sha(temp)==entry['sha256'],entry['name'];temp.replace(target);return entry['name']
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(receive,entry) for entry in inventory['files']]
        for future in as_completed(futures):print('RECEIVED',future.result(),flush=True)
    (OUT/'remote_receive_receipt.json').write_text(json.dumps({'files_verified':len(inventory['files']),'all_transfers_sha256_verified':True,'data_sha256':json.loads((OUT/'protocol.json').read_text())['data_sha256']},indent=2)+'\n',encoding='utf-8')
    # Strong controls have already been fitted locally; all neural views now share one remote backend.
    with (OUT/'analysis.log.txt').open('w',encoding='utf-8') as logfile:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/ddet_sufficient_analysis_2026_10_14.py')],cwd=ROOT,stdout=logfile,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('local analysis failed; inspect analysis.log.txt')
    with (OUT/'unseen_families_analysis.log.txt').open('w',encoding='utf-8') as logfile:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/ddet_unseen_families_2026_10_14.py')],cwd=ROOT,stdout=logfile,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('family-heldout analysis failed')
    with (OUT/'verification.log.txt').open('w',encoding='utf-8') as logfile:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/ddet_sufficient_verify_2026_10_14.py')],cwd=ROOT,stdout=logfile,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('independent verification failed')
    progress('训练、条件对照、可视化、整家族留出检测与独立复算均已完成。')
    (OUT/'local_pipeline_done.json').write_text(json.dumps({'status':'complete','timestamp_local':time.strftime('%Y-%m-%d %H:%M:%S')},indent=2)+'\n',encoding='utf-8')
    files=sorted(path for path in OUT.rglob('*') if path.is_file() and path.name not in {'SHA256SUMS.txt','WORK_STATE.md'} and not path.name.endswith('_features.npz'))
    (OUT/'SHA256SUMS.txt').write_text(''.join(f'{sha(path)}  {path.relative_to(OUT).as_posix()}\n' for path in files),encoding='utf-8');print('LOCAL_FULL_BATCH_DONE',flush=True)


if __name__=='__main__':main()
