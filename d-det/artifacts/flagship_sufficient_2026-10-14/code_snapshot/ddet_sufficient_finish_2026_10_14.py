"""Finish this single local batch after its encoder process completes."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'d-det/artifacts/flagship_sufficient_2026-10-14'


def progress(status):
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    lines=['# 全量旗舰双任务：本机执行进度','',status,'',f'当前版本 {p["rows"]:,} 份代码、{p["task_count"]:,} 个任务；训练侧每轮使用 {sum(p["counts"]["train"].values()):,} 份。','',
           '| 任务 | 方法 | 已完成轮次 | 最佳校准分数 | 选中轮次 | 状态 |','|---|---|---:|---:|---:|---|']
    records={}
    for task in ['F','D']:
        for arm in ['B0','B1']:
            path=OUT/f'{task}_{arm}_training.json'
            if path.exists():
                try:fit=json.loads(path.read_text(encoding='utf-8'))
                except json.JSONDecodeError:continue
                records[(task,arm)]=fit;lines.append(f'| {"家族归因" if task=="F" else "检测"} | {"普通微调" if arm=="B0" else "AMMC微调"} | {len(fit["history"])} | {fit["best_calibration_metric"]:.4f} | {fit["selected_epoch"]} | '+('已停止' if fit['completed'] else '训练中')+' |')
            else:lines.append(f'| {task} | {arm} | 0 | 待测 | 待测 | 等待或首轮训练中 |')
    lines += ['', '只按校准集选择 checkpoint，未使用开发与泛化成绩选轮次。原始数据和旧实验保留。编码器完成后，本批次继续执行条件/普通/全局校正、强词法对照、泛化评估和归因空间可视化。','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    if records:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(10,6))
        for row,task in enumerate(['F','D']):
            for arm,color in [('B0','#6b7280'),('B1','#2563eb')]:
                if (task,arm) not in records:continue
                h=records[task,arm]['history'];ep=[r['epoch'] for r in h]
                axes[row,0].plot(ep,[r['weighted_train_loss'] for r in h],marker='o',label=arm,color=color);axes[row,1].plot(ep,[r['calibration_metric'] for r in h],marker='o',label=arm,color=color)
            axes[row,0].set_title(task+' training loss');axes[row,1].set_title(task+' calibration metric')
            for ax in axes[row]:
                ax.set_xlabel('Completed epoch');ax.grid(alpha=.2)
                if ax.lines:ax.legend()
        fig.tight_layout();fig.savefig(OUT/'training_progress.png',dpi=150);plt.close(fig)


def main():
    deadline=time.monotonic()+43200
    while time.monotonic()<deadline:
        progress('编码器正在按预定平台规则训练；下表只列已经完成的周期。')
        log=(OUT/'train.log.txt').read_text(encoding='utf-8',errors='replace') if (OUT/'train.log.txt').exists() else ''
        if 'SUFFICIENT_ENCODERS_AND_FEATURES_DONE' in log:break
        if 'Traceback (most recent call last)' in log[-10000:]:raise RuntimeError('encoder process failed; preserve logs and resume after repair')
        time.sleep(45)
    else:raise TimeoutError('batch monitor deadline reached')
    progress('编码器训练与特征提取完成；正在执行条件对照、评估与可视化。')
    with (OUT/'analysis.log.txt').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/ddet_sufficient_analysis_2026_10_14.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('analysis failed; inspect analysis.log.txt and rerun analysis only')
    with (OUT/'verification.log.txt').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/ddet_sufficient_verify_2026_10_14.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('independent verification failed')
    progress('本批次训练、评估、可视化和独立复算完成。最终结论见 conclusions.json 与结果文档。')
    for name in ['ddet_sufficient_finish_2026_10_14.py','ddet_sufficient_verify_2026_10_14.py']:
        import shutil
        shutil.copy2(ROOT/'scripts'/name,OUT/'code_snapshot'/name)
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt' and not p.name.endswith('_features.npz'))
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
        return h.hexdigest()
    (OUT/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(OUT).as_posix()}\n' for p in files),encoding='utf-8')
    print('FULL_BATCH_VERIFIED_DONE',flush=True)


if __name__=='__main__':main()
