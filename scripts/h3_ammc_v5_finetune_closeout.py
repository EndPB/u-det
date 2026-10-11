#!/usr/bin/env python3
"""Close single-seed local encoder adaptation pilot, never a formal result."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import h3_v4_local_experiment as prior

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'d-det/artifacts/h3_ammc_v5_local_finetune_2026-10-11'

def main():
    m=json.loads((OUT/'metrics.json').read_bytes()); assert m['status']=='completed' and m['runs']==4
    comparisons=[];checks=0
    for f in ['fold_0','fold_1']:
        for view in ['clean','hardneg']:
            scores={}
            for arm in ['original','composed']:
                x=np.load(OUT/f'{f}_{arm}_20261011_{view}.npz',allow_pickle=False)
                assert np.isfinite(x['scores']).all()
                rows=[{'task_id':t,'label':int(y)} for t,y in zip(x['task_ids'],x['labels'])]
                got=prior.summary(rows,x['scores'],.5)
                claimed=next(r for r in m['runs_detail'] if r['fold']==f and r['arm']==arm)['eval'][view]
                for key in ['row_auroc','task_macro_auroc','balanced_accuracy']:
                    assert abs(got[key]-claimed[key])<1e-12;checks+=1
                scores[arm]=prior.task_aucs(rows,x['scores'])
            delta=prior.paired([scores['composed']],scores['original'],20261011)
            comparisons.append({'fold':f,'eval':view,**delta})
    close={'status':'exploratory_single_seed_feasibility_completed','saved_score_checks':checks,
        'comparisons_composed_minus_original':comparisons,'formal_gpu_ready':False,'gate':'revise_data',
        'scope':'last two blocks of encoder adapted; not full encoder fine tuning; detection only; one seed',
        'protocol_manifest_timing':'training config reconstructed after pilot; no claim of preregistration',
        'dropout_behavior':'encoder kept from_pretrained eval mode during adaptation; head dropout active; gradients update unfrozen weights',
        'corrections_before_full_pilot':['discard unstable FP16-parameter attempt','FP32 master weights + autocast + GradScaler + clipping',
            'filter dev AI to heldout generators before four-run pilot'],
        'checkpoint_saved':False,'dev_stress_test':False,'test_read':False,
        'decision':'feasible_local_encoder_adaptation; augmentation_gain_not_consistent'}
    prior.write(OUT/'closeout.json',close)
    shutil.copyfile(ROOT/'scripts/h3_ammc_v5_finetune_local.py',OUT/'executed_script.py')
    lines=['# H3 v5 本机编码器部分微调可行性实验','','单 seed，最后两层与 final norm 解冻；原始/组合增强两臂 × 两折 × 两 epoch，共四运行。',
        '这是探索性可行性试验；配置在运行后整理，不称预注册。test 未读取。',
        'FP32 master weights + autocast + GradScaler + clip norm 1.0；encoder dropout 关闭、head dropout 开启；两臂相同。',
        '', '| fold | view | original task AUROC | composed task AUROC | delta / task CI95 |','|---|---|---:|---:|---|']
    for c in comparisons:
        lo,hi=c['task_bootstrap_ci95']
        lines.append(f"| {c['fold']} | {c['eval']} | {c['control_task_macro_auroc']:.4f} | {c['mean_seed_task_macro_auroc']:.4f} | {c['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    lines+=['',f"保存分数复算 {checks}/{checks}；PyTorch peak allocated {m['peak_cuda_mib']:.1f} MiB，不是进程/驱动总显存。",
        '结论：本机可以训练该部分编码器；增强增益不一致，不能用一个 seed 支持或否定 AMMC/H3。',
        '任务内 AUROC 与 pooled row AUROC 分别报告；BA 较低，不以 task AUROC 替代全局检测表现。',
        '未保留 checkpoint、未做变换压力评估，也未使用 joint/source 目标。下一轮需先冻结配置、明确 encoder train/eval 行为、保留 checkpoint 后再运行成对多 seed 评测。']
    (OUT/'report.md').write_bytes(('\n'.join(lines)+'\n').encode())
    (OUT/'commands.txt').write_bytes(b'python -u scripts/h3_ammc_v5_finetune_local.py\npython scripts/h3_ammc_v5_finetune_closeout.py\n')
    (OUT/'SHA256SUMS.txt').write_bytes(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT).as_posix()}\n' for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.txt').encode())
    print(json.dumps(close,ensure_ascii=False))

if __name__=='__main__':main()
