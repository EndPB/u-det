#!/usr/bin/env python3
"""Local lexical shortcut decomposition for the H3 v2 data package."""
from __future__ import annotations
import json, re, hashlib, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "d-det/data/h3_stacad_revision_v1"
V2 = ROOT / "d-det/data/h3_stacad_revision_v2"
OUT = ROOT / "d-det/artifacts/h3_lexical_diagnostic_v2_2026-10-11"

def load(path):
    with path.open(encoding="utf8") as f: return [json.loads(x) for x in f if x.strip()]
def auc(y,s): return float(roc_auc_score(y,s)) if len(set(y))==2 else float("nan")
def tm(rows,s):
    g=defaultdict(list)
    for i,r in enumerate(rows): g[r["task_id"]].append(i)
    a=[]
    for idx in g.values():
        y=[rows[i]["label"] for i in idx]
        if len(set(y))==2: a.append(auc(y,s[idx]))
    return float(np.mean(a))
def strip_comments(s):
    s=re.sub(r"/\\*.*?\\*/", " ", s, flags=re.S)
    s=re.sub(r"//[^\\n]*", " ", s)
    s=re.sub(r"(^|[\\s;{}])#[^\\n]*", r"\\1 ", s)
    return s
def norm_ws(s): return re.sub(r"\\s+", " ", s).strip()
def mask_tokens(s):
    kw={"if","else","for","while","return","class","struct","public","private","static","void","int","float","double","char","new","try","catch","def","import","from","package","func","var","const","true","false","null","nil","switch","case","break","continue","this","self"}
    pat=re.compile(r"(?:\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*')|(?:\\b\\d+(?:\\.\\d+)?\\b)|(?:\\b[A-Za-z_][A-Za-z0-9_]*\\b)")
    def repl(m):
        x=m.group(0)
        if x.startswith(('"',"'")): return "<STR>"
        if x[0].isdigit(): return "<NUM>"
        return x if x in kw else "<ID>"
    return pat.sub(repl,s)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    train=[r for r in load(V1/'train_all.jsonl') if r.get('task_split')=='train']
    clean=[r for r in load(V1/'dev_clean.jsonl') if r.get('task_split')=='dev']
    ctrl=load(V2/'controlled_dev_v2.jsonl')
    views={"raw":lambda x:x,"no_comments":strip_comments,"normalized_ws":norm_ws,"masked_tokens":lambda x:mask_tokens(norm_ws(strip_comments(x)))}
    result={}
    for name,fn in views.items():
        print('fit',name,flush=True)
        vec=TfidfVectorizer(token_pattern=r"(?u)\\b\\w+\\b|[^\\w\\s]",ngram_range=(1,2),min_df=2,max_features=120000,sublinear_tf=True)
        xt=vec.fit_transform([fn(r['code']) for r in train])
        clf=LogisticRegression(C=2.0,max_iter=100,solver='liblinear',random_state=20261011).fit(xt,[r['label'] for r in train])
        one={}
        for label,rows in [('clean_dev',clean),('controlled_dev_v2',ctrl)]:
            s=clf.decision_function(vec.transform([fn(r['code']) for r in rows]))
            one[label]={"row_auroc":auc([r['label'] for r in rows],s),"task_macro_auroc":tm(rows,s),"rows":len(rows)}
        result[name]=one
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2)+"\n",encoding='utf8')
    (OUT/'config.json').write_text(json.dumps({"train":"v1/train_all","eval":["v1/dev_clean","v2/controlled_dev_v2"],"test_deserialized":False,"views":list(views)},indent=2)+"\n",encoding='utf8')
    report=['# H3 v2 词法捷径分解','', '四种 CPU 词法视图均在相同 train/dev task 口径下拟合；这不是 GPU 主实验。','']
    report.append('| view | clean-dev task-macro | controlled-dev task-macro |')
    report.append('|---|---:|---:|')
    for k,v in result.items(): report.append(f"| {k} | {v['clean_dev']['task_macro_auroc']:.4f} | {v['controlled_dev_v2']['task_macro_auroc']:.4f} |")
    report += ['', '若 masked_tokens 仍高，捷径主要来自标识符/字面量/结构组合；下一轮应建立同题、同语言、同长度且 project-heldout 的 hard-negative 配对，而不是继续堆叠模型。']
    (OUT/'report.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    sums=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS.txt':
            h=hashlib.sha256(p.read_bytes()).hexdigest(); sums.append(f'{h}  {p.relative_to(OUT).as_posix()}')
    (OUT/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n',encoding='utf8')
if __name__=='__main__': main()
