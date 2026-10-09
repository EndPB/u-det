"""模式假设探针：参考 p0 组件是用 complete 模式还是 instruct 模式的文本计算的？

方法：用 r0 缓存的两种模式文本/嵌入，对同一 (member, task) 行分别按同一候选规格
（char tf-idf + SGD ensemble；语义 LR on CodeT5-small）拟合 fold-train，
再看哪个模式的 eval 分数与参考组件相关更高。
用法：OMP_NUM_THREADS=8 python scripts/cc_mode_probe.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cc_common as cc

R0 = cc.R0
REF = cc.ROOT / "d-det/artifacts/code_conditioned_p0_reference_2026-10-09/folds"
C0_LOCAL = cc.ROOT / "d-det/artifacts/code_conditioned_c0_strong_p0_2026-10-09/local"


def sgd_ens(Xtr, ytr, Xev, seeds=(0, 1, 2), epochs=5, bs=4096, alpha=1e-6):
    out = np.zeros(Xev.shape[0])
    for s in seeds:
        clf = SGDClassifier(loss="log_loss", alpha=alpha, random_state=s)
        rng = np.random.default_rng(s)
        for _ in range(epochs):
            perm = rng.permutation(Xtr.shape[0])
            for i in range(0, len(perm), bs):
                sel = perm[i:i + bs]
                clf.partial_fit(Xtr[sel], ytr[sel], classes=np.array([0, 1]))
        out += clf.decision_function(Xev)
    return out / len(seeds)


def main():
    design = cc.load_design()
    b = design["bundle"]
    idx = json.loads((R0 / "row_index.json").read_text(encoding="utf-8"))
    models_all, tasks_all = idx["models"], idx["tasks"]
    members = design["members_order"]
    gmidx = {m: models_all.index(m) for m in members}
    nT = len(tasks_all)
    tasks = design["tasks_all"]
    rows = design["rows"]
    n_rows = len(rows)

    # 需要两种模式的行号
    need = {}  # i -> key
    comp_i, ins_i = np.zeros(n_rows, dtype=np.int64), np.zeros(n_rows, dtype=np.int64)
    for r_i, r in enumerate(rows):
        gm = gmidx[r["model_id"]]
        ti = tasks.index(r["task_id"])
        ci = gm * nT * 2 + ti * 2 + 0      # complete
        ii = gm * nT * 2 + ti * 2 + 1      # instruct
        comp_i[r_i], ins_i[r_i] = ci, ii
        need[ci] = r_i
    need_set = set(need)
    print(f"need {len(need_set)} complete rows")

    # 读取 complete 文本
    texts_complete = [None] * n_rows
    with gzip.open(R0 / "texts.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            i = int(o["i"])
            if i in need_set:
                texts_complete[need[i]] = o["text"]
    miss = sum(1 for t in texts_complete if t is None)
    print("missing complete texts:", miss)
    print("sample check (complete != instruct):",
          texts_complete[0][:40].replace("\n", "\\n"), "|", rows[0]["code"][:40].replace("\n", "\\n"))

    # complete 嵌入
    emb = np.load(R0 / "emb_small.npz")["emb"]
    hy_comp = emb[comp_i].astype(np.float32)
    hy_ins = b["hy_small"].astype(np.float32)
    print("emb available:", hy_comp.shape)

    # 参考侧映射
    b_names = [members[i] for i in b["member_idx"]]
    t_names = [tasks_all[i] for i in b["task_idx"]]

    probes = ["codellama--CodeLlama-13b-Instruct-hf", "deepseek-ai--deepseek-coder-6.7b-instruct"]
    for fold in design["folds"]:
        h = fold["heldout_generator_member"]
        if h not in probes:
            continue
        fit_mask, ev_mask, pos_mask, y = cc.fold_setup(design, fold, split="dev")
        fit_rows = np.where(fit_mask)[0]
        ev_rows = np.where(ev_mask)[0]
        tp = np.array([tasks.index(rows[i]["task_id"]) for i in ev_rows])
        # ev 行按 taskpos 排序（与 C0 相同）
        dev_tasks = sorted({rows[i]["task_id"] for i in ev_rows})
        tpos_of = {t: i for i, t in enumerate(dev_tasks)}
        tp2 = np.array([tpos_of[rows[i]["task_id"]] for i in ev_rows])
        order = np.argsort(tp2, kind="mergesort")
        ev_rows_s = ev_rows[order]
        y_ev = pos_mask[ev_rows_s].astype(int)

        # 参考映射
        fam = design["member_series"][h]
        em = sorted(m for m in members if m == h or design["member_series"][m] != fam)
        taskrank = {t: i for i, t in enumerate(dev_tasks)}
        blockrank = {m: i for i, m in enumerate(em)}
        ref_idx = np.array([blockrank[b_names[i]] * 171 + taskrank[t_names[i]] for i in ev_rows_s])
        z_ref = np.load(REF / h / "p0_scores.npz")
        ref_char = z_ref["component_char"][ref_idx]
        ref_word = z_ref["component_word"][ref_idx]
        ref_lin = z_ref["component_linear"][ref_idx]

        y_fit = cc.fold_series_target(design, fold)[fit_mask]
        res = {}
        for mode, hy, texts in [("instruct", hy_ins, [rows[i]["code"] for i in range(n_rows)]),
                                ("complete", hy_comp, texts_complete)]:
            # 语义
            sc = StandardScaler().fit(hy[fit_rows])
            lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(hy[fit_rows]), y_fit)
            s_sem = lr.decision_function(sc.transform(hy[ev_rows_s]))
            # char / word
            tr_texts = [texts[i] for i in fit_rows]
            ev_texts = [texts[i] for i in ev_rows_s]
            vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                                 sublinear_tf=True, lowercase=False)
            Xc_tr = vc.fit_transform(tr_texts); Xc_ev = vc.transform(ev_texts)
            s_char = sgd_ens(Xc_tr, y_fit, Xc_ev)
            vw = TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
                                 ngram_range=(1, 3), min_df=1, sublinear_tf=True, lowercase=False)
            Xw_tr = vw.fit_transform(tr_texts); Xw_ev = vw.transform(ev_texts)
            s_word = sgd_ens(Xw_tr, y_fit, Xw_ev)
            res[mode] = {
                "sem_vs_ref_linear": float(np.corrcoef(s_sem, ref_lin)[0, 1]),
                "sem_auroc": cc.auroc(y_ev, s_sem),
                "char_vs_ref_char": float(np.corrcoef(s_char, ref_char)[0, 1]),
                "char_auroc": cc.auroc(y_ev, s_char),
                "word_vs_ref_word": float(np.corrcoef(s_word, ref_word)[0, 1]),
                "word_auroc": cc.auroc(y_ev, s_word),
            }
        print(f"== fold {h.split('--')[-1]}")
        for mode in ("instruct", "complete"):
            r = res[mode]
            print(f"  [{mode:8s}] sem:r={r['sem_vs_ref_linear']:+.3f} auroc={r['sem_auroc']:.4f} | "
                  f"char:r={r['char_vs_ref_char']:+.3f} auroc={r['char_auroc']:.4f} | "
                  f"word:r={r['word_vs_ref_word']:+.3f} auroc={r['word_auroc']:.4f}")
        # 参考组件自身的 AUROC（同一行序）
        print(f"  [REF     ] char_auroc={cc.auroc(y_ev, ref_char):.4f} word_auroc={cc.auroc(y_ev, ref_word):.4f} lin_auroc={cc.auroc(y_ev, ref_lin):.4f}")


if __name__ == "__main__":
    main()
