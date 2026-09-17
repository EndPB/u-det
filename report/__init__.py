"""报告模块：把代码统计渲染成短报告，注入到代码 token 之前（按名称切换）。

    from report import build_report

    report = build_report("handcrafted", tokenizer=tok, max_tokens=64)   # 手工统计（默认）
    report = build_report("none")                                        # 消融：不注入
    ids = report.ids(code)                                               # 报告 token

新增报告：在 report/ 下新建单文件，实现一个类（有 ``name`` 与 ``ids(code) -> list[int]``），
并在下面 REPORTS 字典里加一行映射。
"""

from .handcrafted import HandcraftedReport, NullReport

# 报告注册表：名称 -> 类
REPORTS = {
    "handcrafted": HandcraftedReport,
    "none": NullReport,
}


def list_reports():
    """列出所有可用报告名称。"""
    return sorted(REPORTS)


def build_report(name: str = "handcrafted", **kwargs):
    """按名称构建报告（cfg 里的 report.name 会传到这里）。"""
    if name not in REPORTS:
        raise KeyError(f"未知报告 {name!r}，可选：{list_reports()}")
    return REPORTS[name](**kwargs)
