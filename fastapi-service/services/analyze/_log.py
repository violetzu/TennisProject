# services/analyze/_log.py
"""
Thread-safe print tee：每個分析執行緒可獨立把 print() 輸出同步寫入 log 檔，
互不干擾（利用 threading.local 儲存 per-thread 的 file handle）。

使用方式：
    from ._log import set_log_file, clear_log_file

    with open(log_path, 'w', encoding='utf-8', buffering=1) as lf:
        set_log_file(lf)
        try:
            ...  # 所有 print() 自動 tee 到 lf
        finally:
            clear_log_file()
"""

import builtins
import threading

_log_local = threading.local()
_orig_print = builtins.print


def _patched_print(*args, **kwargs):
    lf = getattr(_log_local, "file", None)
    if lf:
        # 分析期間：只寫 log 檔，不輸出到 stdout（docker logs）
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        lf.write(sep.join(str(a) for a in args) + end)
        lf.flush()
    else:
        _orig_print(*args, **kwargs)


builtins.print = _patched_print           # 模組載入時一次性 patch


def set_log_file(f):
    _log_local.file = f


def clear_log_file():
    _log_local.file = None


def get_log_file():
    return getattr(_log_local, "file", None)
