"""让旧版 MediaPipe 图/模型资源支持中文 Windows 路径。

MediaPipe Python 包在加载 C++ 图（``solution_base``）时会读取
``solution_base`` 模块对象的 ``__file__`` 目录，再去找
``modules/*.binarypb``、``.tflite``、``.pbtxt`` 等资源文件——它内部用
C 风格窄字符接口打开文件，在 Windows 上遇到含有非 ASCII 字符（如中文用
户名 ``C:\\Users\\张三\\...``）的安装路径就会报
``UnicodeDecodeError`` 或 ``OSError: [Errno 22]``。

本模块做的事情：
1. 探测当前 ``mediapipe`` 包的安装目录；如果是非 Windows 路径或本身就
   是纯 ASCII 路径，则直接 ``yield``（无需任何兼容，Linux / macOS 或纯英
   文用户名 Windows 都走这条路）；
2. 否则把整个 ``modules/`` 目录递归复制到
   ``%TEMP%\\mmai_mediapipe\\<sha256[:12]>``
   （或由环境变量 ``MMAI_MEDIAPIPE_CACHE`` 指定）的纯 ASCII 缓存目录；
3. 进入 ``with`` 块时，把 ``solution_base.__file__`` 临时改写到缓存里的
   同名文件，退出 ``with`` 时还原——这样 MediaPipe 在
   ``__enter__`` / ``__exit__`` 之间构造的所有 Solution 图都使用纯 ASCII
   缓存目录里的资源。

约束 / 注意事项：
- 缓存目录必须可写且路径中不能含中文，否则抛 ``RuntimeError``；
- **不修改任何已安装的 ``mediapipe`` 包文件**，仅修改模块对象的
  ``__file__`` 属性，进程退出后自动恢复；
- 进程内线程安全（``_RESOURCE_LOCK``）；
- 资源按 ``size`` + ``mtime_ns`` 双键判断是否需要重新复制。
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
from contextlib import contextmanager

LOGGER = logging.getLogger(__name__)
# 进程内全局锁：多个 HumanDetector 构造 / 析构并发时只让一个走 copy 阶段
_RESOURCE_LOCK = threading.RLock()


def _copy_resources(package_dir: Path, cache_root: Path) -> None:
    """把 ``package_dir/modules`` 下的模型文件复制到 ``cache_root``。

    只复制 MediaPipe 图真正会读取的后缀：``.binarypb`` / ``.tflite`` /
    ``.txt`` / ``.pbtxt``，避免把 ``__pycache__`` 等无关文件带过去。

    已存在且大小 + mtime 完全一致的目的文件会被跳过；否则用
    ``tempfile`` + ``shutil.copy2`` 做原子替换，避免半截文件被 native 端
    读到。
    """
    source_dir = package_dir / "modules"
    if not source_dir.is_dir():
        raise RuntimeError("MediaPipe 模型资源缺失，请安装 requirements.txt 中指定的版本")
    for source in source_dir.rglob("*"):
        # 只关心真正会被 C++ 图读到的文件后缀，其余跳过
        if not source.is_file() or source.suffix not in {".binarypb", ".tflite", ".txt", ".pbtxt"}:
            continue
        # 维持原始目录层级：cache_root/mediapipe/modules/...binarypb
        destination = cache_root / "mediapipe" / source.relative_to(package_dir)
        source_stat = source.stat()
        if destination.is_file():
            # size + mtime_ns 双重比对，避免出现 size 相同但内容被改过的情况
            dest_stat = destination.stat()
            if dest_stat.st_size == source_stat.st_size and dest_stat.st_mtime_ns == source_stat.st_mtime_ns:
                continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        # 用临时文件 + 原子替换，避免复制中途 native 端读到半截文件
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
            staging = Path(handle.name)
        try:
            shutil.copy2(source, staging)
            staging.replace(destination)
        finally:
            # 任何情况下都把残留临时文件清掉
            staging.unlink(missing_ok=True)


@contextmanager
def compatible_resources(mediapipe_module):
    """让 MediaPipe 在含中文路径的 Windows 上也能正常加载模型资源。

    用法（典型：在构造 MediaPipe Solution 图之前）::

        with compatible_resources(mp):
            pose = mp.solutions.pose.Pose(...)

    退出 ``with`` 块后 ``solution_base.__file__`` 会恢复成原值，因此同一
    进程里多次进出上下文是安全的。

    Args:
        mediapipe_module: 已 import 的 ``mediapipe`` 模块对象。

    Raises:
        RuntimeError: 缓存目录不可写 / 含中文 / 源资源缺失。
    """
    package_dir = Path(mediapipe_module.__file__).resolve().parent
    # 非 Windows 或安装路径本身已经是纯 ASCII → 不需要兼容，直接放行
    if os.name != "nt" or str(package_dir).isascii():
        yield
        return
    from mediapipe.python import solution_base

    # 用安装路径的 sha256 前 12 位作为缓存子目录，避免不同 Python 环境
    # 装不同 mediapipe 版本时互相覆盖
    key = hashlib.sha256(str(package_dir).encode("utf-8")).hexdigest()[:12]
    default_root = Path(tempfile.gettempdir()) / "mmai_mediapipe" / key
    # 允许通过环境变量把缓存指到任意盘符（注意：路径必须仍然是纯 ASCII）
    cache_root = Path(os.environ.get("MMAI_MEDIAPIPE_CACHE", str(default_root))).resolve()
    if not str(cache_root).isascii():
        # 缓存目录含中文会让问题回到原点，所以明确报错让用户换一个盘符
        raise RuntimeError("请将 MMAI_MEDIAPIPE_CACHE 设置为不含中文的可写目录")
    with _RESOURCE_LOCK:
        _copy_resources(package_dir, cache_root)
        # SolutionBase resets the native resource directory on every constructor,
        # deriving it from this module attribute. Redirect just that derivation
        # while constructing graphs; do not edit any installed package files.
        original_file = solution_base.__file__
        solution_base.__file__ = str(cache_root / "mediapipe" / "python" / "solution_base.py")
        try:
            LOGGER.info("MediaPipe 使用兼容资源目录: %s", cache_root)
            yield
        finally:
            # 无论 with 块内部是否抛异常，都把 __file__ 还原
            solution_base.__file__ = original_file