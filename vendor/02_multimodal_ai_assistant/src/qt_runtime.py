"""Qt 运行环境修复：PyQt 安装路径丢失 Unicode 时改用手动路径。

Windows 下虚拟环境目录含中文时，Qt5 预置的插件路径可能出现字面 '?'，
导致 "No Qt platform plugin could be initialized"；本模块在创建
QApplication 之前把插件路径显式指向 PyQt5 安装目录下的真实路径。
"""
from __future__ import annotations

import os
from pathlib import Path


def configure_qt_plugins() -> Path:
    """定位并设置 Qt 平台插件路径；返回插件目录。"""
    import PyQt5
    from PyQt5.QtCore import QCoreApplication

    package_dir = Path(PyQt5.__file__).resolve().parent
    candidates = [package_dir / "Qt5" / "plugins", package_dir / "Qt" / "plugins"]
    plugin_dir = next((path for path in candidates if (path / "platforms").is_dir()), None)
    if plugin_dir is None:
        raise RuntimeError("PyQt5 平台插件缺失，请在当前 Python 环境重新安装 PyQt5-Qt5")
    # 必须在创建 QApplication 之前设置：
    # Do this before QApplication: Qt 5's default installation path can contain
    # literal '?' on Windows when the virtual environment lives in a CJK folder.
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugin_dir / "platforms")
    if os.name == "nt" and os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        # 离屏模式补一个系统字体目录，避免字体查找异常
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        if fonts.is_dir():
            os.environ.setdefault("QT_QPA_FONTDIR", str(fonts))
    QCoreApplication.setLibraryPaths([str(plugin_dir)])
    return plugin_dir
