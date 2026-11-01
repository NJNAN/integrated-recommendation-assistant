"""图像读写小工具：在 Windows 中文路径下也能可靠工作。

cv2.imread/imwrite 对非 ASCII 路径不可靠，这里统一走
「fromfile + imdecode」读 /「imencode + write_bytes」写的字节级通道。
"""
from __future__ import annotations

from pathlib import Path


def write_image(path: str | Path, image, quality: int = 92) -> Path:
    """把 OpenCV 图像写到路径（自动建目录；jpg/jpeg 可指定压缩质量）。"""
    import cv2

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower() or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if suffix in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise ValueError(f"图像编码失败: {suffix}")
    output.write_bytes(encoded.tobytes())
    return output


def read_image(path: str | Path):
    """从路径读取彩色图像（读不到直接报错，避免上层拿到 None）。"""
    import cv2
    import numpy as np

    data = np.fromfile(Path(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像: {path}")
    return image
