"""人体关键点检测模块。

本模块把 MediaPipe Solutions 的 ``Pose`` / ``Hands`` / ``Face`` 三个模型
封装进一个 ``HumanDetector``，是视觉链路里"读帧→推理→画图→时序稳定
判断"的唯一入口。向上游（视频采集、UI 渲染）输出 MediaPipe 推理结果和带
关键点的画面，向下游（手势识别、姿态稳定判断）提供帧序列上的历史极短
队列。

主要职责：
1. 在初始化时同时建立姿态、手部、面部三个 MediaPipe 解决方案实例；
2. 把 BGR 帧转成 RGB 后批量推理，并把最近若干帧的 pose 关键点保存到
   ``self.history``；
3. 提供 ``draw_landmarks`` 把三组关键点绘制到画面上；
4. 提供 ``is_full_body_visible`` / ``is_pose_stable`` 这样的判定工具。

被调用方：``scripts/08_test_detection.py``、主程序中的视觉管线；
依赖：``src/mediapipe_runtime.compatible_resources``（用于把 MediaPipe 模
型目录从含中文的安装路径临时重定向到可读的临时缓存，详见该模块注释）。
"""
from __future__ import annotations

import logging
from collections import deque
from contextlib import nullcontext
import threading

from .mediapipe_runtime import compatible_resources

# 模块级 logger，便于上层按 __name__ 过滤日志
LOGGER = logging.getLogger(__name__)


class HumanDetector:
    """姿态 + 手部 + 面部关键点检测器（线程安全）。

    三个 MediaPipe 解决方案都在同一个 ``HumanDetector`` 实例里完成推理
    （无跨模型共享状态），``process_frame`` 用 ``self._lock`` 串行化对底层
    native 图的访问，避免多线程同时提交帧导致内部状态错乱。

    Attributes:
        MAX_HISTORY: 时序稳定判断需要的最短历史长度（帧数）。
        cv2: opencv-python 模块（延迟导入，避免模块加载时强制依赖）。
        mp: mediapipe 模块（允许通过构造参数注入，便于单元测试）。
        _lock: 可重入锁，覆盖 ``__init__``（构造图）/ ``process_frame`` /
            ``close`` 三个会触碰 native 图的方法。
        history: 最近的 ``MAX_HISTORY`` 帧 pose 关键点的三维坐标队列，
            用于 ``is_pose_stable`` 计算平均位移。
    """

    MAX_HISTORY = 10

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        mediapipe_module=None,
    ):
        """构造三组 MediaPipe Solution 图。

        Args:
            model_complexity: Pose 模型复杂度档位，0=lite / 1=full / 2=heavy。
                默认 2 在桌面端 CPU 上仍可跑到 20+ FPS。
            min_detection_confidence: 检测阶段置信度阈值，0~1，默认 0.5 与
                MediaPipe 官方示例保持一致。
            min_tracking_confidence: 跟踪阶段置信度阈值，0~1，默认 0.5。
            mediapipe_module: 可选注入的 mediapipe 模块对象，便于单元测试；
                为 None 时本方法才会 ``import mediapipe``。
        """
        try:
            # 延迟导入 cv2 / mediapipe：让"只 import 本模块"的轻量调用方
            # （例如做静态类型检查）不会触发重量级 native 加载。
            import cv2
            if mediapipe_module is None:
                import mediapipe as mp
            else:
                mp = mediapipe_module
        except ImportError as exc:
            raise RuntimeError("缺少 opencv-python 或 mediapipe") from exc
        self.cv2 = cv2
        self.mp = mp
        # RLock 而非 Lock：__init__ 与 close 都可能与 process_frame 在同一
        # 线程上重入（例如线程池预热或析构流程），RLock 更安全。
        self._lock = threading.RLock()
        solutions = self.mp.solutions
        self.pose = self.hands = self.face = None
        try:
            # Preserve MediaPipe's optional lite/heavy model download, and cache
            # it before native graphs start looking up their resources.
            if mediapipe_module is None and model_complexity in {0, 2}:
                solutions.pose._download_oss_pose_landmark_model(model_complexity)
            # 当调用方注入了 mediapipe 模块对象时，没有"系统安装包"需要兼容，
            # 因此跳过 compatible_resources 上下文，用 nullcontext 占位。
            with (compatible_resources(self.mp) if mediapipe_module is None else nullcontext()):
                self.pose = solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=model_complexity,
                    smooth_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self.hands = solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self.face = solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=min_detection_confidence
                )
        except Exception:
            # 构造期任一步骤失败都要把已创建出来的 native 图关掉，避免
            # native 资源泄漏；再把异常原样抛出。
            self.close()
            raise
        self.history = deque(maxlen=self.MAX_HISTORY)

    def process_frame(self, bgr_image) -> dict:
        """对单帧 BGR 图做姿态 + 手部 + 面部联合推理。

        Args:
            bgr_image: OpenCV 默认格式的 BGR 图（H, W, 3 / uint8），
                不允许为 None。

        Returns:
            dict: ``{"pose": <PoseResult>, "hands": <HandsResult>,
            "face": <FaceDetectionResult>}``。当未检测到目标时，MediaPipe
            的对应字段会是 falsy 值（如空 NormalizedLandmarkList 或 None），
            调用方需要自行 ``getattr(..., None)`` 防御性取值。
        """
        if bgr_image is None:
            raise ValueError("输入图像不能为空")
        # OpenCV 默认 BGR，MediaPipe 要求 RGB，因此要转换一次
        rgb = self.cv2.cvtColor(bgr_image, self.cv2.COLOR_BGR2RGB)
        # MediaPipe 建议：把 buffer 标成 writeable=False，可以让它走更快的
        # 内部连续内存路径，并防止我们后面不小心写到推理输入。
        rgb.flags.writeable = False
        # 用锁把"三次 native 推理 + 一次历史记录"打包成原子操作，避免与
        # 其它线程在同一图上并发 process。
        with self._lock:
            pose_result = self.pose.process(rgb)
            hands_result = self.hands.process(rgb)
            face_result = self.face.process(rgb)
            pose_landmarks = getattr(pose_result, "pose_landmarks", None)
            if pose_landmarks:
                # 只缓存 pose 的历史；hands / face 不参与时序稳定判断。
                self._remember_pose(pose_landmarks)
        return {"pose": pose_result, "hands": hands_result, "face": face_result}

    def draw_landmarks(self, image, results: dict):
        """把姿态、手部、面部关键点画到画面上。

        Args:
            image: 原始 BGR 图。
            results: ``process_frame`` 返回的 dict。

        Returns:
            画好关键点的新 BGR 图（不会就地修改入参 ``image``）。
        """
        # 拷贝一份输出，避免在调用方的原图上原地画点
        output = image.copy()
        drawing = self.mp.solutions.drawing_utils
        styles = self.mp.solutions.drawing_styles
        pose = results.get("pose")
        hands = results.get("hands")
        face = results.get("face")
        # 用 truthy 判断而不是 is not None：
        # MediaPipe 在未检测到时返回的是 NormalizedLandmarkList()（falsy），
        # 这种对象同样可以直接走 hasattr 检查，但 truthy 更直观。
        if pose and pose.pose_landmarks:
            drawing.draw_landmarks(
                output,
                pose.pose_landmarks,
                self.mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=styles.get_default_pose_landmarks_style(),
            )
        if hands and hands.multi_hand_landmarks:
            for hand in hands.multi_hand_landmarks:
                drawing.draw_landmarks(
                    output,
                    hand,
                    self.mp.solutions.hands.HAND_CONNECTIONS,
                    styles.get_default_hand_landmarks_style(),
                    styles.get_default_hand_connections_style(),
                )
        if face and face.detections:
            for detection in face.detections:
                drawing.draw_detection(output, detection)
        return output

    @staticmethod
    def _landmarks(value):
        """兼容两种 landmarks 容器输入。

        MediaPipe 的 ``NormalizedLandmarkList`` 有 ``.landmark`` 属性；
        但测试或纯几何场景下调用方可能直接传 list/tuple，这里统一返回
        可迭代的 ``(x, y, z[, visibility])`` 关键点序列。
        """
        return value.landmark if hasattr(value, "landmark") else value

    def is_full_body_visible(self, pose_landmarks) -> bool:
        """判定上身 + 下身关键点是否同时满足"可见"阈值。

        取 11/12（左右肩）、23/24（左右髋）、25/26（左右膝）、
        27/28（左右踝）共 8 个点，任一缺失或 ``visibility < 0.5`` 都算
        "全身未完整可见"，返回 False。

        Args:
            pose_landmarks: MediaPipe 的 pose_landmarks 或等价可迭代序列。
        """
        if pose_landmarks is None:
            return False
        landmarks = self._landmarks(pose_landmarks)
        # 8 个关键点索引，按 MediaPipe Pose 拓扑约定；详见上方 docstring。
        required = (11, 12, 23, 24, 25, 26, 27, 28)
        return len(landmarks) > max(required) and all(
            getattr(landmarks[index], "visibility", 0.0) >= 0.5 for index in required
        )

    def _remember_pose(self, pose_landmarks) -> None:
        """把当前帧 pose 关键点快照追加到历史队列。"""
        landmarks = self._landmarks(pose_landmarks)
        # 存成 (x, y, z) tuple 列表，方便 numpy 直接堆叠成 (T, 21, 3)。
        self.history.append([(point.x, point.y, point.z) for point in landmarks])

    def is_pose_stable(self, pose_landmarks=None, threshold: float = 0.1) -> bool:
        """判定最近 ``MAX_HISTORY`` 帧的姿态是否整体平稳。

        思路：把历史里每一帧的所有关键点看作一组点云，求它们的质心，再
        计算每个关键点离质心距离的平均值；如果这个均值小于 ``threshold``
        （归一化坐标下），就认为"基本不动"，适合用作"开始录音""暂停"
        等触发条件。

        Args:
            pose_landmarks: 若提供则先把这一帧追加进历史再判断。
            threshold: 平均位移上限，归一化坐标单位，默认 0.1。
        """
        if pose_landmarks is not None:
            self._remember_pose(pose_landmarks)
        # 历史未填满 MAX_HISTORY 时直接判 False，避免冷启动阶段误触发。
        if len(self.history) < self.MAX_HISTORY:
            return False
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("缺少 numpy") from exc
        # (T, 21, 3)：T = MAX_HISTORY，21 个关键点，3 个坐标轴
        positions = np.asarray(self.history, dtype="float32")
        # 沿时间维求质心，得到 (1, 21, 3)
        center = positions.mean(axis=0, keepdims=True)
        # 每个关键点离质心的欧氏距离，沿 (T, 21) 求平均 -> 标量
        mean_displacement = np.linalg.norm(positions - center, axis=2).mean()
        return bool(mean_displacement < threshold)

    def close(self) -> None:
        """关闭底层 native 图，释放 native 资源。

        即使创建过程中途抛异常，外部 ``try/except`` 也可以再调一次以兜底。
        """
        with self._lock:
            for name in ("pose", "hands", "face"):
                model = getattr(self, name, None)
                # close 方法可能不存在（例如 mock 的 mediapipe 模块），
                # 用 getattr 防御性拿。
                close = getattr(model, "close", None)
                if close:
                    close()
                # 关掉之后把句柄置空，避免被析构时重复 close。
                setattr(self, name, None)