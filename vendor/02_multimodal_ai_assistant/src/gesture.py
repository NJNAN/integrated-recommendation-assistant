"""手势识别模块。

本模块基于 MediaPipe Hands 的 21 个归一化关键点，用纯几何判定（指节角度、
指尖距离、伸直/收拢状态）做六类手势分类：

- ``thumbs_up`` / ``thumbs_down``：四指收拢 + 拇指竖直 + 拇指尖明显伸出；
- ``open_palm``：四指伸直 + 拇指张开；
- ``victory``：食指 + 中指伸直并分开，无名指/小指收拢；
- ``index_up`` / ``index_down``：仅食指伸直，且指尖方向明确朝上 / 朝下。

为了让单次手势触发足够稳健（不会因为一两帧抖动就误识别），模块实现了
一个"候选 → 持续 → 锁定 → 释放"三段式状态机：

- **候选**：连续 ``stable_frames`` 帧、累计 ``stable_seconds`` 秒都给出同
  一个标签，才进入锁定阶段；
- **锁定**：触发一次后直到识别到完全放手（``release_seconds`` 秒内都没
  有任何手势输入）才会释放，期间不会再触发；
- **冷却**：所有手势共享一个 ``cooldown_seconds`` 全局冷却，避免刚触发
  完点赞就又触发了音量+。

依赖：上游传入 MediaPipe Hands 输出的 ``multi_hand_landmarks[0]``，并提
供帧宽高比 ``aspect_ratio`` 用于校正 x/y 轴量纲；下游通常交给
``GestureActionMapper`` 把标签映射到系统动作。
"""
from __future__ import annotations

import math
import time
import threading


class GestureRecognizer:
    """六类手势的状态机式识别器。

    Attributes:
        GESTURES: 支持识别的全部手势标签集合（与下游
            ``GestureActionMapper.action_map`` 键一致）。
        TIP_PIP: 各手指指尖与第二指节的 landmark 索引（保留字段，目前
            ``classify`` 未直接使用，留给后续"指尖方向"扩展）。
        FINGERS: 各手指从掌指关节（MCP）到指尖（TIP）的四节索引序列，
            用于 ``_finger_state`` 计算伸直 / 收拢 / 不确定。
    """

    GESTURES = [
        "thumbs_up",
        "thumbs_down",
        "open_palm",
        "index_up",
        "index_down",
        "victory",
    ]
    # (tip_index, pip_index)：指尖与第二指节，留作后续指尖方向扩展
    TIP_PIP = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}

    # (mcp, pip, dip, tip)：从掌指关节到指尖，依次对应 MediaPipe Hands 拓扑
    # 的 #5/#6/#7/#8 等组合（食指 5-8、中指 9-12、无名指 13-16、小指 17-20）
    FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
               "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}

    def __init__(
        self, cooldown_seconds: float = 2.0, clock=time.monotonic, *,
        stable_frames: int = 4, stable_seconds: float = 0.35,
        release_seconds: float = 0.6, max_frame_gap: float = 0.5,
    ):
        """构造手势识别器并初始化内部状态。

        Args:
            cooldown_seconds: 触发同一/不同动作后的最小间隔（全局）。
            clock: 单调时钟源（默认 ``time.monotonic``），便于测试时注入
                假时钟。
            stable_frames: 进入锁定所需的最小连续同标签帧数（最少 1）。
            stable_seconds: 累计稳定的最短持续秒数。
            release_seconds: 完全放手（``gesture = None``）持续多少秒才
                算"释放"，期间不会再次触发。
            max_frame_gap: 两帧之间允许的最大间隔；超过则视为新事件流，
                把已有候选/锁定状态作废。
        """
        self.cooldown = cooldown_seconds
        self.clock = clock
        self.stable_frames = max(1, stable_frames)
        self.stable_seconds = max(0.0, stable_seconds)
        self.release_seconds = max(0.0, release_seconds)
        self.max_frame_gap = max_frame_gap
        # 最近一次返回给调用方的标签，用于 UI / 日志
        self.last_gesture: str | None = None
        # 用 -inf 让首次触发不经过冷却期
        self.last_trigger_time = float("-inf")
        # 串行化"读状态 → 决策 → 写状态"全过程
        self._lock = threading.Lock()
        self._candidate = None
        self._candidate_since = 0.0
        self._candidate_count = 0
        self._last_frame_time = None
        self._release_since = None
        self._latched = None

    @staticmethod
    def _landmarks(value):
        """兼容 MediaPipe 的 ``NormalizedLandmarkList`` 与 list/tuple 输入。"""
        return value.landmark if hasattr(value, "landmark") else value

    def recognize(self, hand_landmarks, *, aspect_ratio: float = 1.0) -> str | None:
        """状态机入口：确认一次持续手势并返回标签，否则返回 ``None``。

        Args:
            hand_landmarks: ``multi_hand_landmarks[0]`` 或 ``None``（无手时）。
            aspect_ratio: 当前帧宽高比（``width / height``）。MediaPipe 把
                x 按图像宽度归一化、把 y 按高度归一化，因此度量前需要把 y
                缩到与 x 同量纲。

        Returns:
            ``str`` 表示这一次确认要触发的手势标签；``None`` 表示还在候选、
            释放或冷却中，调用方不应重复触发。
        """
        gesture = self.classify(hand_landmarks, aspect_ratio=aspect_ratio)
        now = self.clock()
        with self._lock:
            # 帧间隔过大（>max_frame_gap）视为事件流中断：候选、释放计时
            # 都要作废，重新累计。
            gap = self._last_frame_time is not None and now - self._last_frame_time > self.max_frame_gap
            self._last_frame_time = now
            if gesture is None:
                # 本帧未识别到任何手势 → 候选清空、计数清零
                self._candidate = None
                self._candidate_count = 0
                # 释放计时：从首次 None 起累计，达到 release_seconds 后才
                # 解锁 _latched；gap 触发时要重置开始计时点。
                if self._release_since is None or gap:
                    self._release_since = now
                if now - self._release_since >= self.release_seconds:
                    self._latched = None
                return None
            # 识别到任意手势 → 释放计时不需要了
            self._release_since = None
            # 仍在锁定期间：拒绝任何新动作（包括其它标签），避免抖动
            if self._latched is not None:
                return None
            # 标签变化或帧间隔过大 → 重新累计候选
            if gesture != self._candidate or gap:
                self._candidate = gesture
                self._candidate_since = now
                self._candidate_count = 1
            else:
                self._candidate_count += 1
            # 三个闸门同时满足才放行：连续帧数、最短持续时间、全局冷却
            if (self._candidate_count < self.stable_frames
                    or now - self._candidate_since < self.stable_seconds
                    or now - self.last_trigger_time < self.cooldown):
                return None
            # 触发：写状态 + 返回标签
            self.last_gesture = gesture
            self.last_trigger_time = now
            self._latched = gesture
            return gesture

    def reset(self) -> None:
        """重置观察状态（候选/计数/释放计时/帧间隔），但保留冷却。

        典型场景：切换摄像头源、用户手动暂停 / 恢复后想立刻重新累计，
        但不希望冷却被一并清掉导致刚切换就立刻触发一次。
        """
        with self._lock:
            self._candidate = self._latched = self._release_since = None
            self._candidate_count = 0
            self._last_frame_time = None

    @property
    def confirmed_gesture(self) -> str | None:
        """当前是否仍处于"已锁定"状态；只读，不修改任何字段。"""
        with self._lock:
            return self._latched

    def classify(self, hand_landmarks, *, aspect_ratio: float = 1.0) -> str | None:
        """单帧手势分类（不参与状态机、不消耗冷却）。

        与 ``recognize`` 的区别：本方法只做几何判定并返回当前最可能的标签，
        不会更新 ``last_gesture`` / ``_latched`` / ``last_trigger_time``。
        用于只想知道"现在手上是什么姿态"而不希望触发动作的查询场景。

        Args:
            hand_landmarks: ``multi_hand_landmarks[0]`` 或 ``None``。
            aspect_ratio: 当前帧宽高比（``width / height``），用于 y 轴归一。

        Returns:
            六类标签之一；识别不到返回 ``None``。
        """
        if hand_landmarks is None:
            return None
        lm = self._landmarks(hand_landmarks)
        if len(lm) < 21:
            return None
        if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
            return None
        try:
            # MediaPipe normalizes x/z by image width and y by image height.
            # Put all axes in the same units before measuring angles/lengths.
            # 把 y 除以宽高比，等价于把整个手掌在 y 方向拉伸到与 x 同尺度，
            # 这样 3D 距离才可比。
            points = [(float(p.x), float(p.y) / aspect_ratio, float(getattr(p, "z", 0.0))) for p in lm[:21]]
        except (TypeError, ValueError, AttributeError):
            return None
        if not all(math.isfinite(value) for point in points for value in point):
            return None
        # scale 用两个"手掌对角线"的最大值，避免手部姿态极端（手指全伸）
        # 时缩放因子过小。最小 0.015 是经验阈值：太小说明手离摄像头太远。
        scale = max(self._distance(points[0], points[9]), self._distance(points[5], points[17]))
        if scale < 0.015:
            return None
        # 对 4 根非拇指手指分别判定 extended / folded / uncertain
        states = {name: self._finger_state(points, joints, scale) for name, joints in self.FINGERS.items()}
        extended = {name for name, state in states.items() if state == "extended"}
        folded = {name for name, state in states.items() if state == "folded"}
        # 拇指是否伸直：2-3-4 三点形成的角度 ≥140° 视为伸直
        thumb_straight = self._angle(points[2], points[3], points[4]) >= 140
        if len(folded) == 4 and thumb_straight:
            # 四指收 + 拇指直 → 大拇指手势；用 2-4 的方向判定 up/down
            direction = self._vertical_direction(points[2], points[4], scale)
            # 拇指尖到腕部 vs 第三指节到腕部的距离差 > 0.10*scale → 拇指明显伸出
            thumb_reach = self._distance(points[4], points[0]) - self._distance(points[3], points[0])
            if direction and thumb_reach > 0.10 * scale:
                return "thumbs_up" if direction == "up" else "thumbs_down"
        if len(extended) == 4 and thumb_straight:
            # Require the thumb to spread away from the index base, avoiding
            # four extended fingers with a tucked thumb being called a palm.
            # 四指伸直 + 拇指直：要求拇指尖到食指根 > 第三指节到食指根，
            # 即拇指确实"张开"而非紧贴食指根部，避免"四指伸直但拇指紧贴"
            # 的姿态被误判为 palm。
            spread = self._distance(points[4], points[5]) - self._distance(points[3], points[5])
            if spread > 0.12 * scale:
                return "open_palm"
        if extended == {"index", "middle"} and folded == {"ring", "pinky"}:
            # 食指 + 中指伸直、无名指/小指收拢 → "V" 手势
            # 要求 8/12 两指尖距离 ≥ 30%*scale，避免两指并拢被误判
            if self._distance(points[8], points[12]) >= 0.30 * scale:
                return "victory"
        if extended == {"index"} and folded == {"middle", "ring", "pinky"}:
            # 只有食指伸 → 用 5-8 的方向判定 up/down
            direction = self._vertical_direction(points[5], points[8], scale)
            if direction:
                return "index_up" if direction == "up" else "index_down"
        return None

    @staticmethod
    def _distance(a, b) -> float:
        """3D 欧氏距离。"""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def _angle(a, b, c) -> float:
        """三点夹角 ∠ABC（顶点为 b），返回 0~180°。"""
        ab = [x - y for x, y in zip(a, b)]
        cb = [x - y for x, y in zip(c, b)]
        denominator = math.sqrt(sum(x * x for x in ab) * sum(x * x for x in cb))
        if denominator < 1e-10:
            # 三点重合或极接近 → 返回 0 避免 acos 数值错误
            return 0.0
        cosine = sum(x * y for x, y in zip(ab, cb)) / denominator
        # 钳位防止 acos 输入略超 [-1,1]（浮点误差）
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    def _finger_state(self, points, joints, scale) -> str:
        """判定单根手指是 "extended" / "folded" / "uncertain"。

        用三段角度（PIP/DIP）+ 指尖/掌指关节的距离比 + 指尖外伸量综合判断。
        角度阈值选 155°/150°（伸直接近平角）和 140°（明显弯曲）。
        """
        mcp, pip, dip, tip = joints
        lengths = [self._distance(points[a], points[b]) for a, b in zip(joints, joints[1:])]
        # 太小说明这段手离摄像头太远或被截断，置信度不够，不参与判定
        if min(lengths) < 0.025 * scale:
            return "uncertain"
        pip_angle = self._angle(points[mcp], points[pip], points[dip])
        dip_angle = self._angle(points[pip], points[dip], points[tip])
        # reach ≈ 1 表示四节近似共线（伸直），越小越弯
        reach = self._distance(points[mcp], points[tip]) / sum(lengths)
        # 指尖到腕部 > 第二指节到腕部 → 手指确实"往外"伸
        outward = self._distance(points[tip], points[0]) - self._distance(points[pip], points[0])
        if pip_angle >= 155 and dip_angle >= 150 and reach >= 0.85 and outward >= 0.08 * scale:
            return "extended"
        if pip_angle < 140 or dip_angle < 140 or reach < 0.70:
            return "folded"
        return "uncertain"

    @staticmethod
    def _vertical_direction(base, tip, scale) -> str | None:
        """根据 base→tip 向量判断垂直方向是 up / down / 不可判定。

        用 dy 阈值 0.25*scale、且 |dy| > 1.15*|dx| 来过滤"基本水平"的方
        向；只接受明确朝上或朝下。
        """
        dx, dy = tip[0] - base[0], tip[1] - base[1]
        if abs(dy) < 0.25 * scale or abs(dy) < 1.15 * abs(dx):
            return None
        # MediaPipe 的 y 轴向下为正；y 减小意味着屏幕方向"朝上"
        return "up" if dy < 0 else "down"

    def _is_within_cooldown(self) -> bool:
        """当前是否仍在全局冷却期内。"""
        return self.clock() - self.last_trigger_time < self.cooldown