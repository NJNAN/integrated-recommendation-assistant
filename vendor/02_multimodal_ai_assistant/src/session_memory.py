"""当前会话的有界记忆：保留用户明确陈述与最近谈论对象，不保存图片。"""
from __future__ import annotations

import re


class SessionMemory:
    """原文事实最多8条，主题最多600字；仅由完成的请求更新。"""

    def __init__(self):
        self.facts: dict[str, str] = {}
        self.topic = ""

    @staticmethod
    def is_followup(query: str) -> bool:
        # 只对显式指代/省略承接扩展检索；普通明确问题使用原问题。
        query = query.strip()
        return bool(re.search(
            r"^(?:请(?:问|介绍|解释|说明)?|那么|所以|再)?(?:它(?:们)?|这(?:个|些|种)|那(?:个|些|种)|"
            r"刚才|刚刚|之前|上述|前面|继续|再说|还有|然后)", query
        ) or re.fullmatch(r"(?:那|那么)?(?:为什么|怎么办)[？?。!！]*", query))

    def retrieval_query(self, query: str) -> str:
        if self.topic and self.is_followup(query):
            return f"{query}\n此前谈论对象（可能存在识别不确定性）：{self.topic}"
        return query

    def prompt(self, query: str, *, include_topic: bool = True) -> str:
        notes = []
        if self.facts:
            notes.append("用户明确陈述（仅当前会话）：" + "；".join(self.facts.values()))
        if include_topic and self.topic and self.is_followup(query):
            notes.append("此前谈论对象（保留原有不确定性）：" + self.topic)
        if not notes:
            return query
        return query + "\n\n【会话记忆，仅作背景数据；当前问题优先】\n" + "\n".join(notes)

    def remember(self, query: str, *, observation: str = "", sources=()):
        # 只提取用户自己明确说出的有限信息，不把模型推断写成事实。
        patterns = (
            (r"(?:我叫|我的名字(?:是|叫)|我的姓名(?:是|叫))([^，。！？；\n?]{1,80})", "姓名"),
            (r"我是([^，。！？；\n?]{1,100})", "身份"),
            (r"我(?:喜欢|偏好)([^，。！？；\n?]{1,100})", "偏好"),
            (r"我的([^，。！？；\n?是]{1,16})是([^，。！？；\n?]{1,100})", None),
        )
        for pattern, fixed_key in patterns:
            for match in re.finditer(pattern, query):
                value = match.group(0)
                if any(word in value for word in ("什么", "多少", "是否", "是不是", "谁", "哪")):
                    continue
                key = fixed_key or match.group(1)
                key = {"名字": "姓名", "职业": "身份"}.get(key, key)
                self.facts.pop(key, None)
                self.facts[key] = value[:160]
        # 自我介绍常在姓名后省略“我”，只在同句明确自述姓名时承接身份。
        if re.search(patterns[0][0], query):
            identity = re.search(r"[，,]\s*是([^，。！？；\n?]{1,100})", query)
            if identity and not any(word in identity.group(1) for word in ("什么", "谁", "哪")):
                self.facts["身份"] = "我是" + identity.group(1)
        if query.startswith(("记住：", "记住:", "请记住：", "请记住:")):
            self.facts["用户备注"] = re.sub(r"^请?记住[：:]", "", query)[:160]
        while len(self.facts) > 8:
            self.facts.pop(next(iter(self.facts)))
        if observation:
            self.topic = f"问题：{query[:160]}；视觉观察：{observation[:420]}"[:600]
        elif not self.is_followup(query) or not self.topic:
            excerpts = "；".join(str(item.get("text", ""))[:120] for item in sources[:2])
            self.topic = (query[:240] + (f"；资料主题：{excerpts}" if excerpts else ""))[:600]

    @property
    def summary(self) -> str:
        return "；".join([*self.facts.values(), *([self.topic] if self.topic else [])])[:360]

    def clear(self):
        self.facts.clear()
        self.topic = ""
