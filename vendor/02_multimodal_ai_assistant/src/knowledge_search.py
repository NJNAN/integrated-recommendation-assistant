"""中文双字词、英文词与文件名的 BM25 检索，补充向量召回。"""
import math
import re
from collections import Counter
from pathlib import Path


_STOP_WORDS = {'介绍', '一下', '绍下', '绍一', '什么', '么是', '请问', '哪些',
               '有哪', '怎么', '如何', '知识', '识库', '文档', '资料', '里面'}


def _tokens(text):
    result = []
    for word in re.findall(r'[a-z0-9]+|[\u3400-\u9fff]+', text.casefold()):
        if '\u3400' <= word[0] <= '\u9fff':
            result.extend(word[i:i + 2] for i in range(len(word) - 1))
        else:
            result.append(word)
    return [t for t in result if t not in _STOP_WORDS]


class KeywordIndex:
    def __init__(self, documents):
        self.counts = []
        self.titles = []
        self.frequencies = Counter()
        for doc in documents:
            counts = Counter(_tokens(doc['text']))
            title = re.sub(r'^\d{4}-\d+[:：]?', '', Path(doc.get('source', '')).stem)
            self.titles.append(re.sub(r'[^\w]', '', title.casefold()))
            for token, count in Counter(_tokens(title)).items():
                counts[token] += count * 3
            self.counts.append(counts)
            self.frequencies.update(counts.keys())
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths)) or 1

    def scores(self, query):
        terms = set(_tokens(query))
        scores = []
        total = len(self.counts)
        for counts, length in zip(self.counts, self.lengths):
            score = 0.0
            for term in terms & counts.keys():
                frequency = self.frequencies[term]
                idf = math.log1p((total - frequency + .5) / (frequency + .5))
                tf = counts[term]
                score += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * length / self.average_length))
            scores.append(score)
        compact = re.sub(r'[^\w]', '', query.casefold())
        exact = [(i, len(title)) for i, title in enumerate(self.titles)
                 if len(title) >= 3 and title in compact]
        if exact:
            maximum = max(scores, default=0) or 1
            longest = max(length for _, length in exact)
            for i, length in exact:
                scores[i] += maximum * length / longest
        return scores
