"""Bilingual lexical processing used only for topic representation."""
import logging
import re
import jieba
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

jieba.setLogLevel(logging.ERROR)
DOMAIN_TERMS = "水稻 胚乳 糊粉层 胚乳细胞化 淀粉合成 淀粉粒 直链淀粉 支链淀粉 贮藏蛋白 储藏蛋白 谷蛋白 醇溶蛋白 白垩质 垩白 基因表达 转录因子 转基因 生物反应器 籽粒灌浆 籽粒发育 分子机制 程序性细胞死亡 胚乳发育 信号转导 赤霉素 脱落酸 生长素 植酸 胚乳注射法".split()
for term in DOMAIN_TERMS:
    jieba.add_word(term)

CHINESE_STOP_WORDS = set("的 了 和 与 及 在 对 为 是 中 由 于 以 将 等 并 其 该 这 那 一个 一种 我们 本文 本研究 研究 结果 表明 发现 通过 进行 利用 方法 分析 影响 相关 不同 进一步 具有 可以 可能 显著 主要 采用 根据 因此 同时 此外 以及 从而 中的 中有 其中 作为 对于 关于 由于 如果 而且 或者 但是 但是 这些 那些 此 之 所 被 未 无 有 已 也 又 更 很 比 较 最 能 可 应 上 下 时 使 使得 得到 提高 降低 增加 减少 变化 情况 作用 应用 观察 探讨 证明 说明 提供 探索 初步 本试验 试验 结论 目的 摘要 作者 文献 数据 论文 年 月 日 正".split())
STOP_WORDS = set(ENGLISH_STOP_WORDS) | CHINESE_STOP_WORDS | {
    "study", "studies", "results", "result", "using", "used", "showed", "shown", "based",
    "copyright", "reserved", "rights", "elsevier", "springer", "wiley", "et", "al", "fig",
    "conclusion", "conclusions", "abstract", "author", "authors", "respectively",
}
TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]+|[A-Za-z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*")


def tokenize(text):
    result = []
    for token in TOKEN_PATTERN.findall(str(text)):
        words = jieba.lcut(token) if re.match(r"[\u3400-\u9fff]", token) else [token.casefold()]
        result.extend(word for word in words if len(word) >= 2 and word not in STOP_WORDS)
    return result
