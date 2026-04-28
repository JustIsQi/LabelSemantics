"""Shared entity-decoding post-processing for inference.

Both :mod:`tests.batch_test` and :mod:`service.app` import from here so that
batch evaluation and the FastAPI service produce identical predictions.

Two kinds of corrections are applied to each predicted span:

1. ``trim_entity_span``: strip leading/trailing punctuation and whitespace
   that the model may have folded into an enumeration entity (e.g.
   ``"招商银行、"`` -> ``"招商银行"``).

2. ``trim_entity_role_suffix``: when the model overshoots an entity by
   gluing a role / job-title / department word onto a broker or company
   name (e.g. ``"中信证券分析师明明"`` -> ``"中信证券"``), cut the span
   back to the start of that role word. The pattern lists are intentionally
   conservative: only strong signal words that almost never legitimately
   appear inside a BROKER/C span in the training data.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

from .metrics import get_entities

ENTITY_BOUNDARY_STRIP_CHARS = set(
    " \t\u3000、,，。.；;：:!！?？/\\()()[]【】《》<>\"'`~“”‘’\r\n"
)


# Words that, when they appear inside a predicted span, are almost certainly
# the model overrunning the true entity into a downstream role / department /
# job-title / common verb. We cut the span at the first occurrence of any of
# these (after the entity start), keeping only the prefix.
#
# Keep the lists conservative -- false positives here will hurt recall on
# real entities like "高盛资产管理" or "中信证券研究所".
_BROKER_TRAILING_NOISE = (
    # 研究 / 销售类岗位
    "首席分析师", "首席策略师", "策略分析师", "策略师",
    "分析师", "研究员", "首席", "研究主管", "研究总监",
    # 部门 / 实体后缀（这些是 BROKER 之外的下游机构，不应被吃进 BROKER 里）
    "研究所", "研究院", "研究部", "营业部", "资管部", "自营部",
    "投行部", "零售部", "固收团队", "TMT团队", "电新团队",
    # 通用动词 / 状态
    "表示", "认为", "指出", "发布", "披露", "表态", "最新", "最近",
)

_COMPANY_TRAILING_NOISE = (
    # 高管头衔
    "董事长", "总经理", "总裁", "副总裁", "董秘", "董事", "CEO", "CFO", "CTO",
    "发言人", "创始人", "联合创始人",
    # 公司动作 / 文档
    "发布", "披露", "表示", "回应", "声明", "签约", "收购", "并购",
    "子公司", "控股股东", "关联方", "董事会", "股东大会",
    "招股书", "招股说明书", "年报", "半年报", "季报", "中报", "财报",
    "业绩说明会", "电话会议", "电话会", "调研纪要",
)


def _build_pattern(words: Iterable[str]) -> re.Pattern[str]:
    # Sort by length descending so the regex prefers the longest match
    # ("首席分析师" before "首席").
    sorted_words = sorted(set(words), key=len, reverse=True)
    return re.compile("|".join(re.escape(word) for word in sorted_words))


ENTITY_ROLE_SUFFIX_PATTERNS: Dict[str, re.Pattern[str]] = {
    "BROKER": _build_pattern(_BROKER_TRAILING_NOISE),
    "C": _build_pattern(_COMPANY_TRAILING_NOISE),
}


def trim_entity_span(query: str, start: int, end: int) -> Tuple[int, int]:
    """Strip leading/trailing punctuation/whitespace from a predicted span.

    Returns ``(new_start, new_end)``. If the span is fully trimmed away,
    ``new_start > new_end`` and the caller should drop it.
    """
    while start <= end and query[start] in ENTITY_BOUNDARY_STRIP_CHARS:
        start += 1
    while end >= start and query[end] in ENTITY_BOUNDARY_STRIP_CHARS:
        end -= 1
    return start, end


def trim_entity_role_suffix(
    query: str, start: int, end: int, entity_type: str,
) -> Tuple[int, int]:
    """Cut the span at the first role-suffix word inside it.

    For BROKER/C, if the predicted span contains a role/job-title/department
    word (e.g. "分析师", "董事长"), shrink ``end`` so the span ends right
    before that word. If the role word is at the very start of the span the
    span is returned as invalid (``start > end``).
    """
    pattern = ENTITY_ROLE_SUFFIX_PATTERNS.get(entity_type)
    if pattern is None or start >= end:
        return start, end

    # Search inside the span only.
    match = pattern.search(query, pos=start, endpos=end + 1)
    if match is None:
        return start, end

    if match.start() <= start:
        # The whole span starts with a role word -- the model never had a
        # real entity, drop it.
        return start, start - 1

    return start, match.start() - 1


def decode_entities(
    query: str,
    tags: List[str],
    label_descriptions: Dict[str, str],
) -> Dict[str, List[str]]:
    """Run BIO decode + trim_entity_span + trim_entity_role_suffix.

    This is the canonical inference-side decoder used by both batch
    evaluation and the FastAPI service.
    """
    entities: Dict[str, List[str]] = {label: [] for label in label_descriptions}
    for entity_type, start, end in get_entities(tags):
        if entity_type not in entities:
            continue
        start, end = trim_entity_span(query, start, end)
        if start > end:
            continue
        start, end = trim_entity_role_suffix(query, start, end, entity_type)
        if start > end:
            continue
        # trim once more in case the role-suffix trim left punctuation
        # behind ("中信证券-分析师" -> after suffix trim: "中信证券-").
        start, end = trim_entity_span(query, start, end)
        if start > end:
            continue
        entities[entity_type].append(query[start:end + 1])
    return entities
