from __future__ import annotations

"""Determine whether the user's request explicitly requires a visual artifact."""

import re


_CLAUSE_SPLIT = re.compile(r"[，。；;！？!?\n]+")
_NEGATED_VISUAL = re.compile(
    r"(?:不得|不要|无需|不需要|禁止|避免).{0,12}(?:生成|绘制|制作|画|展示|提供)"
)
_POSITIVE_VISUAL = re.compile(
    r"(?:生成|绘制|制作|画出|画一|展示|提供|输出|叠加).{0,80}"
    r"(?:可视化|地图|图表|图像|图|曲线|时间轴)"
)
_DESIRE_VISUAL = re.compile(
    r"(?:需要|希望|想看|给我).{0,30}(?:可视化|地图|图表|图像|曲线|时间轴)"
)
_DIRECT_VISUAL = re.compile(
    r"(?:通过.{0,40}(?:可视化|地图|图表|图像|图|曲线|时间轴)|绘制)"
)


def visual_artifact_required(question: str) -> bool:
    """Return true only for a positive, explicit visual request.

    Negative constraints such as ``不得绘制没有时间序列支持的轨迹`` do not
    turn an otherwise textual question into a mandatory-figure task.
    """

    for raw_clause in _CLAUSE_SPLIT.split(str(question or "")):
        clause = raw_clause.strip()
        if not clause or _NEGATED_VISUAL.search(clause):
            continue
        if (
            _POSITIVE_VISUAL.search(clause)
            or _DESIRE_VISUAL.search(clause)
            or _DIRECT_VISUAL.search(clause)
        ):
            return True
    return False
