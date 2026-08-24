from __future__ import annotations

import re
from typing import Any


TECHNICAL_TERMS = (
    "编码值",
    "网格点",
    "算术平均",
    "质心",
    "离散度",
    "代理阈值",
    "evidence coverage",
    "error budget",
    "forecast_evidence_id",
)


def audit_communication(answer: str, question: str = "") -> dict[str, Any]:
    """Audit novice readability without changing evidence or imposing a report template."""
    text = str(answer or "").strip()
    issues: list[dict[str, Any]] = []
    if not text:
        issues.append({"code": "empty_answer", "message": "没有自然语言回答。"})

    term_counts = {
        term: len(re.findall(re.escape(term), text, flags=re.IGNORECASE))
        for term in TECHNICAL_TERMS
    }
    repeated_terms = {term: count for term, count in term_counts.items() if count >= 3}
    if repeated_terms:
        issues.append({
            "code": "repeated_technical_language",
            "message": "面向小白的正文重复使用技术术语。",
            "terms": repeated_terms,
        })

    technical_question = any(
        marker in str(question or "").lower()
        for marker in ("像素", "pixel", "数组", "代码", "实现", "字段", "接口", "python")
    )
    if not technical_question and re.search(r"像素|pixel|raw_data|ndarray|数组", text, re.IGNORECASE):
        issues.append({
            "code": "storage_representation_in_answer",
            "message": "天气问题的正文暴露了像素、数组或存储表示；应改写为VIL或区域层面的专业指标。",
        })

    number_count = len(re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?%?", text))
    # Do not impose a fixed number or character limit. Broad forecast, ensemble and
    # verification questions legitimately need more values and explanation than a
    # narrow factual question. Relevance and completeness are reviewed separately by
    # the question-specific coverage gate.

    raw_identifiers = sorted(set(re.findall(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b", text)))
    if raw_identifiers:
        issues.append({
            "code": "raw_identifiers_in_answer",
            "message": "正文暴露了代码或数据字段名。",
            "identifiers": raw_identifiers[:12],
        })

    disclaimer_count = sum(
        text.count(marker)
        for marker in ("不直接等同", "不能直接", "并非官方", "不是官方")
    )
    if disclaimer_count > 2:
        issues.append({
            "code": "repeated_disclaimer",
            "message": "限制说明重复，影响主结论阅读。",
            "count": disclaimer_count,
        })

    if re.match(r"^(根据|基于).{0,30}(?:数据|分析|计算|模型)", text):
        issues.append({
            "code": "technical_opening",
            "message": "回答以分析过程开头，而不是先给用户结论。",
        })

    heading_lines = [
        index for index, line in enumerate(text.splitlines())
        if re.fullmatch(r"\s*(?:#{1,6}\s+.+|\*\*[^*\n]+\*\*)\s*", line)
    ]
    lines = text.splitlines()
    orphan_headings = []
    for index in heading_lines:
        following = next((line.strip() for line in lines[index + 1:] if line.strip()), "")
        if not following or re.fullmatch(r"(?:#{1,6}\s+.+|\*\*[^*\n]+\*\*)", following):
            orphan_headings.append(lines[index].strip())
    if orphan_headings:
        issues.append({
            "code": "orphan_headings",
            "message": "回答中有标题但没有对应解释。",
            "headings": orphan_headings,
        })

    penalty = min(1.0, 0.16 * len(issues))
    return {
        "status": "passed" if not issues else "warning",
        "score": round(1.0 - penalty, 2),
        "character_count": len(text),
        "number_count": number_count,
        "issues": issues,
    }
