"""Proposal compliance checker.

Inspired by a GSA entry in the federal AI inventory: an ML tool that
evaluates vendor proposals against solicitation requirements. This module:

1. **extracts requirements** from a solicitation: sentences with binding
   language (shall, must, required, will provide, ...), keeping their
   section numbers;
2. **splits the proposal** into passages;
3. **matches** each requirement to its best supporting passages with a blend
   of TF-IDF cosine similarity and key-term coverage;
4. labels each requirement **Addressed / Partially addressed / Not found**
   and produces a compliance matrix (CSV / HTML) with the evidence quoted.

This is a reviewer's aid that speeds up the first pass of a compliance
check. It is not an evaluation: source-selection decisions must be made by
the evaluation panel against the stated criteria.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BINDING = re.compile(r"\b(shall|must|is required to|are required to|required|will provide|will deliver|"
                     r"will maintain|will submit|offeror will|contractor will|at a minimum|no later than)\b", re.I)
SECTION = re.compile(r"^\s*((?:[A-Z]\.)?\d+(?:\.\d+)*)[.)]?\s+")
_SENT = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z(])")
_STOP = ENGLISH_STOP_WORDS | {"shall", "must", "contractor", "offeror", "vendor", "government", "provide",
                              "required", "ensure", "include", "including", "proposal", "will"}


@dataclass
class Requirement:
    req_id: str
    section: str
    text: str


def extract_requirements(solicitation: str) -> list[Requirement]:
    reqs, section, n = [], "", 0
    for line in solicitation.splitlines():
        m = SECTION.match(line)
        if m:
            section = m.group(1)
            line = line[m.end():]
        for sent in _SENT.split(line.strip()):
            sent = sent.strip(" -•\t")
            if len(sent.split()) >= 5 and BINDING.search(sent):
                n += 1
                reqs.append(Requirement(f"R{n:03d}", section, sent))
    return reqs


def split_passages(text: str, size: int = 3) -> list[str]:
    sents = [s.strip() for para in re.split(r"\n\s*\n", text) for s in _SENT.split(re.sub(r"\s+", " ", para))
             if s.strip()]
    starts = list(range(0, max(1, len(sents) - size + 1), max(1, size - 1)))
    if sents and starts[-1] + size < len(sents):
        starts.append(len(sents) - size)  # make sure the tail is covered
    out = [" ".join(sents[i:i + size]) for i in starts]
    return [o for o in out if len(o.split()) >= 4]


def _key_terms(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9\-]+", text.lower())
    words = [w[:-1] if w.endswith("s") and not w.endswith("ss") and len(w) > 4 else w for w in words]
    return {w for w in words if w not in _STOP and len(w) > 3}


def check_compliance(solicitation: str, proposal: str, addressed: float = 0.45,
                     partial: float = 0.25, top_n_evidence: int = 1) -> pd.DataFrame:
    """Compliance matrix: one row per requirement.

    ``score = 0.6 * tfidf_cosine + 0.4 * key_term_coverage`` (coverage = share
    of the requirement's content words that appear in the evidence passage).
    """
    reqs = extract_requirements(solicitation)
    passages = split_passages(proposal)
    if not reqs:
        raise ValueError("No requirements found (looked for shall/must/required ...)")
    if not passages:
        raise ValueError("Proposal text is empty")
    vec = TfidfVectorizer(stop_words=list(_STOP), ngram_range=(1, 2), sublinear_tf=True)
    vec.fit([r.text for r in reqs] + passages)
    sim = cosine_similarity(vec.transform([r.text for r in reqs]), vec.transform(passages))
    pass_terms = [_key_terms(p) for p in passages]
    rows = []
    for i, r in enumerate(reqs):
        terms = _key_terms(r.text)
        cov = np.array([len(terms & pt) / len(terms) if terms else 0.0 for pt in pass_terms])
        score = 0.6 * sim[i] + 0.4 * cov
        best = np.argsort(-score)[:top_n_evidence]
        s = float(score[best[0]])
        status = "Addressed" if s >= addressed else ("Partially addressed" if s >= partial else "Not found")
        missing = sorted(terms - pass_terms[best[0]])
        rows.append({"req_id": r.req_id, "section": r.section, "requirement": r.text, "status": status,
                     "score": round(s, 3), "evidence": " … ".join(passages[j] for j in best),
                     "missing_terms": ", ".join(missing[:8])})
    return pd.DataFrame(rows)


def matrix_to_html(matrix: pd.DataFrame, title: str = "Compliance matrix") -> str:
    colors = {"Addressed": "#d8f3dc", "Partially addressed": "#fff3bf", "Not found": "#ffd6d6"}
    counts = matrix.status.value_counts()
    rows = "".join(
        f"<tr style='background:{colors[r.status]}'><td>{r.req_id}</td><td>{html.escape(r.section)}</td>"
        f"<td>{html.escape(r.requirement)}</td><td><b>{r.status}</b><br><small>{r.score}</small></td>"
        f"<td><small>{html.escape(r.evidence)}</small></td><td><small>{html.escape(r.missing_terms)}</small></td></tr>"
        for r in matrix.itertuples())
    summary = " · ".join(f"{k}: {counts.get(k, 0)}" for k in colors)
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>"
            "body{font-family:system-ui,sans-serif;margin:2rem;color:#222}table{border-collapse:collapse;font-size:.85rem}"
            "td,th{border:1px solid #ccc;padding:6px;vertical-align:top}th{background:#eee}</style></head><body>"
            f"<h1>{title}</h1><p>{summary}</p><table><tr><th>ID</th><th>Sec.</th><th>Requirement</th>"
            f"<th>Status</th><th>Best evidence in proposal</th><th>Terms not found</th></tr>{rows}</table>"
            "<p><small>Automated first pass for reviewers. Not a source-selection evaluation.</small></p></body></html>")
