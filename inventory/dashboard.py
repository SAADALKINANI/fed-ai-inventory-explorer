"""Build a static HTML dashboard (GitHub Pages ready) from the inventory."""

from __future__ import annotations

import base64
import html
import io
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import analysis as A  # noqa: E402
from .data import load_raw, practices_2025  # noqa: E402
from .textmodel import TopicClassifier, discover_themes  # noqa: E402

BLUE, ORANGE, GREY, GREEN, YELLOW, RED = "#1f5aa6", "#e07b39", "#9aa5b1", "#2e8b57", "#e6b422", "#c0392b"


def _img(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)


def chart_growth(df) -> str:
    g = A.growth_by_agency(df, 12)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    y = range(len(g))
    ax.barh([i + 0.2 for i in y], g[2024], height=0.4, color=GREY, label="2024")
    ax.barh([i - 0.2 for i in y], g[2025], height=0.4, color=BLUE, label="2025")
    ax.set_yticks(list(y), g.index)
    ax.invert_yaxis()
    ax.legend(frameon=False)
    ax.set_title("Reported AI use cases, top 12 agencies (2025)", loc="left")
    _style(ax)
    return _img(fig)


def chart_share(df, col, title, color=BLUE) -> str:
    s = A.share(df, col, 2025)["count"].sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(s) + 1))
    ax.barh(s.index, s.values, color=[GREY if i == "(not reported)" else color for i in s.index])
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    ax.set_title(title, loc="left")
    _style(ax)
    return _img(fig)


def chart_practices(prac: pd.DataFrame) -> str:
    cols = [c for c in prac.columns if c != "agency"]
    counts = pd.DataFrame({c: prac[c].value_counts() for c in cols}).T.fillna(0)
    order = [k for k in ["in place", "in progress", "waived", "not reported"] if k in counts]
    colors = {"in place": GREEN, "in progress": YELLOW, "waived": GREY, "not reported": RED}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    left = pd.Series(0.0, index=counts.index)
    for k in order:
        ax.barh(counts.index, counts[k], left=left, color=colors[k], label=k)
        left += counts[k]
    ax.invert_yaxis()
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    ax.set_title(f"Minimum practices, {len(prac)} deployed high-impact uses (2025)", loc="left")
    _style(ax)
    return _img(fig)


def _table(d: pd.DataFrame | pd.Series, fmt: dict | None = None) -> str:
    d = d.to_frame() if isinstance(d, pd.Series) else d
    return d.to_html(classes="t", border=0, float_format=lambda x: f"{x:,.3g}", na_rep="–", escape=True)


def build_dashboard(df: pd.DataFrame, out: str | Path = "docs/index.html") -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    h = A.headline(df)
    d25 = df[df.year == 2025]
    prac = practices_2025(load_raw(2025))
    in_place_all = (prac.drop(columns="agency") == "in place").all(axis=1).sum()

    clf = TopicClassifier()
    ev = clf.evaluate(d25)
    clf.fit(d25)
    filled = clf.fill_missing(d25)
    themes = discover_themes(d25, n_themes=10)

    k = lambda label, val, sub="": f'<div class="kpi"><div class="v">{val}</div><div class="l">{label}</div><div class="s">{sub}</div></div>'  # noqa: E731
    growth = h.loc[2025, "use_cases"] / h.loc[2024, "use_cases"] - 1
    kpis = "".join([
        k("AI use cases reported (2025)", f"{h.loc[2025, 'use_cases']:,}", f"{growth:+.0%} vs 2024 ({h.loc[2024, 'use_cases']:,})"),
        k("Generative AI use cases", f"{int(h.loc[2025, 'generative_ai']):,}", f"{h.loc[2025, 'generative_ai'] / h.loc[2025, 'use_cases']:.0%} of 2025"),
        k("High-impact use cases", f"{int(h.loc[2025, 'high_impact_or_rights_safety']):,}", f"{len(prac)} deployed"),
        k("Deployed high-impact with all practices in place", f"{in_place_all}", f"of {len(prac)}"),
    ])
    sections = [
        ("Growth by agency", chart_growth(df) + _table(A.growth_by_agency(df, 12))),
        ("What kind of AI?", chart_share(df, "classification", "AI classification (2025)")
         + "<p>Generative AI share by agency (agencies with ≥ 40 use cases):</p>" + _table(A.genai_by_agency(df).head(10))),
        ("What is it used for?", chart_share(df, "topic_area", "Topic area (2025)", ORANGE)),
        ("Where is it in the lifecycle?", chart_share(df, "stage", "Development stage (2025)", BLUE)),
        ("High-impact AI and the minimum practices", chart_practices(prac)
         + "<p>OMB M-25-21 requires deployed high-impact AI to have each minimum practice in place (or a waiver). "
           "Most deployed high-impact entries in the 2025 inventory report the practices as in progress or leave them blank.</p>"
         + _table(A.high_impact_by_agency(df))),
        ("Who are the vendors?", "<p>Most-named vendors (free-text field; 86% of rows leave it blank):</p>" + _table(A.top_vendors(df, 12))),
        ("Themes in the descriptions (NMF topic model)", _table(themes[["top_terms", "use_cases", "example"]])),
        ("Filling in missing topic areas", f"<p>A TF-IDF + logistic-regression classifier trained on the 2025 rows that do report a topic "
         f"reaches <b>{ev['accuracy']:.0%} accuracy</b> (macro F1 {ev['macro_f1']:.2f}) on a held-out {ev['n_test']} rows across "
         f"{d25.topic_area.nunique()} topics. Of the {int(d25.topic_area.isna().sum())} rows with no topic, "
         f"{int(filled.topic_area_imputed.sum())} have enough text for a confident (≥ 50%) suggestion; most blank-topic rows also leave the description blank.</p>"),
        ("Data quality (share of 2025 rows missing a field)", _table(A.data_quality(df).rename("missing"))),
    ]
    body = "".join(f"<section><h2>{html.escape(t)}</h2>{c}</section>" for t, c in sections)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Federal AI Use Case Inventory Explorer</title>
<style>body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:980px;margin:0 auto;padding:2rem 1rem;color:#1d2733}}
h1{{margin:0}} .sub{{color:#5b6b7b}} .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:1.5rem 0}}
.kpi{{background:#f3f6fa;border-radius:10px;padding:14px}} .v{{font-size:1.9rem;font-weight:700;color:{BLUE}}} .l{{font-weight:600}} .s{{color:#5b6b7b;font-size:.85rem}}
section{{margin:2.2rem 0}} img{{max-width:100%}} table.t{{border-collapse:collapse;font-size:.82rem;margin:.6rem 0}}
table.t td,table.t th{{padding:4px 10px;border-bottom:1px solid #e3e8ee;text-align:right}} table.t th{{text-align:left}}
footer{{color:#5b6b7b;font-size:.8rem;margin-top:3rem}}</style></head><body>
<h1>Federal AI Use Case Inventory Explorer</h1>
<p class="sub">OMB's consolidated 2024 and 2025 inventories of how U.S. federal agencies use AI. Built {date.today():%B %Y} with
<a href="https://github.com/SAADALKINANI/fed-ai-inventory-explorer">fed-ai-inventory-explorer</a>.</p>
<div class="kpis">{kpis}</div>{body}
<footer>Data: <a href="https://github.com/ombegov/2024-Federal-AI-Use-Case-Inventory">ombegov/2024-Federal-AI-Use-Case-Inventory</a> and
<a href="https://github.com/ombegov/2025-Federal-Agency-AI-Use-Case-Inventory">ombegov/2025-Federal-Agency-AI-Use-Case-Inventory</a>
(individually reported use cases; consolidated commercial-product reporting is not included). 2024 counts "rights- or safety-impacting" uses,
2025 counts "high-impact" uses (OMB M-25-21); the definitions differ. Agency-reported data, not independently verified.</footer>
</body></html>"""
    out.write_text(page, encoding="utf-8")
    return out
