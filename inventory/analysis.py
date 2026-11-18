"""Descriptive analysis of the normalised inventory."""

from __future__ import annotations

import pandas as pd


def headline(df: pd.DataFrame) -> pd.DataFrame:
    """Key numbers per year."""
    g = df.groupby("year")
    out = pd.DataFrame({
        "use_cases": g.size(),
        "agencies": g["agency"].nunique(),
        "deployed": g["stage"].apply(lambda s: (s == "Deployed").sum()),
        "high_impact_or_rights_safety": g["is_high_impact"].sum(),
        "generative_ai": g["classification"].apply(lambda s: (s == "Generative AI").sum()),
    })
    out.loc[2024, "generative_ai"] = pd.NA  # field did not exist in 2024
    return out


def growth_by_agency(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    t = df.pivot_table(index="agency", columns="year", values="use_case_name", aggfunc="size", fill_value=0)
    t["change"] = t.get(2025, 0) - t.get(2024, 0)
    t["pct_change"] = (t["change"] / t[2024].where(t[2024] > 0)).round(3) if 2024 in t else pd.NA
    return t.sort_values(2025 if 2025 in t else t.columns[0], ascending=False).head(top)


def share(df: pd.DataFrame, col: str, year: int, top: int | None = None) -> pd.DataFrame:
    s = df.loc[df.year == year, col].fillna("(not reported)").value_counts()
    out = pd.DataFrame({"count": s, "share": (s / s.sum()).round(3)})
    return out.head(top) if top else out


def genai_by_agency(df: pd.DataFrame, min_cases: int = 40) -> pd.DataFrame:
    d = df[df.year == 2025]
    g = d.groupby("agency")
    out = pd.DataFrame({"use_cases": g.size(),
                        "generative_ai": g["classification"].apply(lambda s: (s == "Generative AI").sum())})
    out["genai_share"] = (out["generative_ai"] / out["use_cases"]).round(3)
    return out[out.use_cases >= min_cases].sort_values("genai_share", ascending=False)


def high_impact_by_agency(df: pd.DataFrame, year: int = 2025, top: int = 10) -> pd.DataFrame:
    d = df[df.year == year]
    g = d.groupby("agency")
    out = pd.DataFrame({"use_cases": g.size(), "high_impact": g["is_high_impact"].sum()})
    out["share"] = (out.high_impact / out.use_cases).round(3)
    return out.sort_values("high_impact", ascending=False).head(top)


def top_vendors(df: pd.DataFrame, top: int = 15) -> pd.Series:
    """Most-named vendors in 2025 (free text, lightly normalised)."""
    v = df.loc[df.year == 2025, "vendor"].dropna().str.strip()
    v = v[~v.str.lower().isin({"not available", "n/a", "none", "contractor teams", "open source",
                               "ai service provider", "law enforcement sensitive (les)", "tbd", "various"})]
    v = v.str.replace(r"(?i)\b(inc|llc|corp(oration)?|co)\.?$", "", regex=True).str.strip(" ,")
    return v.value_counts().head(top)


def data_quality(df: pd.DataFrame, year: int = 2025) -> pd.Series:
    """Share of rows missing key fields: useful feedback for inventory owners."""
    d = df[df.year == year]
    cols = ["topic_area", "stage", "impact", "purpose", "outputs"] + (["classification", "contracting"] if year == 2025 else [])
    return d[cols].isna().mean().round(3).sort_values(ascending=False)
