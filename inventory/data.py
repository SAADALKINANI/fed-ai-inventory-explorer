"""Download and normalise the OMB consolidated Federal AI Use Case Inventories (2024, 2025).

The raw files live in OMB's public GitHub repositories. They are U.S.
government works and are downloaded on demand rather than vendored here.
"""

from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = {
    2024: ("https://raw.githubusercontent.com/ombegov/2024-Federal-AI-Use-Case-Inventory/main/data/"
           "2024_consolidated_ai_inventory_raw_v2.csv", "cp1252"),
    2025: ("https://raw.githubusercontent.com/ombegov/2025-Federal-Agency-AI-Use-Case-Inventory/main/Data/"
           "2025_individually_reported_AI_use_cases.csv", "utf-8"),
}

COLUMNS = ["year", "agency", "agency_name", "bureau", "use_case_name", "topic_area", "stage", "impact",
           "is_high_impact", "classification", "contracting", "vendor", "purpose", "outputs", "has_pii",
           "has_custom_code"]


def cache_dir() -> Path:
    d = Path(os.environ.get("INVENTORY_CACHE", Path.home() / ".cache" / "fed-ai-inventory"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_path(year: int, download: bool = True) -> Path:
    url, _ = SOURCES[year]
    path = cache_dir() / url.rsplit("/", 1)[-1]
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        urllib.request.urlretrieve(url, path)  # noqa: S310 (fixed https URL)
    return path


def load_raw(year: int, path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else raw_path(year)
    return pd.read_csv(path, encoding=SOURCES[year][1], low_memory=False)


def _clean(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    return s.mask(s.isin(["", "nan", "None", "N/A", "n/a"]))


_STAGE_2024 = {
    "Planned": "Pre-deployment", "Ideation": "Pre-deployment", "Initiated": "Pre-deployment",
    "Acquisition and/or Development": "Pre-deployment", "Implementation and Assessment": "Pilot",
    "Operation and Maintenance": "Deployed", "In production": "Deployed", "In mission": "Deployed",
    "Retired": "Retired",
}


def _impact_2024(v) -> str | None:
    if not isinstance(v, str):
        return None
    v = v.strip().lower()
    if v.startswith("both"):
        return "Rights- and safety-impacting"
    if v.startswith("rights"):
        return "Rights-impacting"
    if v.startswith("safety"):
        return "Safety-impacting"
    if v.startswith("neither"):
        return "Neither"
    return "Not yet assessed"


def _contracting_2024(v) -> str | None:
    if not isinstance(v, str) or not v.strip():
        return None
    v = v.lower()
    if "both" in v or "combination" in v:
        return "Contracting and In House"
    if "in-house" in v or "no contract" in v:
        return "In-house Development"
    if "contract" in v:
        return "Vendor Purchased"
    return None


def normalize_2024(raw: pd.DataFrame) -> pd.DataFrame:
    col = lambda pat: next(c for c in raw.columns if re.search(pat, c, flags=re.I))  # noqa: E731
    impact = raw[col(r"rights-impacting, safety-impacting")].map(_impact_2024)
    df = pd.DataFrame({
        "year": 2024,
        "agency": _clean(raw["Agency Abbreviation"]),
        "agency_name": _clean(raw["Agency"]),
        "bureau": _clean(raw["Bureau"]),
        "use_case_name": _clean(raw["Use Case Name"]),
        "topic_area": _clean(raw["Use Case Topic Area"]),
        "stage": _clean(raw["Stage of Development"]).map(_STAGE_2024),
        "impact": impact,
        "is_high_impact": impact.isin(["Rights-impacting", "Safety-impacting", "Rights- and safety-impacting"]),
        "classification": pd.NA,
        "contracting": raw[col(r"developed \(or is it to be developed\) under contract")].map(_contracting_2024),
        "vendor": pd.NA,
        "purpose": _clean(raw[col(r"intended purpose and expected benefits")]),
        "outputs": _clean(raw[col(r"outputs")]),
        "has_pii": _clean(raw[col(r"personally identifiable information")]).str.startswith("Yes"),
        "has_custom_code": _clean(raw[col(r"custom-developed code")]).str.startswith("Yes"),
    })
    return df[COLUMNS]


def normalize_2025(raw: pd.DataFrame) -> pd.DataFrame:
    topic = _clean(raw["topic_area"]).replace({"Administrative functions": "Admin Functions",
                                                "Other (use other text field)": "Other"})
    topic = topic.where(~topic.fillna("").str.startswith("Other"), "Other")
    purpose = (_clean(raw["problem_solved"]).fillna("") + " " + _clean(raw["benefits"]).fillna("")).str.strip()
    df = pd.DataFrame({
        "year": 2025,
        "agency": _clean(raw["agency"]),
        "agency_name": _clean(raw["agency_name"]),
        "bureau": _clean(raw["agency_bureau"]),
        "use_case_name": _clean(raw["use_case_name"]),
        "topic_area": topic,
        "stage": _clean(raw["development_stage"]),
        "impact": _clean(raw["is_high_impact"]),
        "is_high_impact": _clean(raw["is_high_impact"]).eq("High-impact").fillna(False),
        "classification": _clean(raw["classification"]),
        "contracting": _clean(raw["contracting_usage"]),
        "vendor": _clean(raw["vendor_name"]),
        "purpose": purpose.mask(purpose == ""),
        "outputs": _clean(raw["system_outputs"]),
        "has_pii": _clean(raw["has_pii"]).eq("Yes"),
        "has_custom_code": _clean(raw["has_custom_code"]).eq("Yes"),
    })
    return df[COLUMNS]


#: 2025 high-impact minimum-practice columns, with the values that count as "in place".
PRACTICE_COLUMNS = {
    "hi_testing_conducted": ("Pre-deployment testing", ["Yes"]),
    "hi_assessment_completed": ("Impact assessment", ["Yes"]),
    "hi_independent_review": ("Independent review", ["Yes", "CAIO Review", "Internal Independent Review",
                                                      "Oversight Board Review"]),
    "hi_ongoing_monitoring": ("Ongoing monitoring", ["Yes"]),
    "hi_training_established": ("Operator training", ["Yes", "Training Established", "a) Yes"]),
    "hi_failsafe_presence": ("Fail-safe / human oversight", ["Yes", "Not Applicable"]),
    "hi_appeal_process": ("Appeal process", ["Appeal Process Established", "Not Applicable", "Appeal Precluded by Law"]),
    "hi_public_consultation": ("Public / end-user feedback", ["['Usability", "['Other", "['Public", "['a)", "['b)",
                                                               "['c)", "['d)"]),
}


def practices_2025(raw: pd.DataFrame) -> pd.DataFrame:
    """Status of each minimum practice for *deployed high-impact* 2025 use cases.

    Values: ``in place`` / ``in progress`` / ``waived`` / ``not reported``.
    """
    hi = raw[(raw["is_high_impact"] == "High-impact") & (raw["development_stage"] == "Deployed")]
    out = pd.DataFrame({"agency": hi["agency"].values})
    for colname, (label, done) in PRACTICE_COLUMNS.items():
        v = hi[colname].astype("string").fillna("")
        status = np.select(
            [v.str.startswith(tuple(done)), v.str.contains("progress", case=False), v.str.contains("waive", case=False)],
            ["in place", "in progress", "waived"], default="not reported")
        out[label] = status
    return out


def load_inventory(years: tuple[int, ...] = (2024, 2025)) -> pd.DataFrame:
    """Both inventories in one normalised table (one row per use case)."""
    frames = []
    for y in years:
        raw = load_raw(y)
        frames.append(normalize_2024(raw) if y == 2024 else normalize_2025(raw))
    df = pd.concat(frames, ignore_index=True)
    # 2024 used different abbreviations for a few agencies; harmonise on the 2025 ones by name
    if 2025 in years and 2024 in years:
        name_to_abbr = df[df.year == 2025].dropna(subset=["agency_name"]).groupby("agency_name")["agency"].first()
        mapped = df["agency_name"].map(name_to_abbr)
        df["agency"] = mapped.fillna(df["agency"])
    return df
