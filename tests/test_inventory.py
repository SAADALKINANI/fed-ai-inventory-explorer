from pathlib import Path

import pandas as pd
import pytest

from inventory.analysis import headline, top_vendors
from inventory.data import cache_dir, load_inventory, normalize_2024, normalize_2025, practices_2025
from inventory.proposal import check_compliance, extract_requirements, matrix_to_html, split_passages
from inventory.textmodel import TopicClassifier, UseCaseSearch

EX = Path(__file__).parent.parent / "examples" / "proposals"


def raw24():
    return pd.DataFrame({
        "Use Case Name": ["Claims triage", "Chatbot"], "Agency": ["Dept A", "Dept B"],
        "Agency Abbreviation": ["DA", "DB"], "Bureau": ["X", "Y"],
        "Use Case Topic Area": ["Health & Medical", None],
        "What is the intended purpose and expected benefits of the AI?": ["Prioritize claims", "Answer questions"],
        "Describe the AI system\x92s outputs.": ["Scores", "Text"],
        "Stage of Development": ["Operation and Maintenance", "Initiated"],
        "Is the AI use case rights-impacting, safety-impacting, both, or neither?": ["Rights-Impacting\n", "Neither"],
        "Was the AI system involved in this use case developed (or is it to be developed) under contract(s) or in-house? ":
            ["Developed in-house.", "Developed with contracting resources."],
        "Does this AI use case involve personally identifiable information (PII) that is maintained by the agency?": ["Yes", "No"],
        "Does this project include custom-developed code?": ["Yes", "No"],
    })


def raw25():
    base = {c: [None, None] for c in ["hi_testing_conducted", "hi_assessment_completed", "hi_independent_review",
                                      "hi_ongoing_monitoring", "hi_training_established", "hi_failsafe_presence",
                                      "hi_appeal_process", "hi_public_consultation"]}
    base.update({
        "agency": ["DA", "DB"], "agency_name": ["Dept A", "Dept B"], "agency_bureau": ["X", "Y"],
        "use_case_name": ["Fraud model", "Drafting assistant"], "topic_area": ["Administrative functions", None],
        "development_stage": ["Deployed", "Pilot"], "is_high_impact": ["High-impact", "Not High-impact"],
        "classification": ["Classical ML", "Generative AI"], "contracting_usage": ["Vendor Purchased", None],
        "vendor_name": ["Acme Inc.", None], "problem_solved": ["Find fraud", "Draft memos"],
        "benefits": ["Savings", "Time"], "system_outputs": ["Risk score", "Draft text"],
        "has_pii": ["Yes", "No"], "has_custom_code": ["No", "No"],
    })
    base["hi_testing_conducted"] = ["Yes", None]
    base["hi_ongoing_monitoring"] = ["In-progress", None]
    return pd.DataFrame(base)


def test_normalize_2024():
    d = normalize_2024(raw24())
    assert d.stage.tolist() == ["Deployed", "Pre-deployment"]
    assert d.is_high_impact.tolist() == [True, False]
    assert d.contracting.tolist() == ["In-house Development", "Vendor Purchased"]


def test_normalize_2025_and_practices():
    d = normalize_2025(raw25())
    assert d.topic_area.iloc[0] == "Admin Functions" and d.is_high_impact.tolist() == [True, False]
    assert d.purpose.iloc[0] == "Find fraud Savings"
    p = practices_2025(raw25())
    assert len(p) == 1 and p["Pre-deployment testing"].iloc[0] == "in place"
    assert p["Ongoing monitoring"].iloc[0] == "in progress" and p["Appeal process"].iloc[0] == "not reported"


def test_vendor_normalisation():
    df = pd.DataFrame({"year": 2025, "vendor": ["Acme Inc.", "Acme", "Not available", "Beta LLC"]})
    assert top_vendors(df).to_dict() == {"Acme": 2, "Beta": 1}


def test_requirement_extraction():
    reqs = extract_requirements((EX / "solicitation.txt").read_text())
    assert len(reqs) == 11 and reqs[0].section == "C.2.1" and reqs[-1].section == "C.4.3"


def test_passages_cover_tail():
    text = "One two three four. " * 6 + "Final sentence about the transition plan."
    assert "transition plan" in split_passages(text)[-1]


def test_compliance_strong_vs_weak():
    sol = (EX / "solicitation.txt").read_text()
    a = check_compliance(sol, (EX / "proposal_vendor_a.txt").read_text())
    b = check_compliance(sol, (EX / "proposal_vendor_b.txt").read_text())
    assert (a.status == "Addressed").sum() >= 9
    assert (b.status == "Not found").sum() >= 8
    assert "<table>" in matrix_to_html(a)


def test_text_models_on_toy_data():
    df = pd.DataFrame({
        "use_case_name": ["chatbot answers questions"] * 10 + ["satellite image wildfire detection"] * 10,
        "purpose": ["help desk virtual assistant answers employee questions"] * 10 + ["detect smoke in imagery"] * 10,
        "outputs": ["text"] * 20, "topic_area": ["IT"] * 10 + ["Science"] * 10,
        "year": 2025, "agency": "X", "stage": "Pilot",
    })
    clf = TopicClassifier().fit(df)
    assert clf.predict(df.iloc[[0, 15]]).predicted_topic.tolist() == ["IT", "Science"]
    assert UseCaseSearch(df).query("wildfire smoke").iloc[0].topic_area == "Science"


@pytest.mark.skipif(not (cache_dir() / "2025_individually_reported_AI_use_cases.csv").exists(),
                    reason="real inventory not downloaded")
def test_real_inventory_counts():
    df = load_inventory()
    h = headline(df)
    assert h.loc[2024, "use_cases"] == 2133 and h.loc[2025, "use_cases"] == 3611
