import pytest
from ai_engine import (
    categorize_expense,
    parse_receipt_text,
    get_model_evaluation,
    CATEGORIES
)

def test_food_categorization():
    result = categorize_expense("Swiggy dinner order")
    assert result["category"] == "Food"
    assert 0.0 <= result["confidence"] <= 1.0
    assert "distribution" in result

def test_transport_categorization():
    result = categorize_expense("Uber cab ride to office")
    assert result["category"] == "Transport"
    assert result["confidence"] > 0.3

def test_utilities_categorization():
    result = categorize_expense("Bescom electricity bill monthly payment")
    assert result["category"] == "Utilities"

def test_income_categorization():
    result = categorize_expense("Monthly salary payroll credited")
    assert result["category"] == "Income"

def test_empty_description_falls_back():
    result = categorize_expense("   ")
    assert result["category"] == "Other"
    assert result["confidence"] == 0.0

def test_model_evaluation_metrics():
    metrics = get_model_evaluation()
    assert metrics["accuracy"] >= 80.0
    assert metrics["macro_precision"] > 70.0
    assert metrics["macro_recall"] > 70.0
    assert metrics["macro_f1"] > 70.0
    assert len(metrics["confusion_matrix"]) > 0

def test_receipt_text_parsing():
    sample_text = """
    STARBUCKS STORE #12
    Date: 2026-10-14
    1x Cold Brew 250.00
    1x Chocolate Croissant 180.00
    TOTAL: Rs. 430.00
    """
    res = parse_receipt_text(sample_text)
    assert res["amount"] == 430.00
    assert res["category"] == "Food"
