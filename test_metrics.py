import pandas as pd
import pytest
from metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)
from load_data import clean_employee_data


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "Yes", "No"],
            "overtime": ["Yes", "No", "Yes", "No"],
            "job_satisfaction": [1, 2, 1, 2],
            "monthly_income": [4000, 6000, 5000, 7000],
            "age": [25, 30, 35, 40],
            "travel_frequency": ["Frequent", "Rarely", "Frequent", "Rarely"],
            "years_at_company": [1, 5, 2, 8],
        }
    )


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent(sample_df):
    assert attrition_rate(sample_df) == 50.0


def test_attrition_rate_all_stay(sample_df):
    sample_df["attrition"] = "No"
    assert attrition_rate(sample_df) == 0.0


def test_attrition_rate_all_leave(sample_df):
    sample_df["attrition"] = "Yes"
    assert attrition_rate(sample_df) == 100.0


# --- attrition_by_department ---

def test_attrition_by_department_returns_expected_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_rates(sample_df):
    result = attrition_by_department(sample_df)
    rates = result.set_index("department")["attrition_rate"]
    assert rates["Sales"] == 50.0
    assert rates["HR"] == 50.0


def test_attrition_by_department_sorted_descending():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5],
            "department": ["Sales", "Sales", "HR", "HR", "HR"],
            "attrition": ["Yes", "Yes", "Yes", "No", "No"],
        }
    )
    result = attrition_by_department(df)
    assert result.iloc[0]["department"] == "Sales"  # 100% > 33.33%


# --- attrition_by_overtime ---

def test_attrition_by_overtime_returns_expected_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_rates(sample_df):
    result = attrition_by_overtime(sample_df)
    rates = result.set_index("overtime")["attrition_rate"]
    assert rates["Yes"] == 100.0
    assert rates["No"] == 0.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_returns_expected_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values(sample_df):
    result = average_income_by_attrition(sample_df)
    values = result.set_index("attrition")["avg_monthly_income"]
    assert values["Yes"] == 4500.0
    assert values["No"] == 6500.0


# --- satisfaction_summary ---

def test_satisfaction_summary_returns_expected_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_within_group_rate(sample_df):
    # satisfaction 1: 2 employees, both left  → 100%
    # satisfaction 2: 2 employees, none left  → 0%
    result = satisfaction_summary(sample_df)
    rates = result.set_index("job_satisfaction")["attrition_rate"]
    assert rates[1] == 100.0
    assert rates[2] == 0.0


def test_satisfaction_summary_sorted_by_satisfaction(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result["job_satisfaction"]) == sorted(result["job_satisfaction"])


# --- clean_employee_data ---

def test_clean_employee_data_raises_on_missing_column():
    df = pd.DataFrame({"employee_id": [1], "department": ["Sales"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_employee_data(df)


def test_clean_employee_data_normalizes_attrition_casing():
    df = pd.DataFrame(
        {
            "employee_id": [1], "department": ["Sales"], "age": [30],
            "monthly_income": [5000], "job_satisfaction": [3],
            "overtime": ["No"], "travel_frequency": ["Rarely"],
            "years_at_company": [2], "attrition": ["yes"],
        }
    )
    result = clean_employee_data(df)
    assert result["attrition"].iloc[0] == "Yes"


def test_clean_employee_data_fills_overtime_null():
    df = pd.DataFrame(
        {
            "employee_id": [1], "department": ["Sales"], "age": [30],
            "monthly_income": [5000], "job_satisfaction": [3],
            "overtime": [None], "travel_frequency": ["Rarely"],
            "years_at_company": [2], "attrition": ["No"],
        }
    )
    result = clean_employee_data(df)
    assert result["overtime"].iloc[0] == "No"


def test_clean_employee_data_fills_satisfaction_null():
    df = pd.DataFrame(
        {
            "employee_id": [1], "department": ["Sales"], "age": [30],
            "monthly_income": [5000], "job_satisfaction": [None],
            "overtime": ["No"], "travel_frequency": ["Rarely"],
            "years_at_company": [2], "attrition": ["No"],
        }
    )
    result = clean_employee_data(df)
    assert result["job_satisfaction"].iloc[0] == 3


def test_clean_employee_data_fills_income_with_median():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3], "department": ["Sales", "HR", "IT"], "age": [30, 35, 40],
            "monthly_income": [4000.0, None, 6000.0], "job_satisfaction": [3, 3, 3],
            "overtime": ["No", "No", "No"], "travel_frequency": ["Rarely", "Rarely", "Rarely"],
            "years_at_company": [2, 3, 4], "attrition": ["No", "No", "No"],
        }
    )
    result = clean_employee_data(df)
    assert result["monthly_income"].iloc[1] == 5000.0
