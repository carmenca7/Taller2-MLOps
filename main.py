from pandas import DataFrame
from joblib import load
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException
import pandas as pd

import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# Load the pipeline
pipeline = load('bankruptcy_pipeline.joblib')

# Define the input data model
class BankruptcyInput(BaseModel):
    roa_c_before_interest_and_depreciation_before_interest: float
    roa_a_before_interest_and_percentage_after_tax: float
    roa_b_before_interest_and_depreciation_after_tax: float
    operating_gross_margin: float
    realized_sales_gross_margin: float
    operating_profit_rate: float
    pre_tax_net_interest_rate: float
    after_tax_net_interest_rate: float
    non_industry_income_and_expenditure_to_revenue: float
    continuous_interest_rate_after_tax: float
    operating_expense_rate: float
    research_and_development_expense_rate: float
    cash_flow_rate: float
    interest_bearing_debt_interest_rate: float
    tax_rate_a: float
    net_value_per_share_b: float
    net_value_per_share_a: float
    net_value_per_share_c: float
    persistent_eps_last_four_seasons: float
    cash_flow_per_share: float
    revenue_per_share: float
    operating_profit_per_share: float
    per_share_net_profit_before_tax: float
    realized_sales_gross_profit_growth_rate: float
    operating_profit_growth_rate: float
    after_tax_net_profit_growth_rate: float
    regular_net_profit_growth_rate: float
    continuous_net_profit_growth_rate: float
    total_asset_growth_rate: float
    net_value_growth_rate: float
    total_asset_return_growth_rate_ratio: float
    cash_reinvestment_percentage: float
    current_ratio: float
    quick_ratio: float
    interest_expense_ratio: float
    total_debt_to_total_net_worth: float
    debt_ratio_percentage: float
    net_worth_to_assets: float
    long_term_fund_suitability_ratio_a: float
    borrowing_dependency: float
    contingent_liabilities_to_net_worth: float
    operating_profit_to_paid_in_capital: float
    net_profit_before_tax_to_paid_in_capital: float
    inventory_and_accounts_receivable_to_net_value: float
    total_asset_turnover: float
    accounts_receivable_turnover: float
    average_collection_days: float
    inventory_turnover_rate: float
    fixed_assets_turnover_frequency: float
    net_worth_turnover_rate: float
    revenue_per_person: float
    operating_profit_per_person: float
    allocation_rate_per_person: float
    working_capital_to_total_assets: float
    quick_assets_to_total_assets: float
    current_assets_to_total_assets: float
    cash_to_total_assets: float
    quick_assets_to_current_liability: float
    cash_to_current_liability: float
    current_liability_to_assets: float
    operating_funds_to_liability: float
    inventory_to_working_capital: float
    inventory_to_current_liability: float
    current_liabilities_to_liability: float
    working_capital_to_equity: float
    current_liabilities_to_equity: float
    long_term_liability_to_current_assets: float
    retained_earnings_to_total_assets: float
    total_income_to_total_expense: float
    total_expense_to_assets: float
    current_asset_turnover_rate: float
    quick_asset_turnover_rate: float
    working_capital_turnover_rate: float
    cash_turnover_rate: float
    cash_flow_to_sales: float
    fixed_assets_to_assets: float
    current_liability_to_liability: float
    current_liability_to_equity: float
    equity_to_long_term_liability: float
    cash_flow_to_total_assets: float
    cash_flow_to_liability: float
    cfo_to_assets: float
    cash_flow_to_equity: float
    current_liability_to_current_assets: float
    liability_assets_flag: float
    net_income_to_total_assets: float
    total_assets_to_gnp_price: float
    no_credit_interval: float
    gross_profit_to_sales: float
    net_income_to_stockholders_equity: float
    liability_to_equity: float
    degree_of_financial_leverage: float
    interest_coverage_ratio: float
    net_income_flag: float
    equity_to_liability: float

# Mapping dictionary from standardized variable names to original feature names
feature_name_mapping = {
    'roa_c_before_interest_and_depreciation_before_interest': ' ROA(C) before interest and depreciation before interest',
    'roa_a_before_interest_and_percentage_after_tax': ' ROA(A) before interest and % after tax',
    'roa_b_before_interest_and_depreciation_after_tax': ' ROA(B) before interest and depreciation after tax',
    'operating_gross_margin': ' Operating Gross Margin',
    'realized_sales_gross_margin': ' Realized Sales Gross Margin',
    'operating_profit_rate': ' Operating Profit Rate',
    'pre_tax_net_interest_rate': ' Pre-tax net Interest Rate',
    'after_tax_net_interest_rate': ' After-tax net Interest Rate',
    'non_industry_income_and_expenditure_to_revenue': ' Non-industry income and expenditure/revenue',
    'continuous_interest_rate_after_tax': ' Continuous interest rate (after tax)',
    'operating_expense_rate': ' Operating Expense Rate',
    'research_and_development_expense_rate': ' Research and development expense rate',
    'cash_flow_rate': ' Cash flow rate',
    'interest_bearing_debt_interest_rate': ' Interest-bearing debt interest rate',
    'tax_rate_a': ' Tax rate (A)',
    'net_value_per_share_b': ' Net Value Per Share (B)',
    'net_value_per_share_a': ' Net Value Per Share (A)',
    'net_value_per_share_c': ' Net Value Per Share (C)',
    'persistent_eps_last_four_seasons': ' Persistent EPS in the Last Four Seasons',
    'cash_flow_per_share': ' Cash Flow Per Share',
    'revenue_per_share': ' Revenue Per Share (Yuan ¥)',
    'operating_profit_per_share': ' Operating Profit Per Share (Yuan ¥)',
    'per_share_net_profit_before_tax': ' Per Share Net profit before tax (Yuan ¥)',
    'realized_sales_gross_profit_growth_rate': ' Realized Sales Gross Profit Growth Rate',
    'operating_profit_growth_rate': ' Operating Profit Growth Rate',
    'after_tax_net_profit_growth_rate': ' After-tax Net Profit Growth Rate',
    'regular_net_profit_growth_rate': ' Regular Net Profit Growth Rate',
    'continuous_net_profit_growth_rate': ' Continuous Net Profit Growth Rate',
    'total_asset_growth_rate': ' Total Asset Growth Rate',
    'net_value_growth_rate': ' Net Value Growth Rate',
    'total_asset_return_growth_rate_ratio': ' Total Asset Return Growth Rate Ratio',
    'cash_reinvestment_percentage': ' Cash Reinvestment %',
    'current_ratio': ' Current Ratio',
    'quick_ratio': ' Quick Ratio',
    'interest_expense_ratio': ' Interest Expense Ratio',
    'total_debt_to_total_net_worth': ' Total debt/Total net worth',
    'debt_ratio_percentage': ' Debt ratio %',
    'net_worth_to_assets': ' Net worth/Assets',
    'long_term_fund_suitability_ratio_a': ' Long-term fund suitability ratio (A)',
    'borrowing_dependency': ' Borrowing dependency',
    'contingent_liabilities_to_net_worth': ' Contingent liabilities/Net worth',
    'operating_profit_to_paid_in_capital': ' Operating profit/Paid-in capital',
    'net_profit_before_tax_to_paid_in_capital': ' Net profit before tax/Paid-in capital',
    'inventory_and_accounts_receivable_to_net_value': ' Inventory and accounts receivable/Net value',
    'total_asset_turnover': ' Total Asset Turnover',
    'accounts_receivable_turnover': ' Accounts Receivable Turnover',
    'average_collection_days': ' Average Collection Days',
    'inventory_turnover_rate': ' Inventory Turnover Rate (times)',
    'fixed_assets_turnover_frequency': ' Fixed Assets Turnover Frequency',
    'net_worth_turnover_rate': ' Net Worth Turnover Rate (times)',
    'revenue_per_person': ' Revenue per person',
    'operating_profit_per_person': ' Operating profit per person',
    'allocation_rate_per_person': ' Allocation rate per person',
    'working_capital_to_total_assets': ' Working Capital to Total Assets',
    'quick_assets_to_total_assets': ' Quick Assets/Total Assets',
    'current_assets_to_total_assets': ' Current Assets/Total Assets',
    'cash_to_total_assets': ' Cash/Total Assets',
    'quick_assets_to_current_liability': ' Quick Assets/Current Liability',
    'cash_to_current_liability': ' Cash/Current Liability',
    'current_liability_to_assets': ' Current Liability to Assets',
    'operating_funds_to_liability': ' Operating Funds to Liability',
    'inventory_to_working_capital': ' Inventory/Working Capital',
    'inventory_to_current_liability': ' Inventory/Current Liability',
    'current_liabilities_to_liability': ' Current Liabilities/Liability',
    'working_capital_to_equity': ' Working Capital/Equity',
    'current_liabilities_to_equity': ' Current Liabilities/Equity',
    'long_term_liability_to_current_assets': ' Long-term Liability to Current Assets',
    'retained_earnings_to_total_assets': ' Retained Earnings to Total Assets',
    'total_income_to_total_expense': ' Total income/Total expense',
    'total_expense_to_assets': ' Total expense/Assets',
    'current_asset_turnover_rate': ' Current Asset Turnover Rate',
    'quick_asset_turnover_rate': ' Quick Asset Turnover Rate',
    'working_capital_turnover_rate': ' Working capitcal Turnover Rate',
    'cash_turnover_rate': ' Cash Turnover Rate',
    'cash_flow_to_sales': ' Cash Flow to Sales',
    'fixed_assets_to_assets': ' Fixed Assets to Assets',
    'current_liability_to_liability': ' Current Liability to Liability',
    'current_liability_to_equity': ' Current Liability to Equity',
    'equity_to_long_term_liability': ' Equity to Long-term Liability',
    'cash_flow_to_total_assets': ' Cash Flow to Total Assets',
    'cash_flow_to_liability': ' Cash Flow to Liability',
    'cfo_to_assets': ' CFO to Assets',
    'cash_flow_to_equity': ' Cash Flow to Equity',
    'current_liability_to_current_assets': ' Current Liability to Current Assets',
    'liability_assets_flag': ' Liability-Assets Flag',
    'net_income_to_total_assets': ' Net Income to Total Assets',
    'total_assets_to_gnp_price': ' Total assets to GNP price',
    'no_credit_interval': ' No-credit Interval',
    'gross_profit_to_sales': ' Gross Profit to Sales',
    'net_income_to_stockholders_equity': ' Net Income to Stockholder\'s Equity',
    'liability_to_equity': ' Liability to Equity',
    'degree_of_financial_leverage': ' Degree of Financial Leverage (DFL)',
    'interest_coverage_ratio': ' Interest Coverage Ratio (Interest expense to EBIT)',
    'net_income_flag': ' Net Income Flag',
    'equity_to_liability': ' Equity to Liability',
}

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(input_data: BankruptcyInput):
    try:
        input_df = pd.DataFrame([input_data.dict()])

        # Rename the columns to match the feature names expected by the pipeline
        input_df = input_df.rename(columns=feature_name_mapping)

        # Ensure all expected columns are present
        expected_features = pipeline.named_steps['preprocessor'].feature_names_in_
        missing_cols = set(expected_features) - set(input_df.columns)
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

        # Reorder columns to match the order expected by the pipeline
        input_df = input_df[expected_features]

        probability = pipeline.predict_proba(input_df)[:, 1][0]
        prediction = pipeline.predict(input_df)[0]

        return {
            'probability_of_bankruptcy': float(probability),
            'prediction': int(prediction),  # 0 = Not Bankrupt, 1 = Bankrupt
            'prediction_label': 'Bankrupt' if prediction == 1 else 'Not Bankrupt'
        }
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(input_data: list[BankruptcyInput]):
    try:
        input_df = pd.DataFrame([data.dict() for data in input_data])

        # Rename the columns to match the feature names expected by the pipeline
        input_df = input_df.rename(columns=feature_name_mapping)

        # Ensure all expected columns are present
        expected_features = pipeline.named_steps['preprocessor'].feature_names_in_
        missing_cols = set(expected_features) - set(input_df.columns)
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

        # Reorder columns to match the order expected by the pipeline
        input_df = input_df[expected_features]

        probabilities = pipeline.predict_proba(input_df)[:, 1]
        predictions = pipeline.predict(input_df)

        return {
            'probabilities': probabilities.tolist(),
            'predictions': predictions.tolist(),
            'prediction_labels': ['Bankrupt' if pred == 1 else 'Not Bankrupt' for pred in predictions]
        }
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {
        'name': 'Bankruptcy Prediction API',
        'version': '1.0',
        'description': 'API for predicting company bankruptcy based on financial indicators',
        'endpoints': {
            '/predict': 'Make prediction for a single company',
            '/predict_batch': 'Make predictions for multiple companies'
        }
    }
