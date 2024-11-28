from pandas import DataFrame
from joblib import load
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException
import pandas as pd

# Cargar el pipeline
pipeline = load('bankruptcy_pipeline.joblib')

# Definir las variables de entrada al modelo
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

# Iniciar FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(input_data: BankruptcyInput):
    try:
        input_df = pd.DataFrame([input_data.dict()])

        probability = pipeline.predict_proba(input_df)[:, 1][0]
        prediction = pipeline.predict(input_df)[0]

        return {
            'probability_of_bankruptcy': float(probability),
            'prediction': int(prediction),  # 0 = Not Bankrupt, 1 = Bankrupt
            'prediction_label': 'Bankrupt' if prediction == 1 else 'Not Bankrupt'
        }
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(input_data: list[BankruptcyInput]):
    try:
        input_df = pd.DataFrame([data.dict() for data in input_data])

        probabilities = pipeline.predict_proba(input_df)[:, 1]
        predictions = pipeline.predict(input_df)

        return {
            'probabilities': probabilities.tolist(),
            'predictions': predictions.tolist(),
            'prediction_labels': ['Bankrupt' if pred == 1 else 'Not Bankrupt' for pred in predictions]
        }
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
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
