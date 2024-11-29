from pandas import DataFrame
from joblib import load
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException
import pandas as pd

# Cargar el pipeline
pipeline = load('bankruptcy_pipeline.joblib')

# Definir las variables de entrada al modelo
class BankruptcyInput(BaseModel):
    ROA(C) before interest and depreciation before interest: float
    ROA(A) before interest and % after tax: float
    ROA(B) before interest and depreciation after tax: float
    Operating Gross Margin: float
    Realized Sales Gross Margin: float
    Operating Profit Rate: float
    Pre-tax net Interest Rate: float
    After-tax net Interest Rate: float
    Non-industry income and expenditure/revenue: float
    Continuous interest rate (after tax): float
    Operating Expense Rate: float
    Research and development expense rate: float
    Cash flow rate: float
    Interest-bearing debt interest rate: float
    Tax rate (A): float
    Net Value Per Share (B): float
    Net Value Per Share (A): float
    Net Value Per Share (C): float
    Persistent EPS in the Last Four Seasons: float
    Cash Flow Per Share: float
    Revenue Per Share (Yuan Â¥): float
    Operating Profit Per Share (Yuan Â¥): float
    Per Share Net profit before tax (Yuan Â¥): float
    Realized Sales Gross Profit Growth Rate: float
    Operating Profit Growth Rate: float
    After-tax Net Profit Growth Rate: float
    Regular Net Profit Growth Rate: float
    Continuous Net Profit Growth Rate: float
    Total Asset Growth Rate: float
    Net Value Growth Rate: float
    Total Asset Return Growth Rate Ratio: float
    Cash Reinvestment %: float
    Current Ratio: float
    Quick Ratio: float
    Interest Expense Ratio: float
    Total debt/Total net worth: float
    Debt ratio %: float
    Net worth/Assets: float
    Long-term fund suitability ratio (A): float
    Borrowing dependency: float
    Contingent liabilities/Net worth: float
    Operating profit/Paid-in capital: float
    Net profit before tax/Paid-in capital: float
    Inventory and accounts receivable/Net value: float
    Total Asset Turnover: float
    Accounts Receivable Turnover: float
    Average Collection Days: float
    Inventory Turnover Rate (times): float
    Fixed Assets Turnover Frequency: float
    Net Worth Turnover Rate (times): float
    Revenue per person: float
    Operating profit per person: float
    Allocation rate per person: float
    Working Capital to Total Assets: float
    Quick Assets/Total Assets: float
    Current Assets/Total Assets: float
    Cash/Total Assets: float
    Quick Assets/Current Liability: float
    Cash/Current Liability: float
    Current Liability to Assets: float
    Operating Funds to Liability: float
    Inventory/Working Capital: float
    Inventory/Current Liability: float
    Current Liabilities/Liability: float
    Working Capital/Equity: float
    Current Liabilities/Equity: float
    Long-term Liability to Current Assets: float
    Retained Earnings to Total Assets: float
    Total income/Total expense: float
    Total expense/Assets: float
    Current Asset Turnover Rate: float
    Quick Asset Turnover Rate: float
    Working capitcal Turnover Rate: float
    Cash Turnover Rate: float
    Cash Flow to Sales: float
    Fixed Assets to Assets: float
    Current Liability to Liability: float
    Current Liability to Equity: float
    Equity to Long-term Liability: float
    Cash Flow to Total Assets: float
    Cash Flow to Liability: float
    CFO to Assets: float
    Cash Flow to Equity: float
    Current Liability to Current Assets: float
    Liability-Assets Flag: int
    Net Income to Total Assets: float
    Total assets to GNP price: float
    No-credit Interval: float
    Gross Profit to Sales: float
    Net Income to Stockholder's Equity: float
    Liability to Equity: float
    Degree of Financial Leverage (DFL): float
    Interest Coverage Ratio (Interest expense to EBIT): float
    Net Income Flag: int
    Equity to Liability: float

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
