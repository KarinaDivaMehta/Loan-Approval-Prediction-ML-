from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load('../models/loan_status_predictor.pkl')

loan_app = FastAPI()
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
scaler = joblib.load('../models/scaler.pkl')

class LoanApproval(BaseModel):
    Gender: float
    Married: float
    Dependents: float
    Education: float
    Self_Employed: float
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: float

@loan_app.post('/predict')

async def predict_loan_status(application: LoanApproval):
    input_data = pd.DataFrame([application.dict()])

    # Convert Property_Area
    input_data['Property_Area_Semiurban'] = (input_data['Property_Area']==1).astype(int)
    input_data['Property_Area_Urban'] = (input_data['Property_Area']==2).astype(int)

    # Convert Dependents
    input_data['Dependents_1'] = (input_data['Dependents']==1).astype(int)
    input_data['Dependents_2'] = (input_data['Dependents']==2).astype(int)
    input_data['Dependents_3'] = (input_data['Dependents']>=3).astype(int)

    input_data = input_data.drop(['Dependents','Property_Area'], axis=1)
    
    input_data[num_cols] = scaler.transform(input_data[num_cols])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        return {'Loan Status': 'Approved'}
    else:
        return {'Loan Status': 'Not Approved'}

 